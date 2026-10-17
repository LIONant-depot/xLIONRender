# Surfel GI: dynamic scenes under a fixed budget (master plan)

Status: **phases 0–7, 2b, 6b built; Part II (the GPU) G1–G5 built** (2026-10-17). Written 2026-10-17 from the engine study
(Unreal, O3DE, Godot, Flax, CryEngine), three review rounds, and what the code did then. Section 5 compares the result with the
references; section 7 orders what is next.

Revision 2 folds in the third review: bounded contribution rather than invariant frame time, phase-level versus end-to-end
claims, fairness separate from importance, a memory budget, publication progress, GPU-side publication, filter dependency
halos, mesh-shared versus per-instance state, qualified basis rotation, and tighter competitive claims.

## 1. The contract

These are the rules every phase must keep. They are what the tests check.

1. **GI never waits and never catches up.** GI does not wait for convergence and never drains deferred work synchronously.
   Its main-thread maintenance contribution is limited by the granted allowance plus the declared maximum work-unit overshoot.
   Under load the lighting gets less accurate or later. Releasing the mouse never means "finish the backlog now".
2. **One application-wide allowance.** GI maintenance gets **2 ms of main-thread time per frame for the whole application**
   (a 60 FPS editor target), granted by the host per frame and shared by every renderer instance (the Level, the geometry and
   material previews, thumbnails). About **1.5 ms is scheduled**; the rest absorbs prediction error and overhead. Every stage
   counts: dirty marking, scheduling, links, tracing, filtering, upload preparation, allocation, publication. The numbers are
   starting settings, tuned from measurements in Debug and Release.
3. **This is a CPU-maintenance contract.** GPU cost is reported separately; nothing here promises that presentation always
   meets 60 FPS.
4. **Bounded indivisible work.** The bound is the allowance plus at most one unfinished work unit, so the unit size is part of
   the contract: after Phase 0 a numerical maximum unit time is declared (a starting target: 0.25 ms in Debug) and any unit
   type that exceeds it is subdivided. Long computations yield inside, not only between top-level operations. OS scheduling,
   allocation and driver stalls are outside the bound, and the log tells the three causes apart (unit too large / blocking
   inside a unit / external delay).
5. **Bounded memory.** A memory allowance covers queued task metadata, pending link candidates, working irradiance and filter
   buffers, retained input snapshots, retired GPU allocations and superseded candidates awaiting destruction. Reclaiming
   storage is itself sliced work: cleanup never becomes a release-time operation.
6. **Progress, not only safety.** Bounded work must reach the screen. Under sustained editing, useful results are still
   published; the log shows time since the last useful publication and work discarded as superseded.
7. **Published state and working state are separate.** Rendering reads only published state; resumable jobs write working
   state. A unit is published only when complete and when its own dependencies are still compatible (not merely because the
   global scene revision has not moved). "Still allocated" is not "safe to sample".
8. **Correctness failures and quality compromises are different things.** NaNs, stale handles, uninitialised links and
   incompatible publication are bugs, always. Noise, late indirect occlusion and temporarily stale light are allowed under load,
   and they are visible in the log.
9. **Transport participation is explicit.** Whether an instance receives, occludes, bounces or emits is a declared role, never
   inferred from "it moved". Making something receiver-only is an approximation chosen by the author, not a performance trick.
10. **Response is measured in time.** Blend constants are not copied between representations. What is specified and tested is
    the time for a surfel to recover from a change, in milliseconds, separately from its smoothing.

## 2. What the code does today (the starting point)

From reading the solver (`features/gi/xlionrender_surfel_gi.h`) and the forward plan:

| Area | Today | Problem |
|---|---|---|
| Tracing | `Update` takes `RayBudget / RaysPerSurfel` surfels | Counts hemisphere rays only (not the shadow rays of each hit); count-bounded, not time-bounded. The main thread waits for the worker pool to finish. |
| A move (`SetTransform`) | `DirtyNear` marks every instance whose bounds, plus a 4 m margin, overlap the old or new bounds dirty; a dirty instance's history goes to 0 | One noisy 8-ray pass replaces its published light (blend 1). Confidence, variance, moments and priority are one flag. |
| Emitter links | Rebuilt for the whole world in one call, 8 updates after the last move | About 1.7 s in Debug. Delayed, not bounded. Links are owned by receivers, so an emitter change affects many owners. |
| Display filter + upload | `BasisLight` filters every shown instance on every frame, then the whole instance is copied into the frame's ring region | 2–4 ms in the small scenes, growing with surfels; recomputed even when nothing changed; a ring region holds complete payloads only, so it cannot simply be sliced. Overflowing the region (131072 surfels) drops GI for the instances that do not fit. |
| First use of a mesh's GI | Decoding plus the rays' BVH (`BuildRays`) at load | About 630 ms in the Police level, synchronous. |
| Several renderers | Each world's renderer has its own full budget | They add up. Whether hidden editor tabs keep rendering is not known yet. |
| Roles | Per asset: `RECEIVE` or `RECEIVE_AND_BOUNCE` (`m_bBounce`) | No per-instance override; not checked that every path keeps a `RECEIVE` mesh out of other surfels' rays and links. |
| Sample rotation | The seed is a per-frame counter, per-pass rotation of the stratified set | Probably fine; no test proves it advances across a move. |
| Solve order | Within one `Update` the hits read last committed light (Jacobi); across updates slices commit in turn | Fine for a progressive solve and kept. The display filter is different: each of its passes must read one completed version of the previous pass. |

**What is shared and what is not.** Instances of a mesh share its mesh-local geometry and reconstruction data: surfel
placement, the read grid, the mesh BVH, neighbour lists and the other static metadata. Scene-dependent lighting and
visibility stay per instance: irradiance, confidence and history, the runtime depth moments, emitter links and visibility,
update age. Two identical chairs in different rooms share no lighting state. No memory optimisation may share the second
group.

## 3. Phases

Each phase lists what it delivers and the test that ends it. Every test runs with the budget enforcement on.

**What each phase may claim.** A phase's test proves only the paths it converted. Paths not yet converted stay visible in the
log and are explicitly outside the guarantee; a stage-specific pass is never reported as an application-wide pass.

| Milestone | What may be claimed |
|---|---|
| Phase 1 | Scheduler overhead and synthetic resumable jobs are bounded; tracing is scheduled units (no synchronous worker wait) |
| Phase 2 | Emitter-link maintenance is bounded |
| Phase 3 | Resident-scene filtering, upload preparation and publication are bounded |
| Phase 7 | Interactive first-use initialisation is bounded |
| All of the above | The end-to-end GI maintenance contract |

### Phase 0: Instrumentation (first, before any behaviour change)

- Per stage, per frame: allowance granted, main-thread time used, units done, longest unit per type, primary and secondary ray
  counts, links processed, filter surfels, upload bytes, allocations, publications, publication rejections, pending work, age
  of the oldest item, superseded and discarded work, repeated cancellations, time since the last useful publication, memory
  held by each category of rule 5, overruns (classified as in rule 4).
- Traced through one move: `SetTransform` → what was marked → history changes → link work → publication.
- Written to a permanent test log, with the slowest stage named, not one "GI took N ms" figure.
- **Done when:** the drag-and-release of a wall in the Police level produces a log that shows exactly where the 1.7 s and the
  per-frame 2–4 ms go, and the maximum unit time of rule 4 can be declared from it.

**Phase 0 results (2026-10-17, Debug, the Police level with the puppy).** Built: per-stage times and counts in
`gi::world::stats`, the renderer's filter / ring copy / moments / capture / load times, a 1200-frame log
(`instance::DumpGIFrames`, written by the Level's `DumpGI -File path` to `path.frames.csv`), and test 6j10. Measured by moving
an entity with `SetProperty` every round trip for 4 s, then 8 s still:

| | Puppy dragged | Building dragged |
|---|---|---|
| GI main thread, still | 19.0 ms mean | 20.3 ms mean |
| during the drag | 21.2 ms mean, 37 ms max | 20.2 ms mean, 43 ms max |
| the frame after release | **1839 ms**: `RebuildEmitterLinks` 1769 ms + `UpdateEmittedLight` 53 ms | **1719 ms**: links 1656 ms + emitted light 46 ms |
| mean by stage, settled | ambient cube 11.8, display filter 3.0, statistics pass 1.7, ring copy 1.4, mean light 0.5 | same order |
| traced per frame | 8–12 surfels (64–96 rays) | same |
| longest unit | one surfel's trace: up to 0.47 ms | |
| memory | links 6.2 MB (275k links, 350 emitters), light state 37.3 MB, filter 2.2 MB, upload 1.1 MB per frame | |

What it shows:
- **The release stall is the link rebuild alone** (one 1.7 s unit), plus 50 ms of emitted light right after it.
- **The settled cost is bookkeeping, not tracing.** Tracing is 0.3–0.6 ms; the per-frame passes over every surfel (the cube
  blend, the filter, the statistics, the ring copy, the mean light) are about 18 ms.
- **A bug found by it, not a cost to schedule:** both instances are held as ambient cubes, the building included (23221
  parents blended from 6 samples every frame). The screen-area rule compares an instance's whole surfel count with the pixels
  it covers, and a large instance always has more surfels than the screen can show, even with the eye inside it. The rule must
  apply to small instances only. To be fixed as its own change (it changes behaviour), before Phase 1 measures anything.
- **After that fix** (same measurement): the building is on its fine surfels (63142), only the puppy is a cube (49 parents).
  The settled cost becomes real work: tracing 6.6 ms mean (2048 surfels a frame, the 16384-ray budget), the display filter
  4.0 ms, the ring copy 3.2 ms, the statistics pass 2.4 ms, the commit 1.8 ms; about 20 ms of GI main-thread time a frame. The
  release stall grows with the finer receivers: the link rebuild takes **2816 ms**. Memory: light state 40.0 MB, filter 6.1 MB,
  upload 3.0 MB a frame. One surfel's trace now reaches **2.4–4.3 ms** on the building's geometry: the surfel is too big a unit
  (its 8 rays and each hit's direct light), so Phase 1 splits it (rays in chunks), and the unit target below stays a target.
- **The unit target.** One surfel's trace reaches 0.47 ms in Debug (the puppy scene; 4.3 ms on the building after the fix), above the 0.25 ms starting target of rule 4. The declared
  maximum unit time for now is **0.5 ms in Debug**, the surfel being the unit; the link rebuild (1.7 s) and the per-instance
  passes over every surfel have no unit yet and are what Phases 1–3 cut up.

### Phase 1: The maintenance scheduler

- Typed, resumable work units (link chunk, trace chunk, filter chunk, upload range, publication, reclamation), each saving its
  progress and yielding inside long computations.
- Tracing becomes scheduled units in this phase: the main thread no longer waits for a whole `Update` on the worker pool.
- One prioritised queue with one shared allowance. Per-type cost estimates that are conservative (a high quantile, not the
  mean) and start with small chunks until observed. A unit that does not fit the remaining allowance is not started; a minimum
  unit that never fits is subdivided, not starved. A count limit stays as a second guard.
- **Importance and fairness are separate.** An importance queue orders work by visible impact, change magnitude and transport
  relevance; a fixed share of each frame's allowance (configurable, a starting value of 20 %) serves the oldest outstanding
  relevant work whatever its importance. Off-screen contributors (a wall bouncing light onto what is visible) keep their
  transport relevance; being off screen is not being unimportant.
- The host grants the allowance per frame; an instance reports what it used and what is left; the host decides who gets more.
  The foreground Level comes before previews and thumbnails. Hidden views do not request image refreshes; shared world
  maintenance continues while an active consumer needs it; a world with no active consumer may pause.
- Overload: coalesce changes to the same target, drop superseded work, reduce optional refinement before using more time, and
  respect the memory allowance (Phase 2 defines the candidate policy).
- **Done when:** a synthetic overload (a stream of changes faster than the allowance, within the envelope of section 4) keeps
  every frame within allowance plus one unit, the queue and its memory stay bounded, aged work is still served, and the work
  converges after the changes stop.

**Phase 1 results so far (2026-10-17, Debug, the Police level, the same drag).** Built: picked surfels stay pending across
updates and are traced in units of rays (`work_item`, `TraceChunk`), round after round within `setup::m_TraceAllowanceMs`
(scene default 1 ms; 0 keeps the old one-update behaviour, with the same rays and seeds); a unit is started only if the
governor's conservative estimate (the slowest recent ray, decaying, at least the mean plus twice the deviation) fits; the
fairness share (20 %) for the instance waiting longest; the trace's worker time split into casting, direct light and reads.
Test 6j11 (a wall moving every update, a tight allowance) passes: within the allowance plus one unit, pending never above one
pick, the far box served, and the floor converges to the scene without an allowance (0.914 against 0.914).

| | Before (Phase 0) | Now |
|---|---|---|
| Trace time per frame | 6.6 ms mean | 0.91 ms mean, 1.20 ms p99, 1.65 ms max (allowance 1 ms) |
| Unit | one surfel, up to 4.3 ms | one ray; 135 us mean, 508 us p99, 672 us max |
| Rays per frame | about 16000 | about 1000 (convergence is that much slower: the contract's trade) |
| Pending | none | up to one pick (2048 surfels) |

- **Casting dominates the trace, not shadow rays**: over the workers, casting 7.8 ms, reads of the hit's light 3.9 ms, direct
  light 1.3 ms. The bounded light sampling of section 5c is not needed for now; faster casting (the rays' BVH in Debug) would be.
- **A single ray is now the indivisible unit**, and its tail (672 us in Debug) is above the 0.5 ms declared maximum. The current
  traversal cannot interrupt a ray; what would cut the tail is cheaper casting and reads. (Historical: the declared units are
  consolidated in Phase 6b's results.)
- **The estimator is conservative by design**: it sizes by the slowest recent ray (about 340 us against a 13 us mean), so the
  allowance is used safely but not fully. A high quantile instead of the maximum would recover rays; to tune after Phase 3.
- **The host's shared allowance:** `xeditor::gi_allowance` (`host::m_GIAllowance`, reset at the top of each frame) holds 1.5 ms
  for the whole application. The Level's viewport (the foreground) asks first through `view_options::m_GITraceAllowanceMs`
  and reports through `xRenderEditor::GITraceMs` (interface version 9); the static geometry preview gets what is left. What the
  foreground used in the last frame stays reserved for it until it renders, whatever the render order; a foreground that did
  not render keeps nothing. Only the trace is charged so far. Test: `smoke/test_gi_allowance.py` (the rule, no editor).
- **Not done yet in Phase 1:** the other per-surfel stages (filter, ring copy, statistics, mean light: about 9 ms a frame),
  which are Phase 3's, and charging them to the allowance.

### Phase 2: Bounded, incremental emitter links

- A dependency index: per emitter, the receivers that link to it; per link, its segment's bounds.
- Changes become typed events: emitter changed (recompute its dependants, discover new ones), occluder moved (recheck only the
  links whose segments cross its old or new box), receiver moved (rebuild its own list). A receiver-only, non-emissive
  instance triggers none of them.
- The rebuild is a cursor over receivers. Candidates are built beside the published links, carrying the input revisions of
  their own dependencies, and swapped in when complete and those dependencies are still compatible.
- **Candidate policy.** At most one pending candidate per product. A new revision of its inputs repairs the candidate where the
  change is local (the receivers it touches go back on the cursor) and restarts it only where it is not; repeated edits
  coalesce. Abandoned storage is reclaimed in sliced units. When the memory allowance is exhausted, new candidates wait and the
  published links stay in use (logged).
- The publication product starts as one instance. If the log shows a large instance (the Police building) whose candidate
  never completes under sustained editing, it is subdivided into stable receiver blocks. This is not built ahead of need, but
  the API takes a product id, not an instance id, so it can be.
- **Done when (link maintenance only):** dragging and releasing a wall keeps link work within its share of the allowance;
  links are published during sustained dragging, not only after it; after settling, the links equal a fresh deterministic
  build of the same final scene (a coherence check). Test C passes: no stale emitter connection survives a closed door.

**Phase 2 results** (2026-10-17, Police level, Debug, the same wall drag as Phase 0):

| | Phase 1 | Phase 2 |
|---|---|---|
| Worst GI main-thread frame after the release | 3157 ms (links 3131 ms) | 15.9 ms |
| Worst frame during the drag | 15.6 ms | 16.0 ms |
| Link work per frame | 0, then 3.1 s at once | 0.8 ms at most (its allowance) |

What was built, in `gi::world`:
- **Granularity: the receiver surfel, not the instance.**
  - A receiver that moves marks all its surfels.
  - An emitter that moves or appears, or a change of which surfels emit, marks every receiver.
  - An occluder marks only the surfels whose box of examined segments (the surfel and the emitters it examined) crosses
    its old or new box.
  - A change of how much an emitter emits changes no link (a link's weight is geometric).
- **The links store the emitter by (instance, surfel)**, so they stay valid while the emitter list is rebuilt.
- **A queue of instances (`PumpLinks`)**:
  - The marked surfels are built in chunks on the scheduler's workers.
  - The links get `m_LinkShare` (0.3) of the allowance while the trace has pending work, and all of it otherwise.
  - Every update makes progress: at least one surfel per worker, the one-unit overshoot.
  - A unit is bounded: one receiver surfel casts at most 64 shadow rays (`max_examined_v`).
- **Publication (`FlattenLinks`)**:
  - The built links and the others' published links are copied into staging storage, 512 surfels a slice, within the
    allowance. The emitted light per fine surfel is made in the same slices.
  - It is swapped in at the end. A partial publication updates the parents' emitted light by deltas.
  - Two Debug spikes (44 ms, then 32 ms) came from publishing and recomputing the Police building's 246k links in one
    frame; slicing removed them.
- **A mark while an instance is being built queues it again.** Repeated edits coalesce, and a drag still publishes as it
  goes.
- **The settle delay (`links_settle_v`) is gone.**
- **The GI's own thread pool is gone too.** `ParallelFor` runs on xscheduler. LIONRender.dll starts its own copy of
  `g_System` on first use, because xscheduler's globals and its `thread_local` worker ids are per module.

Tests:
- 6j9 (rewritten): the link work of an update stays within the allowance; links are published while the floor moves; once
  it stops, the links equal a fresh build, compared by a digest.
- 6j9b (new, test C): a door slides shut between a lamp and a floor, so its links and emitted light go to 0. It opens
  again, and they come back equal to before. The partial-publication deltas are checked by the same comparison.
- 6j10 now expects link rebuilds during the drag.

Still open:
- The initial build of a large level is spread over many frames, so emitter light appears progressively after a load.
  This is Phase 7's question.
- A receiver surfel loops over every emitter (O(emitters)). An emitter hierarchy would cut that if emitter counts grow.

### Phase 2b: Consolidation (one mechanism per job)

The solver grew several mechanisms for the same job, one per case found, and sections 5c and 5d would add more. Before Phase 3
makes any of them bounded and resumable, each job keeps exactly one mechanism. Every merge stands alone: one change, the GI tests
(6j*) plus a before/after capture of the Police and Sponza interiors at equal maintenance time, and a commit. A merge that
looks worse in the capture is not kept until the cause is understood. More code and setup knobs are removed than added.

- **One estimator.** Today a running average (`m_PassAlpha`) warms a surfel up after a change, and the multiscale estimator
  takes over afterwards and is reseeded from it (`Update`, the finished surfels). The estimator was designed to do both jobs:
  it catches up on a change and it settles. Seed it from the current light (as SurfelGI and SurfelPlus do) and let it follow
  from the first pass. `m_PassAlpha`, the warm-up count and the reseed go away. The validity and the moments keep their own
  slow blends, which are a separate quantity, not a second estimator. Its variance, normalised by the batch's ray count
  (section 5d), is the uncertainty that the priority and the sharing read.
- **One sharing mechanism.** There are three candidates for one job:
  - the display filter (`m_FilterPasses`: 4 passes, 3.5 ms a frame, done after the solve);
  - 5c's sharing by confidence;
  - 5d's neighbours as one extra estimator sample.

  Keep the last one. It runs inside the solver's own pass, over the surfels finished this update only, and its weight follows
  the uncertainty, so sharing by confidence comes with it. It uses the same neighbour rules as the filter today: the same
  layer, normal and distance, and backfaces rejected. Compare it against the filter at equal time on the settled interior and
  a light toggle. If it holds, the filter is removed, and Phase 3's "the filter becomes a resumable job" becomes publishing
  the solver's result. If it does not hold, the filter stays and the estimator sample is not added.
- **One level-of-detail rule.** Today a mix of rules decides how many surfels an instance shows:
  - distance tiers (`m_bDistanceTiers`, `m_LodDistance`);
  - screen area (`m_PixelsPerSurfel`, `m_ProjScale`);
  - the small-on-screen gate (`m_SmallFraction`);
  - the ambient cube (`m_bCube`) and the cube parents;
  - the fine and parent levels.

  These become a single rule: the instance's projected pixels per surfel pick its level, with hysteresis, and the cube is the
  coarsest level. The rule has one knob, the pixel target. The distance tiers and the small gate are removed if the pixel
  rule reproduces them (a far instance and a small instance both have few pixels). The Phase 0 bug (large instances turned
  into cubes) gets a test: a large near instance never takes the cube.
- **One priority function.** Each of these becomes a term of one priority, served from one queue under the one allowance:
  - the pick score;
  - the age;
  - the change marks (`m_ChangeMargin`);
  - dormant sampling (5c);
  - the requested rays (5c, 5d);
  - not on screen this frame (5d: a quarter of the rays);
  - and later, Phase 5's transport roles and Phase 6's relighting.

  The fairness share stays outside the priority, as the guarantee that old work is still served. Nothing gets a pass or a
  queue of its own. This item is mostly a rule for the later phases: today's code already has one queue (Phase 1).
  Writing the function down here keeps it that way.
- **Not merged, because the parts do different jobs:**
  - the plane-distance and moment visibility tests (a different leak each);
  - the emitter links and the rays (small emitters the rays miss);
  - the host's allowance and the solver's governor (two levels of one budget).
- **Done when:** each job above has one mechanism in the code, the setup has fewer knobs than before, the GI tests pass, the
  captures at equal time are no worse, and the per-frame bookkeeping is measured again for Phase 3. Removing the filter
  alone would take 3.5 ms off the 9 ms.

**Phase 2b results** (2026-10-17; Police, a fresh Debug editor, 60 s from one view, the shown light and its flicker over 3 s from
two DumpGI files, `b2cap.py`/`b2cmp.py`):

1. **One estimator: kept.**
   - The warm-up (an exact 1/n average after a change) moved into the estimator, per surfel (`m_N`, `Restart`). The call site
     has one path.
   - Result: shown light 0.0151 against 0.0149, flicker 2.7 % against 3.0 %. The GI tests changed by less than 2 %.
   - The multiscale blend still starts from no variance. A variance measured from the warm-up's 8-ray samples would make it
     slow to follow the light while it builds up.
   - Lesson: the first comparison was against an editor that had had the level open for minutes, and it looked like a 50 %
     loss. Comparisons are made from a fresh editor only.
2. **One sharing mechanism: the display filter.**
   - Neighbours as an estimator sample (1/(1+n) while warming up, 0.04 settled, untraced neighbours weighted out) was worse at
     equal trace time: light 0.0143 against 0.0151, flicker 3.7 % against 2.7 %.
   - It is not adopted, and neither is 5c's separate sharing by confidence. The filter's 3.5 ms is Phase 3's to make
     incremental.
3. **One level rule: kept.**
   - A level's surfels must each cover `m_PixelsPerSurfel` pixels where the instance is nearest the eye, else the next coarser
     level; the ambient cube is the last.
   - Removed: the distance threshold for levels, the small-on-screen gate (`m_SmallFraction`, `m_ViewPixels`) and the view-pixel
     plumbing. `m_LodDistance` remains as the priority's distance scale.
   - Police: the building is unchanged (the eye is inside it). The statue 6 m away is at its fine surfels, where the gate held
     it as a 6-sample cube.
   - Tests: 6j7 and 6j7b follow the rule (26 m away: the parents, never a cube); three tests now set a screen scale.
4. **One priority function: kept.**
   - `Priority` (dirty, a pass in progress, age over distance) and `TracePeriod` (buried surfels, the distance tiers) are the
     only places that decide who is served and how often. Fairness stays outside them.
   - Later phases add terms there. There was no behaviour change; the test output is identical.

### Phase 3: Publication on the GPU side

- Persistent per-instance GI storage on the GPU in place of copying every instance's full payload into the frame's ring.
- **Publication protocol** (no CPU wait anywhere):

```
working CPU result
  -> candidate GPU storage / ranges (uploaded in slices)
  -> upload complete and visible to the GPU
  -> publish the descriptor / range / version the renderer reads
  -> retire the old storage after the frames that may still read it
```

  Rendering keeps the previous allocation until the candidate is published; nothing writes into the storage the renderer is
  reading. If partial ranges are ever published separately, each range is its own publication unit with a compatible version;
  an arbitrary partial upload is never called a completed product. Allocation and descriptor updates are inside the measured
  path.
- The display filter first becomes a resumable job that recomputes a whole instance into scratch storage, each pass reading
  one completed version of the previous pass, then publishes. Only after that is correct does it become incremental, and then
  with a **dependency halo**: the changed surfels, their filter neighbours, the expansion for each further pass, and changes of
  validity and weights as well as of colour.
- **Done when (resident scenes only):** the per-frame filter and upload cost of a settled scene is close to zero, a changing
  scene stays within its share of the allowance, and an image comparison shows no mixing between published and in-progress
  data.

**Phase 3 results** (2026-10-17, Police, Debug, the Phase 0 wall drag and release):

| GI main thread per frame | Phase 2 | Phase 3 |
|---|---|---|
| Mean while settled | 10.9 ms | 2.7–2.9 ms |
| Worst during the drag, after the release | 16.0 ms | 6.5 ms |
| Upload | 3.03 MB/frame | 0.2–0.4 MB/frame |

What was built:
- **The shown light (`UpdateShown`).**
  - The filter's passes are kept per instance. The surfels finished in an update are pass 0's dirty set, and each further
    pass is made again for the surfels that read any made again by the pass before: the reverse neighbour lists, a halo that
    grows one ring a pass.
  - The halo work runs in chunks of 64 dirty surfels within `m_ShowAllowanceMs` (half the trace's allowance), and a backlog
    waits for the next update. Past a quarter of the instance, the whole filter runs in parallel instead.
  - A level switch, a new layout or a new setting (passes, saturation) makes all of it again.
  - `BasisLight` returns the kept result.
- **The rings (`updateGI`).**
  - Every instance has a stable place in every frame slot: its mean, then its surfels. The places come from a first-fit
    allocator, compacted when it is full.
  - Each instance keeps, per slot, the surfels the slot has not had since they changed (or "all of it"), for the light and
    for the moments. A slot is written only those.
  - Each frame's slot is therefore one complete version, never written while the GPU reads it. This is the publication
    protocol at the scale of a ring slot.
- **The statistics.**
  - Per instance, kept where they change: the mean light and the bad count by deltas over finished surfels, summed again
    exactly once per surfel count of deltas; the material's colours when it is set; the emitted light's maximum with the
    emitted light.
  - No loop over every surfel per update remains. A census every 16th update was tried first and caused 3 ms spikes.
- **Tests.** 6j12 checks that the incremental shown light equals the whole filter made again (difference 0), through a
  sliced backlog. A settled update makes again 507 of 3469 floor surfels at most. All GI and GPU render tests pass, with
  output unchanged.
- **Image.** An overview capture of the Police level matches the Phase 2b build. The large dark block in both images is
  unconverged light, not this change.

Left for later phases:
- Persistent GPU storage per instance, without the per-slot copies. Today the 3 slots hold 3 copies, which costs memory, not
  time.
- Publication of partial ranges.
- The remaining spikes (6.5 ms) are a link chunk plus a filter chunk landing in the same frame. Phase 4 and the host
  allowance can schedule them against each other.

### Phase 4: Splitting the solver's state

Separate what one dirty flag does today:

| State | After a supported change |
|---|---|
| Published irradiance | Kept as the warm start |
| Confidence (effective sample count) | Reduced by how strongly the change affects the surfel |
| Change detector / variance | Reinitialised or adapted; a large detected change rejects history (as production DDGI does) |
| Visibility moments | Refreshed where geometry or support changed |
| Priority | Raised for the affected surfels |
| Unsupported reads | Shown as unsupported, not hidden behind old confidence |

- "Affected" is scored (overlap, distance in reach, visibility) instead of marking whole instances within 4 m.
- A moved instance keeps its layout and starts from its old light.
- **Rotation.** The three-direction basis is an approximation, and an arbitrary rotation is not exactly representable in it.
  Directional history is transformed only through a validated basis conversion; otherwise its confidence is reduced and the
  surfels are prioritised for resampling. The angle may weight that priority as a heuristic, not as a measured error bound.
  Rotation is tested separately from translation: same position with a different orientation, uniform and strongly
  directional lighting, a large rotation, repeated rotations.
- Time since each surfel's last update is recorded, and the recovery time in milliseconds is reported.
- A regression test proves the ray sequence advances per pass, across a move, and is not reused between surfels.
- **Done when:** after a move, no surfel's published light jumps to a single noisy pass, and recovery time is reported per test.

**Phase 4 results** (2026-10-17):
- **A change lowers confidence; it never restarts.** `estimator::Lower(a)` sets the sample count to (1 − a) of the warm-up,
  one sample at least, so a single noisy pass moves a surfel's light by half at most. What the surfel had is the warm start.
- **A real change is still taken.** Until the surfel is confident again, two samples in a row that differ from the light it
  had by half, the same way, replace it with their own mean: two passes of evidence.
  - "The light doubled" now reaches 90 % in 5 updates (10 with the old restart). A sun change is reached by the same rule.
- **"Affected" is scored per surfel (`AffectNear`)**, by its distance to the changed box: 1 at the box, 0 at its reach.
  - The reach is twice the box's diagonal, between 0.5 m and `m_ChangeMargin`, so a small object changes the light near it
    and a large one farther.
  - An affected surfel is lowered, its moments refresh if it is affected by half or more, and it goes on its instance's
    urgent list. The urgent list is traced first and is a term of `Priority`.
  - Neighbour instances are no longer marked dirty as a whole. The city test: a moved building leaves 1 instance dirty and
    affects 15.
  - Surfels are found through per-instance world buckets (cells of the margin), built once per placement.
- **The moved instance keeps its layout and its light**, lowered by half, more by its rotation angle (the basis turns with
  it; the angle weighs it as a heuristic). Its fractions and validity blend by the surfel's confidence, not by a per-instance
  pass blend (`m_PassAlpha` is gone).
- **Recorded per surfel:** the update that last finished it (`LastPass`) and its last ray offsets (`RayOffsets`).
- **Test 6j13.**
  - A 30 cm move: 0 surfels restarted, 1661 of 6222 floor surfels lowered and none beyond the reach, the first update moving
    a surfel's light by 32 % at most.
  - Recovery to a fresh solve of the final scene in 1 update (2.2 ms; this move barely changes the floor's mean).
  - A 90° rotation in place under a uniform sky: 0 restarted, the wall's light moved 0.0 %.
  - The ray offsets advance per pass and across the move, and differ between surfels.
- **Police, Debug, the drag.** A move costs 1.5 ms on average (`AffectNear` and the link marks in plain floats over buckets),
  against 33 ms in the first version. The worst frame is the first bucket build of the building (16 ms, once per placement).
  Mean GI main thread: 3.7 ms during the drag, 2.3 ms after.

### Phase 5: Transport roles

- Per-instance roles, overriding the asset's default:

| Role | Receives | Occludes indirect transport | Bounces / emits |
|---|---|---|---|
| Receiver only | yes | no | no (emission separately, if supported) |
| Proxy occluder | yes | approximate (its box or ambient cube) | no |
| Full transport | yes | yes | yes |

- Full transport stays the default for structural geometry and for anything placed in the editor. Proxy occlusion is built
  only if the receiver-only result shows a measurable need.
- Direct shadows are a separate system and are not affected by any of this.

**Phase 5 results** (2026-10-17):
- **The role.**
  - The Mesh component has `GIRole`: ASSET (as the geometry's GI mode says), RECEIVER or FULL. It is shown in the inspector
    and serialized, and the default keeps existing levels as they were.
  - It reaches the renderer in the draw's instance flags (bits 32–33, `draw_gi_role_shift_v`) and the solver through
    `world::SetRole`.
- **What a receiver is.** It is out of the rays' scene and out of the emitter list. Moving or removing it affects no other
  surfel, marks no other link and changes no scene for the rays: only its own light and its own links.
- **A role change is a change for the others.** It updates the rays' scene, `AffectNear` around it, the links crossing it
  and, if it emits, the emitters.
- **Test 6j14 (test A).** A 40 cm box is dragged from the sun into a wall's shadow.
  - As RECEIVER: 0 floor surfels lowered or queued, 0 updates with link work for others, rays pass through it, and its
    light follows it (0.746 → 0.241).
  - As FULL: 4580 lowered, rays hit it.
  - A receiver is brighter than the same full box (0.746 against 0.125 in the sun): it casts no GI shadow on the floor next
    to it, so it does not see one. This is the role's intended approximation.
- **Police, Debug.** The statue set to RECEIVER and dragged: a move costs 0.01 ms (1.5 ms as full transport), 0 instances
  affected, no link work. The worst GI frame was 5.8 ms.
- **The frame log** gains LinkPending, LinkSurfels and UrgentSurfels. They showed that the 1.2 ms of link work in the first
  minute after loading is the level's first link build, which is Phase 7's to schedule.
- **Proxy occluders are not built.** Nothing measured needs them yet, per the plan.

### Phase 6: Prioritised relighting

- A changed light raises the importance of the surfels it reaches (by transport relevance and visible impact); a sun or sky
  change is a sliced relight, never a synchronous one. An explicit "relight everything" exists, rate-limited and still sliced.
- Fairness is the scheduler's (Phase 1), not a term of this score.

**Phase 6 results** (2026-10-17):
- **What earlier phases already gave.** A changed point light lowers and queues the surfels in its reach (Phase 4,
  `AffectNear`). A sun or sky change marks every instance dirty: its surfels keep their light as the warm start, it is traced
  under the allowance nearest first (`Priority`), and it is never synchronous.
- **Rays requested by uncertainty (`RequestedRays`, section 5d).**
  - A warming or watched surfel asks for all `m_RaysPerSurfel`. A settled one asks for `m_RaysPerSurfel × relative deviation
    × m_RayUncertainty` (3), a quarter to all of them.
  - The pick counts its budget in rays. This is on by default (`m_bAdaptiveRays`).
  - Test 6j15, a point light turned on in a room at the same 2048-ray budget: the recovery error over 60 updates is equal
    (0.278 both: during recovery every surfel is warming or watched and gets all its rays). The settled error is lower (0.135
    against 0.150). The Cornell comparison to the path tracer improves (5.5 % mean against 5.7 %).
- **`RelightAll`.** Every instance traced again under the allowance, at most once in `m_RelightPeriod` (120) updates; a call
  that comes too soon is deferred, not dropped. Test 6j15: deferred at once, 3 instances dirty, at most the budget per update.
- **Found while measuring: the first link build starved the trace.** After a load, a link chunk overran the allowance (one
  surfel per worker, 1.2 ms in Debug), and the trace was left 0.001 ms: 99 rays a frame instead of about 700. The trace now
  keeps its own share, `(1 − m_LinkShare)` of the allowance, at least.
  - Police, a fresh editor, 60 s: shown light 0.0159 against the Phase 2b build's 0.0146, flicker 3.8 % against 3.9 %.
  - **This means the frame can exceed the allowance by that link chunk.** Phase 6b closes it.

### Phase 6b: Budget closure (before Phase 7)

From a review of these results: the stalls are much smaller, but the 2 ms contract is not yet true end to end. Before more GI
features:

1. **One allowance for every stage.** Events (moves, materials), links, trace, the shown light and the upload preparation all
   draw from the same frame allowance. A stage's own ceiling may remain, but not as spending outside it.
2. **No unbudgeted fallbacks.**
   - The whole-filter shortcut becomes a resumable cursor.
   - "Progress" is one unit, never one per worker.
   - The first bucket build and the allocator's compaction are sliced.
3. **Batch accounting.** Per batch: main-thread elapsed time, workers, aggregate worker CPU, the longest task, and units
   admitted after the allowance was spent.
4. **Publication semantics, stated and tested** while the light keeps changing, not only once it settles.
5. **Localized recovery.** The affected region's mean absolute error and a high percentile, the error over time, and the time
   to reach and stay below a threshold, in updates and ms. Four events: a bright light off, a door closing across an emitter
   link, a receiver moving from a bright room to a dark one, and a directional change at similar energy.
6. **The change detector defined and tested.** High variance at constant light, near black, rapid on/off, coloured light, a
   moving bright source, repeated lowerings.
7. **A closure run.** GI main-thread median, p95, p99 and maximum; the longest unit by type; worker CPU; memory; the oldest
   work; the time since a useful publication. Several renderers open, sustained overload.

**Phase 6b results** (2026-10-17, Police, Debug, a fresh editor, the statue dragged 4 s and released, 1200 frames; `p6b_close.py`):

| GI main thread per frame (the Level) | Median | p95 | p99 | Max |
|---|---|---|---|---|
| The whole run (settling, drag, release) | 1.84 ms | 2.32 ms | 2.58 ms | 3.56 ms |
| During the drag | 1.61 ms | 2.15 ms | 2.39 ms | 2.42 ms |
| For comparison, Phase 3 (before the ledger) | 2.7 ms mean | | | 6.5 ms |

Per frame:
- The stages went past the allowance by 0.91 ms at p99 and 1.75 ms at most, about one link unit.
- Means: pre-charges 0.25 ms; trace 0.51 ms on the main thread (3.67 ms on the workers); links 0.45 (2.63); filter 0.57; ring 0.14.
- 101 units were admitted with the token in 1200 frames.

What was built:
- **The ledger (`Update`).** One allowance is shared, and each stage's grant is what is left of it: the allowance, minus the
  pre-charges (the events since the last update, and the renderer's own GI work outside the update smoothed as `m_OverheadMs`),
  minus the time the update has used so far.
  - The stages are, in order: the events that wait for buckets, the links, the trace (keeping `m_ShowShare` for the shown light
    while it has a backlog), and the shown light.
  - No stage overshoots on its own. A token goes to the stage that has had work for the most updates and done none; only it may
    start one unit past the budget. That bounds the overshoot to one unit and still lets every stage progress.
  - The host is charged the whole GI main-thread time (`GITraceMs` now returns the whole of `updateGI`), not only the trace.
- **No unbudgeted fallbacks.**
  - The whole filter is a resumable cursor (pass by pass, blocks of 4096). On a layout change, pass 0 is shown at once and the
    filtered values replace it as the passes complete. The "a quarter of it: all at once" shortcut is gone.
  - A link chunk is one round at most, a surfel per worker, and is admitted again before the next. The minimum progress is one
    surfel, and only with the token.
  - A change near an instance whose buckets are not built yet waits (`m_PendingAffect`). The buckets are built in slices of
    2048 surfels under the ledger, and the change is applied once they are ready. The first-build spike (16 ms) is gone.
- **Two bugs found by the closure run.**
  - A link chunk's size truncated the time divided by the per-surfel cost before multiplying by the workers, so a budget under
    one surfel admitted nothing, and the first link build crawled.
  - Chunks could hold several rounds, so their wall time could pass the estimate by several units (5.5 ms over).
- **Batch accounting**, per update in the frame log:
  - each stage's main-thread time;
  - the trace's and the links' worker time summed (`TraceCpuMs`, `LinkCpuMs`);
  - the longest ray unit and link surfel;
  - how far past the allowance the stages went (`OverAllowMs`);
  - the units admitted with the token (`TokenUnits`);
  - the pre-charges (`PreMs`).
- **The declared units (Debug, Police; this replaces the earlier figures).**
  - A ray: 0.65 ms at most.
  - A link surfel: 1.2 ms at most (O(emitters) candidates plus up to 64 shadow rays).
  - A shown-light chunk: 64 dirty surfels and their halo, or one 4096-surfel block of the whole filter.
  - A bucket slice: 2048 surfels.
  - The frame's bound is therefore the allowance plus the largest of these, the link surfel. Release units are smaller; a
    Release editor run is still to do.
- **Publication, stated.** The unit is one surfel's shown basis, written whole into a ring slot. Neighbouring surfels in a slot
  may be from different revisions while a backlog is worked off: each is made from the passes as they are when it is made, and
  its own value is never torn. When the inputs stop changing, every surfel reaches the whole filter of the final inputs.
  - Test 6j16: a wall moving every update and a light toggling every 7, under 0.3 ms. The shown light stays within 0.2 % of the
    whole filter of its current inputs, made apart, with no bad values. 198 updates after the changes stop, it equals that filter.
- **The affected radius is a priority and confidence heuristic, not visibility.** `AffectNear` decides which surfels are
  lowered and queued. Whether a stored visibility result still holds is decided elsewhere:
  - the emitter links by their own segment tests (`MarkLinksCrossing`, all of an occluder's crossings, whatever the radius);
  - the moments by the rays that refresh them;
  - and the surfels outside the radius by their recorded age (`LastPass`), "not refreshed yet" rather than valid.
- **The change detector, defined.**
  - It compares the luminance (0.299, 0.587, 0.114) of a sample with the light the surfel had when the change came (`m_Ref`),
    with a floor of 1e-3 under the denominator. Two samples in a row more than half of that apart, the same way, replace the
    light with their mean. The samples are the same batch size: one pass of the surfel's rays.
  - Test 6j18, one case at a time:
    - noisy constant light (±90 %): 3 takes in 60 samples, mean 0.91 for a true 1;
    - near black: 0 takes;
    - rapid on/off: the mean stays in [0, 1];
    - a colour change at equal luminance: not detected, followed by the warm-up average (green 0.48 of 0.51);
    - a ramp to 5: followed (5.00);
    - lowered every 3 samples for 600 samples: an unbiased mean (1.99 for 2).
- **Localized recovery (test 6j17)**, over the affected region against a fresh solve, under 1 ms: when the error gets under
  10 % and stays there, in updates and ms of updates.
  - A bright light off: 21 updates.
  - A door shut across an emitter's link: 5 updates.
  - A receiver into a dark room: 2 updates.
  - The sun turned 90° at the same energy: 1 update.
  - The 95th percentile of the error at the end is 3–9 %.
- **Not done.**
  - The Release editor run.
  - Several renderers drawing at once: in this editor only one GI view draws at a time, since a resource editor takes the
    screen from the Level. The shared host allowance is unit-tested.
  - Slicing the allocator's compaction, which is rare: it writes every instance whole into each slot once.

### Phase 7: First use is bounded too

- Separate compiler work, runtime loading and runtime GI initialisation. Precompute in the compiler what can be (the rays'
  BVH is the first candidate, once Phase 0 confirms it dominates the 630 ms); slice the rest.
- Until an instance's GI is ready it draws with a declared fallback (its mean light or the scene's ambient). A blocking loading
  screen is a separate, explicit mode.

**Phase 7 results** (2026-10-17, Police, Debug, a fresh editor opening the level):

| A GI mesh's first use (both meshes) | Before | Phase 7 |
|---|---|---|
| Main thread, in one frame | 639 ms (decode 26, surfels and rays' BVH 562, renderer 51) | 15 ms (copy 5, GPU buffer 5) |
| On a worker | – | 705 ms (decode, surfels, BVH, read grid, the GPU buffer's data) |

- **The split (`scene_renderer::MeshOf`).** The main thread copies the geometry's arrays (`gi::geom_gi_copy`: the same accessors
  as the geometry, so a geometry released mid-job cannot be read). An xscheduler job decodes it, makes its surfels, the rays'
  BVH and the read grid if the compiler did not pack one, and packs the GPU buffer's data (`PackMeshGI`, now pure). When the
  job is done, the main thread creates the buffer and the mesh joins the GI.
- **The declared fallback.** Until then the mesh has no surfels and draws with the scene's ambient, as a mesh without GI does.
  No loading screen.
- **Compiler precomputation of the BVH is not needed for the main thread.** The worker absorbs it. It would still shorten the
  time until GI appears (0.7 s in Debug) and is left for when that matters.
- **The first link build** is scheduled by Phase 6b's ledger. It was the 1.2 ms of link work seen for the first minute after a
  load, and now progresses within the allowance.
- **Two timing-dependent tests, found and fixed.** The GPU render test (`xlionrender_test_gi_render`, sparse-mesh reads) and 6j12
  depended on how much the allowance let through in a given run. The GPU test now runs with no allowance (as its comment
  intends). 6j12 works under its tiny allowance and drains without a limit before comparing. Both now pass repeatedly.
- **Remaining.**
  - The main thread's 5 ms copy and 5 ms buffer creation are one-time costs, outside the per-frame allowance.
  - A Release editor run.
  - The allocator's compaction is still not sliced (Phase 6b).

### Part II: the trace on the GPU (decided 2026-10-17)

**Why.** Phases 1–7 made the per-frame cost bounded: Police in Debug runs at a 1.84 ms median and a 3.56 ms maximum. But the
time until the light is resolved grew to a minute or more, where the target is 1 s, on PC first and on mobile too.

The cause is that the trace runs synchronously inside the main thread's allowance: 250–700 rays a frame in Police (Debug).
Converging the building needs about 8M rays (63k surfels × 8 rays × a 16-sample warm-up). Meanwhile the workers are about 99 %
idle.

A background CPU trace would reach about 1 s on a many-core PC, but not on mobile. So the decision is to go to the GPU directly,
with software BVH traversal in compute shaders: no hardware ray tracing required (as the WebGPU surfel GI of Juretriglav and
three-mesh-bvh do). The background CPU trace is not built.

**What stays.** The CPU keeps the policy:
- what to trace: priority, fairness, urgency, roles, confidence;
- the emitter links;
- the change events;
- the ledger, now for the CPU's part, with a GPU time budget beside it.

The CPU solver stays as the reference that every GPU step is compared against, and the unit tests stay on it.

**What moves.** The rays (`TraceChunk`), the light at the hits, the reads of other surfels' light, the estimator, the moments,
the basis fractions, and probably the display filter. The light then stays on the GPU: no upload ring for it.

**Steps.** Each is compared numerically with the CPU before the next:

| Step | What | Done when |
|---|---|---|
| G1 | Compute pipelines and dispatch in xGPU, with storage buffers and a readback. The CPU's BVHs (each mesh's BLAS, the TLAS) packed into GPU buffers. A traversal compute shader | For thousands of random rays over a real mesh and a scene of instances, the GPU's nearest hit (instance, triangle, distance) equals the CPU's `Cast`, distance within 1e-4 relative (1 m at least: 0.1 mm on short hits, the float rounding of the instance transform); the rays per ms are measured |
| G2 | A whole surfel pass on the GPU: the picked surfels' rays, the light at each hit (sun with a shadow ray through the same BVH, point lights, the sky's octahedral map, the hit triangle's albedo and emission), the reads of the hit's surfels, the accumulation | On the unit-test scenes (Cornell, rooms, blinds), the GPU's per-surfel results equal the CPU's for the same rays within a stated tolerance |
| G3 | The light resident on the GPU: the estimator, the moments and the basis fractions updated there. The renderer reads it directly. The CPU picks the work and uploads only the list | Police converges in about 1 s on PC; the CPU's per-frame GI time is the list and the events |
| G4 | The display filter on the GPU, the CPU's per-frame GI work removed, mobile measured | Per-frame CPU cost near zero; a mobile GPU budget declared and measured |

**G1 results** (2026-10-17, RTX 4070 Ti SUPER, Release; `xlionrender_test_gi_gpu`):

| | Rays | Differ from the CPU | Worst distance | CPU (one thread) | GPU |
|---|---|---|---|---|---|
| One mesh (a sphere, 18,624 triangles) | 65,536 | 0 | 8e-7 | 7k rays/ms | 146k rays/ms |
| 400 instances of three meshes, rotated, non-uniformly scaled | 65,536 | 1 | 2e-4 | 276 rays/ms | 20.8k rays/ms |

The worst distance is relative, with a 1 m floor. On hits a few millimetres away the two differ by 1e-5 m: the instance
transform's float rounding.

- **xGPU.**
  - `xgpu::compute` (a compute shader, up to 8 storage buffers at set 0, push constants) is created with `device::Create`.
  - `window::Dispatch` records it in the frame's command buffer between render passes, like `CopyBuffer`, with barriers
    before and after (draws, copies, host writes, other dispatches; later shaders and a readback after the frame). Its
    descriptor set comes from the frame's ring.
  - The shader stage gained COMPUTE. Readback uses `CPU_WRITE_GPU_READ` storage buffers (host-visible), so no new buffer
    usage was needed.
- **The rays' structures (`gi::world::PackGPU`).** Every mesh's BVH4 nodes and the TLAS's go in one array, with global child
  codes. The nodes and the 4-triangle rows are copied as they are, since `bvh::node` and `tri4` already have the std430
  layout. Per TLAS primitive it writes the world-to-mesh rows, the root code and the world id.
- **The walk (`shaders/xlionrender_gi_bvh.glsl`).** It is `Cast`'s, mirrored: the same 4-wide box test with the near plane per
  axis, nearest children first, a stack of 64, Möller–Trumbore per row, and the same `SafeInverse`. A mesh is walked with the
  ray in its space, at the same parameter t.
  - Not handled in G1: see-through triangles (added in G2) and an any-hit shadow query (still to do).
- **What it means for the target.** About 0.4 ms of GPU time per million rays in a dense instanced scene. Police's roughly 8M
  rays to converge are about 0.4 s of GPU time in all, so 1 s is within reach once G2 and G3 move the rest of the surfel pass
  to the GPU.

**G2 results** (2026-10-17, RTX 4070 Ti SUPER, Release; `xlionrender_test_gi_surfel_pass`). Every surfel of each scene gets
16 rays (32 for the sign), and its sums are compared with the CPU's (`gi::world::TraceReference`, which is `TraceChunk` over
the same rays). The sums are the light, its share per basis direction, the depth moments, and the back-face, unsupported and
hit counts.

| Scene | Surfels | Differ from the CPU | Worst of the others | CPU (one thread) | GPU |
|---|---|---|---|---|---|
| Cornell box (the box rotated), point light, read grid | 226 | 0 | 3e-6 | 1.1k rays/ms | 2.7k rays/ms (226 threads) |
| The same, 64 passes per surfel (the GPU full) | 14,464 | 0 | 4e-5 | 1.2k rays/ms | **66.7k rays/ms** |
| Cornell box, vertex reads (no read grid) | 226 | 0 | 3e-6 | 1.5k rays/ms | 3.6k rays/ms |
| A see-through pane over a floor and a slab, the sun and the sky by direction | 604 | 0 | 4e-6 | 4.1k rays/ms | 23k rays/ms |
| An emissive sign over the ground, the emitter links off | 330 | 0 | 9e-6 | 3.5k rays/ms | 18k rays/ms |

"Differ" means a count differs, or a sum is off by more than 1e-3 of the surfel's light (for the moments, of its largest
moment). A GPU ray here is a whole ray of the pass: the cast, the shadow rays, the reads and the material.

- **The pass (`shaders/xlionrender_gi_surfel_pass_comp.glsl`).**
  - One invocation per work item traces its rays in `TraceChunk`'s order and writes the item's sums.
  - The CPU keeps the finishing step: the emitter links, E = π·mean(L), and the validity.
  - The hit's light is a mirror of the CPU's: `SurfaceAt`, `Indirect` (`ReadPoint` through the read grid with `Visibility`,
    or `ReadVertex` with the level-1 parent dedupe), `Unsupported`, `Direct` (sun and point lights), and `SkyAt`.
- **The data (`gi::world::PackGPUSurfels`).** One array of words: per mesh a header of offsets, per instance a record, then the
  lights and the setup. The layout is written beside the function.
- **The BVH walk** gained see-through triangles. A hook in `xlionrender_gi_bvh.glsl` reads the instance's bits.
  - Shadow rays turn see-through off, as the CPU's `Occluded` does. They use the nearest-hit walk, so they are correct but
    slower than an any-hit query would be.
- **Not covered yet.**
  - Level-1 (LOD) instances. Their reads are ported but untested here.
  - The whole scene is packed on every call. G3 keeps it resident.
- **What it means for the target.** 66.7k full rays/ms is about 57 times one CPU thread. Police's roughly 8M rays are about
  0.12 s of GPU time, once G3 makes the pass resident and has it pick its own work.

**G3 as built** (2026-10-17).
- **The CPU keeps the pick and, in G3, the commit.** As "What stays" says, the policy (priority, fairness, urgency, roles,
  tiers, adaptive rays) does not move.
  - Committing on the CPU kept one code path for both traces, so a GPU world can be compared with a CPU world update by
    update.
  - The measurements below show that this decision is what limits G3, so G4 moves the commit.
- **The scene is resident** (`gi::gpu_trace`, `features/gi/xlionrender_surfel_gi_gpu.h`).
  - The BVH and `PackGPUSurfels`' data sit in **device-local** buffers. xGPU's `GPU_READ` usage copies them through staging.
    At first they were host-visible, so Police's rays read the scene across PCIe and ran 70 times slower than the Cornell
    box's, while doing less work per ray.
  - Everything is packed again only when the world's structure changes (`world::GPUGen`: add, remove, move, material,
    see-through, a level switch, the emitter set), or when the number of point lights changes. ponytail: a full repack costs
    about 250 ms in the Debug editor, and the material captures at load cause seven of them; in-place material updates are
    still to do.
  - Per slice, the changed words (the surfels committed, the instances whose moments were reset or which hold a cube, the
    means, the lights) go up as (index, value) pairs. A scatter compute shader writes them
    (`shaders/xlionrender_gi_scatter_comp.glsl`). The first version used one `vkCmdCopyBuffer` region per run of words, and
    the Debug validation layer compares every region with every other: 42 s for 7,000 surfels.
- **Over time, as the CPU's trace was.** A batch is the world's pick: `BatchRays`, four frames of slices.
  - Each frame traces the next slice of it that fits `m_GPUBudgetMs`.
  - The frame after, the slice's sums go to `world::DeliverGPU`, and the next `Update` commits them.
  - No frame has to finish a batch. The world picks again once the whole batch is delivered: `Update` picks when no item is
    unfinished, so the update that commits the last slice also picks.
- **The pass spreads a surfel's rays over 8 lanes** (a subgroup cluster, `subgroupClusteredAdd`; SPIR-V for Vulkan 1.2,
  which xGPU requires). With one invocation per surfel, a small batch was bound by latency (426 items, 3.4k rays, 5 ms). Now
  the full Cornell batch runs at 267k rays/ms, against 66.7k before.
- **The budget is a controller on the slices' measured GPU time.**
  - Within the budget it grows by at least a quarter (at most doubling); past it, it shrinks to the fit.
  - A single rate for every batch kept the budget at a small batch's.
- **Shadow rays are any-hit** (`g_GIBvhAnyHit`): the first hit ends the walk, as in `world::Occluded`.
- **Fixed on the way.**
  - The shown light's deadline: with the allowance all pre-charged (the GPU's part, the uploads), it was 0, which means no
    limit, and the filter ran 6 ms a frame.
  - `UpdateShown` with a quarter of an instance dirty or more: the whole filter replaces the dirty chunks' halos. Each chunk
    made its own halo again: millions of surfels for one floor.
  - Tests: 6j19 in `xlionrender_test_surfel_gi`.

**G3 results** (RTX 4070 Ti SUPER):

| Test | Result |
|---|---|
| `xlionrender_test_gi_trace`, Cornell box (point light; emissive panel through links), 8 updates, one slice per batch | 0 of 226 surfels differ from the CPU world, worst 2e-6; packed once |
| The same, the budget at its floor (1,024 rays a slice), 40 batches | the instances' mean light within 0.8 % of the CPU world's |
| `xlionrender_test_gi_render` | traces on the GPU by default; the GI images unchanged |
| Police, Debug editor, 2 ms budget | 450–600k rays/s at 56–58 fps (the CPU trace: about 30k). The 8M rays take about 15 s, not 1 s. GI main thread 8 ms a frame |
| Police, the same with an 8 ms budget (a measurement only) | 840k rays/s at 16 fps. The GI's main-thread work is 42 ms a frame: 25–29 ms is the per-surfel work of collect, commit and refresh, at 6.5k surfels a frame |
| Police, the GPU's clocks during the editor runs | 210–450 MHz (P3, 25–35 % use; the maximum is 3,105). The editor never loads the GPU enough to clock it up, so Police gets 4.7k–6.6k rays/ms |

- Per Police ray: 19.5 BVH nodes, 0.1 shadow rays and almost no read entries (`gpu_result::m_Work`, in the frame log). That is
  less work than a Cornell ray does; the time is memory and clocks.
- **Not met: "Police converges in about 1 s".** Two limits remain:
  - The CPU's per-surfel work, about 4 µs a surfel in Debug. G4 removes it.
  - A GPU that stays at low clocks in the editor. Nothing in the application can control that; a game that loads the GPU
    runs at full clocks.

**G4a, G4b as built** (2026-10-17).
- **The commit on the GPU** (`shaders/xlionrender_gi_commit_comp.glsl`). It runs after each slice's pass and ports `FinishPass`
  and `Update`'s commit loop, `estimator::Add` included. The record grew to 47 words: fractions, estimators (`PackEst`, 21
  words), moment counts, the links' emitted light (re-sent when `EmitGen` changes), the transparent surfels' bits.
- **The world's copy.** Each committed surfel's state comes back (`gpu_state`, 91 words) and `DeliverGPUState` copies it
  in. `Update` only does the bookkeeping for those (`m_bCommitted`).
- **The estimator changes as ops** (`TakeGPUOps`; `shaders/xlionrender_gi_ops_comp.glsl`): `Lower` at a dirty instance's
  pass and in `ApplyAffect`, with the moment counts' reset.
  - The world applies an op to its copy at once; the GPU applies it before the next slice.
  - A change made while a slice is in flight (the renderer's material captures come after the dispatch) comes back without
    the op, so `DeliverGPUState` applies the ops the GPU has not had yet again (both are the same applied twice).
    Without that, the repack from the copy lost them: the render test's light was 3 % darker and asked for 18 % fewer rays.
- **The shown light on the GPU** (`shaders/xlionrender_gi_filter_comp.glsl`, `gpu_trace::Show`).
  - Pass 0, the filter passes (the meshes' neighbor lists, packed with the scene), then saturated.
  - It is written straight into the renderer's ring with the moments (the CPU's `Half`, bit for bit). The CPU writes only
    each instance's mean entry.
  - `m_Setup.m_bGPUShown` skips `UpdateShown`.
  - ponytail: every instance's whole filter every frame. The dirty-only version can come when it shows.
- **`m_GPUBudgetMs = 0`** means no limit: every batch in one slice, as `m_TraceAllowanceMs = 0` does for the CPU (the render
  test uses it).
- **Tests** (`xlionrender_test_gi_trace`). In each, the GPU world matches the CPU world on every surfel (light, validity,
  estimator):
  - the Cornell box after 24 updates, with the adaptive rays and the tiers on;
  - the Cornell box with events: a wall recolored, another's triangles captured in two chunks, one recolored while a slice
    is in flight, the box moved, the coarse level. Without the re-applied ops, 44 of 48 surfels differ;
  - the render test's room;
  - and the GPU's shown light against the CPU's filter of the same state: 3e-7, and 0 moment words differ.

**G4a, G4b results** (Police, Debug editor, 2 ms budget):

| | G3 | G4a + G4b |
|---|---|---|
| Rays per second | 450–600k | 890–910k |
| GI main thread per frame | 8 ms | 4.6 ms: the pick 1.9, `Collect` + `Submit` 2.2, the filter 0 |
| Surfels committed per frame | 1,150–1,300 | 2,300 |

- 8M rays are now about 9 s.
- For 1 s, the GPU must commit about 20k surfels a frame. The CPU's remaining work is per surfel (the pick, the list, the
  copy), at about 2 µs a surfel in Debug: 40 ms. So it must go.

**G4c as built** (2026-10-17). No per-surfel work on the CPU; the light lives only on the GPU (`m_Setup.m_bGPUOwned`).
- **The pick hands out instance ranges** (`world::PickRanges`, `GPURanges`): an instance's urgent surfels one each, then its
  pass from the cursor, sized by what a step of it cost (`m_RaysPerStep`, from the ranges' stats).
  - The expand shader (`xlionrender_gi_expand_comp.glsl`) makes the surfels' items: the tiers (`TracePeriod`), the adaptive
    rays (`RequestedRays`) from the GPU's estimators, and the offsets.
  - The offsets are a hash of (update seed, instance, surfel) on both sides (`world::PassOffsets`), so the CPU's pick draws
    the same rays.
  - The commit adds each item to its range's stats with atomics (rays, surfels traced, unsupported, hits), and
    `DeliverGPURanges` hands them to the world.
- **The world's copy is gone.**
  - The means and the bad counts come from a reduction per instance (`_means_comp`, `SetGPUMeans`).
  - The cubes are expanded on the GPU (`_cube_comp`).
  - A repack moves the light from the old layout into the new one (`_migrate_comp`): arrays copied at the same level,
    re-levelled at another (a child takes its parent's light, a parent its children's mean, the rest new as `ResetMoments`
    makes it). `ResetMoments` now makes the estimators all new on the CPU too.
  - `ReadBack` fills the world's arrays on request: the tests, `DumpGI` (written a frame or two later), the CPU taking over
    when the GPU is turned off.
- **No repack for a material or a move.**
  - A material (the renderer's captures) is changed in place: the arrays are always allocated, record word 47 holds the
    flags, and `world::GPUMaterial` and `MatGen` say what changed. The arrays go up as runs of words.
  - A move (`GPUXformGen`) rebuilds the TLAS only. The meshes' BVHs now come first and the TLAS after them, with a reserve
    in the node buffer (`PackGPUTlas`, `Retlas`); the records' placements are refreshed.
  - The emissive scale and the "no emitters" flag are setup words refreshed every slice.
  - Repacks are left for add, remove and a level switch. ponytail: a level switch still repacks; both levels' arrays
    allocated per instance would remove that.
- **The emitters' light before an instance's links are built** comes from its rays' hits, on the CPU and the GPU alike
  (`I.m_EmitB.empty()`; record word 47's bit 5). Before, it was missing until the links came: for Police, a minute.
  - In GPU mode the link pump gets everything left of the allowance, since the CPU trace needs none.
- **Tests** (`xlionrender_test_gi_trace`, through `ReadBack`). The GPU world matches the CPU world on every surfel:
  - the Cornell box (24 updates, the adaptive rays and the tiers on);
  - the box with an emissive panel;
  - the events: a recolor, a triangle capture, a change made in flight, a four-update drag through the TLAS only, the coarse
    level through the migration. Packed twice: the first pack and the level switch;
  - the render test's room;
  - and the shown light against the CPU's filter: 4e-7, 0 moment words differ.
  - Over time, at the smallest budget (sliced; the slices commit as they come), the GPU world is within 1.8 % at 80 passes.

**G4c results** (Police, Debug editor, RTX 4070 Ti SUPER, `m_GPUBudgetMs` 2 ms; times from the first GI frame, from its frame
log's new `TimeMs` and `MeanLum`):

| | CPU trace (Phases 0–7) | G3 | G4a + G4b | G4c |
|---|---|---|---|---|
| Rays per second | about 30k | 450–600k | 900k | **4.2–5.3M** (85–94k a frame) |
| GI main thread per frame | 1.84 ms (median) | 8 ms | 4.6 ms | **0.8–0.9 ms**; 1.6 ms while the links build (they use the rest of the allowance) |
| A drag (the puppy, p0_drag) | 2.45 ms mean, 5.8 max | | | **1.89 ms mean, 3.7 max**; 0.9 before and after |
| Scene packs at load | | 8 | 8 | 3 |
| 8M rays | a few minutes | about 15 s | about 9 s | **2.6 s** |

- The light is within 10 % at 3.3 s and within 5 % at 4.4 s, of the level it holds before the links. The material captures
  end at 2 s (24 frames).
- Before the links the hits' light is biased low: at 2–8 adaptive rays a surfel the estimator's outlier clamp cuts the rare
  bright hit on a small lamp. Police's links (250k, 350 emitters) are published at about 25 s, and the light rises 40 % to
  its level (0.021 to 0.030).
- **The link build is now what Police waits for.** It is 63k receivers × 350 emitters of visibility tests on the CPU,
  under the main thread's allowance. On the GPU that is about 22M shadow rays, a fraction of a second.
- The first 2 s are the Debug editor's load: three packs at about 350 ms each, the captures.
- **Mobile:** declared (`m_GPUBudgetMs`); not measured, there is no device here.
- The GPU ran at P3 (about 400 MHz) in these runs.

**G5 design: the emitters' light by light sampling** (2026-10-17; replaces the CPU link build for the solve).
- **What it estimates.** It is the links' sum for a surfel: over the emitters it sees, L_e·w_e, with w_e = A·cos_r·cos_e / (d² + A).
  That sum is split into the basis by the direction to each emitter (3·Bw_i), and goes where the links' `EmitB` went in the
  pass' finish.
- **How.** Each pass, each of a surfel's 8 lanes takes one sample:
  - an emitter picked by weighted reservoir sampling over the emitter table (weight w_e·lum(L_e)), with one shadow ray to it;
  - the estimate is L_e·Q/lum(L_e) if the emitter is visible, Q being the sum of the weights;
  - the 8 samples' mean is added to the pass.
  - The random numbers are hashes of the pass' offsets, the lane and the emitter, so the CPU (`TraceChunk`) and the GPU pass
    draw the same emitters. The estimator averages the samples over passes.
- **The links are not built** (`m_Setup.m_bEmitterSampling`), and the rays' hits leave the emission out (the sampling brings
  it).
  - The emitter list is made again every update: it is small, and it follows moved emitters.
  - On the GPU it is a table in the data, with its count and capacity in the header (words 13..15), refreshed every slice.
- **The renderer turns it on** (`scene::gi_settings::m_bEmitterSampling`). The world's default stays off, so the links'
  tests still test the links.
- **Done when.** The GPU world equals the CPU world with sampling, the sampled light equals the links' and a many-ray
  reference within its noise, and Police no longer waits for the links.

**G5 results** (2026-10-17):
- **Tests.**
  - `xlionrender_test_surfel_gi` 6j: under a 20 cm sign the light sampling gives 0.637 (spread 11 %). The links give 0.637
    (11 %), the reference 0.636, the hits 0.161 (141 %).
  - `xlionrender_test_gi_surfel_pass`: the sign by sampling, the GPU's pass = the CPU's (1e-6).
  - `xlionrender_test_gi_trace`: the Cornell panel by sampling, the GPU world = the CPU world after 24 updates (5e-7).
- **Police** (Debug editor, 2 ms GPU budget):
  - No links are built. The light settles at 0.0334; the links reached 0.030 after 25 s, their caps on the emitters a
    surfel takes losing some.
  - The light is within 10 % at 3.65 s and within 5 % at 5 s from the first GI frame. The material captures end at 2.9 s
    (24 frames: the Debug editor's start), and the light is within 10 % 0.7 s after them.
  - GI main thread 0.57 ms a frame; 85–90k rays a frame.

**The flicker** (2026-10-17, measured: `xlionrender_test_gi_flicker`, and Police's light dumped 0.5 s apart once settled).
- **Cause.** The change detector (`estimator::Lower`, then `Add` while watching) compared the samples with the light the surfel
  had when the change came, and never moved that reference. Two samples 50 % off it the same way replaced the light with their
  mean. So a noisy surfel took a "change" every few passes, and its light was a two-sample mean for ever.
  - In Police, two thirds of the surfels were still watching after 25 s, at 2–5 samples, and the worst 1 % jumped 30–40 % in
    half a second. The material captures had lowered them.
- **Fix** (the CPU estimator and the GPU commit alike).
  - A change must be beyond 3 times the samples' noise, measured from the steps between samples that did not deviate (for
    noise a step is about √2 σ, for a ramp small, for a change of level one jump).
  - The two samples must agree with each other.
  - After a take, the reference is the warm-up's mean, so a ramp is still followed.
- **Results.**
  - Police: the lit surfels' p99 change over half a second went from 31 % to 0.3 %, and no surfel is left watching.
  - 6j18: noise makes 0 takes (it made 8 with only the reference moving), and the ramp reaches 4.90.
  - The convergence: within 10 % at 2.9 s; the last 5 % now takes about 9 s, since small changes are followed by the
    estimator's averaging rather than by takes.


**Risks.**
- xGPU has no compute path yet. It declares a compute stage, but nothing creates a compute pipeline or dispatches one.
- Debugging on the GPU means readbacks.
- Divergent traversal performance on mobile GPUs.
- Synchronising the CPU's policy with GPU-resident state, especially the change events against work in flight.

### Held from earlier work (not part of this plan's critical path)

- The dark shade in Sponza at `SurfelSize` 1 (little light reaches it; not a solver defect so far).
- The pale "hidden" ghost under the Police building's outline (geometry the mask draws and the scene does not).

## 4. Acceptance tests

**The envelope.** Bounded queues and eventual service can only both be promised inside a declared workload: a maximum of
loaded instances and surfels, resident emitters, queued changes, and the memory allowance. The first envelope is the Police
level with the puppy and up to four moving objects; it is widened as measurements allow.

| Test | Setup | Expected |
|---|---|---|
| A. Receiver-only motion | The puppy, `RECEIVE`, dragged around and between rooms | Its own light adapts to the new place within the measured recovery time; the GI field elsewhere does not change; no link work is scheduled |
| B. Full-transport motion | A wall dragged, then released | No frame over allowance plus one unit on the converted paths; useful publications during the drag; old occlusion and bounce fade out, new ones appear; convergence to the reference below |
| C. Blocking event | A door closed between an emitter and a receiver | Direct visibility updates at once; indirect light responds; no stale emitter link remains |
| D. Overload | Continuous and repeated movement, several full-transport movers, a moving emitter, rapid light toggling, deleting an object during a pending build, several renderers open, a release with a large backlog | No synchronous catch-up, bounded queues and memory, no incompatible publication, every frame within allowance plus one unit on the converted paths, aged work still served, convergence after the changes stop |
| E. Unrelated edits | Repeated edits far from a large pending candidate | The candidate is not invalidated and still publishes |

**Three different comparisons**, never mixed:
1. **Link coherence** against a deterministic fresh link build of the same final scene (exact).
2. **Lighting convergence** against the same solver with the same roles, freshly initialised on the final scene. This is
   statistical: a stated tolerance on the mean irradiance per region, a fixed seed and initialisation procedure, and a stated
   number of passes, never a pixel-for-pixel match of two noisy solves. A fresh solve can share a defect: it checks coherence,
   it is not an absolute reference.
3. **Physical quality** against analytic or higher-quality references (the Cornell box, the room tests of the unit suite).

Each report keeps per-frame cost by stage, the longest unit, recovery error, queue age, memory by category, and the receiver's
appearance. A speed-up is attributed to its cause: better scheduling, less work, or a deliberately simpler role.

## 5. Where this beats the competition, and where it does not (revised 2026-10-17, after G1–G5)

How to read this section:
- The references are the engine study (Unreal's Lumen, O3DE and Flax's DDGI, Godot's SDFGI and VoxelGI, CryEngine's SVOGI) and
  the surfel systems read in 5b–5d (Wicked Engine, GIBS, SurfelGI, SurfelPlus, Juretriglav's WebGPU surfel GI).
- Ours are measured: Police (63k surfels, 350 emitters) in the Debug editor on an RTX 4070 Ti SUPER running at P3 clocks
  (210–450 MHz: the editor never loads the GPU enough to clock it up), and the unit tests.
- Theirs come from reading their code or papers. **None of them has been run side by side with ours on the same scene**, so
  no claim below says "faster than X" in milliseconds.

**Where we are ahead (measured or tested here)**

| What | Ours (measured) | The references (as read) |
|---|---|---|
| Radiometric correctness | Checked analytically: an open white sky gives E = π (test 6m), the Cornell box matches a path tracer (test 3), light sampling matches a many-ray reference within 0.2 % (6j). | SurfelGI looks 2× too dark and SurfelPlus π× too bright per bounce, from reading their code (5d; unconfirmed on an analytic case). |
| A CPU reference for the GPU | The GPU world equals the CPU world surfel for surfel (light, validity, estimator; 5e-7) through events: a recolor, a capture, a change in flight, a drag, a level switch (`xlionrender_test_gi_trace`). The shown light equals the CPU's filter to 4e-7, the moments bit for bit. | None of the open surfel implementations has a reference solver; the engines test visually. |
| Main-thread cost | 0.57–0.9 ms a frame settled, with nothing per surfel on the CPU (G4c). A drag: 1.9 ms mean, 3.7 ms max (it was 2.45 / 5.8 on the CPU solver). In Debug. | GPU-driven systems (Lumen, GIBS) are also near zero on the CPU. Parity, not a lead; the lead is that it holds in the Debug editor. |
| No hardware ray tracing | Software BVH4 traversal in compute: 267k full surfel rays/ms on the Cornell batch; 4.2–5.3M rays/s in Police at a 2 ms budget and low clocks. | Juretriglav and three-mesh-bvh also do software traversal. GIBS needs hardware RT. Lumen falls back to software traces of distance fields. |
| Many small emitters | Weighted light sampling over every emitter, unbiased. Police's 350 emitters need no build; under 64 lamps the links' caps lost 17 % of the light, the sampling none. | Wicked picks one light per ray, uniformly. GIBS and the open ports rely on the hemisphere rays (a small lamp is rarely hit). |
| Stability once settled | Police's lit surfels: p99 change 0.3 % over half a second; the room, 0.04 % an update (`xlionrender_test_gi_flicker`, which now guards it). | The estimator is Wicked's (MultiscaleMeanEstimator); ours adds a change detector that is noise-aware (after the flicker fix). |
| Bounded work, by design and in tests | A per-frame GPU budget (a controller on the measured slice time) and a CPU allowance with an overload test. Effort follows screen area (levels, ambient cubes, distance tiers). | All of them amortise. The two open GIBS ports cap by ray count only, never by time (5d). |
| Mesh-local representation | Baked surfels, read grids and BVHs are per mesh. A moved instance rebuilds only the TLAS (`Retlas`), a recolor only its material arrays (no repack). | Lumen's cards are also mesh-local. Screen-spawned surfels (GIBS, Wicked) are respawned instead. |

**Where we are behind, or not yet proven**
- **Time to resolution.** Police, counted from the first GI frame in the Debug editor: 8M rays at 2.2 s, the light within
  10 % at 2.9 s, within 5 % at about 9 s. The Debug start (the material captures, 1.6–2.9 s; three packs) is in that time. The
  last 5 % is slow because the estimator follows small changes by its averaging (the price of the flicker fix). The 1 s target
  is not met. Lumen's screen probes and radiance cache react within frames for what is on screen.
- **Variance with hidden emitters.** The emitter pick weighs each emitter by its unoccluded weight, so behind walls most samples
  pick hidden lamps. It is unbiased but noisy; the estimator hides it at the cost of convergence. ReSTIR-style temporal reuse
  was tried and measured worse (section 7, item 1).
- **Diffuse only.** Glossy indirect comes from the reflection probes and the sky. Lumen and SVOGI do glossy indirect.
- **Detail is bounded by surfel size.** The per-pixel read grid helps at edges and walls, but features smaller than a surfel
  are averaged. Lumen's screen traces keep small detail on screen.
- **No ray guiding.** The rays are cosine-distributed. GIBS guides them by a per-surfel radiance map (5c, with the pdf done
  right).
- **No hardware RT path.** Where hardware RT exists we don't use it; the software walk is all there is.
- **Skinned and transparent receivers.** Baked surfels follow rigid instances only. GIBS uses world probes for what surfels
  cannot cover.
- **Scale and mobile.** Not proven on open worlds (memory, the 131,072-surfel ring region). The mobile GPU budget is declared
  (`m_GPUBudgetMs`), not measured: there is no device here.
- **A level switch still repacks the scene:** about 150 ms in Debug since the pack's rewrite, a hitch when the camera crosses
  the switch distance.

**Verdict.** The architecture is in the same class as the GPU surfel systems (GIBS, Wicked): a world-space surfel cache, solved
on the GPU, read per pixel. It is ahead on correctness, testability, light handling and cost discipline. It is behind on
response time, glossy light and variance reduction. Those are specific items with known techniques (section 7), not a
reason to change the approach.

## 5b. What Wicked Engine's surfel GI adds (the local checkout: a heavily modified fork, not upstream)

Already in xLION: its `MultiscaleMeanEstimator` with the firefly clamp (`gi::estimator`), 4 x 4 tangent-space depth moments,
the plane-distance anti-leak term, multiple bounces through the previous light at a hit, a display-only spatial filter, and a
re-trace period by distance with a staggered phase (the distance tiers).

Taken into the phases:
- **No invalidation on a move or a light change** (Phase 4). Wicked resets nothing; the estimator's inconsistency drives the
  catch-up. Ours already has the same estimator, but a dirty instance resets its history, which sends it back to the 1/n
  warm-up average and a first pass that replaces the published light. Keeping the estimator running (warm start) and letting
  its inconsistency react is the first thing Phase 4 tries.
- **Rays by inconsistency** (Phases 1 and 6): a surfel asks for rays in proportion to its inconsistency, within the budget, so
  changed surfels get more and settled ones fewer. Explicit "dirty" bumps (inconsistency raised for the surfels a changed light
  or mesh reaches) cover what Wicked has no trigger for.
- **Recency** (Phase 1): an instance not seen for a while gets 1 ray per surfel, then none, from a per-instance "seen" mark of
  the reads. This sits inside the fairness rule: an off-screen surfel with transport relevance keeps being served.
- **Seeding from neighbours** (Phase 4): a surfel whose history is rejected starts from its neighbours' light, clamped to the
  brightest of them, and fades in.
- **Guided bounce rays** toward the dominant direction of the basis, if Phase 0 shows noise is the limit. Unverified for our
  3-direction basis; measured before it is kept.

Not taken (GPU or screen-space specific): screen-space spawning and recycling (our surfels are baked), the hashed cascade grid
and its binning, ray sorting, the gather skip, cascade dithering. Wicked picks one light per ray uniformly; our explicit emitter
links stay.

## 5c. What GIBS (EA SEED, SIGGRAPH 2021) adds

From a review of the presentation (`seed-siggraph21-surfel-gi.pdf`; only its opening was read directly here). GIBS is GPU-driven
with hardware ray tracing and spawns surfels at runtime; its performance numbers say nothing about a CPU solver under 2 ms.

Taken into the phases:
- **Rays requested, budget granted** (Phase 1, then 6). Three separate questions: does a surfel need service (change events,
  age, transport relevance), how many samples would help (uncertainty from the estimator's inconsistency, recent change,
  initialisation), and how much can be afforded (the scheduler's measured allowance, which stays the final authority). Settled
  surfels go almost dormant (a minimum sampling rate that still discovers changes); fairness stays independent. This also
  solves Phase 0's oversized unit: rays are granted in chunks, so a surfel's 8+ rays are no longer one indivisible unit.
  Test: at equal maintenance time, fixed versus adaptive allocation on a settled interior, a light toggle, an opening door and
  a moving full-transport object, judged by recovery error over time.
- **Sharing by confidence** (Phase 3). The display blends raw and neighbourhood light by each surfel's uncertainty and support:
  a settled, well-supported surfel shares little, a new or noisy one more, never across an incompatible layer (the same
  neighbour rules as today). One pass with a configurable strength first, after the whole-instance resumable filter is correct.
  It should also cut the 4.0 ms filter cost.
- **Bounded light sampling at the hits** (after a measurement). Each hit evaluates the direct light of every light in reach with
  a shadow ray each; if that dominates the trace, small light sets stay exact and larger ones pick one or a few lights from cheap
  candidate weights (a reservoir), with correctly derived weights, tracing only the chosen shadow rays. Visible surfaces keep
  their direct lighting unchanged. First, Phase 1 splits the trace's time into casting, direct light and reads, to know.
- **Directional guiding** (after the above are stable). A small per-surfel directional map of incoming radiance (GIBS: 6 x 6)
  and a mixture of it with cosine sampling, so window-lit rooms and bright walls get the rays. The estimator must change with
  the distribution: E = (1/N) sum L cos / q, not pi x mean(L); the basis accumulation likewise; the distribution is frozen for a
  batch; the depth moments keep their own unguided rays. Compared with cosine sampling at equal cost before it is kept.
- **Resident-cache accounting** (rule 5, now). Mesh assets (surfels, read grid, BVH) versus resident instance GI (light,
  moments, confidence, guiding data, pending work, published GPU storage): the second has limits and a retention rule, and a
  useful result survives temporary invisibility while memory allows.

Later or not at all:
- Ray binning by position and direction: only if traversal dominates, measured end to end (generation, sorting, tracing,
  shading), and itself resumable.
- A world-space receiver cache (GIBS's probes for transparent surfaces), reusing the tracer, the query, the scheduler and the
  estimator, if test A shows unacceptable receiver latency or skinned meshes need GI.
- Not adopted: screen-space spawning (surfels are baked), the first-N reads with a cell-average fallback (it reopens the
  fail-open reads removed earlier), single-bone attachment for skinned surfels, the nonlinear grid, blue-noise LOD selection.

## 5d. What two open GIBS implementations add (SurfelGI on Falcor, SurfelPlus on Vulkan)

From a read of their sources (`E:\source_code\surfles\SurfelGI`, `E:\source_code\surfles\SurfelPlus`; SurfelPlus is a SIGGRAPH 2025
poster for low-end hardware). Both are GPU-driven, spawn surfels from the screen and cap their cost by a ray count only, never by
time. **From reading their code, not shown on an analytic test:**
- **SurfelGI** draws its rays uniformly over the hemisphere (pdf 1/2π per steradian). Its weight cancels to 1, so a surfel stores
  mean(L cos θ) = E/2π, and its shading multiplies that by the albedo as if the rays were cosine-distributed. That would be a fixed
  factor of 2 too dark if no other factor is applied elsewhere.
- **SurfelPlus** stores E = mean(L cos θ / pdf), which is irradiance, and its shading has the 1/π commented out. That would be π
  too bright per bounce.

Both would have to be confirmed on an analytic case (a uniform white sky over a plane: E = π) before being quoted as fact. Ours
was checked: cosine rays (pdf cos θ/π), E = π × mean(L) published as irradiance, and a hit's outgoing radiance is albedo × E/π.
Test 6m compares an open sky with π (3.142), and test 3 compares the Cornell box with a path tracer. We also already have three things that
either one lacks, done more soundly:
- a Chebyshev variance floor and bleeding reduction (the term cubed);
- moments capped at the surfel's reach;
- an estimator seeded from the current light.

Taken into the phases:
- **Concrete constants for "rays requested"** (Phase 6, inside Phase 1's allowance; it refines 5c's first item). Both use
  rays = lerp(low, high, k x estimator variance): SurfelGI uses 16..64 rays, SurfelPlus 4..64. A surfel not seen this frame
  gets a quarter of that, and a new surfel gets the full count for its first 20 updates. A change raises the variance on
  purpose (SurfelPlus: variance x (1 + 10 x change)), so a moved surfel asks for more rays. For us, the change would be
  SetTransform or a dirty mark.
  - Guard the feedback loop. The estimator's variance is that of a batch mean, so more rays lower it, and the lower variance
    then cuts the rays. Normalise it by the batch's ray count, or use the deviation relative to the luminance.
  - Neither one is fair once over budget: they fill in index order and starve the higher indices. Our oldest-first share stays.
- **Snapshot reads** (Phase 3). Each update starts by copying every surfel's mean into its published light, and every reader
  uses only the published value: the pixels, the neighbour sharing and the hits of other surfels' rays. On the CPU this is the
  publication double buffer of Phase 3, and it also lets the workers read other surfels without locks.
- **Sharing as one estimator sample** (Phase 3; it refines 5c's "sharing by confidence").
  - Today a separate display filter runs (3.5 ms). In SurfelPlus, the neighbours within two radii are instead fed into the
    estimator as one extra sample with a small blend (0.04).
  - Neighbour weights: normal times distance, times the neighbour's age (smoothstep over 10 updates). A neighbour is rejected
    when either one faces the other's back (cos < -0.2).
  - SurfelGI lerps toward the neighbour mean by saturate(40 x variance).
  - The log of SurfelPlus says this sped up convergence "a lot". For us it can run over the fine siblings or the parent level,
    in the solver's own pass, and may replace part of the filter and its cost. Test: equal time, against the current filter,
    on the settled interior and a light toggle.
- **The read grid built aside, then swapped** (Phase 2, as an alternative to incremental links). SurfelGI rebuilds its whole
  surfel grid every frame as a counting sort: count, prefix, scatter. A rebuild that runs as a job into a second buffer, with
  the buffers swapped when it is done, never stalls a release. Measure it against the bounded incremental links. It depends
  on whether the emitter links can be rebuilt the same way, which needs one pass per emitter set.
- **Fading in what is new** (Phase 3, at the level switch and when an instance is first lit). SurfelGI weighs each surfel in
  the blend by its age (smoothstep over 240 frames), then normalises. A young surfel next to old ones gives way to them, and a
  young one alone still shows at full strength. This keeps a parent-to-fine switch, or a new instance, from popping.
- **A stochastic cache bypass at path ends** (later; it applies only if we trace more than one bounce per ray). SurfelGI skips
  the surfel read 20 % of the time and keeps the path going instead. This lowers the feedback bias of reading the cache. We
  trace one bounce and read the cache at every hit, so this has no use now.

Not adopted:
- Their ray guiding. A discrete texel probability is a valid pdf only when converted to a solid-angle density (divided by the
  texel's solid angle, with the sample spread within the texel). As read, neither does that, and SurfelGI jitters the
  direction afterwards. 5c's guiding stands, to be built with that conversion.
- Screen-space spawning, recycling and lifetimes (our surfels are baked).
- SurfelPlus's denoise chain (it applies only to its reflections; the diffuse GI gets none).
- Their compression (a stock 32-bit octahedral normal).
- Reflections that end on the surfel cache, with 16 GGX candidates (SurfelPlus): a later option for rough specular, after
  this plan.

## 6. Order and dependencies

```
Phase 0 instrumentation
  -> Phase 1 scheduler (tracing scheduled, no synchronous worker wait)
       -> Phase 2 links          (bounded link maintenance; removes the release stall)
       -> Phase 2b consolidation (one estimator, one sharing, one LOD rule, one priority)
       -> Phase 3 publication    (bounded filtering, upload, publication)
            -> Phase 4 state split
                 -> Phase 5 roles      (tests A, B, C)
                 -> Phase 6 relighting (tests D, E)
  Phase 7 first use: after Phase 0 says where the 630 ms goes
  End-to-end contract: only when Phases 1, 2, 3 and 7 all pass
```

One phase at a time; the plan file is updated at the end of each phase with what was measured.

## 7. Next items, in order (proposed 2026-10-17)

Each is measured before and after on Police and guarded by a test, as G1–G5 were.

1. **Visibility-aware emitter sampling (ReSTIR-style temporal reuse).**
   - Each surfel keeps a small reservoir (an emitter, its weight, its count). Each pass merges the reservoir with new
     candidates, and the shadow ray tests the winner. Visible emitters then win and stay.
   - Same hashes on the CPU and the GPU, so the comparisons hold.
   - Done when: under 64 lamps with 60 hidden, the per-pass noise drops by half or more; in Police, within 5 % sooner, with
     no flicker regression.
   - **Tried and rejected (2026-10-17). Do not rebuild it under another name without meeting the conditions at the end.**
     - Tested on xLIONRender dcc2be7 plus an uncommitted patch, never committed.
     - The patch: a reservoir per surfel (emitter, W, M capped at 80, the emitter list's version), merged each pass with the
       pass' 8 candidates, each of them shadow tested. The CPU world and the GPU pass were equal surfel for surfel
       (`xlionrender_test_gi_trace`).
     - The scene: `xlionrender_test_gi_flicker`, 64 lamps with 60 hidden behind a slab, no bounces (every albedo 0), so the
       exact light is known per surfel. The same rays and shadow rays a pass with and without the reuse.
     - Measured, the floor's rms error against the exact light after 4 / 16 / 120 updates:
       - with the reuse 35.5 / 19.6 / 8.2 %;
       - without it 29.4 / 14.8 / 7.8 % (the bias −0.2 % with it, −0.8 % without);
       - the flicker p90 an update 0.54 % with it, 0.35 % without.
     - Why it lost:
       - ReSTIR's gain comes from shadow testing only the winner. Our pass already tests all 8 candidates, and its estimate
         already uses all 8 results (the visible share of the weight).
       - Carrying the reservoir over adds no information. It averages over about 10 passes, and the estimator then
         averages those correlated passes as if they were independent.
     - Revisit only if:
       - a design shadow-tests fewer rays than candidates (the real ReSTIR trade: about the same noise for fewer shadow
         rays); or
       - a scene where a surfel sees only 1–5 % of its emitters' weight shows the estimator's clipping, or flicker, from the
         rare visible samples.
     - The test now guards the sampling's per-surfel error: under 12 % rms after 120 updates, 7.8 % measured.
2. **The convergence tail.**
   - Split "a change of level" from "noise" in the estimator's settled blend: a confidence-aware catch-up once the samples'
     mean has moved beyond their noise for several passes.
   - Fade new light in by age (5d), so a level switch or a new instance doesn't pop.
   - Done when: Police is within 5 % in about 3 s, with the flicker test unchanged.
3. **Remove the level switch's repack.** Allocate both levels' arrays per instance and re-level in place (the migrate shader's
   kind 1, on the same buffer). Done when: a camera dolly across the switch distance shows no frame over 5 ms in Debug.
4. **Faster start.**
   - The material captures are now the first second in Debug: batch them per frame by area, not one mesh at a time.
   - The first pack, which in Debug is still about 150 ms with its uploads: upload in slices.
   - Done when: the captures finish within 1 s of the first GI frame.
5. **Ray guiding** (5c, with the solid-angle pdf), compared with cosine sampling at equal cost. Kept only if it wins.
6. **Glossy indirect from the surfel cache** (rough reflections ending on the cache, as SurfelPlus does with GGX candidates). A
   separate plan.
7. **An optional hardware RT path** for the cast and the shadow rays, behind the same interface. Then mobile: declare and
   measure the budget on a device.

## 7a. The startup diagnosis, and the order after it (2026-10-17)

Items 2 (the convergence tail) and 3 (the level switch) of section 7 are paused by this diagnosis. So are two ideas that grew out of
item 2: "representatives first" (one fine surfel per parent group traced first, seeding its siblings) and "catch-up by group
agreement". The measurements below found simpler causes to fix first.

**How it was measured (reproducible)**
- The tool: `source/Editors/LevelEditor/smoke/gi_startup.py` (`run <tag> [--captures-first]`, then `analyse <tag>`; the metric
  definitions are in its docstring). Dumps go to `Build/gi_startup/`.
- The build: xLION 4a913ea, xLIONRender dcc2be7 plus this diagnostic commit. The Debug editor, the harness's own (never one the
  user runs). RTX 4070 Ti SUPER. The GI GPU-owned, the scene's GI settings as saved, one run each.
- The scene: Police (63,285 level 0 surfels in 23,221 parent groups, 350 emitters), the fixed orbit camera of the script.
- The control: `--captures-first` sets `XLION_GI_HOLD_CAPTURES`. The renderer then solves nothing until the material captures
  have been done for 10 frames (`plans/forward/xlionrender_plan_forward.cpp`, updateGI).
- The error: the solver light (the dump's `e`) against the last dump (about 60 s), area weighted (radius²), split into
  between groups (each parent group's mean error) and within groups (each surfel's error less its group's).

**Results**

| GI time | Today: mean / ref | Today: between, within | Captures first: mean / ref | Captures first: between, within |
|---|---|---|---|---|
| 1 s | 1.98 | 287 %, 195 % | ≈1.03 (the frame log) | (no dump) |
| 3 s | 1.83 | 383 %, 405 % | 1.009 | 12.0 %, 14.5 % |
| 5 s | 1.17 | 76 %, 40 % | 1.009 | 11.8 %, 14.2 % |
| 9 s | 1.07 | 34 %, 17 % | 1.007 | 10.2 %, 13.0 % |
| 15–19 s | 1.02 (15 s) | 12 %, 10 % | 1.002 (19 s) | 5.5 %, 8.1 % |

- Rays by 1 / 3 / 5 / 9 s: today 1.1 / 1.6 / 6.2 / 19.2 M; captures first 5.1 / 14.3 / 15.1 / 18.0 M.
- The captures end at 4.3 s of GI time today. With nothing solving, they take about 1 s.
- Between 2 and 15 s, about 30 % of the frames trace under 5,000 rays (about 1,150), against about 85,000 on the others.
- The parent groups average 2.7 children, and 10 % of them have normals more than about 37° apart.

**What it establishes**
- The startup's brightness error comes from provisional material input. Before its capture, an instance's surfels use its
  material's base color factor (`world::Add`), usually white for textured materials. A 2× overshoot follows, and the
  estimator takes about 15 s to work it off.
- With final materials, the global mean is within about 3 % at 1 s, but a spatial error of 12–15 % remains at 3 s.
- That error then stays flat from 3 to 7 s. Every surfel is out of warm-up and none is watching, so the service drops to about
  1k rays a frame while the surfels are still about 12 % off. Low variation here is not convergence: a surfel that barely
  updates barely changes.

**What it does not establish**
- That the remaining error is the estimator's inertia.
- That the reference is accurate. The 60 s dump changes by only 0.2–0.3 % in 0.5 s because settled surfels are barely served;
  it is suspected of being under-converged.
- That a group-level mechanism is needed. Groups are small, and some are incoherent.

**The order now**
1. **Material readiness.**
   - Before its first surfel capture, an instance stays in the ray scene for occlusion. It gets a provisional lower-energy
     response: zero diffuse and no emission.
   - Each instance gets a material revision. The GI reads the surfel capture; the triangle captures feed only the
     SURFACE_COLOR view.
   - Once all initial captures are done and GPU-visible, one scene-wide confidence transition. The means stay as a prior, the
     history's weight is lowered by the tested policy (no count to zero, so the flicker fix holds), and the geometry moments
     are kept.
   - It happens at a batch boundary without blocking: frames go on, and work picked before it commits under the old revision.
   - A test: streamed captures against captures ready from the start.
2. **Service, measured first.**
   - Per frame: the work requested, admitted and executed, the GPU time, the unused budget, the eligible surfels skipped and
     why, and the time since each surfel's last fresh evaluation.
   - Then one control with the estimator unchanged: today's adaptive policy against fixed service (no dormancy) at the same
     GPU budget.
3. **A reference that is allowed to converge.** A test mode with final materials, no dormancy, fair service, several
   checkpoints and more than one seed, until regional values stop drifting. The next "within 5 %" claim uses it.
4. **The dormancy or service rule**, if 2 points at it.
5. **The estimator's catch-up**, only if a tail remains after 1–4.

**Step 1 done (2026-10-17): material readiness**
- What it does:
  - `world::SetProvisional`. The host sets it when it adds an instance whose captures it queues, and clears it once no capture
    of that instance is queued (all in, or dropped).
  - While it is set, the instance's hits bring no albedo and no emission (`SurfaceAt` on the CPU, flag bit 64 on the GPU) and it
    has no emitters. Its geometry stays in the ray scene.
  - When the last of the initial instances clears, `DirtyAll`: each instance's `estimator::Lower` at its next pick, after the
    batch in flight.
  - Later instances relight only their neighbors.
- Tests:
  - `xlionrender_test_gi_flicker`: the room with its colors streamed over 18 updates, against the captured-first room after 800
    updates. At 200 updates:

    | Run | Peak after the captures | Within 5 % from update | Mean at 200 | rms at 200 |
    |---|---|---|---|---|
    | captured first | 0.95× | 183 | 0.954× | 8.0 % |
    | streamed | 1.65× | never | 1.116× | 18.9 % |
    | provisional | 0.98× | 26 | 0.984× | 4.4 % |

    Even captured first is still 8 % off at 200 updates with every surfel traced every update. Its slow climb from dark is the
    estimator's history, which the provisional run's startup transition drops.
  - `xlionrender_test_gi_surfel_pass` (6): a provisional sign, CPU = GPU, no emitters.
  - `xlionrender_test_gi_render`: now waits for the captures before it checks the bleeding and the per-pixel read. Base colors no
    longer stand in for them.
- Police (`gi_startup.py run ready`), in the same table form as above, today's startup in brackets:

  | GI time | Mean / ref | Between, within |
  |---|---|---|
  | 1.7 s | 0.43 | 207 %, 71 % |
  | 3.75 s | 0.85 | 78 %, 56 % |
  | 5.7 s | 1.06 | 38 %, 23 % |
  | 9.7 s | 1.03 | 19 %, 13 % (34 %, 17 % at 9 s) |
  | 15.7 s | 1.02 | 13 %, 10 % (12 %, 10 % at 15 s) |

  - The overshoot is gone (1.06× at most, it was 1.98×).
  - The captures still end at 4.7 s of GI time, and only 0.1 M rays are traced in the first second. The start is now bounded by
    the captures: captured first, they take about 1 s and the error is 12 % by 3 s.
  - Faster captures (section 7, item 4: batched per frame, not one mesh at a time) is therefore the lever for the first seconds,
    before the service step.
- A trade-off to decide: a constant-color material's base color is already its capture, so it could skip being provisional
  and not be dark for the first seconds. Not done; it needs the host to know which materials are constant.

**Correction (2026-10-17, after step 1): the dumps froze the GI, and there is no service stall**
- What was wrong: each `DumpGI` is a frame of about 0.9 s in Debug (the read-back and the CSV), and the GI makes no progress in it.
  The GI times above include those frames. The captures-first run dumped at 4, 5, 6, 7 and 8 s, so its "flat from 3 to 7 s" was
  0.27 s of real GI time: 2.99 s to 3.26 s.
- `gi_startup.py` now reports the time without the frames the dumps froze. Corrected (between groups, within groups):

  | Effective GI time | Today | Readiness | Captures first |
  |---|---|---|---|
  | about 1 s | 287 %, 195 % (0.80 s) | 207 %, 71 % (0.86 s) | |
  | about 2 s | 76 %, 40 % (2.1 s) | 38 %, 23 % (2.25 s) | |
  | 3 s | | | 12.0 %, 14.5 % |
  | about 5 s | 34 %, 17 % (5.2 s) | 19 %, 13 % (5.3 s) | 10.2 %, 13.0 % (4.3 s) |
  | 7.4 s | | | 7.2 %, 10.0 % |
  | about 10 s | 12 %, 10 % (10.3 s) | 13 %, 10 % (10.4 s) | 5.5 %, 8.1 % (12.5 s) |

- What changes:
  - The captures end at 1.2–1.4 s, not 4.3–4.7 s. Faster captures are a small lever.
  - With stable inputs, the error falls steadily but slowly: 12 % at 3 s, 5.5 % at 12.5 s, at 4–5 M rays/s.
  - About a third of the frames trace about 1.1k rays (the batch's tail: 0.9 ms of the 2 ms GPU budget), against about 80k on
    the others. That is roughly 20–25 % of the throughput.
  - The room test agrees: captured first and every surfel traced every update, it is still 8 % off at 200 updates, while the run
    whose history was lowered once is at 4.4 %. The slow part is the estimator's history over the bounces' build-up, not the
    service.
- The order after step 1 is therefore:
  1. the estimator's lag as the bounces build up (the room test is the clean case: stable inputs, full service);
  2. the batch's tail (frames that leave the GPU budget unused);
  3. a reference allowed to converge.

## 7b. Reflections (2026-10-17)

**Layer 1 done: rough reflections from the surfel cache**
- The read:
  - The per-pixel GI read also weighs each surfel's 3 basis colors toward the reflection direction R (`g_GIReflDir`, R in mesh
    space through the two tangent frames). These are the values it already fetched.
  - The result is `g_GIRefl`, the light around R. A weak read (confidence under 0.25) keeps the probes' reflection.
- `EnvLighting` (`shaders/xlionrender_environment.glsl`):
  - The probes' and the sky's reflection is scaled by min(1, the cache's light around R / their roughest level along R). That is
    a directional normalization, as Unreal 4 scales its captures by the local GI, so a dark corner facing a bright doorway keeps
    the doorway.
  - From roughness 0.3 to 0.7, the cache's light takes over, so a rough floor reflects the red wall's bounce.
  - Their second sample, which was along N for a diffuse the GI replaces anyway, is now along R: the denominator, at no extra
    cost.
  - The vertex read path keeps the old scalar ratio, labelled as a heuristic, not visibility.
- Box projection: the exit distance per axis only where the ray is not parallel to it. The plain reflection when the projection
  fails (outside the box, degenerate).
- Test: `xlionrender_test_gi_render`, a rough metal floor. Red/green is 1.22 by the red wall and 0.85 by the green with GI, 1.00 and
  1.00 without. The old shaders fail it (1.00 and 1.00 with GI).
- Not seen in Police yet:
  - Its materials are SPIR-V from the material compiler (`xmaterial_compiler.cpp`), which includes the shaders when it compiles,
    so they probably still hold the old code.
  - The two camera views tried were in a wall or in the dark. The "1 error" in the editor's log was there before the change too.
  - To check it there: recompile the materials, and pick a view with a rough surface near colored light.
- The limit: 3 basis directions is a cosine-wide lobe. Medium roughness needs layer 2 (rays ending on the cache, SurfelPlus' GGX
  candidates); mirrors stay on the probes and the sky.

**The review's remaining probe items (not done)**
1. **Specular occlusion.** Keep the scalar GI ratio as a labelled option. Compare no occlusion, the ratio, and a roughness- and
   view-dependent specular AO from a real occlusion input (Filament treats diffuse AO and specular occlusion apart). The
   split-sum's grazing term is Fresnel, not the bug: a washed-out wall more likely comes from the sky fallback, probe fading,
   captures or normalization.
2. **Order-dependent blending.** `w = weight × (1 − covered)` is front-to-back. Make the policy explicit: nested override probes
   versus peers blended within a priority class, a stable order, and the same order from the cell list and the full list. Review
   the `Covered < 0.999` early exit against a bright HDR sky.
3. **Probe fading lets the sky back in** at a room probe's walls. Separate the capture origin, the projection box and the
   influence volume. An enclosing room probe under smaller local ones, so fading falls back to the room, not the outdoor sky.
4. **The diffuse from the roughest specular level** is not Lambertian irradiance. SH9 per probe and for the sky (fits the light
   buffer), or a small cosine-convolved map.
5. **One AO for everything.** Split it into diffuse AO and specular AO, and settle what AO means (material, SSAO) and how much
   visibility the surfel GI already has.
6. **Atlas and conventions:** octahedral seams and borders, isolation between roughness levels and between probe slots, the
   smallest levels, the same roughness mapping in the prefilter, the sampling and the DFG LUT, linear HDR storage. Test with
   neighboring slots of very different HDR colors, the reflection turned through every seam, and the roughness swept.
7. **The capture contract:** HDR before tone mapping, pre-exposure, a defined origin, never sampling the probe being written, a
   policy for other probes during capture, versioned publication, and the six faces captured at different times declared.
8. **Smaller items:**
   - typed accessors for the probe records (not `Color`, `DirCosOuter`, `Shadow`);
   - a probe weighs nothing until its slot holds a published capture of it (generation);
   - test the `g_GI.w > 0.5` support switch (initialization, a level change, missing support), and never smooth it with an
     unsupported read as black.
9. **Tests to add:**
   - a constant white environment;
   - a bright doorway reflected by a surface with low diffuse GI;
   - a dark room probe next to a bright sky;
   - two equal overlapping colored probes;
   - axis-aligned and near-parallel reflections;
   - neighboring atlas slots;
   - GI on/off with unchanged reflection inputs;
   - a probe recaptured while the light changes.
