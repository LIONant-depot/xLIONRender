# Surfel GI: dynamic scenes under a fixed budget (master plan)

Status: **proposal, revision 2, not approved.** Written 2026-10-17 from the engine study (Unreal, O3DE, Godot, Flax,
CryEngine), three review rounds, and what the current code does. Nothing here is built yet.

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
- **A single ray is now the indivisible unit**, and its tail (672 us in Debug) is above the 0.5 ms declared maximum. Splitting
  a ray is not possible; what would cut the tail is cheaper casting and reads.
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

### Phase 6: Prioritised relighting

- A changed light raises the importance of the surfels it reaches (by transport relevance and visible impact); a sun or sky
  change is a sliced relight, never a synchronous one. An explicit "relight everything" exists, rate-limited and still sliced.
- Fairness is the scheduler's (Phase 1), not a term of this score.

### Phase 7: First use is bounded too

- Separate compiler work, runtime loading and runtime GI initialisation. Precompute in the compiler what can be (the rays'
  BVH is the first candidate, once Phase 0 confirms it dominates the 630 ms); slice the rest.
- Until an instance's GI is ready it draws with a declared fallback (its mean light or the scene's ambient). A blocking loading
  screen is a separate, explicit mode.

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

## 5. Where this can beat the competition, and where it cannot

These are design goals to be proven by the tests above, not results. The engine study read one checkout of each engine; its
constants are that checkout's, and it can show what it found, not what does not exist.

**Where it can be better**
- **A documented, tested time contract.** All five engines amortise their GI (per-frame caps, slices, round-robin), and the
  study also found full-refresh paths in some of them for global changes. xLION's claim is only about itself: its maintenance
  contract will be written down, cover every stage, and be checked by an overload test.
- **Small emitters.** Explicit emitter sampling through the links, with emission conserved over a surfel's patch, aims to
  preserve a small emitter's contribution more reliably than hemisphere sampling alone, within the limits of the material
  capture, the footprint it is merged into, the link cap, and occluders near the emitter.
- **Effort follows screen area.** Fine surfels, parents and the ambient cube are chosen by the pixels an instance covers, and
  far surfels are traced less often. Other systems prioritise relevant work in their own ways (probe classification,
  visibility, multiresolution); this is xLION's mechanism, not a claim that they lack one.
- **Explicit transport roles with reference tests.** Receiver-only, proxy and full transport are declared per instance and
  each is tested against its matching reference, so an approximation is always a decision, never a side effect.
- **No ray-tracing hardware required,** within xLION's supported graphics capabilities and memory limits. The solver runs on
  the CPU and the GPU only reads the result, which also keeps it usable in the Debug editor.
- **Mesh-local representation.** Instances share mesh-local geometry and reconstruction data, so a moved instance needs no
  rebuild of its representation; its lighting and visibility remain its own and are updated at its new place.

**Where it cannot (honestly)**
- **Throughput.** CPU ray tracing does far fewer rays per millisecond than hardware ray tracing on the GPU. Under the same
  budget, convergence is slower; the plan trades latency for responsiveness, it does not remove the gap.
- **Diffuse only.** Specular indirect light still comes from the reflection probes and the sky; Lumen and SVOGI also do glossy
  indirect.
- **Detail is bounded by surfel size.** Features smaller than a surfel are averaged. (The GI saturation setting is an artistic
  display adjustment and plays no part in transport accuracy or energy conservation.)
- **An object's own light is attached to it.** A world probe grid answers "what is the light here" at once; a moved instance
  answers it after its surfels are re-traced. Phase 4 and test A measure how long that takes.
- **Not yet proven at scale.** Large open worlds (memory of surfels and links, the 131072-surfel frame region) are untested.

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

## 6. Order and dependencies

```
Phase 0 instrumentation
  -> Phase 1 scheduler (tracing scheduled, no synchronous worker wait)
       -> Phase 2 links          (bounded link maintenance; removes the release stall)
       -> Phase 3 publication    (bounded filtering, upload, publication)
            -> Phase 4 state split
                 -> Phase 5 roles      (tests A, B, C)
                 -> Phase 6 relighting (tests D, E)
  Phase 7 first use: after Phase 0 says where the 630 ms goes
  End-to-end contract: only when Phases 1, 2, 3 and 7 all pass
```

One phase at a time; the plan file is updated at the end of each phase with what was measured.
