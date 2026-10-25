# GI handover, 2026-10-24: for the next session (Opus)

Written by the Fable session that did experiment D, the chunks plan and the Bistro autopsy. Everything a fresh session needs to
carry on: the state of the repos, the rules, the tools, what was measured, and the next jobs with their acceptance tests.

## 1. Read these first

1. `documentation/gi_working_set_design.md` (v2): the representation and working-set plan, the memory ledger, the order of work.
2. `documentation/gi_chunks_plan.md` (v3): the address-space plan (no 65k cap), owners and ghosts, the residency audit.
3. Memory files in `C:\Users\user\.claude\projects\D--LIONant-xLION\memory\` (MEMORY.md indexes them): the standing rules.

## 2. Standing rules (Tomas's, do not relax)

- **Commit only when asked. Never push.** Stage explicit paths, never `git add -A` at a root. All work on `main` in every repo.
- **Never touch the other thread's uncommitted files**: `src/xlionrender_instance.cpp`, `src/xlionrender_instance.h`,
  `src/tests/xlionrender_test_static_geom.cpp` (they show as modified in xLIONRender; not ours).
- **Never drive an editor you did not start**; never the real mouse (virtual input + CaptureWindow if a screenshot is needed).
- **No smoke tests unless asked; never the full suite.** The C++ unit tests of the change are fine.
- **Never build Game.vcxproj by hand** (breaks the editor's PCH). Build with the solution's MSBuild for the editor, the scratch
  build for the compiler (section 4).
- **Measure maximized, frame budget off** (small windows bias GI timings). Fill the screen with pixels: Police, not the mesh
  preview alone.
- **Parallelism through xscheduler**, never a private pool or std::thread (per-module `g_System` gotcha in the memory file).
- **Ask before a direction change**; short status posts in the thread; one job at a time, each leaving a protecting test
  (never delete tests).
- Tomas wants a big leap on realistic scenes (Bistro), not the toy Test Models. He reviews design notes before building.

## 3. State of the repos (nothing pushed)

xLIONRender `main`, committed today (b4802f1 .. 927c40d): relevel, adaptive read grid + cap 32, XLION_GPU_TIMES, coalesced Show
writes, ring MemoryMapFrameRegion, GI lag readback, CPU transport prototype, XLION_GI_TRANSPORT, GPU link build, per-frame solve +
pump (XLION_GI_LINKS), settle 30. xGPU: 3f2f4d2, 611432a (frames in flight). xLION: 1680bdf (XLION_FRAME_TIMES).

**Uncommitted (ours), awaiting Tomas's commit request:**

| File | What |
|---|---|
| `src/features/gi/xlionrender_surfel_gi_chunks.h` | `PartitionChunks`, `chunked_mesh`, `chunk_report`, `chunk_local_cap_v`: the weighted-median partition by clusters, owners and ghosts |
| `src/features/gi/xlionrender_surfel_gi.h` | `m_pOccluder` (bake visibility through the whole geometry), instance `m_nOwned` / `m_GhostOwner`, `SetGhosts`, `OwnedSurfels`, `SyncGhosts` (after the commit when !GPUOwned), picks and emitters skip ghosts |
| `src/tests/xlionrender_test_surfel_gi.cpp` | test 6l: the room split into 6 chunks; chunked vs whole light at common points: 95th pct 0.66 % vs a 0.65 % noise floor |
| `plugins/xgeom_static.plugin/source/Compiler/xgeom_static_compiler.cpp` (xLION repo) | `#include <chrono>`, a WIN32 block (NOMINMAX, psapi), the `XLION_GI_UNCAPPED=<R>,<stop>` autopsy in PrecheckGI writing `gi_autopsy.txt` by stage |
| `documentation/gi_chunks_plan.md`, `gi_working_set_design.md`, this file | plans |

Two surfel_gi timing tests failed while the Bistro bake loaded the CPU; re-run the surfel_gi suite once (expected to pass idle).

## 4. Tools

- **Scratch compiler build** (so the editor's build is untouched): `C:\Users\user\AppData\Local\Temp\claude\D--LIONant-xLION\70688f38-924d-4208-878f-f793fd8a16a8\scratchpad\gcomp\xgeom_static_compiler.exe`,
  built from `Build/xLION.vs2022` with OutDir/IntDir overridden to that folder, assimp DLLs copied beside it. Its `gi_autopsy.txt`
  is the Bistro trail. Rebuild with MSBuild on the compiler's vcxproj with `/p:OutDir=...;IntDir=...`.
- **Autopsy**: `XLION_GI_UNCAPPED=1.0,5` (R = 1 m, stop stage 5 = read grid; 4 = bake; 3 = BVH; 2 = Poisson count; 1 = candidates)
  on the Bistro's compile (resource `4854F8D7DA54C001`, the editor's usual compiler command line, see `scratchpad/gout`).
- **Editor measurements**: `scratchpad/gi_models.py <Debug|Release> [hold_s] [models]` (maximizes, GI frame log medians per
  Test Model); `police_motion.py`, `flicker*.py`, `converge_time.py`, `maximize.py`. The live editor tools: `live.py`, `xcmd.py`,
  `impact.py` (memory file "use the live editor tools").
- Env switches: `XLION_GI_LINKS` (links on), `XLION_GI_TRANSPORT`, `XLION_GPU_TIMES`, `XLION_FRAME_TIMES`, `XLION_GI_REGRID`
  (rebuild the read grid at load, for measurement), `XLION_GI_ABLATE`, `XLION_GI_FREEZE_S`, `XGPU_SERIAL_FRAMES`.

## 5. What was measured (the facts to build on)

Bistro, 2.84M triangles, 108 x 32 x 115 m, SurfelSize 1 m, full autopsy, exit 0, 27.6 min:

| Stage | Count | Time | Working set |
|---|---|---|---|
| import (assimp + raw) | | | 3.7 GB, **peak 9.8 GB before any GI** |
| candidates by the rule | 3,706,189 (1,819,124 from the floor of 1 on tiny triangles; the area asks 17,364 R-disks) | | +106 MB |
| Poisson count without rays | 161,165 | 6.3 s | |
| BVH (Tri4) | 923,548 leaves | 1 s | +200 MB |
| bake with rays | **619,448 fine, 361,727 parents** | **1,063 s** | +280 MB (4.2 GB) |
| read grid | **1,241,390,045 list words (5 GB)**, 524,288 table entries | **583 s** | 9.0 GB, **peak 19.1 GB** |

Police for scale: 63k fine, 23k parents, read lists in the tens of words per surfel. Ducati GPU link build 39 ms, Sponza 192 ms.
Links solve: <= 5 iterations to 0.1-0.26 % of converged; top-16 + instance tail 15 MB on Police at 0.48 %.

**Read grid, the mechanism** (`BuildReadGridAt`, `xlionrender_surfel_gi.h` ~832-1075). The table is a power-of-two open hash of
the cells the surface crosses (524,288 entries = cells, **not a cap**; a thread post of mine said "the table stops at 2^19": wrong,
ignore it). Per cell, per *part* (a connected piece of one plane in the cell), the list holds every fine surfel within `Reach = 2R`
of the cell that faces the part (normal dot > 0.5) and that any corner's points see (4 x 4-bit corner fractions + 16-bit surfel per
word). The cell starts at R and is halved up to 4 times while patches per cell > 8. On the Bistro ~2,400 words per cell: many parts
per cell (layered, doubled, detailed geometry: the same reason the bake kept 3.8x the no-ray count) x many facing surfels inside a
2 m reach at 1 m spacing. The size is parts x surfels-in-reach, quadratic in local density. It is also a 583 s stage (the `Sees`
rays per corner point per surfel per part).

## 6. The next jobs, in order, with acceptance

Each one is a measurable saving on its own and leaves a test. Report numbers, not adjectives.

### Job 1. Bound the read grid (the 19 GB peak, the 583 s)

Options to measure (pick by numbers, on Police + the room test first, then the Bistro trail):
- (a) Per part, keep only the K nearest facing surfels with a nonzero corner fraction (K = 32 as the pixel read's cap already is:
  `chunk_local_cap_v`/the read cap 32 in the renderer); the list is then bounded per part. Check: the read's result at common points
  (test 6l's method) unchanged within noise on the room, Police's light unchanged.
- (b) Merge parts that are the same plane across sub-cells more aggressively, or cap parts per cell, with a measured light error.
- (c) Build the lists per chunk (the chunk plan's "derived structures per chunk") so the working set is one chunk's, and stream the
  words out: bounds the peak even if the total stays.
- Also time: the `Sees` work per (corner point, surfel, part) is the 583 s; the GPU traversal kernel exists for offline visibility
  (plan, section 8) but is a bigger job; start with (a), which also cuts the rays.
Acceptance: Bistro trail stage "read grid" under 1 GB of list words and under 120 s; Police unchanged in light (6l-style compare)
and in frame time (gi_models.py Release, maximized); a unit test that a dense layered slab's list words stay under the bound.

### Job 2. The CPU mirror in GPU-owned mode (about 610 B a surfel)

`world` in `xlionrender_surfel_gi.h` allocates its full solver arrays per instance (estimators, accumulators, moments, filter passes)
even when `m_Setup.m_bGPUOwned`. Find what the host still reads in GPU-owned mode (placement, scheduling, the pick, the means,
`InstanceLight` for tests) and allocate the rest on demand (tests, the CPU fallback, explicit read back). Measure on Police: the
world's bytes before and after (add a `Bytes()` accounting if none). Acceptance: the surfel_gi suite passes (it uses the CPU world),
the GPU path's light unchanged, the host allocation per fine surfel reported before/after.

### Job 3. The complete memory ledger on Police

Measure, not estimate, per category of the design note's section 2: shared mesh (record, read grid, neighbours, origins, vertex
surfels, BVH), per instance (light, moments, GPU state, CPU state, links host-visible + device), ghosts (the room test), workspace by
stage. Put the table in gi_working_set_design.md section 2 replacing the "proposed" column's assumptions where they differ.

### Job 4. The candidate rule + the visibility stage's profile

The floor of one candidate per triangle (`surfel_mesh::Candidates`): area-proportional candidates plus explicit preservation rules
(thin or disconnected features, emitters), with the five-case test set in the design note's section 1. Then profile the bake's
1,063 s by stage (candidates, Poisson acceptance, Sees rays, parents) with the autopsy's `Stage` lambda. Do not change the
acceptance's visibility semantics without a before/after at common points.

### Later (design decisions: ask Tomas, and consider a Fable session for them)

Published-light encoding validation (RGB9E5, display vs feedback), the links' precision layout (section 6 of the design note),
residency with link-target resolution, far-field option A on the Bistro (parents only 1.7x fewer there: measure the third level
before trusting A).

## 7. Pitfalls met today

- Shader/CPP patches: write them with the Write tool (heredocs eat backslashes); the sources are CRLF; tolerant patch scripts in the
  scratchpad show the pattern (`patch_*.py`).
- `(void)` casts are invalid GLSL. `std::size_t(x)` in a declaration is a vexing parse. windows.h needs NOMINMAX (xcolor breaks).
- New shaders need `cmake .` re-run in `Build/xLION.vs2022` before the editor's vcxproj sees them.
- GPU timestamp reads with frames in flight: only read frames <= `m_LastDoneFrame` (xGPU).
- Tomas kills long compiles; a trail file flushed per stage (as `gi_autopsy.txt`) survives the kill. The Bistro bake is 18 min:
  run it in the background with a time limit and say so in the thread.
- The Remote Control link drops now and then; after a reconnect, re-fetch the thread before replying (your own last reply may
  already answer).
