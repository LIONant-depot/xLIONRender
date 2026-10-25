# GI at city scale: representation and working set

Status: design note v2, 2026-10-24, for review. v1 was written after the Bistro autopsy; v2 takes in an outside review (relayed by
Tomas) that corrected the accounting and several assumptions. The chunks plan (gi_chunks_plan.md) is the address-space part of this.
Nothing here is built. Numbers come from the audit of the current code and the autopsy runs; *pending* ones from the full Bistro
autopsy still running.

**The claim to test.** Millions of lighting samples are normal (a lightmap has them). What does not scale is a construction that
materializes everything at once, and a runtime that keeps a full simulation record for every sample. The representation should be
proportional to the useful lighting information and to the live working set. Each compact representation below (moments, iterative
light, links, estimator-free transport) is validated on its own; the memory forecast is recomputed from the complete layout before it
sizes anything.

## 1. The failed bake: what the autopsy says

Bistro, one mesh, 2,837,209 triangles, 108 x 32 x 115 m, SurfelSize 1 m (scratch compiler, `XLION_GI_UNCAPPED=<R>,<stop>`, a stage
trail flushed to `gi_autopsy.txt`):

| Stage | Count | Working set |
|---|---|---|
| after the import, before any GI | 2.84M triangles | 3.7 GB; **the import itself peaked at 9.8 GB** (assimp + raw geometry) |
| candidates by the rule (counted, not allocated) | 3,706,189; 1,819,124 of them the floor of one on tiny triangles | +0 |
| R-disks the area alone asks for | 17,364 | |
| candidates materialized | 3,706,189 | +106 MB |
| Poisson count **without visibility rays** (6.3 s) | **161,165** | +0 |
| BVH (Tri4 leaves), 7.7 s | 923,548 | +200 MB |
| bake with visibility rays (1,063 s) | **619,448 fine surfels, 361,727 parents** | +280 MB (4.2 GB) |
| read grid (583 s) | 1,241,390,045 list words (**5 GB**), 524,288 table entries | 9.0 GB; **peak 19.1 GB** |
| exit record | exit 0: the full run completes in 27.6 min | |
| **read grid, lists bounded to 32 (job 1)** | **49,215,458 list words (197 MB)**, 25x fewer | **107 s** (was 583 s); +214 MB; **peak 9.8 GB: the import's, the GI no longer raises it** |

Conclusions, with their qualifications:
- The import's 9.8 GB comes first; the GI then reaches **19.1 GB at the read grid**, not before. The bake itself adds 280 MB.
- The full run completes (exit 0) in 27.6 min: 18 min of bake with rays, 10 min of read grid. The first run was killed by hand in
  the bake: time, not a compiler failure.
- **The accepted count with visibility is 619k, 3.8x the 161k without.** Visibility lets surfels that cannot see each other both stay
  (walls' two faces, layers, occluded neighbours). 619k is the number to plan with at 1 m; area estimates for 0.5 m are void.
- **Parents are only 1.7x fewer (361k)**: the coarse level barely coarsens on this geometry (Police: 2.7x). Far-field option A's
  premise is weak here; that is now measured, not assumed.
- **The read grid is the memory defect**: 2,000 list words per surfel, against tens on Police. Per cell, per part (a connected
  piece of one plane), the list holds every facing fine surfel within 2 R that any corner sees: parts x surfels-in-reach, quadratic in
  local density, and the Bistro's layered geometry has many parts per cell. The 524,288 table entries are the cells (a power-of-two
  hash), not a cap. A bounded list per part (the K nearest seen, as the pixel read already caps at 32) removes it; this is
  construction, not representation. See gi_handover_2026-10-24.md, job 1.
- 161k is 9x what the area alone asks for: the acceptance keeps one surfel per facing per R (detail, layers, orientation).
- **A construction defect:** the candidate rule's floor of one per triangle is tessellation, not geometry (1.8M of 3.7M candidates on
  tiny triangles). Whether those candidates are useless is to be shown, not assumed: some tiny triangles are real small features. The
  replacement is area-proportional sampling plus explicit preservation rules for thin or disconnected features and for emitters, with a
  coverage test independent of triangle count. Test set: the same geometry (a) as authored, (b) heavily subdivided coplanar, (c) with a
  small disconnected strip, (d) with a tiny emissive feature, (e) with thin parallel layers. The new rule must cut the redundant work
  without deleting the cases the old rule protected.

## 2. The memory ledger (per fine surfel unless noted)

**Measured on Police (job 3, 2026-10-24; 63,125 fine surfels, 23,006 parents, 91,778 triangles; autopsy stage 7, capacities):**

| Category | MB | B a fine surfel |
|---|---|---|
| Shared per mesh: triangles (the CPU twin: positions, normals, UVs, materials) | 10.55 | 175 |
| Shared per mesh: BVH + Tri4 | 6.52 | 108 |
| Shared per mesh: surfels + parents (+ materials) | 6.67 | 111 |
| Shared per mesh: vertex surfels | 4.20 | 70 |
| Shared per mesh: neighbour lists (2 levels) | 2.12 | 35 |
| Shared per mesh: ray origins (position + normal each) | 6.57 | 109 |
| Shared per mesh: read grid (lists bounded, job 1) | 8.46 | 141 |
| **GI shared, without triangles and BVH** | **28.03** | **466** |
| Per instance, CPU light state, CPU-solved (+ the filter's 302 B) | 36.12 | 600 |
| Per instance, CPU light state, GPU-owned (job 2: the solver's arrays freed; no filter) | 11.80 | **196** |
| Per instance, GPU state (layout, the residency audit: E, validity, moments, basis, estimator, counts, EmitB) | | ~376 (+72 filter scratch) |

**Compression census (Police, 2026-10-24, autopsy stage 10):** 275k vertices (4.4 a fine surfel); vertex bindings: 21% of the
vertices carry a unique ordered tuple; origins: 68% of the surfels have only their center; read grid: payload 5.2 MB, directory 2.7 MB
(table 49% full, 16 B a part header), 51% of the corner-weight words all-seen (0xFFFF), the top 256 patterns cover 97%, identical lists
3% (not worth interning); neighbours: degree 2.4 on average, already flat; the CPU surfel record is 48 B (padded vectors) and arrays keep
8-31% unused capacity. Done since: ray origins as vertex indices with the center implicit (24 B -> 4 B an origin, exact).

**The GPU keeps two copies** of a mesh's read grid, surfels and vertex bindings: the display buffer (xlionrender_instance.cpp
PackMeshGI, read by the material shaders) and the trace scene (world::PackGPUSurfels, read by the GI passes). About 45% of the GPU's
shared GI memory; one copy is exact. Blocked on the other thread's uncommitted xlionrender_instance.cpp.

So the cold shared data is the larger lever after job 2: 466 B a fine surfel, not the ~60-100 B v2 assumed. The ray origins (109 B) and
the vertex surfels (70 B, per vertex) are the surprises; the read grid is 141 B after the bound.

Measured today (audit, gi_chunks_plan.md): GPU active state 376 B per instance at the fine count whatever the level; CPU 610 B + 36 B
per filter pass, at the fine count once an instance has been fine, **also in GPU-owned mode**; published light 16 B + moments 64 B per
instance; links (experiment D) 16 + 16 entries x 16 B = 512 B host-visible; shared mesh data ~60-100 B plus variable structures.

Proposed, as an honest total, not a best case:

| Lifetime | Item | Bytes | Notes |
|---|---|---|---|
| A cold, per mesh | compact geometry record | 24 | position quantized to the chunk (3 x 16), octahedral normal (2 x 16), radius class (8), parent (16), flags |
| A cold, per mesh | **variable structures**: read-grid entries, neighbour lists, origins, vertex surfels, chunk and ghost tables, BVH | *measured separately*, per asset | not per-surfel constants; reported as totals per mesh (the Police / Golem read lists are the first numbers to take) |
| B published, per instance | light | 16 | 3 x RGB9E5 + validity, as the ring today |
| B published, per instance | depth moments | 32-64 | **scene-dependent, not cold mesh data** (section 4): per instance where the reads need them; 32 B only if 8-bit quantization passes its own test; 0 where a surface domain no longer needs them |
| per instance | links (transport-valid static surfels) | ~100 | 16 retained x (16-bit local index + 8-bit weight) = 48 B, the retained row mass 4 B, the tail per target instance 16 x (16-bit instance + 8-bit weight) = 48 B; cross-chunk targets through the ghost tables |
| per instance | solve state | 32 | the iteration's old and new values (float, not the published encoding): 2 x 16 B; residual reductions per chunk |
| per ghost | copies of B | 16-80 | the ghost ratio measured per asset (the room test: 160 % of the owned; big flat walls less) |
| C active, per actively refined sample | estimator, accumulators, change detection | ~400 | only the bounded active set: moving objects, chunks whose links are not built or not valid |
| workspace | construction, link build, publication copies | transient, bounded by stage | reported by stage |

Totals per fine surfel of transport-valid static geometry: **24 + 16 + (32-64) + ~100 + 32 = ~200-240 B**, plus the variable
structures and ghosts, against today's ~60-100 + 16 + 64 + 376 + 610 + 512 = ~1,650 B. At 161k surfels that is ~35-40 MB (plus
variables) against ~270 MB; at 6.5M (a city at 0.5 m) ~1.4 GB against ~11 GB. Still large at city scale: residency (section 6), not
compaction alone, bounds it. v1's "40 B" counted only the geometry record and the light; this table replaces it.

## 3. The first win: the CPU mirror

In GPU-owned mode the CPU world still allocates its full solver arrays (estimators, accumulators, moments, filter passes: ~610 B a
surfel at the fine count). Check what the host actually needs (placement, scheduling, the pick, the means) and allocate the rest on
demand: tests, the CPU fallback, an explicit read back. This is an allocation-lifetime fix, earlier and simpler than any compression.

## 4. Depth moments are scene data

A surfel's moments come from its rays into the assembled scene: a chair's self-visibility is mesh-local and could be shared; its
visibility toward the wall, the other chair or a moving door is not. Two instances of one mesh in two rooms have different moments.
So: moments stay per instance (B), refreshed by the solver as today for the active set; for transport-valid static chunks they are
refreshed with the links' build (the same rays) and then fixed until the chunk's transport is invalidated. Where a surface domain's
reads do not need the moments' test (to be measured: the reads' second tier already covers the wall's foot), drop them there rather
than compress them everywhere. Quantization is tested on the visibility output (the rejection near a thin surface, negative variance
guarded), not on the depth error alone, and per radius class.

## 5. Compiled transport: what it removes and what it does not

The operator `E = D + T E` over published light needs no hemisphere estimator: with a fixed operator and a deterministic input, repeated
sparse products are not Monte Carlo. This removes C for every surfel whose transport is valid. Three situations are kept apart:

| Situation | What remains |
|---|---|
| valid operator, deterministic D | the iteration's state (section 2) and a convergence policy |
| valid operator, D from stochastic sources | a policy for the source noise, or D made deterministic |
| the operator changed (a blocker moved, geometry changed) | invalidation, maintenance of the affected rows, or a residual correction |

**D today** (CompileTransport / the link shader): the sky, the direct light at the compile's fixed 256 rays' hits (shadow-tested), the
emission at the hits, and the emitters' light as the sampling's **expectation** (every emitter in sight, by weight: deterministic). So D
is a fixed-sample estimate: deterministic, with a bias that shrinks with the ray count, and no noise over time. D depends on the lights:
a sun or an emitter that changes recomputes D (the links' rays again on the GPU, 39 ms on the Ducati; or the hits' direct terms
stored and re-shaded, to be weighed), not T. The condition is therefore not "static geometry" but **transport-valid, with current
source input**: T valid (no blocker moved through its rays) and D current (the lights it was computed under). A moving external
blocker invalidates the rows its rays crossed, static receivers or not.

## 6. Links: layout and precision

Retained links, the tail and their encoding are one representation to validate, not three constants:
- Retained: 16 per surfel today, by heaviest luminance; the tail per target instance (the instances' means), 16 entries.
- A scalar geometric weight per link is possible if the material response stays separate: today's weight already carries the hit's
  albedo (colored). The option: store the geometric weight and multiply by the target surfel's material sample at solve time, with the
  approximation (the hit's albedo vs the target surfel's) measured.
- Precision: keep the retained row mass exactly and quantize the normalized retained weights; represent the omitted mass through the
  declared tail policy; never renormalize the retained links to absorb the tail without measuring the redistribution. Near unit gain,
  coefficient error compounds over bounces.
- Test the iteration with the compressed light and the compressed weights together: quantized feedback can stall above the residual or
  cycle where the exact operator converges. The display encoding passing is not the feedback encoding passing.
- Disk or GPU rebuild: by cost, with explicit dependency and version keys. Intra-asset reusable data may well be worth storing (editor
  start, mobile, streaming, reproducibility); scene-dependent links belong in a scene or chunk cache. Not an ideological choice.

## 7. Residency closes the transport's dependencies

Three sets, three budgets:

```
Potential samples        defined by the assets (A), loaded or streamed per chunk
Resident samples         B (+ links) under a memory budget
Actively refined samples C under a compute budget
```

Resident is not "near enough to be read this frame": a near receiver's links reach offscreen and distant sources, a bright emitter
outside the near range still lights it, a distant facade still blocks the sun. Every link target resolves to a resident target, a
valid coarse representation of it (section 8), or an explicitly handled residual; eviction never turns a coefficient into zero. The
light ring is sized to the resident set. Loading more geometry allocates more A, not more B or C.

## 8. Bounded construction

Stages kept distinct, each with its own parallelism:

```
candidate generation      per cluster block, area-proportional + the feature rules (section 1); bounded
cheap geometric rejection parallel
visibility testing        parallel against the one immutable BVH; or the validated GPU traversal kernel, offline
conflict / acceptance     Poisson acceptance is not embarrassingly parallel: priorities or a sequential pass over tested candidates;
                          the result checked across worker counts
derived structures        read grid, neighbours, origins per chunk (whole-geometry visibility through m_pOccluder, built)
```

Candidate storage is 106 MB on the Bistro: out-of-core sampling is not the problem; the pending visibility stage is where to profile
before engineering. A memory ceiling and the stage trail stay in the compiler by default.

## 9. The far field: three options, compared before choosing

| Option | What is far | Visibility | Pros | Cons | What to measure |
|---|---|---|---|---|---|
| **A. coarse surface cache** | the parents (3 R) and a third level (~9 R) | the geometry BVH, unchanged | one representation; the most conservative | coarse surfels lose orientation detail; Police's parents were only 2.7x fewer (63k -> 23k): the reduction per level is uncertain; **links across levels need a defined restriction (fine sources -> one coarse source, deduplicated, with its represented area and orientation) and prolongation (coarse -> fine on the way back), transition publication and mixed-level visibility - not "unchanged"** | the third level's count and quality on Bistro; the reduction per level; the level-border error |
| **B. surface cache + a coarse light volume** | a world-space clipmap filled from the surfels | the geometry BVH, unchanged | fixed memory for the far light | a second lighting representation: irradiance in, radiance out at a ray's end (`L_o = L_e + rho/pi E`); ownership of a bounce in the overlap | the boundary contract and energy conservation on a test scene |
| **C. surface cache + voxel light and opacity** | radiance + opacity clipmaps | voxels | far light and visibility bounded together | the most new infrastructure (voxelization, thin walls, directional radiance, materials, moving blockers); 5 x 128^3 RGBA16F is 80 MB per field before directions | only if far visibility itself becomes the measured cost |

**Measured (2026-10-24, autopsy stage 8: the bake's own dart rule with visibility, each level from the one below, radius x3):**

| Asset | fine | parents (3 R) | level 2 (9 R) | level 3 (27 R) |
|---|---|---|---|---|
| Police (R 0.5 m) | 63,125 | 23,006 (x2.7) | 12,110 (x1.9) | 6,746 (x1.8) |
| Bistro (R 1 m) | 619,448 | 361,727 (x1.7) | 213,317 (x1.7) | 66,997 (x3.2) |

Coarse surfels shrink far less than the radius suggests (x9 a level by area): a surfel stands for one plane in one orientation, and a
detailed or layered scene keeps one per facing piece whatever the radius. The Bistro at 27 m still needs 67k surfels (x9 fewer than the
fine level over three levels, not x729). Option A alone does not bound the far field; it buys about one order of magnitude. Option B (a
coarse light volume for the far field, the surfels kept near) is the next experiment.

Order: A, then B if A's levels do not shrink enough, C only if visibility is the problem. A is not reopened as C because A has not yet
been compressed.

## 10. Published light encoding

RGB9E5 (one shared exponent, unsigned) is the storage target for the published value. Validated separately for (1) the final published
value and (2) every iteration's feedback value; passing the first does not establish the second (the iteration keeps float state,
section 2). Cases: bright red with a weak blue, near-black indirect light, long multibounce chains, a light switched off, exposure-scaled
scenes, different iteration counts. Signed quantities (residuals) never go through it.

## 11. Order of work

1. Finish the autopsy; keep the accepted counts, the parents, the entry counts, the time by stage, the peak and the retained output.
2. The CPU mirror: no full solver arrays in GPU-owned mode (allocated on demand for tests, fallback, read back).
3. The complete memory ledger measured on Police: moments, links, ghosts, variable structures, workspace, publication copies.
4. The candidate rule (section 1, with its test set) and the profile of the visibility stage.
5. The published encoding validated apart from the compressed feedback.
6. The compiled operator validated under current source input and moving blockers (invalidation of rows).
7. Bounded per-chunk residency with explicit link-target resolution.
8. Far field: coarse level A measured on the Bistro; B if needed.

Each step is a saving on its own; none requires another unbounded bake.
