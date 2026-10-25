# GI chunks: big geometry without the 65k surfel cap

Status: plan v3, 2026-10-24. v1 agreed in principle with Tomas ("sounds like a winning strategy"); v2 and v3 take in two outside
reviews (relayed by Tomas the same day). Nothing built yet beyond a measurement (the uncapped Bistro bake).

**Two problems, kept apart.** Chunking is the *address-space* solution (no 65k cap). It does not make the data affordable to keep
active: the *scaling* solution is a bounded resident lighting field (below). The Bistro decides whether the design scales, not merely
whether the compiler can finish; its numbers are read as implementation overhead, an acceptable offline cost, or evidence that the
representation must change.

**The invariant.** A chunk owns a subset of the lighting's degrees of freedom: a page of locally indexed lighting data. It does not own
an isolated part of the physical lighting problem. Chunking changes storage and scheduling boundaries; it must not create new physical
boundaries in the geometry or in the lighting solution.

## The problem

A geometry's surfels are addressed with 16 bits everywhere: the read lists (16-bit surfel + 4x4-bit corner weights in one word), the
per-vertex surfels (4 x uint16), the GPU records, and the transport links (`instance << 16 | surfel`). A geometry over 65,534 fine
surfels gets no GI: the compiler's precheck turns it off. The Bistro (one mesh, 2.8M triangles, 108 x 32 x 115 m) needs more than
65,534 surfels even at SurfelSize 3.1 m; the compiler suggests 21 m, which is useless.

## Three identities, kept apart

| Identity | What it is | Who owns what |
|---|---|---|
| Logical instance | What the user moves, edits, deletes, assigns materials and roles to | owns its geometry and all its chunks |
| Geometry / ray identity | The triangles and their placement | one acceleration structure per geometry (below) |
| GI chunk | A page of locally indexed lighting data | belongs to one logical instance |

The first implementation may reuse the existing GI instance record for a chunk, but through explicit mapping helpers (logical instance
-> its chunks; chunk -> its logical instance). Everything that belongs to the object stays on the object: a move is one transform
update for all its chunks; a material edit, a role, a capture are per object, never repeated per chunk; scheduling budget and fairness
are per object (a heavily chunked asset gets no more rays for having more pages); deleting an object retires all its chunks and their
pending work together; "self" tests (a surfel and its own instance) mean the logical instance.

## Compiler (xgeom_static)

1. **A canonical bake, then the partition.** The surfels are baked once over the whole geometry (the bake already works in wide
   indices internally): Poisson acceptance, parents, neighbours, origins, read grid, vertex surfels, all as if there were no cap. The
   partition comes after: each surfel is assigned an owner chunk; references are then remapped to 16-bit chunk-local indices for
   serialization. Splitting first and baking each chunk separately would change the approximation at the borders (samples across a
   border could not reject each other, parents and neighbours would differ, represented area would change), whatever the visibility
   rays see. One whole-geometry BVH serves the whole bake.
2. **Partition.** Recursive median split of the clusters by their bounding-box centres along the longest axis, **weighted by the
   surfels each cluster owns** (the canonical bake's, not the cluster count), until each chunk fits its target. Clusters are not cut.
   Recorded per compile: owned surfels per chunk, chunk extent, bounds overlap, boundary adjacency, ghost-to-owned ratio. A
   surface-adjacency refinement only if these show bad splits.
3. **Ghosts.** A chunk's read lists, vertex surfels, neighbour lists and filter supports may reference surfels another chunk owns
   (within reach of its border). Those get local **ghost** slots: entries in the chunk's local index space that name their owner
   (chunk, local index). See "Owners and ghosts" below.
4. **A hard cap, checked after the bake.** `owned + repair (defect patches) + ghosts <= 65,534` local indices per chunk, verified on the
   real result, after the ghosts are made. A chunk that fails is split again and its ghosts rebuilt. The target (48k owned) is a
   starting point, not a proof.
5. **One cluster that cannot fit** (a few huge triangles at a small R can need many surfels): a precise diagnostic naming the cluster
   and its area, and a deterministic repair (its triangles subdivided into more clusters) rather than GI off for the whole asset.
6. **Identity mapping.** One mapping, used by the CPU and GPU intersections, the material capture, the emission resolve, the debug
   views and capture updates: original triangle <-> cluster primitive <-> chunk <-> material-capture entry <-> the surfel's sample
   source. The material capture runs once per logical geometry, not per chunk.
7. **Format (xserializer version 7).** Independent blocks per chunk: bounds, counts, byte ranges, owner/ghost tables, cross-chunk
   references, the source/recipe version; per cluster its chunk (uint16, checked). Version 6 resources load as one chunk, unchanged.

## Renderer

1. **Rays: one acceleration structure per geometry.** A hit gives the logical instance and the original triangle; triangle -> cluster
   -> chunk -> local lighting address. No geometry is duplicated or split for the GI; the TLAS does not grow. (A split geometry BVH is a
   later, measured option: reported if tried - TLAS entries, BVH memory, traversal per ray, transform updates, repacks.)
2. **Draws.** Clusters are drawn one at a time already; each cluster's draw carries its chunk's GI reference. The per-pixel and vertex
   reads read their chunk's grid; a ghost entry reads its owner's record.
3. **Transport links (experiment D).** `instance << 16 | surfel` becomes central encode/decode functions with range assertions; the
   upper field's width, its reserved values, slot reuse after a delete (a generation), stability across repacks and separate renderer
   namespaces are audited before chunks multiply the instances.

## Owners and ghosts

Every surfel has exactly one owner chunk. Ghosts are **copies** in the chunk's local index space (decided after reading the readers: a
reference would change every reader, the material shaders included; a copy keeps their format). That convenience is not the reason
to accept them: the epoch contract and the level rule below are what make them correct.

- **Owned surfels** are traced and updated, own their estimator and history, contribute emitting area, count in the statistics and the
  scheduling.
- **Ghosts** are never traced, never emit, never get their own history, never count as physical coverage or as traced work. Their
  copying, filtering, memory and synchronization do count in the costs reported.
- **The epoch contract.** Every consumer reads ghost data from the same input epoch as the owned data it reads beside it. Per consumer:
  the solver's trace reads the published epoch (ghosts copied from it before the trace); the commit publishes the owners' new epoch and
  the ghosts are refreshed from it before the next consumer; the filter's passes need their own exchange: either the boundary values
  exchanged between passes, or a halo wide enough for all passes with the intermediate values computed locally (copying an owner's final
  filtered value is not its pass-1 value). All of it as GPU ordering inside the frame, never a new CPU wait. The CPU world follows the
  same epochs.
- **Mixed levels: gated.** Copying a parent's centre and moments into a record built for a fine surfel changes what the record means
  (its read lists, weights, corner visibility and normal tests were built for the fine one). Not done. The first tests keep adjacent
  chunks at the same level. Two policies to evaluate later, each with its own test: (a) keep the fine ghost's geometry and support and
  use a declared coarse approximation for its light only (never parent-centred moments read as fine-centred visibility); (b) real
  parent ghosts, with support and visibility compiled for the parent, and a level-aware selection.
- **Halo width by consumer.** The ghosts cover the widest support of all their consumers: the fine pixel read (2R), the parent read
  (2 R_parent), the filter's neighbour lists over all its passes (each pass widens the support), the solver's hit reads, the ray
  origins, the bounded fallback. Listed per consumer in the code; a halo that only repairs the pixel read would leave the next bounce
  wrong.
- **Query work stays bounded.** Ghost entries count against the same caps as owned ones (the per-pixel cap, the fallback cap, the read
  grid's patch refinement): chunking adds no new path to long lists.

## Bounded residency: levels must reduce allocation, not just updates

Update-rate LOD (every surfel's fine state allocated, distant ones updated less often) saves time, not memory. The Bistro needs
**resident working-set LOD**: shared asset data kept as needed, fine runtime light allocated only for selected regions, a coarser
representation for the rest, promotion and demotion through bounded publication. Today's fine / parent / cube switch is not assumed to
do this: which buffers stay allocated while a chunk is coarse is measured first. Coarse is not automatically cheap (Police: 63k fine
surfels, 23k parents, not a ninefold reduction): fine, parent and coarsest counts, bytes per resident level and their quality are
measured on the Bistro, not extrapolated from the radius.

Far is not irrelevant: a distant or offscreen chunk still blocks sky and sun, lights a visible interior, holds emitters, carries
bounces. The policy is fine detail near important queries and coarse but usable transport elsewhere, never "only nearby chunks
exist". Geometry visibility and light residency stay separate: evicting a chunk's fine light never removes its triangles from the rays.

Global limits are audited too: removing the per-mesh cap does not remove application-wide ones (the renderer's light-ring region of
131,072 surfels a frame, the instance record table, the GPU buffers' sizes and offsets).

## Memory, by category

Reported this way, so an experiment's cost does not drive the architecture and no working copy is missed:

- **Shared asset data**: surfel geometry, read structures, neighbour and origin data, the BVH, ownership and ghost tables.
- **Per logical instance**: active light, estimators, moments, displayed state, pending and published versions, ghost replicas.
- **Optional experiments**: compiled transport links (experiment D), guiding data.

## Per-chunk levels: after the storage is right

Each chunk picking its own level (the far end of the Bistro coarse, the street fine) is the far-versus-near the single mesh never had.
It comes after storage and correctness, and only with: no repack per chunk transition (the in-place relevel), hysteresis, stable
addresses across a switch, the mixed-level ghost rule above, no cascade of sibling updates.

## Streaming: the format allows it, the policy waits

A chunk's serialized identity is kept separate from the runtime slot holding its data. Geometry residency, GI metadata residency,
light residency and ghost dependencies are different things: dropping a chunk's light is not removing its geometry from the rays
(offscreen geometry still blocks and bounces light). No eviction policy in this work.

## Tests

| Test | What it catches |
|---|---|
| A constant light field across an artificial chunk border | normalization seams in the reads |
| Localized light crossing a border | lost spatial detail |
| An emitter split across chunks | duplicated area or power |
| A door in chunk A between surfaces in B and C | global visibility and dependencies |
| Adjacent chunks at different levels | ghost and parent compatibility |
| A chunk's object deleted with work pending | handle lifetime |
| Two instances of one chunked asset | runtime light not shared by accident |
| The camera crossing many level boundaries | repack and scheduling spikes |
| A forced low chunk cap | the split, retry and overflow paths, cheaply |
| The same geometry partitioned two ways | partition-dependent light |
| Whole against forced-chunked, at common surface points (not surfel to surfel) | the display and the light carried to another surface |
| CPU = GPU through chunks | (necessary, not sufficient: both can share a border error) |
| Version 6 resources | load as one chunk, unchanged |

## Measures

The uncapped bake by stage: candidate generation, Poisson selection, visibility tests, parents, neighbours and origins, read grid,
partition, ghosts; peak memory; serialized size; what each stage scales with. And a check that the uncapped path uses wide indices in
all its temporary data (nothing truncated to 16 bits before the partition).

Surfels, chunks, ghosts, memory (by the categories above), compile time, frame cost near and far, time to settle, border
steps; the compiler run at 16k, 32k and 48k targets (chunk extent, ghost overhead, instance count, compile time, traversal, level
behaviour) before fixing one. Index capacity and level granularity are separate goals; one number does not settle both.

## What is authorized now (v3)

1. The isolated Bistro bake, with stage and memory data.
2. The canonical bake's ownership and the partition, with their tests.
3. Ghost copies only with the epoch contract above.
4. Equal-level borders first; mixed levels gated.
5. The runtime memory envelope established before loading all fine state. Not authorized: "fine everywhere, residency later" as the
   Bistro's runtime.

## v1 scope (narrow)

1. The versioned container and the owner/chunk mapping; one-chunk geometry unchanged.
2. The canonical bake, the deterministic weighted partition, ghosts as owner references, the hard post-bake check.
3. The renderer: chunk per cluster draw, ghost reads, one BVH per geometry, the identity mapping.
4. Forced chunking on small known scenes: the tests above.
5. The Bistro with GI.
6. Per-chunk levels only after 1-5 hold.

Not in v1: streaming, partition optimization beyond the weighted median, split geometry BVHs, new runtime cache structures.

## Answers so far

- **48k?** As the initial owned target under the hard check; 16k / 32k / 48k measured before choosing.
- **Chunks as the streaming unit?** Yes as a format capability; not yet a runtime commitment.

## Residency audit (2026-10-24, read-only, before building)

What a coarse instance keeps today - it saves trace work and light-ring space, not memory:
- **GPU:** every per-instance surfel array is sized for the fine level (E 3, validity 1, moments 32, basis 9, estimator 24, moment
  counts 16, EmitB 9 words: about **376 B a fine surfel**, plus about 72 B of the filter's scratch; `m_RelevelB` keeps a full copy of the
  data after the first level switch). The ambient cube resizes nothing.
- **CPU:** only `m_E` / `m_Valid` are replaced at a switch; every other per-surfel array keeps its fine capacity (no `shrink_to_fit`):
  about 610 B + 36 B per filter pass a surfel, at the fine count once an instance has been fine.
- **Shared per mesh:** all of `surfel_mesh` (levels, read grid, neighbours, origins, vertex surfels, BVH), packed once per mesh.

Global limits that cap a big scene whatever the per-mesh cap:
- The light ring: 2^17 surfels a frame slot (x3 slots); an instance that does not fit gets **no GI** (counted as dropped). Counts the
  current level: the one cap coarse levels help. The moment ring has the same layout (24 MB host-visible).
- Fine surfels per mesh < 0xFFFF: only an `assert` in the renderer (nothing in Release).
- Read-list patch counts clamped to 0xFFFF.
- Link keys `instance << 16 | surfel` with bit 31 the tail flag: about 32,767 GI instances, unchecked.
- Emitter table `max(1024, 2 x count)`: past it, a full repack.

So bounded residency needs real changes: per-level allocation on the GPU (and the CPU), the light ring sized or paged for a big scene,
the link key encode/decode with checks. Measured before the Bistro's runtime is attempted.
