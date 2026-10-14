# xLION surfel GI — the algorithm as built (for review)

Status: working, in the editor, on a 92k-triangle interior level ("Police Station", 59k surfels). Written 2026-10-14 to get an
outside review. Code: `src/features/gi/xlionrender_surfel_gi.h` (CPU: bake, solver, read structures), `src/shaders/xlionrender_surfel_gi.glsl`
(GPU reads), `src/shaders/xlionrender_static_geom_body.glsl` (vertex read), `src/shaders/mb_standard_pbr*.frag` (per-pixel read),
`src/plans/forward/xlionrender_plan_forward.cpp` (upload, material capture). Tests: `src/tests/xlionrender_test_surfel_gi.cpp`,
`src/tests/xlionrender_test_gi_render.cpp`.

## 1. Goal and constraints

- Diffuse indirect lighting (multi-bounce) for static meshes, dynamic lights, movable instances, no hardware ray tracing.
- Must look stable (no swimming, no flicker), must not leak light through walls, must not show its own structure (no blotches,
  squares, seams). Low frequency is acceptable; wrong contrast is not.
- Reference points we compare against: Wicked Engine's surfel GI, DDGI, Lumen, Godot SDFGI. Godot looks washed out but clean;
  our goal is Godot-clean with more correct contrast.
- Direct light (sun, point/spot lights with shadow maps) and the sky probe are rendered separately; the GI only adds indirect light.
  Short-range contact shading (below surfel size) is meant to come from SSAO later, not from the GI.

## 2. Data model

### 2.1 Surfels belong to meshes (baked offline)

A surfel is a disk on the surface, in **mesh space**: center, normal, radius (one radius per mesh, the asset's "SurfelSize", e.g.
0.5 m for the Police level, 0.25–0.3 m typical). Surfels are baked by the mesh compiler and stored in the compiled asset, so every
instance of a mesh shares them. Two levels: fine surfels (radius R) and parents (radius 3R, the LOD), each fine surfel knowing its
parent.

### 2.2 Bake (compiler, `surfel_mesh::Bake`)

1. **Candidates** on the triangles: about 4 per R² of area, plus at least one per R/2 of each triangle's longest edge (thin
   strips such as frames and sills need surfels along their length). Each candidate takes its triangle's face normal (oriented to
   the smoothed normal's side).
2. **Poisson-disk dart throwing** (shuffled, fixed seed) through a hash grid of cell R. A candidate is rejected only if an
   already-kept surfel (a) is closer than R, (b) has a similar normal (cos > 0.8), (c) **lies in the candidate's own plane**
   (|dot(center − P, N_candidate)| < 0.1 R), and (d) **sees** the candidate (an unoccluded segment between the two points,
   each lifted 2% of R along its normal, cast against the mesh BVH). (c) and (d) stop surfels on a different layer from claiming a
   surface: e.g. a window frame 10 cm behind blinds that sees the blinds through the slat gaps.
3. **Per vertex, its 4 nearest compatible surfels** (cos ≥ 0.5) that the vertex sees (same visibility test), widening the search
   ring if needed. Stored as 4 × 16-bit indices per vertex (meshes are limited to 65 535 fine surfels).
4. **Parents**: the same Poisson over the fine surfels' centers with radius 3R; each fine surfel's parent is its nearest compatible
   parent.
5. **Per surfel material sample** for the material capture (see 4.1): its triangle's material slot, the UV at its center, and its
   UV-space radius.
6. A BVH (4-wide, binned SAH, Embree-style triangle rows) over the triangles, used by the bake's visibility tests and at run time
   by the solver's rays.

### 2.3 Run-time per mesh (built at load, `SurfelsFromXGeomStatic`)

- **Neighbor lists** for the spatial filter (2.5): per surfel the surfels within 2R with cos ≥ 0.8 that it sees, or that it sees
  from points lifted R/2 off both surfaces (so stacked geometry like closed blinds filters together, while a wall between two floor
  surfels still blocks), weighted by linear distance falloff × cos.
- **Ray origins** per fine surfel: its center plus up to 7 mesh vertices within R/2 of it whose nearest surfel it is (each with its
  own normal). Rays start across the surfel's patch, not only from its center (2.4).
- **The per-pixel read grid** (section 3.3).

### 2.4 Solver (CPU, `world::Update`)

Instances = (mesh, transform with uniform scale, base albedo). A TLAS over instances; rays go TLAS → mesh BVH in mesh space.

Per frame a **ray budget** (e.g. 16 384 rays) is spent on instances: dirty ones first (moved, near a moved instance, reached by a
changed light), then by age / distance. Beyond 40 m an instance switches to its parent level (fewer surfels).

Per surfel update (R rays, default 8):
- **Directions**: cosine-distributed and **stratified**: ray i of R takes the radial coordinate in stratum i and the angle along
  the golden ratio, both offset by a random per-pass rotation (Cranley–Patterson), as DDGI's rotated spherical Fibonacci.
- **Origins**: ray i starts from origin (i + random shift) mod n of the surfel's origin list (2.3), cosine-distributed around that
  origin's own normal.
- **A hit** on a front face returns `albedo(hit) / π × (direct(hit) + E(hit))`, where direct = sun + point lights at the hit point,
  each shadow-ray traced; E(hit) = the cached irradiance read at the hit (vertex read on the CPU: barycentric blend of the hit
  triangle's vertices' surfels). That is one more bounce per update (stochastic Jacobi of L = D + T L, temporally accumulated).
- **A back-face hit** counts toward "buried" (no light). A surfel whose rays hit > ~25% back faces fades its validity to 0.
- **A miss** returns the sky radiance (the sky probe's mean color, or flat ambient / π).
- **Emitters** (self-illuminated materials): not found by random rays (too small). Precomputed **emitter lists**: every surfel
  links to the emissive surfels it sees (≤ 32 links, shadow-tested at build, weight A·cosθr·cosθe / (d² + A), with A = 1.47 r²),
  evaluated deterministically every update. Emission × `GIEmissiveScale` (default 100: texture emission 0..1 treated as a light).
  With emitter lists, emission found by random hits is ignored (no double count).
- **Albedo at a hit** = the captured color of the **hit triangle** (4.1); fallback the surfels' captured colors, then the
  instance's base albedo.
- **Transparent materials** (material flag `bAlpha`): their triangles are skipped by the rays (pass-through), their surfels get
  validity 0 (never read, never bounce). Same choice as Lumen/DDGI/Wicked for translucency.

Outputs per surfel, per update:
- E = π × mean(L) (irradiance), split into an **HL2 basis**: 3 directions around the surfel normal; each ray's radiance is
  distributed to the 3 directions by squared-cosine weights; the per-direction **fractions** are blended slowly (0.1). Read with a
  (normal-mapped) normal, the 3 colors are weighted by squared cosines → directional detail from normal maps.
- **Depth moments** (DDGI / Wicked style): 4×4 hemi-octahedral texels of mean and mean² ray length (capped at 2R), from the
  surfel **center** (for rays from other origins, the moment uses the vector from the center to the ray's end point).
- **Validity** (0..1), blended at ≥ 0.05 per update.

Temporal accumulation: a running average for the first 16 updates after a change, then Wicked's **MultiscaleMeanEstimator**
(short-window mean + variance; the long mean follows at a rate down to 1/256 while samples agree within their noise, catches up when
they leave it; firefly clamp at mean + 8σ). Validity below 1e-6 snaps to 0 (a denormal once produced inf → NaN that spread).

### 2.5 Spatial filter (display only)

Before upload, each surfel's 3 basis colors are blended with its neighbors' (2.3 lists, weights × neighbor validity, itself at
weight 1), `GIFilterPasses` times (default 4, each pass ≈ one radius wider). The solver keeps its unfiltered values (a filter fed
back every update would keep spreading light).

### 2.6 Upload

Per frame, once per instance: 3 × vec4 per surfel of the active level (the 3 basis colors; validity in the first .w) into a ring
buffer (1<<17 surfels per frame region), and the moments (16 × half2 per surfel) into a second ring. The mesh's static data
(surfels, vertex lists, read grid, triangle color table) is one GPU buffer per mesh (`GIMesh`).

## 3. Reading the GI on the GPU

### 3.1 Common surfel weight

For a surface point P (mesh space) with normal N and a surfel s (center C, normal Ns, radius R):
- distance falloff f = max(0, 1 − |P − C| / 2R), used as f²
- normal match c = dot(Ns, N), rejected below 0.5, faded in by smoothstep(0.5, 0.75, c)
- validity (2.4)
- **visibility (GIVisibility)**: plane weight clamp(1 − |dot(P − C, Ns)| / (0.7 · 2R)), times a **Chebyshev test** on the depth
  moments: the query point is P lifted 0.15·2R along Ns (DDGI's normal bias: on thin repeated geometry the tangent directions'
  texels read short), direction/distance from C, bilinear over 4 texels, variance floor (0.05·2R)², weight
  (σ² / (σ² + (d − μ)²))³ when d > μ.
- The value read is the HL2 basis evaluated with the normal-mapped normal.

### 3.2 Vertex read (vertex shader)

Each vertex sums w·E over its 4 baked surfels (bake-time visibility), interpolated across the triangle, divided by the summed
weight. Smooth and leak-free (its surfels were chosen with line of sight), but only as detailed as the mesh's vertices: a wall made
of 2 triangles only gets its 4 corners' surfels.

### 3.3 Per-pixel read (fragment shader) — the read grid

Purpose: every pixel reads the surfels around it, not only its triangle's corners'.

**Structure** (built at load, CPU, `BuildReadGrid`): a sparse hash grid in mesh space with cells of size R (only cells the
surface crosses; 21-bit keys per axis, open addressing, the longest probe stored so lookups stop early).
- The surface is sampled on every triangle every R/2. A point's cell is the cell of P lifted 5% of a cell along N (a floor exactly
  at a multiple of R would otherwise flip between two cells with the GPU's interpolation).
- Per cell: **patches** = groups of its sample points facing the same way (cos > 0.97) in the same plane (within 0.1 cell); up to 4
  points kept per patch. (Blinds and the frame 10 cm behind them are two patches even if they face the same way.)
- Per patch: a **list of surfels** = the fine surfels whose reach (2R) touches the cell's box, facing the patch's way (cos ≥ 0.5).
  Each entry carries **how much of the patch sees the surfel, at the 4 corners of the cell in the patch's plane** (the two axes
  across the patch's dominant normal axis): per corner, the fraction of the 4 nearest same-plane sample points (from the cell and
  its 26 neighbors, within 0.75 cell of the corner) that see the surfel (bake visibility test), cubed, 4 bits. Entry = 16-bit
  surfel index + 4 × 4-bit corner weights (one 32-bit word). A corner shared by neighboring cells is computed from the same points
  whichever cell asks, so the blended weight is continuous across cells (no square seams). Entries with all corners 0 are
  dropped; if nothing at all is seen from the patch, every facing surfel is kept at full weight (the pre-visibility behavior).
- GPU layout: table entry {key lo, key hi, first patch word, patch count}; a patch = 4 words {octahedral normal as 2 × snorm16,
  plane distance, list start, list count}, followed by its list.

**Lookup** (`GIReadCell`): hash the pixel's lifted position → the cell → among its patches with cos(patch, N) ≥ 0.5, the one with the
best score cos − |dot(P, Np) − d| / cell. N is the **face normal** from screen derivatives of the interpolated mesh-space position
(smoothed normals make the read step along triangles).

**Read** (`SurfelGIPixel`): over the chosen patch's list, weight = f² · c · fade · (corner weight blended bilinearly at P) ·
validity · GIVisibility; sum of weight × basis read. If the chosen patch gives no weight at all (a thin edge seen edge-on, whose
derived normal is unreliable), all of the cell's patches are read. Far instances (level 1): the fine surfels still decide who
reaches the pixel; their parents give the light (no pop at the level switch).

**Blend with the vertex read**: if the per-pixel weight sum W is small, the result fades toward the vertex read:
mix(vertex, pixel, smoothstep(0, 0.25, W)). (Where the per-pixel surfels are mostly out of sight, the weights fade out
continuously, so this has no edge.)

## 4. Materials

### 4.1 Material capture (GPU, once per mesh instance)

The GI needs the surface color (what a hit bounces). The real material fragment shaders are run in a capture mode: one 2×2-pixel
quad per sample in a 1024×512 RGBA16F target, UVs spanning the sample's footprint (the texture mips average the material over it),
writing albedo × (1 − metallic) in one pass and emission in another, read back after the frames in flight.
- **Per surfel** (center UV ± UV radius): used for emission (emitter lists) and as a fallback.
- **Per triangle** (UV centroid ± half the square root of its UV area): used at ray hits, so colors stop at material edges (a door
  frame's brown no longer spreads onto the wall). Batches of 65 536 triangles.
The SURFACE_COLOR debug view shows the per-triangle colors (triangle = its cluster's first triangle + gl_PrimitiveID).

### 4.2 Sky

A procedural gradient sky or a **cubemap texture** (HDR or not): drawn into an octahedral source, prefiltered into a roughness
atlas (the specular/diffuse environment); the GI's sky radiance for missed rays is the cubemap's mean color (read back from the
roughest prefiltered level).

## 5. Parameters (current defaults)

| What | Value |
|---|---|
| Surfel radius R | per asset (0.25–0.5 m) |
| Parents | 3R, used beyond 40 m |
| Rays per surfel update | 8 (`GIRaysPerSurfel`), stratified |
| Ray budget | 16 384 / frame (`GIRayBudget`) |
| Warm-up | running average for 16 updates, then the multiscale estimator (rate ≥ 1/256) |
| Spatial filter | 4 passes (`GIFilterPasses`) |
| Moments | 4×4 hemi-octahedral, capped at 2R, floor (0.05·2R)², query lift 0.15·2R |
| Emissive scale | 100 (`GIEmissiveScale`) |
| Read grid cell | R; patch tolerance cos 0.97 / 0.1 cell; 4 points per patch; corner weights 4 bits |
| Vertex/pixel blend | smoothstep(0, 0.25, W) |

Debug views (Environment / Static Geom preview): SURFELS (dominant surfel per pixel), LIGHT (GI only), SURFACE_COLOR
(per-triangle captured color); switches GIMoments (moment test on/off), GIPerPixel.

## 6. Problems met and how they were addressed (history worth reviewing)

| Symptom | Cause | Fix |
|---|---|---|
| Light through walls / door lips | reads from surfels on the other side | bake-time visibility (vertex lists, Poisson), depth moments, plane weight |
| Square ghosts | per-cell visibility cut at cell edges | removed; later per-patch corner-weighted visibility (continuous) |
| Black spreading after flying far | validity denormal → inf → NaN | snap tiny validity to 0 |
| Dark 4-pointed stars on blinds | moment texels on tangent directions read short | lifted moment query (normal bias) |
| Dashes on thin strips | too few candidates on strips | length-based candidates |
| Brown spill from door frame | one color per surfel footprint | per-triangle captured color |
| Window glass blocking the sky | glass in the rays | `bAlpha` materials pass-through |
| Large dark blobs on blinds (the latest) | (1) per-pixel lists had no line of sight: a pixel on blinds read the window frame's dark surfels 10 cm behind (one had 45% of the weight; moments failed: seen from behind, the pixel lies sideways along the gap); (2) the bake let the frame's surfels "cover" the blinds through their gaps, so the blinds had few surfels of their own | per-patch visibility-weighted lists (3.3); the in-plane condition in Poisson (2.2) |

Process note: a stale-material bug in the asset pipeline (a material is not recompiled when a renderer shader it includes changes,
and a running editor does not reload a recompiled material) hid several GPU-side fixes for a while; they are verified now.

## 7. Known weaknesses / open questions for the reviewer

1. **Remaining patchiness at window detail** with R = 0.5 m: the surfels are big for 5–10 cm geometry. Is per-mesh constant R the
   right model, or should the bake adapt R to local feature size (smaller surfels in detailed areas)?
2. **The per-pixel read structure is complex** (cells → patches → visibility-weighted lists with corner weights + a fallback blend
   to the vertex read). Is there a simpler formulation that gives line-of-sight-correct, continuous per-pixel reads? Alternatives we
   considered: per-triangle visible-surfel lists (needs triangle → list, seams at triangle edges), screen-space probe gather,
   storing a small "visible surfels" set per vertex with more than 4 entries.
3. **Visibility by ray tests at load** (read grid ≈ 1–3 s on a 92k-triangle mesh in Release, too slow in an unoptimised build).
   It belongs in the compiler; is there a cheaper visibility proxy?
4. **Depth moments are weak for thin parallel layers** (gaps between blinds and frames): is the moment test still worth its cost
   once the read lists carry visibility?
5. **Variance on high-variance surfaces** (slats): stratified directions + spread origins + estimator; is a per-surfel adaptive ray
   count (more rays where the variance is high) the better lever?
6. **No short-range AO yet**: contact shading below the surfel size is intended for SSAO/GTAO. Is that the right split?
7. **Skinned meshes** have no GI yet.
8. Per-triangle colors are one average per triangle (a large triangle with a varied texture bounces one color).
