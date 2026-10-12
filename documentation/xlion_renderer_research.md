# xLION Renderer — Research and Direction

Status: research and recommendation, 2026-10-12 (rev. 2: review corrections applied). A repository-specific
extension of `E:\source_code\xLION_Rendering_Implementation_Handoff.pdf` (the "handoff"); it does not
replace it. **Recommend** marks a reversible default that needs the owner's yes before it goes into content
contracts (handoff §15). Items marked **experiment** are gated: they ship only after they win a matched
measurement, and the renderer stays correct and usable without them.

Inputs: the handoff; xLION and xGPU as of `2970f58` / `947bb98`; the references in `E:\source_code`
(Filament, Bevy, Unreal, Wicked, Godot, O3DE, CRYENGINE, Stride, Flax, id Tech 5, Quake 4 alpha, HTrace SSGI,
ARTnGAME). Lore is version control, not rendering, and was skipped.

---

## Owner decisions (2026-10-12)

- **Depot:** `xLIONRender` (this repo). Documentation goes under `documentation/`. Test *source* lives
  under `src/`; test build files (CMake etc.) under `Build/test/`.
- **Instances, not a singleton.** The renderer is an `xlionrender::instance` created on an
  `xgpu::device`. A process may hold several: e.g. the Static Geom editor has its own, the level DLL has
  its own. Code duplicated across binaries (exe and DLL) is accepted. No globals. Consequence for D1:
  "one rendering service" means **one code path**, not one instance; products are shared within an
  instance, so views that should share (two viewports on one Level) use the same instance.
- **ECS components** go in `src/ecs/`, the only folder that includes xECS. They are defined once the
  renderer matures; the mesh/material/texture resources they will reference already exist
  (`xgeom_static`, `xgeom_skin`, `xmaterial`, `xmaterial_instance`, `xtexture`). These resources may
  need updates as the renderer's needs become concrete (e.g. vertex layout, material dependency metadata);
  handled when we get there, not designed ahead.
- **Existing editor code is tolerated** while the new renderer grows alongside it; cleanup later (text
  rendering likely moves to another library).
- **Conventions:** follow the LIONant depot conventions; reuse before inventing. Namespace `xlionrender`.
  Cameras use `xgpu::view` and its clip-space convention (Y-up NDC, forward 0..1 depth), not standard Vulkan.
- **Slang** is the intended future; not a priority now. GLSL + shaderc meanwhile.
- **Windows first.** Other platforms after it works.
- **Experiments:** as an xGPU example (when it grows xGPU) or in `D:\LIONant\render_tests`.
- **Target devices** are design considerations for now, not commitments. **VR/AR may become
  significant** — see §4.3.

### Source layout

```
src/
  xlionrender.h     the one include a tool needs
  core/             contract: units, color/exposure, conventions, handles + generations
  scene/            snapshot data + builder API tools fill (meshes, materials, lights, env, cameras)
  view/             view, per-view temporal state, update policy, optional outputs
  products/         registry, compatibility, publication
  graph/            frame graph, transients, timings
  plans/            forward/ (A), deferred/ (B, later)
  features/         lighting/ shadows/ environment/ post/  (each with its own shaders)
  shaders/          shared shader library generated materials include
  debug/            editor overlays (today's primitive/outline/picking code, when cleaned up)
  ecs/              components, extract system, plugin entry
  tests/            structural (Null) + GPU tests
```

---

## 0. The short version

xLION does not win by having more features than Unreal. It aims to do a few things well at once:

1. **One rendering service for every view.** Level viewport, Play view, material preview, thumbnails,
   asset previews and test captures all go through snapshot -> plan -> graph. With equivalent inputs and
   settings, views use equivalent rendering behavior.
2. **Cost follows change.** Shadows, GI, probes, thumbnails and pipelines are persistent products rebuilt
   only when they become incompatible with the snapshot. Stable views can skip scene rendering once
   exposure, lighting, histories and resources have settled.
3. **Dynamic lighting without per-light bakes.** Diffuse transport `L = D + R·F·L` (handoff §11). On RT
   PCs `F` is estimated online (**experiment**). On non-RT devices `F` can be precomputed for static
   geometry while lights stay dynamic (**separate experiment**). This is static transport preprocessing,
   not "no bake", and it is not the baseline until it wins.
4. **The material compiler emits per-pass dependency signatures** (position, coverage, response,
   emission, view, time, resources, unknown). They drive invalidation, plan routing and variant pruning.
5. **Correctness is a feature.** Reference path, metamorphic and differential tests, capture/replay, on
   the existing CI and live-editor tools.
6. **Mobile shapes the baseline.** Tile-friendly passes, physical camera, frame pacing and an upscaler
   adapter interface are in the shared path, so other tiers add optional features instead of duplicating
   the renderer.

Deferred until measured: virtual geometry, stochastic many-light, layered materials, virtual texturing,
virtual shadow maps, geometry-cached probe relighting.

**Immediate sequence:** timestamps -> snapshot-based forward rendering -> shared preview integration ->
verification harness -> conservative shadow reuse. Then the RT transport lab and the offline-transport
mobile lab as independently gated experiments.

---

## 1. Where xLION is today (audit summary)

| Area | State | Implication |
|---|---|---|
| xGPU | Vulkan 1.2 minimum. Graphics pipelines, render passes with subpasses, textures, buffers, push constants, a descriptor pool ring, deferred destroy per frame index. Windows, Xlib and a Null system. | Good base. Per the owner, xGPU grows with the renderer (§7). No second abstraction. |
| xGPU gaps | No compute dispatch path, indirect draw, timeline semaphores, dynamic rendering, ray query or acceleration structures. `EndFrame` waits for the GPU every frame (no frames in flight). **Added 2026-10-12:** `xgpu::timestamps` (non-stalling, ring of query sets) and raw-texel `ReadTexture` for HDR targets, verified by `D:\LIONant\render_tests` `hdr_timestamps_lab`. | Growth items, not blockers. |
| Existing shader library | `xgeom_static.plugin/source/runtime/`: `mb_standard_pbr.frag`, `mb_material_pbr*.frag`, exposure function, tone mappers (AgX, PBR Neutral, GT7, Lottes, Uchimura, lion), shadow, quantized cluster vertex input. `xgeom_static` is already a quantized, clustered (meshlet-like) format. | Reuse: this is the seed of the shared contract shaders (`src/shaders/`), not something to rewrite. |
| xLIONRender | Editor/debug renderer: primitives, capsules, outlines, text, picking. Walks the live ECS inside `Draw` (`xlionrender_system.h` `Collect`). `view::SCENE/GAME` and context/hidden roles exist. | Keep the view and role concepts. The snapshot replaces walking the live ECS during recording (handoff §5). |
| Materials | Graph editor + compiler -> GLSL via shaderc on `mb_standard_pbr.frag` / `mb_material_pbr.frag`; instances; texture pipeline with Basis/Compressonator. | The compiler is where dependency signatures come from (D4). |
| Editor previews | Each editor draws its own preview (`xeditor_mesh_preview.h` with private lighting, `xeditor_thumbnail_cache.h`). | Evidence for D1: previews and viewport can disagree today. |
| Resources | Static and skinned geometry (`xgeom_static`, `xgeom_skin`), materials, material instances, textures — all with resource pipelines. No render ECS components yet (mesh, light, camera, environment). | The renderer consumes these resources; components come once the renderer matures. |
| ECS | xECSV2 queries (must/one_of/none_of); no change tracking. Per-Level engine copies (N worlds live). | N worlds x M views. Change records come from the extraction boundary (§4.2). |
| Tooling | CI with sanitizers (Windows/Linux), `live.py`/`xcmd.py`, virtual input + CaptureWindow. | Harness reuses this (D5). |

---

## 2. What the references teach

Licences: Filament (Apache-2), Bevy (MIT/Apache), Godot (MIT), Wicked (MIT), O3DE (Apache/MIT) and Stride
(MIT) can be read and adapted with attribution. **Unreal and CRYENGINE (restrictive EULAs), Flax (own
licence), HTrace and ARTnGAME (Asset Store) and the id Tech 5 / Quake 4 alpha trees (unclear provenance):
ideas only, no code.**

| Reference | Take | Leave |
|---|---|---|
| **Filament** | Closest model: mobile-first PBR, physical light units and camera (EV100, pre-exposure), froxel light clustering (`Froxelizer`), small frame graph (`fg/`), material compiler with per-pass variants, `FramePacer`/`FrameSkipper`. Its PBR document is the best starting text for our contract. | Single-path rigidity at the high end. |
| **Bevy** | The extract stage (= our snapshot). **Solari**: ReSTIR DI/GI plus a hashed world cache (`world_cache_update/query/compact`) — readable candidate for the handoff's adaptive ray-updated cache baseline. | Rust/WGSL specifics; ReSTIR GI in the first slice. |
| **Unreal** | Lumen's split of where lighting lives (surface cache) from how it is gathered; VSM page caching keyed on what moved; MegaLights shows where many-light goes. | Nanite, MegaLights, Substrate, RDG/permutation scale. |
| **Wicked** | DDGI, surfel GI, VXGI, SSGI, RT shadows (FFX denoiser) in clear code; DDGI is a useful non-RT comparison point. | Several GI systems at once (handoff §14). |
| **Godot** | FSR2/MetalFX/TAA behind one interface; SDFGI as no-RT dynamic GI; the cost of three separate renderers. | Separate renderers per platform. |
| **O3DE (Atom)** | Pipelines as authored pass templates. | JSON pass assets — our plans are C++ functions. |
| **CRYENGINE** | Pass scheduler with explicit transitions; SVOGI. | Code (licence). |
| **Stride** | Shader composition. | Its own shader language. |
| **id Tech 5** | Virtual texturing. | Until texture memory is the measured bottleneck. |
| **HTrace SSGI / ARTnGAME Lumina** | Screen-space and voxel GI as middle tiers. | Code (licence). |
| **Flax `RENDERING_ANALYSIS.md`** | Mobile-first, render-list pooling. | Generic survey; superseded here. |

---

## 3. What sets xLION apart (each item earns its place)

### D1 — One rendering service, every view

```
view
    world / snapshot
    camera
    output target (+ optional outputs: entity IDs, HDR capture)
    execution plan
    quality profile
    exposure / output configuration
    temporal state (owned by the view)
    update policy (continuous, on change, on request)
```

- Guarantee: **with equivalent inputs and settings, views use equivalent rendering behavior.** A
  preview may use a different environment, exposure or quality than a game view — but through the same
  code, never a private copy.
- Temporal state belongs to the view. "Previous" means the last frame *that view* rendered; a view that
  was idle (thumbnail) or jumped resets its history.
- Views share **world** products when their requirements are compatible (world-space GI, static shadow
  regions). View-dependent cascades, clipmaps, reconstruction histories and screen-space resources are
  keyed per view or by compatible coverage.
- **Replacement passes (owner requirement, 2026-10-12):** editors draw the scene with special shaders —
  object detection/picking (e.g. `E23_Pick_frag.glsl`), wireframe, debug normals. A view can draw the same
  snapshot with a caller-supplied pass/shader through the same draw path, so these keep working.
- Picking: an entity-ID target is an **optional view output**, with declared policies for transparency,
  MSAA, overlays and readback — not an attachment in every pass.
- Why it fits xLION: `view` and roles already exist; per-Level engine copies already give N worlds; the
  material preview becomes a tiny world, deleting `xeditor_mesh_preview.h`'s private lighting.

### D2 — Cost follows change
The handoff's product registry (§9) applied to shadows, GI, probes, thumbnails and pipelines:

- **Stable views can skip scene rendering** once exposure, lighting, histories and resources have
  settled. UI composition and presentation still cost GPU; animated materials, streaming and temporal
  refinement keep a view live.
- **Edits rebuild only incompatible persistent products.** Moving a light rebuilds that light's shadow
  region and GI injection; visible shading and indirect propagation still cost frame time.
- Conservative tracking first (handoff §9, §10). Finer tracking only when it beats it.

### D3 — Dynamic lighting without per-light bakes (experiments)

Convention:

```
L = D + R·F·L        solved with batched Jacobi iterations on the GPU (handoff §11)
F: geometry / visibility transfer between surface elements
R: supported diffuse material response (albedo) per element
D: emission + current direct-light injection, same convention
```

Keeping `R` separate makes albedo edits cheap without pretending material response is geometry. An
implementation may store `T = R·F`, but then its update rules must treat albedo changes as operator
changes. Normal, coverage and topology edits change `F`; the material contract states which material
edits are `R`-only.

Two **independent** experiments share the solver:

- **RT lab (PC):** `F` estimated online with ray queries, rows maintained on geometry change. Compared
  against a Solari-style world cache (handoff §11.7).
- **Offline-transport lab (non-RT):** `F` precomputed by the resource pipeline for static geometry.
  Storage, asset invalidation and a rebuild delay after geometry edits are real costs. Precedent:
  Enlighten (precomputed visibility, runtime lighting).

What fixed offline transport supports — stated, not implied:

| Behavior | Fixed offline `F` |
|---|---|
| Light intensity / color / on-off changes | Yes, via current injection |
| Moving lights | Yes, if direct visibility is updated |
| Dynamic receivers | Yes, through the chosen interpolation |
| Dynamic direct-light shadows | Yes, by the shadow system |
| Dynamic indirect occlusion (closing door, moving blocker) | **No** — old transport still carries light through the opening |
| Destruction changing transport | Requires rebuild or a declared approximation |

Precomputed per-state rows (door open/closed) are a possible separate feature for selected content, not a
general answer.

"One solver" is the goal; one identical data layout on every target is a preference, not a requirement.

**Baseline until an experiment wins:** direct lighting + shadows, sky/environment diffuse, AO, and the
specular baseline below.

### Specular
- **Baseline:** simple environment and reflection probes (prefiltered for roughness); SSR on PC.
- **Experiment:** geometry-cached probes (store albedo/normal/depth, relight from current lights and GI).
  Open issues: newly visible geometry, current shadow visibility, offscreen coverage, roughness refiltering,
  N probes x 6 faces cost, transparent/layered surfaces. If it wins, update only affected probes or texels
  — never all probes every frame.

### D4 — Per-pass material dependency signatures
The compiler records, **per pass and per output**, which inputs can affect it:

```
position     (vertex deformation, world-position offset)
coverage     (alpha test/blend, discard)
response     (BRDF inputs)
emission
view         (camera-dependent terms)
time         (attributed to the category it reaches, not "reads time")
resources    (textures, buffers, parameter identities)
unknown      (custom code -> conservative)
```

- Time driving emission must not invalidate shadow geometry; time driving position must; time driving
  alpha affects coverage; time in an unreachable branch of the selected recipe affects nothing.
- Shader reflection gives bindings, not which parameter reaches which output — that comes from the graph.
- Uses: invalidation (handoff §5), forward/deferred routing (Plan B eligibility), variant pruning.
- Effort: a real compiler pass with tests, not a bitset bolted on. Starting point is still the existing
  graph traversal.

### D5 — Correctness as a feature
The handoff's harness (§12) on existing tools:
- **Null/headless:** structural tests only — resource legality, routing, handle generations,
  synchronization, publication compatibility. Not pixel correctness.
- **Real GPU:** shader output, analytic cases and image metamorphic tests (zero-intensity light, split
  light, exposure reciprocity, draw-order independence, translate-everything) via **HDR render-target
  readback**. CaptureWindow stays for end-to-end editor tests.
- Metamorphic relations need no golden images, so they survive driver and GPU changes.

### D6 — Mobile shapes the baseline
- Tile-friendly passes: local attachment reads via existing subpasses/input attachments first;
  dynamic rendering local read is an alternative mechanism. Tile residency is measured, never assumed
  (handoff §7).
- Physical camera and exposure.
- Frame pacing and dynamic resolution driven by GPU time.
- Reconstruction behind **explicit adapters**: one internal convention (jitter, motion vectors,
  exposure), each external upscaler (FSR, DLSS, XeSS, MetalFX) translates it. TAA baseline in-house.
- Formats per quality profile (§6).

### Deliberately not differentiators
Virtual geometry, stochastic many-light, layered material models, virtual texturing, virtual shadow maps,
runtime path tracer. An **offline reference path tracer** for differential tests (handoff §12) is allowed
later.

---

## 4. Architecture mapped onto xLION

```
xECS worlds (one per Level engine copy)        editors (material, mesh, thumbnail)
        |  extract system, end of sim tick             |  small worlds
        v                                              v
  render snapshot (per world, immutable, alive until every CPU/GPU consumer retires)
        |
        +-- products registry (keyed by requirements; world- or view-scoped)   <- D2
        |
  views (§D1)                                                                  <- D1
        |
  plan (C++ function, authored; A = forward, B = compact deferred)             <- handoff §7
        |
  frame graph (resource access, barriers, transient lifetimes, timings)
        |
  xGPU (Vulkan; WebGPU later)
```

### 4.1 Snapshot
- Built by one extraction system per world after simulation; `Draw` never iterates the live ECS.
- Flat arrays (instances, meshes, materials, lights, probes, environment, cameras), stable handles with
  generations, explicit snapshot ids.
- Lifetime by reference/epoch retirement, not "double buffered": a snapshot stays alive while any CPU job
  or in-flight GPU frame reads it.
- Previous transforms are resolved per view against the snapshot that view last rendered.
- **Recommend:** editor overlays (gizmos, outlines, text) become ordinary snapshot entries drawn by the
  same graph, collapsing xLIONRender's per-feature `Collect` loops.

### 4.2 Change records
xECS has no change tracking; adding it ECS-wide is out of scope. Instead:
- Extraction compares render-relevant components with the previous snapshot slot (or a revision bumped
  by the setter) and emits typed records: transform, topology, material, light, coverage, activation.
- Editor edits pass through xundo commands, which already know what changed.
- GPU-written data (particles, GPU skinning, compute deformation) uses a declared conservative policy.

### 4.3 XR (design consideration, not scheduled)
Keep the door open without building it:
- A view's camera is an **array of eyes** (each an `xgpu::view`) (1 for flat, 2 for stereo); never hard-code one camera per view
  in the snapshot or view API.
- Temporal state and jitter are per eye.
- Plans should not assume one camera per pass, so multiview rendering (Vulkan multiview, core since 1.1)
  can drop in later.
- Late-latched poses and foveated rendering / VRS are later features; nothing now should block them.

### 4.4 Plans
**A forward** (clustered/froxel lights; the reference) and **B compact deferred** (eligible opaque
materials, local reads where supported, forward fallback). Chosen per device profile offline; runtime loads
a manifest and never searches.

---

## 5. Platform tiers

Four separate claims per platform, never inferred from each other: (1) shaders compile, (2) the backend
runs, (3) required features exist, (4) performance/quality targets are met.

| Tier | Backend | Plan | Diffuse GI | Specular | Shadows |
|---|---|---|---|---|---|
| Web | WebGPU (later) | A | baseline; offline-`F` lab | env/probes | cached atlas |
| Mobile | Vulkan; Apple via translation layer (to verify) | A or B (measured) | baseline; offline-`F` lab | env/probes | cached atlas |
| PC raster | Vulkan | B (measured) | baseline; offline-`F` lab | env/probes + SSR | cached atlas |
| PC RT | Vulkan + ray query | B | baseline; RT `F` lab vs world cache | env/probes + SSR | cached atlas |

- **Vulkan-first.** Apple: evaluate MoltenVK first on named devices/OS versions. KosmicKrisp is a
  macOS Mesa driver, not an iOS route — verify capabilities and packaging before relying on it.
- **Vulkan floor:** stays 1.2 until the target phones are named and the 1.2 features xGPU uses are
  listed; lowering to 1.1 must earn its compatibility and maintenance cost.
- **Web:** WGSL is one part. Binding model, limits, submission, synchronization, presentation and browser
  capability differences are the rest. Timestamp queries are optional in WebGPU — the measurement path
  needs an unavailable-feature fallback.
- **Shader language:** evaluate Slang early against an acceptance suite — existing PBR template, a
  generated material graph, alpha test, vertex deformation, clustered-lighting compute, sparse GI
  iteration, resource reflection/layout, SPIR-V and WGSL validation. WGSL lowering has target-specific
  differences (bindings, instance indices, atomics, entry points). The **material IR is the durable
  interface**; the emitted language is a backend choice. Alternative: GLSL + SPIR-V -> WGSL translation.

---

## 6. Contract candidates (handoff §6, §15.3) — versioned, owner approval needed

| Convention | Candidate | Notes |
|---|---|---|
| Working space | Scene-linear, Rec.709/sRGB primaries | Document texture/source conversion and clipping. Wider output later cannot restore color discarded earlier. |
| HDR targets | RGBA16F or R11G11B10 | Quality-profile candidates, not a fixed platform rule. |
| Units | Lux (directional), lumens (point/spot), nits (emissive/sky), EV100 with pre-exposure | Approve after the exact light types and exposure equations are written. |
| View transform | One project transform (game default) used by previews too | Plus an explicitly labeled neutral inspection mode and an albedo/debug mode. All transforms versioned. |
| Camera / clip space | **Keep `xgpu::view`** (`xGPU/source/Tools/xgpu_view.h`): view matrix from `xmath` look-at (Z forward), projection with X and Y negated so NDC is Y-up, depth 0 at near -> 1 at far (optional infinite far), screen from `getC2S`. | Owner decision 2026-10-12: keep the existing convention, not standard Vulkan. Reversed-Z, if ever wanted, is a change proposed to `xgpu::view` itself, never a renderer-only convention. Depth format D32F where supported. |
| Motion vectors | Screen UV delta current -> previous, render resolution | Internal convention; upscalers get adapters. |
| Jitter | Halton(2,3) | Initial sequence, not a permanent contract. |
| Material | glTF-style metallic/roughness + emission + alpha, plus unlit | Initial scope. |

---

## 7. xGPU growth, in the order the renderer needs it

1. Timestamp queries read without stalling; validated lifetime/barrier behavior.
2. HDR/depth render targets and a capability report.
3. Compute dispatch plus storage buffers/images.
4. Existing subpass/input-attachment support exercised properly (compact deferred experiment).
5. Acceleration structures and ray queries (RT transport lab).
6. Indirect draw, timeline semaphores, dynamic rendering, descriptor indexing, async compute — each when
   its use case is concrete.

The frame graph lives in the renderer, above xGPU, so xGPU stays a thin, testable layer.

---

## 8. Milestones (handoff §13, made concrete)

- **M0 — audit + baseline.** This document, build/run commands, a timestamped capture of the current
  viewport.
- **M1 — forward reference slice.** Timestamps; snapshot + extraction; plan A; standard opaque PBR +
  unlit; directional/point/spot lights; HDR + exposure + project view transform; HDR readback.
- **M1b — shared previews.** Material preview and thumbnails rendered as views over small worlds (D1).
- **M2 — harness + graph.** Structural tests on Null; analytic and metamorphic tests on GPU; frame graph
  with explicit accesses and transient reuse; group and whole-frame timings; plan manifests.
- **M3 — conservative shadow reuse** vs always-rebuild (Track A).
- **Gated experiments, independent, sequenced by staffing:** B compact deferred vs forward; C RT
  transport lab vs world cache; C-mobile offline-`F` lab on one interior; geometry-cached probes;
  per-pass material signatures beyond the conservative default.
- **M4 — dynamic behavior.** Async publication, pending-build stress, door/blocker events, scrubbing,
  settled-view skipping (D2).
- **M5 — selection + flagship.** Frozen manifests per device profile; the dynamic-interior sequence with
  matched comparisons.

---

## 9. Out of scope now
Everything in handoff §14, plus: native Metal or D3D12 backends; JSON-authored pipelines; several
concurrent GI systems; runtime path tracer; per-platform renderers.

## 10. Decisions needed from the owner

1. Named target devices per tier (phones, GPUs, browsers, OS versions) — this decides the Vulkan floor and
   the Apple route.
2. First RT device for the RT lab and the profiling hardware available.
3. Approve or change the §6 candidates.
4. Run the Slang acceptance suite (§5) now?
5. Order of the gated experiments after M3. **Recommend:** RT lab and offline-`F` lab as independent
   tracks; compact deferred when mobile measurements say forward is bandwidth-bound.
6. Budgets: frame time and memory per tier.

## 11. Risks
- **GI experiments lose.** The baseline stays correct without them; the world cache can become the PC
  answer; an offline-`F` niche win still counts (handoff §11.8).
- **Change tracking is wrong** -> stale shadows/GI. Conservative tracking first, scrubbing,
  rebuild-vs-incremental differential tests (handoff §10).
- **Per-vendor tile behavior differs.** Plans selected per device profile; plan A is always the fallback.
- **Signature analysis is wrong** -> missed invalidation. `unknown` defaults conservative; differential
  tests against always-rebuild.
- **Scope creep from references.** §3's deferred list and handoff §14.

---

## Shadows (decided 2026-10-12)

One system that starts as a **cached shadow atlas** and grows into **virtual shadow maps**, not three separate systems.
Reference survey: Unreal's VSM clipmap (per-level depth range with headroom, depth clamp "pancaking", static/dynamic
pages kept apart), Filament (cascades + LiSPSM, no caching), Godot / Wicked (atlas, packed per frame). Perspective
warps (PSM / LiSPSM / TSM) are out: camera-dependent, so nothing caches, and they shimmer; the clipmap gives the same
"more texels near the camera" with stable, cached tiles.

- **Phase 1 (done):** `features/shadows/xlionrender_shadow_clipmap.h`, `shaders/xlionrender_shadow_clipmap.glsl`.
  D32F atlas 4096^2 of 512^2 tiles; the directional light's clipmap: 4 levels x 4x4 world-anchored tiles in light
  space (slot = tile index mod 4, no allocator, no page table); conservative invalidation (a caster that appears,
  disappears or moves marks the tiles under its old and new bounds; skinned casters every frame; a new light
  direction everything); exact texel fetches (a filtered fetch at a tile edge blended in the unrelated neighbor slot).
  Tested (`xlionrender_test_shadows`): shadows present, an unchanged frame renders 0 tiles, the cached image equals a
  full rebuild after a caster moves (10 of 64 tiles re-rendered), a camera move re-renders only the new tiles (7 of 64).
- **Phase 2 (done):** local lights and their shadows. `scene` has `m_Sun` (directional) and `m_LocalLights` (point / spot,
  up to 32, in the lighting UBO; the HDR PBR shaders loop over them: `shaders/xlionrender_local_lights.glsl`). The atlas
  is 8192x4096: the clipmap in the left half, local tiles (512^2) in the right: 6 per point light (a cube), 1 per spot
  (`features/shadows/xlionrender_shadow_local.h`). A light keeps its tiles while it does not move; they re-render when a
  changed caster's bounds touch its sphere. The caster diff (new, gone, skinned) is done once per frame for both.
  Tested (`xlionrender_test_local_shadows`): the light lights, its shadow is there, an unchanged frame renders 0 tiles,
  a moved caster re-renders only the near light's 6 faces and the cached image equals a full rebuild, a spot's 1 tile.
  Image-based PBR materials got the directional branch (they treated the sun as a zero-radius point: no light).
- **Phase 3 (done):** one 16-bit atlas (`DEPTH_U16`): the clipmap's 8x8 tiles on the left, the local lights' square on the
  right, both sized per platform (`shadows::setup` at `instance::Init`; PC default 4096 + 4096 -> 8192x4096, 64 MB). Local
  maps hold linear distance / radius; a point light's map is octahedral (one tile, down at the center, seams up), a spot's
  its frustum, both with a 2 texel border. Casters render into a D32F scratch (point: 6 cube faces at half the map size),
  then a remap pass writes the map (`shaders/xlionrender_shadow_remap_frag.glsl`, shared mapping in
  `xlionrender_octahedral.glsl`). Map sizes: powers of two from the light's size on screen with hysteresis, quadtree buddy
  allocator. At most `m_ShadowUpdatesPerFrame` maps re-render per frame (never-rendered first, then by screen size). Normal
  offset bias for the local lights (a constant bias left acne where the light grazes the ground). Tested: the allocator,
  1 map per light, caching, cached == rebuilt, the budget (2, 1, 0 maps over three frames, then equal to a rebuild).
- **Phase 2b:** per-object tiles for dynamic objects (characters) - their own tight, high-resolution tile each frame;
  tiles sized by importance; per-face invalidation; normal-offset bias (grazing-angle speckle at the terminator).
- **Later:** static/dynamic tile separation (copy the static depth, draw only the movers); virtual shadow maps once xGPU
  has compute + indirect (smaller tiles = pages, GPU page marking from the depth buffer - same pool, cache, sampling);
  contact shadows.
- The editors' grid still receives the editor's own shadow map (xeditor_tools grid); it moves when the grid becomes a
  renderer shadow receiver.

## HypeHype perspective (added 2026-10-12)

Sources in `E:\source_code`: Aaltonen, *HypeHype Mobile Rendering Architecture* (SIGGRAPH 2023 Advances); Aaltonen,
*Transitioning HypeHype to PBR* (SIGGRAPH 2024 MMG); Lempiäinen, *Stochastic Tile-Based Lighting in HypeHype*
(SIGGRAPH 2025 Advances). A UGC platform on $99 phones up to PCs, with the editor running on the device: the
closest reference to xLION's situation. Their governing rule: **prefer fixed-cost techniques**. Kitbashed user
content is unpredictable, so the cost must not follow scene complexity. Bandwidth, not ALU, is the budget.

**Confirms what we do:**
- Persistent shadow atlas, static/dynamic maps, updates only when a light moves (our cached tiles; static/dynamic
  separation is already under *Later*).
- One shadow render pass with a viewport per tile.
- A separate position stream (tiled GPUs bin with positions only; shadow passes). `xgeom_static` has one.
- Bump-allocated per-draw data with dynamic offsets, global data separate (`allocEntry` + `setDynamicUBO`).
- Mock user code first; validation-layer errors fail the tests (`g_nErrors`).
- Compact deferred (Plan B) stays a gated experiment. HypeHype went deferred because dense kitbashed geometry gave
  ~60% extra pixel waves in forward lighting, and SSAO wants the depth first. Our Level scenes will get there.

**Adopted:**
1. **Atlas memory per platform.** 8192x4096 D32F is ~134 MB: a PC number. The sun's clipmap (the camera's
   "little world": world-anchored tiles around the camera) keeps D32F at a per-platform size. Local lights get a
   separate 16-bit atlas (`DEPTH_U16`) holding *linear* distance (written by the remap pass below; hyperbolic depth
   in 16 bits loses the far range), also sized per platform. STB uses 16 bits.
2. **A per-frame tile budget.** Shadow tiles to re-render go through one priority queue (importance: screen size,
   distance, how often the lighting picks the light); past the budget they stay a frame stale. Fifty lights moving
   at once must not spike the frame (STB load-balances the same way).
3. **Light selection that does not scale with light count:** the world-cell reservoirs below, superseding the
   32-light loop and the CPU 32x32 screen-tile binning HypeHype 2024 ships (also no compute, but per view).
4. **Slim per-draw data.** `ubo_static_geom_mesh` drops `m_L2CShadowT` (always `NoShadow` now) and `m_L2w` becomes
   3x4: 192 -> 112 bytes per draw. Indexed loads are slow on mobile; uniform-address loads are the fast path.
5. **Pipelines built at material load,** not at first draw (first-sight stutter). HypeHype builds every PSO at load.

**Octahedral maps (adopted for point-light shadows, planned for reflections):**
- **Point shadows:** render the 6 cube faces into a scratch D32F (3x2 tiles), then a remap pass writes **one**
  square octahedral tile of linear radial distance into the local atlas, with a 2 px border for PCF (STB). One tile
  instead of six, one lookup, no face selection in the shader, any power-of-two size (sized by importance). Spot
  lights go through the same remap (linearize + border), so the atlas has one format and one sampling path.
- **Reflections:** octahedral probe maps instead of cubemaps: filtered in one pass instead of six, packable in a 2D
  atlas with the same tile manager, and WebGL2 has no cubemap arrays (the web target).

**Taking STB further: world-cell light reservoirs.** STB picks, per *screen* tile, 1-4 lights out of all of them
(two-stage stratified reservoir sampling) and leans on TAA to remove the noise, which they have not solved. Four
changes fit xLION better:

- **Cells in the little world, not on the screen.** Reservoirs live in a camera-centered clipmap of world cells
  (the shadow clipmap's structure: world-anchored, slot = index mod window). A cell's light set does not depend on
  the view, so it is **cached** (re-sampled only when a light that reaches it changes - the same conservative
  invalidation as shadow tiles), **shared by every view** (editor viewports, thumbnails), and **identical for both
  XR eyes** - screen-tile stochastic lighting gives each eye different noise, which is visible in stereo.
- **Exact head, sampled tail.** Each cell keeps its top-K lights (K = 4..8, by the importance PDF) evaluated
  **exactly**, and only the remainder - the tail of weak contributors - is stochastic (1 weighted sample) or merged
  into one unshadowed aggregate light (lightcuts-style). With at most K lights reaching a cell the result equals
  today's loop: noise-free, so still frames in the editors are clean and our image tests stay deterministic (cached
  == rebuilt). Noise appears only where many weak lights overlap, where it matters least.
- **Visibility from the cached shadow tiles.** The PDF uses a coarse occlusion term taken from the light's (cached)
  atlas tile over the cell's bounds: a light fully hidden from a cell gets weight 0, without STB's 4 stochastic
  shadow samples per tile (their source of bias at light boundaries).
- **One importance number.** The same per-light importance drives the cell sampling, the shadow tile's size and
  its place in the update queue (adopted item 2): the lights the cells pick most get shadow resolution first.

**A clustered builder for PC.** The cells are clusters: froxels moved into the little world (clipmap levels give the
fine-near / coarse-far resolution froxel depth slices give, without depending on the view). One data layout and one
shader lookup, two builders: the CPU on mobile (no compute, as HypeHype), a compute pass on PC once xGPU has compute
(rebuilt each frame or only the dirty cells, more exact lights per cell, depth-bounds culling against the depth buffer).

No compute needed on mobile: the CPU fills the cells (HypeHype bins 32x32 tiles on one thread inside L1 - a cell clipmap is
the same order of data) and uploads them as a small buffer; the shader finds its cell from the world position like
`ShadowClipmap` finds its tile. Fixed cost per pixel: K exact lights + 1 tail sample. Compute-side culling and
temporal reservoir reuse come with xGPU compute. Open points: cell size per clipmap level, K per platform, the tail
(sample vs aggregate - measure both), translucent and volumetric lookups (cells are 3D, so they work there too,
which screen tiles do not).

**Worth knowing, later:** fp16 lighting math (Filament's Lagrange-identity GGX for half precision); barriers that
let vertex work overlap the previous pass (~15% GPU when missing); post-processing folded into the tone-map pass;
half resolution + 30 fps on slow GPUs; GTAO at half resolution with a bilateral upscale; material data packed as
fp16 in one 128-bit load; draw streams (only the changed fields per draw) and user-land bind groups (xGPU
direction); generational handles (index + generation, weak references).

**Order:** items 1 and 2 with the octahedral point shadows are done (shadows phase 3; one 16-bit atlas instead of two:
the clipmap's orthographic depth is linear, so 16 bits hold it too). Next: item 4, the world-cell reservoirs, item 5.
