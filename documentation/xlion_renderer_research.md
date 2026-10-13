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
- **Stale results (the contract; current behavior / gap):**

  | Case | Behavior |
  |---|---|
  | Map never rendered | Not sampled: the light is unshadowed until its first render (done) |
  | Light moves / resizes before its refresh | Gets a new map; the old one is freed, never sampled with the new mapping; unshadowed until rendered (done; gap: a moved light under backlog pops) |
  | Over budget | A dirty map keeps its last render - an allowed approximation, not a cache hit (done) |
  | Continuous overload | Never-rendered first, then maps waiting `urgent_age_v` (4) frames or more, the longest first, then screen size: none starves (done, tested) |
  | Caster appears, disappears, moves, skinned | Its old and new bounds invalidate (done) |
  | Caster material changes (alpha test, displacement) | The caster key is mesh + transform + materials (done, tested) |

  The budget is a map count, not time: record scratch raster time, remap time, texels updated, caster draws, waiting
  age and frame spikes before it grows into a cost-based queue.
- **Precision and mapping tests still owed** (cached == rebuilt passes when both share a defect): large clipmap depth
  ranges (16 bits over 2 x the level's half range), thin casters, grazing receivers, clipmap level boundaries,
  octahedral folds and corners, PCF at tile borders, the cube -> octahedral remap against a cube-map reference.
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
1. **Atlas memory per platform (done, shadows phase 3 is the specification).** One `DEPTH_U16` atlas: the clipmap's
   linear orthographic depth, the local maps' linear distance (hyperbolic depth in 16 bits loses the far range), a
   D32F scratch for the local maps' casters. 8192x4096 = 64 MiB on PC, scratch excluded. STB uses 16 bits.
2. **A per-frame update budget (done for the local maps; see the stale-result contract).** Fifty lights moving at
   once must not spike the frame (STB load-balances the same way). How often the lighting picks a light may become one
   input of the priority, never the only one (a rarely picked light would get poor shadows and be picked even less).
3. **Light selection that does not scale with light count:** the world-cell reservoirs below, superseding the
   32-light loop and the CPU 32x32 screen-tile binning HypeHype 2024 ships (also no compute, but per view).
4. **Slim per-draw data (done).** `ubo_static_geom_mesh` dropped `m_L2CShadowT` and `m_L2w` became the affine matrix's 3
   rows: 192 -> 112 bytes. xGPU pads each uniform entry to the GPU's `minUniformBufferOffsetAlignment` (was: the
   renderer padded every struct to 256), so the stride is per device: 128 bytes per draw and 64 per material on the
   RTX 4070 Ti SUPER here (alignment 64), still 256 where the alignment is 256. Report logical bytes, stride and
   uploaded bytes per target device.
   **Persistent instance buffer (done):** `draw::m_Instance` (low 32 bits: the caller's id, high 32: flags) keeps a slot
   whose 3x4 transform (relative to a snapped "little world" origin) is written only when it changes - visibility never
   touches it; id 0 takes a transient slot written that frame. The per-draw data is now just the slot index in the
   push constants; the view block (w2C + origin offset) is written once per view / shadow tile. Slots are evicted
   least-recently-seen only when the buffer is full (invisible is not dead). `features/instances/xlionrender_instance_slots.h`.
   One device-local instance buffer, kept up to date like Unreal's GPUScene and Bevy: the changed slots (and the
   transient ones) are written into this frame's region of a CPU ring (a region per frame in flight), and the GPU copies
   them over (`xgpu::window::CopyBuffer`: recorded between render passes, barriers on both sides) before the passes
   that read them. What does not fit a frame's region stays pending for the next render. Indexed loads are slow on mobile; uniform-address loads are the fast path.
5. **Pipelines built at material load,** not at first draw (first-sight stutter). HypeHype builds every PSO at load.

**Octahedral maps (adopted for point-light shadows, planned for reflections):**
- **Point shadows:** render the 6 cube faces into a scratch D32F (3x2 tiles), then a remap pass writes **one**
  square octahedral tile of linear radial distance into the local atlas, with a 2 px border for PCF (STB). One tile
  instead of six, one lookup, no face selection in the shader, any power-of-two size (sized by importance). Spot
  lights go through the same remap (linearize + border), so the atlas has one format and one sampling path.
- **Reflections:** octahedral probe maps instead of cubemaps: filtered in one pass instead of six, packable in a 2D
  atlas with the same tile manager, and WebGL2 has no cubemap arrays (the web target).

**Taking STB further: world-cell light reservoirs.** STB picks, per *screen* tile, 1-4 lights out of all of them
(two-stage stratified reservoir sampling) and leans on TAA to remove the noise, which they have not solved. The
world-space version exists: **ReGIR** (Boksansky et al., *Rendering Many Lights with Grid-Based Reservoirs*, Ray
Tracing Gems II): reservoirs in world cells (scrolling clipmap or hashed), proposals from approximate importance,
resampled at the shading point with its BRDF. Its known limits are our test cases: cell granularity and temporal reuse
bias, and it suits secondary hits better than primary shading in complex scenes. What xLION adds:

- **Cells in the little world, not on the screen.** Reservoirs live in a camera-centered clipmap of world cells
  (the shadow clipmap's structure: world-anchored, slot = index mod window). A cell's light set does not depend on
  the view, so it is **cached** (re-sampled only when a light that reaches it changes - the same conservative
  invalidation as shadow tiles), **shared by every view** whose window covers it (distant views need their own
  windows over the same world cells), and **the same candidates for both XR eyes** - not the same samples: the eyes can
  fall in different cells and see different surfaces (stereo cell-boundary tests when XR starts). View-specific
  refinement (depth bounds) never removes candidates from the shared cells (other views, translucents, volumes).
- **Exact head, sampled tail.** Each cell keeps its top-K lights (K = 4..8, by the importance PDF) evaluated
  **exactly**, and only the remainder - the tail of weak contributors - is stochastic (1 weighted sample) or merged
  into one unshadowed aggregate light (lightcuts-style). With at most K lights reaching a cell the result equals
  today's loop: noise-free, so still frames in the editors are clean and our image tests stay deterministic (cached
  == rebuilt). "Weak" is the cell's scalar estimate: a tail light can still dominate a sharp highlight on one surface
  (test rough metals, grazing angles, colored materials, spots, receivers near a light). The estimator is
  L = sum over the head of f_i(x) + f_J(x) / p_c(J), J drawn from the tail with its real probability p_c: head and
  tail never share a light, and every tail light that can contribute keeps p_c > 0.
- **Visibility from the cached shadow tiles.** The PDF uses a coarse occlusion term from the light's (cached) atlas
  tile over the cell's bounds. It **lowers** a light's importance, with a positive floor: a few coarse samples do not
  prove a whole cell is occluded (STB documents that darkening bias). Weight 0 only where occlusion is established
  conservatively. These proposals are dirty when casters change, the light's map is refreshed or resized, or the
  light changes - not only when the light changes.
- **One importance estimate, several inputs.** Screen size, distance and how often the cells pick a light drive the
  shadow map's size and its update priority, with a minimum service so no light starves.
- **Tail refresh policy (to measure).** Re-showing the same tail sample every frame adds no evidence. Options: a
  fresh sample each frame (temporal refinement), a fixed approximation, a settled/capture mode of higher quality, or a
  small cached pool with a changing pick. A fixed seed makes a test reproducible, not exact.

**A clustered builder for PC.** The cells are clusters: froxels moved into the little world (clipmap levels give the
fine-near / coarse-far resolution froxel depth slices give, without depending on the view). One data layout and one
shader lookup, two builders: the CPU on mobile (no compute, as HypeHype), a compute pass on PC once xGPU has compute
(rebuilt each frame or only the dirty cells, more exact lights per cell, depth-bounds culling against the depth buffer).

No compute needed on mobile: the CPU fills the cells (HypeHype bins 32x32 tiles on one thread inside L1 - a cell clipmap is
the same order of data) and uploads them as a small buffer; the shader finds its cell from the world position like
`ShadowClipmap` finds its tile. Fixed cost per pixel: K exact lights + 1 tail sample. Compute-side culling and
temporal reservoir reuse come with xGPU compute. Open points: cell size per clipmap level, K per platform, the tail
(sample vs aggregate - measure both), translucent and volumetric lookups (cells are 3D, so they work there too,
which screen tiles do not). Bounded is the shading stage only: the cell build and upload follow the light count -
measure them too.

**Next lighting deliverable: a three-mode lab** on the current shadow atlas and the same PBR shading:

| Mode | What it tells |
|---|---|
| Exact reference | Every light, exactly (the fastest exact baseline, not only today's 32-light loop) |
| Exact world-cell candidate lists | Whether the cells help - construction, sharing, lookup - with no stochastic error |
| Exact head + sampled tail | The bounded-cost proposal |

**Lab, first results (2026-10-13; `xlionrender_test_light_lab`, `instance::m_LightMode`).** Local lights moved to storage
buffers (up to 4096; the first 32 keep shadow maps); cells: 3 levels x 16^3, level 0 cells 2 units, rebuilt on the CPU
each render (`features/lights/xlionrender_light_cells.h`), spot cones culled conservatively (a light is left out of a
cell only where it is exactly 0). Measured at 160x160, 256 lights of radius 3-7 (RTX 4070 Ti SUPER, Debug build):
- CELLS == EXACT bit for bit (few and dense lights); SAMPLED == CELLS bit for bit when every light fits in the heads.
- SAMPLED: no bias detected - the mean error of N frames falls as expected for an unbiased estimator (0.32 -> 0.12 from
  8 to 64 frames). Consistent with unbiasedness, not a proof (a bias floor can appear later; ensembles of seeds against
  CELLS are the stronger test).
- But one frame is very noisy: 0.81 mean relative error with 1 tail sample, 0.44 with 4, 0.19 with 16 stratified
  samples - and 16 samples already cost what CELLS costs. More samples is not the road.
- Release, 1280x720 (`XLIONRENDER_LAB_BENCH=1`), GPU scene / CPU cell build / cell data:

  | Lights | EXACT | CELLS | CELLS build | Cells |
  |---|---|---|---|---|
  | 256 | 0.231 ms | 0.023 ms | 0.16 ms | 214 KB |
  | 1024 | 0.996 ms | 0.066 ms | 0.53 ms | 283 KB |
  | 4096 | 5.653 ms | 0.248 ms | 1.97 ms | 562 KB |

- The builder is light-centric (each light visits only its own cell range; count / prefix / fill; light order kept
  without sorting). Boundary tests: points just inside every radius in 26 directions and just inside spot cones, a light
  enclosing the camera, a very narrow cone, a camera far in negative space, all levels. Outside the cells or past the
  cell buffer: an explicit EXACT fallback (`getLightModeUsed`), never a truncated list.

**CELLS vs clustered shading (context).** For exact shading the two are near equivalent - both rebuild a 3D grid every
frame (we track nothing yet either). Clustered (view froxels) is better at: a GPU build that scales to many lights, a
resolution fitted to the screen, full-screen coverage (ours: a window around the camera, outside it every light is
looped). World cells are kept for what froxels cannot do: lights at points outside the view frustum (reflection
probes and their captures, GI injection, ray-traced secondary hits - ReGIR's reason -, off-screen particles), no
compute (xGPU today, mobile), one build for several views (editor viewports, XR eyes, probe faces), and lists that do
not change when the camera turns (what temporal reuse in SAMPLED needs). To close the gaps: a compute builder once
xGPU has compute, a coarse far level instead of the all-lights fallback, optional per-view trimming of the lists.

**Decision: CELLS is the default** (`instance::m_LightMode`), EXACT the reference and fallback. Next for CELLS:
separate membership (position, radius, cone, add/remove, window) from proposal data (intensity, color, visibility) so
a color edit does not rebuild lists; reuse across views with compatible coverage; measure the complete frame (snapshot,
build, upload, recording, GPU, CPU wait). **SAMPLED stays experimental.** Order: noise diagnostics per pixel (which
light, its probability, f/p, diffuse vs specular, shadowed) -> shading-point resampling (a few cell candidates
re-weighted with the exact falloff, cone, orientation and a BRDF estimate; visibility kept out or conservative) ->
diffuse / specular split (exact specular + sampled diffuse is a candidate) -> demodulated diffuse -> a small spatial
filter -> temporal reuse last, judged on moving sequences, not 64-frame stills. The 32 shadowed lights make this a
many-light selection test, not yet a many-shadowed-light one: add shadowed lights in the tail, a moving blocker.

Test matrix: few lights (the head alone), dense overlap, glossy receivers, a thin occluder (zero support), a moving
blocker with fixed lights (visibility invalidation), cell and clipmap crossings, two distant views, a sustained shadow
backlog. **Acceptance before it replaces the exact path:** correct estimator weights with nonzero support,
visibility-dependent invalidation, multi-view coverage without shared depth-based exclusions, and a measured tail
refresh / settled-view policy.

**Replacement passes (picking, wireframe, debug) - the contract:** a replacement shader declares the vertex / geometry
kinds it supports, whether it keeps deformation (skinning) and coverage (alpha test), the resources and outputs it
needs, its depth and blend state, and whether it takes part in material dependency tracking. "Same draw path" means
the same instance traversal and geometry submission, not a way around the resource validation.

**Timings:** the timestamp ring collects GPU times without stalling, but `EndFrame` still waits for the GPU. Measure the
CPU wait separately from the GPU time before reading throughput or latency from them.

**Later (obvious, related):** atlas pages (an array of local atlases, a buddy allocator each) with an LRU that evicts the
maps of lights not seen for a while, and far lights shrinking before anything is evicted.

**Research needed - from inferred lighting** (Kircher & Lawrance, *Inferred Lighting*, Sandbox 2009). The full pipeline
(G-buffer, light buffer, material pass) is not adopted: GGX specular does not factor out of the material, and the geometry
is drawn twice (HypeHype rejected that for dense kitbashed scenes, and tiled GPUs bin twice). Two ideas to study:
- **Discontinuity-sensitive upsampling** of any term computed at a lower resolution (AO, the reservoirs' stochastic tail,
  per-quad shadows - STB's silhouette artifacts): a low-resolution sample counts only if its object ID and normal-group ID
  match the pixel's. Open: normal-group IDs are per-face data the pipeline must produce and keep (cluster splits, LODs,
  skinning, procedural meshes); depth + normal comparisons may be the practical substitute.
- **Irradiance demodulation:** the stochastic diffuse part is written as irradiance, denoised / accumulated, then multiplied
  by albedo in the material shader, so the denoiser never blurs texture detail. Diffuse only (specular does not factor).

**Worth knowing, later:** fp16 lighting math (Filament's Lagrange-identity GGX for half precision); barriers that
let vertex work overlap the previous pass (~15% GPU when missing); post-processing folded into the tone-map pass;
half resolution + 30 fps on slow GPUs; GTAO at half resolution with a bilateral upscale; material data packed as
fp16 in one 128-bit load; draw streams (only the changed fields per draw) and user-land bind groups (xGPU
direction); generational handles (index + generation, weak references).

**Order:** items 1 and 2 with the octahedral point shadows are done (shadows phase 3; one 16-bit atlas instead of two:
the clipmap's orthographic depth is linear, so 16 bits hold it too). Next: item 4, the world-cell reservoirs, item 5.

## Performance baseline: the stress scene (2026-10-13)

`src/tools/stress/xlionrender_stress.cpp` (target `xlionrender_stress` in Build/test; windowed, scripted, a CSV per frame):
a procedural kitbashed city of xprim_geom primitives as static cluster geometry with persistent instance ids, street
lights (the first 16 shadowed, a quarter spots), the sun with the clipmap. Release, RTX 4070 Ti SUPER, 1280x720, no
vsync, 40x40 blocks = 8001 draws, 1024 lights, CELLS:

| Scenario | Frame | CPU Render | CPU shadows | CPU cells | CPU wait | GPU scene | GPU shadows | Maps / tiles per frame |
|---|---|---|---|---|---|---|---|---|
| flyover | 11.5 ms | 4.5 ms | 1.1 ms | 0.08 ms | 7.0 ms | 6.7 ms | 0.01 ms | 1 (p95 8) |
| street | 7.3 ms | 4.5 ms | 0.9 ms | 0.27 ms | 2.8 ms | 2.5 ms | 0.004 ms | 0.3 |
| moving_lights | 9.4 ms | 6.3 ms | 2.7 ms | 0.29 ms | 3.1 ms | 2.7 ms | 0.04 ms | 16 |

Reading (what to do next, by the size of the win):
1. **No culling.** The street walk draws all 8001 draws, most behind the camera or the next building; flyover is
   GPU-bound (7 ms waiting on a 6.7 ms scene pass). Frustum culling first, occlusion later.
2. **Draw recording ~3 ms of CPU for 8001 draws** (Render minus its measured stages): one pipeline bind, material
   entry and push per draw. Sort by mesh/material and instance the repeats (the instance buffer already holds the
   per-draw data); move the constant material into the instance record (the material ring is a stopgap).
3. **The caster diff hashes every draw every frame** (~1 ms): key it by instance id and the slot's change flag instead.
4. Moving shadowed lights cost CPU (2.7 ms recording their maps), not GPU.
The cell build is not the bottleneck at 1024 lights (0.1-0.3 ms).

**Frustum culling (done; `features/culling/xlionrender_frustum.h`, `instance::m_bFrustumCull`, tested: the culled image
equals the unculled one).** Same scene:

| Scenario | Frame | CPU Render | GPU scene | Draws drawn |
|---|---|---|---|---|
| flyover | 10.0 ms (11.5) | 3.4 ms (4.5) | 6.2 ms (6.7) | 5180 of 8001 |
| street | 5.7 ms (7.3) | 3.1 ms (4.5) | 2.2 ms (2.5) | 4460 of 8001 |
| moving_lights | 7.7 ms (9.4) | 5.0 ms (6.3) | 2.3 ms (2.7) | 4460 of 8001 |

Less than the street walk promised: looking down a long street keeps half the city in the frustum - the buildings
hide it, which is occlusion culling's job (a hierarchical depth test, or portals for interiors). Next by size: draw
recording (~2 ms for ~5k draws), the caster diff (~1 ms), shadow map recording for moving lights (2.7 ms).

**Then (same scene, same machine):** the constant material in the instance record + draws sorted by state with binds
only on change; shadow casters tracked by instance slot (a compare, not a hash); 5 cell levels instead of 3 (from above,
the city was outside the +-64 unit window and every pixel looped all 1024 lights - CELLS cost what EXACT cost):

| Scenario | Frame (first baseline) | CPU Render | GPU scene | FPS |
|---|---|---|---|---|
| flyover | 4.2 ms (11.5) | 1.1 ms (4.5) | 2.8 ms (6.7) | ~240 |
| street | 2.3 ms (7.3) | 1.0 ms (4.5) | 1.0 ms (2.5) | ~440 |
| moving_lights | 3.8 ms (9.4) | 2.5 ms (6.3) | 1.0 ms (2.7) | ~260 |

Then: the cell headers packed into one uint (320 -> 80 KB per render; an overflowing list is shaded exactly), and local
light maps gather their casters once per light (sphere) and draw per face only the ones in that face's frustum (near
plane skipped: pancaking) - moving_lights CPU shadows 1.7 -> 0.47 ms, frame 3.8 -> 2.8 ms (~360 FPS).
Left: occlusion culling for the street (~4.4k of 8k draws drawn); flyover is GPU-bound (2.8 ms scene).

**xGPU fix found by this scene:** the descriptor pool ring reset one pool per frame and grew each new pool by 1.5x; a
frame that overflowed (the city binds ~100k sets) made every later frame create a larger pool - CPU time per frame
exploded and the driver stalled the machine. Now: fixed-size pools all reset after the frame's fence, a hard cap, and a
frame reuses a descriptor set already written with the same buffers (~100k allocations -> a few dozen).

## Reflections (step 1 done: the sky probe)

`scene::m_Sky` -> an octahedral source pyramid (256 .. 8, folded borders) -> a 6-level roughness atlas (GGX importance
sampling, filtered: each sample reads the pyramid level matching its solid angle - 64 samples) + the split-sum DFG table;
rebuilt only on a sky change. Lit pipelines bind it at samplers 8 and 9. Tests: a uniform sky gives every texel of every
level the sky's value, levels keep the mean, a furnace sphere shades to ~1, a static sky builds once.

Against the references: Godot 4 (octahedral radiance like ours, filtered importance sampling - adopted -, update modes
quality / incremental / realtime); Flax (cube maps, probes rendered one face per frame - to adopt for local probes -, then
filtered per mip). Roughness -> level mapping, measured (CPU, exact GGX prefilter of a high-contrast sky vs the blend of
6 levels): linear 1.50% mean / 2.4% max error, square root (Godot) 1.53% / 3.0% - kept linear.
**Step 2 done: local reflection probes** (`scene::m_Probes`). The renderer captures 6 faces (128^2, the same scene
pass as the main view), remaps them into the octahedral source, then the sky's pyramid + prefilter into the probe's slot of
a 4x4 probe atlas. `m_ProbeFacesPerFrame` faces per frame (default 1, Flax's way; never-captured first, then the nearest);
re-captured when a changed caster's bounds touch its box. Shading: box-projected (parallax) specular + diffuse, probes
blended innermost first by their boxes' fades, the sky (or the flat ambient) for the rest. Tests: the captured map is red
toward the red box and green toward the green one (faces, orientation, remap), a capture at 1 face/frame completes on
frame 6, a static scene / a change outside the box does not re-capture, a mirror sphere reflects the probe.
Stress (street, 8 probes): +1.2 ms per frame while capturing (one face), nothing after.
**Step 3 done: many probes, cells, light changes.** Any number of probes per scene: the nearest 32 (eye to box) hold a slot
of an 8x4 atlas (3072x1024 RGBA16F, 24 MiB), the others wait (a probe swapped in is captured again). The probes' records
share the light buffer (64 bytes, like a light; innermost first) and the world cells list them (a second packed header per
cell, `light_cells::AddBoxes`), so a pixel walks only the probes whose box reaches its cell; EXACT mode walks them all.
Re-capture also on a changed local light reaching the box (hash of the lights touching it), and on a sun / sky / ambient
change; a change during a capture lets it finish then captures again (a continuous change never starves it), except a
moved probe. Tests: CELLS == EXACT with nested + overlapping probes (max difference 0); 40 probes -> 32 captured, the camera
at the far end swaps in 8; a far light does not re-capture, a light moving into the box / changing color does, so does the
sun. Stress (street, 64 probes): GPU scene 1.09 ms (1.06 with 8).
ponytail: residency has no hysteresis (a probe on the boundary can swap as the camera moves); the light key tests every
resident probe against every light (the cells' lists when it shows).

## Diffuse GI research (2026-10-13)

Owner direction: grid probes (re-rendered or relit) are rejected. GI, like reflection probes, is "location, location,
location": a probe in the wrong place leaks or misses, and every probe engine spends its complexity fighting that. Study every
reference; do not settle for an existing technique if we can argue for a better one.

### What each reference does (read from the source in E:\source_code)

| Engine | Tracing structure | Where radiance lives | Dynamic geometry | Fights location with | Cost / limits |
|---|---|---|---|---|---|
| **Godot SDFGI** (`renderer_rd/environment/gi.cpp`, `sdfgi_*.glsl`) | Up to 8 cascades of 128^3 voxels around the camera, voxelized from **static** meshes, SDF by jump flood (`sdfgi_preprocess.glsl`); per-voxel light with anisotropy (`sdfgi_direct_light.glsl`, RGBE) | **Grid probes**: 17^3 per cascade (`PROBE_DIVISOR` 16), SH -> 6x6 octahedral (`sdfgi_integrate.glsl`), rays marched through the SDF with history | Not in the SDF: dynamic objects only receive | Per-probe occlusion texture, cascade blend | Blotchy, leaks at thin walls, cascade seams; PC/console only |
| **Godot VoxelGI** (`voxel_gi.glsl`) | Baked octree per volume; dynamic objects re-voxelized each frame into a dynamic map | Voxels, cone traced | Partly (dynamic map) | Voxel occlusion | Needs a bake, local volumes only |
| **Flax** (`Renderer/GI`, `Utils/GlobalSignDistanceFieldPass.cpp`) | **Per-mesh SDFs** (built per model asset) composited into a cascaded global SDF (R8_SNORM, 32^3 chunks, dirty chunks re-rasterized) | **Global Surface Atlas**: each object's 6 axis projections rasterized into an atlas (albedo/normal/emissive/depth), relit by lights + GI; **DDGI** grid probes (up to 4 cascades, 6^2 irradiance / 14^2 distance texels) trace the SDF and read the atlas at hits | Yes (SDF chunks and atlas tiles redrawn when dirty) | DDGI probe relocation + classification, Chebyshev depth visibility | Lumen-lite; still grid probes for the gather |
| **Unreal Lumen** | Mesh SDFs + global SDF clipmaps (or hardware RT) | **Surface cache** (mesh cards, relit + radiosity) for hits; **screen probes** on the visible pixels (about every 16 px, 8^2 octahedral) for the gather; a world radiance cache for distant rays | Yes | Screen probes sit on the surfaces being shaded | ~4+ ms on consoles, temporal noise/ghosting, thin walls (<10 cm) leak, no mobile |
| **Wicked** (`ddgi_*`, `surfel_*`, `vxgi_*`) | Three systems: DDGI (BVH ray tracing, compute or RT API), surfel GI (RT), VXGI (voxel clipmap + jump-flood SDF, cone tracing) | DDGI grid probes / surfels on surfaces / voxels | Yes | Surfels: on the surfaces | RT for two of the three |
| **ARTnGAME Lumina** (SEGI) | Voxelized cascades (6 mip levels), re-voxelized every few frames | Voxels, cone traced (diffuse + specular) | Yes (re-voxelize) | Nothing: cone tracing leaks | Heavy, light leaks, voxel crawl |
| **HTrace SSGI** (`HRestirSSGI.compute`, `HFallbackSSGI.hlsl`) | Depth pyramid ray march (screen) | Per-pixel ReSTIR reservoirs (temporal + spatial resampling, occlusion validation, checkerboarding) | Yes (screen) | Pixels are on the surfaces | Off screen: **falls back to probes (Unity APV) or sky SH**. Screen space alone is not GI |
| **CryEngine SVOGI** (`Cry3DEngine/SVO`) | Sparse voxel octree of static geometry, built in the background | Voxels, cone traced | Mostly static | - | Leaks, static |
| **O3DE DiffuseProbeGrid** | RTXGI: hardware RT (or baked) | Grid probes | Yes with RT | Probe relocation | RT required for real time |
| **Bevy Solari** (`bevy_solari/src/realtime/world_cache_*.wesl`) | Hardware RT only | ReSTIR DI/GI per pixel + a **hashed world cache**: cells keyed by quantized position **and normal**, created where queried, size growing with distance, freed after 10 frames unqueried | Yes | Cache cells are on the surfaces, normal in the key | RT only |
| Filament, Stride, Quake 4, id Tech 5 | No dynamic GI (IBL / baked) | | | | |
| HypeHype (SIGGRAPH 2024) | Mobile: GTAO; plans a top-down Z-buffer for long range sky visibility and screen-space approximate GI | | | | The mobile reality: an AO-class budget |

### What the references teach

1. **Every grid-probe system ends up fighting placement** (DDGI relocation and classification, Chebyshev visibility, SDFGI
   occlusion textures, cascade blending). The leak is structural: a probe interpolates across a wall it cannot know about.
2. **The systems that do not leak keep radiance on the surfaces**: Lumen's surface cache + screen probes, Wicked's surfels,
   Solari's world cache (normal in the key). Location is solved by construction, not by repair.
3. **The tracing structure and the radiance store are separate choices.** SDF, voxels, BVH or screen space only answer "what
   does this ray hit"; the store answers "how bright is it there".
4. **Screen space is a detail layer, never the GI**: HTrace itself falls back to world-space probes off screen.
5. **Static-only SDFs (Godot) fail our rule** that most things are dynamic. Flax/Lumen's per-mesh SDFs composited into a global
   one handle moving and newly spawned objects, at the cost of an asset-pipeline bake.
6. **Nobody spans mobile to RT with one representation.** Godot, Unreal, Wicked and Flax have no mobile dynamic GI; Solari
   and O3DE need RT.

### Proposal: the xLION surface radiance cache (SRC), our own technique

One representation on every tier, radiance on the surfaces, cost follows change.

- **Store: a sparse hashed surface cache** (Solari's idea, without requiring RT). A GPU hash table of cells keyed by quantized
  world position **and** quantized normal, the cell size growing with distance from the camera (our "little world" levels:
  the same snapped origins as the light cells). A cell is created where something asks for it (a visible pixel, a trace hit)
  and freed after N frames unused. Each cell holds irradiance (L1 SH: directional, so normal maps still respond) and a sample
  count. The two sides of a wall have opposite normals, so they are different cells: **no leak through walls by
  construction**, and no probe to place.
- **Update: L = D + T L, stochastically (the handoff's §11 operator).** Each frame a budget of cells traces a few rays; a hit
  returns its direct light (the sun clipmap, local lights through the **world cells** - the reason world cells exist: lookups
  outside the view) + the cache's own value at the hit (one more bounce per frame: infinite bounces over time) + the sky on a
  miss. The cells are the handoff's persistent diffuse elements and the update is a stochastic Jacobi iteration. Storing a
  few hit links per cell later (re-light from the links when only lights change) is the handoff's transport-reuse
  experiment, on the same data.
- **Change response (ADGI-like):** cells whose rays crossed a changed caster's bounds, or whose lights changed (the probes'
  light keys), get more rays and a shorter history. We already produce both signals.
- **Tracing: a tier choice behind one interface.**
  - *Mobile / web:* a coarse camera-centered **brick SDF clipmap** (8^3 R8 bricks, only where geometry is), few rays per
    frame, long histories.
  - *PC raster:* the same SDF at finer levels, plus a short screen-space trace first (HTrace-style) for contact detail.
  - *PC RT:* ray queries instead of the SDF; the same cache (= Solari's world cache).
  The SDF is composited from **per-mesh SDFs baked by the resource pipeline** (Flax/Lumen): moving, rotating and spawned
  objects re-composite only their bricks (we already track changed caster bounds). Skinned meshes are receivers, with
  optional capsule proxies (a stated approximation).
- **Gather: one lookup per pixel.** The pixel reads the cache at its own position and normal (jittered in the tangent plane
  against grid look, as Solari does): the cheapest possible final gather, mobile friendly. PC adds the screen-space detail
  trace, and later per-pixel ReSTIR on top.
- **Specular synergy:** a rough reflection ray traced through the SDF reads the same cache at its hit, so reflections stop
  depending on probe placement too; reflection probes remain the fallback.

**Why it beats the references:** no probe placement (vs Godot, Flax, Wicked DDGI, O3DE); dynamic geometry included (vs Godot
SDFGI, CryEngine); world space, works off screen (vs HTrace); a cheaper gather than Lumen's screen probes (one lookup vs an
8^2 probe per 16 px + filtering); one representation from mobile to RT (nobody else); lighting changes re-light cells and
geometry changes touch only their bricks (D2: cost follows change).

**Known risks:** hash occupancy and collisions (Solari: linear probing, 3 steps; size the table from the visible surface
area); geometry thinner than an SDF voxel still leaks for tracing (mitigated by the per-mesh SDF resolution and a minimum
thickness flag); the cache shows grid structure without jitter; convergence after camera cuts (new cells start from the
coarser level or the sky); **xGPU has no compute pipelines yet** - the cache needs compute + atomics (Vulkan core, within our
mobile floor); the per-mesh SDF bake is a new xgeom pipeline product.

**Steps (each leaves tests, as the probes did):**
1. xGPU compute: pipelines, dispatch, storage buffers/images, barriers. Prerequisite.
2. The cache alone, traced against analytic test scenes (box/sphere SDFs, no assets): hash, allocation/eviction, update
   rays, gather. A Cornell box against a CPU path-traced reference.
3. Per-mesh SDF bake (xgeom) + the brick clipmap compositing, change driven.
4. Change response (caster bounds, light keys), multi-bounce, stress city numbers.
5. Tiers: screen-space detail (PC), ray queries (RT), rough specular through the cache.

### Owner discussion -> chosen direction: mesh-owned vertex surfels (2026-10-13)

The hashed surface cache above is rejected: its cells are still a world grid (sparse voxels) - an up-facing floor and table
top in one cell share light. Chosen instead (owner ideas, refined):

- **Surfels belong to the mesh, not to space.** The pipeline bakes them per mesh asset: patches of the surface (position,
  normal, area, radius in mesh space). Instances store only their surfels' light (a range of one GPU buffer, ~8 B each).
  Skinned meshes carry them with the bones; nothing spawns, so nothing pops (Wicked's surfel header has 55 constants, most
  fighting spawn density, popping, edge leaks and noise - baking removes that class).
- **Read like skinning:** a vertex stores up to 4 surfel indices (the "direction"); the vertex shader computes the weights
  from the distance to each surfel's center/radius and the normal match (a distance field over the surface), normalized.
  No stored weights. Gather cost ~ skinning, no per-pixel search, no denoiser.
- **Hierarchy = LOD:** the pipeline merges surfels into parent surfels down to a few per mesh, plus a parent table. A vertex
  keeps its finest indices; the shader walks `parent[]` to the instance's active level and weighs against the parents'
  centers - one vertex stream serves every mesh LOD and every surfel level. An instance holds only its active level
  (distance / screen size): memory and update cost follow the view. Going finer: children start at the parent's value;
  coarser: the parent is the area-weighted mean of its children. Ray hits resolve to the level the hit instance holds.
- **Solver:** L = D + T L on the surfels, coarse to fine (hierarchical radiosity, Hanrahan 1991 / multigrid: light crosses
  a room in few coarse iterations). Start on CPU worker threads (Enlighten shipped a CPU runtime on mobile): rays against
  per-mesh BVHs + a top level over instances; upload only changed surfels. Compute later.
- **Low vertex density:** the pipeline splits triangles larger than the surfel size. A per-pixel read of the same surfels
  (same weighting function, Wicked-style) is the fallback for content that cannot take more vertices.
- Later: merging surfels across instances (HLOD-like) for distant districts; stored links per surfel (transport reuse, the
  handoff §11 experiment).

**Experiment 1 (next):** bake surfel hierarchy + vertex indices on test meshes (Cornell box, moving box, triangle splitting);
CPU tracing + coarse-to-fine solve; read through vertex indices (runtime weights) and, for comparison, per pixel; measure vs
a CPU path-traced reference: quality, leaks, LOD-change popping, convergence after a change, CPU ms, memory per instance.
- **Descriptor options (xgeom, owner 2026-10-13):** a GI group, two separate settings - *GI* (off / receive only / receive +
  bounce: whether surfels are baked and whether the mesh contributes light) and *GI sample density* (split triangles
  (default) / per-pixel read / vertices as they are), with the target surfel size. Density only applies when GI is on.

### Experiment 1 results (2026-10-13, `src/tools/gi_lab/xlionrender_gi_lab.cpp`, CPU, `xlionrender_gi_lab.exe [--quick]`)

Cornell box (1 m half size) + a box, a point light; 31 test points (floor, back wall, a contact point 5 cm from the box);
error = |indirect irradiance - path-traced reference (2048 paths, 6 bounces)| / the scene's mean, after convergence.

| vertex split / surfel radius (m) | vertices | surfels | vertex read mean / max | per-point read mean / max |
|---|---|---|---|---|
| 2 / 2 (as authored) | 48 | 30 | 83% / 106% | 83% / 106% |
| 0.5 / 0.5 | 192 | 174 | 28% / 45% | 28% / 43% |
| 0.25 / 0.25 | 580 | 536 | 10.6% / 16.7% | 8.5% / 13.8% |
| 0.125 / 0.125 | 1970 | 1852 | 2.3% / 7.0% | 2.2% / 5.9% |
| **0.125 / 0.3** | 1970 | **250** | **2.7% / 17.9%** | 2.9% / 14.0% |

What it shows:
1. **The owner's idea holds: vertices carry the detail, surfels can be few.** 250 surfels under dense vertices (2.7%) match
   1852 (2.3%) and beat 536 surfels on 0.25 m vertices (10.6%). Interpolation density (vertices) drives quality; the
   solver's cost (surfels) can be ~7x lower. Authored low-poly geometry is unusable as is (83%): the pipeline split is
   required, as decided.
2. **The vertex read equals the per-pixel read** once (3) and (4) are in: the per-pixel fallback is for meshes that cannot
   take vertices, not for quality.
3. **Location problem, found and fixed: buried surfels.** Floor surfels under the box read black and darkened the contact
   (57% max error). A surfel whose rays mostly start inside geometry (> 25% back-face hits - DDGI's probe test, per surfel,
   at runtime, so it follows moving instances) gets weight 0: max 57% -> 47% (0.125 / 0.125), 40% -> 18% (0.125 / 0.3).
4. **The homogeneous vertex read:** a vertex under the box still interpolated a zero into the triangle. Interpolating
   (sum w E, sum w) as two varyings and dividing per pixel drops a buried vertex automatically: max 57% -> 7%.
5. **No leak:** a sealed room behind a 2 cm wall reads exactly 0 (lit side 1.35).
6. **Hierarchy pays twice.** A light switching on, same budget (2000 rays/frame, 250 surfels / 55 parents):
   fine only 55% / 35% / 25% at frames 1 / 4 / 8; coarse-to-fine (4 parent bounces per frame, the fine level gathers from
   the parents for 8 frames, starts from them) 16% / 8.7% / 5.3%; both reach the 8-ray noise floor (3-4%) later. As a LOD,
   the parent level read through the same vertex indices is 7.2% from the fine level.
7. Change response (box moves, 8 rays/surfel/frame, 0.2 blend): back wall 7.1% off before, at the noise floor (4-5%) from
   frame 1-2. CPU: 536 surfels x 8 rays = 1.2-2.6 ms/frame brute force on all cores (no BVH yet).

Next: a real BVH + instances (the CPU cost at city scale), skinned and moving instances (validity follows them), the
pipeline bake on xgeom meshes, then the GPU read (vertex shader: 4 indices, parent walk, two varyings) in the forward plan.

### Step 2: the CPU module (2026-10-13, `src/features/gi/xlionrender_surfel_gi.h`, test `xlionrender_test_surfel_gi`)

The lab's method as the renderer's module: `surfel_mesh` (bake: surfels, one parent level, parent table, 4 indices per
vertex, a BVH) shared by instances; `world` (instances with transforms, a TLAS, Update(lights, eye, ray budget)): dirty
instances first (moved, near a moved one within 4 m, in a changed point light's sphere; a sun/sky change dirties all),
then age over distance; per-instance LOD (the parent level beyond 40 m: finer starts from the parent, coarser averages the
children); point lights with the renderer's falloff; any-hit shadow rays; PCG random numbers.

- BVHs (mesh + TLAS, nearest child first) = brute force on 2000 rays. Cornell built from 7 instances (the box rotated):
  4.0% mean / 7.5% max vs path traced; a rotated instance = the same geometry pre-transformed.
- City (400 buildings from 3 shared meshes + a 220 m ground, sun + sky, eye at street level, 28 threads): 103k surfels at
  full detail, 42k active with LOD (61 instances fine, 340 at the parent level), 662 KB of GPU buffer. Every instance lit
  after 7 updates of 64k rays. A moved building dirties 10 instances of 401.
- CPU cost, first version (median-split binary BVH, scalar, threads spawned per update): 64k rays 7.8 ms, 16k 2.5 ms,
  4k 1.3 ms (~1 ms fixed). **After the references' methods** - a binned SAH build (16 bins, Wald 2007 / Jolt's
  TriangleSplitterBinning), collapsed to a 4-wide BVH tested with SSE (Embree's BVH4 node: bounds in rows, the near plane
  from the direction's sign; Triangle4: 4 Moller-Trumbore tests at once), nearest-first traversal that re-checks popped
  nodes, any-hit shadow rays, PCG random numbers, a persistent worker pool: **64k rays 1.8 ms, 16k 0.52 ms, 4k 0.29 ms**
  (4.3x; ~1.3 M rays/s per thread plus shadow rays). Same hits as brute force, same Cornell error.
  ponytail: SSE only (NEON for mobile next), Update still blocks its caller (asynchronous, Enlighten-style, later).

**Renderer integration - decisions needed (owner):**
1. *Which meshes:* `CreateMesh` meshes have CPU vertices; `CreateClusterGeom` (xgeom, the editor's) only GPU buffers - the
   surfel bake belongs in the xgeom compiler (the descriptor's GI options), so xgeom meshes wait for that.
2. *Low-poly render meshes:* the vertex read needs the split vertices *rendered*. With GI on, the renderer renders the split
   mesh (the descriptor's density option), or reads per pixel for meshes that keep their triangles.
3. *Non-uniform scale* (the stress city scales a unit cube per building): surfel radii are mesh-space; non-uniform scale
   stretches them. Bake per scale class, or measure distances in world space in the weight function.
4. *GPU read:* per mesh an SSBO of the 4 surfel indices per vertex (read by gl_VertexIndex: no vertex format change) + the
   surfels and parent table; per instance a base offset into one irradiance buffer (the instance record grows 4 bytes);
   two varyings (sum w E, sum w), divided in the fragment shader; the material's ambient diffuse uses it instead of the
   probes' / sky's diffuse.

### Step 3: in the renderer (2026-10-13) - the decisions above, first pass

- **API:** `instance::SetMeshGI(mesh, gi::surfel_mesh&&)` for cluster (xgeom static) meshes - the caller hands the baked surfels
  (the xgeom compiler's job next); `scene::m_GI { m_bEnable, m_RayBudget }`; `getGIStats()`. Draws with a persistent
  `m_Instance` id and a mesh with surfels become world instances (added, moved when their L2W changes, removed when no longer
  drawn); the sun, the sky (or the flat ambient) and the local lights (as points, found through a BVH over their spheres) are
  the solver's lights. One `Update` per render on the ray budget.
- **GPU read:** the constant-material pipeline uses `xlionrender_static_geom_gi_vert.glsl` (the static body with `XLION_GI`):
  per mesh an SSBO (level offsets; per vertex 4 x 16-bit surfel indices; per surfel center, radius, normal, parent), per
  frame a ring of surfel irradiance + validity (`push_static_geom` carries the instance's offset and level). Weights use
  world-space distances and the transform's mean scale (non-uniform scale: decision 3); normals in the solver go through the
  inverse transpose. Two varyings' worth (`vec4`: sum w E, sum w) at location 9, divided per pixel; `g_GI` in
  `xlionrender_environment.glsl` replaces the probes' / sky's diffuse. xmaterial materials and skinned meshes: not yet (they
  keep the non-GI vertex shader).
- **Test `xlionrender_test_gi_render`:** a room of cluster meshes (red wall, green wall, a box, a point light): without GI the
  floor's red/green is 1.00 on both sides; with GI 1.14 near the red wall and 0.90 near the green one; mean luminance
  0.55 -> 0.98; the shadow behind the box fills; a moved and a removed box; no validation errors.
- **Stress city (`--gi 16384`, 8001 draws, 1024 lights, street walk):** frame 4.3 ms (233 FPS); GPU scene 0.98 ms (1.06 without
  GI: the vertex read is free); CPU GI update 1.26 ms mean on the render thread.
  ponytail: the update blocks Render (asynchronous next), the whole ring region is rewritten per render (changed instances
  only, into a device-local buffer, next), albedo is the constant material's base color.
