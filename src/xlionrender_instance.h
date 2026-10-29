#ifndef XLIONRENDER_INSTANCE_H
#define XLIONRENDER_INSTANCE_H
#pragma once

// One renderer. A process may hold several (each editor its own, the level DLL its own); nothing is global.
// Owns the GPU side: pipelines, meshes, per-frame uniforms and GPU timings. Draws a scene (snapshot) into a
// render_view as seen from an xgpu::tools::view camera.

#include "scene/xlionrender_scene.h"
#include "view/xlionrender_render_view.h"
#include "meshes/xlionrender_cluster_geom.h"
#include "features/shadows/xlionrender_shadow_clipmap.h"
#include "features/shadows/xlionrender_shadow_local.h"
#include "features/instances/xlionrender_instance_slots.h"
#include "features/lights/xlionrender_light_cells.h"
#include "features/culling/xlionrender_frustum.h"
#include "features/gi/xlionrender_surfel_gi_gpu.h"
#include "features/gi/xlionrender_surfel_gi_chunks.h"
#include "plans/forward/xlionrender_plan_forward.h"     // (details::push_static_geom, draw_indexed_indirect: the half-resolution GI pass's lists)
#include "features/cull/xlionrender_gpu_cull.h"
#include "features/shadows/xlionrender_shadow_pairs.h"
#include "features/shadows/xlionrender_shadow_local_pages.h"
#include "xlionrender_view.h"
#include <chrono>
#include <memory>
#include "dependencies/xGPU/source/Tools/xgpu_view.h"
#include "dependencies/xGPU/source/Details/OffsetAllocator/offsetAllocator_inline.hpp"
#include <span>
#include <vector>
#include <functional>
#include <unordered_map>
#include <map>
#include <tuple>
#include <deque>

namespace xlionrender
{
    namespace details { struct ubo_lighting; }     // plans/forward/xlionrender_plan_forward.h

    // The GPU layout of a mesh's surfels (xlionrender_instance.cpp): pure, so a worker can pack it (Phase 7)
    std::vector<std::uint32_t> PackMeshGI( const gi::surfel_mesh& Surfels, std::span<const std::uint32_t> ClusterFirstTri, std::span<const xmath::fvec3> TriColors ) noexcept;

    class instance
    {
    public:

        // GPU time of the last finished frame, in milliseconds (measured with xgpu::timestamps, never stalls)
        struct gpu_times
        {
            double  m_ShadowMs      = 0;        // the instance copy + every shadow pass
            double  m_SceneMs       = 0;
            double  m_ToneMapMs     = 0;
        };

        // The last Render's CPU time per stage (recording, not GPU execution)
        struct cpu_times
        {
            double  m_InstancesMs   = 0;        // instance slots + the ring
            double  m_ShadowsMs     = 0;        // caster diff, clipmap / local updates, recording the shadow passes
            double  m_LightCellsMs  = 0;        // the world-cell build (CELLS / SAMPLED)
            double  m_RenderMs      = 0;        // the whole Render call
        };

                                instance        ( void )                                                        noexcept = default;
                               ~instance        ( void )                                                        noexcept;
                                instance        ( const instance& )                                             = delete;
        instance&               operator=       ( const instance& )                                             = delete;

        // Shadows: per platform sizes (features/shadows/xlionrender_shadow_local.h)
        [[nodiscard]] xgpu::device::error* Init         ( xgpu::device& Device, const shadows::setup& ShadowSetup = {} ) noexcept;

        [[nodiscard]] xgpu::device::error* CreateMesh   ( mesh_id& Id, std::span<const vertex> Vertices, std::span<const std::uint32_t> Indices ) noexcept;
        // An xgeom_static/xgeom_skin mesh (MakeClusterGeom / MakeSkinGeom). References Geom's GPU buffers (not owned, not
        // copied): they must outlive the mesh id
        [[nodiscard]] xgpu::device::error* CreateClusterGeom( mesh_id& Id, cluster_geom Geom )                  noexcept;
                      void                 DestroyMesh  ( mesh_id Id )                                          noexcept;
        // Surfel GI for a cluster mesh (CreateClusterGeom, rigid): its baked surfels (gi::surfel_mesh::Bake) in the mesh's vertex
        // order - the resource pipeline's product (the xgeom GI options); kept by the renderer. scene::m_GI turns GI on.
        [[nodiscard]] xgpu::device::error* SetMeshGI    ( mesh_id Id, gi::surfel_mesh&& Surfels, std::vector<std::uint32_t>&& Packed = {} ) noexcept;     // Packed: PackMeshGI's data made on a worker (empty: packed here)
        // A geometry over one GI mesh's cap (xgeom_static version 7): its GI in sections, each its own surfels (gi::LoadSection) and display
        // buffer; per cluster its section and the delta from its vertices to its section's (gi::LoadSection's VMap)
        struct gi_section_in
        {
            gi::surfel_mesh                 m_Surfels;
            std::vector<std::uint32_t>      m_Packed;       // PackMeshGI's data (made on a worker)
            std::uint32_t                   m_nOwned = 0;   // its owned fine surfels; its ghosts after them
            std::vector<gi::chunk_ghost>    m_Ghosts;       // per ghost: its owner (section, local index)
        };
        [[nodiscard]] xgpu::device::error* SetMeshGISections( mesh_id Id, std::vector<gi_section_in>&& Sections, std::vector<std::uint16_t>&& ClusterSection, std::vector<int>&& ClusterVertexDelta ) noexcept;
        // The mesh the selection's outline is made from (the editor's outline shader draws it expanded in screen space): per vertex position and
        // normal (6 floats), the triangles, the local box (its center and half extents) the shader's cage is made from. Once per mesh.
        [[nodiscard]] xgpu::device::error* SetOutlineProxy( mesh_id Id, std::span<const float> PosNrm, std::span<const std::uint32_t> Indices, const xmath::fvec3& Center, const xmath::fvec3& Half ) noexcept;
                      bool                 HasOutlineProxy( mesh_id Id ) const                                  noexcept { const auto* p = findMesh(Id); return p && p->m_nOutlineIndices > 0; }
        // The last render's GI update (instances, rays, CPU ms)
                      const gi::world::stats& getGIStats( void ) const                                      noexcept { return m_GIStats; }
                      const gi::world*        getGIWorld( void ) const                                      noexcept { return m_pGIWorld.get(); }      // tests, tools
                      bool                    isGICapturingAny( void ) const                                noexcept { return !m_GICaptures.empty(); }
                      int                     DumpGI( const char* pPath ) const                             noexcept;
                      // (diagnostics) What the GI reads where a ray from O along D first hits: the hit, its instance, its cell's list (each
                      // surfel: where, facing, its corner weights, its light) and the triangle's vertex surfels. The light is the CPU's copy:
                      // after DumpGI when the GPU owns it
                      std::string             DescribeGIRead( const xmath::fvec3& O, const xmath::fvec3& D ) const          noexcept;      // gi::world::Dump, and the frame log to pPath + ".frames.csv" (DumpGIFrames)
        // The GI's frame log (Phase 0 of documentation/gi_dynamic_master_plan.md): the last gi_frame_log_v frames' stage times and counts, as CSV
                      int                     DumpGIFrames( const char* pPath ) const                       noexcept;
        // A mesh's GI was made ready on the main thread (scene_renderer::MeshOf): its time goes into this frame's record
                      void                    NoteGILoad( double Ms )                                       noexcept { m_GILoadMsAcc += Ms; }
                      bool                    isGISunMapValid( void ) const                                 noexcept { return m_bGISunMapValid; }     // the GI's sun shadow map is the sun's (renderShadows)
                      void                    setGISunRays( bool b )                                        noexcept { m_bGISunRays = b; }     // the relight's sun by shadow rays, not the map
                      int                     getGIId( std::uint64_t Instance ) const                       noexcept { const auto It = m_GIIds.find(std::uint32_t(Instance)); return It == m_GIIds.end() ? -1 : It->second; }    // a draw's instance in getGIWorld (-1: none)
                      int                     getGISectionId( std::uint64_t Instance, int Section ) const   noexcept { const auto It = m_GISecIds.find((std::uint64_t(std::uint32_t(Instance)) << 16) | std::uint64_t(Section)); return It == m_GISecIds.end() ? -1 : It->second; }    // (GI sections: a draw's section's)
                      bool                    isGICapturing( void ) const                                   noexcept { return !m_GICaptures.empty(); }

        // A material: its HDR fragment shader (xmaterial::rt::getShaderHDR()) and its sampler bindings in shader order - slot 0
        // is the shadow map: the renderer binds its shadow atlas there whatever is given; then the textures. Neither is
        // copied or owned: they must outlive the id. The same material draws rigid and skinned geometry: the pipeline of each
        // vertex path is built the first time the material is drawn with it.
        // State: its pipeline settings (blend, culling, depth, shadows); a blended one is drawn after the opaque ones, sorted, and the
        // GI's rays pass through it (gi::world::SetSeeThrough)
        // Params: its parameter block (vec4s as floats: xmaterial::data_file::m_pDefaultBlock, m_BlockBytes / 4), copied into
        // MaterialParams (set 1 binding 9); the shader reads it from g_Mat.P[MaterialBase + n]. Empty: the shader reads none.
        // TextureTable (xmaterial::data_file::flags::m_bTextureTable): the shader reads the textures (Textures[1..]) through an xgpu::texture_table at
        // set 3 instead of samplers 1-7 - bindless, by the ids the renderer writes after the parameter block (two slots a vec4: texture_id,
        // sampler_id; the sampler: the texture's own); otherwise binding i of the table's set. No limit of 7 textures
        [[nodiscard]] xgpu::device::error* CreateMaterial( material_id& Id, const xgpu::shader& FragmentHDR, std::span<xgpu::texture* const> Textures, const material_state& State = {}, std::span<const float> Params = {}, bool bTextureTable = false ) noexcept;
        // A texture-table material's slot i (Textures[i + 1] of CreateMaterial) now reads Texture: the id in its block and its table (bindless: the
        // block only - a texture swap costs what a color change does)
        [[nodiscard]] xgpu::device::error* SetMaterialTexture( material_id Id, int iSlot, xgpu::texture& Texture ) noexcept;
        // The material's parameter block (CreateMaterial's Params) from vec4 FirstVec4 on: the frames recorded from now read it (a private
        // block's value changed in place: material_overrides)
        [[nodiscard]] xgpu::device::error* SetMaterialParams( material_id Id, std::span<const float> Params, std::uint32_t FirstVec4 = 0 ) noexcept;
        // How many materials exist (tests)
                      int                  getMaterialCount( void ) const                                       noexcept { int n = 0; for (const auto& M : m_Materials) n += M.m_bAlive; return n; }
        // The renderer's default material (shaders/xlionrender_default_material_frag.glsl: PBR with a diffuse texture), for a slot whose
        // material does not resolve. Textures: slot 0 the shadow map (the renderer's atlas), slot 1 the diffuse texture.
        [[nodiscard]] xgpu::device::error* CreateDefaultMaterial( material_id& Id, std::span<xgpu::texture* const> Textures ) noexcept;
                      void                 DestroyMaterial( material_id Id )                                    noexcept;

        [[nodiscard]] xgpu::device::error* CreateView   ( render_view& View, const render_view::setup& Setup )  noexcept;
                      void                 DestroyView  ( render_view& View )                                   noexcept;

        // Optional editor pass (grid, wire frame, debug lines): recorded after tone mapping into View.m_Output with
        // the scene's depth, so it draws display-space colors and is occluded by the scene. Camera is the same view.
        using overlay_fn = std::function<void(xgpu::cmd_buffer&)>;

        // Records the forward plan (plans/forward) into Window's frame: scene -> View.m_HDR, then tone map ->
        // View.m_Output (alpha 0 where nothing was drawn), then Overlay. Call between Window.BeginRendering() and
        // Window.PageFlip(), outside any other render pass.
                      void                 Render       ( xgpu::window& Window, const scene& Scene, render_view& View, xgpu::tools::view& Camera, const overlay_fn& Overlay = {} ) noexcept;

        // Draws View.m_Output into the current render pass and viewport as is (no color conversion, alpha blended), e.g. from an
        // ImGui render callback. Not ImGui::Image: the xGPU ImGui shader gamma encodes textures again.
                      void                 Blit         ( xgpu::cmd_buffer& CmdBuffer, render_view& View )      noexcept;

        // Shadows: false re-renders every shadow tile every frame (the reference the caching is tested against)
        bool                                       m_bShadowCache = true;
        // At most this many local light shadow pages render per frame besides the pinned ones (the others wait, a dirty one stays stale)
        int                                        m_ShadowUpdatesPerFrame = 64;
        // Local light pages draw only the casters in the page's frustum (false: every caster in the light's sphere)
        bool                                       m_bShadowFaceCull = true;
        // The sun's movers (casters changed lately, skinned) in mover pages of their own, never dirtying the static pages (P1b); false: every
        // caster in the pages (the tests' reference: the two layers composed must equal it)
        bool                                       m_bShadowSplitMovers = true;
        // The sun's pages drawn through GPU page/caster pairs (P1c, features/shadows/xlionrender_shadow_pairs.h): the rigid opaque clusters, a
        // draw a geometry; false (or no drawIndirectCount): a draw a cluster and page from the CPU (the tests' reference)
        bool                                       m_bShadowGPUPairs = true;
        // Tests: every filter tap of the pages through the table (the border path everywhere): the image must not change
        bool                                       m_bVSMTapTranslation = false;
        // The pages' debug views (shaders/xlionrender_shadow_vsm.glsl): 0 off, 2 the level, 3 the footprint against the texel, 4 what shaded the pixel
        // (its own level, a coarser one, none). XLION_VSM_DEBUG=n at Init
        int                                        m_VSMDebugView = 0;
        // The pages' numbers: this frame's admission (shadows::vsm::stats) and the request pass's counters of a ring ago (the pixels, the
        // ones shaded by their own level, by a coarser one, by none; crossing a page border; per level; per log2(footprint / texel) + 4)
        struct vsm_report
        {
            shadows::vsm::stats             m_Pages;
            std::array<std::uint32_t, 32>   m_Counters{};
            bool                            m_bCounters = false;     // the counters arrived (a ring after the first request pass)
            double                          m_RequestGpuMs = 0;      // the request pass (a ring late)
            double                          m_RenderGpuMs  = 0;      // the sun's pages rendered (a ring late; 0: none rendered)
        };
                      const vsm_report&    getVSMReport( void ) const                                           noexcept { return m_VSMReport; }
        // The local lights' pages (P1d): this frame's page stats; a ring late the request pass's counters (0 pixel-light pairs, 1 shaded by
        // their level, 2 by a coarser one, 3 by none, 5-9 per level)
        struct local_report { shadows::local_pages::stats m_Pages; std::array<std::uint32_t, 32> m_Counters{}; bool m_bCounters = false; };
                      const local_report&  getLocalReport( void ) const                                         noexcept { return m_LocalReport; }
        // This frame's local pages rendered (tests: which lights)
                      const std::vector<shadows::local_pages::page_to_render>& getLocalPagesRendered( void ) const noexcept { return m_LocalPages.m_ToRender; }
        // Their setup (tests: a small pool, a budget): starts them over
                      void                 configureLocalPages( const shadows::local_pages::setup& Setup )     noexcept { m_LocalPages.Init(Setup); }
        // Invariant 1 on this frame's local table
                      bool                 validateLocalPages( void ) const                                     noexcept { std::vector<std::uint32_t> T; m_LocalPages.BuildTable(T); return m_LocalPages.Validate(T); }
        struct pairs_report { int m_nCandidates = 0, m_nRuns = 0, m_nPairs = 0, m_nDropped = 0; std::int64_t m_Triangles = 0; bool m_bGPU = false; };
        // The screen tail's last frame (light_mode::SCREEN): its passes' GPU time (a ring late) and its targets' memory
        struct tail_report { bool m_bOn = false; int m_Width = 0, m_Height = 0; double m_GBufferMs = 0, m_TailMs = 0, m_DenoiseMs = 0, m_TemporalMs = 0, m_MemoryMB = 0; int m_nGBufferDraws = 0, m_nNoVariant = 0; };     // (m_DenoiseMs: both passes, m_TemporalMs the first)     // (m_nNoVariant: lit submeshes left out, their material without a G-buffer variant)
                      const tail_report&   getTailReport( void ) const                                          noexcept { return m_TailReport; }
        // Reflections milestone A (documentation/reflections_strategy.md sec. 11): the world trace's benchmark (m_ReflTraceBench), the last
        // 120 frames' means: its GPU ms, the rays (a texel of the thin G-buffer each), the hits, the BVH's visits per ray
        struct refl_report { int m_Mode = 0; double m_Ms = 0, m_Rays = 0, m_HitRate = 0, m_VisitsPerRay = 0; int m_Frames = 0; };
                      const refl_report&   getReflReport( void ) const                                          noexcept { return m_ReflReport; }
        // A material's G-buffer variant (its fragment compiled with XLION_GBUFFER_OUTPUT): it is drawn into the screen tail's thin G-buffer
        // (without one, a material shades its tail inline - the sampled path)
        [[nodiscard]] xgpu::device::error* SetMaterialGBuffer( material_id Id, const xgpu::shader& FragmentGBuffer ) noexcept;
        // The sun's GPU page/caster pairs (P1c): this frame's candidates and runs, whether the GPU drew the pages; a ring late the pairs, those
        // dropped and the triangles
                      const pairs_report&  getPairsReport( void ) const                                         noexcept { return m_PairsReport; }
        // The pages' setup (tests: a small pool, a larger budget): starts them over. Its pool must fit the clipmap's part of the atlas
                      void                 configureVSM( const shadows::vsm::setup& Setup )                    noexcept { m_VSM.Init(Setup); }
        // Invariant 1 on this frame's table (tests): every entry names a VALID slot holding exactly that page
                      bool                 validateVSM ( void ) const                                           noexcept { std::vector<std::uint32_t> T; m_VSM.BuildTable(T); return m_VSM.Validate(T); }
        // The pages' CPU side, read only (tests: levels, texel size)
                      const shadows::vsm&  getVSM      ( void ) const                                           noexcept { return m_VSM; }

        // The lighting lab (features/lights/xlionrender_light_cells.h): how local lights are found per pixel
        enum class light_mode : std::uint8_t
        { EXACT                 // every light
        , CELLS                 // the world cell's list of the lights that reach it (the same image as EXACT)
        , SAMPLED               // the cell's head_v most important lights exactly + one tail light, sampled (unbiased, noisy)
        , SCREEN                // the head exactly + the tail sampled at half size in screen space and denoised (documentation/lighting_p3_design.md
                                // "The decision"); the main view only (the others: CELLS); a pixel the screen tail does not cover: the list exactly
        , AUTO                  // CELLS below m_LightScreenThreshold lights, SCREEN from it
        };
        light_mode                                 m_LightMode     = light_mode::AUTO;      // CELLS (exact) or, past the threshold, SCREEN (research doc: lab; lighting_p3_design.md)
        int                                        m_LightScreenThreshold = 512;            // AUTO: the lights from which SCREEN is used (the night-market rig: CELLS costs less up to ~300)
        float                                      m_TailTraceLength = 1.0f;                // SCREEN: the tail's unshadowed lights' screen trace (contact shadows), world units (0: none)
        int                                        m_TailTraceSteps  = 12;
        bool                                       m_bTailTraceBvh   = true;                // SCREEN: those lights' shadow ray also through the GI's BVH (P4; the opaque rigid GI meshes, off screen too)
        int                                        m_SSRSteps        = 48;                  // reflections' screen trace (scene::m_bReflectionScreen): its steps a ray (half size), its length and thickness (world units)
        float                                      m_SSRMaxDist      = 30.0f;
        float                                      m_SSRThickness    = 0.15f;
        float                                      m_ReflSharpMax    = 0.1f;                // reflections: the receivers sharper than this trace at the view's full size (their own G-buffer); 0: all at half size
        int                                        m_ReflTraceBench  = 0;                   // (measurement) SCREEN, the GI on: a reflection ray a thin G-buffer texel through the GI's BVH (shaders/xlionrender_refl_trace_comp.glsl): 1 mirror, 2 roughness 0.5, 3 the materials'; checks shown as the records (with ReflectionDebug): 4 the camera's ray through the BVH against the G-buffer's depth (green agrees, red / blue 10 x the error; hits: within 1%), 5 the G-buffer's normal, 6 its N.V, 7 its F0 luminance (the motion target's w: the hits' texturing selects by it); 0 off. Init: XLION_REFL_BENCH
        // Emissive surfaces as lights (documentation/lighting_gi_interface_notice.md): the GI's emitter view (gi::world::ForEachEmitter) as
        // lights without shadow pages. Off by default: the GI's bounce already holds the emitters' light on the surfaces around them - with
        // both, it counts twice - so this one switch also hands that term to lighting (gi::world::setup::m_bEmitterDirectExternal: the GI keeps
        // the emitters' bounces, not their direct light). Or scene::gi_settings::m_bEmitterLights (the Environment's GIEmitterLights)
        bool                                       m_bEmitterLights   = false;
        int                                        m_EmitterLightsMax = 2048;               // the strongest by power (luminance x area); the rest folded into their nearest (getLightStats m_nEmittersMerged)
        float                                      m_EmitterCutoff    = 0.002f;             // an emitter light's reach: where its luminance x A / (d^2 + A) falls under this
        int                                        m_LightHead     = 4;
        int                                        m_LightTailSamples = 1;     // SAMPLED: stratified tail samples per pixel
        int                                        m_LightTailCandidates = 8;  // SAMPLED: > 1, the tail resampled (RIS) from this many candidates, one shaded (0: the plain samples)
        bool                                       m_bLightTailVisibility = true;  // SAMPLED, resampled: a candidate's weight holds its shadow too (one tap)
        float                                      m_LightCellSize = 2.0f;     // level 0 cells, world units
        bool                                       m_bLightCellsGPU = true;    // the cells built on the GPU (shaders/xlionrender_light_cells_comp.glsl); off: the CPU's Build
        // CPU time of the last render's cell build (ms; with the GPU build its CPU part only), and a frame's GPU build (ms, a ring late)
                      double               getLightBuildMs( void ) const                                        noexcept { return m_LightBuildMs; }
                      double               getLightBuildGpuMs( void ) const                                     noexcept { return m_LightBuildGpuMs; }
        // The last render's cell lists as the shader reads them (headers, then entries: light_cells::m_Data's layout); waits for its frame
                      bool                 readLightCells( std::vector<std::uint32_t>& Out ) noexcept;
        // The last render's cell data (uints; 0 when the mode was EXACT or fell back to it) and the mode it used
                      int                  getLightCellUints( void ) const                                      noexcept { return m_LightCellUints; }
                      light_mode           getLightModeUsed( void ) const                                       noexcept { return m_LightModeUsed; }
        // The last render's local lights and what it could not shade as asked (documentation/lighting_strategy.md sec. 3, 9): lights past
        // the buffer (not shaded at all), shadowed lights shaded unshadowed (no map yet: the declared fallback), maps sampled while waiting
        // for a redraw (stale: what P1a removes), cells whose list overflowed (shaded with every light: exact, slower)
        struct light_stats { int m_nLights = 0; int m_nDropped = 0; int m_nShadowed = 0; int m_nShadowFallback = 0; int m_nShadowStale = 0; int m_nOverflowCells = 0; double m_CellRingMB = 0; int m_nEmitters = 0; int m_nEmittersLeftOut = 0; int m_nEmittersMerged = 0; };     // (m_nEmitters: emitter lights among m_nLights)     // (m_CellRingMB: the light cell ring's memory)
                      light_stats          getLightStats( void ) const                                          noexcept { return m_LightStats; }
        // How many clipmap tiles + local light maps the last Render drew (a static scene: 0 after its first frame)
                      int                  getShadowTilesRendered( void ) const                                 noexcept { return m_nShadowTilesRendered; }

        // The scene pass skips draws whose world bounds are outside the camera's frustum (false: draw everything - the reference)
        bool                                       m_bFrustumCull = true;
        // The main view's opaque rigid clusters with a material are culled on the GPU and drawn indirectly (documentation/gpu_cluster_culling.md;
        // false or XLION_NO_GPU_CULL=1: the CPU path). The last Render's candidates (its clusters handed to the cull) and runs (indirect draws)
        bool                                       m_bGPUCull     = true;
        // with occlusion, in two passes (the clusters seen last frame, then the rest against the Hi-Z of their depth; false or
        // XLION_NO_GPU_OCCLUSION=1: the frustum only)
        bool                                       m_bGPUOcclusion = true;
        // its view-size rule: a view of fewer triangles (in the frustum) skips it - it costs more than it saves there
        std::int64_t                               m_GPUOcclusionMinTriangles = 500'000;
        // Freeze: the last cull's lists drawn again, seen from wherever the camera goes - what was culled shows as missing (a debug view;
        // ignored when the scene's runs change)
        bool                                       m_bFreezeCull   = false;
        // and, read back a few frames late, the clusters the GPU kept (drawn shaded - both passes -, in the depth prepass, shaded by pass 2)
        struct cull_stats { int m_nCandidates = 0; int m_nRuns = 0; int m_nDrawn = 0; int m_nPrepassDrawn = 0; int m_nDrawnPass2 = 0; bool m_bFrozen = false; std::int64_t m_nTriangles = 0;
                            bool m_bOcclusionSkipped = false; float m_OccludedShare = 0; double m_CullGpuMs = 0, m_OccludeGpuMs = 0; };     // (adaptive occlusion; the GPU's times)
                      cull_stats           getCullStats( void ) const                                           noexcept { return { int(m_CullCandidates.size()), int(m_CullRuns.size()), m_pCull ? m_pCull->m_Drawn : 0, m_pCull ? m_pCull->m_PrepassDrawn : 0, m_pCull ? m_pCull->m_Drawn2 : 0, m_bFreezeCull && m_CullFrozen.m_bValid, m_pCull ? m_pCull->m_Triangles : 0
                                                                                                                       , m_pCull && m_pCull->isOcclusionSkipped(), m_pCull ? m_pCull->m_OccludedShare : 0.0f, m_pCull ? m_pCull->m_CullGpuMs : 0.0, m_pCull ? m_pCull->m_OccludeGpuMs : 0.0 }; }
        // How many of the scene's draws the last Render drew in the scene pass
                      int                  getDrawsRendered( void ) const                                       noexcept { return m_nDrawsRendered; }
        // How many persistent instance slots this frame wrote (new or changed: transform, constant material, entity parameters)
                      int                  getSlotsWritten( void ) const                                        noexcept { return m_nSlotsWritten; }

        // The editor's view of huge meshes (view_options::m_bBoxesOnly, m_FrameBudgetMs), in Render's scene pass. Boxes only: no cluster mesh is drawn. A frame budget (> 0, ms;
        // the frame is the time from one Render to the next): only the clusters in the view are drawn, the biggest on the screen first, as many as fit the allowance of indices;
        // the allowance shrinks while the frames are slower than the budget and grows back (to no limit) while they are faster. What was not drawn: getNotDrawn, the boxes the
        // editor draws instead (the clusters outside the view have none)
        bool                                       m_bBoxesOnly    = false;
        float                                      m_FrameBudgetMs = 0;
        struct budget_stats
        {
            int             m_nClusters = 0;        // the clusters of the visible cluster draws, in the view
            int             m_nDrawn    = 0;        // of them, the ones drawn
            std::uint64_t   m_Allowance = 0;        // indices (0: no limit)
            float           m_FrameMs   = 0;        // the frame time the allowance was last decided on
        };
                      std::span<const wire_box> getNotDrawn( void ) const                                      noexcept { return m_NotDrawn; }
                      const budget_stats&  getBudgetStats( void ) const                                         noexcept { return m_BudgetStats; }
                      int                  getSceneBinds( void ) const                                          noexcept { return m_nSceneBinds; }    // the last Render's scene-pass state binds
        // Whether the last scene pass left cluster iCluster (an index into its cluster_geom::m_Clusters) of scene draw iDraw out (its box is drawn, or it is outside the view)
                      bool                 isClusterSkipped( std::size_t iDraw, std::uint32_t iCluster ) const  noexcept
                      {
                          if( iDraw >= m_ClusterBase.size() || m_ClusterBase[iDraw] == ~0u ) return false;
                          const std::size_t i = std::size_t(m_ClusterBase[iDraw]) + iCluster;
                          return i < m_ClusterSkip.size() && m_ClusterSkip[i] != 0;
                      }
        // Whether it was left out as its box (Boxes only, or over the budget; not for being outside the view): the shadows leave it out too (what is left out of the
        // view still casts into it)
                      bool                 isClusterBoxed( std::size_t iDraw, std::uint32_t iCluster ) const    noexcept
                      {
                          if( iDraw >= m_ClusterBase.size() || m_ClusterBase[iDraw] == ~0u ) return false;
                          const std::size_t i = std::size_t(m_ClusterBase[iDraw]) + iCluster;
                          return i < m_ClusterSkip.size() && m_ClusterSkip[i] == cluster_boxed_v;
                      }

        // true when a newer frame's timings arrived since the last call
                      bool                 getGPUTimes  ( gpu_times& Times )                                    noexcept;
        // the latest timings any getGPUTimes read (several readers share them: each set reaches only the first)
                      const gpu_times&     getLastGPUTimes( void ) const                                        noexcept { return m_LastGPUTimes; }
                      const cpu_times&     getCPUTimes  ( void ) const                                          noexcept { return m_CPUTimes; }
        // How many material pipelines exist (materials of one fragment shader and state share them; tests)
                      int                  getPipelineCount( void ) const                                       noexcept { return int(m_SharedPipelines.size()); }

    private:

        struct mesh_slot
        {
            xgpu::buffer            m_VertexBuffer;
            xgpu::buffer            m_IndexBuffer;
            int                     m_nIndices      = 0;
            xmath::fbbox            m_BBox;                             // local bounds of a CreateMesh mesh
            cluster_geom            m_Geom;                             // used when m_bCluster
            std::unique_ptr<gi::surfel_mesh> m_pGI;                     // SetMeshGI: the surfels (CPU) ...
            xgpu::buffer            m_GIBuffer;                         // ... and the vertex shader's copy (xlionrender_static_geom_body.glsl GIMesh)
            bool                    m_bTriColors = false;               // m_GIBuffer has its triangles' captured colors (the SURFACE_COLOR view)
            std::uint64_t           m_GIStamp    = 0;                   // which SetMeshGI made m_pGI (a new one can land at the old one's address)
            // SetMeshGISections: the GI in sections (m_pGI empty); each section's surfels, display buffer, owned count and ghosts' owners
            struct gi_section_slot
            {
                std::unique_ptr<gi::surfel_mesh>    m_pGI;
                xgpu::buffer                        m_Buffer;
                std::uint32_t                       m_nOwned = 0;
                std::vector<gi::chunk_ghost>        m_Ghosts;
            };
            std::vector<gi_section_slot>    m_GISections;
            std::vector<std::uint16_t>      m_GIClusterSection;         // per cluster its section
            std::vector<int>                m_GIClusterVertexDelta;     // per cluster: its section's vertex is its own plus this (push_static_geom)
            std::uint32_t           m_Generation    = 0;
            bool                    m_bAlive        = false;
            bool                    m_bCluster      = false;
            xgpu::buffer            m_OutlineVB, m_OutlineIB;           // SetOutlineProxy: the selection's outline mesh (position, normal) and its triangles
            int                     m_nOutlineIndices = 0;
            xmath::fvec3            m_OutlineCenter{ 0, 0, 0 }, m_OutlineHalf{ 0, 0, 0 };
        };

        static constexpr std::size_t capture_path_v = std::size_t(geom_kind::COUNT);      // the GI material capture's pipelines (getCaptureInstance)
        static constexpr std::size_t shadow_path_v  = capture_path_v + 1;                 // a masked material's shadow casters, per vertex path (getMaterialInstance bShadow)
        static constexpr std::size_t records_path_v = shadow_path_v + capture_path_v;     // the rigid path reading its draws' records (getRecordsInstance: GPU culling)
        static constexpr std::size_t gbuffer_path_v = records_path_v + 1;                 // the screen tail's G-buffer, per vertex path (getGBufferInstance)

        struct material_slot
        {
            // Copies of the handles (they share the GPU objects): its pipelines are built on first use, after whoever handed them over may
            // have let theirs go (a material instance's texture compiled again - pointers into a resource there were freed memory)
            xgpu::shader                                            m_Fragment;
            std::vector<xgpu::texture>                              m_Textures;     // slot 0 (shadow map) may be empty: white
            std::vector<xgpu::texture*> TexturePointers() noexcept { std::vector<xgpu::texture*> P; for (auto& T : m_Textures) P.push_back(T.m_Private ? &T : nullptr); return P; }
            // Per vertex path (geom_kind), the GI capture, the shadow casters per vertex path, the records path: built on first use. Its own
            // textures; the pipeline is shared with every material of the same fragment and state (m_SharedPipelines)
            std::array<xgpu::pipeline_instance, gbuffer_path_v + capture_path_v> m_Instance;
            xgpu::shader            m_GBufferFragment;              // SetMaterialGBuffer: its G-buffer variant (none: not in the thin G-buffer)
            std::uint32_t           m_Generation    = 0;
            bool                    m_bAlive        = false;
            material_state          m_State;                        // CreateMaterial: its pipelines' settings
            std::uint32_t           m_ParamBase     = 0;            // its block in MaterialParams: the first vec4 (push constant MaterialBase), how many
            std::uint32_t           m_ParamCount    = 0;
            OffsetAllocator::Allocation m_ParamRange;               // m_ParamAlloc's: freed by DestroyMaterial
            // A texture-table material (CreateMaterial bTextureTable): its textures' table (set 3), their ids from vec4 m_IDsAt of its block
            bool                    m_bTable        = false;
            xgpu::texture_table     m_Table;
            std::uint32_t           m_IDsAt         = 0;
        };

        const mesh_slot*            findMesh        ( mesh_id Id ) const                                        noexcept;
        material_slot*              findMaterial    ( material_id Id )                                          noexcept;
        void                        renderShadows   ( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye, float ProjScale ) noexcept;
        void                        updateInstances ( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye ) noexcept;
        void                        writeLocalLights( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye, details::ubo_lighting& Light ) noexcept;
        void                        updateEnvironment( xgpu::window& Window, const scene& Scene )               noexcept;
        void                        renderScenePass ( xgpu::window& Window, const scene& Scene, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye, bool bMain ) noexcept;
        // The selection's outline (Scene.m_SelectedUser's draw): its triangles into the view's mask, then the outline over m_Output (before the overlays)
        void                        renderOutline   ( xgpu::window& Window, const scene& Scene, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye ) noexcept;
        xgpu::device::error*        createOutlineTargets( render_view& View ) noexcept;
        void                        destroyOutlineTargets( render_view& View ) noexcept;
        xgpu::device::error*        createGIScreenTargets( render_view& View ) noexcept;
        void                        destroyGIScreenTargets( render_view& View ) noexcept;
        xgpu::device::error*        createCullTargets   ( render_view& View ) noexcept;      // render_view::m_ScenePassLoad, m_HiZ (the first time)
        void                        updateProbes    ( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye ) noexcept;
        void                        renderPlanar    ( xgpu::window& Window, const scene& Scene, const render_view& Main, const xgpu::tools::view& Camera ) noexcept;
        void                        prefilterInto   ( xgpu::window& Window, xgpu::renderpass& Pass, float OriginX, float OriginY ) noexcept;
        std::vector<xgpu::pipeline::sampler>                    LitSamplers ( void ) const                     noexcept;
        std::vector<xgpu::pipeline_instance::sampler_binding>   LitBindings ( std::span<xgpu::texture* const> Textures, bool bShadowAtlas = true ) noexcept;     // false: white in slot 0 (a pass that renders into the atlas)
        xgpu::pipeline_instance*    getMaterialInstance( material_slot& Material, geom_kind Kind, bool bShadow = false ) noexcept;     // bShadow: its shadow caster (a masked material's cutout)
        static void                 ApplyState         ( xgpu::pipeline::setup& Setup, const material_state& S ) noexcept;
        xgpu::pipeline_instance*    getCaptureInstance ( material_slot& Material )                              noexcept;
        xgpu::pipeline_instance*    getRecordsInstance ( material_slot& Material )                              noexcept;     // RIGID, the draw's push constants from its record (binding 12)
        xgpu::pipeline_instance*    getGBufferInstance ( material_slot& Material, geom_kind Kind )              noexcept;     // its G-buffer variant (nullptr: none)
        // The screen tail (light_mode::SCREEN, the main view): the thin G-buffer, the tail pass and the denoiser, before the scene pass
        xgpu::device::error*        createTailTargets  ( render_view& View )                                    noexcept;
        xgpu::device::error*        createSharpTargets ( render_view& View )                                    noexcept;     // (the records grow by the full-size region)
        void                        destroyTailTargets ( render_view& View )                                    noexcept;
        void                        renderTail         ( xgpu::window& Window, const scene& Scene, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye, details::ubo_lighting& Light, bool bPlan ) noexcept;
        xgpu::pipeline              m_ForwardGBPipeline, m_StaticGeomGBPipeline, m_TailPipeline;     // the built-in materials' G-buffer variants; the tail pass
        xgpu::pipeline_instance     m_ForwardGBInstance, m_StaticGeomGBInstance;
        xgpu::shader                m_TailDenoiseShader;
        xgpu::compute               m_TailDenoise;
        xgpu::buffer                m_TailFrames;                   // the denoiser's frame (host visible, a slot a ring frame)
        xgpu::timestamps            m_TailTimes;                    // the screen tail's passes: start, G-buffer, tail, denoiser
        xgpu::shader                m_ReflTraceShader;              // the reflection trace's benchmark (m_ReflTraceBench): its pass, counters (4 per ring slot), timestamps
        xgpu::compute               m_ReflTrace;
        xgpu::shader                m_ReflDenoiseShader;            // the reflections' reconstruction (milestone D: xlionrender_refl_denoise_comp.glsl), its frames (a ring slot each)
        xgpu::compute               m_ReflDenoise;
        xgpu::buffer                m_ReflFrames;
        xgpu::shader                m_ReflSSRShader;                // the screen trace (SSR: xlionrender_refl_ssr_comp.glsl), its timestamps (XLION_GPU_TIMES: SSRGPU)
        xgpu::compute               m_ReflSSR;
        xgpu::timestamps            m_ReflSSRTimes;
        double                      m_ReflSSRSum = 0; int m_ReflSSRSumN = 0;
        // Reflections, the hits' textures (shaders/xlionrender_refl_hits_comp.glsl): the hits binned by their draw's material and shaded by
        // the material's capture variant, a pixel each, into m_HitShadeTex (the albedo's rows, then the emission's); made on first use
        static constexpr std::uint32_t hit_width_v = 1024, hit_cap_v = 1024 * 1024, hit_max_bins_v = 4096;     // (a million hits: Police's mirror from 0.7 m at 3778 x 1600 is 0.85 million)
        xgpu::shader                m_ReflHitsShader;
        xgpu::compute               m_ReflHits;
        xgpu::texture               m_HitShadeTex;
        xgpu::renderpass            m_HitShadePass;
        xgpu::buffer                m_HitVB, m_HitIB, m_HitCapture, m_HitTables, m_HitBins, m_HitCounts, m_HitDraws;
        std::uint32_t               m_HitTableCap   = 0;        // m_HitTables' words a ring slot
        std::vector<std::pair<material_id, std::uint32_t>> m_HitBinKeys;   // per bin: its material and draw slot (this frame's)
        std::vector<std::vector<std::pair<material_id, std::uint32_t>>> m_HitBinKeysAt;      // per ring slot: the bins its counts are of
        xgpu::buffer                m_HitList;                  // the trace's hits to shade: their counts, then three uvec4 a hit (hit_cap_v)
        xgpu::device::error*        createHitTargets( void ) noexcept;
        void                        shadeReflectionHits( xgpu::window& Window, const scene& Scene, render_view& View, const gi::gpu_trace::bvh_view& BvhR, int Slot, std::uint32_t HalfKeys, std::uint32_t FullBase ) noexcept;
        xgpu::buffer                m_ReflStats;
        xgpu::timestamps            m_ReflTimes;
        double                      m_ReflSplit[4] = {}, m_ReflTextured = 0, m_ReflTexturedFull = 0, m_ReflHitSplit[3] = {};     // (the hits' shading: bins, draws, relight)            // (measurement) the sharp trace, the trace, the hits' shading, the reconstruction: ms summed over the report's frames
        refl_report                 m_ReflReport;
        double                      m_ReflSum[4] = {};              // (the report's running sums: ms, rays, hits, visits)
        int                         m_ReflSumN = 0;
        double                      m_ReflPlanarSum = 0, m_ReflPlanarRough = 0, m_ReflIncompleteSum = 0;
        bool                        m_bReflTraceCycle = false;      // (XLION_REFL_BENCH=9: each report the next mode)
        tail_report                 m_TailReport;
        xgpu::buffer*               m_pTailRecords = nullptr;       // binding 14 of the lit pipelines this render: the view's records (SCREEN), else none
        xgpu::buffer*               TailBuffer( void ) noexcept     { return m_pTailRecords ? m_pTailRecords : &m_GIDummy; }
        // The materials' pipelines, shared by (fragment shader, the state the path uses, path): materials differ by their instances (textures)
        using pipeline_key = std::tuple<const void*, std::uint32_t, std::size_t>;
        struct shared_pipeline { xgpu::pipeline m_Pipeline; xgpu::pipeline_instance m_Globals; int m_nUsers = 0; };      // m_Globals: the texture-table materials' set 0 (the renderer's textures only)
        std::map<pipeline_key, shared_pipeline> m_SharedPipelines;
        static pipeline_key         PipelineKey        ( const material_slot& Material, std::size_t k )         noexcept;
        // Material's instance k on the shared pipeline (made by Build when it is new); nullptr when either could not be created
        xgpu::pipeline_instance*    makeInstance       ( material_slot& Material, std::size_t k, bool bShadowAtlas, const std::function<xgpu::device::error*(xgpu::pipeline&)>& Build ) noexcept;
        void                        releaseInstances   ( material_slot& Material )                              noexcept;
        xgpu::device::error*        buildMaterialPipeline( xgpu::pipeline& Pipeline, const material_slot& Material, geom_kind Kind, bool bShadow, bool bRecords = false, bool bGBuffer = false ) noexcept;
        std::uint8_t                TableSlots         ( const material_slot& Material ) const                  noexcept;
        xgpu::shader                m_StaticGeomGIVert, m_SkinGeomVert, m_GICaptureVert, m_StaticGeomGIRecordsVert;     // the material pipelines' vertex shaders
        void                        BindClusterStreams ( xgpu::cmd_buffer& Cmd, const cluster_geom& Geom )      noexcept;   // the core + optional streams (m_StaticGeomVD / m_SkinGeomVD)
        void                        processGICapture( xgpu::window& Window )                                    noexcept;

        // The GI's material capture (processGICapture): a mesh's surfels drawn through their slots' real material shaders into
        // m_GICaptureTex (each a 2 x 2 block: the albedo in the top half, the emission in the bottom), read back a few frames later into
        // the solver (gi::world::SetSurfelMaterial). One job at a time
        struct gi_capture_job
        {
            int                         m_GIId = -1;
            mesh_id                     m_Mesh;
            std::vector<material_id>    m_Materials;
            std::uint32_t               m_Slot = 0;             // the draw's instance slot (the constant material reads it)
            std::vector<int>            m_Order;                // placement -> surfel (sorted by material slot)
            bool                        m_bTriangles = false;   // the mesh's triangles m_First.. (else its surfels)
            std::uint32_t               m_First = 0;
            int                         m_Section = -1;         // GI sections: the section's surfels and triangles (-1: the mesh's m_pGI)
            xgpu::buffer                m_Data;
            std::uint64_t               m_Frame = 0;
            bool                        m_bRecorded = false;
        };
        xgpu::device::error*        buildMeshGI( mesh_slot& M, std::span<const xmath::fvec3> TriColors ) noexcept;
        xgpu::device::error*        createMeshGI( mesh_slot& M, std::span<const std::uint32_t> Data ) noexcept;      // the GPU buffer of PackMeshGI's data     // M.m_GIBuffer from M.m_pGI (+ its triangles' colors)
        static constexpr int        gi_capture_passes_v = 4;       // albedo, emission, F0's color, roughness + metallic (the last two: reflections' secondary hits)
        static constexpr int        gi_capture_w_v = 1024, gi_capture_h_v = 1024, gi_capture_max_v = (gi_capture_w_v / 2) * (gi_capture_h_v / (2 * gi_capture_passes_v));   // 65536 surfels
        std::deque<gi_capture_job>  m_GICaptures;
        int                         m_GICapturesQuiet = 0;      // frames since the captures were last pending (updateGI: XLION_GI_HOLD_CAPTURES)
        xgpu::texture               m_GICaptureTex;
        xgpu::renderpass            m_GICapturePass;
        xgpu::buffer                m_GICaptureVB, m_GICaptureIB;
        xgpu::pipeline              m_GIConstCapturePipeline;   // the constant material's capture (geom standard)
        xgpu::pipeline_instance     m_GIConstCaptureInstance;

        xgpu::device                m_Device;
        xgpu::vertex_descriptor     m_MeshVD;
        xgpu::vertex_descriptor     m_FullscreenVD;
        xgpu::pipeline              m_ForwardPipeline;
        xgpu::pipeline_instance     m_ForwardInstance;
        xgpu::vertex_descriptor     m_StaticGeomVD;                     // the core streams + the optional ones (dynamic strides): BindClusterStreams
        xgpu::pipeline              m_StaticGeomPipeline;
        xgpu::pipeline_instance     m_StaticGeomInstance;
        // The depth prepass (renderScenePass): the rigid clusters' depth, through the same vertex shader (the same positions) and no color,
        // before their shading - which then runs once a pixel, at the depth it wrote (LESS_OR_EQUAL). Its layout is m_StaticGeomPipeline's
        xgpu::pipeline              m_DepthPrepassPipeline;
        xgpu::pipeline_instance     m_DepthPrepassInstance;        // The GI read at half resolution (render_view::m_GIScreenPass): the same vertex shader and layout + the buffer it writes (binding 11),
        // the material's GI read in the fragment (xlionrender_gi_screen_frag.glsl), at the depth the prepass wrote, no color
        xgpu::pipeline              m_GIScreenPipeline;
        xgpu::pipeline_instance     m_GIScreenInstance;
        // Both phases of that pass draw in indirect runs: their vertex shader reads each draw's push constants from the view's record
        // buffer (xlionrender_static_geom_gi_records_vert.glsl, binding 12); the depth phase's pipeline is m_DepthPrepassPipeline's with it
        xgpu::pipeline              m_GIScreenDepthPipeline;
        xgpu::pipeline_instance     m_GIScreenDepthInstance;
        xgpu::buffer                m_StaticGeomUBO;        // the view block of both cluster vertex paths (details::ubo_geom_view)

        // Instance data (features/instances): a device-local buffer (persistent slots, then this frame's transient ones) kept up to
        // date by GPU copies from a CPU ring, one region of the ring per frame in flight
        static constexpr int        max_instances_v         = 65536;
        static constexpr int        max_transient_v         = 16384;
        static constexpr int        ring_frames_v           = 3;        // frames in flight (xGPU keeps resources 2 flips) + the one recorded
        static constexpr float      origin_grid_v           = 1024.0f;  // the instance buffer's origin snaps to this grid near the eye
        xgpu::buffer                m_InstanceBuffer;
        xgpu::buffer                m_InstanceRing;
        instance_slots              m_InstanceSlots;
        int                         m_RingUsed              = 0;        // entries of this frame's ring region already written
        int                         m_ParamRingUsed         = 0;        // vec4s of this frame's MaterialParams ring region already written
        int                         m_nSlotsWritten         = 0;        // persistent slots written this frame (getSlotsWritten)
        std::vector<std::uint32_t>  m_DrawSlot;             // per scene draw: its slot this frame (instance_slots::none_v: not drawn)
        std::vector<std::uint8_t>   m_DrawChanged;          // per scene draw: its persistent slot is new or changed this render
        std::array<int, 8>          m_ChangeWhy{};          // (measurement, VSMEVENT: why casters changed this frame - slot moved, new, mesh, materials, gone, settled, hashed settled, hashed new)

        // Material parameters (plugins/xmaterial.plugin/documentation/material_contract.md, section 0): every material's block in one
        // device-local buffer (set 1 binding 9), a range per material; a new block goes into this frame's ring region (packed: the
        // region is much smaller than the buffer, what does not fit waits for the next frame) and is copied over by updateInstances.
        // EntityParams (binding 10): a 64 B entry per instance slot (draw::m_EntityParams), written with its instance entry: the same
        // ring entry index in its own ring, copied by the same regions.
        static constexpr int        max_material_params_v   = 262144;   // vec4s (4 MiB: 10k override materials of a 20-vec4 block)
        static constexpr int        param_ring_v            = 16384;    // vec4s a frame can write (256 KiB a ring region)
        xgpu::buffer                m_MaterialParams;
        xgpu::buffer                m_MaterialParamsRing;
        xgpu::buffer                m_EntityParams;
        xgpu::buffer                m_EntityParamsRing;
        OffsetAllocator::Allocator  m_ParamAlloc            { max_material_params_v };     // the materials' ranges (vec4s): O(1), freed ranges merge
        std::vector<std::pair<int, std::vector<float>>> m_ParamPending; // blocks not copied yet (first vec4, floats)

        // Surfel GI: the CPU world, draw id -> world instance, per draw this render's (first surfel in the light ring, level)
        static constexpr int        gi_light_region_v       = 1 << 17;  // surfels per frame at the start (a uvec4: the HL2 basis' colors in RGB9E5, the validity)
        int                         m_GIRegion              = gi_light_region_v;     // ... now: grown to the drawn surfels (updateGI; the Bistro in sections: 507k)
        struct gi_track
        {
            xmath::fmat4            m_L2W;
            const gi::surfel_mesh*  m_pMesh     = nullptr;
            std::uint64_t           m_GIStamp   = 0;                    // mesh_slot::m_GIStamp of the surfels it was added with
            std::uint64_t           m_Render    = 0;                    // the last render that drew it
            std::vector<material_id> m_Materials;                       // what its colors are captured from (a change: captured again)
            std::uint32_t           m_DrawSlot  = 0;                    // its draw's instance slot this frame (the reflections' hit shading: the capture's)
            std::vector<std::uint32_t> m_Moments;                       // its moments packed for the GPU (the last update's surfels repacked)
            std::uint32_t           m_MomentsReset = ~0u;               // gi::world::MomentsReset when they were packed (another: all again)
            int                     m_Offset    = -1;                   // its place in every frame slot of the rings (its mean, then its surfels): updateGI
            int                     m_Size      = 0;                    // its surfels
            std::array<bool, ring_frames_v> m_bFull{}, m_bMomFull{};    // a slot gets all of its light, its moments
            std::array<std::vector<int>, ring_frames_v> m_Pending, m_MomPending;   // the surfels a slot has not had since they changed
        };
        std::unique_ptr<gi::world>  m_pGIWorld;
        // The GI's GPU slice (methodology: 2 ms of the frame's 16): its steady cost (the GPU update and Show, measured while no relight is
        // owed, smoothed) and the relight's budget a frame, what the slice leaves (instance::updateGI)
        float                       m_GISliceMs = 2.0f;
        xgpu::timestamps            m_GISliceTS;                    // 0 before Submit, 1 after the probes, 2 before Show, 3 after it, 4 before the probes
        bool                        m_bGISliceTS = false, m_bGISliceTSTried = false;
        double                      m_GISteadyMs = -1;              // (-1: not measured yet)
        double                      m_GISteadyNoProbesMs = -1;      // the same without the probe grid's update (a relight frame skips it)
        int                         m_GIQuietFrames = 0;            // frames since a relight was owed (the timestamps read are a few frames old)
        std::vector<std::uint64_t>  m_GISliceNs;
        // The GI's sun shadow map (renderShadows): the whole level's GI geometry from the sun, ortho over its light-space box, drawn when the
        // sun turns or that box changes; the relight's sun visibility reads it (gi::gpu_trace::SetSunMap). XLION_GI_SUN_RAYS: shadow rays
        static constexpr int        gi_sun_map_v = 2048;
        float                       m_VSMLevelBand = 0.1f;          // the sun's shadow-level blend this frame (scene's sun_light::m_ShadowLevelBlend: the request pass and the shading)
        bool                        m_bGISunRays = std::getenv("XLION_GI_SUN_RAYS") != nullptr;     // (tests, measurement: the shadow rays instead)
        xgpu::texture               m_GISunMap;
        xgpu::renderpass            m_GISunPass;
        bool                        m_bGISunMapMade = false, m_bGISunMapTried = false, m_bGISunMapValid = false;
        xmath::fmat4                m_GISunW2C = xmath::fmat4::fromIdentity();
        float                       m_GISunTexel = 0, m_GISunDepthPerMeter = 0;
        std::array<float, 9>        m_GISunKey{};                   // (the sun's direction and the box it was drawn for)
        xgpu::timestamps            m_GISunTimes;                   // (its draw's GPU time: GISUNMAP with XLION_GPU_TIMES)
        std::unique_ptr<gi::gpu_trace> m_pGITrace;              // G3: the GI's trace on the GPU (scene::gi_settings::m_bGPUTrace)
        bool                        m_bGITraceFailed = false;   // its Init failed: the CPU traces
        bool                        m_bGIToCPU = false, m_bGIToCPUAsked = false;     // the GPU turned off: its light read back first (updateGI)
        mutable std::string         m_GIDumpPath;               // DumpGI with the light on the GPU: written when it is read back (updateGI)
        mutable bool                m_bGIDumpAsked = false;
        std::chrono::steady_clock::time_point m_GIStart{};      // the first GI frame (the frame log's clock)
        std::unordered_map<std::uint32_t, int> m_GIIds;
        std::unordered_map<std::uint64_t, int> m_GISecIds;              // GI sections: (draw id << 16 | section) -> world instance
        std::vector<gi_track>       m_GITrack;                          // per world instance
        std::vector<std::pair<std::uint32_t, std::uint32_t>> m_DrawGI;  // per draw: offset, level (0xFFFFFFFF: none)
        std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> m_DrawGISec;     // GI sections: per draw, per section the same
        xgpu::buffer                m_GILightRing;
        xgpu::buffer                m_GIMomentRing;             // per surfel of m_GILightRing: gi::moment_texels_v uints (half mean, half mean^2) - ponytail: 48 MB host-visible, size it to the scenes if it matters
        // The rings the draws do not read now: while the GPU traces (its Show writes every surfel, every frame) the rings are device-local -
        // host-visible memory written by the GPU and read by every pixel's GI read cost the Bistro 1.9 ms of Show alone - and these the
        // host-visible pair the CPU world writes into (swapped in when the CPU takes over). Made at the first switch, dropped when the region grows
        xgpu::buffer                m_GILightOther, m_GIMomentOther;
        int                         m_GIOtherRegion = 0;
        bool                        m_bGIRingsLocal = false;
        xgpu::buffer                m_GIDummy;                          // bound as GIMesh for meshes without surfels
        xgpu::buffer                m_WhiteColorStream;                 // one RGBA8 white: the COLOR0 stream of a cluster geom without colors (stride 0)
        xgpu::buffer                m_NoUV1Stream;                      // one float2 marker (mb_input_definition_full.vert: the shaders read UV0): the UV1 stream of a cluster geom without one (stride 0)
        std::uint64_t               m_GIRender = 0;
        double                      m_GIOverheadMs = 0;         // the GI's main-thread work in updateGI outside gi::world::Update (the upload), smoothed: its allowance's pre-charge
        struct gi_frame { std::uint64_t m_Frame; gi::world::stats m_Stats; double m_SceneGpuMs = -1; };     // (SceneGpuMs: the scene pass' GPU time, getGPUTimes: -1 not known)
        static constexpr std::size_t gi_frame_log_v = 1200;     // about 20 s at 60 FPS
        std::vector<gi_frame>       m_GIFrames;                 // DumpGIFrames
        double                      m_GILoadMsAcc = 0;
        int                         m_GIAllocTop = 0;                   // the places in a slot of the rings (gi_track::m_Offset): the top, the free blocks
        std::vector<std::pair<int, int>> m_GIFree;
        gi::world::stats            m_GIStats;
        void                        updateGI        ( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye, float ProjScale ) noexcept;

        // Shadow casters by persistent slot: what they were the last time they were seen, so an unchanged caster costs a compare,
        // not a hash, and a caster that stops being drawn is found from the list of the last render's slots
        struct slot_caster
        {
            xmath::fbbox            m_World;
            mesh_id                 m_Mesh;
            std::uint64_t           m_MaterialsHash = 0;
            std::uint64_t           m_BoxedHash     = 0;        // its clusters left out as boxes (renderShadows' BoxedHash): the shadow changes with them
            std::uint64_t           m_Seen          = 0;
            bool                    m_bValid        = false;
            std::uint64_t           m_ChangedPass   = 0;        // the shadow pass it last changed in (a mover until static_frames_v later)
        };
        std::vector<slot_caster>    m_SlotCasters;
        std::vector<std::uint32_t>  m_CasterSlots, m_CasterSlotsPrev;
        std::uint64_t               m_ShadowPassCount       = 0;
        std::vector<xmath::fbbox>   m_DrawWorld;            // per scene draw: its world bounds this frame (computed with the shadow casters)
        struct visible_draw { const mesh_slot* m_pMesh; std::uint32_t m_Material; std::uint32_t m_iDraw; };
        std::vector<visible_draw>   m_Visible;              // this view's visible draws, sorted by state
        // The half-resolution GI pass's draws (renderScenePass): a record and a command a cluster, the runs that share a bind state
        static constexpr int        screen_header_v         = 8;        // the half-resolution GI buffers' header entries (xlionrender_gi_screen_frag.glsl)
        static constexpr int        screen_draws_v          = 1 << 16;  // ponytail: a frame's cap (both phases); past it the rest of the pass is not drawn
        struct screen_run { int m_Phase; const cluster_geom* m_pGeom; const xgpu::buffer* m_pGI; int m_First, m_Count; };
        std::vector<details::push_static_geom>      m_ScreenRecords;
        std::vector<details::draw_indexed_indirect> m_ScreenCommands;
        std::vector<screen_run>                     m_ScreenRuns;
        // GPU culling (features/cull): the main view's opaque rigid clusters with a material, culled into runs of indirect draws - a run a bind
        // state (records pipeline, material, geometry, GI buffer), a prepass run a geometry (m_pInstance null). XLION_NO_GPU_CULL=1: the CPU path
        struct cull_run { xgpu::pipeline_instance* m_pInstance; material_slot* m_pMat; const cluster_geom* m_pGeom; xgpu::buffer* m_pGI; std::uint32_t m_First = 0, m_Max = 0; };
        std::unique_ptr<gpu_cull>                   m_pCull;                // null: not supported (drawIndirectCount)
        std::vector<gpu_cull::candidate>            m_CullCandidates;
        std::vector<xmath::fmat4>                   m_CullMatrices;
        std::vector<cull_run>                       m_CullRuns;
        std::vector<std::uint8_t>                   m_CullPrepass;          // per run: a prepass run
        struct cull_frozen { bool m_bValid = false; std::size_t m_nRuns = 0; std::uint32_t m_Second = 0; bool m_bPass2 = false; } m_CullFrozen;   // the last cull (m_bFreezeCull)
        xmath::fvec3                                m_CullPrevEye{ 0, 0, 0 }, m_CullPrevFwd{ 0, 0, 0 };     // the last cull's camera (a cut: gpu_cull::WantOcclusion)
        std::map<std::tuple<const void*, const void*, const void*, const void*>, std::uint32_t> m_CullRunOf;
        std::vector<std::uint32_t>                  m_CullSubBase;          // per scene draw: its submeshes' first flag in m_CullSub (~0: none)
        std::vector<std::uint8_t>                   m_CullSub;              // 1: the submesh is culled on the GPU (the CPU loops leave it)
        int                         m_nDrawsRendered        = 0;
        // The editor's view of huge meshes (m_bBoxesOnly, m_FrameBudgetMs): planClusters, updateBudget
        void                        planClusters    ( const scene& Scene, const xmath::fvec3& Eye, const frustum& Frustum ) noexcept;
        void                        updateBudget    ( void ) noexcept;
        struct cluster_pick { float m_Priority; std::uint32_t m_iSkip; std::uint32_t m_nIndices; std::uint32_t m_iDraw; std::uint32_t m_iCluster; const cluster_geom* m_pGeom; };
        std::vector<cluster_pick>   m_ClusterPicks;
        std::vector<wire_box>       m_NotDrawn;             // the last scene pass's boxes of what it did not draw
        static constexpr std::uint8_t cluster_culled_v = 1, cluster_boxed_v = 2;
        std::vector<std::uint8_t>   m_ClusterSkip;          // per cluster of each visible cluster draw (from m_ClusterBase): 0 drawn, cluster_culled_v outside the view, cluster_boxed_v its box
        std::vector<std::uint32_t>  m_ClusterBase;          // per scene draw: its first entry in m_ClusterSkip (~0: not planned, every cluster drawn)
        std::vector<std::uint64_t>  m_ClusterPlanKey;       // per scene draw planned: its mesh and entity, so the next frame's renderShadows (before the plan) finds it is the same draw
        double                      m_IndexAllowance        = 0;        // indices a frame may draw (0: no limit)
        std::uint64_t               m_IndicesVisible        = 0;        // the indices of the clusters in the view, last scene pass
        std::chrono::steady_clock::time_point m_LastRender{};
        double                      m_BudgetWindowMs        = 0;        // the frames since the allowance was last decided on
        int                         m_BudgetWindowFrames    = 0;
        budget_stats                m_BudgetStats;
        int                         m_nSceneBinds           = 0;
        xmath::fvec3                m_Origin                {};
        bool                        m_bOriginSet            = false;
        std::uint64_t               m_InstanceFrame         = ~0ull;
        int                         m_TransientCount        = 0;
        xgpu::vertex_descriptor     m_SkinGeomVD;
        xgpu::pipeline              m_ToneMapPipeline;
        xgpu::pipeline              m_BlitPipeline;
        xgpu::texture               m_WhiteTexture;
        xgpu::shader                m_DefaultMaterialFrag, m_DefaultMaterialGBFrag;     // (CreateDefaultMaterial: its HDR fragment and its G-buffer variant)

        // Shadows (features/shadows): the atlas (16 bits: the clipmap's 8x8 tiles on the left, the local maps' square on the right),
        // its passes (a clear for its first use, then load), the tile clear, the casters' depth pipelines per vertex path (float
        // mesh, rigid, skin), the scratch the local maps' casters render into and the remap from it into the atlas
        shadows::setup                          m_ShadowSetup;
        xgpu::texture                           m_ShadowAtlas;
        xgpu::renderpass                        m_ShadowClearPass, m_ShadowPass;
        xgpu::pipeline                          m_ShadowClearPipeline;
        xgpu::pipeline_instance                 m_ShadowClearInstance;
        std::array<xgpu::pipeline, 3>           m_ShadowPipeline;
        std::array<xgpu::pipeline_instance, 3>  m_ShadowInstance;
        xgpu::pipeline                          m_ShadowPairsPipeline;      // RIGID clusters through the pairs' records (xlionrender_shadow_pairs_vert.glsl)
        xgpu::pipeline_instance                 m_ShadowPairsInstance;
        std::unique_ptr<shadow_pairs>           m_pShadowPairs;             // null: not supported (drawIndirectCount)
        pairs_report                            m_PairsReport;
        xgpu::buffer                            m_ShadowUBO;
        // The sun's virtual pages: the CPU side, the request pass (its frame data a ring slot each, its bits and counters zeroed and copied
        // back a ring late), each ring slot's windows (to decode its requests), this frame's table in the cell ring (written once a frame)
        shadows::vsm                            m_VSM;
        xgpu::shader                            m_VSMRequestShader;
        xgpu::compute                           m_VSMRequest;
        xgpu::buffer                            m_VSMFrames, m_VSMReq, m_VSMReqHost, m_VSMZeros;
        xgpu::timestamps                        m_VSMTimes;                 // the request pass's GPU time
        xgpu::timestamps                        m_VSMRenderTimes;           // the sun's pages' rendering GPU time
        std::array<bool, ring_frames_v>         m_VSMRenderTimed{};         // whether a ring slot's frame rendered sun pages (timed)
        std::array<shadows::vsm::snapshot, ring_frames_v> m_VSMSnap{};
        std::vector<shadows::vsm::key>          m_VSMRequests;
        std::vector<std::uint32_t>              m_VSMGot;                   // (the request readback's copy, kept: a frame's room)
        std::vector<std::uint32_t>              m_VSMTable;
        std::uint64_t                           m_VSMTableFrame = ~0ull;
        int                                     m_VSMTableBase  = 0;
        vsm_report                              m_VSMReport;
        int  vsmBitUints( void ) const noexcept { return (m_VSM.getSetup().TableEntries() + 31) / 32; }
        int  vsmReqUints( void ) const noexcept { return vsmBitUints() + 32 + 2 * m_VSM.getSetup().TableEntries(); }      // bits, counters, depth ranges
        // The sun's part of the shadow atlas (from its left): the pages' pool, 8192 x 4096;
        // the local lights' part is to its right
        // renderShadows' light-space boxes of the casters, kept between frames (with the world boxes and the light basis they were of)
        std::vector<shadows::lbox>              m_CasterLight;
        std::vector<xmath::fbbox>               m_CasterLightWorld;
        std::array<xmath::fvec3, 3>             m_CasterLightBasis{};
        std::vector<shadow_pairs::candidate>    m_PairCandidates;           // (the pairs' candidates, kept: their room)
        int                                     m_VSMSoftLevels = 0;        // the coarser levels the request pass asks for too (contact-hardening)
        static constexpr std::uint64_t          static_frames_v = 4;        // a caster unchanged this many shadow passes is static (in the sun's pages), else a mover
        int  sunAtlasWidth( void ) const noexcept { return 2 * shadows::atlas_tiles_v * m_ShadowSetup.m_ClipmapTileSize; }
        void vsmRequestPass( xgpu::window& Window, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye ) noexcept;
        int  vsmTableBase( void ) noexcept;
        // The local lights' shadow pages (P1d, features/shadows/xlionrender_shadow_local_pages.h): their request pass (a ring slot each, with
        // the light slots' generations of then) and this frame's table in the cell ring
        shadows::local_pages                    m_LocalPages;
        std::vector<shadows::local_pages::key>  m_LocalRequests;
        std::vector<std::uint32_t>              m_LocalGot, m_LocalTable;
        std::uint64_t                           m_LocalTableFrame = ~0ull;
        int                                     m_LocalTableBase  = 0;
        xgpu::shader                            m_LocalRequestShader;
        xgpu::compute                           m_LocalRequest;
        xgpu::buffer                            m_LocalFrames, m_LocalReq, m_LocalReqHost;
        std::array<std::array<std::uint32_t, max_local_lights_v>, ring_frames_v> m_LocalReqGen{};
        std::array<bool, ring_frames_v>         m_LocalReqValid{};
        local_report                            m_LocalReport;
        int  localBitUints( void ) const noexcept { return (shadows::local_pages::max_faces_v * shadows::local_pages::face_entries_v + 31) / 32; }
        int  localTableBase( void ) noexcept;
        void localRequestPass( xgpu::window& Window, const scene& Scene, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye ) noexcept;
        struct prev_caster { xmath::fbbox m_World; std::uint64_t m_First; };       // (m_First: the shadow pass the key first appeared in)
        std::unordered_map<std::uint64_t, prev_caster> m_PrevCasters;   // last frame's casters (key -> world bounds), to find what changed
        int                                     m_nShadowTilesRendered = 0;
        double                                  m_SplitShadowMs = 0, m_SplitProbesMs = 0;     // (measurement: the shadow passes alone, the probes + planar - XLION_FRAME_CSV)
        struct { int m_CpuDraws, m_CpuCasters, m_IndirectDraws, m_IndirectMax, m_PageClears; std::int64_t m_CpuTris; } m_ShadowDrawStats{};     // (measurement: the shadow pass's commands a frame - XLION_FRAME_CSV)
        bool                                    m_bShadowAtlasReady    = false;         // stands in for the shadow map until shadows exist
        xgpu::buffer                m_LightUBO;

        // The environment probe (features/environment; shaders/xlionrender_environment*.glsl): the sky into an octahedral source, the
        // source prefiltered into a roughness atlas, the split-sum DFG table. Lit pipelines bind the atlas and the table at samplers
        // 8 and 9 (0-7: the shadow atlas and the material's textures, padded with white).
        static constexpr int        lit_samplers_v          = 11;           // + 10: the probe atlas
        static constexpr int        env_source_size_v       = 256;
        static constexpr int        env_dfg_size_v          = 64;
        static constexpr int        env_source_levels_v     = 6;            // the source pyramid: 256 .. 8 (filtered importance sampling)
        std::array<xgpu::texture, env_source_levels_v>    m_EnvSource;
        std::array<xgpu::renderpass, env_source_levels_v> m_EnvSourcePass;
        std::array<xgpu::pipeline_instance, env_source_levels_v> m_EnvDownsampleInstance;      // [k]: level k from level k-1 (0 unused)
        xgpu::texture               m_EnvAtlas, m_EnvDFG;
        xgpu::renderpass            m_EnvAtlasPass, m_EnvDFGPass;
        xgpu::pipeline              m_EnvSkyPipeline, m_EnvPrefilterPipeline, m_EnvDFGPipeline, m_EnvDownsamplePipeline;
        xgpu::pipeline_instance     m_EnvSkyInstance, m_EnvPrefilterInstance, m_EnvDFGInstance;
        std::uint64_t               m_EnvKey                = 0;            // the sky the atlas holds (0: none yet)
        xgpu::pipeline              m_EnvCubemapPipeline;                   // a cubemap texture (scene::sky::m_pCubemap) into the source
        xgpu::pipeline_instance     m_EnvCubemapInstance;
        xgpu::pipeline              m_MaskPipeline[3];                      // the selection's mask: the casters' vertex paths with the mask's fragment (renderOutline)
        xgpu::pipeline              m_OutlineCompositePipeline;             // the outline over the tone-mapped image
        bool                        m_bOutlineReady = false;
        xgpu::vertex_descriptor     m_OutlineVD;                            // the outline mesh: position and normal
        xgpu::pipeline              m_OutlinePipeline;                      // the editor's outline shader (xlionrender_outline_obb_*), drawn over the scene after it
        xgpu::pipeline_instance     m_OutlineInstance;
        xgpu::pipeline              m_SkyBackgroundPipeline;                // the cubemap as the background (scene::sky::m_bVisible)
        xgpu::pipeline_instance     m_SkyBackgroundInstance;
        const xgpu::texture*        m_pSkyBackground        = nullptr;      // the cubemap m_SkyBackgroundInstance binds
        const xgpu::texture*        m_pEnvCubemap           = nullptr;      // the cubemap m_EnvCubemapInstance binds
        xmath::fvec3                m_EnvMean               { 0, 0, 0 };    // the cubemap's mean color (the GI's sky)
        std::array<xmath::fvec3, 16> m_EnvOct               {};             // and its coarsest prefiltered level, 4 x 4 (gi::lights::m_SkyOct)
        bool                        m_bEnvOct               = false;
        std::uint64_t               m_EnvMeanFrame          = 0;            // the frame it was prefiltered in (0: read)
        bool                        m_bEnvDFGReady          = false;
        bool                        m_bEnvOn                = false;
        float                       m_EnvIntensity          = 0;
        // Reflection probes: an 8x4 atlas of the sky layout's slots (the nearest probe_slots_v probes are resident), 6 capture views
        // (128^2) for the probe being captured, the cube -> octahedral pass into the environment source (then the sky's pyramid and
        // prefilter, into the probe's slot)
        static constexpr int        probe_face_size_v       = 128;
        static constexpr int        probe_slots_v           = 32;
        static constexpr int        probe_slots_x_v         = 8;
        struct probe_state
        {
            std::uint64_t           m_Key       = 0;        // the probe it holds (position, box)
            std::uint64_t           m_LightKey  = 0;        // the lights it saw: the local lights reaching its box, sun, sky, ambient
            int                     m_Slot      = -1;       // its slot in the atlas (-1: not resident)
            bool                    m_bValid    = false;    // captured since it got its slot
            bool                    m_bDirty    = true;
        };
        xgpu::texture               m_ProbeAtlas;
        xgpu::renderpass            m_ProbeAtlasClearPass, m_ProbeAtlasPass;
        bool                        m_bProbeAtlasReady      = false;
        std::array<render_view, 6>  m_CaptureViews;
        render_view                 m_PlanarView;                   // reflections' planar view (renderPlanar): the scene mirrored in scene::m_ReflectionPlane, the screen tail's size
        xgpu::shader                m_PlanarMipsShader;             // its pyramid (shaders/xlionrender_refl_planar_mips_comp.glsl): the pass, the levels, its parameters a ring slot each
        xgpu::compute               m_PlanarMips;
        xgpu::buffer                m_PlanarMipsBuffer, m_PlanarParams;
        int                         m_PlanarLevels = 0;
        static constexpr int        refl_boxes_v = 64;              // the reflection pass's parameters: the boxes of the drawn things the GI's BVH does not hold (a ray crossing one: PREFIX_INCOMPLETE)
        bool                        m_bPlanarOn = false;            // (this frame's: the reflection pass resolves its records from it)
        int                         m_PlanarWhy = 1;                // (diagnostics: 0 on, 1 off, 2 its view, 3 no GI world, 4 no floor under the eye, 5 the oblique plane)
        xmath::fvec4                m_PlanarPlane { 0, 0, 0, 0 };   // (its plane this frame, unit normal: the scene's or the floor found under the eye)
        xgpu::pipeline              m_CubeToOctPipeline;
        xgpu::pipeline_instance     m_CubeToOctInstance;
        std::vector<probe_state>    m_ProbeStates;                      // per scene probe
        std::vector<int>            m_ProbeOrder;                       // this scene pass' captured probes, innermost first
        std::vector<xmath::fbbox>   m_ProbeBoxes;
        int                         m_ProbeCapturing        = -1;       // the probe being captured
        int                         m_ProbeNextFace         = 0;
        int                         m_nProbeCaptures        = 0;
        std::vector<xmath::fbbox>   m_FrameChanged;                     // this frame's changed caster bounds (shadows), for the probes
    public:
        // At most this many probe faces are captured per frame (a probe needs 6; Flax does 1)
        int                         m_ProbeFacesPerFrame    = 1;
        // Completed probe captures so far (tests)
                      int                  getProbeCaptures( void ) const                                       noexcept { return m_nProbeCaptures; }
                      const xgpu::texture& getProbeAtlas( void ) const                                          noexcept { return m_ProbeAtlas; }
        // How many times the environment's atlas was (re)built (tests: a static sky builds it once)
                      int                  getEnvironmentBuilds( void ) const                                   noexcept { return m_nEnvBuilds; }
                      const xmath::fvec3&  getEnvironmentMean  ( void ) const                                   noexcept { return m_EnvMean; }     // a cubemap sky's mean color (the GI's sky)
        // The prefiltered environment atlas (tests read it back)
                      const xgpu::texture& getEnvironmentAtlas( void ) const                                    noexcept { return m_EnvAtlas; }
    private:
        int                         m_nEnvBuilds            = 0;

        // The local lights and their cells: host-visible storage buffers, a region per frame in flight, written per render
        static constexpr int        max_lights_v            = 4096;
        static constexpr int        light_region_v          = 4 * max_lights_v;    // a frame's lights: the main view + probe capture faces
        static constexpr int        max_cell_uints_v        = light_cells::levels_v * light_cells::level_cells_v * light_cells::header_uints_v + 2 * 262144;     // the start
        int                         m_CellStride            = max_cell_uints_v;    // a frame's region of the cell ring (grows: beginCellFrame)
        int                         m_CellNeed              = 0;                   // the largest a frame needed that did not fit
        void                        beginCellFrame( std::uint64_t Frame ) noexcept;
        xgpu::buffer                m_LightRing;
        xgpu::buffer                m_CellRing;
        light_cells                 m_LightCells;
        std::vector<light_cells::light> m_CellLights;
        struct emitter_light { xmath::fvec3 m_Pos, m_Normal, m_Radiance; float m_Area, m_Radius; };
        std::vector<emitter_light>      m_EmitterLights;                    // the GI's emitters as lights (m_bEmitterLights), made again when its generation changes
        std::uint64_t                   m_EmitterLightsGen  = 0;
        int                             m_EmitterLightsMaxUsed = 0;
        int                             m_nEmittersLeftOut  = 0;
        int                             m_nEmittersMerged   = 0;                    // folded into their neighbours (over m_EmitterLightsMax)
        void                            refreshEmitterLights( bool bOn ) noexcept;
        int                         m_LightRingUsed         = 0;
        int                         m_CellRingUsed          = 0;
        std::uint64_t               m_LightFrame            = ~0ull;
        std::uint32_t               m_LightSeed             = 0;
        double                      m_LightBuildMs          = 0;
        double                      m_LightBuildGpuMs       = 0;
        int                         m_LightCellBase         = 0;                   // the last render's cells in the cell ring
        std::vector<std::uint32_t>  m_CellInput;                                   // the GPU build's input (light_cells::GPUInput)
        xgpu::buffer                m_CellScratch;                                 // the GPU build's lists and input (device-local; grows)
        int                         m_CellScratchUints      = 0;
        std::uint64_t               m_CellScratchKey        = 0;                   // what the scratch's lists were built from (light_cells::Key; 0: nothing)
        std::uint32_t               m_CellScratchBound      = 0;
        int                         m_nCellBuilds           = 0;                   // GPU builds run (a render with the same lights and windows reuses the last)
        xgpu::shader                m_LightCellsShader;
        xgpu::compute               m_LightCellsCompute;
        xgpu::timestamps            m_LightCellsTimes;
        bool                        m_bLightCellsTimed      = false;
        int                         m_LightCellUints        = 0;
        light_mode                  m_LightModeUsed         = light_mode::EXACT;
        light_stats                 m_LightStats;
        xgpu::buffer                m_MaterialUBO;
        xgpu::buffer                m_FullscreenVB;
        xgpu::buffer                m_FullscreenIB;
        xgpu::timestamps            m_Timestamps;
        cpu_times                   m_CPUTimes;
        std::vector<mesh_slot>      m_Meshes;
        std::vector<material_slot>  m_Materials;
        std::vector<std::uint64_t>  m_TimestampNs;
        gpu_times                   m_LastGPUTimes          = {};         // (getLastGPUTimes)
        std::uint64_t               m_GPUTimesSerial        = 0;          // counts the timings getGPUTimes read (whoever asked: the ledger records each once)
        bool                        m_bReady        = false;
    };
}

#endif
