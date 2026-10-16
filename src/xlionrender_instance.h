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
#include "features/gi/xlionrender_surfel_gi.h"
#include <memory>
#include "dependencies/xGPU/source/Tools/xgpu_view.h"
#include <span>
#include <vector>
#include <functional>
#include <unordered_map>
#include <deque>

namespace xlionrender
{
    namespace details { struct ubo_lighting; }     // plans/forward/xlionrender_plan_forward.h

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
        [[nodiscard]] xgpu::device::error* SetMeshGI    ( mesh_id Id, gi::surfel_mesh&& Surfels )             noexcept;
        // The last render's GI update (instances, rays, CPU ms)
                      const gi::world::stats& getGIStats( void ) const                                      noexcept { return m_GIStats; }
                      const gi::world*        getGIWorld( void ) const                                      noexcept { return m_pGIWorld.get(); }      // tests, tools
                      bool                    isGICapturingAny( void ) const                                noexcept { return !m_GICaptures.empty(); }
                      int                     DumpGI( const char* pPath ) const                             noexcept { return m_pGIWorld ? m_pGIWorld->Dump(pPath) : -1; }      // gi::world::Dump
                      int                     getGIId( std::uint64_t Instance ) const                       noexcept { const auto It = m_GIIds.find(std::uint32_t(Instance)); return It == m_GIIds.end() ? -1 : It->second; }    // a draw's instance in getGIWorld (-1: none)
                      bool                    isGICapturing( void ) const                                   noexcept { return !m_GICaptures.empty(); }

        // A material: its HDR fragment shader (xmaterial::rt::getShaderHDR()) and its sampler bindings in shader order - slot 0
        // is the shadow map: the renderer binds its shadow atlas there whatever is given; then the textures. Neither is
        // copied or owned: they must outlive the id. The same material draws rigid and skinned geometry: the pipeline of each
        // vertex path is built the first time the material is drawn with it.
        // bAlpha: a transparent material (the material's alpha flag): the GI's rays pass through it (gi::world::SetSeeThrough)
        [[nodiscard]] xgpu::device::error* CreateMaterial( material_id& Id, const xgpu::shader& FragmentHDR, std::span<xgpu::texture* const> Textures, bool bAlpha = false ) noexcept;
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
        // At most this many local light shadow maps re-render per frame (the others stay a frame stale)
        int                                        m_ShadowUpdatesPerFrame = 16;
        // Local light maps draw, per face, only the casters in that face's frustum (false: every caster in the light's sphere)
        bool                                       m_bShadowFaceCull = true;

        // The lighting lab (features/lights/xlionrender_light_cells.h): how local lights are found per pixel
        enum class light_mode : std::uint8_t
        { EXACT                 // every light
        , CELLS                 // the world cell's list of the lights that reach it (the same image as EXACT)
        , SAMPLED               // the cell's head_v most important lights exactly + one tail light, sampled (unbiased, noisy)
        };
        light_mode                                 m_LightMode     = light_mode::CELLS;     // the default: the same image as EXACT, a fraction of the shading (research doc: lab)
        int                                        m_LightHead     = 4;
        int                                        m_LightTailSamples = 1;     // SAMPLED: stratified tail samples per pixel
        float                                      m_LightCellSize = 2.0f;     // level 0 cells, world units
        // CPU time of the last render's cell build (ms)
                      double               getLightBuildMs( void ) const                                        noexcept { return m_LightBuildMs; }
        // The last render's cell data (uints; 0 when the mode was EXACT or fell back to it) and the mode it used
                      int                  getLightCellUints( void ) const                                      noexcept { return m_LightCellUints; }
                      light_mode           getLightModeUsed( void ) const                                       noexcept { return m_LightModeUsed; }
        // How many clipmap tiles + local light maps the last Render drew (a static scene: 0 after its first frame)
                      int                  getShadowTilesRendered( void ) const                                 noexcept { return m_nShadowTilesRendered; }

        // The scene pass skips draws whose world bounds are outside the camera's frustum (false: draw everything - the reference)
        bool                                       m_bFrustumCull = true;
        // How many of the scene's draws the last Render drew in the scene pass
                      int                  getDrawsRendered( void ) const                                       noexcept { return m_nDrawsRendered; }

        // true when a newer frame's timings arrived since the last call
                      bool                 getGPUTimes  ( gpu_times& Times )                                    noexcept;
                      const cpu_times&     getCPUTimes  ( void ) const                                          noexcept { return m_CPUTimes; }

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
            std::uint32_t           m_Generation    = 0;
            bool                    m_bAlive        = false;
            bool                    m_bCluster      = false;
        };

        static constexpr std::size_t capture_path_v = std::size_t(geom_kind::COUNT);      // the GI material capture's pipelines (getCaptureInstance)

        struct material_slot
        {
            // Copies of the handles (they share the GPU objects): its pipelines are built on first use, after whoever handed them over may
            // have let theirs go (a material instance's texture compiled again - pointers into a resource there were freed memory)
            xgpu::shader                                            m_Fragment;
            std::vector<xgpu::texture>                              m_Textures;     // slot 0 (shadow map) may be empty: white
            std::vector<xgpu::texture*> TexturePointers() noexcept { std::vector<xgpu::texture*> P; for (auto& T : m_Textures) P.push_back(T.m_Private ? &T : nullptr); return P; }
            std::array<xgpu::pipeline,          capture_path_v + 1> m_Pipeline;     // per vertex path (geom_kind, the GI capture last), built on first use
            std::array<xgpu::pipeline_instance, capture_path_v + 1> m_Instance;
            std::uint32_t           m_Generation    = 0;
            bool                    m_bAlive        = false;
            bool                    m_bAlpha        = false;        // transparent (CreateMaterial): the GI sees through it
        };

        const mesh_slot*            findMesh        ( mesh_id Id ) const                                        noexcept;
        material_slot*              findMaterial    ( material_id Id )                                          noexcept;
        void                        renderShadows   ( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye, float ProjScale ) noexcept;
        void                        updateInstances ( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye ) noexcept;
        void                        writeLocalLights( const scene& Scene, const xmath::fvec3& Eye, details::ubo_lighting& Light ) noexcept;
        void                        updateEnvironment( xgpu::window& Window, const scene& Scene )               noexcept;
        void                        renderScenePass ( xgpu::window& Window, const scene& Scene, render_view& View, const xmath::fmat4& W2C, const xmath::fvec3& Eye, bool bMain ) noexcept;
        void                        updateProbes    ( xgpu::window& Window, const scene& Scene, const xmath::fvec3& Eye ) noexcept;
        void                        prefilterInto   ( xgpu::window& Window, xgpu::renderpass& Pass, float OriginX, float OriginY ) noexcept;
        std::vector<xgpu::pipeline::sampler>                    LitSamplers ( void ) const                     noexcept;
        std::vector<xgpu::pipeline_instance::sampler_binding>   LitBindings ( std::span<xgpu::texture* const> Textures ) noexcept;
        xgpu::pipeline_instance*    getMaterialInstance( material_slot& Material, geom_kind Kind )              noexcept;
        xgpu::pipeline_instance*    getCaptureInstance ( material_slot& Material )                              noexcept;
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
            xgpu::buffer                m_Data;
            std::uint64_t               m_Frame = 0;
            bool                        m_bRecorded = false;
        };
        xgpu::device::error*        buildMeshGI( mesh_slot& M, std::span<const xmath::fvec3> TriColors ) noexcept;     // M.m_GIBuffer from M.m_pGI (+ its triangles' colors)
        static constexpr int        gi_capture_w_v = 1024, gi_capture_h_v = 512, gi_capture_max_v = (gi_capture_w_v / 2) * (gi_capture_h_v / 4);   // 65536 surfels
        std::deque<gi_capture_job>  m_GICaptures;
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
        xgpu::vertex_descriptor     m_StaticGeomVD;
        xgpu::pipeline              m_StaticGeomPipeline;
        xgpu::pipeline_instance     m_StaticGeomInstance;
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
        std::vector<std::uint32_t>  m_DrawSlot;             // per scene draw: its slot this frame (instance_slots::none_v: not drawn)
        std::vector<std::uint8_t>   m_DrawChanged;          // per scene draw: its persistent slot is new or changed this render

        // Surfel GI: the CPU world, draw id -> world instance, per draw this render's (first surfel in the light ring, level)
        static constexpr int        gi_light_region_v       = 1 << 17;  // surfels per frame (3 vec4: the HL2 basis' colors, the first's w the validity)
        struct gi_track
        {
            xmath::fmat4            m_L2W;
            const gi::surfel_mesh*  m_pMesh     = nullptr;
            std::uint64_t           m_GIStamp   = 0;                    // mesh_slot::m_GIStamp of the surfels it was added with
            std::uint64_t           m_Render    = 0;                    // the last render that drew it
            std::vector<material_id> m_Materials;                       // what its colors are captured from (a change: captured again)
            std::vector<std::uint32_t> m_Moments;                       // its moments packed for the GPU (the last update's surfels repacked)
            std::uint32_t           m_MomentsReset = ~0u;               // gi::world::MomentsReset when they were packed (another: all again)
        };
        std::unique_ptr<gi::world>  m_pGIWorld;
        std::unordered_map<std::uint32_t, int> m_GIIds;
        std::vector<gi_track>       m_GITrack;                          // per world instance
        std::vector<std::pair<std::uint32_t, std::uint32_t>> m_DrawGI;  // per draw: offset, level (0xFFFFFFFF: none)
        xgpu::buffer                m_GILightRing;
        xgpu::buffer                m_GIMomentRing;             // per surfel of m_GILightRing: gi::moment_texels_v uints (half mean, half mean^2) - ponytail: 48 MB host-visible, size it to the scenes if it matters
        xgpu::buffer                m_GIDummy;                          // bound as GIMesh for meshes without surfels
        std::uint64_t               m_GIRender = 0, m_GIFrame = ~0ull;
        int                         m_GIRingUsed = 0;
        std::unordered_map<int, int> m_GIFrameRegion;           // this frame: per GI instance the first surfel of its uploaded light (one upload a frame)
        gi::world::stats            m_GIStats;
        void                        updateGI        ( const scene& Scene, const xmath::fvec3& Eye, float ProjScale ) noexcept;

        // Shadow casters by persistent slot: what they were the last time they were seen, so an unchanged caster costs a compare,
        // not a hash, and a caster that stops being drawn is found from the list of the last render's slots
        struct slot_caster
        {
            xmath::fbbox            m_World;
            mesh_id                 m_Mesh;
            std::uint64_t           m_MaterialsHash = 0;
            std::uint64_t           m_Seen          = 0;
            bool                    m_bValid        = false;
        };
        std::vector<slot_caster>    m_SlotCasters;
        std::vector<std::uint32_t>  m_CasterSlots, m_CasterSlotsPrev;
        std::uint64_t               m_ShadowPassCount       = 0;
        std::vector<xmath::fbbox>   m_DrawWorld;            // per scene draw: its world bounds this frame (computed with the shadow casters)
        struct visible_draw { const mesh_slot* m_pMesh; std::uint32_t m_Material; std::uint32_t m_iDraw; };
        std::vector<visible_draw>   m_Visible;              // this view's visible draws, sorted by state
        int                         m_nDrawsRendered        = 0;
        xmath::fvec3                m_Origin                {};
        bool                        m_bOriginSet            = false;
        std::uint64_t               m_InstanceFrame         = ~0ull;
        int                         m_TransientCount        = 0;
        xgpu::vertex_descriptor     m_SkinGeomVD;
        xgpu::pipeline              m_ToneMapPipeline;
        xgpu::pipeline              m_BlitPipeline;
        xgpu::texture               m_WhiteTexture;
        xgpu::shader                m_DefaultMaterialFrag;

        // Shadows (features/shadows): the atlas (16 bits: the clipmap's 8x8 tiles on the left, the local maps' square on the right),
        // its passes (a clear for its first use, then load), the tile clear, the casters' depth pipelines per vertex path (float
        // mesh, rigid, skin), the scratch the local maps' casters render into and the remap from it into the atlas
        shadows::setup                          m_ShadowSetup;
        xgpu::texture                           m_ShadowAtlas;
        xgpu::texture                           m_ShadowScratch;
        xgpu::renderpass                        m_ScratchPass;
        xgpu::pipeline                          m_RemapPipeline;
        xgpu::pipeline_instance                 m_RemapInstance;
        shadows::buddy                          m_ScratchAlloc;
        xgpu::renderpass                        m_ShadowClearPass, m_ShadowPass;
        xgpu::pipeline                          m_ShadowClearPipeline;
        xgpu::pipeline_instance                 m_ShadowClearInstance;
        std::array<xgpu::pipeline, 3>           m_ShadowPipeline;
        std::array<xgpu::pipeline_instance, 3>  m_ShadowInstance;
        xgpu::buffer                            m_ShadowUBO;
        shadows::clipmap                        m_Clipmap;
        std::vector<shadows::tile_to_render>    m_ShadowTiles;
        shadows::local_shadows                  m_LocalShadows;
        std::vector<shadows::local_map_to_render> m_LocalToRender;
        std::vector<shadows::local_map>         m_LocalMaps;        // per scene local light: its map (m_Size 0: none)
        std::unordered_map<std::uint64_t, xmath::fbbox> m_PrevCasters;  // last frame's casters (key -> world bounds), to find what changed
        int                                     m_nShadowTilesRendered = 0;
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
        static constexpr int        max_cell_uints_v        = light_cells::levels_v * light_cells::level_cells_v * light_cells::header_uints_v + 2 * 262144;
        xgpu::buffer                m_LightRing;
        xgpu::buffer                m_CellRing;
        light_cells                 m_LightCells;
        std::vector<light_cells::light> m_CellLights;
        int                         m_LightRingUsed         = 0;
        int                         m_CellRingUsed          = 0;
        std::uint64_t               m_LightFrame            = ~0ull;
        std::uint32_t               m_LightSeed             = 0;
        double                      m_LightBuildMs          = 0;
        int                         m_LightCellUints        = 0;
        light_mode                  m_LightModeUsed         = light_mode::EXACT;
        xgpu::buffer                m_MaterialUBO;
        xgpu::buffer                m_FullscreenVB;
        xgpu::buffer                m_FullscreenIB;
        xgpu::timestamps            m_Timestamps;
        cpu_times                   m_CPUTimes;
        std::vector<mesh_slot>      m_Meshes;
        std::vector<material_slot>  m_Materials;
        std::vector<std::uint64_t>  m_TimestampNs;
        bool                        m_bReady        = false;
    };
}

#endif
