#ifndef XLIONRENDER_RENDER_VIEW_H
#define XLIONRENDER_RENDER_VIEW_H
#pragma once

// A render_view is one place the renderer draws a snapshot into: its own HDR + depth targets and a tone-mapped
// RGBA8 output (for ImGui::Image, thumbnails, captures). Cameras are xgpu::tools::view, with its conventions (Y-up NDC,
// depth 0 near -> 1 far). Named render_view because xlionrender::view is still the old SCENE/GAME enum.

#include <array>
#include <chrono>
#include <vector>

#include "dependencies/xGPU/source/xGPU.h"
#include "dependencies/xmath/source/xmath.h"
#include "../xlionrender_post_graph.h"      // (and the render graph)
#include "../xlionrender_stage_graph.h"

namespace xlionrender
{
    class instance;

    struct render_view
    {
        struct setup
        {
            int     m_Width     = 256;
            int     m_Height    = 256;
        };

        float                   m_Exposure  = 1.0f;     // linear multiplier before tone mapping; ponytail: EV100 once units are approved
        xmath::fvec4            m_Background{ 0, 0, 0, 0 };     // display-encoded color of m_Output where nothing was drawn (alpha 0: transparent)

        xgpu::texture           m_HDR;                  // R16G16B16A16_SFLOAT, scene-linear
        xgpu::texture           m_HDRPrev;              // last frame's m_HDR (the G-buffer prepass clears m_HDR before anything reads it): the screen trace's and XLION_SSGI's light; made when first needed
        bool                    m_bHDRPrev = false;     // m_HDRPrev holds a frame
        int                     m_HDRPrevPendingW = 0, m_HDRPrevPendingH = 0;     // (the async voxel rays read m_HDRPrev beside the frame's end: its copy waits for the next Render)
        xgpu::texture           m_Depth;                // DEPTH_F32
        xgpu::texture           m_Output;               // R8G8B8A8_UNORM, display-encoded
        xgpu::renderpass        m_ScenePass;
        xgpu::renderpass        m_OutputPass;
        xgpu::renderpass        m_OverlayPass;          // m_Output + m_Depth, loaded (not cleared): instance::Render's Overlay
        xgpu::pipeline_instance m_ToneMapInstance;      // samples m_HDR, m_Bloom[0], m_ExposureTex
        // scene::post_settings (xlionrender_post_comp.glsl). ponytail: made with the view, on or off: half a view's RGBA16F and its smaller levels; on demand if views multiply
        std::array<xgpu::texture, 6> m_Bloom;           // half the view's size, and half again
        xgpu::texture           m_ExposureTex;          // 1 x 1: the adapted exposure
        bool                    m_bPostReady = false;   // (the images written once: they can be sampled)
        bool                    m_bExposureReady = false;   // (the exposure kept across a resize: not zeroed again)
        std::chrono::steady_clock::time_point m_PostLast{};
        // The material tile pass's prototype (XLION_TILE, material_tiles_plan.md): the prepass's attributes, the pass into m_HDR, the pipeline's instance over this view's G-buffer
        xgpu::texture           m_TileAttr0, m_TileAttr1, m_TileColor;   // R32G32_UINT, R32_UINT: the polar codec (xlionrender_tile_attr.glsl); R8G8B8A8_UNORM: the vertex color
        xgpu::renderpass        m_TilePass;
        std::vector<xgpu::pipeline_instance> m_TileInstances;   // a graph each
        // Temporal antialiasing (scene::post_settings::m_bTAA; xlionrender_taa_comp.glsl), made the first time it is on: this frame into one, the other last frame's; tone map from either
        std::array<xgpu::texture, 2> m_TAA;
        std::array<xgpu::pipeline_instance, 2> m_ToneMapTAA;
        int                     m_TAAFrame = 0;
        bool                    m_bTAAHistory = false;
        rg::graph               m_Graph;
        // Frame stages as data (render graph R1, xlionrender_stage_graph.h): the names a stage binds (kernels, resources, frame values), and
        // the stages lowered against them (made the first time). m_bStageLightTiles: the light tiles from their description (off: hand-written)
        stage::registry         m_Stages;
        stage::plan             m_LightTilesPlan;
        bool                    m_bStageLightTiles = true;
        stage::plan             m_ReflReconPlan;        // the reflections' reconstruction (temporal + spatial), the same way
        bool                    m_bStageReflRecon = true;
        stage::plan             m_ReflSSRPlan;          // the reflections' screen trace
        bool                    m_bStageReflSSR = true;
        stage::plan             m_AOPlan;               // the ambient occlusion (raw + filtered)
        bool                    m_bStageAO = true;
        stage::plan             m_GIReadPlan;           // the GI read (classify + a gather a GI buffer: render passes)
        bool                    m_bStageGIRead = true;
        stage::plan             m_VSMRequestPlan, m_LocalRequestPlan;      // the shadow request passes (the sun's pages, the local lights')
        bool                    m_bStageVSMRequest = true, m_bStageLocalRequest = true;
        stage::plan             m_TilePlan;             // the lighting tile pass (a draw a material graph)
        bool                    m_bStageTile = true;
        stage::plan             m_ShadowPagesPlan;      // the shadow pages (two recorded passes: the sun's, the local lights')
        bool                    m_bStageShadowPages = true;
        stage::plan             m_GIUpdatePlan;         // the GI's update (a recorded step)
        bool                    m_bStageGIUpdate = true;
        stage::plan             m_VoxRaysPlan;          // the GI's voxel rays (a recorded step; on the second queue: the code's)
        bool                    m_bStageVoxRays = true;
        stage::plan             m_SunMaskPlan;          // the sun's shadow mask (a draw over the prepass, then its copy - a recorded step)
        bool                    m_bStageSunMask = true;
        // XLION_STAGE_AB (a stage's proof on real materials: the same frame's inputs through the hand-written pass and the stage): the HDR
        // saved, a copy, the two results in rings read back frames later, the count
        xgpu::texture           m_ABSave;
        xgpu::buffer            m_ABScratch;
        std::array<xgpu::buffer, 2> m_ABBuffer;
        int                     m_ABFrames = 0, m_ABChecked = 0;
        std::uint64_t           m_ABDiffWords = 0;
        // the stages' descriptions in place of the built-in ones, by stage name (light_tiles, refl_ssr, refl_recon, ao, gi_read, vsm_request, local_request, tile, shadow_pages, gi_update, gi_voxel_rays, sun_mask; the RenderGraph
        // resource's, xlionrender_stage_graph_text.h): one that does not lower leaves its stage hand-written (and says why once)
        std::vector<std::pair<std::string, stage::desc>> m_StageDescs;
        std::vector<std::pair<std::string, std::uint32_t>>& StageParams( void ) noexcept { return m_Stages.m_Overrides; }      // (the RenderGraph's StageParams: frame values that win)
        std::vector<std::string> m_StagesOff;       // the stages bypassed (stage::bypassable_v: the frame takes its way without them)
        bool StageOn( std::string_view Name ) const noexcept { return std::find(m_StagesOff.begin(), m_StagesOff.end(), Name) == m_StagesOff.end(); }
        void SetStageDesc( const std::string& Name, stage::desc D ) noexcept      // (no passes: the built-in again)
        {
            std::erase_if(m_StageDescs, [&]( const auto& P ) { return P.first == Name; });
            if( !D.m_Passes.empty() ) m_StageDescs.emplace_back(Name, std::move(D));
            m_LightTilesPlan = {}; m_ReflSSRPlan = {}; m_ReflReconPlan = {}; m_AOPlan = {}; m_GIReadPlan = {}; m_VSMRequestPlan = {}; m_LocalRequestPlan = {}; m_TilePlan = {}; m_ShadowPagesPlan = {}; m_GIUpdatePlan = {}; m_VoxRaysPlan = {}; m_SunMaskPlan = {};
        }
        void SetStageDescs( std::vector<std::pair<std::string, stage::desc>> All ) noexcept      // (all of them at once: the RenderGraph bundle's; none: the built-in ones)
        {
            if( All == m_StageDescs ) return;
            m_StageDescs = std::move(All);
            m_LightTilesPlan = {}; m_ReflSSRPlan = {}; m_ReflReconPlan = {}; m_AOPlan = {}; m_GIReadPlan = {}; m_VSMRequestPlan = {}; m_LocalRequestPlan = {}; m_TilePlan = {}; m_ShadowPagesPlan = {}; m_GIUpdatePlan = {}; m_VoxRaysPlan = {}; m_SunMaskPlan = {};
        }
        const stage::desc& StageDesc( std::string_view Name, const stage::desc& BuiltIn ) const noexcept
        {
            for( const auto& [N, D] : m_StageDescs ) if( N == Name ) return D;
            return BuiltIn;
        }
        post::desc              m_PostDesc = post::desc::Default();     // the post chain (xlionrender_post_graph.h; as text: xlionrender_post_graph_text.h) - set it to change the chain
        post::plan              m_PostPlan;             // m_PostDesc lowered for this view's size: made again when either changes
        post::desc              m_PostPlanDesc;
        std::vector<float>      m_PostSnapshot, m_PostNoted;
        // Custom post nodes (render graph stage I): their SPIR-V (set by the owner: the RenderGraph bundle's shader links), and what is made from it
        std::array<std::vector<std::int32_t>, 4>   m_PostCustomSpirv;
        std::array<std::size_t, 4>                 m_PostCustomMade{};      // the SPIR-V's hash the pipeline was made from (0: none)
        std::array<std::size_t, 4>                 m_PostCustomTried{};     // the last SPIR-V tried (a bad one is not tried again every frame)
        std::array<xgpu::shader, 4>                m_PostCustomShader;
        std::array<xgpu::compute, 4>               m_PostCustomCompute;
        std::array<xgpu::texture, 4>               m_PostCustom;            // each node's output (the chain image after it)
        std::array<xgpu::pipeline_instance, 4>     m_ToneMapCustom;         // the tone map over it
        std::vector<std::uint8_t> m_PostRan;            // the last frame: which of the plan's passes ran (its settings skip some) - the editor's view
        std::string             m_PostError;            // the description's error when it did not lower (the chain then tone maps only)    // the frame's parameter values (post::SnapshotParams), and the clamps already reported
        int                     m_PostPlanW = 0, m_PostPlanH = 0;
        float                   m_PostFixedDt = -1.0f;
        const render_view*      m_pPostInputs = nullptr;    // the harness: the post chain starts from this view's lit image (copied in; same size), so two views' chains see the same input  // >= 0: the auto exposure's step in seconds instead of the wall clock (the harness: two views, the same adaptation)                // the render graph's declarations (R0b: the post chain's scopes; xlionrender_render_graph.h)
        xmath::fmat4            m_TAAPrevW2C{};         // without the jitter
        xmath::fvec3            m_TAAPrevEye{};
        // (C22: this frame's jitter in clip space - W2C = T(m_JitterNdc) * the un-jittered one; the screen tail's last camera without it, for the
        // histories accumulated over frames: XLION_JITTER_HISTORY_FIX)
        xmath::fvec3            m_JitterNdc{ 0, 0, 0 };
        xmath::fmat4            m_TailPrevW2CNoJit{};
        xmath::fmat4            m_GIScreenPrevW2CNoJit{};
        int                     m_SharpMotionFrame = -1;     // the TAA frame the full-size G-buffer's motion was drawn in (C22: the TAA reads it then)
        std::array<xgpu::texture, 2> m_TAAVel;          // the TAA's motion a pixel, px (rg16f): this frame's written, last frame's read where the pixel was - velocity disocclusion
        int                     m_TAAVelFrames = 0;     // (frames written since made: last frame's is read only once both were written - a storage image is undefined before)
        xgpu::texture           m_TAADebug;             // XLION_TAA_DEBUG: the TAA's diagnostic, written beside its output (made the first time it is on)
        // XLION_HEAT (measurement): the GI gather's clock a pixel (4-uint header, then the gather's grid), its colours (xlionrender_heat_comp.glsl),
        // the header a ring slot for the HEAT line, the tone mapper over m_HDR with the colours in the bloom's slot; made the first time it is on
        xgpu::buffer            m_HeatBuffer, m_HeatStats;
        xgpu::buffer            m_LightTiles;           // A5.3: each 8 x 8 tile's local lights (xlionrender_light_tiles_comp.glsl), made the first time
        xgpu::texture           m_HeatTex;
        xgpu::pipeline_instance m_HeatToneMap;
        double                  m_HeatSum = 0.0, m_HeatN = 0.0, m_HeatLanes = 0.0, m_HeatGroups = 0.0;
        int                     m_HeatFrames = 0;
        xgpu::pipeline_instance m_BlitInstance;         // samples m_Output, for instance::Blit
        // DebugView (xlionrender_debug_view_frag.glsl): one of the images below, by its render graph name, in place of the frame ("" off).
        // Mode -1: by its format (depth 3, else 0 colour); Error says why nothing is shown (no such image, not made yet, integer data)
        std::string             m_DebugImage;
        int                     m_DebugMode = -1, m_DebugChannel = 0, m_DebugMip = 0;
        float                   m_DebugScale = 1.0f, m_DebugBias = 0.0f;
        std::string             m_DebugError;
        xgpu::pipeline_instance m_DebugInstance;        // samples the shown image (made again when it changes)
        const void*             m_pDebugBound = nullptr;
        // The images DebugView can show, by the names the frame's passes declare them with (rg::graph; their xGPU m_pName). Images made
        // only for some switches are listed even when not made (m_Private null: "not made")
        template< typename T_FN > void ForEachImage( T_FN&& Fn ) noexcept
        {
            Fn("Render target: HDR colour", m_HDR);               Fn("Render target: last frame's HDR", m_HDRPrev);
            Fn("Render target: depth", m_Depth);                  Fn("Render target: prepass normals", m_GPreNormal);
            Fn("Render target: prepass ids", m_GPreId);           Fn("Render target: tile attributes", m_TileColor);
            Fn("Render target: tile graphs", m_TileGraphDepth);   Fn("Render target: exposure", m_ExposureTex);
            Fn("Render target: tail colour", m_TailGColor);       Fn("Render target: tail depth", m_TailGDepth);
            Fn("Render target: tail motion", m_TailGMotion);      Fn("Render target: tail diffuse", m_TailDiffuse);
            Fn("Render target: tail specular", m_TailSpecular);   Fn("Render target: sharp colour", m_SharpGColor);
            Fn("Render target: sharp depth", m_SharpGDepth);      Fn("Render target: sharp motion", m_SharpGMotion);
            Fn("Render target: bloom", m_Bloom[0]);               Fn("Render target: TAA", m_TAA[(m_TAAFrame + 1) & 1]);      /* (the one this frame wrote: m_TAAFrame moved on after it - DebugView runs after the post chain) */
            Fn("Render target: TAA debug", m_TAADebug);
            Fn("GI: screen colour", m_GIScreenColor);             Fn("GI: screen depth", m_GIScreenDepth);
        }
        // The selection's outline (instance::renderOutline), made the first time something is selected: the mask (R: covered, G: in plain sight) with a
        // depth of its own, its pass, the casters' pipelines with the mask's fragment (they sample m_Depth), and the composite over m_Output (it samples m_Mask)
        xgpu::texture           m_Mask;                 // R8G8B8A8_UNORM
        xgpu::texture           m_MaskDepth;            // DEPTH_F32
        xgpu::renderpass        m_MaskPass;
        xgpu::renderpass        m_OutlinePass;          // m_Output, loaded
        xgpu::pipeline_instance m_MaskInstance[3];      // per geometry path: CreateMesh meshes, rigid, skin
        xgpu::pipeline_instance m_OutlineInstance;
        // The GI read at half resolution (instance::renderScenePass, scene::gi_settings::m_bHalfRes; the gather from the G-buffer): its
        // depth (each pixel's GI buffer: the gather's classification) and a color (not written), the pass, the buffer the read writes (entry
        // 0: its size, then a uvec4 a pixel: xlionrender_gi_gather_frag.glsl) and zeros to clear it with
        xgpu::texture           m_GIScreenColor;        // R16G16B16A16_SFLOAT, half size (not written)
        xgpu::texture           m_GIScreenDepth;        // DEPTH_F32, half size
        xgpu::renderpass        m_GIScreenPass;
        xgpu::buffer            m_GIScreenBuffer, m_GIScreenZeros;
        // Temporal: last frame's buffer (swapped with m_GIScreenBuffer each frame: this frame's pixels off the pattern reproject from it),
        // the header's ring (host visible, instance::screen_header_v entries a frame: xlionrender_gi_gather_frag.glsl), last frame's
        // camera-centered world to clip and eye, and whether the last buffer holds a frame of this size
        xgpu::buffer            m_GIScreenFilter;       // (the screen filter's other side: xlionrender_gi_denoise_comp.glsl)
        // (Tomas 2026-11-13: the history holds what stands still only - m_GIScreenHistory written this frame, m_GIScreenPrev last frame's, swapped each
        // frame; under a mover the static entry is carried on. m_GIScreenBuffer is the frame's output, the movers' pixels in it; cleared each frame)
        xgpu::buffer            m_GIScreenHistory;
        xgpu::buffer            m_GIScreenPrev, m_GIScreenHeader, m_GIScreenCounters;     // (the counters: host visible, an entry a ring frame)
        // The ambient occlusion on that grid (xlionrender_ao_comp.glsl): this frame's and last frame's (vec2 a pixel: the AO, the distance)
        xgpu::buffer            m_AO, m_AOPrev, m_AORaw;      // (m_AORaw: this frame's unfiltered)
        bool                    m_bAOPrev           = false;
        xmath::fmat4            m_GIScreenPrevW2C{};
        xmath::fvec3            m_GIScreenPrevEye{};
        std::uint32_t           m_GIScreenFrame     = 0;
        bool                    m_bGIScreenPrev     = false;
        bool                    m_bGIScreenPerPixel = true;     // the read it was (scene::gi_settings::m_bPerPixel): a change drops it
        // GPU culling's occlusion (instance::createCullTargets, the main view's): the scene pass again, loaded (its second pass draws after
        // the Hi-Z), and the Hi-Z pyramid (gpu_cull::HiZFloats floats)
        xgpu::renderpass        m_ScenePassLoad;
        xgpu::buffer            m_HiZ;
        // The G-buffer prepass (instance::createGPreTargets, the main view's; gi_gbuffer_gather_plan.md): the opaque depth with each pixel's
        // world normal (octahedral) and submesh instance (slot + 1, cluster) - the GI gathers from them; the scene pass loads its depth
        xgpu::texture           m_GPreNormal;           // R16G16_SFLOAT
        xgpu::texture           m_GPreId;               // R32G32B32A32_SFLOAT: slot + 1 (0 nothing), the cluster (-(cluster + 1): a back face), the exact distance to the eye
        xgpu::renderpass        m_GPrePass;
        xgpu::renderpass        m_GPrePassLoad;         // (GPU occlusion's pass 2: the G-buffer loaded again after the Hi-Z)
        // The GI gather from them (xlionrender_gi_gather_common.glsl): its classification's and passes' instances (this view's G-buffer
        // bound), the table (a region a ring frame: instance::gather_table_uints_v)
        xgpu::pipeline_instance m_GIClassifyInstance, m_GIGatherInstance;
        xgpu::texture           m_TileGraphDepth;       // the tile pass's classification (A5.2): each pixel's graph as its depth
        xgpu::pipeline_instance m_TileClassifyInstance;
        xgpu::buffer            m_GIGatherTable;
        // The screen tail (instance::renderTail, the main view in light_mode::SCREEN; documentation/lighting_p3_design.md "The decision"),
        // made the first time: the thin G-buffer at half size - the octahedral normal and roughness (10-10-10-2; the materials' G-buffer
        // variants draw into it) and its depth -, its pass; the tail pass's two targets (demodulated diffuse and specular, 11-11-10 floats),
        // its pass and pipeline instance (it samples the G-buffer); the denoiser's records (a uvec4 a texel: xlionrender_tail_denoise_comp.glsl):
        // the result - what the materials read (binding 14) and next frame's history - and the temporal pass's
        xgpu::texture           m_TailGColor, m_TailGDepth;
        // Reflections' sharp receivers (instance::renderTail): the same G-buffer at the view's full size (the materials' G-buffer variants
        // drawn again), their rays a pixel each; their records after the tail's and the reflections' half-size ones, 2 pixels a record
        xgpu::texture           m_SharpGColor, m_SharpGMotion, m_SharpGDepth;
        xgpu::renderpass        m_SharpGPass;
        xgpu::pipeline_instance m_TailFromSharp;        // the half-size G-buffer taken from the full-size one (xlionrender_tail_from_sharp_frag.glsl)
        int                     m_SharpWidth = 0, m_SharpHeight = 0;
        xgpu::texture           m_TailGMotion;          // last frame's screen position (ndc xy), clip w, 1 (0: no motion: the camera's reprojection)
        xgpu::renderpass        m_TailGPass;
        xgpu::texture           m_TailDiffuse, m_TailSpecular;
        xgpu::renderpass        m_TailPass;
        xgpu::pipeline_instance m_TailInstance;
        xgpu::renderpass        m_SunMaskPass;
        xgpu::pipeline_instance m_SunMaskInstance;
        xmath::fmat4            m_SunMaskPrevW2C;       // the reuse (instance::m_SunMaskMode 6): last frame's camera-centred clip, its eye, the sun's parameters
        xmath::fvec3            m_SunMaskPrevEye;
        std::vector<float>      m_SunMaskPrevKey;       // (empty: no last frame - its first, a resize)
        int                     m_SunMaskPrevW = 0, m_SunMaskPrevH = 0;
        bool                    m_bSunMaskValid = false; // (last frame computed the mask: the reuse may keep it)
        xgpu::buffer            m_TailResult, m_TailTemporal;
        xgpu::buffer            m_ReflTemporal, m_ReflHistory;      // the reflections' reconstruction (xlionrender_refl_denoise_comp.glsl): its pass between, last frame's result
        xmath::fmat4            m_TailPrevC2W = xmath::fmat4::fromIdentity();
        xmath::fmat4            m_TailPrevW2C = xmath::fmat4::fromIdentity();
        xmath::fvec3            m_TailPrevEye{ 0, 0, 0 };
        bool                    m_bTailPrev      = false;     // m_TailResult holds last frame's (of this size)
        std::uint32_t           m_TailFrame      = 0;
        int                     m_TailWidth      = 0;
        int                     m_TailHeight     = 0;
        int                     m_GIScreenWidth  = 0;
        int                     m_GIScreenHeight = 0;
        int                     m_MaskWidth  = 0;
        int                     m_MaskHeight = 0;
        int                     m_Width     = 0;
        int                     m_Height    = 0;
    };
}

#endif
