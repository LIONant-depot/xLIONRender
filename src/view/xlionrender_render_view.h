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
        // XLION_HEAT (measurement): the GI gather's clock a pixel (4-uint header, then the gather's grid), its colours (xlionrender_heat_comp.glsl),
        // the header a ring slot for the HEAT line, the tone mapper over m_HDR with the colours in the bloom's slot; made the first time it is on
        xgpu::buffer            m_HeatBuffer, m_HeatStats;
        xgpu::buffer            m_LightTiles;           // A5.3: each 8 x 8 tile's local lights (xlionrender_light_tiles_comp.glsl), made the first time
        xgpu::texture           m_HeatTex;
        xgpu::pipeline_instance m_HeatToneMap;
        double                  m_HeatSum = 0.0, m_HeatN = 0.0, m_HeatLanes = 0.0, m_HeatGroups = 0.0;
        int                     m_HeatFrames = 0;
        xgpu::pipeline_instance m_BlitInstance;         // samples m_Output, for instance::Blit
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
        xgpu::buffer            m_GIScreenPrev, m_GIScreenHeader, m_GIScreenCounters;     // (the counters: host visible, an entry a ring frame)
        // The ambient occlusion on that grid (xlionrender_ao_comp.glsl): this frame's and last frame's (vec2 a pixel: the AO, the distance)
        xgpu::buffer            m_AO, m_AOPrev, m_AORaw;      // (m_AORaw: this frame's unfiltered)
        bool                    m_bAOPrev           = false;
        xmath::fmat4            m_GIScreenPrevW2C;
        xmath::fvec3            m_GIScreenPrevEye;
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
