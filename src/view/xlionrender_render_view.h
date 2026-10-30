#ifndef XLIONRENDER_RENDER_VIEW_H
#define XLIONRENDER_RENDER_VIEW_H
#pragma once

// A render_view is one place the renderer draws a snapshot into: its own HDR + depth targets and a tone-mapped
// RGBA8 output (for ImGui::Image, thumbnails, captures). Cameras are xgpu::tools::view, with its conventions (Y-up NDC,
// depth 0 near -> 1 far). Named render_view because xlionrender::view is still the old SCENE/GAME enum.

#include "dependencies/xGPU/source/xGPU.h"
#include "dependencies/xmath/source/xmath.h"

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
        xgpu::texture           m_Depth;                // DEPTH_F32
        xgpu::texture           m_Output;               // R8G8B8A8_UNORM, display-encoded
        xgpu::renderpass        m_ScenePass;
        xgpu::renderpass        m_OutputPass;
        xgpu::renderpass        m_OverlayPass;          // m_Output + m_Depth, loaded (not cleared): instance::Render's Overlay
        xgpu::pipeline_instance m_ToneMapInstance;      // samples m_HDR
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
        int                     m_SharpWidth = 0, m_SharpHeight = 0;
        xgpu::texture           m_TailGMotion;          // last frame's screen position (ndc xy), clip w, 1 (0: no motion: the camera's reprojection)
        xgpu::renderpass        m_TailGPass;
        xgpu::texture           m_TailDiffuse, m_TailSpecular;
        xgpu::renderpass        m_TailPass;
        xgpu::pipeline_instance m_TailInstance;
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
