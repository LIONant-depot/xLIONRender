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
        // The GI read at half resolution (instance::renderScenePass, scene::gi_settings::m_bHalfRes): its depth and a color of the scene's
        // format (the pass's layout is the scene pass's: the same pipelines), the pass, the buffer the read writes (entry 0: its size, then
        // a uvec4 a pixel: xlionrender_gi_screen_frag.glsl) and zeros to clear it with
        xgpu::texture           m_GIScreenColor;        // R16G16B16A16_SFLOAT, half size (not written)
        xgpu::texture           m_GIScreenDepth;        // DEPTH_F32, half size
        xgpu::renderpass        m_GIScreenPass;
        xgpu::buffer            m_GIScreenBuffer, m_GIScreenZeros;
        // Temporal: last frame's buffer (swapped with m_GIScreenBuffer each frame: this frame's pixels off the pattern reproject from it),
        // the header's ring (host visible, instance::screen_header_v entries a frame: xlionrender_gi_screen_frag.glsl), last frame's
        // camera-centered world to clip and eye, and whether the last buffer holds a frame of this size
        xgpu::buffer            m_GIScreenPrev, m_GIScreenHeader, m_GIScreenCounters;     // (the counters: host visible, an entry a ring frame)
        xmath::fmat4            m_GIScreenPrevW2C;
        xmath::fvec3            m_GIScreenPrevEye;
        std::uint32_t           m_GIScreenFrame     = 0;
        bool                    m_bGIScreenPrev     = false;
        bool                    m_bGIScreenPerPixel = true;     // the read it was (scene::gi_settings::m_bPerPixel): a change drops it
        // and its draws (instance::renderScenePass): a record a cluster (details::push_static_geom, read by gl_InstanceIndex) and an
        // indirect command (details::draw_indexed_indirect), host visible, a region per ring frame (instance::screen_draws_v each)
        xgpu::buffer            m_GIScreenRecords, m_GIScreenCommands;
        int                     m_GIScreenWidth  = 0;
        int                     m_GIScreenHeight = 0;
        int                     m_MaskWidth  = 0;
        int                     m_MaskHeight = 0;
        int                     m_Width     = 0;
        int                     m_Height    = 0;
    };
}

#endif
