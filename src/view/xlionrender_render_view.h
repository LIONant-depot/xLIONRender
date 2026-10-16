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
        int                     m_MaskWidth  = 0;
        int                     m_MaskHeight = 0;
        int                     m_Width     = 0;
        int                     m_Height    = 0;
    };
}

#endif
