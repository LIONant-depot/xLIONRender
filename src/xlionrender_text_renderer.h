#ifndef XLIONRENDER_TEXT_RENDERER_H
#define XLIONRENDER_TEXT_RENDERER_H
#pragma once

// The drawing of the Text components (xlionrender_text.h): every glyph of every label is one instance of a unit quad, and one draw call covers all the labels that share a font (its atlas) and a
// depth mode - a few hundred labels are a handful of draws. The layout (xlionrender_text_layout.h) is made once per label and kept; what is made every frame is the list of instances, because the
// camera, the transform and the color are not in a layout: the vertex shader (xlionrender_text_vert.glsl) places the quads in the world from the origin and axes of their label, the orientation and
// the size mode.
//
// Order: labels that do not ask to be sorted are drawn first, grouped by font and depth mode (they are expected to be in front of something solid: no sorting costs nothing); the ones that ask
// (text::m_bSort) are drawn after them from the farthest to the nearest, in runs of consecutive labels that share a font and a depth mode.
#include "dependencies/xGPU/source/xgpu.h"
#include "dependencies/xmath/source/xmath.h"
#include "dependencies/xLIONCore/src/resources/xlioncore_resources_api.h"
#include "xlionrender_text.h"
#include "xlionrender_text_layout.h"

#include <array>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace xlionrender
{
    class text_renderer
    {
    public:
        bool Init    (xgpu::device& Device) noexcept;
        void Release (void) noexcept;

        // One label for this frame: its layout (kept by the system), where it is and how its axes point (Position, and AxisX/AxisY as long as the scale of the entity; the shader replaces them when
        // the text faces the camera). Appends instances; no GPU work here.
        void Submit  (const xlioncore::resources::font_view& Font, const text_layout::result& Layout, const text& Tx, const xmath::fvec3& Position, const xmath::fvec3& AxisX, const xmath::fvec3& AxisY) noexcept;

        // Draws everything submitted since the last Draw and clears it.
        void Draw    (xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH) noexcept;

        // What the last Draw did, for the tests and the tools.
        struct stats { int m_Labels = 0; int m_Glyphs = 0; int m_Draws = 0; int m_Dropped = 0; };
        const stats& getLastStats(void) const noexcept { return m_LastStats; }

    private:
        // One glyph for the GPU: the layout of xlionrender_text_vert.glsl's inputs. 84 bytes, no padding (the vertex descriptor packs the attributes tight).
        struct instance
        {
            float           m_Rect[4];          // em units: min xy, max zw
            float           m_UV[4];            // uv of the min corner, uv of the max corner
            float           m_Pos[4];           // xyz: the origin of the label, w: the size of one em
            float           m_AxisX[4];         // xyz: the reading direction (as long as the scale), w: the orientation
            float           m_AxisY[4];         // xyz: up (as long as the scale), w: the size mode
            std::uint8_t    m_Color[4];
        };
        static_assert(sizeof(instance) == 84, "the vertex descriptor of the text renderer packs the instance in 84 bytes");

        struct push_constants
        {
            xmath::fmat4    m_W2C;
            xmath::fvec4    m_View;             // xy: the viewport in pixels
            xmath::fvec4    m_Font;             // x: the pixel range of the field, y: the output type of the font
        };

        struct label
        {
            xgpu::texture*          m_pTexture;
            const xfont_rsc::font*  m_pFont;
            text_depth_mode         m_Depth;
            bool                    m_bSort;
            xmath::fvec3            m_Position;
            int                     m_First;    // in m_Instances
            int                     m_Count;
        };

        struct run
        {
            xgpu::texture*          m_pTexture;
            const xfont_rsc::font*  m_pFont;
            text_depth_mode         m_Depth;
            int                     m_First;
            int                     m_Count;
        };

        // The most glyphs one Draw sends (a label that does not fit whole is dropped, and counted): 16384 glyphs are over a thousand labels of a dozen letters.
        static constexpr int        kMaxInstances = 16384;
        // The buffers are rewritten by the CPU every Draw while the GPU may still read the last frames: a ring (index buffer and unit quad in each, they are tiny).
        static constexpr int        kRing         = 8;

        static bool Ok(xgpu::device::error* pErr) noexcept;
        xgpu::pipeline_instance* InstanceOf(xgpu::texture* pTexture, text_depth_mode Depth) noexcept;

        xgpu::device*               m_pDevice = nullptr;
        bool                        m_bReady  = false;
        xgpu::vertex_descriptor     m_VD;
        std::array<xgpu::pipeline, 2>                       m_Pipelines;                // by text_depth_mode
        std::map<std::pair<xgpu::texture*, int>, xgpu::pipeline_instance> m_PipelineInstances;
        std::array<std::array<xgpu::buffer, 3>, kRing>      m_Streams;                  // index, unit quad, instances
        int                         m_Slot = 0;

        std::vector<instance>       m_Instances;
        std::vector<label>          m_Labels;
        stats                       m_LastStats;
    };

    inline text_renderer g_TextRenderer;
}

#endif // XLIONRENDER_TEXT_RENDERER_H
