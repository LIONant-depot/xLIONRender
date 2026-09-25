#ifndef XLIONRENDER_RENDERER_H
#define XLIONRENDER_RENDERER_H
#pragma once

// The actual xGPU rendering work - one mesh per shape (built once from xprim_geom, NOT
// xGPU's own Examples/E19_MaterialEditor - see xlionrender_plugin_entry.cpp's own comment), one
// simple unlit pipeline, and a per-frame submit/draw list. Purely internal to this DLL: nothing
// outside LIONRender.dll ever touches this class directly, not even xLION.exe (see xlionrender_api.h
// for the two functions that DO cross the boundary) - so it needs no export macro at all.
#include "dependencies/xGPU/source/xgpu.h"
#include "dependencies/xmath/source/xmath.h"
#include "dependencies/xprim_geom/source/xprim_geom.h"
#include "xlionrender_primitive.h"
#include <vector>
#include <cstdint>

namespace xlionrender
{
    class renderer
    {
    public:
        bool Init    (xgpu::device& Device) noexcept;
        void Release (void) noexcept;

        // Called by system::Collect (compiled into this same DLL) once per entity per frame - just
        // appends to m_DrawList, no GPU work here. EntityValue (xecs::component::entity::m_Value) is
        // only used to find the selected item again at Draw time - see SetSelected.
        void Submit  (shape Shape, const xmath::fmat4& L2W, const xmath::fvec3& Color, std::uint64_t EntityValue) noexcept;

        // The one entity (if any) to outline this frame - xlionrender::SetSelectedEntity forwards here.
        void SetSelected(std::uint64_t EntityValue) noexcept { m_SelectedEntity = EntityValue; }

        // Called once per frame by the exported xlionrender::Draw (xlionrender_api.h), from the host's
        // own render callback - issues the real draw calls for everything Submitted since the last
        // Draw, then clears the list. ViewportW/H (pixels) size the selected item's outline width.
        void Draw    (xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH) noexcept;

    private:
        struct mesh
        {
            xgpu::buffer m_Verts;
            xgpu::buffer m_Indices;
            int          m_IndexCount = 0;
        };
        struct draw_item
        {
            shape         m_Shape;
            xmath::fmat4  m_L2W;
            xmath::fvec3  m_Color;
            std::uint64_t m_EntityValue;
        };
        struct push_constants
        {
            xmath::fmat4 m_L2C;
            xmath::fvec4 m_Color;
        };
        struct outline_push_constants
        {
            xmath::fmat4 m_L2C;
            xmath::fvec4 m_ViewportAndRadius; // x width, y height, z radius (pixels), w unused
            xmath::fvec4 m_Color;
        };

        static bool Ok(xgpu::device::error* pErr) noexcept;
        bool BuildMesh(xgpu::device& Device, mesh& Mesh, const xprim_geom::mesh& GeneratedMesh) noexcept;
        void DrawItem(xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, const draw_item& Item) noexcept;

        xgpu::device*           m_pDevice = nullptr;
        bool                    m_bReady  = false;
        xgpu::vertex_descriptor m_VD;
        xgpu::pipeline          m_Pipeline;
        xgpu::pipeline_instance m_Instance;

        // Outline pass: expanded-along-normal silhouette, drawn behind the selected item's own normal
        // draw (xlionrender_outline_vert/frag.glsl) - cull FRONT (render only back faces, so the real
        // object's front-face draw fully covers the interior, leaving only the perimeter "skirt"
        // visible) and a depth bias (pushed slightly away from the camera) so it never z-fights the
        // real surface on coplanar/glancing faces. Reuses m_VD - same Position+Normal vertex layout.
        xgpu::pipeline          m_OutlinePipeline;
        xgpu::pipeline_instance m_OutlineInstance;

        mesh                    m_Meshes[4]; // indexed by shape
        std::vector<draw_item>  m_DrawList;
        std::uint64_t           m_SelectedEntity = ~0ull; // xecs::component::entity::invalid_entity_v
    };
}

#endif // XLIONRENDER_RENDERER_H
