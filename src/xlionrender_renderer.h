#ifndef XLIONRENDER_RENDERER_H
#define XLIONRENDER_RENDERER_H
#pragma once

// The actual xGPU rendering work - one mesh per shape (built once from xprim_geom, NOT
// xGPU's own Examples/E19_MaterialEditor - see xlionrender_plugin_entry.cpp's own comment), one
// simple unlit pipeline, and a per-frame submit/draw list. Purely internal to this DLL: nothing
// outside LIONRender.dll ever touches this class directly, not even xLION.exe (see xlionrender_api.h
// for the two functions that DO cross the boundary) - so it needs no export macro at all.
#include "dependencies/xGPU/source/xGPU.h"
#include "dependencies/xmath/source/xmath.h"
#include "dependencies/xprim_geom/source/xprim_geom.h"
#include "xlionrender_primitive.h"
#include "xlionrender_view.h"
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
        // only used to find the selected item again at Draw time - see SetSelected. Scale is the same
        // vector already folded into L2W - carried separately too, only for the capsule draw path
        // (see DrawItem); every other shape ignores it.
        void Submit  (shape Shape, const xmath::fmat4& L2W, const xmath::fvec3& Scale, const xmath::fvec3& Color, std::uint64_t EntityValue) noexcept;

        // The one entity (if any) to outline this frame - xlionrender::SetSelectedEntity forwards here.
        void SetSelected(std::uint64_t EntityValue) noexcept { m_SelectedEntity = EntityValue; }

        // Called once per frame by the exported xlionrender::Draw (xlionrender_api.h), from the host's
        // own render callback - issues the real draw calls for everything Submitted since the last
        // Draw, then clears the list. ViewportW/H (pixels) size the selected item's outline width.
        // pRoles (null: everything is the document): the items that are CONTEXT are drawn first, then a full screen quad of the fade color (only where something was drawn: the depth of an
        // empty pixel is the cleared one), then the rest: the context is there to see and is in the way of nothing (editing in context, prefabs_plan.md phase 7).
        void Draw    (xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH, const roles* pRoles = nullptr) noexcept;

        // What the last Draw did with the roles: the items of the context, the items of the document, whether the fade quad was drawn.
        struct role_stats { int m_Context = 0; int m_Document = 0; bool m_bFaded = false; };
        const role_stats& getLastRoleStats() const noexcept { return m_LastRoleStats; }

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
            xmath::fvec3  m_Scale;
            xmath::fvec3  m_Color;
            std::uint64_t m_EntityValue;
        };
        struct push_constants
        {
            xmath::fmat4 m_L2C;
            xmath::fvec4 m_Color;
        };
        // Capsule only (see DrawItem) - L2C here carries world ROTATION+TRANSLATION but NOT Scale
        // (stripped via Item.m_L2W * fromScale(1/Item.m_Scale)); Scale is reapplied by the vertex
        // shader with the cap radius decoupled from the body's length, so hemisphere caps stay round
        // at any Scale instead of stretching into cones (direct user report: "the top and the bottom"
        // - the caps - "looks more like a sphere... specially when you scale it in Y").
        struct capsule_push_constants
        {
            xmath::fmat4 m_L2C;
            xmath::fvec4 m_Color;
            xmath::fvec4 m_RadiusHeightScale; // x local radius (xprim_geom Generate's Radius, Init); y Scale.x; z Scale.z; w Scale.y
        };
        // Legacy radial-from-pivot outline (96 bytes). Kept as fallback when the projected OBB is
        // unusable (near-plane cross, missing/degenerate bbox after axis validation).
        struct outline_push_constants
        {
            xmath::fmat4 m_L2C;
            xmath::fvec4 m_ViewportAndRadius; // x width, y height, z radius (pixels), w unused
            xmath::fvec4 m_Color;
        };
        // Preferred OBB screen-space expand outline (128 bytes std140: mat4 + 4 vec4).
        struct outline_obb_push_constants
        {
            xmath::fmat4 m_L2C;
            xmath::fvec4 m_ViewportAndRadius; // xy viewport px; z radius px; w max fractional expand
            xmath::fvec4 m_Color;
            xmath::fvec4 m_Bounds;             // xy projected OBB center px; zw half-extents in axis frame
            xmath::fvec4 m_Axis;               // xy normalized long axis in viewport px; zw unused
        };

        static bool Ok(xgpu::device::error* pErr) noexcept;
        bool BuildMesh(xgpu::device& Device, mesh& Mesh, const xprim_geom::mesh& GeneratedMesh) noexcept;
        void DrawItem(xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, const draw_item& Item) noexcept;
        void DrawFade(xgpu::cmd_buffer& CmdBuffer, const roles& Roles) noexcept;

        // Local AABB half-extents of the mesh built for Shape (matches xprim_geom Generate sizes in Init).
        static xmath::fvec3 ShapeLocalHalfExtents(shape Shape) noexcept;

        // Projected OBB of LocalHalfExtents under L2C into viewport pixels (same NDC->px as the OBB
        // shader). Returns false on near-plane cross / unusable extents - caller must use legacy.
        static bool ComputeProjectedOutlineObb( const xmath::fmat4& L2C
                                              , float ViewportW, float ViewportH
                                              , const xmath::fvec3& LocalHalfExtents
                                              , xmath::fvec4& OutBounds
                                              , xmath::fvec2& OutAxis ) noexcept;

        xgpu::device*           m_pDevice = nullptr;
        bool                    m_bReady  = false;
        xgpu::vertex_descriptor m_VD;
        xgpu::pipeline          m_Pipeline;
        xgpu::pipeline_instance m_Instance;

        // Outline pass: expanded silhouette, drawn behind the selected item's own normal draw
        // (cull FRONT so the real front-face fill covers the interior). Depth writes off; existing
        // depth-bias settings left unchanged. Preferred path = OBB expand; legacy = radial-from-pivot.
        xgpu::pipeline          m_OutlinePipeline;
        xgpu::pipeline_instance m_OutlineInstance;
        xgpu::pipeline          m_OutlineObbPipeline;
        xgpu::pipeline_instance m_OutlineObbInstance;

        // Capsule's own pipeline (see capsule_push_constants) - Cube/Sphere/Cylinder still share m_Pipeline/m_Instance.
        xgpu::pipeline          m_CapsulePipeline;
        xgpu::pipeline_instance m_CapsuleInstance;

        // The fade quad of the context (see Draw): the primitive shaders with a quad that fills the screen at the far plane, blended, tested NOT_EQUAL to the cleared depth, writing nothing.
        xgpu::pipeline          m_FadePipeline;
        xgpu::pipeline_instance m_FadeInstance;
        mesh                    m_FadeMesh;
        role_stats              m_LastRoleStats;

        mesh                    m_Meshes[4]; // indexed by shape
        std::vector<draw_item>  m_DrawList;
        std::uint64_t           m_SelectedEntity = ~0ull; // xecs::component::entity::invalid_entity_v
    };
}

#endif // XLIONRENDER_RENDERER_H