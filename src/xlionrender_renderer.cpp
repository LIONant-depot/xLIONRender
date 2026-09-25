#include "xlionrender_renderer.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace xlionrender
{
    inline constexpr std::uint32_t g_VertShader[] =
    {
        #include "xlionrender_primitive_vert.h"
    };
    inline constexpr std::uint32_t g_FragShader[] =
    {
        #include "xlionrender_primitive_frag.h"
    };
    inline constexpr std::uint32_t g_OutlineVertShader[] =
    {
        #include "xlionrender_outline_vert.h"
    };
    inline constexpr std::uint32_t g_OutlineFragShader[] =
    {
        #include "xlionrender_outline_frag.h"
    };
    inline constexpr std::uint32_t g_OutlineObbVertShader[] =
    {
        #include "xlionrender_outline_obb_vert.h"
    };
    inline constexpr std::uint32_t g_OutlineObbFragShader[] =
    {
        #include "xlionrender_outline_obb_frag.h"
    };

    bool renderer::Ok(xgpu::device::error* pErr) noexcept
    {
        if (!pErr) return true;
        std::printf("xlionrender::renderer: %s\n", std::string(xgpu::getErrorMsg(pErr)).c_str());
        return false;
    }

    bool renderer::BuildMesh(xgpu::device& Device, mesh& Mesh, const xprim_geom::mesh& GeneratedMesh) noexcept
    {
        if (!Ok(Device.Create(Mesh.m_Verts, { .m_Type = xgpu::buffer::type::VERTEX, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ
            , .m_EntryByteSize = sizeof(xprim_geom::vertex), .m_EntryCount = static_cast<int>(GeneratedMesh.m_Vertices.size()) }))) return false;
        (void)Mesh.m_Verts.MemoryMap(0, static_cast<int>(GeneratedMesh.m_Vertices.size()), [&](void* pData)
        {
            std::memcpy(pData, GeneratedMesh.m_Vertices.data(), GeneratedMesh.m_Vertices.size() * sizeof(xprim_geom::vertex));
        });

        if (!Ok(Device.Create(Mesh.m_Indices, { .m_Type = xgpu::buffer::type::INDEX, .m_EntryByteSize = sizeof(std::uint32_t)
            , .m_EntryCount = static_cast<int>(GeneratedMesh.m_Indices.size()) }))) return false;
        (void)Mesh.m_Indices.MemoryMap(0, static_cast<int>(GeneratedMesh.m_Indices.size()), [&](void* pData)
        {
            std::memcpy(pData, GeneratedMesh.m_Indices.data(), GeneratedMesh.m_Indices.size() * sizeof(std::uint32_t));
        });

        Mesh.m_IndexCount = static_cast<int>(GeneratedMesh.m_Indices.size());
        return true;
    }

    xmath::fvec3 renderer::ShapeLocalHalfExtents(shape Shape) noexcept
    {
        // Matches the xprim_geom::Generate sizes used in Init - all current primitives fit in a
        // unit cube centered at the origin (half-extents 0.5). Capsule(Height=1,Radius=0.5) too.
        (void)Shape;
        return xmath::fvec3(0.5f, 0.5f, 0.5f);
    }

    bool renderer::ComputeProjectedOutlineObb( const xmath::fmat4& L2C
                                             , float ViewportW, float ViewportH
                                             , const xmath::fvec3& LocalHalfExtents
                                             , xmath::fvec4& OutBounds
                                             , xmath::fvec2& OutAxis ) noexcept
    {
        if (ViewportW <= 0.0f || ViewportH <= 0.0f) return false;
        if (LocalHalfExtents.m_X <= 0.0f || LocalHalfExtents.m_Y <= 0.0f || LocalHalfExtents.m_Z <= 0.0f) return false;

        constexpr float kNearW = 0.00001f;
        const float hx = LocalHalfExtents.m_X;
        const float hy = LocalHalfExtents.m_Y;
        const float hz = LocalHalfExtents.m_Z;

        xmath::fvec2 pixels[8];
        float        centerX = 0.0f, centerY = 0.0f;

        // Same viewport-local mapping as xlionrender_outline_obb_vert.glsl / xGPU's positive-height
        // Vulkan viewport (y=0 at top, height > 0): px = (ndc * 0.5 + 0.5) * viewportPx.
        for (int i = 0; i < 8; ++i)
        {
            const float sx = (i & 1) ? hx : -hx;
            const float sy = (i & 2) ? hy : -hy;
            const float sz = (i & 4) ? hz : -hz;
            const xmath::fvec4 clip = L2C * xmath::fvec4(sx, sy, sz, 1.0f);

            // v1 near-plane: any corner behind/crossing near (w<=eps or z not in [0,w]) → unusable.
            if (clip.m_W <= kNearW) return false;
            if (clip.m_Z < 0.0f || clip.m_Z > clip.m_W) return false;

            const float invW = 1.0f / clip.m_W;
            const float ndcX = clip.m_X * invW;
            const float ndcY = clip.m_Y * invW;
            pixels[i] = xmath::fvec2((ndcX * 0.5f + 0.5f) * ViewportW
                                   , (ndcY * 0.5f + 0.5f) * ViewportH);
            centerX += pixels[i].m_X;
            centerY += pixels[i].m_Y;
        }

        centerX *= 0.125f;
        centerY *= 0.125f;
        const xmath::fvec2 center(centerX, centerY);

        // Project the three local OBB axes (through L2C) into viewport pixels; pick the longest.
        // Axis Y in the shader is screen-space perpendicular(axisX) - we do NOT pass a second 3D axis.
        auto ProjectAxis = [&](float ax, float ay, float az) -> xmath::fvec2
        {
            const xmath::fvec4 c0 = L2C * xmath::fvec4(-ax, -ay, -az, 1.0f);
            const xmath::fvec4 c1 = L2C * xmath::fvec4( ax,  ay,  az, 1.0f);
            if (c0.m_W <= kNearW || c1.m_W <= kNearW) return xmath::fvec2(0.0f, 0.0f);
            if (c0.m_Z < 0.0f || c0.m_Z > c0.m_W) return xmath::fvec2(0.0f, 0.0f);
            if (c1.m_Z < 0.0f || c1.m_Z > c1.m_W) return xmath::fvec2(0.0f, 0.0f);
            const float i0 = 1.0f / c0.m_W;
            const float i1 = 1.0f / c1.m_W;
            const xmath::fvec2 p0((c0.m_X * i0 * 0.5f + 0.5f) * ViewportW, (c0.m_Y * i0 * 0.5f + 0.5f) * ViewportH);
            const xmath::fvec2 p1((c1.m_X * i1 * 0.5f + 0.5f) * ViewportW, (c1.m_Y * i1 * 0.5f + 0.5f) * ViewportH);
            return p1 - p0;
        };

        const xmath::fvec2 axisCandidates[3] =
        { ProjectAxis(hx, 0.0f, 0.0f)
        , ProjectAxis(0.0f, hy, 0.0f)
        , ProjectAxis(0.0f, 0.0f, hz)
        };

        int   iBest = -1;
        float bestLenSq = 0.0f;
        for (int i = 0; i < 3; ++i)
        {
            const float lenSq = axisCandidates[i].LengthSq();
            if (lenSq > bestLenSq) { bestLenSq = lenSq; iBest = i; }
        }

        constexpr float kMinAxisLenSq = 1.0e-4f; // ~0.01 px
        xmath::fvec2 axisX;
        if (iBest < 0 || bestLenSq < kMinAxisLenSq)
        {
            // Degenerate projected axes: screen-aligned bound from projected corners (still OBB path).
            axisX = xmath::fvec2(1.0f, 0.0f);
        }
        else
        {
            axisX = axisCandidates[iBest].NormalizeCopy();
        }
        const xmath::fvec2 axisY(-axisX.m_Y, axisX.m_X);

        float halfX = 0.0f, halfY = 0.0f;
        for (int i = 0; i < 8; ++i)
        {
            const xmath::fvec2 d = pixels[i] - center;
            halfX = std::max(halfX, std::fabs(d.Dot(axisX)));
            halfY = std::max(halfY, std::fabs(d.Dot(axisY)));
        }

        constexpr float kMinHalf = 0.5f; // sub-pixel / empty → unusable for expand
        if (halfX < kMinHalf || halfY < kMinHalf) return false;

        OutBounds = xmath::fvec4(center.m_X, center.m_Y, halfX, halfY);
        OutAxis   = axisX;
        return true;
    }

    bool renderer::Init(xgpu::device& Device) noexcept
    {
        static_assert(sizeof(outline_obb_push_constants) == 128,
            "outline_obb_push_constants must be 128 bytes (mat4 + 4 vec4, std140)");
        static_assert(sizeof(outline_push_constants) == 96,
            "legacy outline_push_constants must stay 96 bytes");
        if (m_bReady) return true;
        m_pDevice = &Device;

        auto Attributes = std::array
        { xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(xprim_geom::vertex, m_Position), .m_Format = xgpu::vertex_descriptor::format::FLOAT_3D }
        , xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(xprim_geom::vertex, m_Normal),   .m_Format = xgpu::vertex_descriptor::format::FLOAT_3D }
        };
        if (!Ok(Device.Create(m_VD, xgpu::vertex_descriptor::setup{ .m_Topology = xgpu::vertex_descriptor::topology::TRIANGLE_LIST
            , .m_VertexSize = sizeof(xprim_geom::vertex), .m_Attributes = Attributes }))) return false;

        // One mesh per shape - built once here, drawn many times per frame
        if (!BuildMesh(Device, m_Meshes[static_cast<int>(shape::CUBE)],     xprim_geom::cube::Generate(0, 0, 0, 0, { 1.0f, 1.0f, 1.0f })))    return false;
        if (!BuildMesh(Device, m_Meshes[static_cast<int>(shape::CAPSULE)],  xprim_geom::capsule::Generate(4, 12, 0.5f, 1.0f)))                 return false;
        if (!BuildMesh(Device, m_Meshes[static_cast<int>(shape::SPHERE)],   xprim_geom::uvsphere::Generate(8, 12, 1.0f, 0.5f)))                return false;
        if (!BuildMesh(Device, m_Meshes[static_cast<int>(shape::CYLINDER)], xprim_geom::cylinder::Generate(4, 12, 1.0f, 0.5f, 0.5f)))          return false;

        auto Shader = [&](xgpu::shader& Out, xgpu::shader::type::bit Type, const std::uint32_t* pCode, std::size_t nWords)
        {
            return Ok(Device.Create(Out, { .m_Type = Type, .m_Sharer = xgpu::shader::setup::raw_data{ std::span{ (std::int32_t*)pCode, nWords } } }));
        };

        {
            xgpu::shader Vert, Frag;
            if (!Shader(Vert, xgpu::shader::type::bit::VERTEX,   g_VertShader, std::size(g_VertShader))) return false;
            if (!Shader(Frag, xgpu::shader::type::bit::FRAGMENT, g_FragShader, std::size(g_FragShader))) return false;
            auto Shaders = std::array<const xgpu::shader*, 2>{ &Frag, &Vert };
            if (!Ok(Device.Create(m_Pipeline, xgpu::pipeline::setup{ .m_VertexDescriptor = m_VD, .m_Shaders = Shaders
                , .m_PushConstantsSize = sizeof(push_constants) }))) return false;
            if (!Ok(Device.Create(m_Instance, { .m_PipeLine = m_Pipeline }))) return false;
        }

        // Legacy radial-from-pivot outline pipeline (fallback).
        {
            xgpu::shader Vert, Frag;
            if (!Shader(Vert, xgpu::shader::type::bit::VERTEX,   g_OutlineVertShader, std::size(g_OutlineVertShader))) return false;
            if (!Shader(Frag, xgpu::shader::type::bit::FRAGMENT, g_OutlineFragShader, std::size(g_OutlineFragShader))) return false;
            auto Shaders = std::array<const xgpu::shader*, 2>{ &Frag, &Vert };
            if (!Ok(Device.Create(m_OutlinePipeline, xgpu::pipeline::setup{ .m_VertexDescriptor = m_VD, .m_Shaders = Shaders
                , .m_PushConstantsSize = sizeof(outline_push_constants)
                , .m_Primitive    = { .m_Cull = xgpu::pipeline::primitive::cull::FRONT }
                , .m_DepthStencil = { .m_DepthBiasConstantFactor = 1.25f, .m_DepthBiasSlopeFactor = 1.75f
                                     , .m_bDepthWriteEnable = false, .m_bDepthBiasEnable = true }
                }))) return false;
            if (!Ok(Device.Create(m_OutlineInstance, { .m_PipeLine = m_OutlinePipeline }))) return false;
        }

        // Preferred OBB screen-space expand outline pipeline.
        {
            xgpu::shader Vert, Frag;
            if (!Shader(Vert, xgpu::shader::type::bit::VERTEX,   g_OutlineObbVertShader, std::size(g_OutlineObbVertShader))) return false;
            if (!Shader(Frag, xgpu::shader::type::bit::FRAGMENT, g_OutlineObbFragShader, std::size(g_OutlineObbFragShader))) return false;
            auto Shaders = std::array<const xgpu::shader*, 2>{ &Frag, &Vert };
            if (!Ok(Device.Create(m_OutlineObbPipeline, xgpu::pipeline::setup{ .m_VertexDescriptor = m_VD, .m_Shaders = Shaders
                , .m_PushConstantsSize = sizeof(outline_obb_push_constants)
                , .m_Primitive    = { .m_Cull = xgpu::pipeline::primitive::cull::FRONT }
                , .m_DepthStencil = { .m_DepthBiasConstantFactor = 1.25f, .m_DepthBiasSlopeFactor = 1.75f
                                     , .m_bDepthWriteEnable = false, .m_bDepthBiasEnable = true }
                }))) return false;
            if (!Ok(Device.Create(m_OutlineObbInstance, { .m_PipeLine = m_OutlineObbPipeline }))) return false;
        }

        m_bReady = true;
        return true;
    }

    void renderer::Release(void) noexcept
    {
        if (!m_pDevice) return;
        m_pDevice->Destroy(std::move(m_OutlineObbInstance));
        m_pDevice->Destroy(std::move(m_OutlineObbPipeline));
        m_pDevice->Destroy(std::move(m_OutlineInstance));
        m_pDevice->Destroy(std::move(m_OutlinePipeline));
        m_pDevice->Destroy(std::move(m_Instance));
        m_pDevice->Destroy(std::move(m_Pipeline));
        m_bReady = false;
    }

    void renderer::Submit(shape Shape, const xmath::fmat4& L2W, const xmath::fvec3& Color, std::uint64_t EntityValue) noexcept
    {
        m_DrawList.push_back({ Shape, L2W, Color, EntityValue });
    }

    void renderer::DrawItem(xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, const draw_item& Item) noexcept
    {
        auto& Mesh = m_Meshes[static_cast<int>(Item.m_Shape)];
        push_constants PushConstants
        { .m_L2C   = W2C * Item.m_L2W
        , .m_Color = xmath::fvec4(Item.m_Color, 1.0f)
        };
        CmdBuffer.setPushConstants(PushConstants);
        CmdBuffer.setBuffer(Mesh.m_Indices);
        CmdBuffer.setBuffer(Mesh.m_Verts);
        CmdBuffer.Draw(Mesh.m_IndexCount);
    }

    void renderer::Draw(xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH) noexcept
    {
        if (!m_bReady || m_DrawList.empty()) { m_DrawList.clear(); return; }

        // Find the selected item, if any, so its normal draw can be skipped in the first pass and
        // redone last (see the class comment) - at most one, matching primary (single) selection.
        int iSelected = -1;
        if (m_SelectedEntity != ~0ull)
            for (int i = 0, end = static_cast<int>(m_DrawList.size()); i < end; ++i)
                if (m_DrawList[i].m_EntityValue == m_SelectedEntity) { iSelected = i; break; }

        CmdBuffer.setPipelineInstance(m_Instance);
        for (int i = 0, end = static_cast<int>(m_DrawList.size()); i < end; ++i)
        {
            if (i == iSelected) continue;
            DrawItem(CmdBuffer, W2C, m_DrawList[i]);
        }

        if (iSelected >= 0)
        {
            const auto& Item = m_DrawList[iSelected];
            auto& Mesh = m_Meshes[static_cast<int>(Item.m_Shape)];
            const xmath::fmat4 L2C = W2C * Item.m_L2W;

            // Keep the selection outline alive without making it distracting. This is computed
            // here, on the host side, so the preferred OBB path and the legacy fallback share the
            // exact same pulse. One second per cycle gives the editor a moderate, ~1 Hz rhythm.
            // 2s cycle; keep the trough bright enough that the outline never looks muddy.
            constexpr float kOutlinePulsePeriodSeconds = 3.0f;
            constexpr float kTwoPi = 6.28318530717958647692f;
            static const auto s_OutlinePulseStart = std::chrono::steady_clock::now();
            const float elapsedSeconds = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - s_OutlinePulseStart).count();
            const float pulse = 0.875f + 0.125f * std::sin(elapsedSeconds * (kTwoPi / kOutlinePulsePeriodSeconds));
            const xmath::fvec4 OutlineColor{ std::min(1.0f,pulse), std::min(1.0f, pulse * 0.65f), std::min(1.0f, pulse * 0.15f), 1.0f };

            xmath::fvec4 ObbBounds;
            xmath::fvec2 ObbAxis;
            const bool bUseObb = ComputeProjectedOutlineObb( L2C, ViewportW, ViewportH
                                                           , ShapeLocalHalfExtents(Item.m_Shape)
                                                           , ObbBounds, ObbAxis );

            if (bUseObb)
            {
                CmdBuffer.setPipelineInstance(m_OutlineObbInstance);
                outline_obb_push_constants OutlinePC
                { .m_L2C               = L2C
                , .m_ViewportAndRadius = { ViewportW, ViewportH, 6.0f, 0.08f }
                , .m_Color             = OutlineColor
                , .m_Bounds            = ObbBounds
                , .m_Axis              = { ObbAxis.m_X, ObbAxis.m_Y, 0.0f, 0.0f }
                };
                CmdBuffer.setPushConstants(OutlinePC);
            }
            else
            {
                // Near-plane / degenerate / no usable bbox → legacy radial-from-pivot.
                CmdBuffer.setPipelineInstance(m_OutlineInstance);
                outline_push_constants OutlinePC
                { .m_L2C               = L2C
                , .m_ViewportAndRadius = { ViewportW, ViewportH, 6.0f, 0.0f }
                , .m_Color             = OutlineColor
                };
                CmdBuffer.setPushConstants(OutlinePC);
            }
            CmdBuffer.setBuffer(Mesh.m_Indices);
            CmdBuffer.setBuffer(Mesh.m_Verts);
            CmdBuffer.Draw(Mesh.m_IndexCount);

            CmdBuffer.setPipelineInstance(m_Instance);
            DrawItem(CmdBuffer, W2C, Item);
        }

        m_DrawList.clear();
    }
}
