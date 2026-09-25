#include "xlionrender_renderer.h"

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

    bool renderer::Init(xgpu::device& Device) noexcept
    {
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

        // Outline pass pipeline - see xlionrender_renderer.h's own comment on the member. Cull FRONT
        // (render only back faces - the real object's own front-face draw, done after, fully covers
        // the interior projection of those expanded back faces) and a depth bias pushing it slightly
        // further from the camera so coplanar/glancing faces don't z-fight the real surface.
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

        m_bReady = true;
        return true;
    }

    void renderer::Release(void) noexcept
    {
        if (!m_pDevice) return;
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

            CmdBuffer.setPipelineInstance(m_OutlineInstance);
            outline_push_constants OutlinePC
            { .m_L2C              = W2C * Item.m_L2W
            , .m_ViewportAndRadius = { ViewportW, ViewportH, 6.0f, 0.0f }
            , .m_Color             = { 1.0f, 0.65f, 0.0f, 1.0f }
            };
            CmdBuffer.setPushConstants(OutlinePC);
            CmdBuffer.setBuffer(Mesh.m_Indices);
            CmdBuffer.setBuffer(Mesh.m_Verts);
            CmdBuffer.Draw(Mesh.m_IndexCount);

            CmdBuffer.setPipelineInstance(m_Instance);
            DrawItem(CmdBuffer, W2C, Item);
        }

        m_DrawList.clear();
    }
}
