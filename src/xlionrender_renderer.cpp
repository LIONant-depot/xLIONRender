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

        xgpu::shader Vert, Frag;
        auto Shader = [&](xgpu::shader& Out, xgpu::shader::type::bit Type, const std::uint32_t* pCode, std::size_t nWords)
        {
            return Ok(Device.Create(Out, { .m_Type = Type, .m_Sharer = xgpu::shader::setup::raw_data{ std::span{ (std::int32_t*)pCode, nWords } } }));
        };
        if (!Shader(Vert, xgpu::shader::type::bit::VERTEX,   g_VertShader, std::size(g_VertShader))) return false;
        if (!Shader(Frag, xgpu::shader::type::bit::FRAGMENT, g_FragShader, std::size(g_FragShader))) return false;

        auto Shaders = std::array<const xgpu::shader*, 2>{ &Frag, &Vert };
        if (!Ok(Device.Create(m_Pipeline, xgpu::pipeline::setup{ .m_VertexDescriptor = m_VD, .m_Shaders = Shaders
            , .m_PushConstantsSize = sizeof(push_constants) }))) return false;
        if (!Ok(Device.Create(m_Instance, { .m_PipeLine = m_Pipeline }))) return false;

        m_bReady = true;
        return true;
    }

    void renderer::Release(void) noexcept
    {
        if (!m_pDevice) return;
        m_pDevice->Destroy(std::move(m_Instance));
        m_pDevice->Destroy(std::move(m_Pipeline));
        m_bReady = false;
    }

    void renderer::Submit(shape Shape, const xmath::fmat4& L2W, const xmath::fvec3& Color) noexcept
    {
        m_DrawList.push_back({ Shape, L2W, Color });
    }

    void renderer::Draw(xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C) noexcept
    {
        if (!m_bReady || m_DrawList.empty()) return;

        CmdBuffer.setPipelineInstance(m_Instance);
        for (auto& Item : m_DrawList)
        {
            auto& Mesh = m_Meshes[static_cast<int>(Item.m_Shape)];
            push_constants PushConstants
            { .m_L2C  = W2C * Item.m_L2W
            , .m_Color = xmath::fvec4(Item.m_Color, 1.0f)
            };
            CmdBuffer.setPushConstants(PushConstants);
            CmdBuffer.setBuffer(Mesh.m_Indices);
            CmdBuffer.setBuffer(Mesh.m_Verts);
            CmdBuffer.Draw(Mesh.m_IndexCount);
        }
        m_DrawList.clear();
    }
}
