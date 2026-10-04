#include "xlionrender_text_renderer.h"
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

namespace xlionrender
{
    inline constexpr std::uint32_t g_TextVertShader[] =
    {
        #include "xlionrender_text_vert.h"
    };
    inline constexpr std::uint32_t g_TextFragShader[] =
    {
        #include "xlionrender_text_frag.h"
    };

    bool text_renderer::Ok(xgpu::device::error* pErr) noexcept
    {
        if (!pErr) return true;
        std::printf("xlionrender::text_renderer: %s\n", std::string(xgpu::getErrorMsg(pErr)).c_str());
        return false;
    }

    bool text_renderer::Init(xgpu::device& Device) noexcept
    {
        if (m_bReady) return true;
        m_pDevice = &Device;

        // Stream 0: the four corners of a unit quad (per vertex). Stream 1: one record per glyph (per instance).
        auto Attributes = std::array
        { xgpu::vertex_descriptor::attribute{ .m_Offset = 0,                                    .m_Format = xgpu::vertex_descriptor::format::FLOAT_2D,            .m_iStream = 0 }
        , xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(instance, m_Rect),           .m_Format = xgpu::vertex_descriptor::format::FLOAT_4D,            .m_iStream = 1, .m_bPerInstance = true }
        , xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(instance, m_UV),             .m_Format = xgpu::vertex_descriptor::format::FLOAT_4D,            .m_iStream = 1, .m_bPerInstance = true }
        , xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(instance, m_Pos),            .m_Format = xgpu::vertex_descriptor::format::FLOAT_4D,            .m_iStream = 1, .m_bPerInstance = true }
        , xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(instance, m_AxisX),          .m_Format = xgpu::vertex_descriptor::format::FLOAT_4D,            .m_iStream = 1, .m_bPerInstance = true }
        , xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(instance, m_AxisY),          .m_Format = xgpu::vertex_descriptor::format::FLOAT_4D,            .m_iStream = 1, .m_bPerInstance = true }
        , xgpu::vertex_descriptor::attribute{ .m_Offset = offsetof(instance, m_Color),          .m_Format = xgpu::vertex_descriptor::format::UINT8_4D_NORMALIZED, .m_iStream = 1, .m_bPerInstance = true }
        };
        if (!Ok(Device.Create(m_VD, xgpu::vertex_descriptor::setup{ .m_bUseStreaming = true, .m_Topology = xgpu::vertex_descriptor::topology::TRIANGLE_LIST, .m_Attributes = Attributes }))) return false;

        xgpu::shader Vert, Frag;
        if (!Ok(Device.Create(Vert, { .m_Type = xgpu::shader::type::bit::VERTEX,   .m_Sharer = xgpu::shader::setup::raw_data{ std::span{ (std::int32_t*)g_TextVertShader, std::size(g_TextVertShader) } } }))) return false;
        if (!Ok(Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ std::span{ (std::int32_t*)g_TextFragShader, std::size(g_TextFragShader) } } }))) return false;

        auto Shaders  = std::array<const xgpu::shader*, 2>{ &Frag, &Vert };
        auto Samplers = std::array{ xgpu::pipeline::sampler{ .m_AddressMode = { xgpu::pipeline::sampler::address_mode::CLAMP, xgpu::pipeline::sampler::address_mode::CLAMP, xgpu::pipeline::sampler::address_mode::CLAMP } } };
        for (int i = 0; i < 2; ++i)
        {
            const bool bTested = i == static_cast<int>(text_depth_mode::DEPTH_TESTED);
            if (!Ok(Device.Create(m_Pipelines[i], xgpu::pipeline::setup
                { .m_VertexDescriptor   = m_VD
                , .m_Shaders            = Shaders
                , .m_PushConstantsSize  = sizeof(push_constants)
                , .m_Samplers           = Samplers
                , .m_Primitive          = { .m_Cull = xgpu::pipeline::primitive::cull::NONE }        // both sides: text seen from behind is mirrored, not missing
                , .m_DepthStencil       = { .m_bDepthTestEnable = bTested, .m_bDepthWriteEnable = false }
                , .m_Blend              = xgpu::pipeline::blend::getAlphaOriginal()
                }))) return false;
        }

        constexpr float Corners[4][2] = { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } };
        constexpr std::uint32_t Indices[6] = { 0, 1, 2, 0, 2, 3 };
        for (auto& Stream : m_Streams)
        {
            if (!Ok(Device.Create(Stream[0], { .m_Type = xgpu::buffer::type::INDEX,  .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = 6 }))) return false;
            if (!Ok(Device.Create(Stream[1], { .m_Type = xgpu::buffer::type::VERTEX, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(Corners[0]),    .m_EntryCount = 4 }))) return false;
            if (!Ok(Device.Create(Stream[2], { .m_Type = xgpu::buffer::type::VERTEX, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(instance),      .m_EntryCount = kMaxInstances }))) return false;
            (void)Stream[0].MemoryMap(0, 6, [&](void* pData) { std::memcpy(pData, Indices, sizeof(Indices)); });
            (void)Stream[1].MemoryMap(0, 4, [&](void* pData) { std::memcpy(pData, Corners, sizeof(Corners)); });
        }

        m_bReady = true;
        return true;
    }

    void text_renderer::Release(void) noexcept
    {
        if (!m_pDevice) return;
        for (auto& [Key, Instance] : m_PipelineInstances) m_pDevice->Destroy(std::move(Instance));
        m_PipelineInstances.clear();
        for (auto& Pipeline : m_Pipelines) m_pDevice->Destroy(std::move(Pipeline));
        m_bReady = false;
    }

    xgpu::pipeline_instance* text_renderer::InstanceOf(xgpu::texture* pTexture, text_depth_mode Depth) noexcept
    {
        const auto Key = std::pair{ pTexture, static_cast<int>(Depth) };
        auto It = m_PipelineInstances.find(Key);
        if (It != m_PipelineInstances.end()) return &It->second;

        auto Bindings = std::array{ xgpu::pipeline_instance::sampler_binding{ *pTexture } };
        xgpu::pipeline_instance Instance;
        if (!Ok(m_pDevice->Create(Instance, { .m_PipeLine = m_Pipelines[static_cast<int>(Depth)], .m_SamplersBindings = Bindings }))) return nullptr;
        return &m_PipelineInstances.emplace(Key, std::move(Instance)).first->second;
    }

    void text_renderer::Submit(const xlioncore::resources::font_view& Font, const text_layout::result& Layout, const text& Tx, const xmath::fvec3& Position, const xmath::fvec3& AxisX, const xmath::fvec3& AxisY) noexcept
    {
        if (!m_bReady || Layout.m_Quads.empty() || !Font.m_pTexture || Tx.m_Opacity <= 0.0f) return;

        const auto Byte = [](float V) noexcept { return static_cast<std::uint8_t>(std::clamp(V, 0.0f, 1.0f) * 255.0f + 0.5f); };
        const bool bScreen = Tx.m_SizeMode == text_size_mode::SCREEN;
        const float Orientation = static_cast<float>(static_cast<int>(Tx.m_Orientation));

        label Label{ Font.m_pTexture, Font.m_pFont, Tx.m_DepthMode, Tx.m_bSort, Position, static_cast<int>(m_Instances.size()), static_cast<int>(Layout.m_Quads.size()) };
        for (const auto& Quad : Layout.m_Quads)
        {
            instance I
            { { Quad.m_Min.m_X, Quad.m_Min.m_Y, Quad.m_Max.m_X, Quad.m_Max.m_Y }
            , { Quad.m_UVMin.m_X, Quad.m_UVMin.m_Y, Quad.m_UVMax.m_X, Quad.m_UVMax.m_Y }
            , { Position.m_X, Position.m_Y, Position.m_Z, Tx.m_Size }
            , { AxisX.m_X, AxisX.m_Y, AxisX.m_Z, Orientation }
            , { AxisY.m_X, AxisY.m_Y, AxisY.m_Z, bScreen ? 1.0f : 0.0f }
            , { Byte(Tx.m_Color.m_X), Byte(Tx.m_Color.m_Y), Byte(Tx.m_Color.m_Z), Byte(Tx.m_Opacity) }
            };
            m_Instances.push_back(I);
        }
        m_Labels.push_back(Label);
    }

    void text_renderer::Draw(xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH) noexcept
    {
        m_LastStats = {};
        if (!m_bReady || m_Labels.empty()) { m_Labels.clear(); m_Instances.clear(); return; }

        // The order labels are drawn in: the ones that did not ask to be sorted by font and depth mode, then the sorted ones from far to near.
        std::vector<int> Order(m_Labels.size());
        for (int i = 0, end = static_cast<int>(Order.size()); i < end; ++i) Order[i] = i;
        std::vector<float> Distance(m_Labels.size());
        for (std::size_t i = 0; i < m_Labels.size(); ++i)
            Distance[i] = (W2C * xmath::fvec4(m_Labels[i].m_Position.m_X, m_Labels[i].m_Position.m_Y, m_Labels[i].m_Position.m_Z, 1.0f)).m_W;      // along the view direction
        std::stable_sort(Order.begin(), Order.end(), [&](int A, int B) noexcept
        {
            const auto& LA = m_Labels[A];
            const auto& LB = m_Labels[B];
            if (LA.m_bSort != LB.m_bSort) return !LA.m_bSort;
            if (LA.m_bSort) return Distance[A] > Distance[B];
            if (LA.m_pTexture != LB.m_pTexture) return LA.m_pTexture < LB.m_pTexture;
            return LA.m_Depth < LB.m_Depth;
        });

        m_Slot = (m_Slot + 1) % kRing;
        auto& Stream = m_Streams[m_Slot];

        std::vector<run> Runs;
        int nCopied = 0;
        (void)Stream[2].MemoryMap(0, kMaxInstances, [&](void* pData)
        {
            auto* pOut = static_cast<instance*>(pData);
            for (const int iLabel : Order)
            {
                const auto& L = m_Labels[iLabel];
                if (nCopied + L.m_Count > kMaxInstances) { ++m_LastStats.m_Dropped; continue; }
                std::memcpy(pOut + nCopied, m_Instances.data() + L.m_First, static_cast<std::size_t>(L.m_Count) * sizeof(instance));
                if (!Runs.empty() && Runs.back().m_pTexture == L.m_pTexture && Runs.back().m_Depth == L.m_Depth) Runs.back().m_Count += L.m_Count;
                else Runs.push_back({ L.m_pTexture, L.m_pFont, L.m_Depth, nCopied, L.m_Count });
                nCopied += L.m_Count;
                ++m_LastStats.m_Labels;
            }
        });

        for (const auto& Run : Runs)
        {
            auto* pInstance = InstanceOf(Run.m_pTexture, Run.m_Depth);
            if (!pInstance) continue;
            push_constants PC
            { W2C
            , xmath::fvec4(ViewportW, ViewportH, 0.0f, 0.0f)
            , xmath::fvec4(Run.m_pFont->m_PixelRange, static_cast<float>(static_cast<int>(Run.m_pFont->m_OutputType)), 0.0f, 0.0f)
            };
            CmdBuffer.setPipelineInstance(*pInstance);
            CmdBuffer.setStreamingBuffers(Stream);
            CmdBuffer.setPushConstants(PC);
            CmdBuffer.DrawInstance(Run.m_Count, 6, Run.m_First);
            ++m_LastStats.m_Draws;
        }
        m_LastStats.m_Glyphs = nCopied;

        m_Labels.clear();
        m_Instances.clear();
    }
}
