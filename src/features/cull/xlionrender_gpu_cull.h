#pragma once
// GPU cluster culling (documentation/gpu_cluster_culling.md): the main view's opaque rigid clusters, a candidate each, culled by
// shaders/xlionrender_cull_comp.glsl into runs of indirect draws - a command and a record (the vertex shader's push constants) each;
// instance::renderScenePass draws a run with one DrawIndexedIndirectCount
#include "plans/forward/xlionrender_plan_forward.h"
#include <cstring>
#include <span>

namespace xlionrender
{
    class gpu_cull
    {
    public:

        static constexpr int max_candidates_v  = 1 << 16;     // ponytail: a frame's cap; past it the clusters take the CPU path
        static constexpr int max_draws_v       = 1 << 14;     // their draws' local to clip
        static constexpr int max_runs_v        = 1 << 14;
        static constexpr int ring_frames_v     = 3;           // (instance::ring_frames_v)

        struct candidate        // std430 (xlionrender_cull_comp.glsl)
        {
            std::uint32_t               m_Run, m_RunFirst, m_PrepassRun = ~0u, m_PrepassFirst = 0;
            std::uint32_t               m_Draw, m_nIndices, m_iFirstIndex;
            std::int32_t                m_VertexOffset;
            details::push_static_geom   m_Record;
            std::uint32_t               m_Pad[2]{};
            float                       m_Center[4], m_Half[4];
        };
        static_assert(sizeof(candidate) == 96);

        // (its GPU objects go with it: xGPU defers their release)
        bool Init( xgpu::device& Device, std::span<const std::uint32_t> SPIRV ) noexcept
        {
            const auto Raw = xgpu::shader::setup::raw_data{ std::span{ reinterpret_cast<const std::int32_t*>(SPIRV.data()), SPIRV.size() } };
            using usage = xgpu::buffer::setup::usage;
            const std::vector<std::uint32_t> Zeros(std::size_t(2) * max_candidates_v * sizeof(details::push_static_geom) / sizeof(std::uint32_t), 0u);    // (a device-local buffer is made from its first data)
            return Device.hasDrawIndirectCount()
                && !Device.Create(m_Shader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = Raw })
                && !Device.Create(m_Compute, { .m_pShader = &m_Shader, .m_nBuffers = 5, .m_PushConstantsSize = 16 })
                && !Device.Create(m_Candidates, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(candidate), .m_EntryCount = ring_frames_v * max_candidates_v })
                && !Device.Create(m_Matrices, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(xmath::fmat4), .m_EntryCount = ring_frames_v * max_draws_v })
                && !Device.Create(m_Zeros, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = max_runs_v, .m_pData = const_cast<std::uint32_t*>(Zeros.data()) })
                && !Device.Create(m_CountsHost, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = ring_frames_v * max_runs_v })
                && !Device.Create(m_Counts, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = max_runs_v, .m_pData = const_cast<std::uint32_t*>(Zeros.data()) })
                // a run's commands: its candidates' room (a candidate is in two runs: its own, its prepass's)
                && !Device.Create(m_Commands, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(details::draw_indexed_indirect), .m_EntryCount = 2 * max_candidates_v, .m_pData = const_cast<std::uint32_t*>(Zeros.data()) })
                && !Device.Create(m_Records, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(details::push_static_geom), .m_EntryCount = 2 * max_candidates_v, .m_pData = const_cast<std::uint32_t*>(Zeros.data()) });
        }

        // This frame's candidates and their draws' local to clip into its region, the runs' counts zeroed, the cull (outside a render pass)
        // Prepass: per run, whether it is a prepass run (the stats count the others' draws)
        void Cull( xgpu::window& Window, std::uint64_t Frame, std::span<const candidate> Candidates, std::span<const xmath::fmat4> L2C, std::span<const std::uint8_t> Prepass ) noexcept
        {
            const int Slot = int(Frame % ring_frames_v);
            // What the GPU kept the last time this slot was used (that frame is done): its runs' counts
            if( !m_SlotPrepass[Slot].empty() )
            {
                const auto& P = m_SlotPrepass[Slot];
                m_Drawn = m_PrepassDrawn = 0;
                (void)m_CountsHost.MemoryMapFrameRegion(Slot * max_runs_v, int(P.size()), [&](void* p)
                {
                    const auto* pC = static_cast<const std::uint32_t*>(p);
                    for( std::size_t r = 0; r < P.size(); ++r ) (P[r] ? m_PrepassDrawn : m_Drawn) += int(pC[r]);
                });
            }
            m_SlotPrepass[Slot].assign(Prepass.begin(), Prepass.end());
            if( Candidates.empty() ) return;
            const int nRuns = int(Prepass.size());
            (void)m_Candidates.MemoryMapFrameRegion(Slot * max_candidates_v, int(Candidates.size()), [&](void* p) { std::memcpy(p, Candidates.data(), Candidates.size_bytes()); });
            (void)m_Matrices.MemoryMapFrameRegion(Slot * max_draws_v, int(L2C.size()), [&](void* p) { std::memcpy(p, L2C.data(), L2C.size_bytes()); });
            const xgpu::buffer::copy Z{ 0, 0, nRuns };
            Window.CopyBuffer(m_Zeros, m_Counts, std::span{ &Z, 1 });
            const std::uint32_t Push[4] = { std::uint32_t(Slot * max_candidates_v), std::uint32_t(Candidates.size()), std::uint32_t(Slot * max_draws_v), 0u };
            auto Buffers = std::array<xgpu::buffer*, 5>{ &m_Candidates, &m_Matrices, &m_Counts, &m_Commands, &m_Records };
            Window.Dispatch(m_Compute, Buffers, Push, sizeof(Push), (int(Candidates.size()) + 63) / 64);
            const xgpu::buffer::copy C{ 0, Slot * max_runs_v, nRuns };      // (read when the slot comes round again)
            Window.CopyBuffer(m_Counts, m_CountsHost, std::span{ &C, 1 });
        }

        // The clusters the cull kept, a few frames late (ring_frames_v): drawn shaded, drawn in the depth prepass
        int                 m_Drawn = 0, m_PrepassDrawn = 0;

        xgpu::buffer        m_Commands, m_Records, m_Counts;        // the scene pass's: the runs' commands, their records (binding 12), a count a run

    private:

        xgpu::shader        m_Shader;
        xgpu::compute       m_Compute;
        xgpu::buffer        m_Candidates, m_Matrices, m_Zeros, m_CountsHost;
        std::array<std::vector<std::uint8_t>, ring_frames_v> m_SlotPrepass;
    };
}
