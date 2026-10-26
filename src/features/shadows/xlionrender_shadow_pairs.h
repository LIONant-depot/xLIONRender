#pragma once
// The sun's page/caster pairs on the GPU (documentation/lighting_strategy.md P1c): this frame's pages to render and the rigid opaque
// clusters of the casters that may touch them, a candidate each; shaders/xlionrender_shadow_pairs_comp.glsl pairs each cluster with the
// pages its light-space box overlaps (a record each) and appends one instanced indirect command to the cluster's run (its geometry);
// instance::renderShadows draws a run with one DrawIndexedIndirectCount over the whole sun's atlas (xlionrender_shadow_pairs_vert.glsl
// puts each instance in its page). The CPU records a draw a run instead of a draw a cluster and page.
// ponytail: a thread a cluster loops over every page of the frame (a few hundred: 27k clusters x 188 pages in the benchmark); a light-space
// grid of the pages when an epoch's thousands of pages show in the GPU time
#include "plans/forward/xlionrender_plan_forward.h"
#include <cstring>
#include <span>

namespace xlionrender
{
    class shadow_pairs
    {
    public:

        static constexpr int max_candidates_v = 1 << 17;     // a frame's caps; past them the frame takes the CPU path
        static constexpr int max_draws_v      = 1 << 14;
        static constexpr int max_pages_v      = 1 << 12;
        static constexpr int max_runs_v       = 1 << 12;
        static constexpr int max_records_v    = 1 << 21;     // the pairs (past it: dropped, counted - the pages rendered that frame go again)
        static constexpr int ring_frames_v    = 3;           // (instance::ring_frames_v)
        static constexpr int counts_v         = max_runs_v + 3;

        struct candidate        // std430 (xlionrender_shadow_pairs_comp.glsl)
        {
            std::uint32_t   m_Run, m_RunFirst, m_Cluster, m_Slot;
            std::uint32_t   m_nIndices, m_iFirstIndex;
            std::int32_t    m_VertexOffset;
            std::uint32_t   m_Draw;                 // | mover_v: a mover's
            float           m_Center[4], m_Half[4];
        };
        static_assert(sizeof(candidate) == 64);
        static constexpr std::uint32_t mover_v = 0x80000000u;

        struct page             // std430 (both shaders)
        {
            xmath::fmat4    m_W2C;
            float           m_Atlas[4];             // the page's scale and offset in the sun atlas's clip space
            float           m_Rect[4];              // light space: min x, min y, size; 1: a mover page
        };
        static_assert(sizeof(page) == 96);

        bool Init( xgpu::device& Device, std::span<const std::uint32_t> SPIRV ) noexcept
        {
            using usage = xgpu::buffer::setup::usage;
            const std::vector<std::uint32_t> Zeros(std::size_t(3) * max_records_v, 0u);     // (a device-local buffer is made from its first data)
            auto* pZeros = const_cast<std::uint32_t*>(Zeros.data());
            return Device.hasDrawIndirectCount()
                && !Device.Create(m_Shader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = xgpu::shader::setup::raw_data{ std::span{ reinterpret_cast<const std::int32_t*>(SPIRV.data()), SPIRV.size() } } })
                && !Device.Create(m_Compute, { .m_pShader = &m_Shader, .m_nBuffers = 6, .m_PushConstantsSize = 32 })
                && !Device.Create(m_Candidates, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(candidate), .m_EntryCount = ring_frames_v * max_candidates_v })
                && !Device.Create(m_Matrices, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(xmath::fmat4), .m_EntryCount = ring_frames_v * max_draws_v })
                && !Device.Create(m_Pages, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(page), .m_EntryCount = ring_frames_v * max_pages_v })
                && !Device.Create(m_Zeros, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = counts_v, .m_pData = pZeros })
                && !Device.Create(m_Counts, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = counts_v, .m_pData = pZeros })
                && !Device.Create(m_CountsHost, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = ring_frames_v * counts_v })
                && !Device.Create(m_Commands, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(details::draw_indexed_indirect), .m_EntryCount = max_candidates_v, .m_pData = pZeros })
                && !Device.Create(m_Records, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = 3 * sizeof(std::uint32_t), .m_EntryCount = max_records_v, .m_pData = pZeros });
        }

        // A frame's start: what the GPU did the last time this frame's slot was used (that frame is done)
        void Begin( std::uint64_t Frame ) noexcept
        {
            m_Slot = int(Frame % ring_frames_v);
            m_bDropped = false;
            if( !m_bSlotUsed[m_Slot] ) return;
            m_bSlotUsed[m_Slot] = false;
            (void)m_CountsHost.MemoryMapFrameRegion(m_Slot * counts_v + max_runs_v, 3, [&](void* p)
            {
                const auto* pC = static_cast<const std::uint32_t*>(p);
                m_Pairs = int(pC[0]); m_Dropped = int(pC[1]); m_Triangles = std::int64_t(pC[2]);
            });
            m_bDropped = m_Dropped > 0;
        }

        // Whether a frame's lists fit (else the CPU path takes it)
        static bool Fits( std::size_t nCandidates, std::size_t nDraws, std::size_t nPages, std::size_t nRuns ) noexcept
        {
            return nCandidates <= std::size_t(max_candidates_v) && nDraws <= std::size_t(max_draws_v) && nPages <= std::size_t(max_pages_v) && nRuns <= std::size_t(max_runs_v);
        }

        // This frame's pages, candidates and their draws' local to light into its region, the counts zeroed, the pairs made (outside a render pass)
        void Pair( xgpu::window& Window, std::span<const candidate> Candidates, std::span<const xmath::fmat4> L2L, std::span<const page> Pages, std::size_t nRuns ) noexcept
        {
            m_bSlotUsed[m_Slot] = true;
            m_nCandidates = int(Candidates.size());
            (void)m_Pages.MemoryMapFrameRegion(m_Slot * max_pages_v, int(Pages.size()), [&](void* p) { std::memcpy(p, Pages.data(), Pages.size_bytes()); });
            if( Candidates.empty() ) return;
            (void)m_Candidates.MemoryMapFrameRegion(m_Slot * max_candidates_v, m_nCandidates, [&](void* p) { std::memcpy(p, Candidates.data(), Candidates.size_bytes()); });
            (void)m_Matrices.MemoryMapFrameRegion(m_Slot * max_draws_v, int(L2L.size()), [&](void* p) { std::memcpy(p, L2L.data(), L2L.size_bytes()); });
            const std::array Z{ xgpu::buffer::copy{ 0, 0, int(nRuns) }, xgpu::buffer::copy{ max_runs_v, max_runs_v, 3 } };
            Window.CopyBuffer(m_Zeros, m_Counts, Z);
            const std::uint32_t Push[8] = { std::uint32_t(m_Slot * max_candidates_v), std::uint32_t(m_nCandidates), std::uint32_t(m_Slot * max_draws_v), std::uint32_t(m_Slot * max_pages_v)
                                          , std::uint32_t(Pages.size()), std::uint32_t(max_records_v), std::uint32_t(max_runs_v), 0u };
            auto Buffers = std::array<xgpu::buffer*, 6>{ &m_Candidates, &m_Matrices, &m_Pages, &m_Counts, &m_Commands, &m_Records };
            Window.Dispatch(m_Compute, Buffers, Push, sizeof(Push), (m_nCandidates + 63) / 64);
            const xgpu::buffer::copy C{ max_runs_v, m_Slot * counts_v + max_runs_v, 3 };        // (read when the slot comes round again)
            Window.CopyBuffer(m_Counts, m_CountsHost, std::span{ &C, 1 });
        }

        // The vertex shader's first page of this frame (its records name pages from it)
        std::uint32_t PageBase( void ) const noexcept { return std::uint32_t(m_Slot * max_pages_v); }

        // A ring late: the pairs made, those dropped past max_records_v, the triangles drawn into the pages
        int                 m_Pairs = 0, m_Dropped = 0;
        std::int64_t        m_Triangles = 0;
        bool                m_bDropped = false;     // this frame read a frame that dropped pairs (its pages are not complete)
        int                 m_nCandidates = 0;

        xgpu::buffer        m_Commands, m_Records, m_Counts, m_Pages;

    private:

        xgpu::shader        m_Shader;
        xgpu::compute       m_Compute;
        xgpu::buffer        m_Candidates, m_Matrices, m_Zeros, m_CountsHost;
        std::array<bool, ring_frames_v> m_bSlotUsed{};
        int                 m_Slot = 0;
    };
}
