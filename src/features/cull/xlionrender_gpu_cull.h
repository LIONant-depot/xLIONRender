#pragma once
// GPU cluster culling (documentation/gpu_cluster_culling.md): the main view's opaque rigid clusters, a candidate each, culled by
// shaders/xlionrender_cull_comp.glsl into runs of indirect draws - a command and a record (the vertex shader's push constants) each;
// instance::renderScenePass draws a run with one DrawIndexedIndirectCount. Occlusion in two passes: Cull keeps what was visible last
// frame (pass 1); the scene pass lays its depth; Occlude makes the Hi-Z pyramid of it (shaders/xlionrender_hiz_comp.glsl, one dispatch,
// the depth read where the pass left it) and keeps what is visible now and pass 1 did not draw (pass 2). Adaptive: skipped while it hides little
#include "plans/forward/xlionrender_plan_forward.h"
#include <cstring>
#include <span>
#include <unordered_map>

namespace xlionrender
{
    class gpu_cull
    {
    public:

        static constexpr int max_candidates_v  = 1 << 16;     // ponytail: a frame's cap; past it the clusters take the CPU path
        static constexpr int max_draws_v       = 1 << 14;     // their draws' local to clip
        static constexpr int max_runs_v        = 1 << 14;
        static constexpr int max_vis_v         = 1 << 20;     // visibility entries (a cluster of an instance slot each)
        static constexpr int ring_frames_v     = 3;           // (instance::ring_frames_v)
        static constexpr int counts_v          = 2 * max_runs_v + 3;     // a count a run and pass (pass 2's at max_runs_v); the triangles shaded, pass 2's tested and occluded

        struct candidate        // std430 (xlionrender_cull_comp.glsl)
        {
            std::uint32_t               m_Run, m_RunFirst, m_PrepassRun = ~0u, m_PrepassFirst = 0;
            std::uint32_t               m_Draw, m_nIndices, m_iFirstIndex;
            std::int32_t                m_VertexOffset;
            details::push_static_geom   m_Record;
            std::uint32_t               m_VisKey = 0;           // its visibility's entry (VisKey)
            std::uint32_t               m_Pad    = 0;
            float                       m_Center[4], m_Half[4];
        };
        static_assert(sizeof(candidate) == 96);

        // (its GPU objects go with it: xGPU defers their release)
        bool Init( xgpu::device& Device, std::span<const std::uint32_t> CullSPIRV, std::span<const std::uint32_t> HiZSPIRV ) noexcept
        {
            const auto Raw = []( std::span<const std::uint32_t> W ) { return xgpu::shader::setup::raw_data{ std::span{ reinterpret_cast<const std::int32_t*>(W.data()), W.size() } }; };
            using usage = xgpu::buffer::setup::usage;
            const std::vector<std::uint32_t> Zeros(std::size_t(4) * max_candidates_v * sizeof(details::push_static_geom) / sizeof(std::uint32_t), 0u);    // (a device-local buffer is made from its first data)
            auto* pZeros = const_cast<std::uint32_t*>(Zeros.data());
            return Device.hasDrawIndirectCount()
                && !Device.Create(m_Shader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = Raw(CullSPIRV) })
                && !Device.Create(m_Compute, { .m_pShader = &m_Shader, .m_nBuffers = 7, .m_PushConstantsSize = 32 })
                && !Device.Create(m_HiZShader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = Raw(HiZSPIRV) })
                && !Device.Create(m_HiZCompute, { .m_pShader = &m_HiZShader, .m_nBuffers = 2, .m_nTextures = 1, .m_PushConstantsSize = 32 })
                && !Device.Create(m_HiZCounter, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = 1, .m_pData = pZeros })
                && !Device.Create(m_Times, { .m_Count = 4 })
                && !Device.Create(m_Candidates, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(candidate), .m_EntryCount = ring_frames_v * max_candidates_v })
                && !Device.Create(m_Matrices, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(xmath::fmat4), .m_EntryCount = ring_frames_v * max_draws_v })
                && !Device.Create(m_Zeros, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = counts_v, .m_pData = pZeros })
                // a count a run and pass (pass 2's at max_runs_v), read back a ring later (a region of both a ring frame)
                && !Device.Create(m_Counts, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = counts_v, .m_pData = pZeros })
                && !Device.Create(m_CountsHost, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = ring_frames_v * counts_v })
                && !Device.Create(m_Vis, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(std::uint32_t), .m_EntryCount = max_vis_v, .m_pData = pZeros })
                // a run's commands: its candidates' room, a candidate in two runs (its own, its prepass's), in each pass
                && !Device.Create(m_Commands, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(details::draw_indexed_indirect), .m_EntryCount = 4 * max_candidates_v, .m_pData = pZeros })
                && !Device.Create(m_Records, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = usage::GPU_READ, .m_EntryByteSize = sizeof(details::push_static_geom), .m_EntryCount = 4 * max_candidates_v, .m_pData = pZeros });
        }

        // An instance slot's first visibility entry (its cluster c: + c), kept while the slot keeps its cluster count. Full: every slot
        // starts again (a frame of mostly pass 2 - never a wrong picture: a cluster's visibility only picks the pass that draws it)
        std::uint32_t VisKey( std::uint32_t Slot, std::uint32_t nClusters ) noexcept
        {
            auto& E = m_VisOf[Slot];
            if( E.second == nClusters ) return E.first;
            if( m_VisNext + nClusters > std::uint32_t(max_vis_v) ) { m_VisOf.clear(); m_VisNext = 0; return VisKey(Slot, nClusters); }
            E = { m_VisNext, nClusters };
            m_VisNext += nClusters;
            return E.first;
        }

        // The Hi-Z pyramid's floats for a W x H view (level 0: half its size; xlionrender_hiz_comp.glsl)
        static int HiZFloats( int W, int H ) noexcept
        {
            int n = 0;
            W = (W + 1) / 2; H = (H + 1) / 2;
            for( ;; ) { n += W * H; if( W == 1 && H == 1 ) return n; W = (W + 1) / 2; H = (H + 1) / 2; }
        }

        // A frame's start (before WantOcclusion and Cull): what the GPU did the last time this frame's slot was used (that frame is done) -
        // its runs' counts, both passes, the occlusion's numbers and its GPU times
        void Begin( std::uint64_t Frame ) noexcept
        {
            m_Slot = int(Frame % ring_frames_v);
            if( !m_SlotPrepass[m_Slot].empty() )
            {
                const auto& P = m_SlotPrepass[m_Slot];
                m_Drawn = m_PrepassDrawn = m_Drawn2 = 0;
                std::uint32_t Tested = 0, Occluded = 0;
                (void)m_CountsHost.MemoryMapFrameRegion(m_Slot * counts_v, counts_v, [&](void* p)
                {
                    const auto* pC = static_cast<const std::uint32_t*>(p);
                    m_Triangles = std::int64_t(pC[2 * max_runs_v]);
                    Tested = pC[2 * max_runs_v + 1]; Occluded = pC[2 * max_runs_v + 2];
                    for( std::size_t r = 0; r < P.size(); ++r )
                    {
                        (P[r] ? m_PrepassDrawn : m_Drawn) += int(pC[r] + pC[max_runs_v + r]);
                        if( !P[r] ) m_Drawn2 += int(pC[max_runs_v + r]);
                    }
                });
                // Adaptive occlusion: a frame that tested and hid little counts toward skipping it
                if( m_SlotOccluded[m_Slot] && Tested > 0 )
                {
                    m_OccludedShare = float(Occluded) / float(Tested);
                    m_nLowFrames    = m_OccludedShare < low_share_v ? m_nLowFrames + 1 : 0;
                    if( m_nLowFrames >= low_frames_v ) { m_nSkipFrames = skip_frames_v; m_nLowFrames = 0; }
                }
            }
            if( std::vector<std::uint64_t> Ns; m_Times.getResults(Ns) && Ns.size() >= 4 )
            {
                const auto Ms = [&]( int a, int b ) { return Ns[a] != ~0ull && Ns[b] != ~0ull && Ns[b] >= Ns[a] ? double(Ns[b] - Ns[a]) * 1e-6 : 0.0; };
                m_CullGpuMs = Ms(0, 1); m_OccludeGpuMs = Ms(2, 3);
            }
        }

        // Whether this frame runs the occlusion (the Hi-Z and pass 2): not while the last frames hid under low_share_v of what they tested
        // (skip_frames_v frames, then it tries again); a camera cut tries again at once
        bool WantOcclusion( bool bCameraCut ) noexcept
        {
            if( bCameraCut ) m_nSkipFrames = 0;
            if( m_nSkipFrames > 0 ) { --m_nSkipFrames; return false; }
            return true;
        }
        bool isOcclusionSkipped( void ) const noexcept { return m_nSkipFrames > 0; }

        // This frame's candidates and their draws' local to clip into its region, the runs' counts zeroed, pass 1 (outside a render pass):
        // bOcclusion, what was visible last frame (Occlude follows); else every cluster in the view. Prepass: per run, whether it is a
        // prepass run (the stats); Second: pass 2's first command (pass 1's runs' room)
        void Cull( xgpu::window& Window, std::span<const candidate> Candidates, std::span<const xmath::fmat4> L2C, std::span<const std::uint8_t> Prepass, std::uint32_t Second, bool bOcclusion ) noexcept
        {
            m_SlotPrepass[m_Slot].assign(Prepass.begin(), Prepass.end());
            m_SlotOccluded[m_Slot] = bOcclusion;
            m_nCandidates = int(Candidates.size());
            m_Second      = Second;
            if( Candidates.empty() ) return;
            const int nRuns = int(Prepass.size());
            Window.WriteTimestamp(m_Times, 0);
            (void)m_Candidates.MemoryMapFrameRegion(m_Slot * max_candidates_v, m_nCandidates, [&](void* p) { std::memcpy(p, Candidates.data(), Candidates.size_bytes()); });
            (void)m_Matrices.MemoryMapFrameRegion(m_Slot * max_draws_v, int(L2C.size()), [&](void* p) { std::memcpy(p, L2C.data(), L2C.size_bytes()); });
            const std::array Z{ xgpu::buffer::copy{ 0, 0, nRuns }, xgpu::buffer::copy{ max_runs_v, max_runs_v, nRuns }, xgpu::buffer::copy{ 2 * max_runs_v, 2 * max_runs_v, 3 } };
            Window.CopyBuffer(m_Zeros, m_Counts, Z);
            Dispatch(Window, bOcclusion ? 0u : 2u, m_Zeros, 0, 0);
            Window.WriteTimestamp(m_Times, 1);
            if( !bOcclusion ) CountsBack(Window);
        }

        // Pass 2: the pyramid of Depth (the view's, W x H, as pass 1 left it) into HiZ (HiZFloats) in one dispatch, then every cluster in the
        // view against it
        void Occlude( xgpu::window& Window, const xgpu::texture& Depth, xgpu::buffer& HiZ, int W, int H ) noexcept
        {
            if( m_nCandidates == 0 ) return;
            Window.WriteTimestamp(m_Times, 2);
            const int W0 = (W + 1) / 2, H0 = (H + 1) / 2, GX = (W0 + 63) / 64, GY = (H0 + 63) / 64;
            std::uint32_t nLevels = 1;
            for( int w = W0, h = H0; w > 1 || h > 1; w = (w + 1) / 2, h = (h + 1) / 2 ) ++nLevels;
            const std::uint32_t Push[8] = { std::uint32_t(W), std::uint32_t(H), std::uint32_t(GX), std::uint32_t(GX * GY), nLevels, 0u, 0u, 0u };
            auto Buffers  = std::array<xgpu::buffer*, 2>{ &HiZ, &m_HiZCounter };
            auto Textures = std::array<const xgpu::texture*, 1>{ &Depth };
            Window.Dispatch(m_HiZCompute, Buffers, Textures, Push, sizeof(Push), GX, GY);
            Dispatch(Window, 1u, HiZ, W, H);
            Window.WriteTimestamp(m_Times, 3);
            CountsBack(Window);
        }

        // The clusters the cull kept, a few frames late (ring_frames_v): drawn shaded (both passes), in the depth prepass, shaded by pass 2
        int                 m_Drawn = 0, m_PrepassDrawn = 0, m_Drawn2 = 0;
        std::int64_t        m_Triangles = 0;            // the triangles shaded
        float               m_OccludedShare = 0;        // of the triangles pass 2 tested, the share it hid (adaptive occlusion)
        double              m_CullGpuMs = 0, m_OccludeGpuMs = 0;     // the GPU's time: pass 1's cull; the pyramid and pass 2's cull

        xgpu::buffer        m_Commands, m_Records, m_Counts;        // the scene pass's: the runs' commands, their records (binding 12), a count a run and pass

    private:

        static constexpr float low_share_v   = 0.10f;   // adaptive occlusion: a frame hiding less than this share of the triangles it tested is "low",
        static constexpr int   low_frames_v  = 30;      //   this many low frames in a row skip it for
        static constexpr int   skip_frames_v = 120;     //   this many frames

        void Dispatch( xgpu::window& Window, std::uint32_t Mode, xgpu::buffer& HiZ, int W, int H ) noexcept
        {
            const std::uint32_t Push[8] = { std::uint32_t(m_Slot * max_candidates_v), std::uint32_t(m_nCandidates), std::uint32_t(m_Slot * max_draws_v), Mode
                                          , m_Second, std::uint32_t(max_runs_v), std::uint32_t(W), std::uint32_t(H) };
            auto Buffers = std::array<xgpu::buffer*, 7>{ &m_Candidates, &m_Matrices, &m_Counts, &m_Commands, &m_Records, &m_Vis, &HiZ };
            Window.Dispatch(m_Compute, Buffers, Push, sizeof(Push), (m_nCandidates + 63) / 64);
        }

        void CountsBack( xgpu::window& Window ) noexcept        // (read when the slot comes round again)
        {
            const xgpu::buffer::copy C{ 0, m_Slot * counts_v, counts_v };
            Window.CopyBuffer(m_Counts, m_CountsHost, std::span{ &C, 1 });
        }

        xgpu::shader        m_Shader, m_HiZShader;
        xgpu::compute       m_Compute, m_HiZCompute;
        xgpu::buffer        m_Candidates, m_Matrices, m_Zeros, m_CountsHost, m_Vis, m_HiZCounter;
        xgpu::timestamps    m_Times;
        std::array<std::vector<std::uint8_t>, ring_frames_v> m_SlotPrepass;
        std::array<bool, ring_frames_v>                      m_SlotOccluded{};
        std::unordered_map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> m_VisOf;    // instance slot: its first entry, its clusters
        std::uint32_t       m_VisNext = 0;
        int                 m_Slot = 0, m_nCandidates = 0;
        std::uint32_t       m_Second = 0;
        int                 m_nLowFrames = 0, m_nSkipFrames = 0;
    };
}
