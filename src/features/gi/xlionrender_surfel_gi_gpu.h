#ifndef XLIONRENDER_SURFEL_GI_GPU_H
#define XLIONRENDER_SURFEL_GI_GPU_H

// Part II, G3 and G4 (documentation/gi_dynamic_master_plan.md): the GI on the GPU, over time as the CPU's trace was. The scene and the
// light live on the GPU (world::m_Setup.m_bGPUOwned): the scene is packed again (world::PackGPU, PackGPUSurfels) only when the world's
// structure changes (world::GPUGen) or the number of point lights does, the light moved from the last layout into the new one on the GPU
// (shaders/xlionrender_gi_migrate_comp.glsl: a level switch re-levels it there). A batch is the world's pick as instance ranges
// (world::GPURanges, a few frames of rays: BatchRays); every frame the next slice of them that fits the frame's GPU budget is expanded
// into surfels (xlionrender_gi_expand_comp.glsl: the tiers, the adaptive rays), traced (xlionrender_gi_surfel_pass_comp.glsl) and
// committed (xlionrender_gi_commit_comp.glsl); the ambient cubes are expanded (_cube_comp), the instances' means summed (_means_comp),
// and the frame after the slice's ranges and the means go back to the world (world::DeliverGPURanges, SetGPUMeans): nothing per
// surfel on the CPU. Before each slice the world's changes go up: its estimator changes as ops (world::TakeGPUOps, _ops_comp), the
// links' emitted light that changed, the means, the setup and the lights (_scatter_comp). The shown light is made on the GPU (Show,
// _filter_comp), straight into the renderer's ring. ReadBack fills the world's arrays from the GPU's light (its tests, its dump)
#include "xlionrender_surfel_gi.h"
#include <span>

namespace xlionrender::gi
{
    class gpu_trace
    {
    public:

        // The SPIR-V of the shaders (shaders/xlionrender_gi_*_comp.glsl)
        struct shaders { std::span<const std::uint32_t> m_Pass, m_Scatter, m_Commit, m_Ops, m_Filter, m_Expand, m_Means, m_Cube, m_Migrate, m_ReadBack, m_Links, m_Solve, m_Ghosts, m_Probes; };     // (m_Links, m_Solve: optional, experiment D)

        bool Init(xgpu::device& Device, const shaders& S) noexcept
        {
            m_pDevice = &Device;
            const auto Make = [&](xgpu::shader& Sh, xgpu::compute& C, std::span<const std::uint32_t> W, int nBuffers, int Push)
            {
                const auto Raw = xgpu::shader::setup::raw_data{ std::span{ reinterpret_cast<const std::int32_t*>(W.data()), W.size() } };
                return !Device.Create(Sh, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = Raw }) && !Device.Create(C, { .m_pShader = &Sh, .m_nBuffers = nBuffers, .m_PushConstantsSize = std::size_t(Push) });
            };
            return Make(m_Shader[0], m_Pass, S.m_Pass, 8, 16) && Make(m_Shader[1], m_Scatter, S.m_Scatter, 2, 16) && Make(m_Shader[2], m_Commit, S.m_Commit, 5, 16)
                && Make(m_Shader[3], m_OpsC, S.m_Ops, 2, 16) && Make(m_Shader[4], m_Filter, S.m_Filter, 6, 32) && Make(m_Shader[5], m_Expand, S.m_Expand, 4, 32)
                && Make(m_Shader[6], m_Means, S.m_Means, 2, 16) && Make(m_Shader[7], m_Cube, S.m_Cube, 2, 16) && Make(m_Shader[8], m_Migrate, S.m_Migrate, 3, 16)
                && Make(m_Shader[9], m_ReadBackC, S.m_ReadBack, 2, 16) && !Device.Create(m_Times, { .m_Count = 2 })
                && (S.m_Links.empty() || (m_bLinks = Make(m_Shader[10], m_LinksC, S.m_Links, 15, 32)) || true)
                && (S.m_Solve.empty() || (m_bSolve = Make(m_Shader[11], m_SolveC, S.m_Solve, 7, 16)) || true)
                && (S.m_Ghosts.empty() || (m_bGhosts = Make(m_Shader[12], m_GhostsC, S.m_Ghosts, 2, 16)) || true)
                && (S.m_Probes.empty() || (m_bProbes = Make(m_Shader[13], m_ProbesC, S.m_Probes, 6, 32)) || true);
        }

        // Before world::Update: what the GPU wrote two frames ago or earlier (that frame is done: xGPU keeps one in flight, its EndFrame
        // waited for the one before) into the world - the means, the oldest slice's stats; never a wait for the frame in flight. A read
        // back asked for, into its arrays. true: a slice was delivered
        static constexpr std::uint64_t lag_frames_v = 2;
        bool Collect(world& W, std::uint64_t Frame) noexcept
        {
            if (m_ReadFrame != ~0ull && Frame > m_ReadFrame) FinishReadBack(W);
            for (int Pass = 0; Pass < 2; ++Pass)            // (both done: the older first, the newer last)
            {
                int k = -1;
                for (int j = 0; j < 2; ++j)
                    if (m_MeansFrame[j] != ~0ull && Frame >= m_MeansFrame[j] + lag_frames_v && (k < 0 || m_MeansFrame[j] < m_MeansFrame[k])) k = j;
                if (k < 0) break;
                m_MeansOut.resize(std::size_t(5) * m_MeansCount[k]);
                if (!m_MeansOut.empty() && !m_MeansB2[k].MemoryMapFrameRegion(0, int(m_MeansOut.size()), [&](void* p) { std::memcpy(m_MeansOut.data(), p, m_MeansOut.size() * sizeof(float)); })) W.SetGPUMeans(m_MeansOut);
                m_MeansFrame[k] = ~0ull;
            }
            if (m_nSlicesOut == 0 || Frame < m_SlicesOut[0].m_Frame + lag_frames_v) return false;
            const auto Sl = m_SlicesOut[0];
            m_SlicesOut[0] = m_SlicesOut[1]; --m_nSlicesOut;
            std::vector<std::uint64_t> Ns;
            if (m_Times.getResults(Ns) && Ns.size() >= 2 && Ns[0] != ~0ull && Ns[1] != ~0ull && Ns[1] > Ns[0])
            {
                m_LastMs = double(Ns[1] - Ns[0]) * 1e-6;
                m_RaysPerMs = double(m_LastRays) / std::max(m_LastMs, 1e-3);
                m_bMeasured = true;
            }
            if (Sl.m_Count == 0) return false;
            m_StatsOut.resize(Sl.m_Count);
            if (m_StatsB.MemoryMapFrameRegion(int(Sl.m_First), int(Sl.m_Count), [&](void* p) { std::memcpy(m_StatsOut.data(), p, m_StatsOut.size() * sizeof(world::gpu_range_stats)); })) return false;
            if (W.m_Setup.m_bAblateStats)      // measurement only (GI review, experiment A): the work counters per hit, printed about every 2 s
            {
                for (auto& S : m_StatsOut)
                {
                    m_Diag[0] += S.m_nHits; m_Diag[1] += std::bit_cast<int>(S.m_UnsupportedAt[0]); m_Diag[2] += std::bit_cast<int>(S.m_UnsupportedAt[1]);
                    m_Diag[3] += std::bit_cast<int>(S.m_UnsupportedAt[2]); m_Diag[4] += S.m_bAt; m_Diag[5] += S.m_Rays; m_Diag[6] += S.m_nUnsupported;
                    S.m_UnsupportedAt[0] = S.m_UnsupportedAt[1] = S.m_UnsupportedAt[2] = 0; S.m_bAt = 0;
                }
                if (++m_nDiag % 120 == 0 && m_Diag[0] > 0)
                    std::printf("GIABLATE rays %lld hits %lld per hit: read entries %.1f, accepted %.1f, hash probes %.2f; BVH nodes per ray %.1f; unsupported hits %.3f; entries per unsupported hit %.0f\n",
                        m_Diag[5], m_Diag[0], double(m_Diag[1]) / double(m_Diag[0]), double(m_Diag[3]) / double(m_Diag[0]), double(m_Diag[2]) / double(m_Diag[0]),
                        double(m_Diag[4]) / double(std::max<long long>(m_Diag[5], 1)), double(m_Diag[6]) / double(m_Diag[0]),
                        double(m_Diag[1]) / double(std::max<long long>(m_Diag[6], 1))), std::fflush(stdout);
            }
            W.DeliverGPURanges(std::span<const world::gpu_range>(m_Ranges).subspan(Sl.m_First, Sl.m_Count), m_StatsOut);
            m_LastRays = 0; m_LastTraced = 0;
            for (const auto& S : m_StatsOut) { m_LastRays += S.m_Rays; m_LastTraced += S.m_Traced; }
            return true;
        }

        // After world::Update: its changes up; with no slice in flight, the next slice of the batch (a new batch when it is all traced: the
        // world's pick) expanded, traced and committed in this frame, BudgetMs of the GPU's time (0: all of the batch), the cubes and the
        // means after it (dispatches between render passes). Eye: the update's (the tiers). true: a slice was dispatched
        bool Submit(xgpu::window& Window, world& W, const lights& L, std::uint64_t Frame, float BudgetMs, const v3& Eye) noexcept
        {
            { auto Ops = W.TakeGPUOps(); m_Ops.insert(m_Ops.end(), Ops.begin(), Ops.end()); }
            const bool bStructural = W.GPUGen() != m_Gen || L.m_Points.size() != m_nPoints || m_Data.empty() || W.EmitterCount() > m_EmRoom
                                  || W.GPUXformGen() != m_XformGen || W.GPULevelGen() != m_LevelGen;
            // Slices out (their stats not read yet: lag_frames_v): another of the same batch may go, two at most; a new batch, a pack,
            // a move or a level switch, an op wait for them to come back (the ranges and the records they name must stay as they were)
            if (m_nSlicesOut > 0 && (m_nSlicesOut >= 2 || bStructural || !m_Ops.empty() || m_NextRange >= m_Ranges.size())) return false;
            if (m_nSlicesOut == 0)
            {
                m_OldDataB = {};                                        // (the last pack's old layout: its frame is done)
                if (W.GPUGen() != m_Gen || L.m_Points.size() != m_nPoints || m_Data.empty() || W.EmitterCount() > m_EmRoom) { if (!Pack(Window, W, L)) return false; }     // (the emitter table past its room too)
                else if (W.GPUXformGen() != m_XformGen && !Retlas(Window, W) && !Pack(Window, W, L)) return false;     // (moved: the TLAS; its reserve full: all)
                if (W.GPULevelGen() != m_LevelGen && !Relevel(Window, W) && !Pack(Window, W, L)) return false;       // (a level switch: in place; no room: all)
            }
            const auto TP0 = std::chrono::steady_clock::now();
            if (!ApplyOps(Window, W)) return false;
            const auto TP1 = std::chrono::steady_clock::now();
            if (!Refresh(Window, W, L)) return false;
            const auto TP2 = std::chrono::steady_clock::now();
            if (std::getenv("XLION_GI_CPU_TRACE"))         // measurement: the main thread's parts of a Submit, mean ms, about every 200 frames
            {
                m_CpuTrace[0] += std::chrono::duration<double, std::milli>(TP1 - TP0).count(); m_CpuTrace[1] += std::chrono::duration<double, std::milli>(TP2 - TP1).count();
                if (++m_nCpuTrace % 200 == 0) { std::printf("GICPU ops %.2f refresh %.2f ranges %.2f ms, staged words %zu, run words %zu, run jobs %zu\n", m_CpuTrace[0] / 200, m_CpuTrace[1] / 200, m_CpuTrace[2] / 200, m_Stage.size() / 2, m_RunStage.size(), m_RunJobs.size()); std::fflush(stdout); m_CpuTrace = {}; }
            }
            bool bSlice = false;
            if (m_LinkState == 2 && m_LinkSolves > 0) return false;      // experiment D: the links own the light (no more slices: they would overwrite it)
            if (m_NextRange >= m_Ranges.size())
            {
                const auto TR0 = std::chrono::steady_clock::now();
                const auto R = W.GPURanges();
                m_CpuTrace[2] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - TR0).count();
                m_Ranges.assign(R.begin(), R.end());
                m_NextRange = 0;
                if (!m_Ranges.empty())
                {
                    std::vector<std::array<int, 8>> G(m_Ranges.size());
                    for (std::size_t k = 0; k < G.size(); ++k) { const auto& r = m_Ranges[k]; G[k] = { r.m_Inst, r.m_Level, r.m_First, r.m_Count, r.m_Flags, r.m_History, int(r.m_Seed), r.m_Start }; }
                    std::vector<world::gpu_range_stats> Zeros(m_Ranges.size());
                    // (no slice out reads them: a new batch waits for the last one's stats, two frames on)
                    if (!Fit(m_RangeB, int(sizeof(G[0])), G.size()) || m_RangeB.MemoryMapFrameRegion(0, int(G.size()), [&](void* p) { std::memcpy(p, G.data(), G.size() * sizeof(G[0])); })) return false;
                    if (!Fit(m_StatsB, int(sizeof(world::gpu_range_stats)), Zeros.size()) || m_StatsB.MemoryMapFrameRegion(0, int(Zeros.size()), [&](void* p) { std::memcpy(p, Zeros.data(), Zeros.size() * sizeof(Zeros[0])); })) return false;
                    ++m_nBatches;
                }
            }
            if (m_NextRange < m_Ranges.size())
            {
                // the slice: the ranges from m_NextRange whose estimated rays fit this frame's budget (one at least; no budget: all of them)
                const int Rays = BudgetMs > 0 ? Budget(BudgetMs) : std::numeric_limits<int>::max();
                m_SliceFirst = m_NextRange;
                m_NextRange = SliceEnd(m_Ranges, m_SliceFirst, Rays);
                int nThreads = 0;
                for (std::size_t k = m_SliceFirst; k < m_NextRange; ++k) nThreads += m_Ranges[k].m_Count;
                m_SliceCount = m_NextRange - m_SliceFirst;
                const std::uint32_t Zero[4] = {};
                auto& CountB = m_CountB2[m_nSlices % 2];                // (two: the slice before may still be running)
                if (!FitLocal(m_ItemB, int(sizeof(world::gpu_item)), std::size_t(nThreads)) || !FitLocal(m_Results, int(sizeof(world::gpu_result)), std::size_t(nThreads))
                    || !Fit(CountB, int(sizeof(std::uint32_t)), 4) || CountB.MemoryMapFrameRegion(0, 4, [&](void* p) { std::memcpy(p, Zero, sizeof(Zero)); })) return false;
                const struct { int m_RangeFirst, m_nRanges, m_nThreads, m_StartBase; float m_Eye[4]; } EPush{ int(m_SliceFirst), int(m_SliceCount), nThreads, m_Ranges[m_SliceFirst].m_Start, { Eye.m_X, Eye.m_Y, Eye.m_Z, 0 } };
                const struct { int m_nItems, m_TlasRoot, m_PrimOffset, m_First; } Push{ nThreads, m_Scene.m_TlasRoot, m_Scene.m_PrimOffset, 0 };
                const struct { int m_nItems, m_First, m_Count, m_Pad; } CPush{ nThreads, 0, nThreads, 0 };
                xgpu::buffer* EBufs[] = { &m_DataB, &m_RangeB, &m_ItemB, &CountB };
                xgpu::buffer* Bufs[]  = { &m_Nodes, &m_Tri4, &m_Inst, &m_Tlas, &m_DataB, &m_ItemB, &m_Results, &CountB };
                xgpu::buffer* CBufs[] = { &m_DataB, &m_ItemB, &m_Results, &m_StatsB, &CountB };
                Window.WriteTimestamp(m_Times, 0);
                Window.Dispatch(m_Expand, EBufs, &EPush, sizeof(EPush), (nThreads + 63) / 64);
                Window.Dispatch(m_Pass, Bufs, &Push, sizeof(Push), world::GPUGroups(std::size_t(nThreads)));
                Window.Dispatch(m_Commit, CBufs, &CPush, sizeof(CPush), (nThreads + 63) / 64);
                Window.WriteTimestamp(m_Times, 1);
                m_SlicesOut[m_nSlicesOut++] = { Frame, m_SliceFirst, m_SliceCount };
                ++m_nSlices;
                bSlice = true;
            }
            if (m_bGhosts && m_nGhosts)      // GI sections: the ghosts take their owners' committed set, every frame (before the means and Show)
            {
                xgpu::buffer* B[] = { &m_DataB, &m_GhostB };
                const int Push[4] = { int(m_nGhosts), 0, 0, 0 };
                Window.Dispatch(m_GhostsC, B, Push, sizeof(Push), int((m_nGhosts + 63) / 64));
            }
            // the cubes (their parents from the traced ones) and the means, every frame (what the next update reads)
            std::vector<std::array<int, 8>> Cubes;
            for (int i = 0; i < W.InstanceCount(); ++i)
                if (W.IsAlive(i) && W.IsCube(i))
                {
                    std::array<int, 8> C{ i, 0, 0, 0, 0, 0, 0, 0 };
                    for (const int k : W.CubeSet(i)) if (C[1] < 6) C[std::size_t(2 + C[1]++)] = k;
                    Cubes.push_back(C);
                }
            if (!Cubes.empty())
            {
                auto& CubeB = m_CubeB2[m_nSubmits % 2];          // (two: the frame in flight may still read the other; mapped without a wait)
                if (!Fit(CubeB, int(sizeof(Cubes[0])), Cubes.size()) || CubeB.MemoryMapFrameRegion(0, int(Cubes.size()), [&](void* p) { std::memcpy(p, Cubes.data(), Cubes.size() * sizeof(Cubes[0])); })) return false;
                xgpu::buffer* B[] = { &m_DataB, &CubeB };
                const int Pad[4] = {};
                Window.Dispatch(m_Cube, B, Pad, sizeof(Pad), int(Cubes.size()));
            }
            if (m_Data[1] > 0 && m_MeansFrame[Frame % 2] == ~0ull)         // (two copies: the one written the frame before may still be running; this one read)
            {
                const int k = int(Frame % 2);
                if (!Fit(m_MeansB2[k], int(sizeof(float)), std::size_t(5) * m_Data[1])) return false;
                xgpu::buffer* B[] = { &m_DataB, &m_MeansB2[k] };
                const int Pad[4] = {};
                Window.Dispatch(m_Means, B, Pad, sizeof(Pad), int(m_Data[1]));
                m_MeansFrame[k] = Frame; m_MeansCount[k] = m_Data[1]; m_LastMeans = k;
            }
            return bSlice;
        }

        // Experiment D (compiled transport) on the GPU, world::CompileTransport's twin: the links of level-0 surfels, one workgroup each
        // (xlionrender_gi_links_comp.glsl): Items { instance, surfel }, Rays each, into a host-visible table (2 x links_v a surfel: the
        // heaviest links, the tail per instance) and D. Packs the scene first if it must. First / Count: a range of Items (a frame's share)
        static constexpr int links_v = 16;
        bool m_bLinkReadback = false;       // (tests: CopyLinksOut copied the links and hit clusters out for ReadLinks)
        static constexpr std::size_t hits_v = 8;       // the hit clusters a link row keeps: its first bounce (xlionrender_gi_links_comp.glsl gi_hits_v)
        bool BuildLinks(xgpu::window& Window, world& W, const lights& L, std::span<const std::array<int, 2>> Items, int Rays, std::size_t First, std::size_t Count, unsigned Seed = 1) noexcept
        {
            if (!m_bLinks || Items.empty()) return false;
            if (W.GPUGen() != m_Gen || L.m_Points.size() != m_nPoints || m_Data.empty() || W.EmitterCount() > m_EmRoom) { if (!Pack(Window, W, L)) return false; }
            if (First == 0)
            {
                m_bLinksLocal = false;                                  // (SolveLinks copies the new ones)
                struct item { std::int32_t m_Ids[4]; float m_Offsets[4]; };
                std::vector<item> It(Items.size());
                rng R(Seed);
                for (std::size_t k = 0; k < Items.size(); ++k) It[k] = { { Items[k][0], Items[k][1], 0, Rays }, { R.Next(), R.Next(), 0, 0 } };
                if (!Fit(m_LinkItemB, int(sizeof(item)), It.size()) || m_LinkItemB.MemoryMap(0, int(It.size()), [&](void* p) { std::memcpy(p, It.data(), It.size() * sizeof(item)); })) return false;
                if (!FitLocal(m_LinkB, int(2 * sizeof(std::uint32_t)), 2 * links_v * Items.size()) || !Fit(m_LinkDB, int(4 * sizeof(float)), Items.size())) return false;
                // per instance its first item (+1: the total): mode 4 finds a neighbor's item by it
                std::vector<std::uint32_t> InstFirst(std::size_t(W.InstanceCount()) + 1, 0);
                for (const auto& It2 : Items) if (std::size_t(It2[0]) < InstFirst.size() - 1) ++InstFirst[std::size_t(It2[0]) + 1];
                for (std::size_t i = 1; i < InstFirst.size(); ++i) InstFirst[i] += InstFirst[i - 1];
                if (!Local(m_LinkDirB, std::vector<std::array<float, 4>>(Items.size())) || !Local(m_LinkDirTmpB, std::vector<std::array<float, 4>>(Items.size()))
                 || !Local(m_LinkInstFirstB, InstFirst) || !FitLocal(m_LinkHitB, int(4 * sizeof(std::uint32_t)), hits_v * Items.size())) return false;
                m_bLinkDirValid = false;
            }
            Count = std::min(Count, Items.size() - std::min(First, Items.size()));
            if (Count == 0) return true;
            const struct { int m_nItems, m_TlasRoot, m_PrimOffset, m_First, m_Mode, m_Pad[3]; } Push{ int(Items.size()), m_Scene.m_TlasRoot, m_Scene.m_PrimOffset, int(First), 0, {} };
            if (!LinkAux()) return false;
            xgpu::buffer* B[] = { &m_Nodes, &m_Tri4, &m_Inst, &m_Tlas, &m_DataB, &m_LinkItemB, &m_LinkB, &m_LinkDB, &m_LinkDirB, &m_LinkListB, &m_LinkChangedB, &m_LinkFlagB, &m_LinkHitB, &m_LinkInstFirstB, &m_LinkDirTmpB };
            Window.Dispatch(m_LinksC, B, &Push, sizeof(Push), int(Count));
            return true;
        }
        // (tests) The links and hit clusters (device-local) copied out for ReadLinks: in a frame after the build's, read after this one
        bool CopyLinksOut(xgpu::window& Window, std::size_t nItems) noexcept
        {
            // (in words: FitLocal's entries, a copy's unit is its source's entry)
            const xgpu::buffer::copy CL{ 0, 0, int(2 * 2 * links_v * nItems) }, CH{ 0, 0, int(4 * hits_v * nItems) };
            if (!Fit(m_LinkReadB, int(sizeof(std::uint32_t)), 2 * 2 * links_v * nItems) || !Fit(m_LinkHitReadB, int(sizeof(std::uint32_t)), 4 * hits_v * nItems)) return false;
            Window.CopyBuffer(m_LinkB, m_LinkReadB, std::span{ &CL, 1 });
            Window.CopyBuffer(m_LinkHitB, m_LinkHitReadB, std::span{ &CH, 1 });
            m_bLinkReadback = true;
            return true;
        }
        // The links' first bounce of the lights (E = D + F + T E): each item's hit clusters lit (the links shader's mode 1: a shadow ray to
        // the sun and to each lamp from each cluster). After a build and whenever a light changes: the links stay
        // A slice [First, First + Count) when the lights move (PumpLinks' per-frame budget): its smoothing reads the others as they are
        bool RelightLinks(xgpu::window& Window, std::size_t nItems, std::size_t First = 0, std::size_t Count = ~std::size_t(0)) noexcept
        {
            if (!m_bLinks || !m_bSolve || nItems == 0 || m_Data.empty() || !CopyLinksLocal(Window, nItems)) return false;
            const std::size_t End = std::min(nItems, First + std::min(Count, nItems));
            if (First >= End) return true;
            const int nGroups = int((End - First + 63) / 64);
            const struct { int m_nItems, m_TlasRoot, m_PrimOffset, m_First, m_Mode, m_Pad[3]; } Push{ int(End), m_Scene.m_TlasRoot, m_Scene.m_PrimOffset, int(First), 1, {} };
            if (!LinkAux()) return false;
            xgpu::buffer* B[] = { &m_Nodes, &m_Tri4, &m_Inst, &m_Tlas, &m_DataB, &m_LinkItemB, &m_LinkB, &m_LinkDB, &m_LinkDirB, &m_LinkListB, &m_LinkChangedB, &m_LinkFlagB, &m_LinkHitB, &m_LinkInstFirstB, &m_LinkDirTmpB };
            Window.Dispatch(m_LinksC, B, &Push, sizeof(Push), nGroups);
            const struct { int m_nItems, m_TlasRoot, m_PrimOffset, m_First, m_Mode, m_Pad[3]; } Smooth{ int(End), m_Scene.m_TlasRoot, m_Scene.m_PrimOffset, int(First), 4, {} };
            Window.Dispatch(m_LinksC, B, &Smooth, sizeof(Smooth), nGroups);
            m_bLinkDirValid = true;
            ++m_nRelights;
            if (std::getenv("XLION_GI_LINK_TRACE")) { std::printf("GILINKS relit %zu items, %u lamps in the GPU data\n", nItems, m_Data.size() > 3 ? m_Data[3] : 0u); std::fflush(stdout); }     // (measurement)
            return true;
        }
        // The links' D and items copied once into device-local buffers (the links and their hit clusters are built there): every frame's
        // solve read them over the bus - the Bistro's 506k surfels took 49 ms a frame
        bool CopyLinksLocal(xgpu::window& Window, std::size_t nItems) noexcept
        {
            if (m_bLinksLocal) return true;
            if (!Local(m_LinkDLocalB, std::vector<std::array<std::uint32_t, 4>>(nItems)) || !Local(m_LinkItemLocalB, std::vector<std::array<std::uint32_t, 8>>(nItems))) return false;
            const xgpu::buffer::copy CN{ 0, 0, int(nItems) };
            Window.CopyBuffer(m_LinkDB, m_LinkDLocalB, std::span{ &CN, 1 });
            Window.CopyBuffer(m_LinkItemB, m_LinkItemLocalB, std::span{ &CN, 1 });
            m_bLinksLocal = true;
            return true;
        }
        // One Jacobi iteration over the links built (xlionrender_gi_solve_comp.glsl): into a scratch, then into the surfels' light. After
        // Submit (its means: the tail's instance means) and before Show (what it shows)
        bool SolveLinks(xgpu::window& Window, std::size_t nItems) noexcept
        {
            if (!m_bSolve || nItems == 0 || m_Data.empty() || m_LastMeans < 0) return false;
            if (!FitLocal(m_LinkNextB, int(4 * sizeof(float)), nItems)) return false;
            if (!CopyLinksLocal(Window, nItems)) return false;
            const int Groups = int((nItems + 63) / 64);
            if (!LinkAux()) return false;
            xgpu::buffer* B[] = { &m_DataB, &m_LinkB, &m_LinkDLocalB, &m_LinkItemLocalB, &m_MeansB2[m_LastMeans], &m_LinkNextB, &m_LinkDirB };
            struct { int m_nItems, m_Mode, m_nInst, m_Links; } Push{ int(nItems), 0, int(m_Data[1]), links_v };
            Window.Dispatch(m_SolveC, B, &Push, sizeof(Push), Groups);
            Push.m_Mode = 1;
            Window.Dispatch(m_SolveC, B, &Push, sizeof(Push), Groups);
            return true;
        }
        // The probe grid (xlionrender_gi_probes_comp.glsl): Dim probes Spacing apart around Eye, PerFrame of them updated a frame (round
        // robin), each from the scene's GI as a ray's hit sees it. For what has no surfels of its own (skinned meshes: the material's
        // SurfelGIProbes reads ProbeBuffer, bound as their GIMesh). The header is written every frame (the grid moves with the eye; the
        // scene pass's positions are the eye's: its header carries it). After Submit
        bool UpdateProbes(xgpu::window& Window, const v3& Eye, float Spacing, std::array<int, 3> Dim, int PerFrame, std::uint64_t Frame) noexcept
        {
            if (!m_bProbes || m_Data.empty() || m_Scene.m_bEmpty || Spacing <= 0) return false;     // (an empty scene's BVH is a placeholder: its root walked off the buffers)
            const std::size_t n = std::size_t(Dim[0]) * Dim[1] * Dim[2];
            if (!FitLocal(m_ProbeB, int(sizeof(std::uint32_t)), probe_header_v + probe_words_v * n)) return false;
            std::array<std::uint32_t, probe_header_v> H{};
            const float E[3] = { Eye.m_X, Eye.m_Y, Eye.m_Z };
            for (int k = 0; k < 3; ++k) { H[std::size_t(k)] = std::uint32_t(int(std::floor(E[k] / Spacing)) - Dim[std::size_t(k)] / 2); H[std::size_t(4 + k)] = std::uint32_t(Dim[std::size_t(k)]); H[std::size_t(8 + k)] = std::bit_cast<std::uint32_t>(E[k]); }
            H[3] = std::bit_cast<std::uint32_t>(Spacing); H[7] = std::uint32_t(n);
            auto& HB = m_ProbeHdrB2[Frame % 2];          // (two: the frame in flight may still read the other; mapped without a wait)
            if (!Fit(HB, int(sizeof(std::uint32_t)), probe_header_v) || HB.MemoryMapFrameRegion(0, int(probe_header_v), [&](void* p) { std::memcpy(p, H.data(), sizeof(H)); })) return false;
            const xgpu::buffer::copy C{ 0, 0, int(probe_header_v) };
            Window.CopyBuffer(HB, m_ProbeB, std::span{ &C, 1 });
            const int Count = std::min(PerFrame, int(n));
            const struct { int m_nProbes, m_TlasRoot, m_PrimOffset, m_First, m_Frame, m_Count, m_Pad0, m_Pad1; } Push{ int(n), m_Scene.m_TlasRoot, m_Scene.m_PrimOffset, int(m_ProbeCursor % n), int(Frame), Count, 0, 0 };
            xgpu::buffer* B[] = { &m_Nodes, &m_Tri4, &m_Inst, &m_Tlas, &m_DataB, &m_ProbeB };
            Window.Dispatch(m_ProbesC, B, &Push, sizeof(Push), Count);
            m_ProbeCursor = (m_ProbeCursor + std::size_t(Count)) % n;
            return true;
        }
        xgpu::buffer* ProbeBuffer(void) noexcept { return m_bProbes && m_ProbeB.m_Private ? &m_ProbeB : nullptr; }
        // (tests) The probe buffer copied out (a frame after its update), then read (a frame after this one)
        bool CopyProbesOut(xgpu::window& Window) noexcept
        {
            if (!m_ProbeB.m_Private) return false;
            const int n = m_ProbeB.getEntryCount();
            const xgpu::buffer::copy C{ 0, 0, n };
            if (!Fit(m_ProbeReadB, int(sizeof(std::uint32_t)), std::size_t(n))) return false;
            Window.CopyBuffer(m_ProbeB, m_ProbeReadB, std::span{ &C, 1 });
            return true;
        }
        std::vector<std::uint32_t> ReadProbes(void) noexcept
        {
            std::vector<std::uint32_t> V(m_ProbeB.m_Private ? std::size_t(m_ProbeB.getEntryCount()) : 0);
            if (V.empty() || m_ProbeReadB.MemoryMap(0, int(V.size()), [&](void* p) { std::memcpy(V.data(), p, V.size() * 4); })) V.clear();
            return V;
        }
        static constexpr std::size_t probe_header_v = 16, probe_words_v = 16;      // (xlionrender_gi_probes_comp.glsl)

        // The local rebuild's buffers bound by every links / solve dispatch (one word each until used)
        bool LinkAux() noexcept
        {
            if (!m_LinkListB.m_Private    && !Local(m_LinkListB, std::vector<std::uint32_t>(1, 0u))) return false;
            if (!m_LinkChangedB.m_Private && !Local(m_LinkChangedB, std::vector<std::uint32_t>(1, 0u))) return false;
            if (!m_LinkFlagB.m_Private    && !Fit(m_LinkFlagB, int(sizeof(std::uint32_t)), 1)) return false;
            return true;
        }
        void SnapshotLinked(const world& W) noexcept
        {
            m_LinkedMatGen.resize(std::size_t(W.InstanceCount())); m_LinkedL2W.resize(std::size_t(W.InstanceCount()));
            for (int i = 0; i < W.InstanceCount(); ++i) { m_LinkedMatGen[std::size_t(i)] = W.MatGen(i); m_LinkedL2W[std::size_t(i)] = W.InstanceL2W(i); }
        }
        // A static object moved or a material changed once the links solve: only the rows it touches are built again - its own, the rows
        // whose links or tail read it (the GPU flags them: mode 3, read back), and for a move the rows near it, before and after (their
        // rays may meet it now, or no more). The others keep their links and the solve goes on; the rows built again have new hit
        // clusters, lit at once (RelightLinks). The rays stay off. Moving objects are the ray solver's (not here)
        void LocalRebuild(xgpu::window& Window, world& W, const lights& L, std::uint64_t Frame, int Rays, int PerFrame) noexcept
        {
            const std::size_t nItems = m_LinkItems.size(), nInst = std::size_t(W.InstanceCount());
            if (m_LocalState == 0)
            {
                if (m_LinkedMatGen.size() != nInst) { SnapshotLinked(W); return; }
                std::vector<std::uint32_t> Changed(nInst, 0);
                bool bAny = false;
                m_LocalNear.assign(nItems, 0);
                for (std::size_t i = 0; i < nInst; ++i)
                {
                    const bool bMat = W.MatGen(int(i)) != m_LinkedMatGen[i];
                    const bool bMoved = std::memcmp(&W.InstanceL2W(int(i)), &m_LinkedL2W[i], sizeof(xmath::fmat4)) != 0;
                    if (!bMat && !bMoved) continue;
                    Changed[i] = 1; bAny = true;
                    if (bMoved)         // the rows near it, where it was and where it is (its bounds now, the old ones moved back)
                    {
                        const auto [Lo, Hi] = W.InstanceBounds(int(i));
                        const v3 Ext = Hi - Lo, Margin = v3{ 1, 1, 1 } * std::max({ Ext.m_X, Ext.m_Y, Ext.m_Z, 0.5f });
                        const xmath::fmat4 Back = m_LinkedL2W[i] * W.InstanceL2W(int(i)).Inverse();
                        const v3 C0 = Back * ((Lo + Hi) * 0.5f);
                        for (std::size_t k = 0; k < nItems; ++k)
                        {
                            const int Id = m_LinkItems[k][0];
                            if (!W.IsAlive(Id)) continue;
                            const v3 P = W.InstanceL2W(Id) * W.InstanceMesh(Id).m_Levels[0][std::size_t(m_LinkItems[k][1])].m_Center;
                            const auto In = [&](const v3& c) { const v3 d = P - c; return std::abs(d.m_X) <= Ext.m_X * 0.5f + Margin.m_X && std::abs(d.m_Y) <= Ext.m_Y * 0.5f + Margin.m_Y && std::abs(d.m_Z) <= Ext.m_Z * 0.5f + Margin.m_Z; };
                            if (In((Lo + Hi) * 0.5f) || In(C0)) m_LocalNear[k] = 1;
                        }
                    }
                }
                if (!bAny) return;
                SnapshotLinked(W);                  // (a change while this one is built: the next round)
                if (!Local(m_LinkChangedB, Changed) || !Fit(m_LinkFlagB, int(sizeof(std::uint32_t)), nItems)) return;
                const struct { int m_nItems, m_TlasRoot, m_PrimOffset, m_First, m_Mode, m_Pad[3]; } Push{ int(nItems), m_Scene.m_TlasRoot, m_Scene.m_PrimOffset, 0, 3, {} };
                xgpu::buffer* B[] = { &m_Nodes, &m_Tri4, &m_Inst, &m_Tlas, &m_DataB, &m_LinkItemB, &m_LinkB, &m_LinkDB, &m_LinkDirB, &m_LinkListB, &m_LinkChangedB, &m_LinkFlagB, &m_LinkHitB, &m_LinkInstFirstB, &m_LinkDirTmpB };
                Window.Dispatch(m_LinksC, B, &Push, sizeof(Push), int((nItems + 63) / 64));
                m_LocalFrame = Frame; m_LocalState = 1;
                return;
            }
            if (m_LocalState == 1)
            {
                if (Frame < m_LocalFrame + lag_frames_v + 1) return;
                std::vector<std::uint32_t> Flags(nItems, 0);
                if (m_LinkFlagB.MemoryMap(0, int(nItems), [&](void* p) { std::memcpy(Flags.data(), p, nItems * sizeof(std::uint32_t)); })) { m_LocalState = 0; return; }
                m_LocalList.clear();
                for (std::size_t k = 0; k < nItems; ++k) if (Flags[k] || (k < m_LocalNear.size() && m_LocalNear[k])) m_LocalList.push_back(std::uint32_t(k));
                if (m_LocalList.empty() || !Local(m_LinkListB, m_LocalList)) { m_LocalState = 0; return; }
                m_LocalCursor = 0; m_LocalState = 2;
                return;
            }
            if (m_LocalState == 2)
            {
                const std::size_t n = std::min(std::size_t(PerFrame), m_LocalList.size() - m_LocalCursor);
                const struct { int m_nItems, m_TlasRoot, m_PrimOffset, m_First, m_Mode, m_Pad[3]; } Push{ int(nItems), m_Scene.m_TlasRoot, m_Scene.m_PrimOffset, int(m_LocalCursor), 2, {} };
                xgpu::buffer* B[] = { &m_Nodes, &m_Tri4, &m_Inst, &m_Tlas, &m_DataB, &m_LinkItemB, &m_LinkB, &m_LinkDB, &m_LinkDirB, &m_LinkListB, &m_LinkChangedB, &m_LinkFlagB, &m_LinkHitB, &m_LinkInstFirstB, &m_LinkDirTmpB };
                if (n) Window.Dispatch(m_LinksC, B, &Push, sizeof(Push), int(n));
                m_LocalCursor += n;
                if (m_LocalCursor >= m_LocalList.size()) { m_LocalState = 3; m_LocalFrame = Frame; }
                return;
            }
            if (m_LocalState == 3)
            {
                // the solve's copies again (the rows built again are in the host-visible ones), their first bounce now
                m_bLinksLocal = false;
                (void)Rays; (void)L;
                (void)RelightLinks(Window, nItems);
                ++m_nLocalRebuilds; m_nLocalRows += m_LocalList.size();
                if (std::getenv("XLION_GI_LINK_TRACE")) { std::printf("GILINKS rebuilt %zu of %zu rows (a move or a material)\n", m_LocalList.size(), nItems); std::fflush(stdout); }
                m_LocalState = 0;
            }
        }
    public:
        std::size_t LocalRebuilds(void) const noexcept { return m_nLocalRebuilds; }      // (tests)
        std::size_t LocalRows(void) const noexcept { return m_nLocalRows; }
        std::size_t LinkBuilds(void) const noexcept { return m_nLinkBuilds; }
        std::size_t Relights(void) const noexcept { return m_nRelights; }
        std::size_t LinkItemCount(void) const noexcept { return m_LinkItems.size(); }
        void        SetRelightPerFrame(std::size_t n) noexcept { m_RelightPerFrame = std::max<std::size_t>(n, 1); }     // the lights' relight budget
        // Experiment D in a renderer (XLION_GI_LINKS): after Settle frames of the ray solver (the reads' validity and moments), the links of
        // every level-0 surfel are built, PerFrame surfels a frame; two frames after the last slice they are solved, one iteration a frame.
        // A pack, a move or a level switch drops them (built again Settle frames on). After Submit, before Show
        void PumpLinks(xgpu::window& Window, world& W, const lights& L, std::uint64_t Frame, int Rays, int Settle, int PerFrame) noexcept
        {
            if (!m_bLinks || !m_bSolve || m_Data.empty()) return;
            // (and the materials: the links' hits bring their surfaces' albedo - a capture landing after they were built left them grey)
            // (and the sky: it is in D, made again with the links. The sun and the lamps are not: their light at each surfel (Dir) is made
            // again on its own when they change - RelightLinks - and the links stay)
            const auto Hash = [](std::uint64_t h, const void* p, std::size_t n) { const auto* b = static_cast<const unsigned char*>(p); for (std::size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull; return h; };
            std::uint64_t SkyKey = Hash(1469598103934665603ull, &L.m_Sky, sizeof(L.m_Sky));
            SkyKey = Hash(SkyKey, L.m_SkyOct.data(), sizeof(L.m_SkyOct)); SkyKey = Hash(SkyKey, &L.m_bSkyOct, sizeof(L.m_bSkyOct));
            std::uint64_t DirKey = Hash(1469598103934665603ull, &L.m_SunDirection, sizeof(L.m_SunDirection));
            DirKey = Hash(DirKey, &L.m_SunColor, sizeof(L.m_SunColor));
            if (!L.m_Points.empty()) DirKey = Hash(DirKey, L.m_Points.data(), L.m_Points.size() * sizeof(L.m_Points[0]));
            // (lighting shades the emitters' direct light - setup::m_bEmitterDirectExternal -: D without it, F with it at the hits; switched: all again)
            const std::uint64_t External = W.m_Setup.m_bEmitterDirectExternal ? 0x5EEDC0DE5EEDC0DEull : 0ull;
            SkyKey ^= External; DirKey ^= External;
            // (a move or a material while the links settle or build: all of them again; once they solve, the rows the change touches -
            // LocalRebuild - the rest of the scene keeps its links)
            const std::uint64_t Local = m_LinkState == 2 ? m_LinkLocalKey : ((m_XformGen << 20) * 0x9E3779B97F4A7C15ull ^ W.MaterialGenSum());
            if (m_LinkState != 2) m_LinkLocalKey = Local;
            const std::uint64_t Key = ((m_Gen ^ (m_LevelGen << 40)) * 0x9E3779B97F4A7C15ull ^ Local) * 0xC2B2AE3D27D4EB4Full ^ SkyKey;
            if (Key != m_LinkKey && std::getenv("XLION_GI_LINK_TRACE")) { std::printf("GILINKS dropped at frame %llu: gen %llu xform %llu level %llu materials %llu sky %llx (state %d)\n", (unsigned long long)Frame, (unsigned long long)m_Gen, (unsigned long long)m_XformGen, (unsigned long long)m_LevelGen, (unsigned long long)W.MaterialGenSum(), (unsigned long long)SkyKey, m_LinkState); std::fflush(stdout); }     // (measurement)
            if (Key != m_LinkKey) { m_LinkKey = Key; m_LinkState = 0; m_LinkSince = Frame; m_LinkItems.clear(); m_bLinksLocal = false; }
            if (m_LinkState == 0 && Frame >= m_LinkSince + std::uint64_t(Settle))
            {
                for (int i = 0; i < W.InstanceCount(); ++i)
                    if (W.IsAlive(i) && W.Level(i) == 0) for (std::size_t s = 0; s < W.InstanceSurfels0(i); ++s) m_LinkItems.push_back({ i, int(s) });
                m_LinkCursor = 0; m_LinkState = m_LinkItems.empty() ? 3 : 1;
                ++m_nLinkBuilds;
                SnapshotLinked(W);                  // (what the links are built against: each instance's placement and material)
                m_LocalState = 0;
                m_LinkT0 = std::chrono::steady_clock::now();
            }
            if (m_LinkState == 1)
            {
                if (!BuildLinks(Window, W, L, m_LinkItems, Rays, m_LinkCursor, std::size_t(PerFrame))) { m_LinkState = 3; return; }
                m_LinkCursor += std::size_t(PerFrame);
                if (m_LinkCursor >= m_LinkItems.size()) { m_LinkState = 2; m_LinkReady = Frame + 2; }
            }
            if (m_LinkState == 2 && Frame >= m_LinkReady)
            {
                if (m_LinkSolves == 0) std::printf("GILINKS %zu surfels' links built in %.0f ms (%d rays each, %d a frame); solving\n", m_LinkItems.size(),
                                                   std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m_LinkT0).count(), Rays, PerFrame), std::fflush(stdout);
                if (m_LinkSolves == 0 && std::getenv("XLION_GEOM_SIZES"))      // measurement: the GI's GPU memory (MB)
                {
                    const auto MB = [](std::size_t b) { return double(b) / (1024.0 * 1024.0); };
                    const std::size_t n = m_LinkItems.size();
                    std::printf("GIMEM data %.1f, BVH nodes %.1f + tri4 %.1f, links %.1f (+ its device-local copy %.1f), hit clusters %.1f (+ %.1f), D/F/items %.1f MB\n",
                                MB(m_Data.size() * 4), MB(m_Scene.m_Nodes.size() * sizeof(m_Scene.m_Nodes[0])), MB(m_Scene.m_Tri4.size() * sizeof(m_Scene.m_Tri4[0])),
                                MB(2 * links_v * n * 8), 0.0, MB(hits_v * n * 16), 0.0, MB(n * (16 * 2 + 32 * 2 + 16 * 3))), std::fflush(stdout);
                }
                LocalRebuild(Window, W, L, Frame, Rays, PerFrame);
                // The lights' first bounce: all at once the first time; after a light changes, relight_per_frame_v items a frame round the
                // items from where the last frame stopped - each change owes one whole round (a slow sun: an even refresh; a sudden one spread
                // over frames). HTraceWSGI's per-frame budget (xLION_Docs research/htrace_wsgi.md): its cost bounded, not the scene's size
                // ponytail: round-robin; the camera's distance / visibility / the sun's turn since an item's last relight first when the lag shows
                static const bool        s_bNoRelight = std::getenv("XLION_GI_NO_RELIGHT") != nullptr;      // (measurement: the relight's upper bound)
                static const char*       s_pPerFrame  = std::getenv("XLION_GI_RELIGHT_PER_FRAME");         // (measurement: the budget)
                const std::size_t        Budget       = s_pPerFrame ? std::size_t(std::max(1, std::atoi(s_pPerFrame))) : m_RelightPerFrame;
                const std::size_t nItems = m_LinkItems.size();
                if (!m_bLinkDirValid) { if (RelightLinks(Window, nItems)) { m_LinkDirKey = DirKey; m_RelightOwed = 0; } }
                else
                {
                    if (DirKey != m_LinkDirKey && !s_bNoRelight) { m_LinkDirKey = DirKey; m_RelightOwed = nItems; }
                    if (m_RelightOwed > 0 && nItems > 0)
                    {
                        const std::size_t n = std::min({ Budget, m_RelightOwed, nItems });
                        m_RelightCursor %= nItems;
                        (void)RelightLinks(Window, nItems, m_RelightCursor, n);
                        if (m_RelightCursor + n > nItems) (void)RelightLinks(Window, nItems, 0, m_RelightCursor + n - nItems);     // (the wrap)
                        m_RelightCursor = (m_RelightCursor + n) % nItems; m_RelightOwed -= n;
                    }
                }
                if (SolveLinks(Window, m_LinkItems.size())) ++m_LinkSolves;
                // GI sections: the solve wrote every surfel's light, the ghosts' too (their own links): they take their owners' again, or
                // a section's border steps (Submit's copy ran before the solve)
                if (m_bGhosts && m_nGhosts)
                {
                    xgpu::buffer* B[] = { &m_DataB, &m_GhostB };
                    const int Push[4] = { int(m_nGhosts), 0, 0, 0 };
                    Window.Dispatch(m_GhostsC, B, Push, sizeof(Push), int((m_nGhosts + 63) / 64));
                }
            }
        }

        // What BuildLinks wrote (its frame done) as the world's transport: rows in Items' order (all of the world's level-0 surfels, in
        // order: the CPU's rows), the tail grouped by instance (world::TruncateTransport's tail through the instances)
        static float HalfToFloat(std::uint16_t h) noexcept
        {
            const std::uint32_t e = (h >> 10) & 31u, m = h & 1023u;
            const float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? std::numeric_limits<float>::infinity() : std::ldexp(float(m | 1024u), int(e) - 25);
            return (h & 0x8000u) ? -v : v;
        }
        bool ReadLinks(const world& W, std::span<const std::array<int, 2>> Items, world::transport& T, double* pDropped = nullptr) noexcept
        {
            if (!m_bLinks || Items.empty() || !m_bLinkReadback) return false;
            std::vector<std::array<std::uint32_t, 2>> Lk(2 * links_v * Items.size());
            std::vector<std::array<float, 4>> Dk(Items.size());
            std::vector<std::array<std::uint32_t, 4>> Hk(hits_v * Items.size());
            if (m_LinkReadB.MemoryMap(0, int(2 * Lk.size()), [&](void* p) { std::memcpy(Lk.data(), p, Lk.size() * 8); })) return false;
            if (m_LinkDB.MemoryMap(0, int(Dk.size()), [&](void* p) { std::memcpy(Dk.data(), p, Dk.size() * 16); })) return false;
            if (m_LinkHitReadB.MemoryMap(0, int(4 * Hk.size()), [&](void* p) { std::memcpy(Hk.data(), p, Hk.size() * 16); })) return false;
            T = {};
            T.m_InstFirst.push_back(0);
            for (int i = 0; i < W.InstanceCount(); ++i) T.m_InstFirst.push_back(T.m_InstFirst.back() + std::uint32_t(W.IsAlive(i) ? W.InstanceSurfels0(i) : 0));
            T.m_First.push_back(0); T.m_AFirst.push_back(0); T.m_HitFirst.push_back(0);
            double Drop = 0;
            for (std::size_t k = 0; k < Items.size(); ++k)
            {
                T.m_D.push_back(v3{ Dk[k][0], Dk[k][1], Dk[k][2] }); Drop += Dk[k][3];
                // its hit clusters (the shader's packing: half offsets from its center, half albedo, an octahedral snorm normal)
                const v3 C = W.InstanceL2W(Items[k][0]) * W.InstanceMesh(Items[k][0]).m_Levels[0][std::size_t(Items[k][1])].m_Center;
                for (std::size_t q = 0; q < hits_v; ++q)
                {
                    const auto& h = Hk[hits_v * k + q];
                    const auto Half = [](std::uint32_t u) { return HalfToFloat(std::uint16_t(u & 0xFFFFu)); };
                    const v3 A{ Half(h[1] >> 16), Half(h[2]), Half(h[2] >> 16) };
                    if (A.m_X + A.m_Y + A.m_Z <= 0) continue;
                    const auto Sn = [](std::uint32_t u) { return std::max(float(std::int16_t(u & 0xFFFFu)) / 32767.0f, -1.0f); };
                    float ex = Sn(h[3]), ey = Sn(h[3] >> 16);
                    v3 Nn{ ex, ey, 1 - std::abs(ex) - std::abs(ey) };
                    if (Nn.m_Z < 0) { const float x = Nn.m_X, y = Nn.m_Y; Nn.m_X = (1 - std::abs(y)) * (x >= 0 ? 1.0f : -1.0f); Nn.m_Y = (1 - std::abs(x)) * (y >= 0 ? 1.0f : -1.0f); }
                    T.m_Hits.push_back({ C + v3{ Half(h[0]), Half(h[0] >> 16), Half(h[1]) }, Nn.NormalizeCopy(), A });
                }
                T.m_HitFirst.push_back(std::uint32_t(T.m_Hits.size()));
                T.m_Group.push_back(std::uint32_t(Items[k][0]));
                for (int q = 0; q < 2 * links_v; ++q)
                {
                    const auto& e = Lk[2 * links_v * k + std::size_t(q)];
                    if (e[0] == 0xFFFFFFFFu) continue;
                    const v3 w = UnpackRGB9E5(e[1]);
                    if (e[0] & 0x80000000u) { T.m_AGroup.push_back(e[0] & 0x7FFFFFFFu); T.m_AW.push_back(w); }
                    else { T.m_Col.push_back(T.m_InstFirst[e[0] >> 16] + (e[0] & 0xFFFFu)); T.m_W.push_back(w); }
                }
                T.m_First.push_back(std::uint32_t(T.m_Col.size())); T.m_AFirst.push_back(std::uint32_t(T.m_AGroup.size()));
            }
            T.m_GroupSize.assign(std::size_t(W.InstanceCount()), 0);
            for (int i = 0; i < W.InstanceCount(); ++i) T.m_GroupSize[std::size_t(i)] = T.m_InstFirst[std::size_t(i) + 1] - T.m_InstFirst[std::size_t(i)];
            if (pDropped) *pDropped = Drop;
            return true;
        }

        // A slice: the ranges from First whose estimated rays fit Rays (one at least); the end. What is left of the batch after it, when
        // under a quarter of Rays, goes in it too (up to 25% over this frame): a slice costs a fixed part whatever its size, and the next
        // batch is picked only once this one is all in - the leftover took a frame of its own (Police: ~1.1k rays every third frame)
        static std::size_t SliceEnd(std::span<const world::gpu_range> Ranges, std::size_t First, int Rays) noexcept
        {
            std::size_t End = First;
            int Est = 0;
            while (End < Ranges.size() && (End == First || Est + Ranges[End].m_Rays <= Rays)) Est += Ranges[End++].m_Rays;
            int Rest = 0;
            for (std::size_t k = End; k < Ranges.size(); ++k) Rest += Ranges[k].m_Rays;
            return Rest > 0 && Rest <= Rays / 4 ? Ranges.size() : End;
        }

        // The rays the world may pick for a batch: batch_frames_v frames of slices
        static constexpr int batch_frames_v = 4;
        int BatchRays(void) const noexcept { return m_Budget * batch_frames_v; }
        std::size_t LinkSolves(void) const noexcept { return m_LinkSolves; }     // (tests)
        int LinkState(void) const noexcept { return m_LinkState; }

        // The rays the next slice may have: what fits BudgetMs at the last slice's rate. Within it the budget grows (at most doubling), past
        // it it shrinks (at most halving). A slice within it never shrinks it: a small one is bound by the rays' latency, not the GPU's
        // width, and its rate says too little (one rate for every batch kept the budget at a small batch's: 3.4k rays a frame in Police)
        int Budget(float BudgetMs) noexcept
        {
            if (m_bMeasured && m_LastRays > 0)
            {
                const double Fit = double(m_LastRays) * double(BudgetMs) / std::max(m_LastMs, 1e-3);
                // (within it, a quarter more at least: a latency-bound slice takes about the same time with more rays, so the fit alone
                // never finds the GPU's width; past it the next slice shrinks to the fit)
                const double B   = m_LastMs <= BudgetMs ? std::max(1.25 * m_Budget, std::min(Fit, 2.0 * m_Budget)) : std::max(Fit, 0.5 * m_Budget);
                m_Budget = std::clamp(int(B), 1024, 1 << 22);
            }
            m_bMeasured = false;
            return m_Budget;
        }

        // G4: the shown light of every instance (UpdateShown's: pass 0, Passes filter passes, saturated) into the renderer's ring: Light (3
        // vec4 a surfel) and Moments (16 uints), from surfel Base of them (this frame's region). Places: per world instance its place in
        // the region (its mean there, its surfels after it; -1: not drawn). After Submit (the slice committed this frame is in it)
        bool Show(xgpu::window& Window, std::span<const int> Places, xgpu::buffer& Light, xgpu::buffer& Moments, int Base, int Passes, float Saturation) noexcept
        {
            if (m_nShown == 0 || m_Data.empty()) return false;
            std::vector<int> P(Places.begin(), Places.end());
            P.resize(2 * std::size_t(std::max<std::uint32_t>(m_Data[1], 1)), -1);     // (per instance: its place, its mean or -1)
            auto& PlaceB = m_PlaceB2[m_nShows++ % 2];                  // (two: the frame before may still read its copy)
            if (!Fit(PlaceB, int(sizeof(int)), P.size()) || PlaceB.MemoryMapFrameRegion(0, int(P.size()), [&](void* p) { std::memcpy(p, P.data(), P.size() * sizeof(int)); })) return false;
            xgpu::buffer* Bufs[] = { &m_DataB, &m_OwnerB, &m_ScratchB, &PlaceB, &Light, &Moments };
            struct { int m_Mode, m_N, m_Src, m_Base; float m_Saturation; int m_Pad0, m_Pad1, m_Pad2; } Push{ 0, int(m_nShown), 1, Base, Saturation, 0, 0, 0 };
            const int Groups = int((m_nShown + 63) / 64);
            static const bool s_bTimes = std::getenv("XLION_GPU_TIMES") != nullptr;     // measurement: the three steps' GPU time
            if (s_bTimes && !m_ShowTimesOk) { xgpu::device D; Window.getDevice(D); m_ShowTimesOk = D.Create(m_ShowTimes, { .m_Count = 4 }) == nullptr; }
            const bool bT = s_bTimes && m_ShowTimesOk;
            if (bT)
            {
                std::vector<std::uint64_t> Ns;
                if (m_ShowTimes.getResults(Ns) && Ns.size() >= 4 && Ns[3] > Ns[0])
                {
                    for (int k = 0; k < 3; ++k) m_ShowSum[k] += (Ns[k + 1] - Ns[k]) * 1e-6;
                    if (++m_nShowSum == 120)
                    {
                        std::size_t nDrawn = 0; for (std::size_t i = 0; i < P.size(); i += 2) nDrawn += P[i] >= 0;
                        std::printf("GISHOW %zu surfels: pass 0 %.2f, %d filter passes %.2f, into the rings %.2f ms; %zu of %zu instances drawn\n", m_nShown, m_ShowSum[0] / 120, Passes, m_ShowSum[1] / 120, m_ShowSum[2] / 120, nDrawn, P.size() / 2);
                        std::fflush(stdout); m_nShowSum = 0; for (auto& v : m_ShowSum) v = 0;
                    }
                }
                Window.WriteTimestamp(m_ShowTimes, 0);
            }
            Window.Dispatch(m_Filter, Bufs, &Push, sizeof(Push), Groups);             // pass 0 into the first array
            if (bT) Window.WriteTimestamp(m_ShowTimes, 1);
            const int nP = std::clamp(Passes, 0, 16);
            for (int p = 1; p <= nP; ++p) { Push.m_Mode = 1; Push.m_Src = (p - 1) % 2; Window.Dispatch(m_Filter, Bufs, &Push, sizeof(Push), Groups); }
            if (bT) Window.WriteTimestamp(m_ShowTimes, 2);
            Push.m_Mode = 2; Push.m_Src = nP % 2;
            Window.Dispatch(m_Filter, Bufs, &Push, sizeof(Push), int((16 * m_nShown + 63) / 64));      // (16 invocations a surfel: coalesced stores)
            if (bT) Window.WriteTimestamp(m_ShowTimes, 3);
            return true;
        }

        // The GPU's light into the world's arrays (InstanceLight, InstanceValid, InstanceMoments, BasisFractions, Estimator, MomentCounts):
        // copied in this frame, there at the next Collect (its frame done). Its tests, its dump, the CPU taking over (m_bGPUOwned off)
        bool ReadBack(xgpu::window& Window, std::uint64_t Frame) noexcept
        {
            if (m_Data.empty()) return false;
            if (!Fit(m_ReadB, int(sizeof(std::uint32_t)), m_Data.size())) return false;
            const struct { int m_nWords, m_Pad0, m_Pad1, m_Pad2; } Push{ int(m_Data.size()), 0, 0, 0 };
            xgpu::buffer* B[] = { &m_DataB, &m_ReadB };
            Window.Dispatch(m_ReadBackC, B, &Push, sizeof(Push), int((m_Data.size() + 63) / 64));
            m_ReadFrame = Frame; m_ReadWords = m_Data.size();
            return true;
        }
        bool ReadBackPending( void ) const noexcept { return m_ReadFrame != ~0ull; }

        bool            InFlight    ( void ) const noexcept { return m_nSlicesOut > 0; }
        bool            BatchDone   ( void ) const noexcept { return m_NextRange >= m_Ranges.size(); }    // the batch's slices are all dispatched
        double          LastMs      ( void ) const noexcept { return m_LastMs; }        // the last slice's GPU time
        double          RaysPerMs   ( void ) const noexcept { return m_RaysPerMs; }
        int             Packs       ( void ) const noexcept { return m_nPacks; }       // the scene packed in full (a structural change)
        int             Relevels    ( void ) const noexcept { return m_nRelevels; }    // level switches done in place (Relevel)
        int             Retlases    ( void ) const noexcept { return m_nRetlas; }      // the TLAS made again (a move)
        int             Batches     ( void ) const noexcept { return m_nBatches; }
        int             Slices      ( void ) const noexcept { return m_nSlices; }
        int             Ops         ( void ) const noexcept { return m_nOps; }          // the estimator changes applied (one per surfel)
        int             LastRays    ( void ) const noexcept { return m_LastRays; }
        int             LastTraced  ( void ) const noexcept { return m_LastTraced; }   // the last slice's surfels traced
        std::size_t     DataWords   ( void ) const noexcept { return m_Data.size(); }
        std::size_t     UploadWords ( void ) const noexcept { return m_UploadWords; }  // the last slice's refresh

        // Lighting's traced visibility (documentation/lighting_gi_interface_notice.md, P4), read only: the BVH as the GI's rays walk it
        // (shaders/xlionrender_gi_bvh.glsl: GINode, GITri4, GIInstance, the TLAS leaves; g_GIBvhTlasRoot, g_GIBvhPrimOffset), absolute world
        // space. Valid for the frame it is asked in, after the GI's update (a pack or a move may replace or rebuild them): ask every frame.
        // Coverage: the opaque rigid meshes that have GI. bEmpty: never walk it (an empty scene's BVH is a placeholder: its root walks off
        // the buffers and loses the device)
        // m_pData: the GI's data (xlionrender_gi_pass_common.glsl GD: the instances, meshes, surfels, their light, the lights, the sky) for
        // a pass that shades its hits with the GI's model (reflections: xlionrender_refl_trace_comp.glsl); nullptr while there is none
        struct bvh_view { const xgpu::buffer *m_pNodes, *m_pTri4, *m_pInstances, *m_pTlas; int m_TlasRoot, m_PrimOffset; bool m_bEmpty; const xgpu::buffer* m_pData = nullptr; };
        bvh_view getBvh( void ) const noexcept
        {
            return { &m_Nodes, &m_Tri4, &m_Inst, &m_Tlas, m_Scene.m_TlasRoot, m_Scene.m_PrimOffset, m_Scene.m_bEmpty || !m_Nodes.m_Private || !m_Tlas.m_Private, m_DataB.m_Private ? &m_DataB : nullptr };
        }

    protected:

        // A device-local buffer made with V (GPU_READ: xGPU copies it through a staging buffer; the GPU reads its own memory, not the host's
        // across the bus: Police's rays were 70 times slower than the Cornell box's with the scene in host memory)
        template <class T> bool Local(xgpu::buffer& B, const std::vector<T>& V) noexcept
        {
            std::vector<T> Tmp = V;
            if (Tmp.empty()) Tmp.push_back(T{});
            B = {};
            return m_pDevice->Create(B, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = int(sizeof(T)), .m_EntryCount = int(Tmp.size()), .m_pData = static_cast<void*>(Tmp.data()) }) == nullptr;
        }
        // B holds at least n entries of Size bytes (made, or made larger: half again, nothing in flight reads it)
        bool Fit(xgpu::buffer& B, int Size, std::size_t n) noexcept
        {
            n = std::max<std::size_t>(n, 1);
            if (B.m_Private && std::size_t(B.getEntryCount()) >= n) return true;
            const int Count = int(n + n / 2);
            if (B.m_Private) return B.Resize(Count) == nullptr;
            return m_pDevice->Create(B, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = Size, .m_EntryCount = Count }) == nullptr;
        }

        // B (device-local, the GPU's only) holds at least n entries of Size bytes (made larger: half again, nothing in flight reads it)
        bool FitLocal(xgpu::buffer& B, int Size, std::size_t n) noexcept
        {
            n = std::max<std::size_t>(n, 1);
            const std::size_t Words = std::size_t(Size / 4) * n;
            if (B.m_Private && std::size_t(B.getEntryCount()) >= Words) return true;
            return Local(B, std::vector<std::uint32_t>(Words + Words / 2, 0u));
        }

        // Everything again (ponytail: any structural change packs the whole scene; incremental records and TLAS refits when it shows), the
        // light moved from the last layout: an instance at the same level its arrays copied, at another level re-levelled (a new one: the
        // world's, as it was made)
        bool Pack(xgpu::window& Window, world& W, const lights& L) noexcept
        {
            // measurement only (GI review): the stages of a pack, printed when the whole takes more than 10 ms
            const auto PT0 = std::chrono::steady_clock::now();
            double PMs[4] = {};
            auto PLap = [&, T = PT0](int k) mutable { const auto N = std::chrono::steady_clock::now(); PMs[k] += std::chrono::duration<double, std::milli>(N - T).count(); T = N; };
            struct pack_report { double* p; std::chrono::steady_clock::time_point t0; ~pack_report() { const double All = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (All > 10) std::printf("GIPACK %.1f ms: BVHs %.1f, surfel data %.1f, uploads %.1f, light migration %.1f\n", All, p[0], p[1], p[2], p[3]), std::fflush(stdout); } } PackReport{ PMs, PT0 };
            std::vector<std::uint32_t> Old = std::move(m_Data);
            W.PackGPU(m_Scene);
            PLap(0);
            W.PackGPUSurfels(m_Data, L);
            PLap(1);
            std::vector<std::uint32_t> Owners;
            for (int i = 0; i < W.InstanceCount(); ++i)
            {
                const std::uint32_t R = m_Data[0] + std::uint32_t(i) * world::gpu_inst_v;
                if (m_Data[R + 40] == 0) continue;
                m_Data[R + 46] = std::uint32_t(Owners.size());
                Owners.insert(Owners.end(), W.InstanceLight(i).size(), std::uint32_t(i));
            }
            m_nShown = Owners.size();
            {   // GI sections: per ghost { its instance, its surfel, its owner's instance, its surfel } (Submit copies them after each commit)
                std::vector<std::array<std::uint32_t, 4>> Ghosts;
                for (int i = 0; i < W.InstanceCount(); ++i)
                {
                    if (!W.IsAlive(i)) continue;
                    const auto& Gh = W.GhostOwners(i);
                    const std::uint32_t nOwned = W.OwnedSurfels(i);
                    for (std::size_t k = 0; k < Gh.size(); ++k) Ghosts.push_back({ std::uint32_t(i), nOwned + std::uint32_t(k), std::uint32_t(Gh[k][0]), std::uint32_t(Gh[k][1]) });
                }
                m_nGhosts = Ghosts.size();
                if (m_nGhosts && !Local(m_GhostB, Ghosts)) return false;
            }
            m_ShownRoom = 0;                                            // (the shown light's scratch: room for every instance at its finer level, Relevel)
            for (int i = 0; i < W.InstanceCount(); ++i) if (m_Data[m_Data[0] + std::uint32_t(i) * world::gpu_inst_v + 40] != 0) { const std::uint32_t M = m_Data[m_Data[0] + std::uint32_t(i) * world::gpu_inst_v]; m_ShownRoom += std::max(m_Data[M + 6], m_Data[M + 8]); }
            m_Gen = W.GPUGen(); m_LevelGen = W.GPULevelGen(); m_nPoints = L.m_Points.size();
            m_EmitGen.assign(std::size_t(W.InstanceCount()), 0); m_MatGen.assign(std::size_t(W.InstanceCount()), 0);
            for (int i = 0; i < W.InstanceCount(); ++i) { m_EmitGen[std::size_t(i)] = W.EmitGen(i); m_MatGen[std::size_t(i)] = W.MatGen(i); }
            ++m_nPacks;
            m_OldDataB = std::move(m_DataB);
            m_XformGen = W.GPUXformGen();
            m_EmRoom = world::GPUEmitterRoom(m_Data[15]);
            {   // the nodes with a reserve after the TLAS's (a move rebuilds it there: Retlas)
                auto Nodes = m_Scene.m_Nodes;
                Nodes.resize(Nodes.size() + 2 * (Nodes.size() - std::size_t(m_Scene.m_nMeshNodes)) + 64);
                m_NodeCap = Nodes.size();
                if (!Local(m_Nodes, Nodes)) return false;
            }
            if (!Local(m_Tri4, m_Scene.m_Tri4) || !Local(m_Inst, m_Scene.m_Instances) || !Local(m_Tlas, m_Scene.m_TlasLeaves) || !Local(m_DataB, m_Data)
                || !Local(m_OwnerB, Owners) || !Local(m_ScratchB, std::vector<float>(std::max<std::size_t>(18 * std::max(m_nShown, m_ShownRoom), 1), 0.0f))) return false;
            PLap(2);
            if (Old.empty() || !m_OldDataB.m_Private) return true;
            // the light from the old layout
            std::vector<std::array<int, 4>> Jobs;
            const std::uint32_t nOld = Old[1];
            for (int i = 0; i < W.InstanceCount() && std::uint32_t(i) < nOld; ++i)
            {
                const std::uint32_t Ro = Old[0] + std::uint32_t(i) * world::gpu_inst_v, Rn = m_Data[0] + std::uint32_t(i) * world::gpu_inst_v;
                if (Old[Ro + 40] == 0 || m_Data[Rn + 40] == 0) continue;
                const std::uint32_t no = Old[Old[Ro] + 6 + 2 * Old[Ro + 1]], nn = std::uint32_t(W.InstanceLight(i).size());
                if (Old[Ro + 1] != m_Data[Rn + 1]) { Jobs.push_back({ 1, int(Ro), int(Rn), 0 }); continue; }
                if (no != nn) continue;
                const std::uint32_t Words[6][2] = { { 2, 3 }, { 3, 1 }, { 4, 2 * moment_texels_v }, { 41, 9 }, { 42, world::gpu_est_v }, { 43, moment_texels_v } };
                for (const auto& [w, k] : Words) Jobs.push_back({ 0, int(Old[Ro + w]), int(m_Data[Rn + w]), int(k * nn) });
            }
            if (Jobs.empty()) return true;
            if (!Fit(m_JobB, int(sizeof(Jobs[0])), Jobs.size()) || m_JobB.MemoryMap(0, int(Jobs.size()), [&](void* p) { std::memcpy(p, Jobs.data(), Jobs.size() * sizeof(Jobs[0])); })) return false;
            xgpu::buffer* B[] = { &m_OldDataB, &m_DataB, &m_JobB };
            const int Pad[4] = {};
            Window.Dispatch(m_Migrate, B, Pad, sizeof(Pad), int(Jobs.size()));
            PLap(3);
            return true;
        }

        // An instance changed level (world::GPULevelGen) and nothing else did: no repack. The arrays have room for both levels (PackGPUSurfels),
        // so the record keeps its offsets. The switched instances' records and light (E, validity) are saved first by the migrate shader's copy
        // jobs into a scratch at the same offsets (a device-local buffer cannot be a copy's source), the new level and the shown order's words
        // are scattered, then the migrate shader's kind 1 re-levels each one in place from the scratch (a child its parent's light, a parent its
        // children's mean, the rest seeded: world::Update's switch on the CPU). false: not possible here (a full Pack)
        bool Relevel(xgpu::window& Window, world& W) noexcept
        {
            m_LevelGen = W.GPULevelGen();
            if (m_Data.empty() || !m_DataB.m_Private) return false;
            std::vector<std::array<int, 4>> Save, Move;
            std::vector<std::uint32_t> Pairs, Owners;
            for (int i = 0; i < W.InstanceCount() && std::uint32_t(i) < m_Data[1]; ++i)
            {
                const std::uint32_t R = m_Data[0] + std::uint32_t(i) * world::gpu_inst_v;
                if (m_Data[R + 40] == 0) continue;
                const std::uint32_t Lv = std::uint32_t(W.Level(i)), Old = m_Data[R + 1], M = m_Data[R];
                if (Lv != Old)
                {
                    const std::uint32_t nOld = m_Data[M + 6 + 2 * Old];
                    Save.push_back({ 0, int(R), int(R), int(world::gpu_inst_v) });
                    Save.push_back({ 0, int(m_Data[R + 2]), int(m_Data[R + 2]), int(3 * nOld) });
                    Save.push_back({ 0, int(m_Data[R + 3]), int(m_Data[R + 3]), int(nOld) });
                    Move.push_back({ 1, int(R), int(R), 0 });
                    m_Data[R + 1] = Lv; Pairs.push_back(R + 1); Pairs.push_back(Lv);
                }
                const std::uint32_t First = std::uint32_t(Owners.size());
                if (m_Data[R + 46] != First) { m_Data[R + 46] = First; Pairs.push_back(R + 46); Pairs.push_back(First); }
                Owners.insert(Owners.end(), W.InstanceLight(i).size(), std::uint32_t(i));
            }
            if (Owners.size() > std::max(m_nShown, m_ShownRoom)) return false;      // (more than the scratch has room for)
            m_nShown = Owners.size();
            if (!Local(m_OwnerB, Owners)) return false;
            if (Move.empty() && Pairs.empty()) return true;
            const int Pad[4] = {};
            if (!Move.empty())
            {
                if (!FitLocal(m_RelevelB, int(sizeof(std::uint32_t)), m_Data.size())) return false;
                if (!Fit(m_RelSaveB, int(sizeof(Save[0])), Save.size()) || m_RelSaveB.MemoryMap(0, int(Save.size()), [&](void* p) { std::memcpy(p, Save.data(), Save.size() * sizeof(Save[0])); })) return false;
                xgpu::buffer* B[] = { &m_DataB, &m_RelevelB, &m_RelSaveB };
                Window.Dispatch(m_Migrate, B, Pad, sizeof(Pad), int(Save.size()));
            }
            if (!Pairs.empty())
            {
                const int nPairs = int(Pairs.size() / 2);
                if (!Fit(m_RelStageB, int(2 * sizeof(std::uint32_t)), std::size_t(nPairs)) || m_RelStageB.MemoryMap(0, nPairs, [&](void* p) { std::memcpy(p, Pairs.data(), Pairs.size() * sizeof(std::uint32_t)); })) return false;
                const struct { int m_nPairs, m_Pad0, m_Pad1, m_Pad2; } Push{ nPairs, 0, 0, 0 };
                xgpu::buffer* Bufs[] = { &m_DataB, &m_RelStageB };
                Window.Dispatch(m_Scatter, Bufs, &Push, sizeof(Push), (nPairs + 63) / 64);
            }
            if (!Move.empty())
            {
                if (!Fit(m_RelMoveB, int(sizeof(Move[0])), Move.size()) || m_RelMoveB.MemoryMap(0, int(Move.size()), [&](void* p) { std::memcpy(p, Move.data(), Move.size() * sizeof(Move[0])); })) return false;
                xgpu::buffer* B[] = { &m_RelevelB, &m_DataB, &m_RelMoveB };
                Window.Dispatch(m_Migrate, B, Pad, sizeof(Pad), int(Move.size()));
            }
            ++m_nRelevels;
            return true;
        }

        // G4c: the instances moved (world::GPUXformGen), the meshes the same: the TLAS again into the nodes' reserve (copied as runs of
        // words: the migrate shader's copy jobs), its instances' rows and leaves, the records' placements (Refresh). false: it does not
        // fit (a full Pack)
        bool Retlas(xgpu::window& Window, world& W) noexcept
        {
            if (!W.PackGPUTlas(m_Scene) || m_Scene.m_Nodes.size() > m_NodeCap) return false;
            m_XformGen = W.GPUXformGen();
            m_bPlaced = true;
            ++m_nRetlas;
            if (!Local(m_Inst, m_Scene.m_Instances) || !Local(m_Tlas, m_Scene.m_TlasLeaves)) return false;
            const std::size_t n0 = std::size_t(m_Scene.m_nMeshNodes), nW = (m_Scene.m_Nodes.size() - n0) * sizeof(bvh::node) / 4;
            if (nW == 0) return true;
            constexpr std::size_t node_words_v = sizeof(bvh::node) / 4;
            const std::array<int, 4> Job{ 0, 0, int(n0 * node_words_v), int(nW) };
            if (!Fit(m_TlasStageB, int(sizeof(std::uint32_t)), nW) || m_TlasStageB.MemoryMap(0, int(nW), [&](void* q) { std::memcpy(q, m_Scene.m_Nodes.data() + n0, nW * 4); })) return false;
            if (!Fit(m_TlasJobB, int(sizeof(Job)), 1) || m_TlasJobB.MemoryMap(0, 1, [&](void* q) { std::memcpy(q, &Job, sizeof(Job)); })) return false;
            xgpu::buffer* B[] = { &m_TlasStageB, &m_Nodes, &m_TlasJobB };
            const int Pad[4] = {};
            Window.Dispatch(m_Migrate, B, Pad, sizeof(Pad), 1);
            return true;
        }

        // The world's estimator changes on the GPU's light (an instance's: one op per surfel)
        bool ApplyOps(xgpu::window& Window, world& W) noexcept
        {
            if (m_Ops.empty()) return true;
            m_OpWords.clear();
            const auto& D = m_Data;
            for (const auto& O : m_Ops)
            {
                if (O.m_Inst < 0 || O.m_Inst >= W.InstanceCount() || D[D[0] + std::uint32_t(O.m_Inst) * world::gpu_inst_v + 40] == 0) continue;
                const int n = int(W.InstanceLight(O.m_Inst).size());
                const int s0 = O.m_Surf < 0 ? 0 : O.m_Surf, s1 = O.m_Surf < 0 ? n : std::min(O.m_Surf + 1, n);
                for (int s = s0; s < s1; ++s) { m_OpWords.push_back(O.m_Inst); m_OpWords.push_back(s); m_OpWords.push_back(std::bit_cast<int>(O.m_A)); m_OpWords.push_back(O.m_bMomCount); }
            }
            m_Ops.clear();
            const int nOps = int(m_OpWords.size() / 4);
            if (nOps == 0) return true;
            if (!Fit(m_OpB, int(4 * sizeof(int)), std::size_t(nOps))) return false;
            if (m_OpB.MemoryMap(0, nOps, [&](void* p) { std::memcpy(p, m_OpWords.data(), m_OpWords.size() * sizeof(int)); })) return false;
            const struct { int m_nOps, m_Pad0, m_Pad1, m_Pad2; } Push{ nOps, 0, 0, 0 };
            xgpu::buffer* Bufs[] = { &m_DataB, &m_OpB };
            Window.Dispatch(m_OpsC, Bufs, &Push, sizeof(Push), (nOps + 63) / 64);
            m_nOps += nOps;
            return true;
        }

        // What the world changed since the last slice into the resident data (and m_Data, its copy of the layout and the world's words):
        // as (index, value) pairs, written into the device-local array by the scatter shader in this frame (before the expand: the
        // dispatches' barriers). The means, the setup, the lights, the links' emitted light that changed; a changed material (the
        // renderer's captures: no repack) as runs of words, copied by the migrate shader's copy jobs
        bool Refresh(xgpu::window& Window, world& W, const lights& L) noexcept
        {
            const auto& D = m_Data;
            m_Stage.clear(); m_RunStage.clear(); m_RunJobs.clear();
            const auto Put = [&](std::uint32_t At, float f) { const std::uint32_t u = std::bit_cast<std::uint32_t>(f); m_Data[At] = u; m_Stage.push_back(At); m_Stage.push_back(u); };
            const auto PutU = [&](std::uint32_t At, std::uint32_t u) { m_Data[At] = u; m_Stage.push_back(At); m_Stage.push_back(u); };
            const auto Rec = [&](int i) { return D[0] + std::uint32_t(i) * world::gpu_inst_v; };
            for (int i = 0; i < W.InstanceCount(); ++i)
            {
                const std::uint32_t R = Rec(i);
                if (D[R + 40] == 0) continue;
                if (m_bPlaced)                                          // moved: its placement (world::GPUPlacement)
                {
                    std::uint32_t P[25];
                    W.GPUPlacement(i, P);
                    for (std::uint32_t k = 0; k < 25; ++k) PutU(R + 15 + k, P[k]);
                }
                const v3& M = W.MeanLight(i);
                Put(R + 5, M.m_X); Put(R + 6, M.m_Y); Put(R + 7, M.m_Z);
                if (W.EmitGen(i) != m_EmitGen[std::size_t(i)])          // the links' emitted light (zeros: none)
                {
                    m_EmitGen[std::size_t(i)] = W.EmitGen(i);
                    const auto& B = W.EmittedLight(i);
                    PutU(R + 47, (m_Data[R + 47] & ~32u) | (B.empty() ? 0u : 32u));     // (has links: world::GPUMaterial's bit 5)
                    for (std::uint32_t u = 0; u < std::uint32_t(W.InstanceLight(i).size()); ++u)
                        for (std::uint32_t k = 0; k < 3; ++k)
                        {
                            const v3 e = u < B.size() ? B[u][k] : v3{ 0, 0, 0 };
                            Put(D[R + 44] + 9 * u + 3 * k, e.m_X); Put(D[R + 44] + 9 * u + 3 * k + 1, e.m_Y); Put(D[R + 44] + 9 * u + 3 * k + 2, e.m_Z);
                        }
                }
                if (W.MatGen(i) != m_MatGen[std::size_t(i)])            // its material (world::GPUMaterial: the same sizes, in place)
                {
                    m_MatGen[std::size_t(i)] = W.MatGen(i);
                    std::vector<std::uint32_t> A[world::gpu_mat_arrays_v];
                    PutU(R + 47, W.GPUMaterial(i, A));
                    const auto& Word = world::gpu_mat_words_v;
                    for (int k = 0; k < world::gpu_mat_arrays_v; ++k)
                    {
                        m_RunJobs.push_back({ 0, int(m_RunStage.size()), int(D[R + Word[k]]), int(A[k].size()) });
                        std::copy(A[k].begin(), A[k].end(), m_Data.begin() + D[R + Word[k]]);
                        m_RunStage.insert(m_RunStage.end(), A[k].begin(), A[k].end());
                    }
                }
            }
            std::uint32_t Setup[world::gpu_header_v] = {};
            W.PackGPUSetup(Setup);
            for (const std::uint32_t w : { 5u, 7u, 8u, 10u, 11u, 12u, 13u }) PutU(w, Setup[w]);
            {   // G5: the emitter table (where they are, their radiance now: the light sampling's), one run
                std::vector<std::uint32_t> Em;
                W.GPUEmitters(Em);
                PutU(15, std::uint32_t(Em.size() / world::gpu_emitter_v));
                if (!Em.empty())
                {
                    m_RunJobs.push_back({ 0, int(m_RunStage.size()), int(D[14]), int(Em.size()) });
                    std::copy(Em.begin(), Em.end(), m_Data.begin() + D[14]);
                    m_RunStage.insert(m_RunStage.end(), Em.begin(), Em.end());
                }
            }
            // the lights (the same count: PackGPUSurfels' layout), the sky's flag
            std::uint32_t At = D[2];
            const auto V = [&](const v3& v) { Put(At, v.m_X); Put(At + 1, v.m_Y); Put(At + 2, v.m_Z); At += 3; };
            V(L.m_SunDirection); V(L.m_SunColor); V(L.m_Sky);
            for (const auto& s : L.m_SkyOct) V(s);
            for (const auto& P : L.m_Points) { V(P.m_Position); V(P.m_Color); Put(At++, P.m_Radius); V(P.m_Dir); Put(At++, P.m_CosOuter); Put(At++, P.m_CosInner); }
            PutU(4, L.m_bSkyOct ? 1u : 0u);
            m_bPlaced = false;
            const int nPairs = int(m_Stage.size() / 2);
            // Two staging buffers, one a frame: the frame in flight may still read the other, so this one is mapped without a wait (MemoryMap
            // waited for the GPU's frame: 6 ms a frame on the Bistro, its whole scene)
            auto& StageB = m_StageB2[++m_nSubmits % 2];
            if (!Fit(StageB, int(2 * sizeof(std::uint32_t)), std::size_t(nPairs))) return false;
            if (StageB.MemoryMapFrameRegion(0, nPairs, [&](void* p) { std::memcpy(p, m_Stage.data(), m_Stage.size() * sizeof(std::uint32_t)); })) return false;
            const struct { int m_nPairs, m_Pad0, m_Pad1, m_Pad2; } Push{ nPairs, 0, 0, 0 };
            xgpu::buffer* Bufs[] = { &m_DataB, &StageB };
            Window.Dispatch(m_Scatter, Bufs, &Push, sizeof(Push), (nPairs + 63) / 64);
            m_UploadWords = std::size_t(nPairs) + m_RunStage.size();
            if (m_RunJobs.empty()) return true;
            if (!Fit(m_RunB, int(sizeof(std::uint32_t)), m_RunStage.size()) || m_RunB.MemoryMap(0, int(m_RunStage.size()), [&](void* p) { std::memcpy(p, m_RunStage.data(), m_RunStage.size() * sizeof(std::uint32_t)); })) return false;
            if (!Fit(m_RunJobB, int(sizeof(m_RunJobs[0])), m_RunJobs.size()) || m_RunJobB.MemoryMap(0, int(m_RunJobs.size()), [&](void* p) { std::memcpy(p, m_RunJobs.data(), m_RunJobs.size() * sizeof(m_RunJobs[0])); })) return false;
            xgpu::buffer* RBufs[] = { &m_RunB, &m_DataB, &m_RunJobB };
            const int Pad[4] = {};
            Window.Dispatch(m_Migrate, RBufs, Pad, sizeof(Pad), int(m_RunJobs.size()));
            return true;
        }

        // The read back's words into the world's arrays (the layouts of the pack it was copied with)
        void FinishReadBack(world& W) noexcept
        {
            m_ReadFrame = ~0ull;
            std::vector<std::uint32_t> G(m_ReadWords);
            if (G.empty() || m_ReadB.MemoryMap(0, int(G.size()), [&](void* p) { std::memcpy(G.data(), p, G.size() * sizeof(std::uint32_t)); })) return;
            for (int i = 0; i < W.InstanceCount() && std::uint32_t(i) < G[1]; ++i)
            {
                const std::uint32_t R = G[0] + std::uint32_t(i) * world::gpu_inst_v;
                if (G[R + 40] == 0 || G[R + 1] != std::uint32_t(W.Level(i)) || G[G[R] + 6 + 2 * G[R + 1]] != W.InstanceLight(i).size()) continue;
                W.SetGPUState(i, &G[G[R + 2]], &G[G[R + 3]], &G[G[R + 4]], &G[G[R + 41]], &G[G[R + 42]], &G[G[R + 43]]);
            }
        }

        xgpu::device*                       m_pDevice = nullptr;
        xgpu::shader                        m_Shader[14];
        xgpu::compute                       m_LinksC;           // experiment D: the link build (optional: shaders::m_Links)
        xgpu::buffer                        m_LinkItemB, m_LinkB, m_LinkDB, m_LinkNextB;
        xgpu::buffer                        m_LinkDLocalB, m_LinkItemLocalB;     // (the device-local copies: CopyLinksLocal; m_LinkB is device-local)
        xgpu::buffer                        m_LinkHitB;         // per item its hit clusters (hits_v, device-local): its first bounce, lit again on a light change
        xgpu::buffer                        m_LinkReadB, m_LinkHitReadB;         // (tests: the links read back, m_bLinkReadback)
        xgpu::buffer                        m_LinkDirB;         // per item its first bounce of the lights, F (RelightLinks)
        xgpu::buffer                        m_LinkDirTmpB, m_LinkInstFirstB;   // F before its neighbors (mode 1 -> 4); per instance its first item
        xgpu::buffer                        m_LinkListB, m_LinkChangedB, m_LinkFlagB;          // the local rebuild: its items, the changed instances, the flags read back
        std::vector<std::uint32_t>          m_LinkedMatGen;                                    // per instance its material's version the links were built against
        std::vector<xmath::fmat4>           m_LinkedL2W;                                       // ... and its placement
        std::vector<std::uint32_t>          m_LocalList;                                       // the items a change touches
        int                                 m_LocalState = 0;                                  // 0 idle, 1 flags out, 2 building, 3 done (copied, relit)
        std::uint64_t                       m_LocalFrame = 0, m_LinkLocalKey = 0;
        std::size_t                         m_LocalCursor = 0, m_nLocalRebuilds = 0, m_nLocalRows = 0;
        std::vector<char>                   m_LocalNear;                                       // per item: near a moved object (the CPU's part of the flags)
        bool                                m_bLinkDirValid = false;
        std::uint64_t                       m_LinkDirKey = 0;
        std::size_t                         m_RelightPerFrame = 16384;     // the lights' relight budget, items a frame (SetRelightPerFrame)
        std::size_t                         m_RelightCursor = 0, m_RelightOwed = 0;     // where the round is, the items it still owes
        std::size_t                         m_nLinkBuilds = 0;
        std::size_t                         m_nRelights = 0;
        bool                                m_bLinksLocal = false;
        bool                                m_bLinks = false, m_bSolve = false;
        xgpu::compute                       m_SolveC;
        std::vector<std::array<int, 2>>     m_LinkItems;        // PumpLinks: the surfels with links, its state (0 settling, 1 building, 2 solving, 3 off)
        int                                 m_LinkState = 0, m_LastMeans = -1;
        std::size_t                         m_LinkCursor = 0, m_LinkSolves = 0;
        std::uint64_t                       m_LinkKey = ~0ull, m_LinkSince = 0, m_LinkReady = 0;
        std::chrono::steady_clock::time_point m_LinkT0{};
        xgpu::compute                       m_Pass, m_Scatter, m_Commit, m_OpsC, m_Filter, m_Expand, m_Means, m_Cube, m_Migrate, m_ReadBackC;
        std::array<double, 3>               m_CpuTrace{}; int m_nCpuTrace = 0;       // (XLION_GI_CPU_TRACE)
        xgpu::compute                       m_GhostsC;          // GI sections: the ghosts take their owners' set (xlionrender_gi_ghosts_comp.glsl)
        xgpu::compute                       m_ProbesC;          // the probe grid (UpdateProbes)
        xgpu::buffer                        m_ProbeB, m_ProbeHdrB2[2], m_ProbeReadB;
        std::size_t                         m_ProbeCursor = 0;
        bool                                m_bProbes = false;
        xgpu::buffer                        m_GhostB;
        std::size_t                         m_nGhosts = 0;
        bool                                m_bGhosts = false;
        xgpu::timestamps                    m_Times;
        xgpu::timestamps                    m_ShowTimes;        // measurement (XLION_GPU_TIMES): Show's steps
        bool                                m_ShowTimesOk = false;
        double                              m_ShowSum[3] = {};
        int                                 m_nShowSum = 0;
        xgpu::buffer                        m_Nodes, m_Tri4, m_Inst, m_Tlas, m_DataB, m_OldDataB, m_OwnerB, m_ScratchB;     // (device-local)
        xgpu::buffer                        m_ItemB, m_Results, m_RangeB, m_StatsB, m_JobB, m_OpB, m_ReadB;
        xgpu::buffer                        m_StageB2[2], m_CubeB2[2];        // (one a frame: mapped without waiting for the GPU)
        std::uint64_t                       m_nSubmits = 0;
        xgpu::buffer                        m_CountB2[2], m_MeansB2[2], m_PlaceB2[2];     // two copies: what the CPU writes or reads every frame (one frame in flight)
        struct slice_out { std::uint64_t m_Frame = 0; std::size_t m_First = 0, m_Count = 0; };
        slice_out                           m_SlicesOut[2] = {};    // dispatched, their stats not read yet (Collect, lag_frames_v on)
        int                                 m_nSlicesOut = 0, m_nShows = 0;
        std::uint64_t                       m_MeansFrame[2] = { ~0ull, ~0ull };     // the frame each copy of the means was written in (~0: read)
        std::uint32_t                       m_MeansCount[2] = {};
        world::gpu_scene                    m_Scene;
        std::vector<std::uint32_t>          m_Data;             // PackGPUSurfels' array: the layout, the world's words (the light is the GPU's)
        std::vector<world::gpu_range>       m_Ranges;           // the batch
        std::size_t                         m_NextRange = 0, m_SliceFirst = 0, m_SliceCount = 0;    // its next range; the slice in flight
        std::vector<world::gpu_range_stats> m_StatsOut;
        std::vector<float>                  m_MeansOut;
        std::vector<world::gpu_op>          m_Ops;              // the world's estimator changes not applied yet
        std::vector<int>                    m_OpWords;
        std::vector<std::uint32_t>          m_Stage;            // Refresh's changed words: (index, value) pairs
        std::vector<std::uint32_t>          m_EmitGen;          // per instance: world::EmitGen when its emitted light last went up
        std::vector<std::uint32_t>          m_MatGen;           // per instance: world::MatGen when its material last went up
        std::vector<std::uint32_t>          m_RunStage;         // Refresh's runs of words (a material), their copy jobs
        std::vector<std::array<int, 4>>     m_RunJobs;
        xgpu::buffer                        m_RunB, m_RunJobB, m_TlasStageB, m_TlasJobB;
        std::uint64_t                       m_XformGen = 0;
        std::size_t                         m_NodeCap = 0;      // the nodes' buffer: the meshes', the TLAS's, its reserve
        std::size_t                         m_EmRoom = 0;       // the emitter table's room (world::GPUEmitterRoom)
        bool                                m_bPlaced = false;  // Retlas: the records' placements go up (Refresh)
        int                                 m_nRetlas = 0;
        std::uint64_t                       m_Gen = 0, m_ReadFrame = ~0ull;
        std::size_t                         m_nPoints = 0, m_nShown = 0, m_UploadWords = 0, m_ReadWords = 0;
        int                                 m_LastRays = 0, m_LastTraced = 0, m_nPacks = 0, m_nBatches = 0, m_nSlices = 0, m_nOps = 0;
        double                              m_LastMs = 0, m_RaysPerMs = 0;
        int                                 m_Budget = 32768;   // (until measured)
        bool                                m_bMeasured = false;
        long long                           m_Diag[7] = {};     // (measurement only, setup::m_bAblateStats: hits, read entries, probes, accepted, BVH nodes, rays, unsupported)
        int                                 m_nDiag = 0;
        std::uint64_t                       m_LevelGen = 0;     // world::GPULevelGen at the last Pack or Relevel
        std::size_t                         m_ShownRoom = 0;    // the shown light's room: every instance at its finer level (Relevel)
        xgpu::buffer                        m_RelevelB, m_RelSaveB, m_RelMoveB, m_RelStageB;     // Relevel: the saved light (device), its jobs, its words
        int                                 m_nRelevels = 0;
    };
}

#endif
