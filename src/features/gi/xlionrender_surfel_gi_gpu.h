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
        struct shaders { std::span<const std::uint32_t> m_Pass, m_Scatter, m_Commit, m_Ops, m_Filter, m_Expand, m_Means, m_Cube, m_Migrate, m_ReadBack; };

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
                && Make(m_Shader[9], m_ReadBackC, S.m_ReadBack, 2, 16) && !Device.Create(m_Times, { .m_Count = 2 });
        }

        // Before world::Update: the slice of an earlier frame (done: that frame's PageFlip waited for it) and the means into the world; a
        // read back asked for, into its arrays. true: a slice was delivered
        bool Collect(world& W, std::uint64_t Frame) noexcept
        {
            if (m_ReadFrame != ~0ull && Frame > m_ReadFrame) FinishReadBack(W);
            if (m_InFlight == ~0ull || Frame <= m_InFlight) return false;
            m_InFlight = ~0ull;
            std::vector<std::uint64_t> Ns;
            if (m_SliceCount > 0 && m_Times.getResults(Ns) && Ns.size() >= 2 && Ns[0] != ~0ull && Ns[1] != ~0ull && Ns[1] > Ns[0])
            {
                m_LastMs = double(Ns[1] - Ns[0]) * 1e-6;
                m_RaysPerMs = double(m_LastRays) / std::max(m_LastMs, 1e-3);
                m_bMeasured = true;
            }
            if (m_bMeansIn)
            {
                m_MeansOut.resize(std::size_t(5) * m_Data[1]);
                if (!m_MeansOut.empty() && !m_MeansB.MemoryMap(0, int(m_MeansOut.size()), [&](void* p) { std::memcpy(m_MeansOut.data(), p, m_MeansOut.size() * sizeof(float)); })) W.SetGPUMeans(m_MeansOut);
                m_bMeansIn = false;
            }
            if (m_SliceCount == 0) return false;
            m_StatsOut.resize(m_SliceCount);
            if (m_StatsB.MemoryMap(int(m_SliceFirst), int(m_SliceCount), [&](void* p) { std::memcpy(m_StatsOut.data(), p, m_StatsOut.size() * sizeof(world::gpu_range_stats)); })) return false;
            W.DeliverGPURanges(std::span<const world::gpu_range>(m_Ranges).subspan(m_SliceFirst, m_SliceCount), m_StatsOut);
            m_LastRays = 0; m_LastTraced = 0;
            for (const auto& S : m_StatsOut) { m_LastRays += S.m_Rays; m_LastTraced += S.m_Traced; }
            m_SliceCount = 0;
            return true;
        }

        // After world::Update: its changes up; with no slice in flight, the next slice of the batch (a new batch when it is all traced: the
        // world's pick) expanded, traced and committed in this frame, BudgetMs of the GPU's time (0: all of the batch), the cubes and the
        // means after it (dispatches between render passes). Eye: the update's (the tiers). true: a slice was dispatched
        bool Submit(xgpu::window& Window, world& W, const lights& L, std::uint64_t Frame, float BudgetMs, const v3& Eye) noexcept
        {
            { auto Ops = W.TakeGPUOps(); m_Ops.insert(m_Ops.end(), Ops.begin(), Ops.end()); }
            if (m_InFlight != ~0ull) return false;
            m_OldDataB = {};                                            // (the last pack's old layout: its frame is done)
            if (W.GPUGen() != m_Gen || L.m_Points.size() != m_nPoints || m_Data.empty() || W.EmitterCount() > m_EmRoom) { if (!Pack(Window, W, L)) return false; }     // (the emitter table past its room too)
            else if (W.GPUXformGen() != m_XformGen && !Retlas(Window, W) && !Pack(Window, W, L)) return false;     // (moved: the TLAS; its reserve full: all)
            if (!ApplyOps(Window, W) || !Refresh(Window, W, L)) return false;
            bool bSlice = false;
            if (m_NextRange >= m_Ranges.size())
            {
                const auto R = W.GPURanges();
                m_Ranges.assign(R.begin(), R.end());
                m_NextRange = 0;
                if (!m_Ranges.empty())
                {
                    std::vector<std::array<int, 8>> G(m_Ranges.size());
                    for (std::size_t k = 0; k < G.size(); ++k) { const auto& r = m_Ranges[k]; G[k] = { r.m_Inst, r.m_Level, r.m_First, r.m_Count, r.m_Flags, r.m_History, int(r.m_Seed), r.m_Start }; }
                    std::vector<world::gpu_range_stats> Zeros(m_Ranges.size());
                    if (!Fit(m_RangeB, int(sizeof(G[0])), G.size()) || m_RangeB.MemoryMap(0, int(G.size()), [&](void* p) { std::memcpy(p, G.data(), G.size() * sizeof(G[0])); })) return false;
                    if (!Fit(m_StatsB, int(sizeof(world::gpu_range_stats)), Zeros.size()) || m_StatsB.MemoryMap(0, int(Zeros.size()), [&](void* p) { std::memcpy(p, Zeros.data(), Zeros.size() * sizeof(Zeros[0])); })) return false;
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
                if (!FitLocal(m_ItemB, int(sizeof(world::gpu_item)), std::size_t(nThreads)) || !FitLocal(m_Results, int(sizeof(world::gpu_result)), std::size_t(nThreads))
                    || !Fit(m_CountB, int(sizeof(std::uint32_t)), 4) || m_CountB.MemoryMap(0, 4, [&](void* p) { std::memcpy(p, Zero, sizeof(Zero)); })) return false;
                const struct { int m_RangeFirst, m_nRanges, m_nThreads, m_StartBase; float m_Eye[4]; } EPush{ int(m_SliceFirst), int(m_SliceCount), nThreads, m_Ranges[m_SliceFirst].m_Start, { Eye.m_X, Eye.m_Y, Eye.m_Z, 0 } };
                const struct { int m_nItems, m_TlasRoot, m_PrimOffset, m_First; } Push{ nThreads, m_Scene.m_TlasRoot, m_Scene.m_PrimOffset, 0 };
                const struct { int m_nItems, m_First, m_Count, m_Pad; } CPush{ nThreads, 0, nThreads, 0 };
                xgpu::buffer* EBufs[] = { &m_DataB, &m_RangeB, &m_ItemB, &m_CountB };
                xgpu::buffer* Bufs[]  = { &m_Nodes, &m_Tri4, &m_Inst, &m_Tlas, &m_DataB, &m_ItemB, &m_Results, &m_CountB };
                xgpu::buffer* CBufs[] = { &m_DataB, &m_ItemB, &m_Results, &m_StatsB, &m_CountB };
                Window.WriteTimestamp(m_Times, 0);
                Window.Dispatch(m_Expand, EBufs, &EPush, sizeof(EPush), (nThreads + 63) / 64);
                Window.Dispatch(m_Pass, Bufs, &Push, sizeof(Push), world::GPUGroups(std::size_t(nThreads)));
                Window.Dispatch(m_Commit, CBufs, &CPush, sizeof(CPush), (nThreads + 63) / 64);
                Window.WriteTimestamp(m_Times, 1);
                m_InFlight = Frame;
                ++m_nSlices;
                bSlice = true;
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
                if (!Fit(m_CubeB, int(sizeof(Cubes[0])), Cubes.size()) || m_CubeB.MemoryMap(0, int(Cubes.size()), [&](void* p) { std::memcpy(p, Cubes.data(), Cubes.size() * sizeof(Cubes[0])); })) return false;
                xgpu::buffer* B[] = { &m_DataB, &m_CubeB };
                const int Pad[4] = {};
                Window.Dispatch(m_Cube, B, Pad, sizeof(Pad), int(Cubes.size()));
            }
            if (m_Data[1] > 0)
            {
                if (!Fit(m_MeansB, int(sizeof(float)), std::size_t(5) * m_Data[1])) return false;
                xgpu::buffer* B[] = { &m_DataB, &m_MeansB };
                const int Pad[4] = {};
                Window.Dispatch(m_Means, B, Pad, sizeof(Pad), int(m_Data[1]));
                m_bMeansIn = true;
                if (m_InFlight == ~0ull) m_InFlight = Frame;             // (the means come back the frame after, as a slice)
            }
            return bSlice;
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
            P.resize(std::size_t(std::max<std::uint32_t>(m_Data[1], 1)), -1);
            if (!Fit(m_PlaceB, int(sizeof(int)), P.size()) || m_PlaceB.MemoryMap(0, int(P.size()), [&](void* p) { std::memcpy(p, P.data(), P.size() * sizeof(int)); })) return false;
            xgpu::buffer* Bufs[] = { &m_DataB, &m_OwnerB, &m_ScratchB, &m_PlaceB, &Light, &Moments };
            struct { int m_Mode, m_N, m_Src, m_Base; float m_Saturation; int m_Pad0, m_Pad1, m_Pad2; } Push{ 0, int(m_nShown), 1, Base, Saturation, 0, 0, 0 };
            const int Groups = int((m_nShown + 63) / 64);
            Window.Dispatch(m_Filter, Bufs, &Push, sizeof(Push), Groups);             // pass 0 into the first array
            const int nP = std::clamp(Passes, 0, 16);
            for (int p = 1; p <= nP; ++p) { Push.m_Mode = 1; Push.m_Src = (p - 1) % 2; Window.Dispatch(m_Filter, Bufs, &Push, sizeof(Push), Groups); }
            Push.m_Mode = 2; Push.m_Src = nP % 2;
            Window.Dispatch(m_Filter, Bufs, &Push, sizeof(Push), Groups);
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

        bool            InFlight    ( void ) const noexcept { return m_InFlight != ~0ull; }
        bool            BatchDone   ( void ) const noexcept { return m_NextRange >= m_Ranges.size(); }    // the batch's slices are all dispatched
        double          LastMs      ( void ) const noexcept { return m_LastMs; }        // the last slice's GPU time
        double          RaysPerMs   ( void ) const noexcept { return m_RaysPerMs; }
        int             Packs       ( void ) const noexcept { return m_nPacks; }       // the scene packed in full (a structural change)
        int             Retlases    ( void ) const noexcept { return m_nRetlas; }      // the TLAS made again (a move)
        int             Batches     ( void ) const noexcept { return m_nBatches; }
        int             Slices      ( void ) const noexcept { return m_nSlices; }
        int             Ops         ( void ) const noexcept { return m_nOps; }          // the estimator changes applied (one per surfel)
        int             LastRays    ( void ) const noexcept { return m_LastRays; }
        int             LastTraced  ( void ) const noexcept { return m_LastTraced; }   // the last slice's surfels traced
        std::size_t     DataWords   ( void ) const noexcept { return m_Data.size(); }
        std::size_t     UploadWords ( void ) const noexcept { return m_UploadWords; }  // the last slice's refresh

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
            std::vector<std::uint32_t> Old = std::move(m_Data);
            W.PackGPU(m_Scene);
            W.PackGPUSurfels(m_Data, L);
            std::vector<std::uint32_t> Owners;
            for (int i = 0; i < W.InstanceCount(); ++i)
            {
                const std::uint32_t R = m_Data[0] + std::uint32_t(i) * world::gpu_inst_v;
                if (m_Data[R + 40] == 0) continue;
                m_Data[R + 46] = std::uint32_t(Owners.size());
                Owners.insert(Owners.end(), W.InstanceLight(i).size(), std::uint32_t(i));
            }
            m_nShown = Owners.size();
            m_Gen = W.GPUGen(); m_nPoints = L.m_Points.size();
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
                || !Local(m_OwnerB, Owners) || !Local(m_ScratchB, std::vector<float>(std::max<std::size_t>(18 * m_nShown, 1), 0.0f))) return false;
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
                    std::vector<std::uint32_t> A[5];
                    PutU(R + 47, W.GPUMaterial(i, A));
                    const std::uint32_t Word[5] = { 11, 12, 13, 14, 45 };
                    for (int k = 0; k < 5; ++k)
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
            for (const auto& P : L.m_Points) { V(P.m_Position); V(P.m_Color); Put(At++, P.m_Radius); }
            PutU(4, L.m_bSkyOct ? 1u : 0u);
            m_bPlaced = false;
            const int nPairs = int(m_Stage.size() / 2);
            if (!Fit(m_StageB, int(2 * sizeof(std::uint32_t)), std::size_t(nPairs))) return false;
            if (m_StageB.MemoryMap(0, nPairs, [&](void* p) { std::memcpy(p, m_Stage.data(), m_Stage.size() * sizeof(std::uint32_t)); })) return false;
            const struct { int m_nPairs, m_Pad0, m_Pad1, m_Pad2; } Push{ nPairs, 0, 0, 0 };
            xgpu::buffer* Bufs[] = { &m_DataB, &m_StageB };
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
        xgpu::shader                        m_Shader[10];
        xgpu::compute                       m_Pass, m_Scatter, m_Commit, m_OpsC, m_Filter, m_Expand, m_Means, m_Cube, m_Migrate, m_ReadBackC;
        xgpu::timestamps                    m_Times;
        xgpu::buffer                        m_Nodes, m_Tri4, m_Inst, m_Tlas, m_DataB, m_OldDataB, m_OwnerB, m_ScratchB;     // (device-local)
        xgpu::buffer                        m_ItemB, m_Results, m_CountB, m_RangeB, m_StatsB, m_MeansB, m_CubeB, m_JobB, m_StageB, m_OpB, m_PlaceB, m_ReadB;
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
        std::uint64_t                       m_Gen = 0, m_InFlight = ~0ull, m_ReadFrame = ~0ull;
        std::size_t                         m_nPoints = 0, m_nShown = 0, m_UploadWords = 0, m_ReadWords = 0;
        int                                 m_LastRays = 0, m_LastTraced = 0, m_nPacks = 0, m_nBatches = 0, m_nSlices = 0, m_nOps = 0;
        double                              m_LastMs = 0, m_RaysPerMs = 0;
        int                                 m_Budget = 32768;   // (until measured)
        bool                                m_bMeasured = false, m_bMeansIn = false;
    };
}

#endif
