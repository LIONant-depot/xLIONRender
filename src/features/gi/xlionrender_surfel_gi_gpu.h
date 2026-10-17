#ifndef XLIONRENDER_SURFEL_GI_GPU_H
#define XLIONRENDER_SURFEL_GI_GPU_H

// Part II, G3 and G4 (documentation/gi_dynamic_master_plan.md): the GI's trace and commit on the GPU, over time as the CPU's trace was.
// The scene and the light stay on the GPU: packed again (world::PackGPU, PackGPUSurfels) only when the world's structure changes
// (world::GPUGen) or the number of point lights does. A batch is the world's picked items (world::GPUWork, a few frames of rays:
// BatchRays); every frame the next slice of it that fits the frame's GPU budget is traced (shaders/xlionrender_gi_surfel_pass_comp.glsl)
// and committed (xlionrender_gi_commit_comp.glsl), and the frame after the surfels' new state goes to the world's copy
// (world::DeliverGPUState): no frame has to finish a batch. Before each slice the world's changes go up: its estimator changes as ops
// (world::TakeGPUOps, xlionrender_gi_ops_comp.glsl), the light of an instance whose moments were reset or that holds an ambient cube,
// the links' emitted light that changed, every instance's mean light and the lights (xlionrender_gi_scatter_comp.glsl). The world
// picks (the policy stays on the CPU) once the whole batch is delivered. The shown light is made on the GPU too (Show,
// xlionrender_gi_filter_comp.glsl), straight into the renderer's ring
#include "xlionrender_surfel_gi.h"
#include <span>

namespace xlionrender::gi
{
    class gpu_trace
    {
    public:

        // The SPIR-V of shaders/xlionrender_gi_surfel_pass_comp.glsl, _scatter_comp, _commit_comp, _ops_comp, _filter_comp
        bool Init(xgpu::device& Device, std::span<const std::uint32_t> Spirv, std::span<const std::uint32_t> Scatter, std::span<const std::uint32_t> Commit, std::span<const std::uint32_t> Ops
                 , std::span<const std::uint32_t> Filter) noexcept
        {
            m_pDevice = &Device;
            const auto Raw = [](std::span<const std::uint32_t> W) { return xgpu::shader::setup::raw_data{ std::span{ reinterpret_cast<const std::int32_t*>(W.data()), W.size() } }; };
            if (Device.Create(m_Shader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = Raw(Spirv) })) return false;
            if (Device.Create(m_Pass, { .m_pShader = &m_Shader, .m_nBuffers = 7, .m_PushConstantsSize = 16 })) return false;
            if (Device.Create(m_ScatterShader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = Raw(Scatter) })) return false;
            if (Device.Create(m_Scatter, { .m_pShader = &m_ScatterShader, .m_nBuffers = 2, .m_PushConstantsSize = 16 })) return false;
            if (Device.Create(m_CommitShader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = Raw(Commit) })) return false;
            if (Device.Create(m_Commit, { .m_pShader = &m_CommitShader, .m_nBuffers = 4, .m_PushConstantsSize = 16 })) return false;
            if (Device.Create(m_OpsShader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = Raw(Ops) })) return false;
            if (Device.Create(m_OpsC, { .m_pShader = &m_OpsShader, .m_nBuffers = 2, .m_PushConstantsSize = 16 })) return false;
            if (Device.Create(m_FilterShader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = Raw(Filter) })) return false;
            if (Device.Create(m_Filter, { .m_pShader = &m_FilterShader, .m_nBuffers = 6, .m_PushConstantsSize = 32 })) return false;
            if (Device.Create(m_Times, { .m_Count = 2 })) return false;
            return true;
        }

        // Before world::Update: the slice of an earlier frame (done: that frame's PageFlip waited for it) into the world. true: one was delivered
        bool Collect(world& W, std::uint64_t Frame) noexcept
        {
            if (m_InFlight == ~0ull || Frame <= m_InFlight) return false;
            m_InFlight = ~0ull;
            std::vector<std::uint64_t> Ns;
            if (m_Times.getResults(Ns) && Ns.size() >= 2 && Ns[0] != ~0ull && Ns[1] != ~0ull && Ns[1] > Ns[0])
            {
                m_LastMs = double(Ns[1] - Ns[0]) * 1e-6;
                m_RaysPerMs = double(m_LastRays) / std::max(m_LastMs, 1e-3);
                m_bMeasured = true;
            }
            m_Out.resize(m_SliceCount);
            if (m_Out.empty() || m_States.MemoryMap(0, int(m_SliceCount), [&](void* p) { std::memcpy(m_Out.data(), p, m_Out.size() * sizeof(world::gpu_state)); })) return false;
            W.DeliverGPUState(std::span<const world::gpu_item>(m_Items).subspan(m_SliceFirst, m_SliceCount), m_Out, m_Ops);
            m_Work = {};
            for (const auto& G : m_Out) for (int w = 0; w < 4; ++w) m_Work[std::size_t(w)] += G.m_Work[w];
            return true;
        }

        // After world::Update: its estimator changes noted; with no slice in flight, the next slice of the batch (a new batch when it is all
        // traced: the world's picked items) traced and committed in this frame, BudgetMs of the GPU's time (dispatches between render
        // passes). true: a slice was dispatched
        bool Submit(xgpu::window& Window, world& W, const lights& L, std::uint64_t Frame, float BudgetMs) noexcept
        {
            { auto Ops = W.TakeGPUOps(); m_Ops.insert(m_Ops.end(), Ops.begin(), Ops.end()); }
            if (m_InFlight != ~0ull) return false;
            // the scene and the world's changes first (also with nothing to trace: Show reads them)
            if (W.GPUGen() != m_Gen || L.m_Points.size() != m_nPoints || m_Data.empty()) { if (!Pack(W, L)) return false; m_Ops.clear(); }   // (the world's copy has the ops)
            else if (!ApplyOps(Window, W) || !Refresh(Window, W, L)) return false;
            if (m_Next >= m_Items.size())
            {
                const auto Work = W.GPUWork();
                if (Work.empty()) return false;
                m_Items.assign(Work.begin(), Work.end());
                m_Next = 0;
                if (!Fit(m_ItemB, int(sizeof(world::gpu_item)), m_Items.size()) || !Fit(m_Results, int(sizeof(world::gpu_result)), m_Items.size())
                    || !Fit(m_States, int(sizeof(world::gpu_state)), m_Items.size())) return false;
                if (m_ItemB.MemoryMap(0, int(m_Items.size()), [&](void* p) { std::memcpy(p, m_Items.data(), m_Items.size() * sizeof(world::gpu_item)); })) return false;
                ++m_nBatches;
            }
            // the slice: the items from m_Next whose rays fit this frame's budget (one at least; no budget: all of them)
            const int Rays = BudgetMs > 0 ? Budget(BudgetMs) : std::numeric_limits<int>::max();
            m_SliceFirst = m_Next; m_LastRays = 0;
            while (m_Next < m_Items.size() && (m_Next == m_SliceFirst || m_LastRays + m_Items[m_Next].m_Rays <= Rays)) m_LastRays += m_Items[m_Next++].m_Rays;
            m_SliceCount = m_Next - m_SliceFirst;
            const struct { int m_nItems, m_TlasRoot, m_PrimOffset, m_First; } Push{ int(m_Items.size()), m_Scene.m_TlasRoot, m_Scene.m_PrimOffset, int(m_SliceFirst) };
            xgpu::buffer* Bufs[] = { &m_Nodes, &m_Tri4, &m_Inst, &m_Tlas, &m_DataB, &m_ItemB, &m_Results };
            const struct { int m_nItems, m_First, m_Count, m_Pad; } CPush{ int(m_Items.size()), int(m_SliceFirst), int(m_SliceCount), 0 };
            xgpu::buffer* CBufs[] = { &m_DataB, &m_ItemB, &m_Results, &m_States };
            Window.WriteTimestamp(m_Times, 0);
            Window.Dispatch(m_Pass, Bufs, &Push, sizeof(Push), world::GPUGroups(m_SliceCount));
            Window.Dispatch(m_Commit, CBufs, &CPush, sizeof(CPush), int((m_SliceCount + 63) / 64));
            Window.WriteTimestamp(m_Times, 1);
            m_InFlight = Frame;
            ++m_nSlices;
            return true;
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

        bool            InFlight    ( void ) const noexcept { return m_InFlight != ~0ull; }
        double          LastMs      ( void ) const noexcept { return m_LastMs; }        // the last batch's GPU time
        double          RaysPerMs   ( void ) const noexcept { return m_RaysPerMs; }
        int             Packs       ( void ) const noexcept { return m_nPacks; }       // the scene packed in full (a structural change)
        int             Batches     ( void ) const noexcept { return m_nBatches; }
        int             Slices      ( void ) const noexcept { return m_nSlices; }
        int             Ops         ( void ) const noexcept { return m_nOps; }          // the estimator changes applied (one per surfel)
        int             LastRays    ( void ) const noexcept { return m_LastRays; }
        const std::array<std::int64_t, 4>& Work( void ) const noexcept { return m_Work; }   // the last slice's: BVH nodes, shadow rays, read entries, hash probes
        std::size_t     DataWords   ( void ) const noexcept { return m_Data.size(); }
        std::size_t     UploadWords ( void ) const noexcept { return m_UploadWords; }    // the last slice's refresh

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

        // Everything again (ponytail: any structural change packs the whole scene; incremental records and TLAS refits when it shows)
        bool Pack(world& W, const lights& L) noexcept
        {
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
            if (!Local(m_OwnerB, Owners) || !Local(m_ScratchB, std::vector<float>(std::max<std::size_t>(18 * m_nShown, 1), 0.0f))) return false;
            m_Gen = W.GPUGen(); m_nPoints = L.m_Points.size();
            m_MomReset.assign(std::size_t(W.InstanceCount()), 0);
            m_EmitGen.assign(std::size_t(W.InstanceCount()), 0);
            for (int i = 0; i < W.InstanceCount(); ++i) { m_MomReset[std::size_t(i)] = W.MomentsReset(i); m_EmitGen[std::size_t(i)] = W.EmitGen(i); }
            ++m_nPacks;
            return Local(m_Nodes, m_Scene.m_Nodes) && Local(m_Tri4, m_Scene.m_Tri4) && Local(m_Inst, m_Scene.m_Instances) && Local(m_Tlas, m_Scene.m_TlasLeaves) && Local(m_DataB, m_Data);
        }

        // The world's estimator changes on the GPU's copy (an instance's: one op per surfel)
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

        // What changed since the last slice into the resident data (and m_Data, its copy): the changed words as (index, value) pairs,
        // written into the device-local array by the scatter shader in this frame (before the pass: the dispatches' barriers)
        bool Refresh(xgpu::window& Window, world& W, const lights& L) noexcept
        {
            const auto& D = m_Data;
            m_Stage.clear();
            const auto Put = [&](std::uint32_t At, float f)       // only the words that change
            {
                const std::uint32_t u = std::bit_cast<std::uint32_t>(f);
                m_Data[At] = u;
                m_Stage.push_back(At); m_Stage.push_back(u);
            };
            const auto Rec = [&](int i) { return D[0] + std::uint32_t(i) * world::gpu_inst_v; };
            const auto PutU = [&](std::uint32_t At, std::uint32_t v) { Put(At, std::bit_cast<float>(v)); };
            const auto Light = [&](int i, int s)            // E and the fractions (a cube's expanded parents)
            {
                const std::uint32_t R = Rec(i), u = std::uint32_t(s);
                const v3& E = W.InstanceLight(i)[u];
                Put(D[R + 2] + 3 * u, E.m_X); Put(D[R + 2] + 3 * u + 1, E.m_Y); Put(D[R + 2] + 3 * u + 2, E.m_Z);
                const auto& Fr = W.BasisFractions(i)[u];
                for (std::uint32_t k = 0; k < 3; ++k) { Put(D[R + 41] + 9 * u + 3 * k, Fr[k].m_X); Put(D[R + 41] + 9 * u + 3 * k + 1, Fr[k].m_Y); Put(D[R + 41] + 9 * u + 3 * k + 2, Fr[k].m_Z); }
            };
            const auto Surfel = [&](int i, int s)           // all of its state (the world's copy is the GPU's: Collect delivered every slice)
            {
                const std::uint32_t R = Rec(i), u = std::uint32_t(s);
                if (D[R + 2] == world::gpu_none_v) return;
                Light(i, s);
                Put(D[R + 3] + u, W.InstanceValid(i)[u]);
                const auto& Mo = W.InstanceMoments(i)[u];
                for (std::uint32_t t = 0; t < std::uint32_t(Mo.size()); ++t) Put(D[R + 4] + u * std::uint32_t(Mo.size()) + t, Mo[t]);
                std::uint32_t Es[world::gpu_est_v];
                world::PackEst(W.Estimator(i, s), Es);
                for (std::uint32_t k = 0; k < world::gpu_est_v; ++k) PutU(D[R + 42] + u * world::gpu_est_v + k, Es[k]);
                const auto& Mc = W.MomentCounts(i)[u];
                for (std::uint32_t t = 0; t < std::uint32_t(Mc.size()); ++t) PutU(D[R + 43] + u * std::uint32_t(Mc.size()) + t, Mc[t]);
            };
            for (int i = 0; i < W.InstanceCount(); ++i)
            {
                const std::uint32_t R = Rec(i);
                if (D[R + 40] == 0) continue;
                const v3& M = W.MeanLight(i);
                Put(R + 5, M.m_X); Put(R + 6, M.m_Y); Put(R + 7, M.m_Z);
                if (W.MomentsReset(i) != m_MomReset[std::size_t(i)])
                {
                    m_MomReset[std::size_t(i)] = W.MomentsReset(i);
                    for (int s = 0; s < int(W.InstanceLight(i).size()); ++s) Surfel(i, s);
                }
                else if (W.IsCube(i)) for (int s = 0; s < int(W.InstanceLight(i).size()); ++s) Light(i, s);
                if (W.EmitGen(i) != m_EmitGen[std::size_t(i)])          // the links' emitted light (zeros: none)
                {
                    m_EmitGen[std::size_t(i)] = W.EmitGen(i);
                    const auto& B = W.EmittedLight(i);
                    for (std::uint32_t u = 0; u < std::uint32_t(W.InstanceLight(i).size()); ++u)
                        for (std::uint32_t k = 0; k < 3; ++k)
                        {
                            const v3 e = u < B.size() ? B[u][k] : v3{ 0, 0, 0 };
                            Put(D[R + 44] + 9 * u + 3 * k, e.m_X); Put(D[R + 44] + 9 * u + 3 * k + 1, e.m_Y); Put(D[R + 44] + 9 * u + 3 * k + 2, e.m_Z);
                        }
                }
            }
            // the lights (the same count: PackGPUSurfels' layout), the sky's flag
            std::uint32_t At = D[2];
            const auto V = [&](const v3& v) { Put(At, v.m_X); Put(At + 1, v.m_Y); Put(At + 2, v.m_Z); At += 3; };
            V(L.m_SunDirection); V(L.m_SunColor); V(L.m_Sky);
            for (const auto& s : L.m_SkyOct) V(s);
            for (const auto& P : L.m_Points) { V(P.m_Position); V(P.m_Color); Put(At++, P.m_Radius); }
            Put(4, std::bit_cast<float>(L.m_bSkyOct ? 1u : 0u));
            const int nPairs = int(m_Stage.size() / 2);
            if (!Fit(m_StageB, int(2 * sizeof(std::uint32_t)), std::size_t(nPairs))) return false;
            if (m_StageB.MemoryMap(0, nPairs, [&](void* p) { std::memcpy(p, m_Stage.data(), m_Stage.size() * sizeof(std::uint32_t)); })) return false;
            const struct { int m_nPairs, m_Pad0, m_Pad1, m_Pad2; } Push{ nPairs, 0, 0, 0 };
            xgpu::buffer* Bufs[] = { &m_DataB, &m_StageB };
            Window.Dispatch(m_Scatter, Bufs, &Push, sizeof(Push), (nPairs + 63) / 64);
            m_UploadWords = std::size_t(nPairs);
            return true;
        }

        xgpu::device*                       m_pDevice = nullptr;
        xgpu::shader                        m_Shader;
        xgpu::compute                       m_Pass;
        xgpu::timestamps                    m_Times;
        xgpu::buffer                        m_Nodes, m_Tri4, m_Inst, m_Tlas, m_DataB, m_ItemB, m_Results;     // (the first five device-local)
        xgpu::shader                        m_ScatterShader, m_CommitShader, m_OpsShader, m_FilterShader;
        xgpu::compute                       m_Scatter, m_Commit, m_OpsC, m_Filter;
        xgpu::buffer                        m_OwnerB, m_ScratchB, m_PlaceB;     // Show: the surfels' instances, the passes, the places
        std::size_t                         m_nShown = 0;
        xgpu::buffer                        m_States, m_OpB;    // the committed states (the world's copy), the ops
        std::vector<world::gpu_op>          m_Ops;              // the world's estimator changes not applied yet
        std::vector<int>                    m_OpWords;
        std::vector<std::uint32_t>          m_EmitGen;          // per instance: world::EmitGen when its emitted light last went up
        int                                 m_nOps = 0;
        xgpu::buffer                        m_StageB;           // Refresh's changed words: (index, value) pairs
        std::vector<std::uint32_t>          m_Stage;
        std::size_t                         m_UploadWords = 0;
        world::gpu_scene                    m_Scene;
        std::vector<std::uint32_t>          m_Data;             // PackGPUSurfels' array, as the GPU has it
        std::vector<world::gpu_item>        m_Items;            // the batch (on the GPU: m_ItemB)
        std::size_t                         m_Next = 0, m_SliceFirst = 0, m_SliceCount = 0;     // its next item; the slice in flight
        std::vector<world::gpu_state>       m_Out;
        std::vector<std::uint32_t>          m_MomReset;         // per instance: world::MomentsReset when its light last went up whole
        std::uint64_t                       m_Gen = 0, m_InFlight = ~0ull;
        std::size_t                         m_nPoints = 0;
        int                                 m_LastRays = 0, m_nPacks = 0, m_nBatches = 0, m_nSlices = 0;
        double                              m_LastMs = 0, m_RaysPerMs = 0;
        int                                 m_Budget = 32768;   // (until measured)
        bool                                m_bMeasured = false;
        std::array<std::int64_t, 4>         m_Work{};
    };
}

#endif
