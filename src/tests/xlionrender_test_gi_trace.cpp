// Part II, G3 and G4 (documentation/gi_dynamic_master_plan.md): the solver on the GPU (gi::gpu_trace, world::m_Setup.m_bGPUOwned: the pick
// as ranges, the light living on the GPU). A world whose light the GPU makes (expanded, traced, committed, its changes as ops, moved and
// re-levelled when the scene is packed again) must hold, read back, the light of the same world on the CPU, update for update: the
// Cornell box (a point light; an emissive panel through its links; events), the GI render test's room, and its shown light against the
// CPU's filter. The CPU's main-thread time per update in GPU mode is printed
#include "xlionrender_test_common.h"
#include "features/gi/xlionrender_surfel_gi_gpu.h"
#include "features/gi/xlionrender_surfel_gi_chunks.h"

using namespace xlionrender::test;
using namespace xlionrender::gi;

namespace
{
    constexpr std::uint32_t s_PassComp[] = {
        #include "xlionrender_gi_surfel_pass_comp.h"
    };
    constexpr std::uint32_t s_ScatterComp[] = {
        #include "xlionrender_gi_scatter_comp.h"
    };
    constexpr std::uint32_t s_CommitComp[] = {
        #include "xlionrender_gi_commit_comp.h"
    };
    constexpr std::uint32_t s_OpsComp[] = {
        #include "xlionrender_gi_ops_comp.h"
    };
    constexpr std::uint32_t s_FilterComp[] = {
        #include "xlionrender_gi_filter_comp.h"
    };
    constexpr std::uint32_t s_ExpandComp[] = {
        #include "xlionrender_gi_expand_comp.h"
    };
    constexpr std::uint32_t s_MeansComp[] = {
        #include "xlionrender_gi_means_comp.h"
    };
    constexpr std::uint32_t s_CubeComp[] = {
        #include "xlionrender_gi_cube_comp.h"
    };
    constexpr std::uint32_t s_MigrateComp[] = {
        #include "xlionrender_gi_migrate_comp.h"
    };
    constexpr std::uint32_t s_LinksComp[] = {
        #include "xlionrender_gi_links_comp.h"
    };
    constexpr std::uint32_t s_SolveComp[] = {
        #include "xlionrender_gi_solve_comp.h"
    };
    constexpr std::uint32_t s_GhostsComp[] = {
        #include "xlionrender_gi_ghosts_comp.h"
    };
    constexpr std::uint32_t s_ProbesComp[] = {
        #include "xlionrender_gi_probes_comp.h"
    };
    constexpr std::uint32_t s_ReadBackComp[] = {
        #include "xlionrender_gi_readback_comp.h"
    };
    const gpu_trace::shaders s_Shaders{ s_PassComp, s_ScatterComp, s_CommitComp, s_OpsComp, s_FilterComp, s_ExpandComp, s_MeansComp, s_CubeComp, s_MigrateComp, s_ReadBackComp, s_LinksComp, s_SolveComp, s_GhostsComp, s_ProbesComp };

    // The GPU's light into the world's arrays: the last slice delivered, the read back copied, then in (two frames)
    // Empty frames until the slices out are back (gpu_trace::Collect reads them lag_frames_v frames on)
    void Drain(app& App, gpu_trace& T, world& W)
    {
        for (int f = 0; f < 8 && T.InFlight(); ++f)
        {
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            T.Collect(W, App.m_Device.getFrameIndex());
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
        }
    }

    void Sync(app& App, gpu_trace& T, world& W)
    {
        Drain(App, T, W);
        for (int f = 0; f < 1 + int(gpu_trace::lag_frames_v); ++f)
        {
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            const std::uint64_t Frame = App.m_Device.getFrameIndex();
            T.Collect(W, Frame);
            if (f == 0) (void)T.ReadBack(App.m_Window, Frame);
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
        }
    }

    void AddQuad(surfel_mesh& M, v3 C, v3 U, v3 V, float Split)        // (xlionrender_test_surfel_gi.cpp's)
    {
        const int nu = std::max(1, int(std::ceil(U.Length() / Split))), nv = std::max(1, int(std::ceil(V.Length() / Split)));
        const v3  N  = U.Cross(V).NormalizeCopy();
        const int Base = int(M.m_Pos.size());
        for (int j = 0; j <= nv; ++j) for (int i = 0; i <= nu; ++i) { M.m_Pos.push_back(C + U * (float(i) / nu) + V * (float(j) / nv)); M.m_Nrm.push_back(N); }
        for (int j = 0; j < nv; ++j) for (int i = 0; i < nu; ++i)
        {
            const int a = Base + j * (nu + 1) + i, b = a + 1, c = a + nu + 1, d = c + 1;
            M.m_Tris.push_back({ a, b, d }); M.m_Tris.push_back({ a, d, c });
        }
    }

    // The Cornell box of the GI lab (the box an instance, rotated), its read grids built; Panel: the ceiling's middle emits (the links)
    void MakeCornell(std::vector<surfel_mesh>& M, world& W, bool bPanel)
    {
        const v3 Wh{ 0.75f, 0.75f, 0.75f }, Rd{ 0.65f, 0.05f, 0.05f }, Gr{ 0.12f, 0.6f, 0.1f };
        M.assign(7, {});
        AddQuad(M[0], { -1, -1, -1 }, { 0, 0, 2 }, { 2, 0, 0 }, 0.125f);
        AddQuad(M[1], { -1,  1, -1 }, { 2, 0, 0 }, { 0, 0, 2 }, 0.125f);
        AddQuad(M[2], { -1, -1,  1 }, { 0, 2, 0 }, { 2, 0, 0 }, 0.125f);
        AddQuad(M[3], { -1, -1, -1 }, { 2, 0, 0 }, { 0, 2, 0 }, 0.125f);
        AddQuad(M[4], { -1, -1, -1 }, { 0, 2, 0 }, { 0, 0, 2 }, 0.125f);
        AddQuad(M[5], {  1, -1, -1 }, { 0, 0, 2 }, { 0, 2, 0 }, 0.125f);
        const v3 S{ 0.5f, 0.8f, 0.6f }, Lo = S * -0.5f, X{ S.m_X, 0, 0 }, Y{ 0, S.m_Y, 0 }, Z{ 0, 0, S.m_Z };
        AddQuad(M[6], Lo, X, Z, 0.125f); AddQuad(M[6], Lo + Y, Z, X, 0.125f); AddQuad(M[6], Lo, Z, Y, 0.125f);
        AddQuad(M[6], Lo + X, Y, Z, 0.125f); AddQuad(M[6], Lo, Y, X, 0.125f); AddQuad(M[6], Lo + Z, X, Y, 0.125f);
        for (auto& m : M) { m.Bake(0.3f, 0.75f); m.BuildReadGrid(); }
        const v3 Albedo[] = { Wh, Wh, Wh, Wh, Rd, Gr };
        for (int m = 0; m < 6; ++m) W.Add(M[std::size_t(m)], xmath::fmat4::fromIdentity(), Albedo[m]);
        W.Add(M[6], xmath::fmat4::fromTranslation({ -0.35f, -0.6f, 0.1f }) * xmath::fmat4::fromRotationY(xmath::radian{ 0.5f }), Wh);
        if (bPanel)
        {
            std::vector<v3> Em(M[1].m_Levels[0].size(), v3{ 0, 0, 0 });
            for (std::size_t s = 0; s < Em.size(); ++s) { const v3& C = M[1].m_Levels[0][s].m_Center; if (std::abs(C.m_X) < 0.35f && std::abs(C.m_Z) < 0.35f) Em[s] = { 6, 6, 5 }; }
            W.SetSurfelMaterial(1, {}, Em);
        }
        // (the adaptive rays and the tiers on: they read the estimators when a surfel is picked, the world's copy of the GPU's (G4: the
        // slice's commit is delivered before the next update, as the CPU's commit is in it))
    }

    // The room of xlionrender_test_gi_render (its pieces boxes of cube::Generate at their size, MakeSurfels: the renderer's meshes)
    // GI sections (gi_chunks_plan.md): a room baked as one mesh (6 x 6 m, 2.5 m high, 1 m tiles as its clusters), cut into sections of about a
    // quarter of its surfels, each its own instance; their ghosts copy their owners (the CPU: world::SyncGhosts; the GPU: the ghost pass)
    void MakeSectionRoom(std::vector<surfel_mesh>& M, world& W)
    {
        surfel_mesh Whole;
        const auto Quad = [&](v3 C, v3 U, v3 V)
        {
            const int n = 48;
            const int Base = int(Whole.m_Pos.size());
            const v3 N = U.Cross(V).NormalizeCopy();
            for (int j = 0; j <= n; ++j) for (int i = 0; i <= n; ++i) { Whole.m_Pos.push_back(C + U * (float(i) / n) + V * (float(j) / n)); Whole.m_Nrm.push_back(N); }
            for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
            {
                const int a = Base + j * (n + 1) + i;
                Whole.m_Tris.push_back({ a, a + 1, a + n + 2 }); Whole.m_Tris.push_back({ a, a + n + 2, a + n + 1 });
            }
        };
        Quad({ -3, 0, -3 }, { 0, 0, 6 }, { 6, 0, 0 }); Quad({ -3, 2.5f, -3 }, { 6, 0, 0 }, { 0, 0, 6 });
        Quad({ -3, 0, 3 }, { 0, 2.5f, 0 }, { 6, 0, 0 }); Quad({ -3, 0, -3 }, { 6, 0, 0 }, { 0, 2.5f, 0 });
        Quad({ -3, 0, -3 }, { 0, 2.5f, 0 }, { 0, 0, 6 }); Quad({ 3, 0, -3 }, { 0, 0, 6 }, { 0, 2.5f, 0 });
        Whole.Bake(0.3f, 0.9f);
        std::vector<int> TriCluster(Whole.m_Tris.size());
        std::unordered_map<std::int64_t, int> Tile;
        for (std::size_t t = 0; t < Whole.m_Tris.size(); ++t)
        {
            const v3 c = (Whole.m_Pos[Whole.m_Tris[t][0]] + Whole.m_Pos[Whole.m_Tris[t][1]] + Whole.m_Pos[Whole.m_Tris[t][2]]) * (1.0f / 3.0f);
            const std::int64_t k = (std::int64_t(std::floor(c.m_X)) + 100) * 1000000 + (std::int64_t(std::floor(c.m_Y)) + 100) * 1000 + (std::int64_t(std::floor(c.m_Z)) + 100);
            TriCluster[t] = Tile.try_emplace(k, int(Tile.size())).first->second;
        }
        chunked_mesh Ch;
        (void)PartitionChunks(Whole, TriCluster, int(Tile.size()), Whole.m_Levels[0].size() / 4, Ch);
        M = std::move(Ch.m_Chunks);
        std::vector<int> Ids;
        for (auto& C : M) Ids.push_back(W.Add(C, xmath::fmat4::fromIdentity(), { 0.7f, 0.7f, 0.7f }));
        for (std::size_t c = 0; c < M.size(); ++c)
        {
            std::vector<std::array<int, 2>> Owners;
            for (const auto& G : Ch.m_Ghosts[c]) Owners.push_back({ Ids[G.m_Chunk], int(G.m_Local) });
            W.SetGhosts(Ids[c], Ch.m_nOwned[c], std::move(Owners));
        }
    }

    void MakeRoom(std::vector<surfel_mesh>& M, world& W)
    {
        struct piece { xprim_geom::float3 m_Size; xmath::fvec3 m_Pos; v3 m_Color; };
        const piece Pieces[] =
        { { { 4.0f, 0.2f, 4.0f }, { 0, -0.1f, 0 }, { 0.75f, 0.75f, 0.75f } }, { { 4.0f, 0.2f, 4.0f }, { 0, 4.1f, 0 }, { 0.75f, 0.75f, 0.75f } }
        , { { 4.0f, 4.0f, 0.2f }, { 0, 2, 2.1f }, { 0.75f, 0.75f, 0.75f } }, { { 0.2f, 4.0f, 4.0f }, { -2.1f, 2, 0 }, { 0.8f, 0.05f, 0.05f } }
        , { { 0.2f, 4.0f, 4.0f }, { 2.1f, 2, 0 }, { 0.05f, 0.7f, 0.05f } }, { { 0.8f, 1.6f, 0.8f }, { -0.6f, 0.8f, 0.6f }, { 0.75f, 0.75f, 0.75f } } };
        M.clear(); M.reserve(std::size(Pieces));
        for (const auto& P : Pieces) M.push_back(MakeSurfels(xprim_geom::cube::Generate(15, 15, 15, 15, P.m_Size), 0.3f, 0.9f));
        for (std::size_t i = 0; i < std::size(Pieces); ++i) W.Add(M[i], xmath::fmat4::fromTranslation(Pieces[i].m_Pos), Pieces[i].m_Color);
    }

    struct result { int m_Surfels = 0, m_Mismatch = 0; double m_Worst = 0, m_UpdateMs = 0, m_TraceMs = 0; int m_Packs = 0, m_Relevels = 0; double m_ShownWorst = -1, m_MomentWorst = -1, m_InitMs = 0; std::size_t m_CpuBytes = 0, m_OwnedBytes = 0; };

    // Updates of the CPU world against the GPU world's (it commits a batch one update later: one update more). bEvents: on both at the
    // same updates, a wall recolored (its material: the scene packed again), the box moved (estimator ops: Lower, the changes near it),
    // the eye far away (the coarse level: the light carried over)
    int Compare(app& App, const lights& L, bool bPanel, int Updates, result& R, bool bEvents = false, bool bRoom = false, bool bSample = false, bool bSections = false)
    {
        R = {};
        std::vector<surfel_mesh> Mc, Mg;
        world Wc, Wg;
        if (bSections) { MakeSectionRoom(Mc, Wc); MakeSectionRoom(Mg, Wg); }
        else if (bRoom) { MakeRoom(Mc, Wc); MakeRoom(Mg, Wg); }
        else { MakeCornell(Mc, Wc, bPanel); MakeCornell(Mg, Wg, bPanel); }
        Wc.m_Setup.m_bEmitterSampling = Wg.m_Setup.m_bEmitterSampling = bSample;
        Wg.m_Setup.m_bGPUTrace = Wg.m_Setup.m_bGPUOwned = true;
        gpu_trace T;
        const auto TInit = std::chrono::steady_clock::now();
        CHECK(T.Init(App.m_Device, s_Shaders), "gpu_trace::Init");
        R.m_InitMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - TInit).count();
        const auto Event = [&](world& W, std::vector<surfel_mesh>& M, int u)
        {
            if (!bEvents) return;
            if (u == 6) W.SetSurfelMaterial(4, std::vector<v3>(M[4].m_Levels[0].size(), v3{ 0.2f, 0.6f, 0.7f }), {});
            if (u == 8 || u == 9) W.SetTriangleAlbedo(5, u == 8 ? 0 : M[5].m_Tris.size() / 2, std::vector<v3>(M[5].m_Tris.size() / 2, v3{ 0.7f, 0.5f, 0.1f }));     // (a capture in two chunks)
            if (u >= 10 && u <= 13) W.SetTransform(6, xmath::fmat4::fromTranslation({ 0.2f + 0.05f * float(u - 10), -0.6f, 0.1f }) * xmath::fmat4::fromRotationY(xmath::radian{ 0.9f + 0.1f * float(u - 10) }));     // (dragged: the TLAS only)
            if (u == 14) { W.m_Setup.m_ProjScale = 300; }
        };
        // between the slice's dispatch and its delivery (as the renderer's material captures come): its ops reach the GPU with the next slice
        const auto Late = [&](world& W, std::vector<surfel_mesh>& M, int u)
        {
            if (bEvents && u == 7) W.SetSurfelMaterial(2, std::vector<v3>(M[2].m_Levels[0].size(), v3{ 0.9f, 0.9f, 0.4f }), {});
        };
        const auto EyeAt = [&](int u) { return bEvents && u >= 14 ? v3{ 0, 0, -60 } : bRoom ? v3{ 0, 1.6f, -1.9f } : v3{ 0, 0, -0.9f }; };
        for (int u = 0; u < Updates; ++u) { Event(Wc, Mc, u); R.m_CpuBytes = Wc.Update(L, EyeAt(u), 1 << 30, 100 + u).m_LightBytes; Late(Wc, Mc, u); }
        for (int u = 0; u < Updates; ++u)       // (the GPU commits a batch in the frame it traces it: the same updates; its stats back first)
        {
            Drain(App, T, Wg);
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            const std::uint64_t Frame = App.m_Device.getFrameIndex();
            const auto T0 = std::chrono::steady_clock::now();
            T.Collect(Wg, Frame);
            Event(Wg, Mg, u);
            R.m_OwnedBytes = Wg.Update(L, EyeAt(u), 1 << 30, 100 + u).m_LightBytes;      // (before Sync: its read back makes the CPU's arrays)
            T.Submit(App.m_Window, Wg, L, Frame, 1000.0f, EyeAt(u));     // (a budget for all of a batch: one slice, the CPU's order)
            Late(Wg, Mg, u);
            if (u > 0) R.m_UpdateMs = std::max(R.m_UpdateMs, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count());
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
            if (u > 0) R.m_TraceMs = std::max(R.m_TraceMs, T.LastMs());
        }
        R.m_Packs = T.Packs(); R.m_Relevels = T.Relevels();
        Sync(App, T, Wg);
        // per surfel: its light within 1e-3 of the instance's brightest (the sums' float order differs; the passes compound it)
        for (int i = 0; i < Wc.InstanceCount(); ++i)
        {
            const auto& Ec = Wc.InstanceLight(i); const auto& Eg = Wg.InstanceLight(i);
            CHECK(Ec.size() == Eg.size(), "instance %d: %zu surfels, %zu on the GPU", i, Ec.size(), Eg.size());
            double Top = 1e-6;
            for (const auto& e : Ec) Top = std::max({ Top, double(e.m_X), double(e.m_Y), double(e.m_Z) });
            for (std::size_t s = 0; s < Ec.size(); ++s)
            {
                const double d = std::max({ std::abs(Ec[s].m_X - Eg[s].m_X), std::abs(Ec[s].m_Y - Eg[s].m_Y), std::abs(Ec[s].m_Z - Eg[s].m_Z) }) / Top;
                const double v = std::abs(Wc.InstanceValid(i)[s] - Wg.InstanceValid(i)[s]);
                const auto& Ec_ = Wc.Estimator(i, int(s)); const auto& Eg_ = Wg.Estimator(i, int(s));
                const bool bEst = Ec_.m_N == Eg_.m_N && Ec_.m_bWatch == Eg_.m_bWatch && std::abs(Ec_.m_Variance.m_X - Eg_.m_Variance.m_X) <= 1e-3 * Top * Top + 1e-9;
                ++R.m_Surfels;
                if (d > 1e-3 || v > 1e-3 || !bEst) ++R.m_Mismatch;
                else R.m_Worst = std::max(R.m_Worst, d);
            }
        }
        if (bRoom)      // G4: the GPU's shown light (gpu_trace::Show into rings as the renderer's) against the CPU's filter of the same state
        {
            std::vector<int> Places(2 * std::size_t(Wg.InstanceCount()), -1);     // (per instance: its place, its mean: -1, the renderer's CPU writes it)
            int nRing = 0;
            for (int i = 0; i < Wg.InstanceCount(); ++i) { Places[2 * std::size_t(i)] = nRing; nRing += int(Wg.InstanceLight(i).size()) + 1; }
            xgpu::buffer Light, Moments;
            CHECK(!App.m_Device.Create(Light, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = nRing })
               && !App.m_Device.Create(Moments, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 4, .m_EntryCount = 16 * nRing }), "rings");
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            CHECK(T.Show(App.m_Window, Places, Light, Moments, 0, Wg.m_Setup.m_FilterPasses, Wg.m_Setup.m_Saturation), "Show");
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
            std::vector<std::array<std::uint32_t, 4>> L4(static_cast<std::size_t>(nRing)); std::vector<std::uint32_t> M4(std::size_t(16 * nRing));
            CHECK(!Light.MemoryMap(0, nRing, [&](void* q) { std::memcpy(L4.data(), q, L4.size() * 16); }) && !Moments.MemoryMap(0, 16 * nRing, [&](void* q) { std::memcpy(M4.data(), q, M4.size() * 4); }), "read the rings");
            Wg.ShowAllAgain();
            R.m_ShownWorst = 0; R.m_MomentWorst = 0;
            const auto Half = [](float f) -> std::uint32_t       // (plans/forward/xlionrender_plan_forward.h, details::Half)
            {
                const std::uint32_t x = std::bit_cast<std::uint32_t>(f), Sign = (x >> 16) & 0x8000u;
                const int E = int((x >> 23) & 0xFF) - 127 + 15;
                if (E <= 0) return Sign;
                if (E >= 31) return Sign | 0x7C00u;
                const std::uint32_t M = x & 0x7FFFFFu, H = Sign | (std::uint32_t(E) << 10) | (M >> 13);
                return (H + ((M >> 12) & 1u)) & 0xFFFFu;
            };
            for (int i = 0; i < Wg.InstanceCount(); ++i)
            {
                const auto& B = Wg.BasisLight(i); const auto& V = Wg.InstanceValid(i); const auto& Mo = Wg.InstanceMoments(i);
                double Top = 1e-6; for (const auto& b : B) for (int k = 0; k < 3; ++k) Top = std::max({ Top, double(b[k].m_X), double(b[k].m_Y), double(b[k].m_Z) });
                for (std::size_t q = 0; q < B.size(); ++q)
                {
                    const std::size_t At = std::size_t(Places[2 * std::size_t(i)] + 1) + q;
                    for (int k = 0; k < 3; ++k)
                    {
                        const v3 g = UnpackRGB9E5(L4[At][std::size_t(k)]);        // (RGB9E5: steps of 2^-9 of the largest channel)
                        R.m_ShownWorst = std::max({ R.m_ShownWorst, std::abs(g.m_X - B[q][k].m_X) / Top, std::abs(g.m_Y - B[q][k].m_Y) / Top, std::abs(g.m_Z - B[q][k].m_Z) / Top });
                    }
                    R.m_ShownWorst = std::max(R.m_ShownWorst, double(std::abs(std::bit_cast<float>(L4[At][3]) - V[q])));
                    for (int t = 0; t < 16; ++t)
                    {
                        const std::uint32_t w = M4[16 * At + std::size_t(t)];
                        R.m_MomentWorst += w != (Half(Mo[q][2 * t]) | (Half(Mo[q][2 * t + 1]) << 16)) ? 1 : 0;
                    }
                }
            }
        }
        return 0;
    }
}

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    for (const int Case : { 0, 1, 2 })
    {
        const bool bPanel = Case > 0;
        lights L;
        if (!bPanel) L.m_Points = { { { 0.2f, 0.4f, -0.3f }, { 1.5f, 1.5f, 1.5f }, 4.0f } };
        result R;
        if (Compare(App, L, bPanel, 24, R, false, false, Case == 2)) return 1;
        const char* pName = Case == 2 ? "Cornell box lit by an emissive panel (light sampling: G5)" : bPanel ? "Cornell box lit by an emissive panel (links)" : "Cornell box, a point light";
        std::printf("GPU trace, %s, 24 updates (the estimators settled, the adaptive rays on): %d surfels, %d differ from the CPU world, the worst of the others %.1e; the scene packed %d times; "
                    "an update in GPU mode (collect, Update, submit) %.2f ms at most, a batch %.3f ms on the GPU; gpu_trace::Init %.0f ms\n", pName, R.m_Surfels, R.m_Mismatch, R.m_Worst, R.m_Packs, R.m_UpdateMs, R.m_TraceMs, R.m_InitMs);
        CHECK(R.m_Mismatch * 200 <= R.m_Surfels, "%s: %d of %d surfels differ from the CPU world", pName, R.m_Mismatch, R.m_Surfels);
        CHECK(R.m_Packs == 1, "%s: the scene was packed %d times (no structural change after the first)", pName, R.m_Packs);
        // GI working set, job 2: GPU-owned, the world keeps no CPU solver state (accumulators, estimators, moment counts): the light state
        // per surfel well under the CPU world's
        std::printf("  the CPU's light state: CPU-solved %.0f B a surfel, GPU-owned %.0f B\n", double(R.m_CpuBytes) / std::max(R.m_Surfels, 1), double(R.m_OwnedBytes) / std::max(R.m_Surfels, 1));
        CHECK(R.m_OwnedBytes * 10 < R.m_CpuBytes * 6, "%s: GPU-owned, the CPU keeps %zu B of light state (CPU-solved %zu B): its solver's arrays", pName, R.m_OwnedBytes, R.m_CpuBytes);
    }
    {
        lights L; L.m_Points = { { { 0.2f, 0.4f, -0.3f }, { 1.5f, 1.5f, 1.5f }, 4.0f } };
        result R;
        if (Compare(App, L, false, 24, R, true)) return 1;
        std::printf("GPU trace with events (a wall recolored, another's triangles captured, one recolored while a slice is in flight, the box dragged, the coarse level), 24 updates: %d surfels, %d differ from the CPU world, the worst of the others %.1e; packed %d times\n",
                    R.m_Surfels, R.m_Mismatch, R.m_Worst, R.m_Packs, R.m_Relevels);
        CHECK(R.m_Mismatch * 200 <= R.m_Surfels, "with events: %d of %d surfels differ from the CPU world", R.m_Mismatch, R.m_Surfels);
        CHECK(R.m_Packs == 1 && R.m_Relevels >= 1, "with events the scene was packed %d times and re-leveled %d (the first pack; the coarse level, the material and the drag in place)", R.m_Packs, R.m_Relevels);
    }
    // The shown light's RGB9E5 (gi::PackRGB9E5, the filter shader's the same steps): a channel within 2^-9 of the color's largest, over
    // 1e-4..1e4; 0, negative and NaN 0; past the range its top
    {
        double Worst = 0; unsigned h = 12345;
        const auto Rnd = [&] { h = h * 1664525u + 1013904223u; return float(h >> 8) * (1.0f / 16777216.0f); };
        for (int k = 0; k < 20000; ++k)
        {
            const float Scale = std::pow(10.0f, -4.0f + 8.0f * Rnd());
            const v3 c{ Rnd() * Scale, Rnd() * Scale, Rnd() * Scale }, d = UnpackRGB9E5(PackRGB9E5(c));
            const float m = std::max({ c.m_X, c.m_Y, c.m_Z });
            if (m < 1e-4f) continue;                                    // (below the format's 9 bits: its smallest step is 2^-24)
            Worst = std::max({ Worst, double(std::abs(d.m_X - c.m_X) / m), double(std::abs(d.m_Y - c.m_Y) / m), double(std::abs(d.m_Z - c.m_Z) / m) });
        }
        const v3 Z = UnpackRGB9E5(PackRGB9E5({ 0, -1, std::numeric_limits<float>::quiet_NaN() })), Big = UnpackRGB9E5(PackRGB9E5({ 1e9f, 1, 0 }));
        std::printf("RGB9E5: the worst channel %.2e of its color's largest (20000 colors over 1e-4..1e4); 0, -1, NaN -> %g %g %g; 1e9 -> %g\n", Worst, Z.m_X, Z.m_Y, Z.m_Z, Big.m_X);
        CHECK(Worst <= 1.0 / 512 && Z.m_X == 0 && Z.m_Y == 0 && Z.m_Z == 0 && Big.m_X == 65408.0f, "RGB9E5: %.2e off; %g %g %g; %g", Worst, Z.m_X, Z.m_Y, Z.m_Z, Big.m_X);
    }
    // The slices of a batch (gpu_trace::SliceEnd): whole ranges up to the budget; a leftover under a quarter of it goes in the slice before
    // (it took a frame of its own: a third of Police's frames all but idle), a larger one is a slice of its own
    {
        const auto Slices = [](std::initializer_list<int> Rays, int Budget)
        {
            std::vector<world::gpu_range> R;
            for (const int r : Rays) { world::gpu_range g; g.m_Rays = r; R.push_back(g); }
            int n = 0;
            for (std::size_t k = 0; k < R.size(); k = gpu_trace::SliceEnd(R, k, Budget)) ++n;
            return n;
        };
        const int a = Slices({ 40, 40, 40, 40, 1 }, 80), b = Slices({ 40, 40, 40, 40, 30 }, 80), c = Slices({ 200 }, 80), d = Slices({ 10, 10 }, 80);
        std::printf("the slices of a batch: 161 rays at 80 a slice in %d (the leftover in the last), 190 in %d, one range over the budget in %d, a batch under it in %d\n", a, b, c, d);
        CHECK(a == 2 && b == 3 && c == 1 && d == 1, "the slices of a batch: %d, %d, %d, %d (2, 3, 1, 1 expected)", a, b, c, d);
    }
    // Over time: a GPU budget far too small for a batch (it is traced in slices over frames, the slices' surfels committed as they come
    // in, so later slices read newer light than the CPU's pass did): the converged light is still the CPU world's, instance by instance
    {
        lights L; L.m_Points = { { { 0.2f, 0.4f, -0.3f }, { 1.5f, 1.5f, 1.5f }, 4.0f } };
        std::vector<surfel_mesh> Mc, Mg;
        world Wc, Wg;
        MakeCornell(Mc, Wc, false); MakeCornell(Mg, Wg, false);
        Wg.m_Setup.m_bGPUTrace = Wg.m_Setup.m_bGPUOwned = true;
        Wc.m_Setup.m_RaysPerSurfel = Wg.m_Setup.m_RaysPerSurfel = 24;     // (a batch of several 1024-ray slices: at 8 it was just over one, and SliceEnd takes such a leftover whole)
        gpu_trace T;
        CHECK(T.Init(App.m_Device, s_Shaders), "gpu_trace::Init");
        const v3 Eye{ 0, 0, -0.9f };
        for (int u = 0; u < 80; ++u) Wc.Update(L, Eye, 1 << 30, 500 + u);
        int Frames = 0;
        for (; Frames < 2000 && !(T.Batches() >= 80 && T.BatchDone()); ++Frames)     // (80 batches, all their slices)
        {
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            const std::uint64_t Frame = App.m_Device.getFrameIndex();
            T.Collect(Wg, Frame);
            Wg.Update(L, Eye, 1 << 30, 500 + Frames);
            T.Submit(App.m_Window, Wg, L, Frame, 0.001f, Eye);      // (1024 rays a slice: the smallest)
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
        }
        Sync(App, T, Wg);
        double Worst = 0;
        for (int i = 0; i < Wc.InstanceCount(); ++i)
        {
            double c = 0, g = 0;
            for (const auto& e : Wc.InstanceLight(i)) c += 0.2126 * e.m_X + 0.7152 * e.m_Y + 0.0722 * e.m_Z;
            for (const auto& e : Wg.InstanceLight(i)) g += 0.2126 * e.m_X + 0.7152 * e.m_Y + 0.0722 * e.m_Z;
            Worst = std::max(Worst, std::abs(g - c) / std::max(c, 1e-6));
        }
        std::printf("GPU trace over time: %d batches in %d slices over %d frames; the instances' mean light off the CPU world's by %.1f%% at most (80 passes each: near converged; the slices commit as they come, a little ahead of the CPU before)\n",
                    T.Batches(), T.Slices(), Frames, 100 * Worst);
        CHECK(2 * T.Slices() >= 3 * T.Batches(), "the batches were not sliced (the budget starts at 32k rays: the first ones are whole) (%d batches, %d slices)", T.Batches(), T.Slices());
        CHECK(Worst < 0.03, "over time the GPU world's light is %.1f%% off the CPU world's", 100 * Worst);
    }
    // Experiment D: the links built on the GPU (gpu_trace::BuildLinks, xlionrender_gi_links_comp.glsl) against the CPU's (CompileTransport,
    // TruncateTransport: top 16, the tail through the instances) on a converged world: both solved on the CPU (SolveTransport), each
    // against the converged world's light. The rays differ (their offsets), so the two agree to their noise, not to the bit
    for (const bool bPanel : { false, true })
    {
        lights L;
        if (!bPanel) L.m_Points = { { { 0.2f, 0.4f, -0.3f }, { 1.5f, 1.5f, 1.5f }, 4.0f } };
        std::vector<surfel_mesh> M; world W;
        MakeCornell(M, W, bPanel);
        W.m_Setup.m_bEmitterSampling = bPanel;
        W.m_Setup.m_RaysPerSurfel = 64;
        W.m_Setup.m_MinBlend = 1;                                        // (converged: 8 bounces replaced, then averaged - as surfel_gi's Converge)
        for (int u = 0; u < 8; ++u) W.Update(L, { 0, 0, -0.9f }, 1 << 30, 3000 + u);
        W.m_Setup.m_MinBlend = 0;
        for (int u = 0; u < 300; ++u) W.Update(L, { 0, 0, -0.9f }, 1 << 30, 3100 + u);
        std::vector<std::array<int, 2>> Items;
        for (int i = 0; i < W.InstanceCount(); ++i) for (std::size_t s = 0; s < W.InstanceSurfels0(i); ++s) Items.push_back({ i, int(s) });
        gpu_trace G;
        CHECK(G.Init(App.m_Device, s_Shaders), "gpu_trace::Init (links)");
        while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
        CHECK(G.BuildLinks(App.m_Window, W, L, Items, 256, 0, Items.size()), "BuildLinks");
        { auto Cmd = App.m_Window.getCmdBuffer(); }
        App.m_Window.PageFlip();
        while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
        CHECK(G.CopyLinksOut(App.m_Window, Items.size()), "CopyLinksOut");      // (the links live in device-local memory)
        { auto Cmd = App.m_Window.getCmdBuffer(); }
        App.m_Window.PageFlip();
        // The probe grid (what has no surfels: skinned meshes) in the box: 8 x 8 x 8 probes 0.25 apart around its center, all of them updated
        // 6 times. The ones in the room are lit; the inner box's inside is flagged (its rays meet its back faces); by the red wall a probe's
        // irradiance toward the wall is redder than toward the green one (the bleeding), and the probes see the point light's bounce
        if (!bPanel)
        {
            for (int f = 0; f < 6; ++f)
            {
                while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
                CHECK(G.UpdateProbes(App.m_Window, v3{ 0, 0, 0 }, 0.25f, { 8, 8, 8 }, 512, std::uint64_t(f)), "UpdateProbes");
                { auto Cmd = App.m_Window.getCmdBuffer(); }
                App.m_Window.PageFlip();
            }
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            CHECK(G.CopyProbesOut(App.m_Window), "CopyProbesOut");
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
            const auto P = G.ReadProbes();
            CHECK(P.size() >= 16 + 16 * 512, "ReadProbes (%zu words)", P.size());
            const auto Slot = [&](int x, int y, int z) { const int lx = ((x % 8) + 8) % 8, ly = ((y % 8) + 8) % 8, lz = ((z % 8) + 8) % 8; return 16 + 16 * std::size_t(lx + 8 * (ly + 8 * lz)); };
            const auto E = [&](std::size_t b, const v3& n)      // its irradiance toward n (SurfelGIProbes' formula)
            {
                v3 c[4];
                for (int k = 0; k < 4; ++k) c[k] = { std::bit_cast<float>(P[b + 3 * k]), std::bit_cast<float>(P[b + 3 * k + 1]), std::bit_cast<float>(P[b + 3 * k + 2]) };
                return c[0] * (3.14159265f * 0.282095f) + (c[1] * n.m_X + c[2] * n.m_Y + c[3] * n.m_Z) * (2.0943951f * 0.488603f);
            };
            int nLit = 0, nInside = 0, nRoom = 0;
            for (int z = -3; z <= 3; ++z) for (int y = -3; y <= 3; ++y) for (int x = -3; x <= 3; ++x)
            {
                const std::size_t b = Slot(x, y, z);
                if ((P[b + 15] & 2u) != 0) { ++nInside; continue; }
                ++nRoom; nLit += (P[b + 15] & 1u) != 0 && E(b, { 0, 1, 0 }).m_X + E(b, { 0, -1, 0 }).m_X > 1e-3f;
            }
            const std::size_t Red = Slot(-3, 1, 0);           // (-0.75, 0.25, 0): by the red wall (x = -1)
            const v3 ToRed = E(Red, { -1, 0, 0 }), ToGreen = E(Red, { 1, 0, 0 });
            std::printf("  probe grid: %d of %d room probes lit, %d inside; by the red wall toward it r/g %.2f, away %.2f\n", nLit, nRoom, nInside,
                        ToRed.m_X / std::max(ToRed.m_Y, 1e-6f), ToGreen.m_X / std::max(ToGreen.m_Y, 1e-6f));
            CHECK(nLit == nRoom && nRoom > 300, "probe grid: %d of %d room probes lit", nLit, nRoom);
            CHECK(nInside > 0, "probe grid: no probe flagged inside the inner box");
            CHECK(ToRed.m_X / std::max(ToRed.m_Y, 1e-6f) > 1.5f * ToGreen.m_X / std::max(ToGreen.m_Y, 1e-6f), "probe grid: no red bleeding by the red wall");
            // An empty scene (a level's first frames, its meshes still loading): no probe update - its BVH is a placeholder, and walking it
            // lost the Vulkan device in the editor
            {
                world We; gpu_trace Ge;
                CHECK(Ge.Init(App.m_Device, s_Shaders), "gpu_trace::Init (empty)");
                while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
                We.Update(L, v3{ 0, 0, 0 }, 1 << 20, 1);
                (void)Ge.Submit(App.m_Window, We, L, 1, 1000.0f, v3{ 0, 0, 0 });
                const bool bUpdated = Ge.UpdateProbes(App.m_Window, v3{ 0, 0, 0 }, 0.25f, { 8, 8, 8 }, 512, 1);
                { auto Cmd = App.m_Window.getCmdBuffer(); }
                App.m_Window.PageFlip();
                CHECK(!bUpdated, "probe grid: updated over an empty scene");
            }
        }
        world::transport Tg; double Dropped = 0;
        CHECK(G.ReadLinks(W, Items, Tg, &Dropped), "ReadLinks");
        W.RelightTransport(Tg, L);                                       // (the GPU's first bounce: its hit clusters lit)
        const auto Tf = W.CompileTransport(L, 256, 1, true);
        const auto Tb = W.CompileTransport(L, 256, 1, false);            // (the direct light baked at the hits: what relightable costs)
        const auto Tc = W.TruncateTransport(Tf, gpu_trace::links_v, true);
        std::vector<v3> Ref;
        for (int i = 0; i < W.InstanceCount(); ++i) for (const auto& e : W.InstanceLight(i)) Ref.push_back(e);
        std::vector<v3> Eg, Ec;
        std::vector<v3> Ef;
        std::vector<v3> Eb;
        world::SolveTransport(Tg, Eg, 20); world::SolveTransport(Tc, Ec, 20); world::SolveTransport(Tf, Ef, 20); world::SolveTransport(Tb, Eb, 20);
        const auto Lum = [](const v3& e) { return 0.2126 * e.m_X + 0.7152 * e.m_Y + 0.0722 * e.m_Z; };
        const auto Off = [&](const std::vector<v3>& A, const std::vector<v3>& B)      // the worst instance's mean, relative
        {
            double Worst = 0;
            for (int i = 0; i < W.InstanceCount(); ++i)
            {
                double a = 0, b = 0;
                for (std::uint32_t r = Tg.m_InstFirst[std::size_t(i)]; r < Tg.m_InstFirst[std::size_t(i) + 1]; ++r) { a += Lum(A[r]); b += Lum(B[r]); }
                Worst = std::max(Worst, std::abs(a - b) / std::max(b, 1e-9));
            }
            return Worst;
        };
        std::printf("links on the GPU (Cornell, %s): %zu links + %zu tail entries (the CPU's %zu + %zu), %.4f of the weight dropped; solved, the instances off the converged world by %.2f%% (the CPU's links %.2f%%, every link %.2f%%), off the CPU's by %.2f%%\n",
                    bPanel ? "the emissive panel, sampled" : "a point light", Tg.m_Col.size(), Tg.m_AGroup.size(), Tc.m_Col.size(), Tc.m_AGroup.size(), Dropped, 100 * Off(Eg, Ref), 100 * Off(Ec, Ref), 100 * Off(Ef, Ref), 100 * Off(Eg, Ec));
        // (the same rays as the CPU's: the same offsets and origins; a closed box is the worst case of the top 16 - surfel_gi 6k: 4.8%)
        // The links with the CPU's exact hits: the same; with the GPU's 8 hit clusters a row (one hit each, picked): their own error
        {
            auto Tx = Tg; Tx.m_HitFirst = Tc.m_HitFirst; Tx.m_Hits = Tc.m_Hits; W.RelightTransport(Tx, L);
            std::vector<v3> Ex; world::SolveTransport(Tx, Ex, 20);
            std::printf("  the GPU's links with the CPU's hits off the CPU's by %.2f%%; its hit clusters' first bounce: %.2f%%\n", 100 * Off(Ex, Ec), 100 * Off(Eg, Ec));
            CHECK(Off(Ex, Ec) < 0.005, "the GPU's links are %.2f%% off the CPU's (the same rays and hits)", 100 * Off(Ex, Ec));
            CHECK(Off(Eg, Ec) < 0.03, "the GPU's hit clusters are %.2f%% off the CPU's hits", 100 * Off(Eg, Ec));
        }
        // relightable (the direct light at each surfel, not at each hit) against baked: close to the converged world as well
        CHECK(Off(Ef, Eb) < 1e-4, "relightable links at their own lights are %.3f%% off the baked ones", 100 * Off(Ef, Eb));
        // A light moved (relit: each row's hits lit again, nothing light-dependent in the links) against the links compiled again at the
        // new light, and against the old links left as they were (the light did not follow). The GPU's: its 8 hit clusters a row
        if (!bPanel)
        {
            lights L2 = L; L2.m_Points[0].m_Position = { -0.3f, 0.5f, 0.3f };
            auto Tr = Tc; W.RelightTransport(Tr, L2);
            const auto Tn = W.TruncateTransport(W.CompileTransport(L2, 256, 1, false), gpu_trace::links_v, true);
            std::vector<v3> Er, En, Es;
            world::SolveTransport(Tr, Er, 20); world::SolveTransport(Tn, En, 20); world::SolveTransport(Tc, Es, 20);
            auto Tgr = Tg; W.RelightTransport(Tgr, L2);
            std::vector<v3> Egr; world::SolveTransport(Tgr, Egr, 20);
            std::printf("  a light moved: relit links off the links compiled again by %.2f%% (the GPU's hit clusters: %.2f%%; the old links, not relit: %.2f%%)\n", 100 * Off(Er, En), 100 * Off(Egr, En), 100 * Off(Es, En));
            CHECK(Off(Er, En) < 0.005, "a light moved: the relit links are %.2f%% off (not relit %.2f%%)", 100 * Off(Er, En), 100 * Off(Es, En));
            CHECK(Off(Egr, En) < 0.25 * Off(Es, En), "a light moved: the GPU's relit links are %.2f%% off (not relit %.2f%%)", 100 * Off(Egr, En), 100 * Off(Es, En));
        }
        CHECK(Off(Ef, Ref) < 0.02, "every link: %.1f%% off the converged world", 100 * Off(Ef, Ref));
        CHECK(Off(Eg, Ref) < 0.15, "the GPU's links: %.1f%% off the converged world (the tail through whole instances: a closed box of big walls is its worst case)", 100 * Off(Eg, Ref));
    }
    {   // the room of the GI render test, its light (a point light, the ambient as the sky)
        lights L; L.m_Points = { { { 0.4f, 3.0f, -0.2f }, { 4, 4, 4 }, 9.0f } }; L.m_Sky = v3{ 0.02f, 0.02f, 0.02f } * (1.0f / pi_v);
        result R;
        if (Compare(App, L, false, 24, R, false, true)) return 1;
        std::printf("GPU trace, the GI render test's room, 24 updates: %d surfels, %d differ from the CPU world, the worst of the others %.1e; the GPU's shown light off the CPU's filter by %.1e (of the brightest), %.0f moment words not the CPU's\n",
                    R.m_Surfels, R.m_Mismatch, R.m_Worst, R.m_ShownWorst, R.m_MomentWorst);
        CHECK(R.m_Mismatch * 200 <= R.m_Surfels, "the room: %d of %d surfels differ from the CPU world", R.m_Mismatch, R.m_Surfels);
        CHECK(R.m_ShownWorst >= 0 && R.m_ShownWorst < 2.5e-3 && R.m_MomentWorst == 0,      // (the light in RGB9E5: 2^-9 steps)
              "the GPU's shown light is off the CPU's filter (%.1e; %.0f moment words differ)", R.m_ShownWorst, R.m_MomentWorst);
    }
    {   // GI sections: a room cut into sections, each its own instance, its ghosts copied from their owners after every commit - the GPU's
        // ghost pass against the CPU's SyncGhosts, every surfel (the ghosts too) the CPU world's
        lights L; L.m_Points = { { { 0.5f, 2.0f, 0.3f }, { 6, 6, 6 }, 8.0f } }; L.m_Sky = { 0.05f, 0.05f, 0.05f };
        result R;
        if (Compare(App, L, false, 24, R, false, false, false, true)) return 1;
        std::printf("GPU trace, a room in GI sections (ghosts copied on the GPU), 24 updates: %d surfels, %d differ from the CPU world, the worst of the others %.1e\n", R.m_Surfels, R.m_Mismatch, R.m_Worst);
        CHECK(R.m_Mismatch * 200 <= R.m_Surfels, "the sections: %d of %d surfels differ from the CPU world", R.m_Mismatch, R.m_Surfels);
    }
    {   // GI sections with the link solve (the renderer's default): the solve writes every surfel's light, the ghosts' too - they must take
        // their owners' again after it (a section's border stepped: the Bistro's wall behind the ivy, 2026-10-24)
        lights L; L.m_Points = { { { 0.5f, 2.0f, 0.3f }, { 6, 6, 6 }, 8.0f } }; L.m_Sky = { 0.05f, 0.05f, 0.05f };
        std::vector<surfel_mesh> M; world W;
        MakeSectionRoom(M, W);
        W.m_Setup.m_bGPUTrace = W.m_Setup.m_bGPUOwned = true;
        gpu_trace T;
        CHECK(T.Init(App.m_Device, s_Shaders), "gpu_trace::Init");
        std::size_t nSolves = 0;
        for (int u = 0; u < 40; ++u)
        {
            Drain(App, T, W);
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            const std::uint64_t Frame = App.m_Device.getFrameIndex();
            T.Collect(W, Frame);
            W.Update(L, v3{ 0, 1.6f, -1.9f }, 1 << 30, 100 + u);
            T.Submit(App.m_Window, W, L, Frame, 1000.0f, v3{ 0, 1.6f, -1.9f });
            T.PumpLinks(App.m_Window, W, L, Frame, 64, 4, 1 << 20);     // (links after 4 frames, all at once)
            if (std::getenv("XLION_TEST_DEBUG")) std::printf("u %d state %d solves %zu\n", u, T.LinkState(), T.LinkSolves());
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
        }
        nSolves = T.LinkSolves();
        Sync(App, T, W);
        int nGhosts = 0, nOff = 0; double Worst = 0;
        for (int i = 0; i < W.InstanceCount(); ++i)
        {
            const auto& Own = W.GhostOwners(i);
            const auto& E = W.InstanceLight(i);
            for (std::size_t k = 0; k < Own.size(); ++k)
            {
                const auto [Oi, Os] = Own[k];
                const std::size_t g = std::size_t(W.OwnedSurfels(i)) + k;
                if (Oi < 0 || g >= E.size() || std::size_t(Os) >= W.InstanceLight(Oi).size()) continue;
                const v3 a = E[g], b = W.InstanceLight(Oi)[std::size_t(Os)];
                const double d = std::max({ std::abs(a.m_X - b.m_X), std::abs(a.m_Y - b.m_Y), std::abs(a.m_Z - b.m_Z) });
                ++nGhosts; Worst = std::max(Worst, d); if (d > 1e-6) ++nOff;
            }
        }
        std::printf("GI sections with the link solve (%zu solves): %d ghosts, %d not their owners' light (worst %.1e)\n", nSolves, nGhosts, nOff, Worst);
        CHECK(nSolves > 0 && nGhosts > 0 && nOff == 0, "the link solve left %d of %d ghosts off their owners' light (%zu solves)", nOff, nGhosts, nSolves);

        // A light moved: the links (they carry each surfel's direct light) are dropped and the rays relight the room (they were kept: the
        // light never followed the lamp)
        CHECK(T.LinkState() == 2, "the links were not solving before the light moved (state %d)", T.LinkState());
        const std::size_t nBuildsBefore = T.LinkBuilds(), nRelightsBefore = T.Relights();
        std::vector<v3> Before;
        for (int i = 0; i < W.InstanceCount(); ++i) for (const auto& e : W.InstanceLight(i)) Before.push_back(e);
        L.m_Points[0].m_Position = { -1.5f, 1.0f, -1.0f };
        for (int u = 40; u < 80; ++u)
        {
            Drain(App, T, W);
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            const std::uint64_t Frame = App.m_Device.getFrameIndex();
            T.Collect(W, Frame);
            W.Update(L, v3{ 0, 1.6f, -1.9f }, 1 << 30, 100 + u);
            T.Submit(App.m_Window, W, L, Frame, 1000.0f, v3{ 0, 1.6f, -1.9f });
            T.PumpLinks(App.m_Window, W, L, Frame, 64, 4, 1 << 20);
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
        }
        Sync(App, T, W);
        double Diff = 0, Sum = 0; std::size_t k = 0;
        for (int i = 0; i < W.InstanceCount(); ++i) for (const auto& e : W.InstanceLight(i))
        {
            if (k >= Before.size()) break;
            const auto& b = Before[k++];
            Diff += std::abs(e.m_X - b.m_X) + std::abs(e.m_Y - b.m_Y) + std::abs(e.m_Z - b.m_Z); Sum += b.m_X + b.m_Y + b.m_Z;
        }
        std::printf("a light moved: the surfels' light changed by %.1f%% (the links relit: state %d)\n", 100.0 * Diff / std::max(Sum, 1e-9), T.LinkState());
        CHECK(Diff > 0.05 * Sum, "a light moved and the GI did not follow it (%.2f%% of the light changed)", 100.0 * Diff / std::max(Sum, 1e-9));
        // ... by relighting the links' hits, not by building them again: nothing light-dependent is in them (with the lamp's light baked
        // and relit by a difference, the Police level's old lamp spot stayed 64% too bright; built again, the Bistro waited 6 s)
        CHECK(T.LinkBuilds() == nBuildsBefore && T.LinkState() == 2 && T.Relights() > nRelightsBefore, "a light moved: the links were built again (%zu builds, %zu relights, state %d)", T.LinkBuilds() - nBuildsBefore, T.Relights() - nRelightsBefore, T.LinkState());

        // A material changed (an instance recolored) once the links solve: only the rows that read it are built again - the links stay
        // (state 2 throughout), the rest of the room keeps its rows, and its light follows the new color
        {
            const std::size_t nBefore = T.LocalRebuilds();
            int nState2 = 0, nFrames = 0;
            W.SetSurfelMaterial(0, std::vector<v3>(M[0].m_Levels[0].size(), v3{ 0.9f, 0.1f, 0.1f }), {});
            for (int u = 80; u < 110; ++u)
            {
                Drain(App, T, W);
                while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
                const std::uint64_t Frame = App.m_Device.getFrameIndex();
                T.Collect(W, Frame);
                W.Update(L, v3{ 0, 1.6f, -1.9f }, 1 << 30, 100 + u);
                T.Submit(App.m_Window, W, L, Frame, 1000.0f, v3{ 0, 1.6f, -1.9f });
                T.PumpLinks(App.m_Window, W, L, Frame, 64, 4, 1 << 20);
                { auto Cmd = App.m_Window.getCmdBuffer(); }
                App.m_Window.PageFlip();
                ++nFrames; nState2 += T.LinkState() == 2;
            }
            std::size_t nRows = 0; for (int i = 0; i < W.InstanceCount(); ++i) nRows += W.InstanceLight(i).size();
            std::printf("a material changed: %zu local rebuilds, %zu rows of %zu built again; the links solving %d of %d frames\n", T.LocalRebuilds() - nBefore, T.LocalRows(), nRows, nState2, nFrames);
            CHECK(T.LocalRebuilds() > nBefore && nState2 == nFrames && T.LocalRows() < nRows, "a material change rebuilt %zu rows of %zu (local rebuilds %zu, solving %d of %d frames)", T.LocalRows(), nRows, T.LocalRebuilds() - nBefore, nState2, nFrames);
        }
        // A static object moved once the links solve: a local rebuild too (its rows, the rows reading it, the rows near it), the links stay
        {
            const std::size_t nBefore = T.LocalRebuilds();
            int nState2 = 0, nFrames = 0;
            W.SetTransform(1, xmath::fmat4::fromTranslation({ 0.05f, 0, 0 }));
            for (int u = 110; u < 140; ++u)
            {
                Drain(App, T, W);
                while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
                const std::uint64_t Frame = App.m_Device.getFrameIndex();
                T.Collect(W, Frame);
                W.Update(L, v3{ 0, 1.6f, -1.9f }, 1 << 30, 100 + u);
                T.Submit(App.m_Window, W, L, Frame, 1000.0f, v3{ 0, 1.6f, -1.9f });
                T.PumpLinks(App.m_Window, W, L, Frame, 64, 4, 1 << 20);
                { auto Cmd = App.m_Window.getCmdBuffer(); }
                App.m_Window.PageFlip();
                ++nFrames; nState2 += T.LinkState() == 2;
            }
            std::printf("a static object moved: %zu local rebuilds; the links solving %d of %d frames\n", T.LocalRebuilds() - nBefore, nState2, nFrames);
            CHECK(T.LocalRebuilds() > nBefore && nState2 == nFrames, "a move: %zu local rebuilds, solving %d of %d frames (the links dropped)", T.LocalRebuilds() - nBefore, nState2, nFrames);
        }
    }
    std::printf("PASS\n");
    return 0;
}
