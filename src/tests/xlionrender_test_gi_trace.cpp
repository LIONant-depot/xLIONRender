// Part II, G3 and G4 (documentation/gi_dynamic_master_plan.md): the solver on the GPU (gi::gpu_trace, world::m_Setup.m_bGPUOwned: the pick
// as ranges, the light living on the GPU). A world whose light the GPU makes (expanded, traced, committed, its changes as ops, moved and
// re-levelled when the scene is packed again) must hold, read back, the light of the same world on the CPU, update for update: the
// Cornell box (a point light; an emissive panel through its links; events), the GI render test's room, and its shown light against the
// CPU's filter. The CPU's main-thread time per update in GPU mode is printed
#include "xlionrender_test_common.h"
#include "features/gi/xlionrender_surfel_gi_gpu.h"

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
    constexpr std::uint32_t s_ReadBackComp[] = {
        #include "xlionrender_gi_readback_comp.h"
    };
    const gpu_trace::shaders s_Shaders{ s_PassComp, s_ScatterComp, s_CommitComp, s_OpsComp, s_FilterComp, s_ExpandComp, s_MeansComp, s_CubeComp, s_MigrateComp, s_ReadBackComp };

    // The GPU's light into the world's arrays: the last slice delivered, the read back copied, then in (two frames)
    void Sync(app& App, gpu_trace& T, world& W)
    {
        for (int f = 0; f < 2; ++f)
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

    struct result { int m_Surfels = 0, m_Mismatch = 0; double m_Worst = 0, m_UpdateMs = 0, m_TraceMs = 0; int m_Packs = 0; double m_ShownWorst = -1, m_MomentWorst = -1, m_InitMs = 0; };

    // Updates of the CPU world against the GPU world's (it commits a batch one update later: one update more). bEvents: on both at the
    // same updates, a wall recolored (its material: the scene packed again), the box moved (estimator ops: Lower, the changes near it),
    // the eye far away (the coarse level: the light carried over)
    int Compare(app& App, const lights& L, bool bPanel, int Updates, result& R, bool bEvents = false, bool bRoom = false, bool bSample = false)
    {
        R = {};
        std::vector<surfel_mesh> Mc, Mg;
        world Wc, Wg;
        if (bRoom) { MakeRoom(Mc, Wc); MakeRoom(Mg, Wg); }
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
        for (int u = 0; u < Updates; ++u) { Event(Wc, Mc, u); Wc.Update(L, EyeAt(u), 1 << 30, 100 + u); Late(Wc, Mc, u); }
        for (int u = 0; u < Updates; ++u)       // (the GPU commits a batch in the frame it traces it: the same updates)
        {
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            const std::uint64_t Frame = App.m_Device.getFrameIndex();
            const auto T0 = std::chrono::steady_clock::now();
            T.Collect(Wg, Frame);
            Event(Wg, Mg, u);
            Wg.Update(L, EyeAt(u), 1 << 30, 100 + u);
            T.Submit(App.m_Window, Wg, L, Frame, 1000.0f, EyeAt(u));     // (a budget for all of a batch: one slice, the CPU's order)
            Late(Wg, Mg, u);
            if (u > 0) R.m_UpdateMs = std::max(R.m_UpdateMs, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count());
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
            if (u > 0) R.m_TraceMs = std::max(R.m_TraceMs, T.LastMs());
        }
        R.m_Packs = T.Packs();
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
            std::vector<int> Places(std::size_t(Wg.InstanceCount()), -1);
            int nRing = 0;
            for (int i = 0; i < Wg.InstanceCount(); ++i) { Places[std::size_t(i)] = nRing; nRing += int(Wg.InstanceLight(i).size()) + 1; }
            xgpu::buffer Light, Moments;
            CHECK(!App.m_Device.Create(Light, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 16, .m_EntryCount = 3 * nRing })
               && !App.m_Device.Create(Moments, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 4, .m_EntryCount = 16 * nRing }), "rings");
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            CHECK(T.Show(App.m_Window, Places, Light, Moments, 0, Wg.m_Setup.m_FilterPasses, Wg.m_Setup.m_Saturation), "Show");
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
            std::vector<std::array<float, 4>> L4(std::size_t(3 * nRing)); std::vector<std::uint32_t> M4(std::size_t(16 * nRing));
            CHECK(!Light.MemoryMap(0, 3 * nRing, [&](void* q) { std::memcpy(L4.data(), q, L4.size() * 16); }) && !Moments.MemoryMap(0, 16 * nRing, [&](void* q) { std::memcpy(M4.data(), q, M4.size() * 4); }), "read the rings");
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
                    const std::size_t At = std::size_t(Places[std::size_t(i)] + 1) + q;
                    for (int k = 0; k < 3; ++k)
                    {
                        const auto& g = L4[3 * At + std::size_t(k)];
                        R.m_ShownWorst = std::max({ R.m_ShownWorst, std::abs(g[0] - B[q][k].m_X) / Top, std::abs(g[1] - B[q][k].m_Y) / Top, std::abs(g[2] - B[q][k].m_Z) / Top });
                    }
                    R.m_ShownWorst = std::max(R.m_ShownWorst, double(std::abs(L4[3 * At][3] - V[q])));
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
    }
    {
        lights L; L.m_Points = { { { 0.2f, 0.4f, -0.3f }, { 1.5f, 1.5f, 1.5f }, 4.0f } };
        result R;
        if (Compare(App, L, false, 24, R, true)) return 1;
        std::printf("GPU trace with events (a wall recolored, another's triangles captured, one recolored while a slice is in flight, the box dragged, the coarse level), 24 updates: %d surfels, %d differ from the CPU world, the worst of the others %.1e; packed %d times\n",
                    R.m_Surfels, R.m_Mismatch, R.m_Worst, R.m_Packs);
        CHECK(R.m_Mismatch * 200 <= R.m_Surfels, "with events: %d of %d surfels differ from the CPU world", R.m_Mismatch, R.m_Surfels);
        CHECK(R.m_Packs == 2, "with events the scene was packed %d times (the first, the coarse level; the material and the drag in place)", R.m_Packs);
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
    {   // the room of the GI render test, its light (a point light, the ambient as the sky)
        lights L; L.m_Points = { { { 0.4f, 3.0f, -0.2f }, { 4, 4, 4 }, 9.0f } }; L.m_Sky = v3{ 0.02f, 0.02f, 0.02f } * (1.0f / pi_v);
        result R;
        if (Compare(App, L, false, 24, R, false, true)) return 1;
        std::printf("GPU trace, the GI render test's room, 24 updates: %d surfels, %d differ from the CPU world, the worst of the others %.1e; the GPU's shown light off the CPU's filter by %.1e (of the brightest), %.0f moment words not the CPU's\n",
                    R.m_Surfels, R.m_Mismatch, R.m_Worst, R.m_ShownWorst, R.m_MomentWorst);
        CHECK(R.m_Mismatch * 200 <= R.m_Surfels, "the room: %d of %d surfels differ from the CPU world", R.m_Mismatch, R.m_Surfels);
        CHECK(R.m_ShownWorst >= 0 && R.m_ShownWorst < 1e-4 && R.m_MomentWorst == 0, "the GPU's shown light is off the CPU's filter (%.1e; %.0f moment words differ)", R.m_ShownWorst, R.m_MomentWorst);
    }
    std::printf("PASS\n");
    return 0;
}
