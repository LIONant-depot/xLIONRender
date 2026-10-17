// Part II, G3 (documentation/gi_dynamic_master_plan.md): the solver with its trace on the GPU (gi::gpu_trace, world::m_Setup.m_bGPUTrace).
// A world whose batches the GPU traces (the scene resident, the light the commits change uploaded between batches, the sums delivered
// the frame after) must hold the light of the same world traced on the CPU, update for update: the Cornell box (a point light), and a
// room lit by an emissive panel through its emitter links. The CPU's main-thread time per update in GPU mode is printed
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
        // the same picks and rays on both sides: every surfel its full rays each pass (the adaptive rays and the tiers read the estimators
        // when it is picked, which the GPU world has committed one update later)
        W.m_Setup.m_bAdaptiveRays = false; W.m_Setup.m_bDistanceTiers = false;
    }

    struct result { int m_Surfels = 0, m_Mismatch = 0; double m_Worst = 0, m_UpdateMs = 0, m_TraceMs = 0; int m_Packs = 0; };

    // Updates of the CPU world against the GPU world's (it commits a batch one update later: one update more)
    int Compare(app& App, const lights& L, bool bPanel, int Updates, result& R)
    {
        R = {};
        std::vector<surfel_mesh> Mc, Mg;
        world Wc, Wg;
        MakeCornell(Mc, Wc, bPanel); MakeCornell(Mg, Wg, bPanel);
        Wg.m_Setup.m_bGPUTrace = true;
        gpu_trace T;
        CHECK(T.Init(App.m_Device, s_PassComp, s_ScatterComp), "gpu_trace::Init");
        const v3 Eye{ 0, 0, -0.9f };
        for (int u = 0; u < Updates; ++u) Wc.Update(L, Eye, 1 << 30, 100 + u);
        for (int u = 0; u <= Updates; ++u)
        {
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            const std::uint64_t Frame = App.m_Device.getFrameIndex();
            const auto T0 = std::chrono::steady_clock::now();
            T.Collect(Wg, Frame);
            Wg.Update(L, Eye, 1 << 30, 100 + u);
            T.Submit(App.m_Window, Wg, L, Frame, 1000.0f);     // (a budget for all of a batch: one slice, the CPU's order)
            if (u > 0) R.m_UpdateMs = std::max(R.m_UpdateMs, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count());
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
            if (u > 0) R.m_TraceMs = std::max(R.m_TraceMs, T.LastMs());
        }
        R.m_Packs = T.Packs();
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
                ++R.m_Surfels;
                if (d > 1e-3 || v > 1e-3) ++R.m_Mismatch;
                else R.m_Worst = std::max(R.m_Worst, d);
            }
        }
        return 0;
    }
}

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    for (const bool bPanel : { false, true })
    {
        lights L;
        if (!bPanel) L.m_Points = { { { 0.2f, 0.4f, -0.3f }, { 1.5f, 1.5f, 1.5f }, 4.0f } };
        result R;
        if (Compare(App, L, bPanel, 8, R)) return 1;
        const char* pName = bPanel ? "Cornell box lit by an emissive panel (links)" : "Cornell box, a point light";
        std::printf("GPU trace, %s, 8 updates: %d surfels, %d differ from the CPU world, the worst of the others %.1e; the scene packed %d times; "
                    "an update in GPU mode (collect, Update, submit) %.2f ms at most, a batch %.3f ms on the GPU\n", pName, R.m_Surfels, R.m_Mismatch, R.m_Worst, R.m_Packs, R.m_UpdateMs, R.m_TraceMs);
        CHECK(R.m_Mismatch * 200 <= R.m_Surfels, "%s: %d of %d surfels differ from the CPU world", pName, R.m_Mismatch, R.m_Surfels);
        CHECK(R.m_Packs == 1, "%s: the scene was packed %d times (no structural change after the first)", pName, R.m_Packs);
    }
    // Over time: a GPU budget far too small for a batch (it is traced in slices over frames, the slices' surfels committed as they come
    // in, so later slices read newer light than the CPU's pass did): the converged light is still the CPU world's, instance by instance
    {
        lights L; L.m_Points = { { { 0.2f, 0.4f, -0.3f }, { 1.5f, 1.5f, 1.5f }, 4.0f } };
        std::vector<surfel_mesh> Mc, Mg;
        world Wc, Wg;
        MakeCornell(Mc, Wc, false); MakeCornell(Mg, Wg, false);
        Wg.m_Setup.m_bGPUTrace = true;
        gpu_trace T;
        CHECK(T.Init(App.m_Device, s_PassComp, s_ScatterComp), "gpu_trace::Init");
        const v3 Eye{ 0, 0, -0.9f };
        for (int u = 0; u < 40; ++u) Wc.Update(L, Eye, 1 << 30, 500 + u);
        int Frames = 0;
        for (; Frames < 2000 && T.Batches() <= 40; ++Frames)
        {
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            const std::uint64_t Frame = App.m_Device.getFrameIndex();
            T.Collect(Wg, Frame);
            Wg.Update(L, Eye, 1 << 30, 500 + Frames);
            T.Submit(App.m_Window, Wg, L, Frame, 0.001f);           // (1024 rays a slice: the smallest)
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
        }
        double Worst = 0;
        for (int i = 0; i < Wc.InstanceCount(); ++i)
        {
            double c = 0, g = 0;
            for (const auto& e : Wc.InstanceLight(i)) c += 0.2126 * e.m_X + 0.7152 * e.m_Y + 0.0722 * e.m_Z;
            for (const auto& e : Wg.InstanceLight(i)) g += 0.2126 * e.m_X + 0.7152 * e.m_Y + 0.0722 * e.m_Z;
            Worst = std::max(Worst, std::abs(g - c) / std::max(c, 1e-6));
        }
        std::printf("GPU trace over time: %d batches in %d slices over %d frames; the instances' mean light off the CPU world's by %.1f%% at most (40 passes each)\n",
                    T.Batches(), T.Slices(), Frames, 100 * Worst);
        CHECK(2 * T.Slices() >= 3 * T.Batches(), "the batches were not sliced (the budget starts at 32k rays: the first ones are whole) (%d batches, %d slices)", T.Batches(), T.Slices());
        CHECK(Worst < 0.03, "over time the GPU world's light is %.1f%% off the CPU world's", 100 * Worst);
    }
    std::printf("PASS\n");
    return 0;
}
