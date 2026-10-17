// The GI's flicker (documentation/gi_dynamic_master_plan.md): once a scene has converged, how much its light moves from one update to the
// next. The room of xlionrender_test_gi_render (boxes of cube::Generate, a point light), on the CPU (the GPU's world is the same, surfel for
// surfel: xlionrender_test_gi_trace): every surfel traced every update, the adaptive rays on. Measured: the solved light (E), the validity,
// the depth moments, and the shown light (the filtered basis the pixels read), each as the luminance-weighted change between updates and
// the 90th percentile of the lit surfels' relative change
#include "xlionrender_test_common.h"

using namespace xlionrender::test;
using namespace xlionrender::gi;

namespace
{
    struct change { double m_Weighted = 0, m_P90 = 0, m_Watching = 0; };
    // the share of a world's surfels still watching for a change (estimator::Lower) and so at a few samples: once settled, none should
    double Watching(const world& W)
    {
        std::size_t n = 0, w = 0;
        for (int i = 0; i < W.InstanceCount(); ++i) for (int s = 0; s < int(W.InstanceLight(i).size()); ++s) { ++n; w += W.Estimator(i, s).m_bWatch ? 1 : 0; }
        return double(w) / double(std::max<std::size_t>(n, 1));
    }
    change Change(const std::vector<double>& A, const std::vector<double>& B)
    {
        double Tot = 0, Moved = 0;
        for (std::size_t k = 0; k < A.size(); ++k) { Tot += std::abs(A[k]); Moved += std::abs(A[k] - B[k]); }
        const double Mean = Tot / std::max<std::size_t>(A.size(), 1);
        std::vector<double> Rel;
        for (std::size_t k = 0; k < A.size(); ++k) if (std::abs(A[k]) > 0.1 * Mean) Rel.push_back(std::abs(A[k] - B[k]) / std::abs(A[k]));
        std::sort(Rel.begin(), Rel.end());
        return { Moved / std::max(Tot, 1e-12), Rel.empty() ? 0 : Rel[Rel.size() * 9 / 10] };
    }
    double Lum(const v3& c) { return 0.2126 * c.m_X + 0.7152 * c.m_Y + 0.0722 * c.m_Z; }
}

int main()
{
    struct piece { xprim_geom::float3 m_Size; xmath::fvec3 m_Pos; v3 m_Color; };
    const piece Pieces[] =
    { { { 4.0f, 0.2f, 4.0f }, { 0, -0.1f, 0 }, { 0.75f, 0.75f, 0.75f } }, { { 4.0f, 0.2f, 4.0f }, { 0, 4.1f, 0 }, { 0.75f, 0.75f, 0.75f } }
    , { { 4.0f, 4.0f, 0.2f }, { 0, 2, 2.1f }, { 0.75f, 0.75f, 0.75f } }, { { 0.2f, 4.0f, 4.0f }, { -2.1f, 2, 0 }, { 0.8f, 0.05f, 0.05f } }
    , { { 0.2f, 4.0f, 4.0f }, { 2.1f, 2, 0 }, { 0.05f, 0.7f, 0.05f } }, { { 0.8f, 1.6f, 0.8f }, { -0.6f, 0.8f, 0.6f }, { 0.75f, 0.75f, 0.75f } } };
    std::vector<surfel_mesh> M;
    for (const auto& P : Pieces) M.push_back(MakeSurfels(xprim_geom::cube::Generate(15, 15, 15, 15, P.m_Size), 0.3f, 0.9f));
    world W;
    for (std::size_t i = 0; i < std::size(Pieces); ++i) W.Add(M[i], xmath::fmat4::fromTranslation(Pieces[i].m_Pos), Pieces[i].m_Color);
    lights L; L.m_Points = { { { 0.4f, 3.0f, -0.2f }, { 4, 4, 4 }, 9.0f } }; L.m_Sky = v3{ 0.02f, 0.02f, 0.02f } * (1.0f / pi_v);
    W.m_Setup.m_FilterPasses = 2;
    const v3 Eye{ 0, 1.6f, -1.9f };
    for (int u = 0; u < 200; ++u) W.Update(L, Eye, 1 << 30, 900 + u);        // converged

    const auto Snap = [&](std::vector<double>& E, std::vector<double>& V, std::vector<double>& Mo, std::vector<double>& S)
    {
        E.clear(); V.clear(); Mo.clear(); S.clear();
        for (int i = 0; i < W.InstanceCount(); ++i)
        {
            for (const auto& e : W.InstanceLight(i)) E.push_back(Lum(e));
            for (const float v : W.InstanceValid(i)) V.push_back(v);
            for (const auto& m : W.InstanceMoments(i)) for (int t = 0; t < moment_texels_v; ++t) Mo.push_back(m[std::size_t(2 * t)]);
            for (const auto& b : W.BasisLight(i)) S.push_back(Lum(b[0]) + Lum(b[1]) + Lum(b[2]));
        }
    };
    std::vector<double> E0, V0, M0, S0, E1, V1, M1, S1;
    change cE, cV, cM, cS; int n = 0;
    Snap(E0, V0, M0, S0);
    for (int u = 0; u < 20; ++u)
    {
        W.Update(L, Eye, 1 << 30, 1100 + u);
        Snap(E1, V1, M1, S1);
        const auto Acc = [&](change& A, const change& B) { A.m_Weighted += B.m_Weighted; A.m_P90 += B.m_P90; };
        Acc(cE, Change(E0, E1)); Acc(cV, Change(V0, V1)); Acc(cM, Change(M0, M1)); Acc(cS, Change(S0, S1)); ++n;
        std::swap(E0, E1); std::swap(V0, V1); std::swap(M0, M1); std::swap(S0, S1);
    }
    const auto P = [&](const char* pName, const change& c) { std::printf("  %-14s moves %5.2f%% an update (luminance weighted), the lit surfels' p90 %5.2f%%\n", pName, 100 * c.m_Weighted / n, 100 * c.m_P90 / n); };
    std::printf("flicker, the GI render test's room after 200 updates (8 rays, adaptive):\n");
    P("light (E)", cE); P("validity", cV); P("moments", cM); P("shown", cS);
    std::printf("  still watching for a change: %.1f%% of the surfels\n", 100 * Watching(W));
    CHECK(Watching(W) < 0.01, "settled, %.1f%% of the room's surfels still watch for a change", 100 * Watching(W));
    CHECK(cS.m_P90 / n < 0.01, "the room's shown light moves %.2f%% an update (p90)", 100 * cS.m_P90 / n);

    // Many lamps, most hidden (a building's): a floor under a grid of 64 emissive tiles, a slab between it and all but 4 of them. With the
    // links (visibility tested once) and with the light sampling (an emitter picked by its weight, its visibility by its shadow ray)
    const auto Lamps = [&](bool bSample, float Power, double& Mean) -> change
    {
        surfel_mesh Floor, Slab, Tile;
        const auto Quad = [](surfel_mesh& Q, v3 C, v3 U, v3 V, float Split)
        {
            const int nu = std::max(1, int(std::ceil(U.Length() / Split))), nv = std::max(1, int(std::ceil(V.Length() / Split)));
            const v3 N = U.Cross(V).NormalizeCopy(); const int Base = int(Q.m_Pos.size());
            for (int j = 0; j <= nv; ++j) for (int i = 0; i <= nu; ++i) { Q.m_Pos.push_back(C + U * (float(i) / nu) + V * (float(j) / nv)); Q.m_Nrm.push_back(N); }
            for (int j = 0; j < nv; ++j) for (int i = 0; i < nu; ++i) { const int a = Base + j * (nu + 1) + i, b = a + 1, c = a + nu + 1, d = c + 1; Q.m_Tris.push_back({ a, b, d }); Q.m_Tris.push_back({ a, d, c }); }
        };
        Quad(Floor, { -4, 0, -4 }, { 0, 0, 8 }, { 8, 0, 0 }, 0.5f);
        Quad(Slab, { -8, 1.5f, -8 }, { 16, 0, 0 }, { 0, 0, 16 }, 1.0f);          // facing down, over the floor, below the lamps
        Quad(Tile, { -0.15f, 0, -0.15f }, { 0.3f, 0, 0 }, { 0, 0, 0.3f }, 0.1f);   // facing down
        Floor.Bake(0.3f, 0.9f); Slab.Bake(0.3f, 0.9f); Tile.Bake(0.08f, 0.24f);
        world D;
        D.m_Setup.m_bEmitterSampling = bSample;
        D.Add(Floor, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        int iSlab = -1;
        for (int z = 0; z < 8; ++z) for (int x = 0; x < 8; ++x)
        {
            const bool bSeen = (x == 3 || x == 4) && (z == 3 || z == 4);
            const int t = D.Add(Tile, xmath::fmat4::fromTranslation({ float(x) * 2 - 7, bSeen ? 1.2f : 3.0f, float(z) * 2 - 7 }), { 0.5f, 0.5f, 0.5f });
            D.SetSurfelMaterial(t, {}, std::vector<v3>(Tile.m_Levels[0].size(), v3{ Power, Power, Power }));
        }
        iSlab = D.Add(Slab, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        (void)iSlab;
        lights Dark;
        // after 30 updates the lamps twice as bright (a change: estimator::Lower, the watch for a new level; as the renderer's material
        // captures come after the first passes), then settled
        for (int u = 0; u < 200; ++u)
        {
            if (u == 30) for (int t = 1; t <= 64; ++t) D.SetSurfelMaterial(t, {}, std::vector<v3>(Tile.m_Levels[0].size(), v3{ 2 * Power, 2 * Power, 2 * Power }));
            D.Update(Dark, { 0, 1, 0 }, 1 << 30, 3000 + u);
        }
        std::vector<double> A, B;
        const auto Floor0 = [&](std::vector<double>& Out) { Out.clear(); for (const auto& e : D.InstanceLight(0)) Out.push_back(Lum(e)); };
        Floor0(A);
        change C; int k = 0;
        for (int u = 0; u < 20; ++u)
        {
            D.Update(Dark, { 0, 1, 0 }, 1 << 30, 3200 + u);
            Floor0(B);
            const auto c = Change(A, B); C.m_Weighted += c.m_Weighted; C.m_P90 += c.m_P90; ++k;
            std::swap(A, B);
        }
        Mean = 0; for (const double a : A) Mean += a; Mean /= std::max<std::size_t>(A.size(), 1);
        C.m_Weighted /= k; C.m_P90 /= k; C.m_Watching = Watching(D);
        return C;
    };
    for (const float Power : { 40.0f, 1.3f })        // (1.3: as dim as Police: its light about 0.03)
    {
        double MLinks, MSample;
        const change cL = Lamps(false, Power, MLinks), cSm = Lamps(true, Power, MSample);
        std::printf("flicker under 64 lamps (radiance %.1f), 60 hidden (the floor, after 200 updates):\n", Power);
        std::printf("  links          moves %5.2f%% an update, p90 %5.2f%%; the floor's light %.4f; still watching %.1f%%\n", 100 * cL.m_Weighted, 100 * cL.m_P90, MLinks, 100 * cL.m_Watching);
        std::printf("  light sampling moves %5.2f%% an update, p90 %5.2f%%; the floor's light %.4f; still watching %.1f%%\n", 100 * cSm.m_Weighted, 100 * cSm.m_P90, MSample, 100 * cSm.m_Watching);
        CHECK(cL.m_Watching < 0.01 && cSm.m_Watching < 0.01, "settled, %.1f%% / %.1f%% of the surfels still watch for a change (at a few samples: the flicker)", 100 * cL.m_Watching, 100 * cSm.m_Watching);
        CHECK(cSm.m_P90 < 0.01, "the light sampling's light moves %.2f%% an update (p90)", 100 * cSm.m_P90);
    }
    // The emitters' light against the exact one: the same 64 lamps, no bounces (every albedo 0), so a floor surfel's light is the sum over
    // the emitter surfels it sees of L w (world::EmitterWeight, the shadow ray as the sampling's). The sampling, the links (their caps).
    // (G5b, ReSTIR's temporal reuse of an emitter reservoir per surfel, was measured here and dropped: 19.6% error after 16 updates against
    // 14.8% without; the 8 candidates are all shadow tested already, so carrying the reservoir over only correlates the passes)
    {
        surfel_mesh Floor, Slab, Tile;
        const auto Quad = [](surfel_mesh& Q, v3 C, v3 U, v3 V, float Split)
        {
            const int nu = std::max(1, int(std::ceil(U.Length() / Split))), nv = std::max(1, int(std::ceil(V.Length() / Split)));
            const v3 Nq = U.Cross(V).NormalizeCopy(); const int Base = int(Q.m_Pos.size());
            for (int j = 0; j <= nv; ++j) for (int i = 0; i <= nu; ++i) { Q.m_Pos.push_back(C + U * (float(i) / nu) + V * (float(j) / nv)); Q.m_Nrm.push_back(Nq); }
            for (int j = 0; j < nv; ++j) for (int i = 0; i < nu; ++i) { const int a = Base + j * (nu + 1) + i, b = a + 1, c = a + nu + 1, d = c + 1; Q.m_Tris.push_back({ a, b, d }); Q.m_Tris.push_back({ a, d, c }); }
        };
        Quad(Floor, { -4, 0, -4 }, { 0, 0, 8 }, { 8, 0, 0 }, 0.5f);
        Quad(Slab, { -8, 1.5f, -8 }, { 16, 0, 0 }, { 0, 0, 16 }, 1.0f);
        Quad(Tile, { -0.15f, 0, -0.15f }, { 0.3f, 0, 0 }, { 0, 0, 0.3f }, 0.1f);
        Floor.Bake(0.3f, 0.9f); Slab.Bake(0.3f, 0.9f); Tile.Bake(0.08f, 0.24f);
        // 0 links, 1 sampling: the floor's mean light; Exact: the exact mean; Err: the floor's error (rms over its surfels, of the exact
        // mean) after 4, 16 and 120 updates
        const auto Run = [&](int Mode, double& Exact, double (&Err)[3]) -> double
        {
            world D;
            D.m_Setup.m_bEmitterSampling = Mode == 1;
            D.Add(Floor, xmath::fmat4::fromIdentity(), { 0, 0, 0 });
            std::vector<v3> At;
            for (int z = 0; z < 8; ++z) for (int x = 0; x < 8; ++x)
            {
                const bool bSeen = (x == 3 || x == 4) && (z == 3 || z == 4);
                At.push_back({ float(x) * 2 - 7, bSeen ? 1.2f : 3.0f, float(z) * 2 - 7 });
                const int t = D.Add(Tile, xmath::fmat4::fromTranslation({ At.back().m_X, At.back().m_Y, At.back().m_Z }), { 0, 0, 0 });
                D.SetSurfelMaterial(t, {}, std::vector<v3>(Tile.m_Levels[0].size(), v3{ 10, 10, 10 }));
            }
            D.Add(Slab, xmath::fmat4::fromIdentity(), { 0, 0, 0 });
            lights Dark;
            const auto& L0 = Floor.m_Levels[0];
            std::vector<double> X(L0.size(), 0.0);
            int iErr = 0;
            for (int u = 0; u < 120; ++u)
            {
                D.Update(Dark, { 0, 1, 0 }, 1 << 30, 5000 + u);
                if (u == 0)
                    for (std::size_t s = 0; s < L0.size(); ++s)
                    {
                        const v3 C = L0[s].m_Center, Nn = L0[s].m_Normal;
                        for (int t = 1; t <= 64; ++t)
                            for (const auto& T : Tile.m_Levels[0])
                            {
                                const v3 Ce = At[std::size_t(t - 1)] + T.m_Center, Ne{ 0, -1, 0 }, Dv = Ce - C;
                                const float d2 = Dv.Dot(Dv), A = 1.47f * T.m_Radius * T.m_Radius;
                                const v3 Dn = Dv * (1.0f / std::sqrt(d2));
                                const float cr = Nn.Dot(Dn), ce = -Ne.Dot(Dn);
                                if (cr <= 0 || ce <= 0) continue;
                                const float w = A * cr * ce / (d2 + A);
                                if (w < 1e-5f) continue;
                                if (D.Occluded(C + Nn * 1e-3f, Dn, std::sqrt(d2) * 0.98f)) continue;
                                X[s] += 10.0 * w;
                            }
                    }
                if (u == 3 || u == 15 || u == 119)
                {
                    double e2 = 0, xm = 0;
                    for (std::size_t s = 0; s < L0.size(); ++s) { const double d = Lum(D.InstanceLight(0)[s]) - X[s]; e2 += d * d; xm += X[s]; }
                    xm /= double(L0.size());
                    Err[iErr++] = std::sqrt(e2 / double(L0.size())) / std::max(xm, 1e-12);
                }
            }
            double m = 0, e = 0;
            for (std::size_t s = 0; s < L0.size(); ++s) { m += Lum(D.InstanceLight(0)[s]); e += X[s]; }
            Exact = e / double(L0.size());
            return m / double(L0.size());
        };
        double X0, X1, E0[3], E1[3];
        const double Links = Run(0, X0, E0), Samp = Run(1, X1, E1);
        std::printf("the emitters' light under 64 lamps, no bounces: exact %.4f; light sampling %.4f (%+.1f%%); links %.4f (%+.1f%%)\n", X1, Samp, 100 * (Samp / X1 - 1), Links, 100 * (Links / X0 - 1));
        std::printf("  the floor's error (rms, of the mean) after 4 / 16 / 120 updates: light sampling %.1f%% / %.1f%% / %.1f%%; links %.1f%% / %.1f%% / %.1f%%\n",
                    100 * E1[0], 100 * E1[1], 100 * E1[2], 100 * E0[0], 100 * E0[1], 100 * E0[2]);
        CHECK(std::abs(Samp / X1 - 1) < 0.03, "the sampled emitters' light is %.1f%% off the exact one", 100 * (Samp / X1 - 1));
        CHECK(E1[2] < 0.12, "the sampled emitters' light is %.1f%% off the exact one per surfel after 120 updates (rms)", 100 * E1[2]);
    }
    // 7a (gi_dynamic_master_plan.md): the start with the materials captured over the first updates, as the host's captures come in. Every
    // piece of the room added with a white albedo (a material's base color factor: what an instance has until its capture), its colors
    // captured one piece every 3 updates. (a) captured before the first update (the control), (b) streamed, (c) streamed with the pieces
    // provisional until captured (world::SetProvisional). The reference: (a) after 800 updates. After 200: (c) never overshoots, is within
    // 5 % no later than (b), and is as close to the reference as (a) is
    {
        const auto Start = [&](int Mode, std::vector<double>& Mean, std::vector<double>& Final, int nUpdates)
        {
            world D;
            for (std::size_t i = 0; i < std::size(Pieces); ++i)
            {
                D.Add(M[i], xmath::fmat4::fromTranslation(Pieces[i].m_Pos), { 1, 1, 1 });
                if (Mode == 2) D.SetProvisional(int(i), true);
            }
            const auto Capture = [&](int i) { D.SetSurfelMaterial(i, std::vector<v3>(M[std::size_t(i)].m_Levels[0].size(), Pieces[i].m_Color), {}); if (Mode == 2) D.SetProvisional(i, false); };
            if (Mode == 0) for (int i = 0; i < int(std::size(Pieces)); ++i) Capture(i);
            D.m_Setup.m_FilterPasses = 2;
            Mean.clear();
            for (int u = 0; u < nUpdates; ++u)
            {
                if (Mode != 0 && u % 3 == 2 && u / 3 < int(std::size(Pieces))) Capture(u / 3);
                D.Update(L, Eye, 1 << 30, 7000 + u);
                double m = 0; std::size_t n = 0;
                for (int i = 0; i < D.InstanceCount(); ++i) for (const auto& e : D.InstanceLight(i)) { m += Lum(e); ++n; }
                Mean.push_back(m / double(n));
            }
            Final.clear();
            for (int i = 0; i < D.InstanceCount(); ++i) for (const auto& e : D.InstanceLight(i)) Final.push_back(Lum(e));
        };
        std::vector<double> MR, MA, MB, MC, FR, FA, FB, FC;
        Start(0, MR, FR, 800); Start(0, MA, FA, 200); Start(1, MB, FB, 200); Start(2, MC, FC, 200);
        const double Ref = MR.back();
        const auto Peak = [&](const std::vector<double>& m) { double p = 0; for (std::size_t u = 18; u < m.size(); ++u) p = std::max(p, m[u] / Ref); return p; };   // (after the last capture)
        const auto Within = [&](const std::vector<double>& m, double Tol) { int Last = -1; for (int u = 0; u < int(m.size()); ++u) if (std::abs(m[std::size_t(u)] / Ref - 1) > Tol) Last = u; return Last + 1; };
        const auto Rms = [&](const std::vector<double>& F) { double e = 0, t = 0; for (std::size_t k = 0; k < F.size(); ++k) { e += (F[k] - FR[k]) * (F[k] - FR[k]); t += FR[k]; } return std::sqrt(e / double(F.size())) / (t / double(F.size())); };
        std::printf("the start, the room's colors captured over the first 18 updates, against the captured-first room after 800 updates:\n");
        const auto P = [&](const char* pName, const std::vector<double>& m, const std::vector<double>& F)
        { std::printf("  %-15s peak after the captures %.2fx, within 5%% from update %3d (of 200); at 200 its mean %.3fx, off by %.1f%% (rms)\n", pName, Peak(m), Within(m, 0.05), m.back() / Ref, 100 * Rms(F)); };
        P("captured first", MA, FA); P("streamed", MB, FB); P("provisional", MC, FC);
        CHECK(Peak(MC) < 1.05, "provisional materials: the light overshoots %.2fx after the captures", Peak(MC));
        CHECK(Within(MC, 0.05) <= Within(MB, 0.05), "provisional materials: within 5%% from update %d, streamed from %d", Within(MC, 0.05), Within(MB, 0.05));
        CHECK(std::abs(MC.back() / Ref - 1) <= std::abs(MA.back() / Ref - 1) + 0.02 && Rms(FC) <= Rms(FA) + 0.02, "provisional materials: at 200 updates the mean %.3fx and %.1f%% off (rms), captured first %.3fx and %.1f%%", MC.back() / Ref, 100 * Rms(FC), MA.back() / Ref, 100 * Rms(FA));
    }
    std::printf("PASS\n");
    return 0;
}
