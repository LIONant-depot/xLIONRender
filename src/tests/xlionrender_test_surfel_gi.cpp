// Surfel GI on the CPU (features/gi/xlionrender_surfel_gi.h), no GPU:
//  - the BVHs (mesh + instances) return the same hits as a brute-force cast
//  - a Cornell box built from instances (the box rotated) converges to a path-traced reference
//  - a rotated instance equals the same geometry pre-transformed (transforms reach surfels, normals and reads)
//  - a city (400 buildings, sun + sky): cost per update, coverage, LOD levels, and a moved building dirtying only its block
#include "features/gi/xlionrender_surfel_gi.h"
#include <cstdio>
#include <random>

using namespace xlionrender::gi;

namespace
{
    int g_nErrors = 0;
    #define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); ++g_nErrors; } } while (0)

    // A quad (corner, two edges; normal U x V) split into cells of at most Split, appended to M
    void AddQuad(surfel_mesh& M, v3 C, v3 U, v3 V, float Split)
    {
        const int nu = std::max(1, int(std::ceil(U.Length() / Split))), nv = std::max(1, int(std::ceil(V.Length() / Split)));
        const v3  N  = U.Cross(V).NormalizeCopy();
        const int Base = int(M.m_Pos.size());
        for (int j = 0; j <= nv; ++j)
        for (int i = 0; i <= nu; ++i) { M.m_Pos.push_back(C + U * (float(i) / nu) + V * (float(j) / nv)); M.m_Nrm.push_back(N); }
        for (int j = 0; j < nv; ++j)
        for (int i = 0; i < nu; ++i)
        {
            const int a = Base + j * (nu + 1) + i, b = a + 1, c = a + nu + 1, d = c + 1;
            M.m_Tris.push_back({ a, b, d });
            M.m_Tris.push_back({ a, d, c });
        }
    }

    // A box of size S centered on the origin (outward faces)
    surfel_mesh Box(v3 S, float Split)
    {
        surfel_mesh M;
        const v3 Lo = S * -0.5f, X{ S.m_X, 0, 0 }, Y{ 0, S.m_Y, 0 }, Z{ 0, 0, S.m_Z };
        AddQuad(M, Lo,     X, Z, Split);         // -Y
        AddQuad(M, Lo + Y, Z, X, Split);         // +Y
        AddQuad(M, Lo,     Z, Y, Split);         // -X
        AddQuad(M, Lo + X, Y, Z, Split);         // +X
        AddQuad(M, Lo,     Y, X, Split);         // -Z
        AddQuad(M, Lo + Z, X, Y, Split);         // +Z
        return M;
    }

    // Path-traced indirect irradiance (the reference): sum over the path of throughput x direct, the sky on a miss
    v3 Reference(const world& W, const lights& L, v3 P, v3 N, int Paths, int Depth, rng& Rng)
    {
        v3 Sum{ 0, 0, 0 };
        for (int p = 0; p < Paths; ++p)
        {
            v3 O = P, Nn = N, Thr{ 1, 1, 1 };
            for (int d = 0; d < Depth; ++d)
            {
                const v3 D = world::CosineDir(Nn, Rng);
                const auto H = W.Cast(O + Nn * 1e-3f, D);
                if (H.m_Inst < 0) { Sum = Sum + Thr * L.m_Sky * pi_v; break; }
                const v3 Nh = W.HitNormal(H);
                if (Nh.Dot(D) > 0) break;
                O = W.HitPosition(H); Nn = Nh;
                Thr = Thr * W.Albedo(H.m_Inst);
                Sum = Sum + Thr * W.Direct(O, Nn, L);
            }
        }
        return Sum * (1.0f / float(Paths));
    }

    float Lum(v3 c) { return 0.2126f * c.m_X + 0.7152f * c.m_Y + 0.0722f * c.m_Z; }

    // The cache's indirect irradiance at a surface point (cast down its normal onto it)
    v3 CacheAt(const world& W, v3 P, v3 N)
    {
        const auto H = W.Cast(P + N * 0.05f, N * -1.0f, 0.1f);
        return H.m_Inst < 0 ? v3{ -1, -1, -1 } : W.Indirect(H);
    }

    struct cornell
    {
        std::vector<surfel_mesh>    m_Meshes;
        world                       m_World;
        lights                      m_Lights;
    };

    // The Cornell box of the GI lab (walls: one instance each), the box an instance rotated by Yaw (or pre-transformed)
    void MakeCornell(cornell& C, float Yaw, bool bPreTransform)
    {
        const v3 W{ 0.75f, 0.75f, 0.75f }, R{ 0.65f, 0.05f, 0.05f }, G{ 0.12f, 0.6f, 0.1f };
        const float Split = 0.125f;
        C.m_Meshes.resize(7);
        AddQuad(C.m_Meshes[0], { -1, -1, -1 }, { 0, 0, 2 }, { 2, 0, 0 }, Split);    // floor
        AddQuad(C.m_Meshes[1], { -1,  1, -1 }, { 2, 0, 0 }, { 0, 0, 2 }, Split);    // ceiling
        AddQuad(C.m_Meshes[2], { -1, -1,  1 }, { 0, 2, 0 }, { 2, 0, 0 }, Split);    // back
        AddQuad(C.m_Meshes[3], { -1, -1, -1 }, { 2, 0, 0 }, { 0, 2, 0 }, Split);    // front
        AddQuad(C.m_Meshes[4], { -1, -1, -1 }, { 0, 2, 0 }, { 0, 0, 2 }, Split);    // left (red)
        AddQuad(C.m_Meshes[5], {  1, -1, -1 }, { 0, 0, 2 }, { 0, 2, 0 }, Split);    // right (green)
        C.m_Meshes[6] = Box({ 0.5f, 0.8f, 0.6f }, Split);
        const xmath::fmat4 BoxL2W = xmath::fmat4::fromTranslation({ -0.35f, -0.6f, 0.1f }) * xmath::fmat4::fromRotationY(xmath::radian{ Yaw });
        if (bPreTransform)
            for (std::size_t v = 0; v < C.m_Meshes[6].m_Pos.size(); ++v)
            {
                C.m_Meshes[6].m_Pos[v] = BoxL2W * C.m_Meshes[6].m_Pos[v];
                C.m_Meshes[6].m_Nrm[v] = BoxL2W.TransformDirection(C.m_Meshes[6].m_Nrm[v]).NormalizeCopy();
            }
        for (auto& M : C.m_Meshes) M.Bake(0.3f, 0.75f);
        const v3 Albedo[] = { W, W, W, W, R, G, W };
        for (int m = 0; m < 6; ++m) C.m_World.Add(C.m_Meshes[m], xmath::fmat4::fromIdentity(), Albedo[m]);
        C.m_World.Add(C.m_Meshes[6], bPreTransform ? xmath::fmat4::fromIdentity() : BoxL2W, W);
        C.m_Lights.m_Points = { { { 0.2f, 0.4f, -0.3f }, { 1.5f, 1.5f, 1.5f }, 4.0f } };
    }

    // Converge: every instance every update (a large budget); 8 bounces of Jacobi (replace), then plain averaging
    void Converge(cornell& C, int Rays, int Updates)
    {
        C.m_World.m_Setup.m_RaysPerSurfel = Rays;
        C.m_World.m_Setup.m_MinBlend      = 1;
        for (int u = 0; u < 8; ++u) C.m_World.Update(C.m_Lights, { 0, 0, -0.9f }, 1 << 30, 100 + u);
        C.m_World.m_Setup.m_MinBlend      = 0;
        for (int u = 0; u < Updates; ++u) C.m_World.Update(C.m_Lights, { 0, 0, -0.9f }, 1 << 30, 200 + u);
    }
}

int main()
{
    // 1. The BVHs against brute force
    {
        cornell C;
        MakeCornell(C, 0.5f, false);
        std::mt19937 Rng(7);
        std::uniform_real_distribution<float> U(-0.95f, 0.95f);
        int nMismatch = 0;
        const xmath::fmat4 BoxL2W = xmath::fmat4::fromTranslation({ -0.35f, -0.6f, 0.1f }) * xmath::fmat4::fromRotationY(xmath::radian{ 0.5f });
        for (int r = 0; r < 2000; ++r)
        {
            const v3 O{ U(Rng), U(Rng), U(Rng) }, D = v3{ U(Rng), U(Rng), U(Rng) }.NormalizeCopy();
            float Best = 1e30f;
            for (int m = 0; m < 7; ++m)
            {
                const auto& M = C.m_Meshes[m];
                const xmath::fmat4 L2W = m == 6 ? BoxL2W : xmath::fmat4::fromIdentity();
                for (const auto& T : M.m_Tris)
                {
                    const v3 A = L2W * M.m_Pos[T[0]], E1 = L2W * M.m_Pos[T[1]] - A, E2 = L2W * M.m_Pos[T[2]] - A;
                    const v3 P = D.Cross(E2); const float det = E1.Dot(P);
                    if (std::abs(det) < 1e-12f) continue;
                    const v3 Tv = O - A; const float u = Tv.Dot(P) / det;
                    if (u < 0 || u > 1) continue;
                    const v3 Q = Tv.Cross(E1); const float v = D.Dot(Q) / det;
                    if (v < 0 || u + v > 1) continue;
                    const float t = E2.Dot(Q) / det;
                    if (t > 1e-5f) Best = std::min(Best, t);
                }
            }
            const auto H = C.m_World.Cast(O, D);
            if (std::abs((H.m_Inst < 0 ? 1e30f : H.m_T) - Best) > 1e-3f && nMismatch++ == 0) std::printf("first mismatch: bvh %d t %g, brute %g\n", H.m_Inst, H.m_T, Best);
        }
        std::printf("BVH: %d of 2000 rays differ from brute force\n", nMismatch);
        CHECK(nMismatch == 0, "the BVHs miss or misplace hits (%d)", nMismatch);
    }

    // 2. Cornell from instances vs a path-traced reference; 3. the rotated instance vs pre-transformed geometry
    std::vector<std::pair<v3, v3>> Pts;
    for (int j = 0; j < 4; ++j) for (int i = 0; i < 4; ++i)
    {
        const float a = -0.8f + 1.6f * (i + 0.37f) / 4, b = -0.8f + 1.6f * (j + 0.61f) / 4;
        if (!(a > -0.75f && a < 0.05f && b > -0.3f && b < 0.5f)) Pts.push_back({ { a, -1, b }, { 0, 1, 0 } });       // the floor, off the box
        Pts.push_back({ { a, b, 1 }, { 0, 0, -1 } });                                                                // the back wall
    }
    cornell Inst, Pre;
    MakeCornell(Inst, 0.5f, false);
    MakeCornell(Pre, 0.5f, true);
    Converge(Inst, 32, 40);
    Converge(Pre, 32, 40);
    {
        std::vector<v3> Ref(Pts.size());
        ParallelFor(int(Pts.size()), [&](int i) { rng Rng(1234 + i); Ref[i] = Reference(Inst.m_World, Inst.m_Lights, Pts[i].first, Pts[i].second, 2048, 6, Rng); });
        double RefMean = 0;
        for (const auto& r : Ref) RefMean += Lum(r) / Ref.size();
        double Err = 0, Max = 0, Diff = 0;
        for (std::size_t i = 0; i < Pts.size(); ++i)
        {
            const double e = std::abs(Lum(CacheAt(Inst.m_World, Pts[i].first, Pts[i].second)) - Lum(Ref[i])) / RefMean;
            Err += e / Pts.size(); Max = std::max(Max, e);
            Diff += std::abs(Lum(CacheAt(Inst.m_World, Pts[i].first, Pts[i].second)) - Lum(CacheAt(Pre.m_World, Pts[i].first, Pts[i].second))) / RefMean / Pts.size();
        }
        std::printf("Cornell from instances vs path traced: mean %.1f%%, max %.1f%%; rotated instance vs pre-transformed: %.1f%%\n", Err * 100, Max * 100, Diff * 100);
        CHECK(Err < 0.06 && Max < 0.2, "the instanced Cornell box is off the reference (%.1f%% / %.1f%%)", Err * 100, Max * 100);
        CHECK(Diff < 0.04, "a rotated instance differs from the same geometry pre-transformed (%.1f%%)", Diff * 100);
    }

    // 4. A city: 20 x 20 buildings (3 shared meshes) on a 220 m ground, sun + sky, the eye at street level
    {
        std::vector<surfel_mesh> Meshes(4);
        AddQuad(Meshes[0], { -110, 0, -110 }, { 0, 0, 220 }, { 220, 0, 0 }, 2.0f);
        Meshes[0].Bake(3.0f, 8.0f);
        const float Heights[3] = { 8, 16, 30 };
        for (int k = 0; k < 3; ++k) { Meshes[1 + k] = Box({ 8, Heights[k], 8 }, 1.0f); Meshes[1 + k].Bake(1.5f, 4.0f); }
        world W;
        W.Add(Meshes[0], xmath::fmat4::fromIdentity(), { 0.4f, 0.4f, 0.4f });
        int FirstBuilding = 1, Moved = -1;
        for (int z = 0; z < 20; ++z) for (int x = 0; x < 20; ++x)
        {
            const int k = (x * 7 + z * 13) % 3;
            const int Id = W.Add(Meshes[1 + k], xmath::fmat4::fromTranslation({ -95.0f + x * 10, Heights[k] * 0.5f, -95.0f + z * 10 }), { 0.6f, 0.55f, 0.5f });
            if (x == 10 && z == 10) Moved = Id;
        }
        lights L;
        L.m_SunDirection = v3{ 0.4f, -1, 0.3f }.NormalizeCopy();
        L.m_SunColor     = { 3, 2.9f, 2.7f };
        L.m_Sky          = { 0.25f, 0.35f, 0.5f };
        int nSurfels = 0;
        for (int i = 0; i < W.InstanceCount(); ++i) nSurfels += int(W.InstanceLight(i).size());
        const v3 Eye{ -100.0f + 10 * 10, 1.7f, -100.0f + 10 * 10 + 5 };
        constexpr int budget_v = 65536;
        double Ms = 0, First = 0;
        int Frames = 0, Fine = 0, Coarse = 0;
        for (; Frames < 40; ++Frames)
        {
            const auto St = W.Update(L, Eye, budget_v, 1000 + Frames);
            if (Frames == 0) First = St.m_Ms; else Ms += St.m_Ms;
            if (St.m_Dirty == 0) break;                     // every instance updated at least once
        }
        for (int i = 0; i < W.InstanceCount(); ++i) (W.Level(i) ? Coarse : Fine)++;
        int nActive = 0;
        for (int i = 0; i < W.InstanceCount(); ++i) nActive += int(W.InstanceLight(i).size());
        std::printf("city: %d instances, %d surfels at full detail, %d active (fine %d / parent level %d instances); %d ray budget: first update %.1f ms, then %.1f ms; every instance lit after %d updates; GPU buffer %d KB\n"
                   , W.InstanceCount(), nSurfels, nActive, Fine, Coarse, budget_v, First, Ms / std::max(Frames - 1, 1), Frames + 1, nActive * 16 / 1024);
        CHECK(Coarse > 0 && Fine > 0, "the city uses one level only (fine %d, coarse %d)", Fine, Coarse);

        // Steady state (nothing changes), then a building moves: its neighbors are dirty, not the city
        double Steady = 0, Trace = 0;
        for (int f = 0; f < 10; ++f) { const auto S = W.Update(L, Eye, budget_v, 2000 + f); Steady += S.m_Ms; Trace += S.m_TraceMs; }
        std::printf("city: %u threads; tracing %.1f of %.1f ms per update; rays per ms: %.0f\n", std::thread::hardware_concurrency(), Trace / 10, Steady / 10, budget_v / (Trace / 10));
        for (int Budget : { 4096, 16384 })
        {
            double Ms = 0;
            for (int f = 0; f < 10; ++f) Ms += W.Update(L, Eye, Budget, 2500 + f).m_Ms;
            std::printf("city: %d rays per update: %.2f ms\n", Budget, Ms / 10);
        }
        W.SetTransform(Moved, xmath::fmat4::fromTranslation({ 2.0f, 0, 1.0f }) * xmath::fmat4::fromTranslation({ 5, 4, 5 }));
        const auto St = W.Update(L, Eye, budget_v, 3000);
        std::printf("city steady: %.1f ms per update; a building moved: %d instances dirty (of %d), updated in %.1f ms\n", Steady / 10, St.m_Dirty, W.InstanceCount(), St.m_Ms);
        CHECK(St.m_Dirty > 1 && St.m_Dirty < 15, "a moved building dirtied %d instances", St.m_Dirty);
    }

    if (g_nErrors) return 1;
    std::printf("PASS\n");
    return 0;
}
