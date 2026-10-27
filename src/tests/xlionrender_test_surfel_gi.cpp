// Surfel GI on the CPU (features/gi/xlionrender_surfel_gi.h), no GPU:
//  - the BVHs (mesh + instances) return the same hits as a brute-force cast
//  - a Cornell box built from instances (the box rotated) converges to a path-traced reference
//  - a rotated instance equals the same geometry pre-transformed (transforms reach surfels, normals and reads)
//  - a city (400 buildings, sun + sky): cost per update, coverage, LOD levels, and a moved building dirtying only its block
//  - surfels saved by the GeomStatic compiler load back as baked, and a receive-only mesh is not in the rays' way
#include "features/gi/xlionrender_surfel_gi.h"
#include "tests/xlionrender_test_no_dialogs.h"
#include "features/gi/xlionrender_surfel_gi_chunks.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <set>
#include <span>

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

    // The cache's indirect irradiance at a surface point (cast down its normal onto it), as the solver reads it (unsupported: its policy)
    v3 CacheAt(const world& W, v3 P, v3 N)
    {
        const auto H = W.Cast(P + N * 0.05f, N * -1.0f, 0.1f);
        if (H.m_Inst < 0) return v3{ -1, -1, -1 };
        const auto Q = W.Indirect(H);
        return Q.m_bSupported ? Q.m_E : W.Unsupported(H);
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
        W.m_Setup.m_ProjScale = 400;                      // a wide view: the far streets (beyond ~66 m for 1.5 m surfels) take their parents
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
        {   // Dump (the GI baseline's DumpGI): every active surfel, one line each after the setup and the column names
            const auto Path = (std::filesystem::temp_directory_path() / "xlionrender_gi_dump_test.csv").string();
            const int  n    = W.Dump(Path.c_str());
            std::ifstream In(Path);
            int nLines = 0;
            for (std::string Line; std::getline(In, Line); ) nLines++;
            In.close();
            std::filesystem::remove(Path);
            CHECK(n == nActive && nLines == nActive + 2, "Dump wrote %d surfels in %d lines (%d active)", n, nLines, nActive);
        }

        // Steady state (nothing changes), then a building moves: its neighbors are dirty, not the city
        double Steady = 0, Trace = 0;
        for (int f = 0; f < 10; ++f) { const auto S = W.Update(L, Eye, budget_v, 2000 + f); Steady += S.m_Ms; Trace += S.m_TraceMs; }
        std::printf("city: %u threads; tracing %.1f of %.1f ms per update; rays per ms: %.0f\n", std::thread::hardware_concurrency(), Trace / 10, Steady / 10, budget_v / (Trace / 10));
        for (int Budget : { 4096, 16384 })
        {
            double Ms = 0;
            int MaxRays = 0;
            for (int f = 0; f < 10; ++f) { const auto S = W.Update(L, Eye, Budget, 2500 + f); Ms += S.m_Ms; MaxRays = std::max(MaxRays, S.m_Rays); }
            CHECK(MaxRays <= Budget, "the budget is not held: %d rays for a budget of %d (instances are updated in slices)", MaxRays, Budget);
            std::printf("city: %d rays per update: %.2f ms\n", Budget, Ms / 10);
        }
        W.SetTransform(Moved, xmath::fmat4::fromTranslation({ 2.0f, 0, 1.0f }) * xmath::fmat4::fromTranslation({ 5, 4, 5 }));
        const auto St = W.Update(L, Eye, budget_v, 3000);
        // (Phase 4: the moved one is dirty, its neighbors' surfels near it are affected, not their whole instances)
        std::printf("city steady: %.1f ms per update; a building moved: %d instances dirty, %d affected (of %d), updated in %.1f ms\n", Steady / 10, St.m_Dirty, St.m_nMarkedDirty, W.InstanceCount(), St.m_Ms);
        CHECK(St.m_Dirty == 1 && St.m_nMarkedDirty > 1 && St.m_nMarkedDirty < 30, "a moved building affected %d instances (%d dirty)", St.m_nMarkedDirty, St.m_Dirty);
    }

    // 5. Surfels saved by the GeomStatic compiler load back as baked (SurfelsFromXGeomStatic), and a receive-only mesh is not in the rays' way
    {
        struct saved_geom       // the GeomStatic resource's GI part (xgeom_static::geom), the fields SurfelsFromXGeomStatic reads
        {
            enum class mode : std::uint8_t { OFF, RECEIVE, RECEIVE_AND_BOUNCE };
            struct vec3 { float m_X, m_Y, m_Z; };
            struct surfel { vec3 m_Center; float m_Radius; vec3 m_Normal; std::uint32_t m_iParent; };
            std::vector<surfel>         m_Surfels;
            std::vector<std::uint16_t>  m_Vert;
            std::uint32_t               m_nFine = 0;
            mode                        m_GIMode = mode::RECEIVE_AND_BOUNCE;
            std::span<const surfel>        getGISurfels()       const { return { m_Surfels.data(), m_nFine }; }
            std::span<const surfel>        getGIParents()       const { return { m_Surfels.data() + m_nFine, m_Surfels.size() - m_nFine }; }
            std::span<const std::uint16_t> getGIVertexSurfels() const { return m_Vert; }
            std::vector<std::uint32_t>  m_Derived;
            std::span<const std::uint32_t> getGIDerived()       const { return m_Derived; }
        };
        surfel_mesh Baked = Box({ 2, 2, 2 }, 0.25f);
        Baked.Bake(0.3f, 0.9f);
        saved_geom G;
        G.m_nFine = std::uint32_t(Baked.m_Levels[0].size());
        for (int L = 0; L < 2; ++L)
            for (std::size_t s = 0; s < Baked.m_Levels[L].size(); ++s)
            {
                const auto& S = Baked.m_Levels[L][s];
                G.m_Surfels.push_back({ { S.m_Center.m_X, S.m_Center.m_Y, S.m_Center.m_Z }, S.m_Radius, { S.m_Normal.m_X, S.m_Normal.m_Y, S.m_Normal.m_Z }, L ? 0u : std::uint32_t(Baked.m_ParentOf[s]) });
            }
        for (const auto& V : Baked.m_VertSurfels) for (int k : V) G.m_Vert.push_back(k < 0 ? 0xFFFF : std::uint16_t(k));

        surfel_mesh Loaded = Box({ 2, 2, 2 }, 0.25f);
        CHECK(SurfelsFromXGeomStatic(Loaded, G), "saved surfels did not load");
        bool bSame = Loaded.m_Levels[0].size() == Baked.m_Levels[0].size() && Loaded.m_Levels[1].size() == Baked.m_Levels[1].size()
                  && Loaded.m_ParentOf == Baked.m_ParentOf && Loaded.m_VertSurfels == Baked.m_VertSurfels && !Loaded.m_Bvh.m_Nodes.empty();
        for (std::size_t s = 0; bSame && s < Baked.m_Levels[0].size(); ++s)
            bSame = (Loaded.m_Levels[0][s].m_Center - Baked.m_Levels[0][s].m_Center).Length() < 1e-6f && Loaded.m_Levels[0][s].m_Radius == Baked.m_Levels[0][s].m_Radius;
        CHECK(bSame, "the loaded surfels differ from the baked ones");
        std::printf("saved surfels: %zu fine, %zu parents load back as baked\n", Loaded.m_Levels[0].size(), Loaded.m_Levels[1].size());

        // The per-pixel read grid: the cell of any point on the surface lists every surfel that reaches it (within 2 radii, a compatible normal)
        Loaded.BuildReadGrid();
        rng R(99);
        int nMissing = 0, nPoints = 0, MaxList = 0;
        for (const auto& T : Loaded.m_Tris)
            for (int k = 0; k < 4; ++k, ++nPoints)
            {
                float u = R.Next(), v = R.Next();
                if (u + v > 1) { u = 1 - u; v = 1 - v; }
                const v3 P = Loaded.m_Pos[T[0]] + (Loaded.m_Pos[T[1]] - Loaded.m_Pos[T[0]]) * u + (Loaded.m_Pos[T[2]] - Loaded.m_Pos[T[0]]) * v;
                const auto List = Loaded.ReadCellList(P, (Loaded.m_Pos[T[1]] - Loaded.m_Pos[T[0]]).Cross(Loaded.m_Pos[T[2]] - Loaded.m_Pos[T[0]]).NormalizeCopy());
                MaxList = std::max(MaxList, int(List.size()));
                for (int s = 0; s < int(Loaded.m_Levels[0].size()); ++s)
                    if ((P - Loaded.m_Levels[0][s].m_Center).Length() < 2 * Loaded.m_Levels[0][s].m_Radius && Loaded.m_Levels[0][s].m_Normal.Dot((Loaded.m_Pos[T[1]] - Loaded.m_Pos[T[0]]).Cross(Loaded.m_Pos[T[2]] - Loaded.m_Pos[T[0]]).NormalizeCopy()) >= 0.5f && std::find_if(List.begin(), List.end(), [&](std::uint32_t e) { return surfel_mesh::ReadEntrySurfel(e) == std::uint32_t(s); }) == List.end()) ++nMissing;
            }
        std::printf("read grid: %zu cells, %zu list entries, at most %d surfels per cell; %d of the surfels reaching %d surface points not listed\n"
                   , std::count_if(Loaded.m_ReadTable.begin(), Loaded.m_ReadTable.end(), [](const auto& E) { return E[1] != 0xFFFFFFFFu; }), Loaded.m_ReadList.size(), MaxList, nMissing, nPoints);
        CHECK(nMissing == 0, "the read grid misses %d surfels that reach a surface point", nMissing);

        // The compiler's packed load-time structures (PackDerived) load back as the renderer would have built them; a block made for
        // other surfels, or cut short, is refused (they are built instead)
        {
            Baked.BuildReadGrid();
            G.m_Derived = Baked.PackDerived();
            surfel_mesh Fast = Box({ 2, 2, 2 }, 0.25f);
            CHECK(SurfelsFromXGeomStatic(Fast, G), "the surfels with their packed structures did not load");
            const auto SameNb = [](const surfel_mesh& A, const surfel_mesh& B)
            {
                for (int Lv = 0; Lv < 2; ++Lv)
                    if (A.m_Neighbors[Lv].m_First != B.m_Neighbors[Lv].m_First || A.m_Neighbors[Lv].m_Index != B.m_Neighbors[Lv].m_Index || A.m_Neighbors[Lv].m_Weight != B.m_Neighbors[Lv].m_Weight) return false;
                return A.m_OriginFirst == B.m_OriginFirst && A.m_OriginVert == B.m_OriginVert;
            };
            const bool bGrid = Fast.m_ReadTable == Loaded.m_ReadTable && Fast.m_ReadList == Loaded.m_ReadList && Fast.m_ReadCell == Loaded.m_ReadCell && Fast.m_ReadMaxProbe == Loaded.m_ReadMaxProbe;
            std::printf("packed load-time structures: %zu words; read grid %s, neighbors and origins %s as built at load\n", G.m_Derived.size(), bGrid ? "the same" : "DIFFERENT", SameNb(Fast, Loaded) ? "the same" : "DIFFERENT");
            CHECK(bGrid && SameNb(Fast, Loaded), "the packed structures load differently from the ones built at load");
            surfel_mesh Other = Box({ 2, 2, 2 }, 0.25f);
            Other.m_Levels[0].resize(3);
            CHECK(!Other.UnpackDerived(G.m_Derived), "a block made for other surfels was taken");
            auto Short = G.m_Derived; Short.pop_back();
            surfel_mesh Cut = Box({ 2, 2, 2 }, 0.25f);
            Cut.m_Levels[0] = Baked.m_Levels[0];
            CHECK(!Cut.UnpackDerived(Short), "a block cut short was taken");
            G.m_Derived.clear();
        }

        saved_geom Empty;
        surfel_mesh None = Box({ 2, 2, 2 }, 0.25f);
        CHECK(!SurfelsFromXGeomStatic(None, Empty), "a geometry without surfels loaded some");

        G.m_GIMode = saved_geom::mode::RECEIVE;
        surfel_mesh Receiver = Box({ 2, 2, 2 }, 0.25f);
        CHECK(SurfelsFromXGeomStatic(Receiver, G) && !Receiver.m_bBounce, "RECEIVE did not make the mesh receive-only");
        world W;
        W.m_Setup.m_ProjScale = 1000;                     // the level rule: 500 m off takes the parents
        const int iBounce  = W.Add(Loaded,   xmath::fmat4::fromTranslation({ 0, 0, 5 }), { 0.5f, 0.5f, 0.5f });
        const int iReceive = W.Add(Receiver, xmath::fmat4::fromTranslation({ 0, 0, -5 }), { 0.5f, 0.5f, 0.5f });
        CHECK(W.Cast({ 0, 0, 0 }, { 0, 0, 1 }).m_Inst == iBounce, "a ray misses the bouncing mesh");
        CHECK(W.Cast({ 0, 0, 0 }, { 0, 0, -1 }).m_Inst < 0, "a ray hit the receive-only mesh (instance %d)", iReceive);

        // Far and back (the parent level, then the fine one again): the surfels' light stays finite and in range
        lights L;
        L.m_SunDirection = v3{ 0.3f, -1, 0.2f }.NormalizeCopy(); L.m_SunColor = { 3, 3, 3 }; L.m_Sky = { 0.2f, 0.3f, 0.5f };
        const auto Bad = [&](int Id)
        {
            int n = 0;
            const auto& E = W.InstanceLight(Id); const auto& V = W.InstanceValid(Id);
            for (std::size_t s = 0; s < E.size(); ++s) n += !(std::isfinite(E[s].m_X) && std::isfinite(E[s].m_Y) && std::isfinite(E[s].m_Z) && std::isfinite(V[s]) && V[s] >= 0 && V[s] <= 1.0001f && E[s].m_X >= 0);
            return n;
        };
        for (int f = 0; f < 8; ++f) W.Update(L, { 0, 0, 0 }, 1 << 20, 4000 + f);
        const std::size_t nNear = W.InstanceLight(iBounce).size();
        for (int f = 0; f < 8; ++f) W.Update(L, { 0, 0, 500 }, 1 << 20, 4100 + f);
        const std::size_t nFar = W.InstanceLight(iBounce).size();
        const int BadFar = Bad(iBounce);
        for (int f = 0; f < 8; ++f) W.Update(L, { 0, 0, 0 }, 1 << 20, 4200 + f);
        std::printf("far and back: %zu surfels near, %zu far; %d bad far, %d bad back\n", nNear, nFar, BadFar, Bad(iBounce));
        CHECK(nFar < nNear && W.InstanceLight(iBounce).size() == nNear, "the levels did not switch (%zu, %zu)", nNear, nFar);
        CHECK(BadFar == 0 && Bad(iBounce) == 0, "the surfels' light went bad far (%d) or back (%d)", BadFar, Bad(iBounce));
    }

    // 6a. No leak through a wall: a floor running under a thin wall (one mesh, as an imported building): a vertex of the floor on one
    //     side takes no surfel of the other side, though they are within reach and face the same way (the pixels: 6b)
    {
        surfel_mesh Room;
        AddQuad(Room, { -2, 0, -2 }, { 0, 0, 4 }, { 4, 0, 0 }, 0.5f);                 // the floor, y = 0, up
        AddQuad(Room, { 0.1f, 0, -2 }, { 0, 1.5f, 0 }, { 0, 0, 4 }, 0.25f);           // the wall's -x face (x = 0.1: the wall is across a read cell)
        AddQuad(Room, { 0.2f, 0, -2 }, { 0, 0, 4 }, { 0, 1.5f, 0 }, 0.25f);           // its +x face
        AddQuad(Room, { 0.1f, 1.5f, -2 }, { 0.1f, 0, 0 }, { 0, 0, 4 }, 0.25f);        // its top
        Room.Bake(0.3f, 0.9f);
        Room.BuildReadGrid();
        const auto Side = [](float x) { return x < 0.1f ? -1 : x > 0.2f ? 1 : 0; };
        int nLeaks = 0, nReads = 0;
        for (std::size_t v = 0; v < Room.m_Pos.size(); ++v)
        {
            if (Room.m_Nrm[v].m_Y < 0.9f || Side(Room.m_Pos[v].m_X) == 0) continue;
            for (int s : Room.m_VertSurfels[v]) if (s >= 0) { ++nReads; nLeaks += Side(Room.m_Levels[0][s].m_Center.m_X) == -Side(Room.m_Pos[v].m_X); }
        }

        std::printf("a wall between: %d of %d reads cross it\n", nLeaks, nReads);
        CHECK(nReads > 0 && nLeaks == 0, "%d of %d reads of the floor cross the wall", nLeaks, nReads);

        // ... nor does the spatial filter (the neighbor lists: lifted off the floor, the wall still blocks)
        const auto& L0 = Room.m_Levels[0];
        const auto& Nb = Room.m_Neighbors[0];
        int nCross = 0;
        for (std::size_t s = 0; s < L0.size(); ++s)
            for (std::uint32_t k = Nb.m_First[s]; k < Nb.m_First[s + 1]; ++k)
                nCross += L0[s].m_Normal.m_Y > 0.9f && Side(L0[s].m_Center.m_X) * Side(L0[Nb.m_Index[k]].m_Center.m_X) < 0;
        CHECK(nCross == 0, "%d floor surfels are filtered with the other side of the wall", nCross);

        // The sparse-grid experiment (XLION_GI_SPARSEGRID): every triangle coarse (ratio 0) keeps every list; none (a huge ratio) empties
        // every cell (their pixels read the vertex surfels)
        {
            auto All = Room, None = Room;
            All.SparsifyReadGrid(0.0f); None.SparsifyReadGrid(1e6f);
            std::size_t nPatches = 0; for (const auto& E : None.m_ReadTable) if (E[1] != 0xFFFFFFFFu) nPatches += E[3];
            CHECK(All.m_ReadList.size() == Room.m_ReadList.size() && None.m_ReadList.empty() && nPatches == 0,
                  "the sparse grid: %zu -> %zu words at ratio 0, %zu words / %zu patches at a huge ratio", Room.m_ReadList.size(), All.m_ReadList.size(), None.m_ReadList.size(), nPatches);
        }
    }

    // 6o. The filter's neighbors are one surface layer (SameLayer), not merely in sight from half a radius off the surface: a 10 cm step
    //     (two layers) and a 15 cm curb between two floors (a barrier) keep the two sides apart, at the Police Station's surfel size
    {
        const auto Across = [](const surfel_mesh& M)
        {
            const auto& L0 = M.m_Levels[0]; const auto& Nb = M.m_Neighbors[0];
            int n = 0;
            for (std::size_t s = 0; s < L0.size(); ++s)
                for (std::uint32_t k = Nb.m_First[s]; k < Nb.m_First[s + 1]; ++k)
                {
                    const auto& A = L0[s]; const auto& B = L0[Nb.m_Index[k]];
                    n += A.m_Normal.m_Y > 0.9f && B.m_Normal.m_Y > 0.9f && (A.m_Center.m_X < -0.05f) != (B.m_Center.m_X < -0.05f) && std::abs(A.m_Center.m_X) > 0.05f && std::abs(B.m_Center.m_X) > 0.05f;
                }
            return n;
        };
        surfel_mesh Step;
        AddQuad(Step, { -2, 0, -2 }, { 0, 0, 4 }, { 2, 0, 0 }, 0.5f);                  // the lower floor
        AddQuad(Step, { 0, 0.1f, -2 }, { 0, 0, 4 }, { 2, 0, 0 }, 0.5f);               // the upper one, 10 cm up
        AddQuad(Step, { 0, 0, -2 }, { 0, 0.1f, 0 }, { 0, 0, 4 }, 0.5f);                // the riser, facing -x
        Step.Bake(0.5f, 1.5f);
        surfel_mesh Curb;
        AddQuad(Curb, { -2, 0, -2 }, { 0, 0, 4 }, { 4, 0, 0 }, 0.5f);                  // one floor
        AddQuad(Curb, { -0.02f, 0, -2 }, { 0, 0.15f, 0 }, { 0, 0, 4 }, 0.5f);          // a curb across it: 15 cm high, 4 cm wide
        AddQuad(Curb, { 0.02f, 0, -2 }, { 0, 0, 4 }, { 0, 0.15f, 0 }, 0.5f);
        AddQuad(Curb, { -0.02f, 0.15f, -2 }, { 0, 0, 4 }, { 0.04f, 0, 0 }, 0.5f);
        Curb.Bake(0.5f, 1.5f);
        const int nStep = Across(Step), nCurb = Across(Curb);
        std::printf("filter neighbors across a 10 cm step: %d, across a 15 cm curb: %d\n", nStep, nCurb);
        CHECK(nStep == 0 && nCurb == 0, "the filter blends surfaces apart: %d across the step, %d across the curb", nStep, nCurb);
    }

    // 6n. Closed blinds (slats stacked tighter than a surfel, each tilted): the slats block the line along the surface between their
    //     surfels, but lifted off it they see each other - the spatial filter blends them (else each keeps its own slat's shadowing: blotches)
    {
        surfel_mesh Blinds;
        for (int i = 0; i < 30; ++i)
            AddQuad(Blinds, { -0.5f, 0.035f * i, 0 }, { 1, 0, 0 }, { 0, 0.04f * 0.966f, -0.04f * 0.259f }, 0.05f);    // 4 cm slats, 3.5 cm apart, 15 degrees
        Blinds.Bake(0.1f, 0.3f);
        const auto& Nb = Blinds.m_Neighbors[0];
        int nAlone = 0;
        for (std::size_t s = 0; s < Blinds.m_Levels[0].size(); ++s) nAlone += Nb.m_First[s + 1] == Nb.m_First[s];
        std::printf("closed blinds: %d of %zu surfels have no neighbor to filter with\n", nAlone, Blinds.m_Levels[0].size());
        CHECK(nAlone * 10 < int(Blinds.m_Levels[0].size()), "the blinds' surfels are not filtered together (%d alone)", nAlone);

        // ... and lit by the sky, their raw light (no filter) varies less from surfel to surfel when each surfel's rays start across its
        // patch (several slats) than from its center (one spot of one slat: in its shadow or not)
        const auto Spread = [&](bool bSpread)
        {
            world W;
            W.m_Setup.m_bSpreadOrigins = bSpread;
            const int Id = W.Add(Blinds, xmath::fmat4::fromIdentity(), { 0.7f, 0.7f, 0.7f });
            lights L; L.m_Sky = { 1, 1, 1 };
            for (int f = 0; f < 200; ++f) W.Update(L, { 0, 0.5f, 1 }, 1 << 30, 17000 + f);
            double m = 0, m2 = 0; int n = 0;
            for (std::size_t s = 0; s < Blinds.m_Levels[0].size(); ++s)
            {
                const auto& C = Blinds.m_Levels[0][s].m_Center;
                if (std::abs(C.m_X) > 0.3f || C.m_Y < 0.2f || C.m_Y > 0.8f) continue;     // away from the edges
                const double e = W.InstanceLight(Id)[s].m_X; m += e; m2 += e * e; ++n;
            }
            m /= std::max(n, 1);
            return std::sqrt(std::max(0.0, m2 / std::max(n, 1) - m * m)) / std::max(m, 1e-9);
        };
        const double SCenter = Spread(false), SPatch = Spread(true);
        std::printf("closed blinds under the sky: surfel to surfel spread %.1f%% from their centers, %.1f%% from across their patches\n", 100 * SCenter, 100 * SPatch);
        CHECK(SPatch < 0.85 * SCenter, "rays from across the patch do not even the blinds out (%.1f%% vs %.1f%%)", 100 * SPatch, 100 * SCenter);
    }

    // 6b. A wall that is another mesh (the bake cannot see it): the floor's surfels on one side learn from their rays that the wall is
    //     in the way (the depth moments), and the reads fade them for points on the other side; points on their own side keep them
    {
        surfel_mesh Floor, Wall = Box({ 0.1f, 1.5f, 4 }, 0.25f);
        AddQuad(Floor, { -2, 0, -2 }, { 0, 0, 4 }, { 4, 0, 0 }, 0.5f);
        Floor.Bake(0.3f, 0.9f); Wall.Bake(0.3f, 0.9f);
        world W;
        const int iFloor = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        W.Add(Wall, xmath::fmat4::fromTranslation({ 0.15f, 0.75f, 0 }), { 0.5f, 0.5f, 0.5f });       // x 0.1 .. 0.2, on the floor
        lights L; L.m_Sky = { 1, 1, 1 };
        for (int f = 0; f < 300; ++f) W.Update(L, { 0, 1, 0 }, 1 << 30, 9000 + f);
        double Across = 0, Same = 0; int nAcross = 0, nSame = 0;
        const auto& S0 = Floor.m_Levels[0];
        for (int s = 0; s < int(S0.size()); ++s)
        {
            const float x = S0[s].m_Center.m_X;
            if (x > -0.05f || x < -0.5f) continue;                                         // the floor's surfels just left of the wall
            for (float dx : { 0.35f, 0.5f })                                               // a point that far to the right: across or not
            {
                const v3 P{ x + dx, 0, S0[s].m_Center.m_Z };
                if ((P - S0[s].m_Center).Length() >= 2 * S0[s].m_Radius) continue;
                const float w = W.Visibility(iFloor, s, P);
                if (P.m_X > 0.2f) { Across += w; ++nAcross; } else { Same += w; ++nSame; }
            }
            const v3 Q{ x - 0.3f, 0, S0[s].m_Center.m_Z };                                 // to the left: open floor
            Same += W.Visibility(iFloor, s, Q); ++nSame;
        }
        std::printf("a wall of another mesh: mean visibility across it %.3f (%d), on the surfels' own side %.3f (%d)\n", Across / std::max(nAcross, 1), nAcross, Same / std::max(nSame, 1), nSame);
        CHECK(nAcross > 0 && Across / nAcross < 0.1, "the moments do not stop the read across the wall (%.3f)", Across / std::max(nAcross, 1));
        CHECK(nSame > 0 && Same / nSame > 0.7, "the moments fade the surfels on their own side too (%.3f)", Same / std::max(nSame, 1));
    }

    // 6c. Coverage: every point of the surface finds a surfel to read in its cell (the debug view's black: none), at a wall's foot and on a
    //     thin plank too (both used to lose their surfels to a gap filter), on big triangles as authored (2 per face), and a hair off the
    //     surface (a floor at y = 0 lies on a cell boundary: the GPU's interpolated position falls on either side), and in a busy cell (24
    //     small faces in as many directions: each finds its surfels, not only the first few directions the cell saw)
    {
        surfel_mesh Room;
        AddQuad(Room, { -3, 0, -3 }, { 0, 0, 6 }, { 6, 0, 0 }, 6.0f);                 // the floor: 2 triangles
        AddQuad(Room, { 0, 0, -3 }, { 0, 2, 0 }, { 0, 0, 6 }, 6.0f);                  // a wall standing on it (x = 0, facing -x)
        const auto Plank = [&](v3 Lo, v3 S)                                          // a 2 cm plank hanging in the room
        {
            const v3 X{ S.m_X, 0, 0 }, Y{ 0, S.m_Y, 0 }, Z{ 0, 0, S.m_Z };
            AddQuad(Room, Lo, X, Z, 6); AddQuad(Room, Lo + Y, Z, X, 6); AddQuad(Room, Lo, Z, Y, 6); AddQuad(Room, Lo + X, Y, Z, 6); AddQuad(Room, Lo, Y, X, 6); AddQuad(Room, Lo + Z, X, Y, 6);
        };
        Plank({ -2.0f, 1.5f, -0.6f }, { 1.2f, 0.02f, 0.3f });
        Plank({ -0.25f, 0.12f, -2.0f }, { 0.25f, 0.02f, 4.0f });                 // a ledge on the wall, low over the floor: floor, wall, its top, front and underside share cells
        for (int k = 0; k < 24; ++k)                                                 // a busy spot (a window's frames and sills: many small faces, as many directions, in one cell)
        {
            const float a = 0.2618f * k, e = 0.3f + 0.05f * float(k % 7);
            const v3 U{ std::cos(a) * 0.08f, 0, std::sin(a) * 0.08f }, V{ -std::sin(a) * std::cos(e) * 0.08f, std::sin(e) * 0.08f, std::cos(a) * std::cos(e) * 0.08f };
            AddQuad(Room, v3{ -1.0f, 0.6f, 1.0f } + U * 0.5f, U, V, 6);
        }
        Room.Bake(0.3f, 0.9f);
        Room.BuildReadGrid();
        rng R(7);
        int nPoints = 0, nUncovered = 0;
        for (const auto& T : Room.m_Tris)
        {
            const v3 A = Room.m_Pos[T[0]], B = Room.m_Pos[T[1]], C = Room.m_Pos[T[2]], N = (B - A).Cross(C - A).NormalizeCopy();
            for (int k = 0; k < 400; ++k, ++nPoints)
            {
                float u = R.Next(), v = R.Next();
                if (u + v > 1) { u = 1 - u; v = 1 - v; }
                const v3 P = A + (B - A) * u + (C - A) * v + v3{ 1e-6f, -1e-6f, 1e-6f } * float((k % 3) - 1);     // the GPU's interpolation is a hair off: a floor at y = 0 lies on a cell boundary
                bool bCovered = false;
                for (auto e : Room.ReadCellList(P, N))
                {
                    const auto& S = Room.m_Levels[0][surfel_mesh::ReadEntrySurfel(e)];
                    bCovered = bCovered || (S.m_Normal.Dot(N) >= 0.5f && (P - S.m_Center).Length() < 2 * S.m_Radius);
                }
                nUncovered += !bCovered;
            }
        }
        std::printf("coverage: %d of %d surface points read no surfel\n", nUncovered, nPoints);
        for (const auto& E : Room.Coverage(4).m_Examples) std::printf("  hole %.3f %.3f %.3f N %.2f %.2f %.2f: %d listed, nearest compatible %.3f, any %.3f\n", E.m_P.m_X, E.m_P.m_Y, E.m_P.m_Z, E.m_N.m_X, E.m_N.m_Y, E.m_N.m_Z, E.m_nList, E.m_NearestCompatible, E.m_NearestAny);
        CHECK(nUncovered == 0, "%d of %d surface points read no surfel (black in the debug view)", nUncovered, nPoints);
        CHECK(Room.m_ReadUnsupported == 0, "%u of the room's patches see none of their surfels", Room.m_ReadUnsupported);

        // A patch lists only the surfels it sees: a small face sealed in a closed box lists its own surfel, none of the floor's or the
        // box top's outside (a cell corner with no surface near it used to count as in plain sight: they leaked in). Without its own
        // surfel it sees none: it lists none (unsupported: the read's policy), not all of them at full weight
        surfel_mesh Sealed;
        AddQuad(Sealed, { -1, 0, -1 }, { 0, 0, 2 }, { 2, 0, 0 }, 6);                                       // a floor facing up
        const v3 Lo{ -0.06f, 0.001f, -0.06f }, S{ 0.12f, 0.12f, 0.12f };                                    // a closed box on it (faces outward)
        const v3 X{ S.m_X, 0, 0 }, Y{ 0, S.m_Y, 0 }, Z{ 0, 0, S.m_Z };
        AddQuad(Sealed, Lo, X, Z, 2); AddQuad(Sealed, Lo + Y, Z, X, 2); AddQuad(Sealed, Lo, Z, Y, 2); AddQuad(Sealed, Lo + X, Y, Z, 2); AddQuad(Sealed, Lo, Y, X, 2); AddQuad(Sealed, Lo + Z, X, Y, 2);
        AddQuad(Sealed, { -0.02f, 0.06f, -0.02f }, { 0, 0, 0.04f }, { 0.04f, 0, 0 }, 1);                     // inside: a tiny face up (apart from the floor's and the top's planes)
        Sealed.Bake(0.3f, 0.9f);
        Sealed.BuildReadGrid();
        const auto Listed = [&] { int nOwn = 0, nOther = 0; for (auto e : Sealed.ReadCellList({ 0, 0.06f, 0 }, { 0, 1, 0 })) (std::abs(Sealed.m_Levels[0][surfel_mesh::ReadEntrySurfel(e)].m_Center.m_Y - 0.06f) < 0.01f ? nOwn : nOther)++; return std::pair{ nOwn, nOther }; };
        const auto [nOwn, nOther] = Listed();
        std::erase_if(Sealed.m_Levels[0], [](const surfel& S) { return std::abs(S.m_Center.m_Y - 0.06f) < 0.01f; });     // its own surfel gone
        Sealed.BuildReadGrid();
        const auto [nOwn2, nOther2] = Listed();
        std::printf("sealed face: lists %d own + %d outside surfels; without its own %d + %d, %u unsupported patches\n", nOwn, nOther, nOwn2, nOther2, Sealed.m_ReadUnsupported);
        CHECK(nOwn == 1 && nOther == 0, "the sealed face lists surfels it does not see (%d outside)", nOther);
        CHECK(nOwn2 + nOther2 == 0 && Sealed.m_ReadUnsupported > 0, "a face that sees no surfel still lists some (%d)", nOther2);

        // A door frame on a wall (the Police Station's black bar): the wall right beside the frame sees the wall's surfels, so it reads
        // them - the wall's points under the frame (never seen) do not decide what its cells see
        surfel_mesh Door;
        // (as the Police Station's: the frame's edge 4 cm short of a cell's, the wall under it first, then beside it in big triangles)
        AddQuad(Door, { 0, 0, -2 }, { 0, 3, 0 }, { 0, 0, 2.46f }, 6);                                       // the wall, facing +x: under the frame,
        AddQuad(Door, { 0, 0, 0.46f }, { 0, 3, 0 }, { 0, 0, 2 }, 1);                                        // then beside it
        const v3 FLo{ 0.001f, 0, -2 }, FS{ 0.06f, 3, 2.46f };                                               // the frame: 6 cm deep, over z -2..0.46
        const v3 FX{ FS.m_X, 0, 0 }, FY{ 0, FS.m_Y, 0 }, FZ{ 0, 0, FS.m_Z };
        AddQuad(Door, FLo, FX, FZ, 2); AddQuad(Door, FLo + FY, FZ, FX, 2); AddQuad(Door, FLo, FZ, FY, 2); AddQuad(Door, FLo + FX, FY, FZ, 4); AddQuad(Door, FLo, FY, FX, 4); AddQuad(Door, FLo + FZ, FX, FY, 4);
        Door.Bake(0.5f, 1.5f);
        Door.BuildReadGrid();
        int nBeside = 0, nRead = 0;
        for (float y = 0.3f; y < 2.8f; y += 0.1f)
            for (float z : { 0.47f, 0.48f, 0.49f })
            {
                const v3 P{ 0, y, z };
                ++nBeside;
                float Wsum = 0;
                for (auto e : Door.ReadCellList(P, { 1, 0, 0 })) if (Door.m_Levels[0][surfel_mesh::ReadEntrySurfel(e)].m_Center.m_X < 0.01f) Wsum += Door.ReadEntryWeight(e, P, { 1, 0, 0 }, { 1, 0, 0 });
                nRead += Wsum > 0.05f;
            }
        std::printf("beside a door frame: %d of %d wall points read the wall's surfels\n", nRead, nBeside);

        // Defects (PatchDefects): a strip of floor between a wall and a beam lying on the floor, its surfels taken away (a strip no surfel
        // serves): BuildReadGrid lists it; baked again with it, it gets a patch surfel (after the others: m_FirstPatch) and reads it
        surfel_mesh Strip;
        AddQuad(Strip, { -2, 0, -2 }, { 0, 0, 4 }, { 2, 0, 0 }, 0.5f);                                     // the floor, up to the wall
        AddQuad(Strip, { 0, 0, -2 }, { 0, 0, 4 }, { 0, 2, 0 }, 0.5f);                                      // a wall at x = 0, facing -x
        const v3 BLo{ -0.5f, 0.001f, -2 }, BS{ 0.4f, 0.3f, 4 };                                            // a beam on the floor, 10 cm from the wall
        const v3 BX{ BS.m_X, 0, 0 }, BY{ 0, BS.m_Y, 0 }, BZ{ 0, 0, BS.m_Z };
        AddQuad(Strip, BLo + BY, BZ, BX, 1); AddQuad(Strip, BLo, BZ, BY, 1); AddQuad(Strip, BLo + BX, BY, BZ, 1);
        Strip.Bake(0.5f, 1.5f);
        std::erase_if(Strip.m_Levels[0], [](const surfel& S) { return S.m_Normal.m_Y > 0.9f && S.m_Center.m_X > -0.1f && S.m_Center.m_X < 0 && S.m_Center.m_Y < 0.01f; });
        Strip.BuildReadGrid();
        const std::size_t nEmpty = Strip.m_EmptyParts.size();
        const auto Forced = Strip.m_EmptyParts;
        Strip.Bake(0.5f, 1.5f, Forced);
        Strip.BuildReadGrid();
        const auto StripList = Strip.ReadCellList({ -0.05f, 0, 0.3f }, { 0, 1, 0 });
        bool bReadsPatch = false;
        for (auto e : StripList) bReadsPatch = bReadsPatch || (e & 0xFFFFu) >= Strip.m_FirstPatch;
        std::printf("defects: %zu empty pieces; patched with %d surfels; the strip reads a patch: %d\n", nEmpty, Strip.m_FirstPatch == 0xFFFFFFFFu ? 0 : int(Strip.m_Levels[0].size() - Strip.m_FirstPatch), int(bReadsPatch));
        CHECK(nEmpty > 0 && Strip.m_FirstPatch != 0xFFFFFFFFu && bReadsPatch, "a strip no surfel serves is not patched");

        // Ivy pressed on a wall: the wall's surfels under its leaves are buried (they see the leaves' backs: validity 0, no light). In a
        // list's nearest slots they left the wall around them reading next to nothing - the Bistro's dark and bright cell blocks behind its
        // ivy, 2026-10-24. Listed after the others: in crowded lists, few (the ones a list needs to fill)
        surfel_mesh Ivy;
        AddQuad(Ivy, { 0, 0, -2 }, { 0, 0, 4 }, { 0, 2, 0 }, 0.25f);                                      // the wall at x = 0, facing -x
        for (int i = 0; i < 19; ++i) for (int j = 0; j < 9; ++j)
            if ((i + j) % 4 != 0) AddQuad(Ivy, { -0.01f, 0.2f * float(j) + 0.02f, -1.9f + 0.2f * float(i) }, { 0, 0, 0.18f }, { 0, 0.18f, 0 }, 1);     // leaves, 1 cm off it
        Ivy.Bake(0.15f, 0.45f);
        Ivy.m_ReadListCap = 8;                                                                              // (crowded: the Bistro's 32 were)
        Ivy.BuildReadGrid();
        int nBuried = 0, nListedBuried = 0, nEntries = 0;
        for (const auto& S : Ivy.m_Levels[0]) nBuried += Ivy.Occluded(S.m_Center + S.m_Normal * (0.02f * Ivy.m_ReadCell), S.m_Normal, 0.1f * Ivy.m_ReadCell);
        for (const auto& E : Ivy.m_ReadTable)
        {
            if (E[1] == 0xFFFFFFFFu) continue;
            for (std::uint32_t p = 0; p < E[3]; ++p)
            {
                const std::uint32_t F = Ivy.m_ReadList[E[2] + 4 * p + 2], n = Ivy.m_ReadList[E[2] + 4 * p + 3] & 0xFFFFu;
                if (n < Ivy.m_ReadListCap) continue;                                                       // (a list with room may name them)
                for (std::uint32_t k = 0; k < n; ++k)
                {
                    const auto& S = Ivy.m_Levels[0][surfel_mesh::ReadEntrySurfel(Ivy.m_ReadList[F + k])];
                    ++nEntries; nListedBuried += Ivy.Occluded(S.m_Center + S.m_Normal * (0.02f * Ivy.m_ReadCell), S.m_Normal, 0.1f * Ivy.m_ReadCell);
                }
            }
        }
        std::printf("ivy pressed on a wall: %d buried surfels, %d of %d list entries name one\n", nBuried, nListedBuried, nEntries);
        CHECK(nBuried > 20 && nEntries > 0 && nListedBuried * 20 < nEntries, "the full lists name %d buried surfels (%d buried): the wall's own slots taken", nListedBuried, nBuried);
        CHECK(nRead == nBeside, "the wall beside a door frame reads no surfel at %d of %d points", nBeside - nRead, nBeside);
    }

    // 6d. Stable and responsive (the estimator): once converged, the light holds still from one update to the next (no swimming), and
    //     when the light doubles it follows within a few dozen passes
    {
        cornell C;
        MakeCornell(C, 0.5f, false);
        auto& W = C.m_World;
        const auto Snapshot = [&]
        {
            std::vector<v3> All;
            for (int i = 0; i < W.InstanceCount(); ++i) for (const auto& e : W.InstanceLight(i)) All.push_back(e);
            return All;
        };
        const auto Mean = [](const std::vector<v3>& A) { double m = 0; for (const auto& e : A) m += e.m_X + e.m_Y + e.m_Z; return m / std::max<std::size_t>(A.size(), 1); };
        for (int f = 0; f < 300; ++f) W.Update(C.m_Lights, { 0, 0, -0.9f }, 1 << 30, 11000 + f);
        double Swim = 0; auto Prev = Snapshot();
        for (int f = 0; f < 20; ++f)
        {
            W.Update(C.m_Lights, { 0, 0, -0.9f }, 1 << 30, 11300 + f);
            const auto Now = Snapshot();
            double d = 0, m = 0;
            for (std::size_t k = 0; k < Now.size(); ++k) { d += (Now[k] - Prev[k]).Length(); m += Now[k].Length(); }
            Swim += d / std::max(m, 1e-9) / 20;
            Prev = Now;
        }
        const double Before = Mean(Prev);
        for (auto& P : C.m_Lights.m_Points) P.m_Color = P.m_Color * 2;
        int Passes = 0;
        for (; Passes < 200 && Mean(Snapshot()) < 1.8 * Before; ++Passes) W.Update(C.m_Lights, { 0, 0, -0.9f }, 1 << 30, 11400 + Passes);
        std::printf("stability: %.2f%% change per update once converged; the light doubled: 90%% of it after %d updates\n", 100 * Swim, Passes);
        CHECK(Swim < 0.01, "the light swims: %.2f%% per update once converged", 100 * Swim);
        CHECK(Passes < 40, "the light follows a change too slowly (%d updates)", Passes);
    }

    // 6e. The read grid's hash table on a big regular layout (a 60 m floor, a row of pillars along a wall: what a building is): its
    //     probes stay short (the GPU looks a cell up within the longest probe; a lattice used to cluster an xor-of-primes hash)
    {
        surfel_mesh Hall;
        AddQuad(Hall, { -30, 0, -30 }, { 0, 0, 60 }, { 60, 0, 0 }, 60.0f);
        AddQuad(Hall, { 0, 0, -30 }, { 0, 4, 0 }, { 0, 0, 60 }, 60.0f);
        for (int k = 0; k < 30; ++k)
        {
            const v3 Lo{ -0.6f, 0, -29.0f + 2.0f * k };
            const v3 X{ 0.6f, 0, 0 }, Y{ 0, 4, 0 }, Z{ 0, 0, 0.6f };
            AddQuad(Hall, Lo, Z, Y, 4); AddQuad(Hall, Lo + Z, X, Y, 4); AddQuad(Hall, Lo, Y, X, 4);
        }
        Hall.Bake(0.5f, 1.5f);
        Hall.BuildReadGrid();
        std::printf("read grid of a 60 m hall: %zu table entries, the longest probe %u\n", Hall.m_ReadTable.size(), Hall.m_ReadMaxProbe);
        CHECK(Hall.m_ReadMaxProbe <= 32, "the read grid's hash clusters: a probe of %u", Hall.m_ReadMaxProbe);
    }

    // 6f. The spatial filter: a floor under a ceiling, a uniform sky (in the middle every surfel's true light is about the same): what the reads show spreads less across
    //     the surfels than their raw light (the independent noise of neighbors averages out), with the same mean
    {
        surfel_mesh Floor;                                                           // a floor under a ceiling 0.5 m above: only grazing rays escape (noisy)
        AddQuad(Floor, { -6, 0, -6 }, { 0, 0, 12 }, { 12, 0, 0 }, 0.5f);
        AddQuad(Floor, { -6, 0.5f, -6 }, { 12, 0, 0 }, { 0, 0, 12 }, 0.5f);
        Floor.Bake(0.3f, 0.9f);
        world W;
        const int Id = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        lights L; L.m_Sky = { 1, 1, 1 };
        for (int f = 0; f < 8; ++f) W.Update(L, { 0, 1, 0 }, 1 << 30, 12000 + f);       // few passes: still noisy
        const auto Spread = [&](const std::vector<v3>& E, double& Mean)             // the floor's middle (the true light nearly uniform there)
        {
            double m = 0, m2 = 0; int n = 0;
            for (std::size_t s = 0; s < E.size(); ++s)
            {
                const auto& S = Floor.m_Levels[0][s];
                if (S.m_Normal.m_Y < 0.9f || std::abs(S.m_Center.m_X) > 2 || std::abs(S.m_Center.m_Z) > 2) continue;
                m += E[s].m_X; m2 += double(E[s].m_X) * E[s].m_X; ++n;
            }
            m /= std::max(n, 1); Mean = m;
            return std::sqrt(std::max(0.0, m2 / std::max(n, 1) - m * m)) / std::max(m, 1e-9);
        };
        double MRaw, MFil;
        const double Raw = Spread(W.InstanceLight(Id), MRaw), Fil = Spread(W.FilteredLight(Id), MFil);
        std::printf("spatial filter: the spread across surfels %.1f%% raw, %.1f%% shown; mean %.3f -> %.3f\n", 100 * Raw, 100 * Fil, MRaw, MFil);
        CHECK(Fil < 0.6 * Raw, "the filter does not smooth: %.1f%% -> %.1f%%", 100 * Raw, 100 * Fil);
        CHECK(std::abs(MFil - MRaw) < 0.1 * MRaw, "the filter moves the mean: %.3f -> %.3f", MRaw, MFil);     // 10%: a few passes of grazing rays, a rare-event mean
    }

    // 6g. The HL2 basis: a floor between a red wall (+x) and a green one (-x), a white sky: a normal leaning toward the red wall reads
    //     redder than one leaning toward the green (the bounce comes from its side), and the floor's own normal reads their mean
    {
        surfel_mesh Floor, Red, Green;
        AddQuad(Floor, { -1, 0, -2 }, { 0, 0, 4 }, { 2, 0, 0 }, 0.25f);
        AddQuad(Red,   {  1, 0, -2 }, { 0, 0, 4 }, { 0, 2, 0 }, 0.25f);       // x = 1, facing -x
        AddQuad(Green, { -1, 0, -2 }, { 0, 2, 0 }, { 0, 0, 4 }, 0.25f);       // x = -1, facing +x
        Floor.Bake(0.3f, 0.9f); Red.Bake(0.3f, 0.9f); Green.Bake(0.3f, 0.9f);
        world W;
        const int iFloor = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        W.Add(Red,   xmath::fmat4::fromIdentity(), { 0.9f, 0.05f, 0.05f });
        W.Add(Green, xmath::fmat4::fromIdentity(), { 0.05f, 0.9f, 0.05f });
        lights L; L.m_Sky = { 0.05f, 0.05f, 0.05f };                            // dim: the walls' bounce shows
        L.m_SunDirection = v3{ 0, -1, 0.2f }.NormalizeCopy(); L.m_SunColor = { 3, 3, 3 };
        for (int f = 0; f < 200; ++f) W.Update(L, { 0, 1, 0 }, 1 << 30, 13000 + f);
        int s0 = 0; float Best = 1e30f;
        for (int s = 0; s < int(Floor.m_Levels[0].size()); ++s) { const float d = Floor.m_Levels[0][s].m_Center.Length(); if (d < Best) { Best = d; s0 = s; } }
        const v3 ToRed   = W.BasisRead(iFloor, s0, v3{ 0.8f, 0.6f, 0 }.NormalizeCopy());
        const v3 ToGreen = W.BasisRead(iFloor, s0, v3{ -0.8f, 0.6f, 0 }.NormalizeCopy());
        const v3 Up = W.BasisRead(iFloor, s0, v3{ 0, 1, 0 }), Mean = W.FilteredLight(iFloor)[s0];
        const float RgRed = ToRed.m_X / std::max(ToRed.m_Y, 1e-6f), RgGreen = ToGreen.m_X / std::max(ToGreen.m_Y, 1e-6f);
        std::printf("HL2 basis: red/green leaning to the red wall %.2f, to the green wall %.2f; the floor's normal %.3f (the mean %.3f)\n", RgRed, RgGreen, Up.m_X + Up.m_Y + Up.m_Z, Mean.m_X + Mean.m_Y + Mean.m_Z);
        CHECK(RgRed > 1.3f * RgGreen, "the basis gives no direction: red/green %.2f toward red, %.2f toward green", RgRed, RgGreen);
        CHECK(std::abs((Up - Mean).Length()) < 0.02f * Mean.Length(), "the surfel's own normal does not read the mean");
    }

    // 6h. Where each surfel samples its material (the bake): two quads of different slots, UVs a known function of the position (u = x/4 +
    //     0.5, v = z/4 + 0.5): every surfel has its quad's slot, the UV of its center, and a footprint of its radius at the UV density (1/4)
    {
        surfel_mesh Two;
        AddQuad(Two, { -2, 0, -2 }, { 0, 0, 4 }, { 2, 0, 0 }, 0.5f);                 // x in [-2, 0]: slot 0
        const std::size_t nFirst = Two.m_Tris.size();
        AddQuad(Two, {  0, 0, -2 }, { 0, 0, 4 }, { 2, 0, 0 }, 0.5f);                 // x in [0, 2]: slot 1
        for (const auto& P : Two.m_Pos) Two.m_UV.push_back({ P.m_X / 4 + 0.5f, P.m_Z / 4 + 0.5f });
        for (std::size_t t = 0; t < Two.m_Tris.size(); ++t) Two.m_TriMaterial.push_back(t < nFirst ? 0 : 1);
        Two.Bake(0.3f, 0.9f);
        int nBad = 0;
        for (std::size_t s = 0; s < Two.m_Levels[0].size(); ++s)
        {
            const auto& S = Two.m_Levels[0][s];
            const auto& M = Two.m_SurfelMaterial[s];
            const bool bSlot = (S.m_Center.m_X < -1e-4f && M.m_iMaterial == 0) || (S.m_Center.m_X > 1e-4f && M.m_iMaterial == 1) || std::abs(S.m_Center.m_X) <= 1e-4f;
            const bool bUV   = std::abs(M.m_U - (S.m_Center.m_X / 4 + 0.5f)) < 1e-3f && std::abs(M.m_V - (S.m_Center.m_Z / 4 + 0.5f)) < 1e-3f;
            nBad += !(bSlot && bUV && std::abs(M.m_UVRadius - 0.3f / 4) < 1e-3f);
        }
        std::printf("surfel materials: %d of %zu surfels with a wrong slot, UV or footprint\n", nBad, Two.m_Levels[0].size());
        CHECK(Two.m_SurfelMaterial.size() == Two.m_Levels[0].size() && nBad == 0, "%d surfels sample their material in the wrong place", nBad);
    }

    // 6i. The surface's colors per surfel (SetSurfelMaterial): (a) both walls one white mesh, colored per surfel (red at +x, green at -x):
    //     the floor's basis leans red toward the red wall as with two colored meshes; (b) no sky, no sun: an emissive panel above a floor
    //     is all that lights it
    {
        surfel_mesh Floor, Walls;
        AddQuad(Floor, { -1, 0, -2 }, { 0, 0, 4 }, { 2, 0, 0 }, 0.25f);
        AddQuad(Walls, {  1, 0, -2 }, { 0, 0, 4 }, { 0, 2, 0 }, 0.25f);
        AddQuad(Walls, { -1, 0, -2 }, { 0, 2, 0 }, { 0, 0, 4 }, 0.25f);
        Floor.Bake(0.3f, 0.9f); Walls.Bake(0.3f, 0.9f);
        world W;
        const int iFloor = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        const int iWalls = W.Add(Walls, xmath::fmat4::fromIdentity(), { 1, 1, 1 });
        std::vector<v3> Albedo;
        for (const auto& S : Walls.m_Levels[0]) Albedo.push_back(S.m_Center.m_X > 0 ? v3{ 0.9f, 0.05f, 0.05f } : v3{ 0.05f, 0.9f, 0.05f });
        W.SetSurfelMaterial(iWalls, Albedo, {});
        lights L; L.m_Sky = { 0.05f, 0.05f, 0.05f };
        L.m_SunDirection = v3{ 0, -1, 0.2f }.NormalizeCopy(); L.m_SunColor = { 3, 3, 3 };
        for (int f = 0; f < 200; ++f) W.Update(L, { 0, 1, 0 }, 1 << 30, 14000 + f);
        int s0 = 0; float Best = 1e30f;
        for (int s = 0; s < int(Floor.m_Levels[0].size()); ++s) { const float d = Floor.m_Levels[0][s].m_Center.Length(); if (d < Best) { Best = d; s0 = s; } }
        const v3 ToRed = W.BasisRead(iFloor, s0, v3{ 0.8f, 0.6f, 0 }.NormalizeCopy()), ToGreen = W.BasisRead(iFloor, s0, v3{ -0.8f, 0.6f, 0 }.NormalizeCopy());
        const float RgRed = ToRed.m_X / std::max(ToRed.m_Y, 1e-6f), RgGreen = ToGreen.m_X / std::max(ToGreen.m_Y, 1e-6f);

        surfel_mesh Ground, Panel;
        AddQuad(Ground, { -2, 0, -2 }, { 0, 0, 4 }, { 4, 0, 0 }, 0.5f);
        AddQuad(Panel,  { -0.5f, 1, -0.5f }, { 1, 0, 0 }, { 0, 0, 1 }, 0.25f);    // facing down
        Ground.Bake(0.3f, 0.9f); Panel.Bake(0.3f, 0.9f);
        world D;
        D.m_Setup.m_ProjScale = 1000;                     // the level rule: far (500 m) takes the parents
        const int iGround = D.Add(Ground, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        const int iPanel  = D.Add(Panel, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        D.SetSurfelMaterial(iPanel, {}, std::vector<v3>(Panel.m_Levels[0].size(), v3{ 2, 2, 2 }));
        lights Dark;
        for (int f = 0; f < 60; ++f) D.Update(Dark, { 0, 1, 0 }, 1 << 30, 15000 + f);
        double Under = 0; int nUnder = 0;
        for (std::size_t s = 0; s < Ground.m_Levels[0].size(); ++s)
            if (Ground.m_Levels[0][s].m_Center.Length() < 0.3f) { Under += D.InstanceLight(iGround)[s].m_X; ++nUnder; }
        Under /= std::max(nUnder, 1);
        std::printf("surfel colors: red/green toward the red wall %.2f, the green %.2f; a dark room lit by an emissive panel: %.3f under it\n", RgRed, RgGreen, Under);
        CHECK(RgRed > 1.3f * RgGreen, "the walls' surfel colors do not reach the floor (%.2f vs %.2f)", RgRed, RgGreen);
        CHECK(Under > 0.5, "the emissive panel lights nothing (%.3f)", Under);

        // The emitters' light is made again only when its inputs change (Update: not every update): a new scale reaches the floor
        D.m_Setup.m_EmissiveScale = 2;
        D.DirtyAll();
        for (int f = 0; f < 60; ++f) D.Update(Dark, { 0, 1, 0 }, 1 << 30, 15100 + f);
        double Under2 = 0;
        for (std::size_t s = 0; s < Ground.m_Levels[0].size(); ++s)
            if (Ground.m_Levels[0][s].m_Center.Length() < 0.3f) Under2 += D.InstanceLight(iGround)[s].m_X;
        Under2 /= std::max(nUnder, 1);
        std::printf("emissive scale 1 -> 2: the floor under the panel %.3f -> %.3f\n", Under, Under2);
        CHECK(std::abs(Under2 / Under - 2) < 0.15, "a new emissive scale does not reach the light (%.3f -> %.3f)", Under, Under2);

        // What a copy of the moments relies on (the renderer keeps them packed): an update changes only the surfels it lists
        // (LastUpdated), and a level switch resets them all (MomentsReset)
        std::vector<moments> Before = D.InstanceMoments(iGround);
        D.Update(Dark, { 0, 1, 0 }, 64 * 8, 15200);
        std::vector<char> Listed(Before.size(), 0);
        for (const auto& [i, s] : D.LastUpdated()) if (i == iGround) Listed[s] = 1;
        int nUnlisted = 0;
        for (std::size_t s = 0; s < Before.size(); ++s) nUnlisted += !Listed[s] && Before[s] != D.InstanceMoments(iGround)[s];
        const auto Reset = D.MomentsReset(iGround);
        D.Update(Dark, { 0, 1, 500 }, 64 * 8, 15201);                               // far: the parent level
        std::printf("moments: %d changed without being listed; a level switch resets them: %d\n", nUnlisted, int(D.MomentsReset(iGround) != Reset));
        CHECK(nUnlisted == 0 && D.MomentsReset(iGround) != Reset, "the moments change where a packed copy would not follow");
    }

    // 6j. The emitter lists: a small emissive tile (20 x 20 cm, an exit sign) 1 m above a floor in the dark, few rays: the floor under it
    //     gets the same light as the rays' hits find with many rays (the reference), and with 8 rays its noise across the floor is far
    //     lower than the hits' (which rarely find a small tile)
    {
        surfel_mesh Ground, Sign;
        AddQuad(Ground, { -2, 0, -2 }, { 0, 0, 4 }, { 4, 0, 0 }, 0.5f);
        AddQuad(Sign,   { -0.1f, 1, -0.1f }, { 0.2f, 0, 0 }, { 0, 0, 0.2f }, 0.1f);   // facing down
        Ground.Bake(0.3f, 0.9f); Sign.Bake(0.05f, 0.15f);
        const auto Run = [&](bool bLinks, int Rays, int Updates, double& Spread, bool bSample = false, bool bExternal = false)
        {
            world D;
            D.m_bEmitterLinks = bLinks;
            D.m_Setup.m_bEmitterSampling = bSample;
            D.m_Setup.m_bEmitterDirectExternal = bExternal;
            D.m_Setup.m_RaysPerSurfel = Rays;
            const int iG = D.Add(Ground, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            const int iS = D.Add(Sign, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            D.SetSurfelMaterial(iS, {}, std::vector<v3>(Sign.m_Levels[0].size(), v3{ 20, 20, 20 }));
            lights Dark;
            for (int f = 0; f < Updates; ++f) D.Update(Dark, { 0, 1, 0 }, 1 << 30, 16000 + f);
            double m = 0, m2 = 0; int n = 0;
            for (std::size_t s = 0; s < Ground.m_Levels[0].size(); ++s)
                if (Ground.m_Levels[0][s].m_Center.Length() < 0.5f) { const double e = D.InstanceLight(iG)[s].m_X; m += e; m2 += e * e; ++n; }
            m /= std::max(n, 1);
            Spread = std::sqrt(std::max(0.0, m2 / std::max(n, 1) - m * m)) / std::max(m, 1e-9);
            return m;
        };
        double SRef, SLinks, SHits;
        const double Ref   = Run(false, 256, 40, SRef);           // the reference: many rays find the tile
        const double Links = Run(true, 8, 20, SLinks);
        const double Hits  = Run(false, 8, 20, SHits);
        std::printf("emitter lists: under a 20 cm sign the reference %.3f (spread %.0f%%); links, 8 rays %.3f (spread %.0f%%); hits, 8 rays %.3f (spread %.0f%%)\n"
                   , Ref, 100 * SRef, Links, 100 * SLinks, Hits, 100 * SHits);
        CHECK(std::abs(Links - Ref) < 0.25 * Ref, "the emitter lists' light is off: %.3f vs the reference %.3f", Links, Ref);
        CHECK(SLinks < 0.5 * SHits, "the emitter lists are no less noisy than the hits (%.0f%% vs %.0f%%)", 100 * SLinks, 100 * SHits);
        // G5: the light sampling instead of the links (8 samples a pass, a shadow ray each): the same light, about as quiet, nothing built
        double SSample;
        const double Sample = Run(true, 8, 20, SSample, true);
        std::printf("light sampling, 8 rays: %.3f (spread %.0f%%)\n", Sample, 100 * SSample);
        CHECK(std::abs(Sample - Ref) < 0.1 * Ref, "the sampled emitters' light is off: %.3f vs the reference %.3f", Sample, Ref);
        // 6j3. Lighting shades the emitters' direct light (setup::m_bEmitterDirectExternal: its emitter lights) - the CPU's solver keeps it
        //      out of what it shows in every mode: the rays' hits, the links, the light sampling. The floor under the sign: none of it
        double SX;
        const double XHits = Run(false, 8, 20, SX, false, true), XLinks = Run(true, 8, 20, SX, false, true), XSample = Run(true, 8, 20, SX, true, true);
        std::printf("the emitters' direct light left to lighting: under the sign hits %.4f, links %.4f, sampling %.4f (the reference %.3f)\n", XHits, XLinks, XSample, Ref);
        CHECK(XHits < 0.05 * Ref && XLinks < 0.05 * Ref && XSample < 0.05 * Ref, "the emitters' direct light is still in the GI: hits %.4f, links %.4f, sampling %.4f (the reference %.3f)", XHits, XLinks, XSample, Ref);
        CHECK(SSample < 0.5 * SHits, "the light sampling is no less noisy than the hits (%.0f%% vs %.0f%%)", 100 * SSample, 100 * SHits);
    }

    // 6j2. A light smaller than its surfels (SetTriangleEmission): a surfel's emission is the mean over its patch, so the emitted power
    //      (the emission times the surfels' area, 1.47 r^2) is the light's at any surfel size - a point sample read 0 or the full emission
    {
        for (const float Rad : { 0.3f, 0.7f })
        {
            surfel_mesh M;
            AddQuad(M, { -2, 1, -2 }, { 4, 0, 0 }, { 0, 0, 4 }, 0.5f);                    // the ceiling (facing down)
            const std::size_t nCeiling = M.m_Tris.size();
            AddQuad(M, { -0.1f, 0.99f, -0.1f }, { 0.2f, 0, 0 }, { 0, 0, 0.2f }, 0.2f);    // the light: 20 cm square, 1 cm below it
            M.Bake(Rad, 3 * Rad);
            world W;
            const int i = W.Add(M, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            W.SetSurfelMaterial(i, {}, std::vector<v3>(M.m_Levels[0].size(), v3{ 0, 0, 0 }));
            std::vector<v3> Tri(M.m_Tris.size(), v3{ 0, 0, 0 });
            for (std::size_t t = nCeiling; t < Tri.size(); ++t) Tri[t] = v3{ 10, 10, 10 };
            W.SetTriangleEmission(i, 0, Tri);
            const auto& E = W.SurfaceEmission(i);
            double Power = 0; int nEmitting = 0;
            for (std::size_t s = 0; s < E.size(); ++s)
            {
                const float r = M.m_Levels[0][s].m_Radius;
                Power += E[s].m_X * 1.47f * r * r;
                nEmitting += E[s].m_X > 0;
            }
            const double Want = 10.0 * 0.2 * 0.2;
            std::printf("a 20 cm light, surfels of %.1f m: %d surfels emit, power %.3f (the light's %.3f)\n", Rad, nEmitting, Power, Want);
            CHECK(nEmitting > 0, "the light has no surfel to emit it (surfels of %.1f m)", Rad);
            CHECK(std::abs(Power - Want) < 0.35 * Want, "the emitted power is off: %.3f vs the light's %.3f (surfels of %.1f m)", Power, Want, Rad);
        }
    }

    // 6j2b. The emission resolve's bound (GI review, experiment A): it splits each emissive triangle's power over the surfels its vertices
    //       name (or, for a triangle larger than they can represent, over samples with a capped search), batch by batch. Three adversarial
    //       cases: (a) many small triangles over few large surfels (the Golem: it used to search every surfel for every triangle), (b) the
    //       same panel at a coarser tessellation: the power and its spread must not change, (c) two triangles over many small surfels (a
    //       big panel: its power must reach the surfels over it, not only its corners'). Each: power conserved, work bounded per triangle
    {
        struct res { double Power, Want, Unassigned; std::size_t Work, nTris; std::vector<v3> E; std::vector<float> R; std::array<double, 9> Irr{}; };
        const auto Run = [](float Split, float Rad, int nBatches) -> res
        {
            surfel_mesh M;
            AddQuad(M, { -2, 0, -2 }, { 0, 0, 4 }, { 4, 0, 0 }, Split);          // a 4 x 4 m emissive floor panel, facing up
            M.Bake(Rad, 3 * Rad);
            world W;
            const int i = W.Add(M, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            W.SetSurfelMaterial(i, {}, std::vector<v3>(M.m_Levels[0].size(), v3{ 0, 0, 0 }));
            const std::size_t n = M.m_Tris.size(), Per = (n + nBatches - 1) / nBatches;
            for (std::size_t f = 0; f < n; f += Per) W.SetTriangleEmission(i, f, std::vector<v3>(std::min(Per, n - f), v3{ 2, 2, 2 }));     // in batches, as the capture
            res R{ 0, 2.0 * 16.0, W.EmissionUnassigned(i).m_X, W.EmissionResolveWork(i), n, W.SurfaceEmission(i), {} };
            for (const auto& S : M.m_Levels[0]) R.R.push_back(S.m_Radius);
            for (std::size_t s = 0; s < R.E.size(); ++s)
            {
                const double p = R.E[s].m_X * 1.47f * R.R[s] * R.R[s];
                R.Power += p;
                const auto& c = M.m_Levels[0][s].m_Center;
                for (int q = 0; q < 9; ++q)     // what it delivers to 9 points 1 m above the panel (each surfel a Lambertian emitter of its power)
                {
                    const v3 X{ float(q % 3 - 1), 1.0f, float(q / 3 - 1) }, D = X - c;
                    const double d2 = D.Dot(D), ct = D.m_Y / std::sqrt(d2);
                    R.Irr[std::size_t(q)] += p * ct * ct / (3.14159265 * d2);
                }
            }
            return R;
        };
        const res A = Run(0.02f, 0.5f, 7), B = Run(0.25f, 0.5f, 1), C = Run(4.0f, 0.15f, 1);
        for (const auto* p : { &A, &B, &C })
            std::printf("emission resolve: %zu triangles, %zu surfels: power %.3f (the panel's %.3f, unassigned %.3f), %.1f candidates a triangle\n",
                        p->nTris, p->E.size(), p->Power + p->Unassigned, p->Want, p->Unassigned, double(p->Work) / double(p->nTris));
        for (const auto* p : { &A, &B, &C })
        {
            CHECK(std::abs(p->Power + p->Unassigned - p->Want) < 1e-3 * p->Want, "the emitted power is not conserved: %.4f vs %.4f", p->Power + p->Unassigned, p->Want);
            CHECK(p->Unassigned < 0.02 * p->Want, "%.1f%% of the power has no surfel", 100 * p->Unassigned / p->Want);
        }
        CHECK(double(A.Work) <= 12.0 * double(A.nTris), "the resolve searched more than the triangles' vertices' surfels: %.1f a triangle", double(A.Work) / double(A.nTris));
        {   // (b): the same panel at 2 cm and at 25 cm triangles (and as 2 triangles over fine surfels, (c), the reference): the irradiance
            //      their surfels deliver 1 m above it. The surfels differ between bakes; the light they give must not
            const auto Rel = [](const res& X, const res& Y) { double d = 0, m = 0; for (int q = 0; q < 9; ++q) { d += std::pow(X.Irr[q] - Y.Irr[q], 2.0); m += std::pow(Y.Irr[q], 2.0); } return std::sqrt(d / std::max(m, 1e-12)); };
            std::printf("emission resolve: the irradiance 1 m above, 2 cm vs 25 cm triangles %.1f%%, 2 cm vs the fine reference %.1f%%, 25 cm vs it %.1f%% (rms)\n", 100 * Rel(A, B), 100 * Rel(A, C), 100 * Rel(B, C));
            CHECK(Rel(A, B) < 0.1, "the tessellation changes the light the panel gives by %.0f%%", 100 * Rel(A, B));
            CHECK(Rel(A, C) < 0.2 && Rel(B, C) < 0.2, "the large surfels' light is off the fine reference: %.0f%%, %.0f%%", 100 * Rel(A, C), 100 * Rel(B, C));
        }
        {   // (c): the big panel lights the surfels over it: most of them emit, and none holds a corner's share of the whole
            int nLit = 0; double Max = 0;
            for (std::size_t s = 0; s < C.E.size(); ++s) { nLit += C.E[s].m_X > 0; Max = std::max(Max, double(C.E[s].m_X) * 1.47 * C.R[s] * C.R[s]); }
            std::printf("emission resolve: a 2-triangle panel over %zu surfels: %d emit, the largest share %.1f%%\n", C.E.size(), nLit, 100 * Max / C.Want);
            CHECK(nLit > int(C.E.size()) / 2, "only %d of %zu surfels under a big panel emit", nLit, C.E.size());
            CHECK(Max < 0.05 * C.Want, "one surfel holds %.0f%% of a big panel's power", 100 * Max / C.Want);
        }
    }

    // 6j2c. The unsupported fallback's bound (GI review, experiment A): 200 small quads facing every way in one 1 m cell, no leak protection, so the
    //       cell holds a patch per orientation and lists hundreds of surfels in all. A query forced through the fallback (world::Unsupported) walks at
    //       most fallback_cap_v of them (it walked the whole cell: 10,000 to 33,000 entries a hit on the Test Models) and still returns light
    //       from the nearest surfels of its own patch, not the instance's mean
    {
        surfel_mesh M;
        std::mt19937 Rng(7);     // 400 small quads, every way, in one cell (what a dense engine or a crumpled prop is to 1 m surfels)
        std::uniform_real_distribution<float> U(-1, 1);
        for (int k = 0; k < 400; ++k)
        {
            v3 N{ U(Rng), U(Rng), U(Rng) }; if (N.Length() < 0.1f) N = { 0, 1, 0 }; N = N.NormalizeCopy();
            const v3 T = (std::abs(N.m_X) < 0.9f ? v3{ 1, 0, 0 } : v3{ 0, 1, 0 }).Cross(N).NormalizeCopy(), Bt = N.Cross(T);
            AddQuad(M, v3{ 0.4f * U(Rng), 0.4f * U(Rng), 0.4f * U(Rng) } - (T + Bt) * 0.05f, Bt * 0.1f, T * 0.1f, 0.1f);
        }
        M.m_bLeakProtection = false;
        M.Bake(1.0f, 3.0f);
        M.BuildReadGridAt(1.0f);        // the radius' cell, not refined (BuildReadGrid's would split it: 6j2d): the worst case a grid can hold
        world W;
        const int i = W.Add(M, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        lights L; L.m_SunDirection = v3{ 0.3f, -1, 0.2f }.NormalizeCopy(); L.m_SunColor = { 3, 3, 3 }; L.m_Sky = { 0.2f, 0.2f, 0.2f };
        for (int u = 0; u < 20; ++u) (void)W.Update(L, { 0, 3, 0 }, 1 << 16, unsigned(u));
        const std::size_t Top = 0;                                       // a triangle of the first quad
        const auto& T = M.m_Tris[Top];
        const v3 P = (M.m_Pos[T[0]] + M.m_Pos[T[1]] + M.m_Pos[T[2]]) * (1.0f / 3.0f);
        const std::size_t ListSize = M.ReadCellList(P, (M.m_Pos[T[1]] - M.m_Pos[T[0]]).Cross(M.m_Pos[T[2]] - M.m_Pos[T[0]]).NormalizeCopy()).size();
        std::size_t AllCell = 0, nPatches = 0;         // what the old fallback walked: every patch's list in the cell
        {
            const v3 Nn = (M.m_Pos[T[1]] - M.m_Pos[T[0]]).Cross(M.m_Pos[T[2]] - M.m_Pos[T[0]]).NormalizeCopy();
            const auto C = surfel_mesh::ReadCellAt(P, Nn, M.m_ReadCell);
            const auto K = surfel_mesh::ReadKey(C);
            const std::uint32_t Mask = std::uint32_t(M.m_ReadTable.size() - 1);
            for (std::uint32_t h = surfel_mesh::ReadHash(C) & Mask, n = 0; n <= Mask; h = (h + 1) & Mask, ++n)
            {
                const auto& Ent = M.m_ReadTable[h];
                if (Ent[1] == 0xFFFFFFFFu) break;
                if (Ent[0] != K[0] || Ent[1] != K[1]) continue;
                nPatches = Ent[3];
                for (std::uint32_t p = 0; p < Ent[3]; ++p) AllCell += M.m_ReadList[Ent[2] + 4 * p + 3] & 0xFFFFu;
                break;
            }
        }
        world::hit H; H.m_Inst = i; H.m_Tri = int(Top); H.m_U = 1.0f / 3.0f; H.m_V = 1.0f / 3.0f;
        int Walked = 0;
        const v3 E = W.Unsupported(H, &Walked);
        const v3 Mean = W.MeanLight(i);
        std::printf("fallback bound: the cell holds %zu patches listing %zu surfels in all (the old walk); its patch lists %zu, the fallback walked %d (cap %d); its light %.3f, the instance's mean %.3f\n", nPatches, AllCell, ListSize, Walked, fallback_cap_v, E.m_X, Mean.m_X);
        CHECK(AllCell > 10 * std::size_t(Walked) && AllCell > 4 * std::size_t(fallback_cap_v), "the test's cell lists only %zu surfels: no worst case", AllCell);
        CHECK(std::size_t(Walked) <= ListSize, "the fallback walked past its own patch (%d of %zu)", Walked, ListSize);
        CHECK(Walked <= fallback_cap_v, "the fallback walked %d entries (cap %d)", Walked, fallback_cap_v);
        CHECK(std::isfinite(E.m_X) && E.m_X > 0, "the fallback returned no light (%.3f)", E.m_X);
    }

    // 6j2d. The read grid follows the surface (GI review: the Golem's 2 m cells held 300 patches each, every pixel scored them all): the same
    //       crumpled cluster as 6j2c, BuildReadGrid halves the cell while the cells hold more than read_patches_v patches on average (at most
    //       read_halvings_v times). Fewer patches per cell than at the radius' cell, and every quad still reads its surfels (none lost)
    {
        surfel_mesh M;
        std::mt19937 Rng(7);
        std::uniform_real_distribution<float> U(-1, 1);
        for (int k = 0; k < 400; ++k)
        {
            v3 N{ U(Rng), U(Rng), U(Rng) }; if (N.Length() < 0.1f) N = { 0, 1, 0 }; N = N.NormalizeCopy();
            const v3 T = (std::abs(N.m_X) < 0.9f ? v3{ 1, 0, 0 } : v3{ 0, 1, 0 }).Cross(N).NormalizeCopy(), Bt = N.Cross(T);
            AddQuad(M, v3{ 0.4f * U(Rng), 0.4f * U(Rng), 0.4f * U(Rng) } - (T + Bt) * 0.05f, Bt * 0.1f, T * 0.1f, 0.1f);
        }
        M.m_bLeakProtection = false;
        M.Bake(1.0f, 3.0f);
        const auto Unread = [&]
        {
            int n = 0;
            for (std::size_t t = 0; t < M.m_Tris.size(); t += 2)
            {
                const auto& T = M.m_Tris[t];
                const v3 P = (M.m_Pos[T[0]] + M.m_Pos[T[1]] + M.m_Pos[T[2]]) * (1.0f / 3.0f);
                n += M.ReadCellList(P, (M.m_Pos[T[1]] - M.m_Pos[T[0]]).Cross(M.m_Pos[T[2]] - M.m_Pos[T[0]]).NormalizeCopy()).empty() ? 1 : 0;
            }
            return n;
        };
        M.BuildReadGridAt(1.0f);
        const float Coarse = M.ReadPatchesPerCell();
        const int   CoarseUnread = Unread();
        M.BuildReadGrid();
        const float Fine = M.ReadPatchesPerCell();
        const int   FineUnread = Unread();
        std::printf("read grid: at the radius' cell (1 m) %.1f patches a cell, %d quads read nothing; refined to %.3f m: %.1f a cell, %d read nothing\n", Coarse, CoarseUnread, M.m_ReadCell, Fine, FineUnread);
        CHECK(Coarse > surfel_mesh::read_patches_v, "the test's cluster is not crumpled enough (%.1f patches a cell at 1 m)", Coarse);
        CHECK(M.m_ReadCell < 1.0f && M.m_ReadCell >= 1.0f / float(1 << surfel_mesh::read_halvings_v), "the cell was not refined within its halvings (%.3f m)", M.m_ReadCell);
        CHECK(Fine < 0.5f * Coarse, "refining did not cut the patches a cell scores (%.1f -> %.1f)", Coarse, Fine);
        CHECK(FineUnread <= CoarseUnread, "the finer grid loses quads (%d read nothing, %d before)", FineUnread, CoarseUnread);
    }

    // 6j2e. The read lists are bounded (GI working set, job 1: the Bistro's layered surface listed thousands of surfels a part, 5 GB of lists,
    //       a 19 GB peak): a bush (1,500 leaves of 10 cm facing about up, in 2 x 1 x 2 m: what the Bistro's trees are to 1 m surfels: many
    //       layers that see each other), every part's list at most read_list_cap_v (its nearest seen surfels from the cell's center) against
    //       the unbounded lists; the words, the build's time, and the converged light at every leaf (the solver reads whole lists, the pixel
    //       its 32 nearest) against a second unbounded world's noise
    {
        surfel_mesh M;
        std::mt19937 Rng(11);
        std::uniform_real_distribution<float> U(-1, 1);
        for (int k = 0; k < 1500; ++k)
        {
            const v3 N = v3{ 0.5f * U(Rng), 1, 0.5f * U(Rng) }.NormalizeCopy();
            const v3 T = v3{ 1, 0, 0 }.Cross(N).NormalizeCopy(), Bt = N.Cross(T);
            AddQuad(M, v3{ U(Rng), 0.5f + 0.5f * U(Rng), U(Rng) } - (T + Bt) * 0.05f, T * 0.1f, Bt * 0.1f, 0.1f);     // (T x Bt: facing N)
        }
        // (leak protection on, the default and the Bistro's: the solver's hits read the lists, world::Indirect)
        M.Bake(1.0f, 3.0f);
        surfel_mesh B = M;                                      // the same surfels: only the lists differ
        const auto Lists = [](const surfel_mesh& X)
        {
            std::size_t nParts = 0, nEnt = 0, Max = 0;
            for (const auto& E : X.m_ReadTable)
            {
                if (E[1] == 0xFFFFFFFFu) continue;
                for (std::uint32_t p = 0; p < E[3]; ++p) { const std::size_t n = X.m_ReadList[E[2] + 4 * p + 3] & 0xFFFFu; ++nParts; nEnt += n; Max = std::max(Max, n); }
            }
            return std::array<std::size_t, 3>{ nParts, nEnt, Max };
        };
        const auto Timed = [](surfel_mesh& X) { const auto T0 = std::chrono::steady_clock::now(); X.BuildReadGridAt(1.0f); return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count(); };
        M.m_ReadListCap = ~std::size_t(0);
        const double Tm = Timed(M), Tb = Timed(B);
        const auto Lm = Lists(M), Lb = Lists(B);
        CHECK(Lm[0] == Lb[0], "the bound changed the parts (%zu, %zu)", Lm[0], Lb[0]);
        CHECK(Lm[2] > 2 * surfel_mesh::read_list_cap_v, "the test's cluster lists only %zu surfels a part: no worst case", Lm[2]);
        CHECK(Lb[2] <= surfel_mesh::read_list_cap_v, "a bounded part lists %zu surfels (cap %zu)", Lb[2], surfel_mesh::read_list_cap_v);
        CHECK(B.m_ReadList.size() <= Lb[0] * (4 + surfel_mesh::read_list_cap_v), "the bounded lists hold %zu words for %zu parts", B.m_ReadList.size(), Lb[0]);
        // the packed lists (the GPU's and the resource's: all-seen entries 16 bits): unpacked, the same table and lists; the unbounded ones
        // stay as they are (longer than the fallback's walk: their order is the nearest first)
        for (auto* X : { &B, &M })
        {
            std::vector<std::array<std::uint32_t, 4>> PT, UT; std::vector<std::uint32_t> PL, UL;
            surfel_mesh::PackReadGrid(true, X->m_ReadTable, X->m_ReadList, PT, PL);
            surfel_mesh::PackReadGrid(false, PT, PL, UT, UL);
            std::size_t nAllSeen = 0;
            for (const auto& E : X->m_ReadTable) if (E[1] != 0xFFFFFFFFu) for (std::uint32_t p = 0; p < E[3]; ++p)
            {
                const std::uint32_t F = X->m_ReadList[E[2] + 4 * p + 2], n = X->m_ReadList[E[2] + 4 * p + 3] & 0xFFFFu;
                if (n <= std::uint32_t(fallback_cap_v)) for (std::uint32_t k = 0; k < n; ++k) nAllSeen += (X->m_ReadList[F + k] >> 16) == 0xFFFFu;
            }
            // (unpacked, a list has its entries with corner weights first, then its all-seen ones: the same entries - SortReadLists orders them again)
            bool bSame = UT == X->m_ReadTable && UL.size() == X->m_ReadList.size();
            for (const auto& E : X->m_ReadTable) if (bSame && E[1] != 0xFFFFFFFFu) for (std::uint32_t p = 0; p < 4 * E[3]; p += 4)
            {
                const std::uint32_t F = X->m_ReadList[E[2] + p + 2], n = X->m_ReadList[E[2] + p + 3] & 0xFFFFu;
                bSame = bSame && std::equal(UL.begin() + E[2] + p, UL.begin() + E[2] + p + 4, X->m_ReadList.begin() + E[2] + p)
                     && std::is_permutation(UL.begin() + F, UL.begin() + F + n, X->m_ReadList.begin() + F);
            }
            CHECK(bSame, "the packed lists unpacked are not the lists (%s)", X == &B ? "bounded" : "unbounded");
            CHECK(X->m_ReadList.size() - PL.size() >= nAllSeen / 4, "packed %zu words of %zu, %zu all-seen entries", PL.size(), X->m_ReadList.size(), nAllSeen);
            std::printf("packed read lists (%s): %zu words of %zu, %zu entries all-seen\n", X == &B ? "bounded" : "unbounded", PL.size(), X->m_ReadList.size(), nAllSeen);
        }
        // the unsupported parts are the same (a part that sees no surfel lists none either way)
        CHECK(M.m_ReadUnsupported == B.m_ReadUnsupported, "the bound changed the unsupported parts (%u, %u)", M.m_ReadUnsupported, B.m_ReadUnsupported);

        lights Lt; Lt.m_SunDirection = v3{ 0.3f, -1, 0.2f }.NormalizeCopy(); Lt.m_SunColor = { 3, 3, 3 }; Lt.m_Sky = { 0.2f, 0.2f, 0.2f };
        world Wa, Wb, Wc;
        Wa.Add(M, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f }); Wb.Add(M, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        Wc.Add(B, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        const auto Run = [&](world& W, unsigned Seed)
        {
            W.m_Setup.m_RaysPerSurfel = 64; W.m_Setup.m_MinBlend = 1;
            for (int u = 0; u < 8; ++u) W.Update(Lt, { 0, 3, 0 }, 1 << 30, Seed + unsigned(u));
            W.m_Setup.m_MinBlend = 0;
            for (int u = 0; u < 150; ++u) W.Update(Lt, { 0, 3, 0 }, 1 << 30, Seed + 100 + unsigned(u));
        };
        Run(Wa, 1000); Run(Wb, 5000); Run(Wc, 1000);
        // Against the path-traced reference at the leaves (the reads are a kernel: the unbounded one blurs the canopy's light into the leaves
        // under it, so "off the unbounded" is not an error by itself)
        std::vector<std::pair<v3, v3>> Pts;
        for (std::size_t t = 0; t < M.m_Tris.size(); t += 20)
        {
            const auto& T = M.m_Tris[t];
            Pts.push_back({ (M.m_Pos[T[0]] + M.m_Pos[T[1]] + M.m_Pos[T[2]]) * (1.0f / 3.0f), (M.m_Pos[T[1]] - M.m_Pos[T[0]]).Cross(M.m_Pos[T[2]] - M.m_Pos[T[0]]).NormalizeCopy() });
        }
        std::vector<v3> Ref(Pts.size());
        ParallelFor(int(Pts.size()), [&](int i) { rng Rng(77 + i); Ref[i] = Reference(Wa, Lt, Pts[i].first, Pts[i].second, 1024, 4, Rng); });
        double Top = 1e-9;
        for (const auto& R : Ref) Top = std::max<double>(Top, Lum(R));
        std::vector<double> Ea, Eb, Ec;
        for (std::size_t i = 0; i < Pts.size(); ++i)
        {
            const v3 a = CacheAt(Wa, Pts[i].first, Pts[i].second), b = CacheAt(Wb, Pts[i].first, Pts[i].second), c = CacheAt(Wc, Pts[i].first, Pts[i].second);
            if (a.m_X < 0 || b.m_X < 0 || c.m_X < 0) continue;
            Ea.push_back(std::abs(Lum(a) - Lum(Ref[i])) / Top); Eb.push_back(std::abs(Lum(b) - Lum(Ref[i])) / Top); Ec.push_back(std::abs(Lum(c) - Lum(Ref[i])) / Top);
        }
        const auto Pct = [](std::vector<double> V, double p) { std::sort(V.begin(), V.end()); return V.empty() ? 0.0 : V[std::size_t(p * double(V.size() - 1))]; };
        const auto Mean = [](const std::vector<double>& V) { return V.empty() ? 0.0 : std::accumulate(V.begin(), V.end(), 0.0) / double(V.size()); };
        std::printf("read lists bounded: %zu parts, unbounded %zu entries (max %zu a part, %.1f ms), bounded %zu (max %zu, %.1f ms); off the path-traced light at %zu leaves "
                    "(of the brightest): unbounded %.2f%% mean, %.2f%% 95th (a second world %.2f%%, %.2f%%), bounded %.2f%% mean, %.2f%% 95th\n",
                    Lm[0], Lm[1], Lm[2], Tm, Lb[1], Lb[2], Tb, Ec.size(), 100 * Mean(Ea), 100 * Pct(Ea, 0.95), 100 * Mean(Eb), 100 * Pct(Eb, 0.95), 100 * Mean(Ec), 100 * Pct(Ec, 0.95));
        CHECK(Ec.size() > Pts.size() / 2, "too few leaves read light (%zu of %zu)", Ec.size(), Pts.size());
        const double Noise = std::abs(Mean(Ea) - Mean(Eb));
        CHECK(Mean(Ec) <= Mean(Ea) + 2 * Noise + 0.005, "the bounded lists are farther from the path-traced light (%.2f%%) than the unbounded (%.2f%%)", 100 * Mean(Ec), 100 * Mean(Ea));
    }

    // 6j2f. The bake's speedups keep the same surfels (GI working set, job 4: the Bistro's bake was 17 min, 11 of them the vertex surfels'
    //       nearest search, 3 the parents' darts): the vertex surfels and the parents of a bake against the rule tested one surfel at a time
    //       (every compatible surfel in reach seen or not, the 4 nearest kept), on the bush of 6j2e (layers that see each other)
    {
        surfel_mesh M;
        std::mt19937 Rng(11);
        std::uniform_real_distribution<float> U(-1, 1);
        for (int k = 0; k < 1500; ++k)
        {
            const v3 N = v3{ 0.5f * U(Rng), 1, 0.5f * U(Rng) }.NormalizeCopy();
            const v3 T = v3{ 1, 0, 0 }.Cross(N).NormalizeCopy(), Bt = N.Cross(T);
            AddQuad(M, v3{ U(Rng), 0.5f + 0.5f * U(Rng), U(Rng) } - (T + Bt) * 0.05f, T * 0.1f, Bt * 0.1f, 0.1f);
        }
        M.Bake(0.25f, 0.75f);
        const auto& L = M.m_Levels[0];
        surfel_grid G{ L[0].m_Radius, {} };
        for (int s = 0; s < int(L.size()); ++s) G.Add(L[s].m_Center, s);
        int nDiff = 0;
        for (std::size_t v = 0; v < M.m_Pos.size(); ++v)
        {
            std::array<std::pair<float, int>, 4> Best; Best.fill({ 1e30f, -1 });
            for (int Ring = 1; Ring <= 4 && Best[0].second < 0; ++Ring)
                G.Near(M.m_Pos[v], Ring, [&](int s)
                {
                    if (L[s].m_Normal.Dot(M.m_Nrm[v]) < 0.5f) return;
                    const float d = (M.m_Pos[v] - L[s].m_Center).Length();
                    if (d >= Best[3].first || std::any_of(Best.begin(), Best.end(), [&](const auto& b) { return b.second == s; })) return;
                    if (!M.Sees(M.m_Pos[v], M.m_Nrm[v], L[s])) return;
                    Best[3] = { d, s }; std::sort(Best.begin(), Best.end());
                });
            nDiff += M.m_VertSurfels[v] != std::array<int, 4>{ Best[0].second, Best[1].second, Best[2].second, Best[3].second };
        }
        // The darts one at a time (the bake's candidates, its shuffle, its rule), the fine level and then the parents: every center and
        // normal the same, bit for bit; each surfel's parent the nearest seen one, every one tested
        const auto Darts = [&](const std::vector<v3>& P, const std::vector<v3>& N, float R)
        {
            std::vector<surfel> Out; surfel_grid Gd{ R, {} };
            std::vector<int> Order(P.size());
            for (std::size_t i = 0; i < Order.size(); ++i) Order[i] = int(i);
            rng Shuffle(0x5EEDu);
            for (std::size_t i = Order.size(); i > 1; --i) std::swap(Order[i - 1], Order[std::size_t(Shuffle.Next() * float(i)) % i]);
            for (const int i : Order)
            {
                bool bTaken = false;
                Gd.Near(P[std::size_t(i)], 1, [&](int s) { bTaken = bTaken || ((P[std::size_t(i)] - Out[std::size_t(s)].m_Center).Length() < R && Out[std::size_t(s)].m_Normal.Dot(N[std::size_t(i)]) > 0.8f
                                         && std::abs((Out[std::size_t(s)].m_Center - P[std::size_t(i)]).Dot(N[std::size_t(i)])) < 0.1f * R && M.Sees(P[std::size_t(i)], N[std::size_t(i)], Out[std::size_t(s)])); });
                if (bTaken) continue;
                Gd.Add(P[std::size_t(i)], int(Out.size())); Out.push_back({ P[std::size_t(i)], N[std::size_t(i)], R });
            }
            return Out;
        };
        const auto Same = [](const std::vector<surfel>& A, const std::vector<surfel>& B)
        {
            if (A.size() != B.size()) return false;
            for (std::size_t s = 0; s < A.size(); ++s)
                if (std::memcmp(&A[s].m_Center, &B[s].m_Center, sizeof(v3)) || std::memcmp(&A[s].m_Normal, &B[s].m_Normal, sizeof(v3)) || A[s].m_Radius != B[s].m_Radius) return false;
            return true;
        };
        std::vector<v3> CP, CN; M.Candidates(0.25f, CP, CN, nullptr);
        const auto Fine = Darts(CP, CN, 0.25f);
        std::vector<v3> SP, SN; for (const auto& S : Fine) { SP.push_back(S.m_Center); SN.push_back(S.m_Normal); }
        const auto Parents = Darts(SP, SN, 0.75f);
        surfel_grid G1{ 0.75f, {} };
        for (int s = 0; s < int(Parents.size()); ++s) G1.Add(Parents[std::size_t(s)].m_Center, s);
        int nParentDiff = 0;
        for (std::size_t s = 0; s < Fine.size() && s < M.m_ParentOf.size(); ++s)
        {
            std::pair<float, int> Best{ 1e30f, -1 };
            for (int Ring = 1; Ring <= 4 && Best.second < 0; ++Ring)
                G1.Near(SP[s], Ring, [&](int p)
                {
                    if (Parents[std::size_t(p)].m_Normal.Dot(SN[s]) < 0.5f) return;
                    const float d = (SP[s] - Parents[std::size_t(p)].m_Center).Length();
                    if (std::pair{ d, p } >= Best || !M.Sees(SP[s], SN[s], Parents[std::size_t(p)])) return;
                    Best = { d, p };
                });
            nParentDiff += M.m_ParentOf[s] != std::max(Best.second, 0);
        }
        const bool bFine = Same(Fine, M.m_Levels[0]), bParents = Same(Parents, M.m_Levels[1]);
        std::printf("bake speedups: %zu fine surfels (one at a time: %s), %zu parents (%s), %d parents of surfels differ; vertex surfels off the one-at-a-time rule at %d of %zu vertices; "
                    "bake ms: candidates %.0f, darts %.0f, vertex surfels %.0f, parents %.0f, neighbours %.0f\n",
                    L.size(), bFine ? "the same, bit for bit" : "DIFFERENT", M.m_Levels[1].size(), bParents ? "the same" : "DIFFERENT", nParentDiff, nDiff, M.m_Pos.size(),
                    M.m_BakeMs[0], M.m_BakeMs[1], M.m_BakeMs[3], M.m_BakeMs[4], M.m_BakeMs[5]);
        CHECK(bFine && bParents && nParentDiff == 0, "the parallel darts differ from one at a time (fine %d, parents %d, parent links %d)", int(bFine), int(bParents), nParentDiff);
        CHECK(nDiff == 0, "the vertex surfels differ from the one-at-a-time rule at %d vertices", nDiff);
    }

    // 6k. Experiment D, step 1 (GI review: compiled transport), the CPU prototype: the Cornell box converged by the solver (the reference),
    //     its transport compiled once from the solver's own rays and reads (world::CompileTransport) and solved with no ray at all
    //     (SolveTransport: E = D + T E). Per iteration the residual |D + T E - E| and the error against the converged world; the links
    {
        cornell C;
        MakeCornell(C, 0.5f, false);
        Converge(C, 64, 200);
        auto& W = C.m_World;
        std::vector<v3> Ref;
        for (int i = 0; i < W.InstanceCount(); ++i) for (const auto& e : W.InstanceLight(i)) Ref.push_back(e);
        const auto T = W.CompileTransport(C.m_Lights, 1024);
        const std::size_t n = T.m_D.size();
        CHECK(n == Ref.size(), "the transport has %zu rows, the world %zu surfels", n, Ref.size());
        const auto Lum = [](const v3& e) { return 0.2126 * e.m_X + 0.7152 * e.m_Y + 0.0722 * e.m_Z; };
        double RefSum = 0; for (const auto& e : Ref) RefSum += Lum(e);
        const auto Err = [&](const std::vector<v3>& E)      // the mean relative error per instance (its mean light) and per surfel (RMS of the brightest)
        {
            double Inst = 0, Top = 1e-9, Rms = 0;
            for (const auto& e : Ref) Top = std::max(Top, Lum(e));
            for (int i = 0; i < W.InstanceCount(); ++i)
            {
                double a = 0, b = 0;
                for (std::uint32_t r = T.m_InstFirst[std::size_t(i)]; r < T.m_InstFirst[std::size_t(i) + 1]; ++r) { a += Lum(E[r]); b += Lum(Ref[r]); }
                Inst = std::max(Inst, std::abs(a - b) / std::max(b, 1e-9));
            }
            for (std::size_t r = 0; r < n; ++r) Rms += (Lum(E[r]) - Lum(Ref[r])) * (Lum(E[r]) - Lum(Ref[r]));
            return std::pair{ Inst, std::sqrt(Rms / double(n)) / Top };
        };
        std::vector<v3> E;
        std::printf("compiled transport (Cornell, %zu surfels, 1024 rays each): %.0f ms to compile, %zu links (%.1f a surfel, %.1f KB at 16 B), %.2f%% of rays unsupported, %.2f%% sky\n",
                    n, T.m_CompileMs, T.m_Col.size(), double(T.m_Col.size()) / double(n), T.m_Col.size() * 16.0 / 1024.0, 100.0 * double(T.m_nUnsupported) / double(T.m_nRays), 100.0 * double(T.m_nSky) / double(T.m_nRays));
        int Within = -1;
        double SolveMs = 0;
        for (int it = 1; it <= 12; ++it)
        {
            const auto Ts = std::chrono::steady_clock::now();
            world::SolveTransport(T, E, 1);
            SolveMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - Ts).count();
            std::vector<v3> F = E; world::SolveTransport(T, F, 1);
            double Res = 0; for (std::size_t r = 0; r < n; ++r) Res += std::abs(Lum(F[r]) - Lum(E[r]));
            const auto [Inst, Rms] = Err(E);
            if (Within < 0 && Inst < 0.01) Within = it;
            std::printf("  iteration %2d: residual %.2e of the light, the converged world's instances off by %.2f%% at most, surfels by %.2f%% RMS (of the brightest)\n", it, Res / RefSum, 100 * Inst, 100 * Rms);
        }
        std::printf("  within 1%% of the converged world's instances after %d iterations (%.3f ms an iteration on the CPU)\n", Within, SolveMs / 12);
        const auto [Inst, Rms] = Err(E);
        CHECK(Inst < 0.03, "the compiled transport's light is %.1f%% off the converged world's (an instance's mean)", 100 * Inst);

        // Top K: each row keeps its K heaviest links; the rest is not dropped and not renormalized away - it reads the mean of the instance
        // each dropped link pointed into (per row, one aggregate link per such instance: a two-level hierarchy)
        for (const bool bParents : { false, true })
        for (const int K : { 8, 16, 32, 64 })
        {
            // the groups the tail reads the mean of: whole instances, or the surfels' parents (their own level 1)
            std::vector<int> InstOf(n);
            int nInst = 0;
            for (int i = 0; i < W.InstanceCount(); ++i)
            {
                const auto& Pa = C.m_Meshes[std::size_t(i)].m_ParentOf;
                int nP = 1;
                for (std::uint32_t r = T.m_InstFirst[std::size_t(i)]; r < T.m_InstFirst[std::size_t(i) + 1]; ++r)
                {
                    const int p = bParents ? Pa[r - T.m_InstFirst[std::size_t(i)]] : 0;
                    InstOf[r] = nInst + p; nP = std::max(nP, p + 1);
                }
                nInst += nP;
            }
            std::vector<int> GroupSize(static_cast<std::size_t>(nInst), 0);
            for (std::size_t r = 0; r < n; ++r) ++GroupSize[std::size_t(InstOf[r])];
            std::vector<std::uint32_t> First{ 0 }, Col; std::vector<v3> Wk; std::vector<std::uint32_t> AFirst{ 0 }; std::vector<int> AInst; std::vector<v3> AW;
            for (std::size_t r = 0; r < n; ++r)
            {
                std::vector<std::uint32_t> Ord;
                for (std::uint32_t k = T.m_First[r]; k < T.m_First[r + 1]; ++k) Ord.push_back(k);
                std::sort(Ord.begin(), Ord.end(), [&](std::uint32_t x, std::uint32_t y) { return Lum(T.m_W[x]) > Lum(T.m_W[y]); });
                std::vector<v3> Tail(std::size_t(nInst), v3{ 0, 0, 0 });
                for (std::size_t q = 0; q < Ord.size(); ++q)
                    if (int(q) < K) { Col.push_back(T.m_Col[Ord[q]]); Wk.push_back(T.m_W[Ord[q]]); }
                    else Tail[std::size_t(InstOf[T.m_Col[Ord[q]]])] = Tail[std::size_t(InstOf[T.m_Col[Ord[q]]])] + T.m_W[Ord[q]];
                for (int i = 0; i < nInst; ++i) if (Lum(Tail[std::size_t(i)]) > 0) { AInst.push_back(i); AW.push_back(Tail[std::size_t(i)]); }
                First.push_back(std::uint32_t(Col.size())); AFirst.push_back(std::uint32_t(AInst.size()));
            }
            std::vector<v3> Ek(n, v3{ 0, 0, 0 }), Next(n), InstMean(static_cast<std::size_t>(nInst));
            int WithinK = -1;
            for (int it = 1; it <= 30; ++it)
            {
                std::fill(InstMean.begin(), InstMean.end(), v3{ 0, 0, 0 });
                for (std::size_t r = 0; r < n; ++r) InstMean[std::size_t(InstOf[r])] = InstMean[std::size_t(InstOf[r])] + Ek[r];
                for (int i = 0; i < nInst; ++i) InstMean[std::size_t(i)] = InstMean[std::size_t(i)] * (1.0f / float(std::max(GroupSize[std::size_t(i)], 1)));
                for (std::size_t r = 0; r < n; ++r)
                {
                    v3 Sum = T.m_D[r];
                    for (auto k = First[r]; k < First[r + 1]; ++k) Sum = Sum + v3{ Wk[k].m_X * Ek[Col[k]].m_X, Wk[k].m_Y * Ek[Col[k]].m_Y, Wk[k].m_Z * Ek[Col[k]].m_Z };
                    for (auto k = AFirst[r]; k < AFirst[r + 1]; ++k) { const v3& m = InstMean[std::size_t(AInst[k])]; Sum = Sum + v3{ AW[k].m_X * m.m_X, AW[k].m_Y * m.m_Y, AW[k].m_Z * m.m_Z }; }
                    Next[r] = Sum;
                }
                Ek.swap(Next);
                if (WithinK < 0 && Err(Ek).first < 0.01) WithinK = it;
            }
            const auto [I2, R2] = Err(Ek);
            std::printf("  top %2d links + the tail to %.1f %s means a surfel (%.1f KB): converged off the world by %.2f%% (instances), %.2f%% RMS (surfels); within 1%% after %d iterations\n",
                        K, double(AInst.size()) / double(n), bParents ? "parent" : "instance", (Col.size() + AInst.size()) * 16.0 / 1024.0, 100 * I2, 100 * R2, WithinK);
        }

        // The solver's own time to the same 1%, from scratch, every surfel every update (64 rays: the reference's)
        {
            cornell C2;
            MakeCornell(C2, 0.5f, false);
            C2.m_World.m_Setup.m_RaysPerSurfel = 64;
            int Updates = 0;
            for (; Updates < 400; ++Updates)
            {
                C2.m_World.Update(C2.m_Lights, { 0, 0, -0.9f }, 1 << 30, 900 + Updates);
                std::vector<v3> E2;
                for (int i = 0; i < C2.m_World.InstanceCount(); ++i) for (const auto& e : C2.m_World.InstanceLight(i)) E2.push_back(e);
                const double e2 = Err(E2).first;
                if (Updates + 1 == 10 || Updates + 1 == 30 || Updates + 1 == 100 || Updates + 1 == 200) std::printf("  the solver from scratch after %d updates: instances off by %.2f%%\n", Updates + 1, 100 * e2);
                if (e2 < 0.01) break;
            }
            std::printf("  the solver from scratch: within 1%% of the converged world after %d updates (64 rays a surfel each: %zu rays; the compile cast %zu once)\n",
                        Updates + 1, std::size_t(Updates + 1) * 64 * n, T.m_nRays);
        }
    }

    // 6l. GI chunks (documentation/gi_chunks_plan.md): a room as one mesh (6 x 6 m, 2.5 m high), its clusters 1 m tiles, baked whole and
    //     partitioned into chunks of about a quarter of its surfels: every surfel owned once, every chunk within the local cap, its vertex
    //     surfels local, its ghosts copies of their owners, and its reads covering its surface as the whole mesh's do
    {
        surfel_mesh M;
        AddQuad(M, { -3, 0, -3 }, { 0, 0, 6 }, { 6, 0, 0 }, 0.125f);            // floor
        AddQuad(M, { -3, 2.5f, -3 }, { 6, 0, 0 }, { 0, 0, 6 }, 0.125f);         // ceiling
        AddQuad(M, { -3, 0, 3 }, { 0, 2.5f, 0 }, { 6, 0, 0 }, 0.125f);          // back
        AddQuad(M, { -3, 0, -3 }, { 6, 0, 0 }, { 0, 2.5f, 0 }, 0.125f);         // front
        AddQuad(M, { -3, 0, -3 }, { 0, 2.5f, 0 }, { 0, 0, 6 }, 0.125f);         // left
        AddQuad(M, { 3, 0, -3 }, { 0, 0, 6 }, { 0, 2.5f, 0 }, 0.125f);          // right
        M.Bake(0.3f, 0.75f);
        M.BuildRays();
        M.BuildReadGrid();
        std::vector<int> TriCluster(M.m_Tris.size());
        std::unordered_map<std::int64_t, int> Tile;
        for (std::size_t t = 0; t < M.m_Tris.size(); ++t)
        {
            const v3 c = (M.m_Pos[M.m_Tris[t][0]] + M.m_Pos[M.m_Tris[t][1]] + M.m_Pos[M.m_Tris[t][2]]) * (1.0f / 3.0f);
            const std::int64_t k = (std::int64_t(std::floor(c.m_X)) + 100) * 1000000 + (std::int64_t(std::floor(c.m_Y)) + 100) * 1000 + (std::int64_t(std::floor(c.m_Z)) + 100);
            TriCluster[t] = Tile.try_emplace(k, int(Tile.size())).first->second;
        }
        const std::size_t nS = M.m_Levels[0].size();
        chunked_mesh Ch; chunk_report Rep;
        const bool bFits = PartitionChunks(M, TriCluster, int(Tile.size()), nS / 4, Ch, &Rep);
        std::size_t Owned = 0; for (const auto n : Ch.m_nOwned) Owned += n;
        int BadVS = 0, BadGhost = 0, EmptyGrid = 0;
        for (std::size_t c = 0; c < Ch.m_Chunks.size(); ++c)
        {
            const auto& C = Ch.m_Chunks[c];
            for (const auto& VS : C.m_VertSurfels) for (const int s : VS) if (s >= int(C.m_Levels[0].size())) ++BadVS;
            for (std::size_t k = 0; k < Ch.m_Ghosts[c].size(); ++k)
            {
                const auto& G = Ch.m_Ghosts[c][k];
                const auto& Own = Ch.m_Chunks[G.m_Chunk].m_Levels[0][G.m_Local];
                const auto& Me  = C.m_Levels[0][Ch.m_nOwned[c] + k];
                if (G.m_Chunk == c || G.m_Local >= Ch.m_nOwned[G.m_Chunk] || (Own.m_Center - Me.m_Center).Length() > 1e-6f) ++BadGhost;
            }
            if (C.m_ReadTable.empty()) ++EmptyGrid;
        }
        // the reads: on every chunk's triangles, the share of points whose read lists something, against the whole mesh's at the same points
        std::size_t nP = 0, nWhole = 0, nChunk = 0;
        for (const auto& C : Ch.m_Chunks)
            for (std::size_t t = 0; t < C.m_Tris.size(); t += 3)
            {
                const auto& T = C.m_Tris[t];
                const v3 P = (C.m_Pos[T[0]] + C.m_Pos[T[1]] + C.m_Pos[T[2]]) * (1.0f / 3.0f);
                const v3 N = (C.m_Pos[T[1]] - C.m_Pos[T[0]]).Cross(C.m_Pos[T[2]] - C.m_Pos[T[0]]).NormalizeCopy();
                ++nP; nWhole += !M.ReadCellList(P, N).empty(); nChunk += !C.ReadCellList(P, N).empty();
            }
        std::printf("GI chunks: %zu surfels, %zu clusters -> %d chunks (owned %zu..%zu, %d split again), %zu ghosts (%.0f%% of the owned; at most %zu a chunk), local at most %zu; "
                    "the reads list something at %.1f%% of %zu points (the whole mesh %.1f%%); partition %.0f ms, chunks built %.0f ms\n",
                    nS, Tile.size(), Rep.m_nChunks, Rep.m_OwnedMin, Rep.m_OwnedMax, Rep.m_nSplitsAgain, Rep.m_GhostsTotal, 100.0 * double(Rep.m_GhostsTotal) / double(nS), Rep.m_GhostsMax, Rep.m_LocalMax,
                    100.0 * double(nChunk) / double(nP), nP, 100.0 * double(nWhole) / double(nP), Rep.m_PartitionMs, Rep.m_BuildMs);
        CHECK(bFits && Rep.m_nChunks >= 4, "the room was not partitioned into fitting chunks (%d chunks, fits %d)", Rep.m_nChunks, int(bFits));
        CHECK(Owned == nS, "the chunks own %zu surfels, the mesh has %zu", Owned, nS);
        CHECK(BadVS == 0 && BadGhost == 0 && EmptyGrid == 0, "%d vertex surfels out of their chunk, %d ghosts not their owners' copies, %d chunks without a read grid", BadVS, BadGhost, EmptyGrid);
        CHECK(Rep.m_GhostsTotal > 0, "no ghosts: the borders read nothing across them");
        CHECK(nChunk + nP / 200 >= nWhole, "the chunks' reads list something at %zu points, the whole mesh's at %zu", nChunk, nWhole);

        // The light: the whole room, a second whole room (another seed: the noise floor) and the chunked room (each chunk an instance, its
        // ghosts synced from their owners), converged alike; at every chunk triangle's centre each one's read
        lights Lt; Lt.m_Points = { { { 0.5f, 2.0f, 0.3f }, { 6, 6, 6 }, 8.0f } }; Lt.m_Sky = { 0.05f, 0.05f, 0.05f };
        world Wa, Wb, Wc;
        const int Ia = Wa.Add(M, xmath::fmat4::fromIdentity(), { 0.7f, 0.7f, 0.7f }), Ib = Wb.Add(M, xmath::fmat4::fromIdentity(), { 0.7f, 0.7f, 0.7f });
        std::vector<int> ChunkId;
        for (const auto& C : Ch.m_Chunks) ChunkId.push_back(Wc.Add(C, xmath::fmat4::fromIdentity(), { 0.7f, 0.7f, 0.7f }));
        for (std::size_t c = 0; c < Ch.m_Chunks.size(); ++c)
        {
            std::vector<std::array<int, 2>> Owners;
            for (const auto& G : Ch.m_Ghosts[c]) Owners.push_back({ ChunkId[G.m_Chunk], int(G.m_Local) });
            Wc.SetGhosts(ChunkId[c], Ch.m_nOwned[c], std::move(Owners));
        }
        const auto Run = [&](world& W, unsigned Seed)
        {
            W.m_Setup.m_RaysPerSurfel = 64; W.m_Setup.m_MinBlend = 1;
            for (int u = 0; u < 8; ++u) W.Update(Lt, { 0, 1.5f, 0 }, 1 << 30, Seed + unsigned(u));
            W.m_Setup.m_MinBlend = 0;
            for (int u = 0; u < 150; ++u) W.Update(Lt, { 0, 1.5f, 0 }, 1 << 30, Seed + 100 + unsigned(u));
        };
        Run(Wa, 1000); Run(Wb, 5000); Run(Wc, 1000);
        const auto Lum = [](const v3& e) { return 0.2126 * e.m_X + 0.7152 * e.m_Y + 0.0722 * e.m_Z; };
        std::vector<double> Dab, Dac;
        double Top = 1e-9;
        for (std::size_t c = 0; c < Ch.m_Chunks.size(); ++c)
            for (std::size_t t = 0; t < Ch.m_Chunks[c].m_Tris.size(); t += 3)
            {
                const auto& C = Ch.m_Chunks[c]; const auto& T = C.m_Tris[t];
                const v3 Pp = (C.m_Pos[T[0]] + C.m_Pos[T[1]] + C.m_Pos[T[2]]) * (1.0f / 3.0f);
                const v3 Nn = (C.m_Pos[T[1]] - C.m_Pos[T[0]]).Cross(C.m_Pos[T[2]] - C.m_Pos[T[0]]).NormalizeCopy();
                const auto qa = Wa.ReadPoint(Ia, Pp, Nn), qb = Wb.ReadPoint(Ib, Pp, Nn), qc = Wc.ReadPoint(ChunkId[c], Pp, Nn);
                if (!qa.m_bSupported || !qb.m_bSupported || !qc.m_bSupported) continue;
                Top = std::max(Top, Lum(qa.m_E));
                Dab.push_back(std::abs(Lum(qa.m_E) - Lum(qb.m_E))); Dac.push_back(std::abs(Lum(qa.m_E) - Lum(qc.m_E)));
            }
        const auto Pct = [&](std::vector<double> V, double p) { std::sort(V.begin(), V.end()); return V.empty() ? 0.0 : V[std::size_t(p * double(V.size() - 1))] / Top; };
        double Ma = 0, Mc = 0;
        for (int i = 0; i < Wa.InstanceCount(); ++i) for (const auto& e : Wa.InstanceLight(i)) Ma += Lum(e);
        for (int i = 0; i < Wc.InstanceCount(); ++i) { const auto& E = Wc.InstanceLight(i); for (std::uint32_t k = 0; k < Wc.OwnedSurfels(i); ++k) Ma += 0, Mc += Lum(E[k]); }
        std::printf("GI chunks, the light at %zu common points (of the brightest): the chunked room off the whole by %.2f%% median, %.2f%% 95th; "
                    "two whole rooms (the noise) %.2f%% median, %.2f%% 95th; the owned surfels' total light %.2f%% off the whole's\n",
                    Dac.size(), 100 * Pct(Dac, 0.5), 100 * Pct(Dac, 0.95), 100 * Pct(Dab, 0.5), 100 * Pct(Dab, 0.95), 100 * std::abs(Mc - Ma) / std::max(Ma, 1e-9));
        CHECK(Pct(Dac, 0.95) < 2 * Pct(Dab, 0.95) + 0.01, "the chunked room's light is off the whole's by %.2f%% (95th), the noise %.2f%%", 100 * Pct(Dac, 0.95), 100 * Pct(Dab, 0.95));

        // The sections through a resource (xgeom_static version 7): flattened as the compiler writes them, each loaded back as the renderer
        // does (its geometry from the whole mesh, SectionGeometry), every array the same as the chunk's
        {
            const auto Flat = FlattenSections(Ch, M.m_Pos.size());
            const auto View = ViewOf(Flat);
            int nBad = 0;
            std::size_t Words = 0;
            std::vector<int> Uses(M.m_Pos.size(), 0);      // (this room's tiles share their border vertices; an xgeom_static's clusters never do)
            for (const auto& VW : Ch.m_VertWhole) for (const std::uint32_t w : VW) ++Uses[w];
            for (std::size_t c = 0; c < Ch.m_Chunks.size(); ++c)
            {
                surfel_mesh L;
                const auto& C = Ch.m_Chunks[c];
                if (!LoadSection(M, TriCluster, View, c, L)) { ++nBad; continue; }
                const auto SameS = [](const std::vector<surfel>& A, const std::vector<surfel>& B) { if (A.size() != B.size()) return false; for (std::size_t s = 0; s < A.size(); ++s) if ((A[s].m_Center - B[s].m_Center).Length() > 0 || (A[s].m_Normal - B[s].m_Normal).Length() > 0 || A[s].m_Radius != B[s].m_Radius) return false; return true; };
                bool bVS = L.m_VertSurfels.size() == C.m_VertSurfels.size();
                for (std::size_t v = 0; bVS && v < C.m_VertSurfels.size(); ++v) bVS = Uses[Ch.m_VertWhole[c][v]] > 1 || L.m_VertSurfels[v] == C.m_VertSurfels[v];
                const bool bOk = L.m_Pos.size() == C.m_Pos.size() && L.m_Tris == C.m_Tris && SameS(L.m_Levels[0], C.m_Levels[0]) && SameS(L.m_Levels[1], C.m_Levels[1])
                              && L.m_ParentOf == C.m_ParentOf && bVS && L.m_ReadTable == C.m_ReadTable && L.m_ReadList == C.m_ReadList
                              && L.m_OriginFirst == C.m_OriginFirst && L.m_OriginVert == C.m_OriginVert && L.m_Neighbors[0].m_Index == C.m_Neighbors[0].m_Index;
                nBad += !bOk;
                if (!bOk) std::printf("  section %zu: verts %d tris %d fine %d parents %d parentof %d vs %d table %d list %d origins %d/%d neighbours %d\n", c, int(L.m_Pos.size() == C.m_Pos.size()), int(L.m_Tris == C.m_Tris),
                    int(SameS(L.m_Levels[0], C.m_Levels[0])), int(SameS(L.m_Levels[1], C.m_Levels[1])), int(L.m_ParentOf == C.m_ParentOf), int(bVS), int(L.m_ReadTable == C.m_ReadTable),
                    int(L.m_ReadList == C.m_ReadList), int(L.m_OriginFirst == C.m_OriginFirst), int(L.m_OriginVert == C.m_OriginVert), int(L.m_Neighbors[0].m_Index == C.m_Neighbors[0].m_Index));
                Words += Flat.m_Sections[c].m_nDerived;
            }
            std::printf("GI sections through a resource: %zu sections, %zu fine surfels (%zu owned), %zu ghosts, %zu derived words, %zu vertices shared by two sections; loaded back: %d differ\n",
                        Flat.m_Sections.size(), Flat.m_Fine.size(), Owned, Flat.m_Ghosts.size(), Words, Flat.m_nSharedVerts, nBad);
            CHECK(nBad == 0, "%d sections load back differently from their chunks", nBad);
        }
    }

    // 6j3. The sky by direction (lights::SkyAt): a 4 x 4 octahedral map with down (-Y) at the center and the seams up (the renderer's
    //      coarsest prefiltered level): a ray leaving downward reads the center texels, one leaving upward the corners
    {
        lights L;
        L.m_Sky = v3{ 0.5f, 0.5f, 0.5f };
        CHECK(std::abs(L.SkyAt({ 0, 1, 0 }).m_X - 0.5f) < 1e-6f, "without a map the sky is its mean");
        L.m_bSkyOct = true;
        for (auto& T : L.m_SkyOct) T = v3{ 1, 1, 1 };                          // the border texels: the sky above
        for (int j = 1; j <= 2; ++j) for (int i = 1; i <= 2; ++i) L.m_SkyOct[std::size_t(j * 4 + i)] = v3{ 0.1f, 0.1f, 0.1f };   // the center: the ground below
        const float Down = L.SkyAt({ 0, -1, 0 }).m_X, Up = L.SkyAt({ 0, 1, 0 }).m_X, Side = L.SkyAt({ 1, 0, 0 }).m_X;
        std::printf("sky by direction: down %.2f, up %.2f, sideways %.2f\n", Down, Up, Side);
        CHECK(std::abs(Down - 0.1f) < 0.05f, "a ray leaving down does not read the center of the map (%.2f)", Down);
        CHECK(std::abs(Up - 1.0f) < 0.05f, "a ray leaving up does not read the map's border (%.2f)", Up);
        CHECK(Side > Down && Side < Up + 1e-3f, "the horizon is not between the ground and the sky (%.2f)", Side);
    }

    // 6j4. The shown light's saturation (setup::m_Saturation, after the filter): the colors move away from their luminance, which is kept; the
    //      solver's light is not touched (its bounces would compound it)
    {
        surfel_mesh Ground, Panel;
        AddQuad(Ground, { -2, 0, -2 }, { 0, 0, 4 }, { 4, 0, 0 }, 0.5f);
        AddQuad(Panel,  { -0.5f, 1, -0.5f }, { 1, 0, 0 }, { 0, 0, 1 }, 0.25f);    // facing down
        Ground.Bake(0.3f, 0.9f); Panel.Bake(0.3f, 0.9f);
        world D;
        const int iGround = D.Add(Ground, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        const int iPanel  = D.Add(Panel, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        D.SetSurfelMaterial(iPanel, {}, std::vector<v3>(Panel.m_Levels[0].size(), v3{ 2, 1, 0.5f }));
        lights Dark;
        for (int f = 0; f < 60; ++f) D.Update(Dark, { 0, 1, 0 }, 1 << 30, 17000 + f);
        std::size_t s0 = 0; float Best = 1e30f;
        for (std::size_t s = 0; s < Ground.m_Levels[0].size(); ++s) { const float d = Ground.m_Levels[0][s].m_Center.Length(); if (d < Best) { Best = d; s0 = s; } }
        const auto Lum = [](const v3& c) { return 0.2126f * c.m_X + 0.7152f * c.m_Y + 0.0722f * c.m_Z; };
        const v3   Plain = D.BasisLight(iGround)[s0][0];
        const v3   Solved = D.InstanceLight(iGround)[s0];
        D.m_Setup.m_Saturation = 2;
        const v3   Sat = D.BasisLight(iGround)[s0][0];
        const float ChromaPlain = Plain.m_X - Lum(Plain), ChromaSat = Sat.m_X - Lum(Sat);
        std::printf("saturation 1 -> 2 under the panel: luminance %.3f -> %.3f, red above its luminance %.3f -> %.3f\n", Lum(Plain), Lum(Sat), ChromaPlain, ChromaSat);
        CHECK(std::abs(Lum(Sat) - Lum(Plain)) < 0.02f * std::max(Lum(Plain), 1e-3f), "the saturation changed the luminance (%.3f -> %.3f)", Lum(Plain), Lum(Sat));
        CHECK(ChromaPlain > 0 && std::abs(ChromaSat - 2 * ChromaPlain) < 0.05f * ChromaPlain, "the saturation did not double the color (%.3f -> %.3f)", ChromaPlain, ChromaSat);
        CHECK(std::abs(D.InstanceLight(iGround)[s0].m_X - Solved.m_X) < 1e-9f, "the saturation reached the solver's light");
    }

    // 6j5. Buried surfels are traced one pass in dead_retrace_v: a sheet inside a closed box (15 cm under its top, the box's back faces over
    //      it) gets validity 0 and stops taking rays, the box's faces keep theirs; every dead_retrace_v-th pass traces all of them again (it can come back)
    {
        surfel_mesh M;
        AddQuad(M, { -0.5f, 0.5f, -0.5f }, { 0, 0, 1 }, { 1, 0, 0 }, 0.5f);       // top (up)
        AddQuad(M, { -0.5f, -0.5f, -0.5f }, { 1, 0, 0 }, { 0, 0, 1 }, 0.5f);      // bottom (down)
        AddQuad(M, { -0.5f, -0.5f, -0.5f }, { 0, 1, 0 }, { 0, 0, 1 }, 0.5f);      // -x
        AddQuad(M, { 0.5f, -0.5f, -0.5f }, { 0, 0, 1 }, { 0, 1, 0 }, 0.5f);       // +x
        AddQuad(M, { -0.5f, -0.5f, -0.5f }, { 1, 0, 0 }, { 0, 1, 0 }, 0.5f);      // -z
        AddQuad(M, { -0.5f, -0.5f, 0.5f }, { 0, 1, 0 }, { 1, 0, 0 }, 0.5f);       // +z
        AddQuad(M, { -0.3f, 0.35f, -0.3f }, { 0, 0, 0.6f }, { 0.6f, 0, 0 }, 0.3f);      // the sheet, facing up
        M.Bake(0.3f, 0.9f);
        world W;
        const int i = W.Add(M, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        lights L;
        L.m_Sky = v3{ 1, 1, 1 };
        for (int f = 0; f < 100; ++f) W.Update(L, { 0, 3, 0 }, 1 << 30, 18000 + f);       // settled: the warm-up passes are over, the sheet's validity has gone
        std::set<int> Seen;
        std::size_t Min = 1u << 30;
        for (int f = 0; f < dead_retrace_v; ++f)
        {
            W.Update(L, { 0, 3, 0 }, 1 << 30, 18100 + f);
            Min = std::min(Min, W.LastUpdated().size());
            for (const auto& [inst, sf] : W.LastUpdated()) Seen.insert(sf);
        }
        std::printf("dead surfels: a pass traces at least %zu of %zu surfels, %d passes trace %zu of them\n", Min, M.m_Levels[0].size(), dead_retrace_v, Seen.size());
        CHECK(Min < M.m_Levels[0].size(), "buried surfels are traced every pass (%zu of %zu)", Min, M.m_Levels[0].size());
        CHECK(Seen.size() == M.m_Levels[0].size(), "%d passes do not trace every surfel (%zu of %zu)", dead_retrace_v, Seen.size(), M.m_Levels[0].size());
        CHECK(Min >= M.m_Levels[0].size() / 2, "more than half the surfels were skipped as dead (%zu of %zu)", Min, M.m_Levels[0].size());
    }

    // 6j6. A surfel far from the eye (in its radii) is traced every 2nd, 4th, 8th pass (Update): a 30 m strip with the eye at one end. The near
    //      end is traced every pass, the far end less, and a few passes together reach every surfel
    {
        surfel_mesh M;
        AddQuad(M, { 0, 0, 0 }, { 0, 0, 30 }, { 1, 0, 0 }, 1);          // facing up
        M.Bake(0.3f, 0.9f);
        world W;
        const int i = W.Add(M, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        lights L;
        L.m_Sky = v3{ 1, 1, 1 };
        const v3 Eye{ 0.5f, 2, 0 };
        for (int f = 0; f < 100; ++f) W.Update(L, Eye, 1 << 30, 19000 + f);      // settled
        std::vector<int> Count(M.m_Levels[0].size(), 0);
        for (int f = 0; f < 8; ++f)
        {
            W.Update(L, Eye, 1 << 30, 19100 + f);
            for (const auto& [inst, sf] : W.LastUpdated()) ++Count[std::size_t(sf)];
        }
        int NearMin = 8, FarMax = 0, nNear = 0, nFar = 0;
        for (std::size_t sf = 0; sf < Count.size(); ++sf)
        {
            const float d = (M.m_Levels[0][sf].m_Center - Eye).Length() / M.m_Levels[0][sf].m_Radius;
            if (d < 10) { NearMin = std::min(NearMin, Count[sf]); ++nNear; }
            if (d > 55) { FarMax = std::max(FarMax, Count[sf]); ++nFar; }
        }
        int Zero = 0; for (int c : Count) Zero += c == 0;
        std::printf("distance tiers: %d near surfels traced at least %d of 8 passes, %d far ones at most %d; %d never\n", nNear, NearMin, nFar, FarMax, Zero);
        CHECK(nNear > 0 && nFar > 0, "the strip has no near or far surfels");
        CHECK(NearMin == 8, "a near surfel is not traced every pass (%d of 8)", NearMin);
        CHECK(FarMax <= 2, "a far surfel is traced %d of 8 passes", FarMax);
        CHECK(Zero == 0, "%d surfels were never traced in 8 passes", Zero);
    }

    // 6j7. The level by the screen area (setup::m_ProjScale, m_PixelsPerSurfel, Phase 2b's one rule): the pixels a fine surfel covers where the
    //      instance is nearest the eye. A small box holds its parent level where they cover too few, the fine one nearer, and it comes back at
    //      1.25 times the area it left at (no flicker)
    {
        surfel_mesh Box;
        AddQuad(Box, { -0.1f, 0.2f, -0.1f }, { 0, 0, 0.2f }, { 0.2f, 0, 0 }, 0.1f);       // a 20 cm box, six faces
        AddQuad(Box, { -0.1f, 0, -0.1f }, { 0.2f, 0, 0 }, { 0, 0, 0.2f }, 0.1f);
        AddQuad(Box, { -0.1f, 0, -0.1f }, { 0, 0.2f, 0 }, { 0, 0, 0.2f }, 0.1f);
        AddQuad(Box, { 0.1f, 0, -0.1f }, { 0, 0, 0.2f }, { 0, 0.2f, 0 }, 0.1f);
        AddQuad(Box, { -0.1f, 0, -0.1f }, { 0.2f, 0, 0 }, { 0, 0.2f, 0 }, 0.1f);
        AddQuad(Box, { -0.1f, 0, 0.1f }, { 0, 0.2f, 0 }, { 0.2f, 0, 0 }, 0.1f);
        Box.Bake(0.05f, 0.15f);
        world W;
        const int i = W.Add(Box, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        W.m_Setup.m_ProjScale = 1000;                     // 1000 pixels per unit at distance 1: a 1080 p view of about 56 degrees
        W.m_Setup.m_PixelsPerSurfel = 256;
        lights L;
        L.m_Sky = v3{ 1, 1, 1 };
        const float r0     = Box.m_Levels[0][0].m_Radius;
        const float dStar  = r0 * 1000.0f * std::sqrt(pi_v / 256.0f);                  // the distance (from the near face) where a fine surfel covers 256 pixels
        const auto LevelAt = [&](float x) { W.Update(L, { 0, 0.1f, 0.1f + x * dStar }, 1 << 20, 20000); return W.Level(i); };
        const int Near = LevelAt(0.5f), Far = LevelAt(2.0f), Between = LevelAt(0.95f), Back = LevelAt(0.5f);
        std::printf("surfel level by screen area (%zu fine surfels, leaves at %.2f m): near %d, far %d, then inside the hysteresis %d, near again %d\n", Box.m_Levels[0].size(), dStar, Near, Far, Between, Back);
        CHECK(Near == 0, "a box covering many pixels is not at the fine level (%d)", Near);
        CHECK(Far == 1, "a box covering few pixels is not at the parent level (%d)", Far);
        CHECK(Between == 1, "the level flickered back inside the hysteresis (%d)", Between);
        CHECK(Back == 0, "the box did not come back to the fine level when near (%d)", Back);
    }

    // 6j7b. The level rule on a large instance (a 40 m floor and wall): with the eye inside its box, or near enough that it fills the view,
    //       it keeps its fine surfels; at 26 m its fine surfels are too small and it takes its parents, but never the ambient cube while
    //       they cover enough (Phase 0 found the Police building held as a 6-sample cube); 2.5 km off, its parents
    {
        surfel_mesh Big;
        AddQuad(Big, { -20, 0, -20 }, { 0, 0, 40 }, { 40, 0, 0 }, 1.0f);              // a 40 m floor ...
        AddQuad(Big, { -20, 0, -20 }, { 40, 0, 0 }, { 0, 8, 0 }, 1.0f);               // ... and a wall
        Big.Bake(0.25f, 0.75f);
        world W;
        const int i = W.Add(Big, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        W.m_Setup.m_ProjScale = 547; W.m_Setup.m_PixelsPerSurfel = 256;               // a 450 x 631 view
        lights L; L.m_Sky = v3{ 1, 1, 1 };
        for (int f = 0; f < 4; ++f) W.Update(L, { 0, 2, 0 }, 1 << 16, 24000 + f);         // the eye inside its box
        const int Inside = W.Level(i); const bool bInsideCube = W.IsCube(i);
        for (int f = 0; f < 4; ++f) W.Update(L, { 0, 10, 22 }, 1 << 16, 24100 + f);       // outside, 3 m from it: it fills the view
        const int Near = W.Level(i); const bool bNearCube = W.IsCube(i);
        for (int f = 0; f < 4; ++f) W.Update(L, { 0, 15, 45 }, 1 << 16, 24150 + f);       // 26 m off: the fine surfels cover ~90 pixels, the parents ~800
        const int Mid = W.Level(i); const bool bMidCube = W.IsCube(i);
        for (int f = 0; f < 4; ++f) W.Update(L, { 0, 30, 2500 }, 1 << 16, 24200 + f);     // 2.5 km off: small on the screen (beyond m_LodDistance too)
        const int Far = W.Level(i);
        std::printf("large instance (%zu surfels): eye inside -> level %d%s, filling the view -> level %d%s, 26 m off -> level %d%s, 2.5 km off -> level %d\n", Big.m_Levels[0].size(),
                    Inside, bInsideCube ? " (cube)" : "", Near, bNearCube ? " (cube)" : "", Mid, bMidCube ? " (cube)" : "", Far);
        CHECK(Inside == 0 && !bInsideCube, "a large instance around the eye left its fine surfels (level %d, cube %d)", Inside, int(bInsideCube));
        CHECK(Near == 0 && !bNearCube, "a large instance filling the view left its fine surfels (level %d, cube %d)", Near, int(bNearCube));
        CHECK(Mid == 1 && !bMidCube, "a large instance 26 m off is not at its parents (level %d, cube %d)", Mid, int(bMidCube));
        CHECK(Far == 1, "a large instance far off did not take its parents (%d)", Far);
    }

    // 6j8. The ambient cube (setup::m_bCube): an instance that covers few pixels for its parents traces 6 of them, one per axis direction, and
    //      blends the others from those by the way they face. A 40 cm box under a bright sky over a dark ground: a pass traces at most 6, and the light on the top and
    //      the bottom of the box is the one the full parents give (within a third), the top lit and the bottom dark
    {
        const auto Make = []()
        {
            surfel_mesh Box;
            AddQuad(Box, { -0.2f, 0.4f, -0.2f }, { 0, 0, 0.4f }, { 0.4f, 0, 0 }, 0.1f);       // top (up)
            AddQuad(Box, { -0.2f, 0, -0.2f }, { 0.4f, 0, 0 }, { 0, 0, 0.4f }, 0.1f);          // bottom (down)
            AddQuad(Box, { -0.2f, 0, -0.2f }, { 0, 0.4f, 0 }, { 0, 0, 0.4f }, 0.1f);          // -x
            AddQuad(Box, { 0.2f, 0, -0.2f }, { 0, 0, 0.4f }, { 0, 0.4f, 0 }, 0.1f);           // +x
            AddQuad(Box, { -0.2f, 0, -0.2f }, { 0.4f, 0, 0 }, { 0, 0.4f, 0 }, 0.1f);          // -z
            AddQuad(Box, { -0.2f, 0, 0.2f }, { 0, 0.4f, 0 }, { 0.4f, 0, 0 }, 0.1f);           // +z
            Box.Bake(0.05f, 0.15f);
            return Box;
        };
        const surfel_mesh Box = Make();
        const auto Run = [&](bool bCube, double& Top, double& Bottom, std::size_t& MaxTraced, bool& bIsCube)
        {
            world W;
            W.m_Setup.m_ProjScale = 1000; W.m_Setup.m_PixelsPerSurfel = 256; W.m_Setup.m_bCube = bCube;
            const int i = W.Add(Box, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            lights L;
            L.m_Sky = v3{ 0.5f, 0.5f, 0.5f }; L.m_bSkyOct = true;                                // the sky above 1, the ground below 0.1 (the center of the map)
            for (auto& T : L.m_SkyOct) T = v3{ 1, 1, 1 };
            for (int j = 1; j <= 2; ++j) for (int k = 1; k <= 2; ++k) L.m_SkyOct[std::size_t(j * 4 + k)] = v3{ 0.1f, 0.1f, 0.1f };
            const v3 Eye{ 0, 0.2f, 40 };                         // far: it covers a few hundred pixels
            for (int f = 0; f < 300; ++f) W.Update(L, Eye, 1 << 30, 21000 + f);
            MaxTraced = 0;
            for (int f = 0; f < 16; ++f) { W.Update(L, Eye, 1 << 30, 21400 + f); MaxTraced = std::max(MaxTraced, W.LastUpdated().size()); }
            bIsCube = W.IsCube(i);
            const auto& E = W.InstanceLight(i);
            double St = 0, Sb = 0; int nt = 0, nb = 0;
            for (std::size_t p = 0; p < E.size() && p < Box.m_Levels[1].size(); ++p)
            {
                const float ny = Box.m_Levels[1][p].m_Normal.m_Y;
                if (ny > 0.9f)  { St += E[p].m_X; ++nt; }
                if (ny < -0.9f) { Sb += E[p].m_X; ++nb; }
            }
            Top = St / std::max(nt, 1); Bottom = Sb / std::max(nb, 1);
        };
        double TopFull, BotFull, TopCube, BotCube; std::size_t MaxFull, MaxCube; bool bFullIsCube, bCubeIsCube;
        Run(false, TopFull, BotFull, MaxFull, bFullIsCube);
        Run(true,  TopCube, BotCube, MaxCube, bCubeIsCube);
        std::printf("ambient cube (%zu parents): a pass traces %zu (all parents: %zu); the top %.3f vs %.3f, the bottom %.3f vs %.3f\n", Box.m_Levels[1].size(), MaxCube, MaxFull, TopCube, TopFull, BotCube, BotFull);
        CHECK(bCubeIsCube && !bFullIsCube, "the box is not (only when asked) an ambient cube");
        CHECK(MaxCube <= 6, "a cube pass traces %zu surfels", MaxCube);
        CHECK(MaxFull > MaxCube, "the cube traces no fewer surfels (%zu vs %zu)", MaxCube, MaxFull);
        CHECK(std::abs(TopCube - TopFull) < 0.33 * TopFull, "the top's light is off: %.3f vs %.3f", TopCube, TopFull);
        CHECK(TopCube > 2 * BotCube, "the top is not lit over the bottom (%.3f vs %.3f)", TopCube, BotCube);
    }

    // 6j9. Phase 2: the emitters' links are maintained in bounded chunks. A sign (an emitter) over a floor; the floor moves every update for 30
    //      updates under a 1 ms allowance: the link work of each update stays within its share plus one chunk, links are published while it
    //      moves (not only after), and once it stops the links equal a fresh build of the final scene
    {
        surfel_mesh Ground, Sign;
        AddQuad(Ground, { -2, 0, -2 }, { 0, 0, 4 }, { 4, 0, 0 }, 0.5f);
        AddQuad(Sign,   { -0.1f, 1, -0.1f }, { 0.2f, 0, 0 }, { 0, 0, 0.2f }, 0.1f);       // facing down
        Ground.Bake(0.1f, 0.3f); Sign.Bake(0.05f, 0.15f);
        const auto Make = [&](world& W, int& iG, int& iS)
        {
            iG = W.Add(Ground, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            iS = W.Add(Sign,   xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            W.SetSurfelMaterial(iS, {}, std::vector<v3>(Sign.m_Levels[0].size(), v3{ 20, 20, 20 }));
        };
        world W; int iGround, iSign; Make(W, iGround, iSign);
        W.m_Setup.m_TraceAllowanceMs = 1.0f;
        lights Dark;
        for (int f = 0; f < 40; ++f) W.Update(Dark, { 0, 1, 0 }, 1 << 30, 22000 + f);
        const int Before = W.LinkRebuilds();
        double WorstOver = -1e9; int MaxPending = 0;
        for (int f = 0; f < 30; ++f)
        {
            W.SetTransform(iGround, xmath::fmat4::fromTranslation({ 0.01f * float(f), 0, 0 }));
            const auto St = W.Update(Dark, { 0, 1, 0 }, 1 << 30, 22100 + f);
            WorstOver  = std::max(WorstOver, St.m_LinksMs - 1.0);         // its share, or all of it when the trace waits for nothing
            MaxPending = std::max(MaxPending, St.m_LinkPending);
        }
        const int During = W.LinkRebuilds() - Before;
        for (int f = 0; f < 60; ++f) W.Update(Dark, { 0, 1, 0 }, 1 << 30, 22200 + f);
        world Ref; int rG, rS; Make(Ref, rG, rS);
        Ref.SetTransform(rG, xmath::fmat4::fromTranslation({ 0.01f * 29.0f, 0, 0 }));
        Ref.Update(Dark, { 0, 1, 0 }, 1 << 30, 22300);                              // no allowance: every link now
        std::printf("links while the floor moves for 30 updates: %d published, the link work at most %.2f ms over the allowance, %d receivers pending at most; "
                    "after: %d links (a fresh build: %d)\n", During, WorstOver, MaxPending, W.LinkCount(iGround), Ref.LinkCount(rG));
        CHECK(During > 0, "no links were published while the floor moved");
        CHECK(WorstOver < 1.0, "an update's link work took %.2f ms more than the allowance", WorstOver);     // (one chunk, and the workers' wake-up in Debug)
        CHECK(MaxPending <= 2, "%d receivers pending (2 instances)", MaxPending);
        CHECK(W.LinkCount(iGround) > 0 && W.LinksDigest(iGround) == Ref.LinksDigest(rG), "the settled links differ from a fresh build (%d vs %d links)", W.LinkCount(iGround), Ref.LinkCount(rG));
    }

    // 6j9b. Phase 2, test C of the plan: no stale emitter connection survives a closed door. A lamp on one side of a doorway, a floor on the
    //       other; the door (a wall that does not emit) slides shut: the floor's links to the lamp are gone and its emitted light is 0. It
    //       opens again: they come back. The lamp and the floor never move; only the occluder's box marks them
    {
        surfel_mesh Floor, Lamp, Door;
        AddQuad(Floor, { 1, 0, -1 }, { 0, 0, 2 }, { 2, 0, 0 }, 0.25f);                    // x 1..3, facing up
        AddQuad(Lamp,  { -1, 0.5f, -0.5f }, { 0, 1, 0 }, { 0, 0, 1 }, 0.25f);              // x = -1, facing +x
        AddQuad(Door,  { 0, -1, -3 }, { 0, 0, 6 }, { 0, 4, 0 }, 0.5f);                     // the plane x = 0, big enough to hide the lamp
        Floor.Bake(0.15f, 0.45f); Lamp.Bake(0.15f, 0.45f); Door.Bake(0.3f, 0.9f);
        world W;
        const int iFloor = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        const int iLamp  = W.Add(Lamp,  xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        const int iDoor  = W.Add(Door,  xmath::fmat4::fromTranslation({ 0, 0, 10 }), { 0, 0, 0 });   // open: far to the side (black: no bounce)
        W.SetSurfelMaterial(iLamp, {}, std::vector<v3>(Lamp.m_Levels[0].size(), v3{ 20, 20, 20 }));
        W.m_Setup.m_TraceAllowanceMs = 1.0f;
        lights Dark;
        const auto Run = [&](int n, unsigned Seed) { for (int f = 0; f < n; ++f) W.Update(Dark, { 2, 1, 0 }, 1 << 30, Seed + f); };
        const auto Emitted = [&] { return W.InstanceEmitted(iFloor); };
        Run(60, 22400);
        const int OpenLinks = W.LinkCount(iFloor); const double OpenLight = Emitted();
        W.SetTransform(iDoor, xmath::fmat4::fromIdentity());                         // shut
        Run(200, 22500);
        const int ShutLinks = W.LinkCount(iFloor); const double ShutLight = Emitted();
        W.SetTransform(iDoor, xmath::fmat4::fromTranslation({ 0, 0, 10 }));          // open again
        Run(60, 22800);
        const int ReopenLinks = W.LinkCount(iFloor); const double ReopenLight = Emitted();
        std::printf("a door between a lamp and a floor: %d links open (emitted light %.3f), %d shut (%.4f), %d open again (%.3f)\n", OpenLinks, OpenLight, ShutLinks, ShutLight, ReopenLinks, ReopenLight);
        CHECK(OpenLinks > 0, "the floor has no links to the lamp with the door open");
        CHECK(ShutLinks == 0, "%d links survive the closed door", ShutLinks);
        CHECK(OpenLight > 0 && ShutLight == 0, "the floor still gets the lamp's light through the closed door (%.4f; open %.3f)", ShutLight, OpenLight);
        CHECK(ReopenLinks == OpenLinks && std::abs(ReopenLight - OpenLight) < 1e-3 * OpenLight, "the links did not come back when the door opened (%d vs %d)", ReopenLinks, OpenLinks);
    }

    // 6j10. Phase 0 instrumentation (documentation/gi_dynamic_master_plan.md): a wall dragged over a floor lit by a sign (an emitter) for 20
    //       updates, then left still. The stage times add up to the update's total, the moves and what they mark are counted, the link
    //       rebuilds show up with their time, and the trace's longest unit is reported. Measuring only: the light is the same as before
    {
        surfel_mesh Floor, Wall, Sign;
        AddQuad(Floor, { -2, 0, -2 }, { 0, 0, 4 }, { 4, 0, 0 }, 0.5f);
        AddQuad(Wall,  { -1, 0, 0 }, { 0, 1.5f, 0 }, { 2, 0, 0 }, 0.5f);
        AddQuad(Sign,  { -0.2f, 2, -0.2f }, { 0.4f, 0, 0 }, { 0, 0, 0.4f }, 0.2f);       // facing down
        Floor.Bake(0.3f, 0.9f); Wall.Bake(0.3f, 0.9f); Sign.Bake(0.15f, 0.45f);
        world W;
        W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        const int iWall = W.Add(Wall, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        const int iSign = W.Add(Sign, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        W.SetSurfelMaterial(iSign, {}, std::vector<v3>(Sign.m_Levels[0].size(), v3{ 10, 10, 10 }));
        lights L; L.m_Sky = v3{ 0.2f, 0.2f, 0.2f };
        for (int f = 0; f < 30; ++f) W.Update(L, { 0, 1, 3 }, 1 << 14, 23000 + f);
        int Moves = 0, Marked = 0, Resets = 0, Rebuilds = 0, Hits = 0; double MaxLinks = 0, MaxUnit = 0, MaxGap = 0, MaxTotal = 0;
        std::size_t LinkBytes = 0, LightBytes = 0;
        for (int f = 0; f < 40; ++f)
        {
            if (f < 20) W.SetTransform(iWall, xmath::fmat4::fromTranslation({ 0.05f * float(f), 0, 0 }));
            const auto St = W.Update(L, { 0, 1, 3 }, 1 << 14, 23100 + f);
            Moves += St.m_nMoves; Marked += St.m_nMarkedDirty; Resets += St.m_nHistoryResets; Rebuilds += St.m_nLinkRebuilds; Hits += int(St.m_Hits);
            MaxLinks = std::max(MaxLinks, St.m_LinksMs); MaxUnit = std::max(MaxUnit, St.m_LongestSurfelUs); MaxTotal = std::max(MaxTotal, St.m_TotalMs);
            const double Sum = St.m_TlasLightsMs + St.m_LevelsMs + St.m_PickMs + St.m_LinksMs + St.m_EmitMs + St.m_MeanMs + St.m_TraceMs + St.m_CommitMs + St.m_CubeMs + St.m_StatsMs;
            MaxGap = std::max(MaxGap, St.m_TotalMs - Sum);
            LinkBytes = St.m_LinkBytes; LightBytes = St.m_LightBytes;
            CHECK(Sum <= St.m_TotalMs * 1.02 + 0.05, "the stage times (%.3f ms) exceed the update's total (%.3f ms)", Sum, St.m_TotalMs);
        }
        std::printf("instrumentation, a wall dragged 20 updates: %d moves marked %d instances dirty, %d history resets, %d link rebuilds (the slowest %.2f ms), "
                    "longest surfel %.1f us, %d hits, slowest update %.2f ms (%.3f ms not in a stage), links %zu B, light state %zu B\n",
                    Moves, Marked, Resets, Rebuilds, MaxLinks, MaxUnit, Hits, MaxTotal, MaxGap, LinkBytes, LightBytes);
        CHECK(Moves == 20, "%d moves counted (20 were made)", Moves);
        CHECK(Marked > 0 && Resets > 0, "the moves marked nothing dirty (%d) or reset no history (%d)", Marked, Resets);
        CHECK(Rebuilds > 0, "no link rebuild was counted (the moved wall's and the floor's links, Phase 2)");
        CHECK(MaxUnit > 0 && Hits > 0 && LinkBytes > 0 && LightBytes > 0, "a counter stayed at 0");
    }

    // 6j11. Phase 1: the trace under a time allowance, in units of rays. A wall moves on every update (a change faster than the allowance can
    //       follow), a box far away waits with a low score. Each update's trace stays within the allowance plus one unit, the pending work
    //       stays bounded by one pick, the far box is still served (fairness), and after the moving stops the light converges to the same
    //       scene traced without an allowance
    {
        surfel_mesh Floor, Wall, Box;
        AddQuad(Floor, { -3, 0, -3 }, { 0, 0, 6 }, { 6, 0, 0 }, 0.5f);
        AddQuad(Wall,  { -1, 0, 0 }, { 0, 1.5f, 0 }, { 2, 0, 0 }, 0.5f);
        AddQuad(Box,   { 40, 0, 40 }, { 0, 0, 1 }, { 1, 0, 0 }, 0.5f);
        Floor.Bake(0.15f, 0.45f); Wall.Bake(0.15f, 0.45f); Box.Bake(0.15f, 0.45f);
        lights L; L.m_SunDirection = v3{ 0.3f, -1, 0.2f }.NormalizeCopy(); L.m_SunColor = v3{ 2, 2, 2 }; L.m_Sky = v3{ 0.3f, 0.3f, 0.3f };
        const v3 Eye{ 0, 1, 4 };
        const auto Make = [&](world& W, int& iFloor, int& iWall, int& iBox)
        {
            iFloor = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
            iWall  = W.Add(Wall,  xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
            iBox   = W.Add(Box,   xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        };
        world W; int iF, iW, iB; Make(W, iF, iW, iB);
        W.m_Setup.m_TraceAllowanceMs = 1.0f;
        W.m_Setup.m_UnitTargetUs     = 60.0f;          // small units: a surfel's rays in several chunks
        const int Budget = 1 << 12;
        double WorstOver = -1e9; int MaxPending = 0, MinChunk = 1 << 30, BoxServed = 0, MaxRounds = 0; double LongestUnit = 0;
        for (int f = 0; f < 60; ++f)
        {
            W.SetTransform(iW, xmath::fmat4::fromTranslation({ 0.03f * float(f % 20), 0, 0 }));
            const auto St = W.Update(L, Eye, Budget, 25000 + f);
            WorstOver   = std::max(WorstOver, St.m_TraceMs - (1.0 + St.m_LongestSurfelUs * 1e-3));
            LongestUnit = std::max(LongestUnit, St.m_LongestSurfelUs);
            MaxPending  = std::max(MaxPending, St.m_Pending);
            MinChunk    = std::min(MinChunk, St.m_ChunkRays);
            MaxRounds   = std::max(MaxRounds, St.m_Rounds);
            for (const auto& [inst, sf] : W.LastUpdated()) BoxServed += inst == iB;
        }
        for (int f = 0; f < 400; ++f) W.Update(L, Eye, Budget, 25100 + f);          // still: it converges
        world Ref; int rF, rW, rB; Make(Ref, rF, rW, rB);                          // the same final scene without an allowance
        Ref.SetTransform(rW, xmath::fmat4::fromTranslation({ 0.03f * float(59 % 20), 0, 0 }));
        for (int f = 0; f < 200; ++f) Ref.Update(L, Eye, Budget, 25600 + f);
        const auto Mean = [](const world& Wd, int Id) { double m = 0; const auto& E = Wd.InstanceLight(Id); for (const auto& e : E) m += e.m_X; return m / double(std::max<std::size_t>(E.size(), 1)); };
        const double Floor1 = Mean(W, iF), Floor0 = Mean(Ref, rF);
        std::printf("trace allowance 1 ms, units of %d rays (%.0f us at most): worst trace %.2f ms over the allowance + one unit, %d pending at most (a pick is %d), "
                    "%d rounds at most, the far box served %d times while the wall moved; floor light %.3f vs %.3f without an allowance\n",
                    MinChunk, LongestUnit, WorstOver, MaxPending, Budget / W.m_Setup.m_RaysPerSurfel, MaxRounds, BoxServed, Floor1, Floor0);
        CHECK(MinChunk < W.m_Setup.m_RaysPerSurfel, "a surfel's rays were never split into units (%d rays a unit)", MinChunk);
        CHECK(WorstOver < 1.0, "an update's trace took %.2f ms more than the allowance plus one unit", WorstOver);     // (1 ms: the workers' wake-up in Debug)
        CHECK(MaxPending <= Budget / W.m_Setup.m_RaysPerSurfel, "pending work grew past one pick (%d)", MaxPending);
        CHECK(BoxServed > 0, "the far box was never served while the wall moved (fairness)");
        CHECK(std::abs(Floor1 - Floor0) < 0.1 * std::max(Floor0, 1e-3), "the floor did not converge to the scene without an allowance (%.3f vs %.3f)", Floor1, Floor0);
    }

    // 6j12. Phase 3: the shown light is kept up to date incrementally (only the changed surfels and their dependency halo, one ring of
    //       neighbors a filter pass), in slices of its allowance (a backlog waits for the next update). A wall moves over a lit floor, then
    //       rests: once the backlog drained, the shown light equals the whole filter made again, and a settled update (8 surfels finished)
    //       makes again only those and their halo, not the whole floor
    {
        surfel_mesh Floor, Wall;
        AddQuad(Floor, { -6, 0, -6 }, { 0, 0, 12 }, { 12, 0, 0 }, 0.5f);
        AddQuad(Wall,  { -1, 0, 0 }, { 0, 1.5f, 0 }, { 2, 0, 0 }, 0.5f);
        Floor.Bake(0.15f, 0.45f); Wall.Bake(0.15f, 0.45f);
        lights L; L.m_SunDirection = v3{ 0.3f, -1, 0.2f }.NormalizeCopy(); L.m_SunColor = v3{ 2, 2, 2 }; L.m_Sky = v3{ 0.3f, 0.3f, 0.3f };
        world W;
        const int iF = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        const int iW = W.Add(Wall,  xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        W.m_Setup.m_FilterPasses = 4; W.m_Setup.m_Saturation = 1.3f;
        W.m_Setup.m_TraceAllowanceMs = 0.05f;                                  // a tiny allowance: the shown light's backlog carries over while the wall moves
        double Worst = 0; std::size_t MaxChanged = 0;
        for (int f = 0; f < 160; ++f)
        {
            if (f < 30) W.SetTransform(iW, xmath::fmat4::fromTranslation({ 0.03f * float(f), 0, 0 }));
            W.Update(L, { 0, 1, 4 }, f < 30 ? 1 << 10 : 64, 26000 + f);
            if (f >= 40) MaxChanged = std::max(MaxChanged, W.ShownChanged(iF).size());
            if (f % 10 != 9 || f < 139) continue;
            W.m_Setup.m_TraceAllowanceMs = 0; W.Update(L, { 0, 1, 4 }, 8, 26500 + f);     // drained without a limit (under 0.05 ms how far it got is timing): then compared
            W.m_Setup.m_TraceAllowanceMs = 0.05f;
            const std::vector<basis> Inc = W.BasisLight(iF);
            W.ShowAllAgain();
            const auto& Full = W.BasisLight(iF);
            for (std::size_t k = 0; k < Full.size(); ++k) for (int i = 0; i < 3; ++i) Worst = std::max(Worst, double((Full[k][i] - Inc[k][i]).Length()));
        }
        std::printf("shown light kept incrementally: worst difference to the whole filter %.2e; a settled update made again %zu of %zu floor surfels at most\n",
                    Worst, MaxChanged, W.InstanceLight(iF).size());
        CHECK(Worst < 1e-5, "the incremental shown light differs from the whole filter (%.2e)", Worst);
        CHECK(MaxChanged > 0 && MaxChanged < W.InstanceLight(iF).size() / 2, "a settled update made %zu floor surfels again (of %zu)", MaxChanged, W.InstanceLight(iF).size());
    }

    // 6j19. Part II, G3: the GPU mode's plumbing on the CPU (TraceReference as the GPU: GPUWork, DeliverGPU; the update that commits a
    //       batch picks the next), and the shown light's deadline when the frame's GI work outside Update (m_OverheadMs: the GPU's part,
    //       the uploads) takes the whole allowance: the filter gets the token's block an update, not all of it (it was no limit at all)
    {
        surfel_mesh Floor;
        AddQuad(Floor, { -12, 0, -12 }, { 0, 0, 24 }, { 24, 0, 0 }, 0.5f);
        Floor.Bake(0.15f, 0.45f);
        lights L; L.m_SunDirection = v3{ 0.3f, -1, 0.2f }.NormalizeCopy(); L.m_SunColor = v3{ 2, 2, 2 }; L.m_Sky = v3{ 0.3f, 0.3f, 0.3f };
        world W;
        const int iF = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        W.m_Setup.m_bGPUTrace = true; W.m_Setup.m_TraceAllowanceMs = 0.2f; W.m_Setup.m_OverheadMs = 5;
        const std::size_t n = W.InstanceLight(iF).size();
        std::size_t MaxShown = 0; int nCommitted = 0, nUpdates = 0;
        for (int f = 0; f < 10; ++f)
        {
            const auto St = W.Update(L, { 0, 1, 4 }, 1 << 30, 27000 + f);
            nCommitted += St.m_Surfels; ++nUpdates;
            if (f > 0) MaxShown = std::max(MaxShown, W.ShownChanged(iF).size());
            const auto Work = W.GPUWork();
            const std::vector<world::gpu_item> Items(Work.begin(), Work.end());
            std::vector<world::gpu_result> R;
            W.TraceReference(Items, L, R);
            W.DeliverGPU(Items, R);
        }
        std::printf("GPU mode on the CPU: %d surfels committed in %d updates (%zu a pass); the overhead taking the allowance, an update showed %zu at most\n", nCommitted, nUpdates, n, MaxShown);
        CHECK(nCommitted == int(9 * n), "GPU mode: %d surfels committed, %zu expected (9 batches of the whole floor)", nCommitted, 9 * n);
        CHECK(MaxShown < n, "with the allowance pre-charged an update showed all %zu surfels (the filter had no limit)", n);
        W.m_Setup.m_OverheadMs = 0; W.m_Setup.m_TraceAllowanceMs = 0;              // no allowance: the backlog drains in one update
        W.Update(L, { 0, 1, 4 }, 1 << 30, 27100);
        CHECK(W.ShownChanged(iF).size() >= n && W.ShownChanged(iF).size() <= 2 * n, "without an allowance the shown light did not drain (%zu of %zu)", W.ShownChanged(iF).size(), n);
    }

    // 6j13. Phase 4: a change lowers confidence, it does not restart. A wall on a floor in the sun, settled; the wall moves 30 cm.
    //       (a) no floor surfel restarts (confidence 0) and none takes more than half of a single pass: the first update after the move
    //       changes each finished surfel's light by at most half of its distance to the new sample; the surfels near the wall are lowered,
    //       the far ones untouched. (b) The recovery: updates (and ms) until the floor's mean light is within 2 % of a fresh solve of the
    //       final scene. (c) A 90 degree rotation in place under a uniform sky: the wall's light barely moves (the sky is the same from
    //       every side) and recovers. (d) The rays: a surfel's offsets advance from pass to pass and across the move, and two surfels'
    //       differ in the same pass
    {
        surfel_mesh Floor, Wall;
        AddQuad(Floor, { -8, 0, -8 }, { 0, 0, 16 }, { 16, 0, 0 }, 0.5f);
        AddQuad(Wall,  { -1, 0, 0 }, { 0, 1.5f, 0 }, { 2, 0, 0 }, 0.5f);
        Floor.Bake(0.15f, 0.45f); Wall.Bake(0.15f, 0.45f);
        lights L; L.m_SunDirection = v3{ 0.3f, -1, 0.2f }.NormalizeCopy(); L.m_SunColor = v3{ 2, 2, 2 }; L.m_Sky = v3{ 0.3f, 0.3f, 0.3f };
        const v3 Eye{ 0, 1, 4 };
        const auto Mean = [](const world& Wd, int Id) { double m = 0; const auto& E = Wd.InstanceLight(Id); for (const auto& e : E) m += e.m_X; return m / double(std::max<std::size_t>(E.size(), 1)); };
        world W;
        const int iF = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        const int iW = W.Add(Wall,  xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        for (int f = 0; f < 300; ++f) W.Update(L, Eye, 1 << 13, 27000 + f);
        const int nF = int(W.InstanceLight(iF).size());
        const std::array<float, 2> Off0 = W.RayOffsets(iF, 0), Off1 = W.RayOffsets(iF, 1);
        // (a)
        const std::vector<v3> Before = W.InstanceLight(iF);
        const xmath::fmat4 Moved = xmath::fmat4::fromTranslation({ 0.3f, 0, 0 });
        W.SetTransform(iW, Moved);
        W.Update(L, Eye, 1 << 13, 27300);                                          // (Phase 6b: the change is applied in the update, its buckets made in slices)
        int Restarted = 0, Lowered = 0, FarLowered = 0;
        for (int k = 0; k < nF; ++k)
        {
            const int c = W.Confidence(iF, k);
            Restarted += c == 0; Lowered += c < 16;
            FarLowered += c < 16 && std::abs(Floor.m_Levels[0][k].m_Center.m_Z) > 5.5f;   // more than 5 m from the wall's plane z = 0 (beyond the 4 m margin)
        }
        double WorstShare = 0;
        for (const auto& [i, k] : W.LastUpdated())
        {
            if (i != iF) continue;
            const float d = (W.InstanceLight(iF)[k] - Before[k]).Length(), ref = std::max(Before[k].Length(), 1e-3f);
            WorstShare = std::max(WorstShare, double(d / ref));
        }
        // (b)
        world Ref;
        const int rF = Ref.Add(Floor, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        Ref.Add(Wall, Moved, { 0.6f, 0.6f, 0.6f });
        for (int f = 0; f < 300; ++f) Ref.Update(L, Eye, 1 << 13, 27400 + f);
        const double Target = Mean(Ref, rF);
        int Recovery = -1; double RecoveryMs = 0;
        for (int f = 0; f < 300 && Recovery < 0; ++f)
        {
            const auto St = W.Update(L, Eye, 1 << 13, 27700 + f);
            RecoveryMs += St.m_TotalMs;
            if (std::abs(Mean(W, iF) - Target) < 0.02 * Target) Recovery = f + 1;
        }
        // (d)
        const std::array<float, 2> Off0b = W.RayOffsets(iF, 0);
        // (c)
        lights Sky; Sky.m_Sky = v3{ 1, 1, 1 };
        world R;
        R.Add(Floor, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        const int rW = R.Add(Wall, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        for (int f = 0; f < 300; ++f) R.Update(Sky, Eye, 1 << 13, 28000 + f);
        const double WallBefore = Mean(R, rW);
        R.SetTransform(rW, xmath::fmat4::fromRotation(xmath::fquat(xmath::fvec3(0, 1, 0), xmath::radian(pi_v * 0.5f))));
        R.Update(Sky, Eye, 1 << 13, 28300);
        int RotRestarted = 0;
        for (int k = 0; k < int(R.InstanceLight(rW).size()); ++k) RotRestarted += R.Confidence(rW, k) == 0;
        const double WallJump = std::abs(Mean(R, rW) - WallBefore) / WallBefore;
        int RotRecovery = -1;
        for (int f = 0; f < 300 && RotRecovery < 0; ++f) { R.Update(Sky, Eye, 1 << 13, 28400 + f); if (std::abs(Mean(R, rW) - WallBefore) < 0.03 * WallBefore) RotRecovery = f + 1; }
        std::printf("a change lowers confidence: %d restarted, %d of %d floor surfels lowered (%d of them far), the first update moved a surfel's light by %.0f %% at most; "
                    "recovery to a fresh solve in %d updates (%.1f ms); rotated 90 degrees: %d restarted, the wall's light moved %.1f %%, back within 3 %% after %d updates; "
                    "ray offsets %.3f,%.3f -> %.3f,%.3f (a neighbor %.3f,%.3f)\n",
                    Restarted, Lowered, nF, FarLowered, WorstShare * 100, Recovery, RecoveryMs, RotRestarted, WallJump * 100, RotRecovery,
                    Off0[0], Off0[1], Off0b[0], Off0b[1], Off1[0], Off1[1]);
        CHECK(Restarted == 0 && RotRestarted == 0, "a change restarted surfels (%d moved, %d rotated)", Restarted, RotRestarted);
        CHECK(Lowered > 0 && Lowered < nF && FarLowered == 0, "the move lowered %d of %d floor surfels (%d far from it)", Lowered, nF, FarLowered);
        CHECK(Recovery > 0 && RotRecovery > 0, "no recovery (%d, %d updates)", Recovery, RotRecovery);
        CHECK(WorstShare < 0.5, "a floor surfel's light moved %.0f %% in the first update after the move (a single pass replaced it?)", WorstShare * 100);
        CHECK(WallJump < 0.1, "the wall's light jumped %.1f %% when it turned under a uniform sky", WallJump * 100);
        CHECK(Off0 != Off0b && Off0 != Off1, "the ray offsets did not advance, or two surfels share them");
    }

    // 6j14. Phase 5, test A of the plan: a receiver-only instance moves. A floor and a wall in the sun, a sign (an emitter) over the floor,
    //       and a 40 cm box dragged from the sun into the wall's shadow. As RECEIVER: rays pass through it, no floor surfel is lowered or
    //       queued, no link work is scheduled for anyone but itself, and its own light follows it into the shadow. As FULL (the same drag):
    //       the floor near it is affected
    {
        surfel_mesh Floor, Wall, Box, Sign;
        AddQuad(Floor, { -8, 0, -8 }, { 0, 0, 16 }, { 16, 0, 0 }, 0.5f);
        AddQuad(Wall,  { -2, 0, 0 }, { 0, 3, 0 }, { 4, 0, 0 }, 0.5f);                       // z = 0, facing -z
        AddQuad(Sign,  { 3, 2, 3 }, { 0.4f, 0, 0 }, { 0, 0, 0.4f }, 0.2f);                  // facing down
        const v3 Lo{ -0.2f, 0, -0.2f }, Hi{ 0.2f, 0.4f, 0.2f };
        AddQuad(Box, { Lo.m_X, Hi.m_Y, Lo.m_Z }, { 0, 0, 0.4f }, { 0.4f, 0, 0 }, 0.1f);
        AddQuad(Box, { Lo.m_X, Lo.m_Y, Lo.m_Z }, { 0, 0.4f, 0 }, { 0, 0, 0.4f }, 0.1f);
        AddQuad(Box, { Hi.m_X, Lo.m_Y, Lo.m_Z }, { 0, 0, 0.4f }, { 0, 0.4f, 0 }, 0.1f);
        AddQuad(Box, { Lo.m_X, Lo.m_Y, Lo.m_Z }, { 0.4f, 0, 0 }, { 0, 0.4f, 0 }, 0.1f);
        AddQuad(Box, { Lo.m_X, Lo.m_Y, Hi.m_Z }, { 0, 0.4f, 0 }, { 0.4f, 0, 0 }, 0.1f);
        Floor.Bake(0.25f, 0.75f); Wall.Bake(0.25f, 0.75f); Box.Bake(0.05f, 0.15f); Sign.Bake(0.1f, 0.3f);
        lights L; L.m_SunDirection = v3{ 0, -1, 1 }.NormalizeCopy(); L.m_SunColor = v3{ 3, 3, 3 }; L.m_Sky = v3{ 0.1f, 0.1f, 0.1f };   // the sun toward +z: the wall shades z > 0
        const v3 Eye{ 0, 2, -6 };
        struct result { int Lowered = 0, Queued = 0, OtherLinks = 0; double SunLight = 0, ShadeLight = 0; bool bRayHits = false; };
        const auto Run = [&](role Role)
        {
            result R;
            world W;
            const int iF = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            W.Add(Wall, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            const int iS = W.Add(Sign, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            W.SetSurfelMaterial(iS, {}, std::vector<v3>(Sign.m_Levels[0].size(), v3{ 10, 10, 10 }));
            const int iB = W.Add(Box, xmath::fmat4::fromTranslation({ 0, 0, -3 }), { 0.6f, 0.6f, 0.6f });
            W.SetRole(iB, Role);
            const auto Mean = [&] { double m = 0; for (const auto& e : W.InstanceLight(iB)) m += e.m_X; return m / double(W.InstanceLight(iB).size()); };
            for (int f = 0; f < 200; ++f) W.Update(L, Eye, 1 << 14, 29000 + f);
            R.SunLight = Mean();
            R.bRayHits = W.Cast({ 0, 0.2f, -5 }, { 0, 0, 1 }, 4.0f).m_Inst == iB;
            for (int f = 0; f < 60; ++f)                                                   // dragged to z = +1.5 (behind the wall: in its shadow, 3 m long)
            {
                W.SetTransform(iB, xmath::fmat4::fromTranslation({ 0, 0, -3 + 0.075f * float(f + 1) }));
                const auto St = W.Update(L, Eye, 1 << 14, 29200 + f);
                for (int k = 0; k < int(W.InstanceLight(iF).size()); ++k) R.Lowered += W.Confidence(iF, k) < 16;
                R.Queued += W.UrgentCount(iF);
                R.OtherLinks += St.m_LinkPending > 1 ? 1 : 0;
            }
            for (int f = 0; f < 200; ++f) W.Update(L, Eye, 1 << 14, 29300 + f);
            R.ShadeLight = Mean();
            return R;
        };
        const result Rec = Run(role::RECEIVER), Full = Run(role::FULL);
        std::printf("a receiver dragged into a shadow: %d floor surfels lowered, %d queued, %d updates with others' link work, the rays %s it; its light %.3f -> %.3f; "
                    "the same drag as FULL: %d lowered, the rays %s it, its light %.3f -> %.3f\n",
                    Rec.Lowered, Rec.Queued, Rec.OtherLinks, Rec.bRayHits ? "hit" : "pass", Rec.SunLight, Rec.ShadeLight,
                    Full.Lowered, Full.bRayHits ? "hit" : "pass", Full.SunLight, Full.ShadeLight);
        CHECK(Rec.Lowered == 0 && Rec.Queued == 0 && Rec.OtherLinks == 0 && !Rec.bRayHits, "a receiver changed the others (%d lowered, %d queued, %d link updates, rays hit %d)", Rec.Lowered, Rec.Queued, Rec.OtherLinks, int(Rec.bRayHits));
        CHECK(Rec.ShadeLight < 0.5 * Rec.SunLight, "the receiver's light did not follow it into the shadow (%.3f -> %.3f)", Rec.SunLight, Rec.ShadeLight);
        CHECK(Full.Lowered > 0 && Full.bRayHits, "a full-transport box changed nothing around it (%d lowered, rays hit %d)", Full.Lowered, int(Full.bRayHits));
    }

    // 6j15. Phase 6: rays requested by uncertainty (setup::m_bAdaptiveRays) against a fixed count, at the same ray budget: a room (a floor
    //       and two walls under a dim sky) settles, then a point light turns on; the recovery error is the mean relative difference of the
    //       floor's light to the same scene solved long with the light on, summed over the first 60 updates. And RelightAll: at most once
    //       a period (a call too soon is done later), traced under the budget (sliced), not at once
    {
        surfel_mesh Floor, WallA, WallB;
        AddQuad(Floor, { -4, 0, -4 }, { 0, 0, 8 }, { 8, 0, 0 }, 0.5f);
        AddQuad(WallA, { -4, 0, -4 }, { 0, 3, 0 }, { 0, 0, 8 }, 0.5f);           // x = -4, facing +x
        AddQuad(WallB, { -4, 0, -4 }, { 8, 0, 0 }, { 0, 3, 0 }, 0.5f);           // z = -4, facing +z
        Floor.Bake(0.2f, 0.6f); WallA.Bake(0.2f, 0.6f); WallB.Bake(0.2f, 0.6f);
        lights Off; Off.m_Sky = v3{ 0.05f, 0.05f, 0.05f };
        lights On = Off; On.m_Points.push_back({ v3{ -2.5f, 2.0f, -2.5f }, v3{ 30, 30, 30 }, 12.0f });
        const v3 Eye{ 2, 1.6f, 2 };
        const auto Make = [&](world& W) { const int f = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f }); W.Add(WallA, xmath::fmat4::fromIdentity(), { 0.7f, 0.7f, 0.7f }); W.Add(WallB, xmath::fmat4::fromIdentity(), { 0.7f, 0.7f, 0.7f }); return f; };
        world Ref; const int rF = Make(Ref);
        for (int f = 0; f < 600; ++f) Ref.Update(On, Eye, 1 << 16, 30000 + f);
        const auto Error = [&](const world& W, int iF)
        {
            double e = 0, n = 0;
            const auto& A = W.InstanceLight(iF); const auto& B = Ref.InstanceLight(rF);
            for (std::size_t k = 0; k < A.size() && k < B.size(); ++k) { e += std::abs(A[k].m_X - B[k].m_X); n += B[k].m_X; }
            return e / std::max(n, 1e-9);
        };
        const auto Run = [&](bool bAdaptive, double& SettledErr, int& Rays)
        {
            world W; const int iF = Make(W);
            W.m_Setup.m_bAdaptiveRays = bAdaptive;
            for (int f = 0; f < 300; ++f) W.Update(Off, Eye, 1 << 11, 31000 + f);
            double Sum = 0; Rays = 0;
            for (int f = 0; f < 60; ++f) { const auto St = W.Update(On, Eye, 1 << 11, 31300 + f); Sum += Error(W, iF); Rays += St.m_Rays; }
            for (int f = 0; f < 240; ++f) W.Update(On, Eye, 1 << 11, 31400 + f);
            SettledErr = Error(W, iF);
            return Sum / 60;
        };
        double SetFixed, SetAdapt; int RaysFixed, RaysAdapt;
        const double Fixed = Run(false, SetFixed, RaysFixed), Adapt = Run(true, SetAdapt, RaysAdapt);
        // RelightAll
        world W; Make(W);
        for (int f = 0; f < 300; ++f) W.Update(On, Eye, 1 << 11, 32000 + f);
        const bool bFirst = W.RelightAll(), bSoon = W.RelightAll();
        int MaxRays = 0, Dirty0 = 0;
        for (int f = 0; f < 10; ++f) { const auto St = W.Update(On, Eye, 1 << 11, 32300 + f); MaxRays = std::max(MaxRays, St.m_Rays); if (f == 0) Dirty0 = St.m_Dirty; }
        for (int f = 0; f < 130; ++f) W.Update(On, Eye, 1 << 11, 32400 + f);
        std::printf("a light turned on, the recovery error over 60 updates: fixed rays %.3f (%d rays), adaptive %.3f (%d rays); settled %.3f vs %.3f; "
                    "RelightAll: %d, again at once %d (deferred), %d instances dirty, at most %d rays an update (budget %d)\n",
                    Fixed, RaysFixed, Adapt, RaysAdapt, SetFixed, SetAdapt, int(bFirst), int(bSoon), Dirty0, MaxRays, 1 << 11);
        CHECK(Adapt <= Fixed * 1.05 && SetAdapt <= SetFixed * 1.1 + 0.01, "the adaptive rays recover worse (%.3f vs %.3f; settled %.3f vs %.3f)", Adapt, Fixed, SetAdapt, SetFixed);
        CHECK(bFirst && !bSoon && Dirty0 == 3 && MaxRays <= (1 << 11), "RelightAll is not rate-limited or not sliced (%d %d, %d dirty, %d rays)", int(bFirst), int(bSoon), Dirty0, MaxRays);
    }

    // 6j16. Phase 6b, publication while the light keeps changing: a wall moves on every update and a point light toggles every 7, under a
    //       0.3 ms allowance (the shown light's backlog is worked off a slice at a time). Each update's shown light is compared with the
    //       whole filter of the inputs it has then (made apart): the difference stays bounded while it changes, no value is torn or bad,
    //       and once the changes stop it goes to 0
    {
        surfel_mesh Floor, Wall;
        AddQuad(Floor, { -6, 0, -6 }, { 0, 0, 12 }, { 12, 0, 0 }, 0.5f);
        AddQuad(Wall,  { -1, 0, 0 }, { 0, 1.5f, 0 }, { 2, 0, 0 }, 0.5f);
        Floor.Bake(0.15f, 0.45f); Wall.Bake(0.15f, 0.45f);
        lights Off; Off.m_Sky = v3{ 0.1f, 0.1f, 0.1f };
        lights On = Off; On.m_Points.push_back({ v3{ 1.5f, 1.5f, 1.0f }, v3{ 20, 20, 20 }, 8.0f });
        world W;
        const int iF = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        const int iW = W.Add(Wall,  xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f });
        W.m_Setup.m_TraceAllowanceMs = 0.3f;
        const auto Diff = [&]
        {
            const auto& A = W.BasisLight(iF); const auto B = W.FilterOf(iF);
            double d = 0, n = 0; int Bad = 0;
            for (std::size_t k = 0; k < A.size() && k < B.size(); ++k)
                for (int i = 0; i < 3; ++i) { d += (A[k][i] - B[k][i]).Length(); n += B[k][i].Length(); Bad += !std::isfinite(A[k][i].m_X + A[k][i].m_Y + A[k][i].m_Z); }
            return std::pair{ d / std::max(n, 1e-9), Bad };
        };
        for (int f = 0; f < 100; ++f) W.Update(Off, { 0, 1, 4 }, 1 << 12, 33000 + f);
        double WorstDuring = 0; int BadAll = 0;
        for (int f = 0; f < 80; ++f)
        {
            W.SetTransform(iW, xmath::fmat4::fromTranslation({ 0.02f * float(f % 40), 0, 0 }));
            W.Update((f / 7) % 2 ? On : Off, { 0, 1, 4 }, 1 << 12, 33100 + f);
            const auto [d, Bad] = Diff(); WorstDuring = std::max(WorstDuring, d); BadAll += Bad;
        }
        int Settle = -1;
        for (int f = 0; f < 400 && Settle < 0; ++f) { W.Update(On, { 0, 1, 4 }, 64, 33200 + f); if (Diff().first < 1e-6) Settle = f + 1; }   // (a few surfels an update: the backlog drains)
        std::printf("publication while the light changes: the shown light off the whole filter of its inputs by %.1f %% at most, %d bad values; %d updates after the changes stop it is the whole filter\n",
                    WorstDuring * 100, BadAll, Settle);
        CHECK(BadAll == 0 && Settle > 0 && WorstDuring < 0.5, "publication under change: %.1f %% off, %d bad, settled after %d", WorstDuring * 100, BadAll, Settle);
    }

    // 6j17. Phase 6b, localized recovery: four events, the error over the region they affect against a fresh solve of the final scene
    //       (mean absolute relative error, its 95th percentile, its sum over the updates, and when it reaches and stays under 10 %, in
    //       updates and ms), under a 1 ms allowance. (1) a bright light off; (2) a door closing across an emitter's link; (3) a receiver
    //       moving from a lit room to a dark one; (4) the sun turning 90 degrees at the same energy
    {
        struct report { double Mae0 = 0, P95End = 0, Integral = 0, Ms = 0; int Below = -1; };
        const auto Measure = [](world& W, const lights& L, const v3& Eye, const std::vector<std::pair<int, int>>& Region, const std::vector<v3>& Ref, unsigned Seed)
        {
            report R; int Run = 0;
            const auto Err = [&](double* pP95)
            {
                std::vector<double> e; double Sum = 0, RefSum = 0;
                for (std::size_t k = 0; k < Region.size(); ++k)
                {
                    const v3 A = W.InstanceLight(Region[k].first)[Region[k].second], B = Ref[k];
                    const double d = std::abs(A.m_X - B.m_X) + std::abs(A.m_Y - B.m_Y) + std::abs(A.m_Z - B.m_Z), r = B.m_X + B.m_Y + B.m_Z;
                    Sum += d; RefSum += r; e.push_back(d / std::max(r, 0.05 * 3));
                }
                if (pP95 && !e.empty()) { std::sort(e.begin(), e.end()); *pP95 = e[std::size_t(0.95 * double(e.size() - 1))]; }
                return Sum / std::max(RefSum, 1e-6);
            };
            R.Mae0 = Err(nullptr);
            for (int f = 0; f < 400; ++f)
            {
                const auto St = W.Update(L, Eye, 1 << 13, Seed + f);
                R.Ms += St.m_TotalMs;
                const double e = Err(nullptr);
                R.Integral += e;
                if (e < 0.10) { if (++Run == 1) R.Below = f + 1; } else { Run = 0; R.Below = -1; }
            }
            Err(&R.P95End);
            return R;
        };
        const auto Settle = [](world& W, const lights& L, const v3& Eye, unsigned Seed) { for (int f = 0; f < 400; ++f) W.Update(L, Eye, 1 << 13, Seed + f); };
        const auto RegionOf = [](const world& W, int Id) { std::vector<std::pair<int, int>> R; for (int s = 0; s < int(W.InstanceLight(Id).size()); ++s) R.push_back({ Id, s }); return R; };
        const auto RefOf = [](const world& W, const std::vector<std::pair<int, int>>& R) { std::vector<v3> V; for (const auto& [i, s] : R) V.push_back(W.InstanceLight(i)[s]); return V; };
        report Rep[4];
        // (1) a bright light off, a room
        {
            surfel_mesh Floor, WallA;
            AddQuad(Floor, { -4, 0, -4 }, { 0, 0, 8 }, { 8, 0, 0 }, 0.5f);
            AddQuad(WallA, { -4, 0, -4 }, { 0, 3, 0 }, { 0, 0, 8 }, 0.5f);
            Floor.Bake(0.2f, 0.6f); WallA.Bake(0.2f, 0.6f);
            lights Off; Off.m_Sky = v3{ 0.05f, 0.05f, 0.05f };
            lights On = Off; On.m_Points.push_back({ v3{ -2, 2, 0 }, v3{ 40, 40, 40 }, 10.0f });
            const auto Make = [&](world& W) { const int f = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f }); W.Add(WallA, xmath::fmat4::fromIdentity(), { 0.7f, 0.7f, 0.7f }); W.m_Setup.m_TraceAllowanceMs = 1.0f; return f; };
            world Ref; const int rF = Make(Ref); Settle(Ref, Off, { 2, 1.6f, 2 }, 34000);
            world W; const int iF = Make(W); Settle(W, On, { 2, 1.6f, 2 }, 34500);
            const auto Region = RegionOf(W, iF);
            Rep[0] = Measure(W, Off, { 2, 1.6f, 2 }, Region, RefOf(Ref, RegionOf(Ref, rF)), 35000);
        }
        // (2) a door closes across an emitter's link (the lamp, the door, the floor of 6j9b)
        {
            surfel_mesh Floor, Lamp, Door;
            AddQuad(Floor, { 1, 0, -1 }, { 0, 0, 2 }, { 2, 0, 0 }, 0.25f);
            AddQuad(Lamp,  { -1, 0.5f, -0.5f }, { 0, 1, 0 }, { 0, 0, 1 }, 0.25f);
            AddQuad(Door,  { 0, -1, -3 }, { 0, 0, 6 }, { 0, 4, 0 }, 0.5f);
            Floor.Bake(0.15f, 0.45f); Lamp.Bake(0.15f, 0.45f); Door.Bake(0.3f, 0.9f);
            lights Dark; Dark.m_Sky = v3{ 0.02f, 0.02f, 0.02f };
            const auto Make = [&](world& W, bool bShut, int& iD)
            {
                const int f = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
                const int l = W.Add(Lamp, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
                iD = W.Add(Door, bShut ? xmath::fmat4::fromIdentity() : xmath::fmat4::fromTranslation({ 0, 0, 10 }), { 0, 0, 0 });
                W.SetSurfelMaterial(l, {}, std::vector<v3>(Lamp.m_Levels[0].size(), v3{ 20, 20, 20 }));
                W.m_Setup.m_TraceAllowanceMs = 1.0f;
                return f;
            };
            int d0, d1;
            world Ref; const int rF = Make(Ref, true, d0); Settle(Ref, Dark, { 2, 1, 0 }, 36000);
            world W; const int iF = Make(W, false, d1); Settle(W, Dark, { 2, 1, 0 }, 36500);
            W.SetTransform(d1, xmath::fmat4::fromIdentity());
            Rep[1] = Measure(W, Dark, { 2, 1, 0 }, RegionOf(W, iF), RefOf(Ref, RegionOf(Ref, rF)), 37000);
        }
        // (3) a receiver from a lit room into a dark one (a wall between, the light in the first)
        {
            surfel_mesh Floor, Wall, Box;
            AddQuad(Floor, { -6, 0, -3 }, { 0, 0, 6 }, { 12, 0, 0 }, 0.5f);
            AddQuad(Wall,  { 0, 0, -3 }, { 0, 3, 0 }, { 0, 0, 6 }, 0.5f);
            AddQuad(Wall,  { 0, 0, -3 }, { 0, 0, 6 }, { 0, 3, 0 }, 0.5f);                  // both sides
            AddQuad(Box,   { -0.2f, 0.4f, -0.2f }, { 0, 0, 0.4f }, { 0.4f, 0, 0 }, 0.1f);
            AddQuad(Box,   { -0.2f, 0, -0.2f }, { 0, 0.4f, 0 }, { 0, 0, 0.4f }, 0.1f);
            AddQuad(Box,   { 0.2f, 0, -0.2f }, { 0, 0, 0.4f }, { 0, 0.4f, 0 }, 0.1f);
            Floor.Bake(0.25f, 0.75f); Wall.Bake(0.25f, 0.75f); Box.Bake(0.05f, 0.15f);
            lights L; L.m_Sky = v3{ 0.01f, 0.01f, 0.01f }; L.m_Points.push_back({ v3{ -3, 2, 0 }, v3{ 30, 30, 30 }, 6.0f });
            const auto Make = [&](world& W, float x) { W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f }); W.Add(Wall, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
                                                       const int b = W.Add(Box, xmath::fmat4::fromTranslation({ x, 0, 0 }), { 0.6f, 0.6f, 0.6f }); W.SetRole(b, role::RECEIVER); W.m_Setup.m_TraceAllowanceMs = 1.0f; return b; };
            world Ref; const int rB = Make(Ref, 3); Settle(Ref, L, { 0, 2, -5 }, 38000);
            world W; const int iB = Make(W, -3); Settle(W, L, { 0, 2, -5 }, 38500);
            W.SetTransform(iB, xmath::fmat4::fromTranslation({ 3, 0, 0 }));
            Rep[2] = Measure(W, L, { 0, 2, -5 }, RegionOf(W, iB), RefOf(Ref, RegionOf(Ref, rB)), 39000);
        }
        // (4) the sun turns 90 degrees, the same energy: a floor and a wall
        {
            surfel_mesh Floor, Wall;
            AddQuad(Floor, { -4, 0, -4 }, { 0, 0, 8 }, { 8, 0, 0 }, 0.5f);
            AddQuad(Wall,  { -2, 0, 0 }, { 0, 3, 0 }, { 4, 0, 0 }, 0.5f);
            Floor.Bake(0.2f, 0.6f); Wall.Bake(0.2f, 0.6f);
            lights A; A.m_SunDirection = v3{ 0.5f, -1, 0.5f }.NormalizeCopy(); A.m_SunColor = v3{ 2, 2, 2 }; A.m_Sky = v3{ 0.1f, 0.1f, 0.1f };
            lights B = A; B.m_SunDirection = v3{ -0.5f, -1, -0.5f }.NormalizeCopy();
            const auto Make = [&](world& W) { const int f = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f }); W.Add(Wall, xmath::fmat4::fromIdentity(), { 0.6f, 0.6f, 0.6f }); W.m_Setup.m_TraceAllowanceMs = 1.0f; return f; };
            world Ref; const int rF = Make(Ref); Settle(Ref, B, { 0, 2, 5 }, 40000);
            world W; const int iF = Make(W); Settle(W, A, { 0, 2, 5 }, 40500);
            Rep[3] = Measure(W, B, { 0, 2, 5 }, RegionOf(W, iF), RefOf(Ref, RegionOf(Ref, rF)), 41000);
        }
        const char* Names[4] = { "light off", "door shut", "receiver into the dark", "sun turned" };
        for (int k = 0; k < 4; ++k)
            std::printf("recovery, %s: error %.0f %% at the event, under 10 %% and staying after %d updates (of 400, %.0f ms of updates in all), the error summed over updates %.2f, at the end its p95 %.0f %%\n",
                        Names[k], Rep[k].Mae0 * 100, Rep[k].Below, Rep[k].Ms, Rep[k].Integral, Rep[k].P95End * 100);
        for (int k = 0; k < 4; ++k) CHECK(Rep[k].Below > 0, "%s: the region's error did not stay under 10 %% (%.0f %% at the event)", Names[k], Rep[k].Mae0 * 100);
    }

    // 6j18. Phase 6b, the change detector (estimator::Lower / Add, defined: the luminance of a sample against the light the surfel had when
    //       the change came, two in a row more than half of it apart the same way, with a floor of 1e-3 for the dark; the samples are the
    //       same batch size, a pass of the surfel's rays): (a) constant light, noisy samples: few false takes, the mean near the truth;
    //       (b) near black: the floor keeps noise from triggering; (c) rapid on/off: the mean stays between them; (d) a color change at the
    //       same luminance is not detected but followed by the warm-up; (e) a bright source ramping up is followed; (f) lowered again and
    //       again before it reconverges, it stays near the light
    {
        rng R(4242);
        const auto Noise = [&](float m, float a) { return m + (R.Next() * 2 - 1) * a; };
        const auto Settled = [](const v3& E) { estimator Es; Es.Restart(); for (int k = 0; k < 40; ++k) Es.Add(E); return Es; };
        // (a)
        estimator A = Settled(v3{ 1, 1, 1 });
        A.Lower(1);
        for (int k = 0; k < 60; ++k) { const float y = Noise(1, 0.9f); A.Add(v3{ y, y, y }); }
        // (b)
        estimator B = Settled(v3{ 1e-4f, 1e-4f, 1e-4f });
        B.Lower(1);
        for (int k = 0; k < 60; ++k) { const float y = std::max(Noise(1e-4f, 1e-4f), 0.0f); B.Add(v3{ y, y, y }); }
        // (c)
        estimator C = Settled(v3{ 1, 1, 1 });
        float CMin = 1e9f, CMax = -1e9f;
        for (int k = 0; k < 60; ++k) { if (k % 2 == 0) C.Lower(1); const float y = (k / 2) % 2 ? 1.0f : 0.0f; C.Add(v3{ y, y, y }); CMin = std::min(CMin, C.m_Mean.m_X); CMax = std::max(CMax, C.m_Mean.m_X); }
        // (d) red to green at the same luminance
        const float g = 0.299f / 0.587f;
        estimator D = Settled(v3{ 1, 0, 0 });
        D.Lower(1);
        for (int k = 0; k < 20; ++k) D.Add(v3{ 0, g, 0 });
        // (e) a ramp 1 -> 5 over 20 samples, then held
        estimator E = Settled(v3{ 1, 1, 1 });
        E.Lower(1);
        for (int k = 0; k < 40; ++k) { const float y = 1 + 4 * std::min(1.0f, float(k) / 20.0f); E.Add(v3{ y, y, y }); }
        // (f) lowered every 3 samples, constant noisy light
        estimator F = Settled(v3{ 2, 2, 2 });
        double FAvg = 0;                                                       // (its mean over the run: a bias, not one noisy value of a 3-sample mean)
        for (int k = 0; k < 600; ++k) { if (k % 3 == 0) F.Lower(0.8f); const float y = Noise(2, 1.5f); F.Add(v3{ y, y, y }); FAvg += F.m_Mean.m_X / 600.0; }
        std::printf("change detector: (a) noisy constant light: %d takes, mean %.2f (1); (b) near black: %d takes; (c) rapid on/off: the mean in [%.2f, %.2f]; "
                    "(d) red to green at equal luminance: %d takes, green %.2f (%.2f) red %.2f; (e) a ramp to 5: %.2f; (f) lowered again and again: %.2f (2)\n",
                    A.m_nTaken, A.m_Mean.m_X, B.m_nTaken, CMin, CMax, D.m_nTaken, D.m_Mean.m_Y, g, D.m_Mean.m_X, E.m_Mean.m_X, FAvg);
        CHECK(A.m_nTaken <= 6 && std::abs(A.m_Mean.m_X - 1) < 0.25f, "noisy constant light: %d takes, mean %.2f", A.m_nTaken, A.m_Mean.m_X);
        CHECK(B.m_nTaken == 0, "near black, noise triggered %d takes", B.m_nTaken);
        CHECK(CMin >= -1e-6f && CMax <= 1.0f + 1e-6f, "rapid on/off left the range [0, 1] (%.2f, %.2f)", CMin, CMax);
        CHECK(D.m_Mean.m_Y > 0.8f * g && D.m_Mean.m_X < 0.2f, "an equal-luminance color change was not followed (green %.2f, red %.2f)", D.m_Mean.m_Y, D.m_Mean.m_X);
        CHECK(E.m_Mean.m_X > 4.5f, "the ramp was not followed (%.2f)", E.m_Mean.m_X);
        CHECK(std::abs(FAvg - 2) < 0.2, "repeated lowering drifted (its mean %.2f over the run)", FAvg);
    }

    // 6k. A recessed panel behind a lip (a door 3 cm behind its frame): it gets surfels of its own next to the lip, not left to the
    //     surfels past the lip that cannot see it (read through the lip, they faded in wedges)
    {
        surfel_mesh M;
        AddQuad(M, { -2, 0, -1 }, { 0, 0, 2 }, { 2, 0, 0 }, 0.25f);                       // the wall: x in [-2, 0], y = 0
        AddQuad(M, { 0.05f, -0.03f, -1 }, { 0, 0, 2 }, { 1, 0, 0 }, 0.25f);              // the door: x in [0.05, 1.05], 3 cm lower
        AddQuad(M, { 0, 0.05f, -1 }, { 0, 0, 2 }, { 0.05f, 0, 0 }, 0.25f);               // the lip's top (5 cm above the wall)
        AddQuad(M, { 0, -0.03f, -1 }, { 0, 0.08f, 0 }, { 0, 0, 2 }, 0.25f);              // its faces
        AddQuad(M, { 0.05f, -0.03f, -1 }, { 0, 0, 2 }, { 0, 0.08f, 0 }, 0.25f);
        M.Bake(0.3f, 0.9f);
        int nNear = 0;
        for (const auto& S : M.m_Levels[0]) nNear += S.m_Center.m_Y < -0.02f && S.m_Center.m_X < 0.35f && S.m_Normal.m_Y > 0.9f;
        std::printf("a door behind its frame's lip: %d surfels of its own within 0.3 m of the lip\n", nNear);
        CHECK(nNear >= 5, "the door next to its frame has few surfels of its own (%d; 3 when the surfels past the lip cover it)", nNear);
    }

    // 6l. Smoothed normals on a flat box (a pillar whose corner normals are averaged): the surfels still face their face's way (the GI's
    //     geometry uses the faces: tilted surfels near the corners made the reads step along the triangles - a sawtooth)
    {
        surfel_mesh Pillar = Box({ 1, 3, 1 }, 0.25f);
        for (std::size_t v = 0; v < Pillar.m_Pos.size(); ++v)                    // averaged at the corners: the direction of the vertex
            Pillar.m_Nrm[v] = (Pillar.m_Nrm[v] * 0.5f + Pillar.m_Pos[v].NormalizeCopy() * 0.5f).NormalizeCopy();
        Pillar.Bake(0.3f, 0.9f);
        int nTilted = 0;
        for (const auto& S : Pillar.m_Levels[0])
            nTilted += std::max({ std::abs(S.m_Normal.m_X), std::abs(S.m_Normal.m_Y), std::abs(S.m_Normal.m_Z) }) < 0.999f;
        std::printf("smoothed normals on a box: %d of %zu surfels tilted off their face\n", nTilted, Pillar.m_Levels[0].size());
        CHECK(nTilted == 0, "%d surfels follow the smoothed normals instead of their faces", nTilted);
    }

    // 6. Buried surfels decay for long (the box's bottom on the floor: their validity heads to 0, then underflows), far and back:
    //    the reads at the ray hits divide by a total weight that is ~0, and nothing may come out as inf / NaN (it spreads to every surfel)
    {
        cornell C;
        MakeCornell(C, 0.5f, false);
        const auto Bad = [&]
        {
            int n = 0;
            for (int i = 0; i < C.m_World.InstanceCount(); ++i)
            {
                const auto& E = C.m_World.InstanceLight(i);
                for (const auto& e : E) n += !(std::isfinite(e.m_X + e.m_Y + e.m_Z) && e.m_X >= 0 && e.m_Y >= 0 && e.m_Z >= 0);
            }
            return n;
        };
        int Worst = 0;
        for (int f = 0; f < 3000; ++f)
        {
            const v3 Eye = (f / 250) % 2 ? v3{ 0, 0, 200 } : v3{ 0, 0, -0.9f };          // near, far, near...
            C.m_World.Update(C.m_Lights, Eye, 1 << 30, 5000 + f);
            if (f % 50 == 49) Worst = std::max(Worst, Bad());
        }
        std::printf("long decay, far and back: at most %d bad surfels\n", Worst);
        CHECK(Worst == 0, "%d surfels went inf / NaN after a long decay and level switches", Worst);
    }

    // 7. A buried surfel (a quad through the wall of a closed box: inside, its rays meet back faces; at the wall, some do) decays to a validity of exactly 0, never to the denormals
    //    under 1e-6 (the Police Station: a sum of such weights divided into inf, then NaN that spread to the whole mesh)
    {
        surfel_mesh Shell = Box({ 2, 2, 2 }, 0.25f), Inner;
        AddQuad(Inner, { -0.5f, 0, -0.5f }, { 0, 0, 1 }, { 2, 0, 0 }, 0.25f);     // through the +x wall: the surfels at the wall half buried (noisy validity)
        Shell.Bake(0.3f, 0.9f); Inner.Bake(0.3f, 0.9f);
        world W;
        W.Add(Shell, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        const int iInner = W.Add(Inner, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        lights L; L.m_Sky = { 1, 1, 1 };
        int nDenormal = 0, nZero = 0;
        for (int f = 0; f < 2000; ++f)
        {
            W.Update(L, { 0, 0, 0 }, 1 << 30, 7000 + f);
            for (int i = 0; i < W.InstanceCount(); ++i) for (float v : W.InstanceValid(i)) nDenormal += v > 0 && v < 1e-6f;
        }
        for (float v : W.InstanceValid(iInner)) nZero += v == 0;
        std::printf("buried surfels: %d of %zu at validity 0; %d validities under 1e-6 over 2000 updates\n", nZero, W.InstanceValid(iInner).size(), nDenormal);
        CHECK(nZero > 0, "the buried quad has no buried surfels");
        CHECK(nDenormal == 0, "%d validities decayed under 1e-6 instead of to 0", nDenormal);
    }

    if (g_nErrors) return 1;
    // 6m. A transparent pane (SetSeeThrough: its material has alpha) over a floor under the sky: the floor's light comes through it as
    //     if it were not there; opaque, the pane shades the floor
    {
        surfel_mesh Floor, Pane;
        AddQuad(Floor, { -1, 0, -1 }, { 0, 0, 2 }, { 2, 0, 0 }, 0.25f);
        AddQuad(Pane, { -3, 0.3f, -3 }, { 6, 0, 0 }, { 0, 0, 6 }, 0.5f);        // facing down, over the floor (and well past it)
        Floor.Bake(0.2f, 0.6f); Pane.Bake(0.2f, 0.6f);
        const auto FloorLight = [&](bool bSeeThrough)
        {
            world W;
            const int iFloor = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            const int iPane  = W.Add(Pane,  xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            if (bSeeThrough) W.SetSeeThrough(iPane, { true });
            lights L; L.m_Sky = { 1, 1, 1 };
            for (int f = 0; f < 100; ++f) W.Update(L, { 0, 1, 0 }, 1 << 30, 15000 + f);
            double Sum = 0;
            for (const auto& e : W.InstanceLight(iFloor)) Sum += e.m_X;
            const auto& V = W.InstanceValid(iPane);
            const double PaneValid = V.empty() ? 0 : std::accumulate(V.begin(), V.end(), 0.0) / V.size();
            return std::pair{ Sum / std::max<std::size_t>(W.InstanceLight(iFloor).size(), 1), PaneValid };
        };
        const auto [Opaque, OpaqueValid] = FloorLight(false);
        const auto [Clear, ClearValid]   = FloorLight(true);
        std::printf("a pane over the floor: floor light %.3f opaque, %.3f transparent (open sky pi = 3.142); the pane's validity %.2f -> %.2f\n", Opaque, Clear, OpaqueValid, ClearValid);
        CHECK(Clear > 0.95 * pi_v && Opaque < 0.5 * Clear, "the transparent pane still shades the floor (%f vs %f)", Clear, Opaque);
        CHECK(ClearValid < 1e-3, "the transparent pane's surfels are still read (validity %f)", ClearValid);
    }

    // 8. An unsupported solver query (world::Indirect: the hit's vertices read no surfel) is not dark: a floor next to a sunlit wall whose
    //    vertices list no surfels gets about the light it gets from the same wall read normally (the policy: the wall's mean light, the ray
    //    still one of the surfel's rays), and every such query is counted, with where it hit
    {
        surfel_mesh Floor, Wall;
        AddQuad(Floor, { -1, 0, -1 }, { 0, 0, 2 }, { 2, 0, 0 }, 0.25f);
        AddQuad(Wall, { -1, 0, -1 }, { 0, 1.5f, 0 }, { 0, 0, 2 }, 0.25f);         // at x = -1, facing +x
        Floor.Bake(0.2f, 0.6f); Wall.Bake(0.2f, 0.6f);
        surfel_mesh Unread = Wall;
        for (auto& v : Unread.m_VertSurfels) v = { -1, -1, -1, -1 };
        struct result { double m_Floor; int m_nUnsupported; std::size_t m_nAt; bool m_bOnWall; };
        const auto FloorLight = [&](const surfel_mesh& W2)
        {
            world W;
            const int iFloor = W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
            W.Add(W2, xmath::fmat4::fromIdentity(), { 0.7f, 0.7f, 0.7f });
            lights L; L.m_Sky = { 0.3f, 0.3f, 0.3f }; L.m_SunDirection = v3{ -1, -0.5f, 0 }.NormalizeCopy(); L.m_SunColor = { 3, 3, 3 };
            int nUnsupported = 0;
            for (int f = 0; f < 100; ++f) nUnsupported += W.Update(L, { 0, 1, 0 }, 1 << 30, 16000 + f).m_nUnsupported;
            double Sum = 0;
            for (const auto& e : W.InstanceLight(iFloor)) Sum += Lum(e);
            bool bOnWall = !W.UnsupportedAt().empty();
            for (const auto& P : W.UnsupportedAt()) bOnWall = bOnWall && std::abs(P.m_X + 1) < 0.01f;
            return result{ Sum / std::max<std::size_t>(W.InstanceLight(iFloor).size(), 1), nUnsupported, W.UnsupportedAt().size(), bOnWall };
        };
        const auto Ref = FloorLight(Wall), Pol = FloorLight(Unread);
        std::printf("unsupported queries: the floor's light %.3f with the wall read, %.3f with it unsupported (the policy); %d unsupported (%zu places kept, on the wall: %d), %d read normally\n"
                   , Ref.m_Floor, Pol.m_Floor, Pol.m_nUnsupported, Pol.m_nAt, int(Pol.m_bOnWall), Ref.m_nUnsupported);
        CHECK(Ref.m_nUnsupported == 0 && Pol.m_nUnsupported > 0 && Pol.m_bOnWall, "the unsupported queries are not counted where they are");
        CHECK(std::abs(Pol.m_Floor - Ref.m_Floor) < 0.05 * Ref.m_Floor, "an unsupported query changes the light: %.3f vs %.3f", Pol.m_Floor, Ref.m_Floor);
    }

    // 9. Light leaks through bad topology (the benchmark of the leak work): two sealed rooms whose floor and ceiling are each ONE quad (two
    //    big triangles) running under the wall between them; a point light in room A, nothing in room B. Physically room B is black: every
    //    bit of light there is a leak. Measured: the solver's light of room B's surfels, what a ray hitting room B's floor reads (the hit
    //    triangle's vertex surfels: the solver's Indirect, the GPU's vertex read), and how much of the per-pixel read's weight on room B's
    //    floor comes from room A's surfels
    {
        constexpr float half_v = 0.05f;                                           // the wall: 10 cm thick at x = 0
        surfel_mesh Rooms;
        for (float x0 : { -4.0f, -2.0f, 2.0f })                                   // the floor and the ceiling: the middle piece, x -2..2, is one quad
        {                                                                         // (2 triangles) under the wall, its vertices 2 m into each room
            const float w = x0 == -2.0f ? 4.0f : 2.0f;
            AddQuad(Rooms, { x0, 0, -2 }, { 0, 0, 4 }, { w, 0, 0 }, 100);
            AddQuad(Rooms, { x0, 3, -2 }, { w, 0, 0 }, { 0, 0, 4 }, 100);
        }
        for (float s : { -1.0f, 1.0f })                                           // each room's other walls (facing in)
        {
            const float x0 = s < 0 ? -4.0f : half_v, x1 = s < 0 ? -half_v : 4.0f;
            AddQuad(Rooms, { x0, 0, -2 }, { x1 - x0, 0, 0 }, { 0, 3, 0 }, 0.5f);    // z = -2, facing +z
            AddQuad(Rooms, { x0, 0, 2 }, { 0, 3, 0 }, { x1 - x0, 0, 0 }, 0.5f);     // z = +2, facing -z
        }
        AddQuad(Rooms, { -4, 0, -2 }, { 0, 3, 0 }, { 0, 0, 4 }, 0.5f);              // x = -4, facing +x
        AddQuad(Rooms, { 4, 0, -2 }, { 0, 0, 4 }, { 0, 3, 0 }, 0.5f);               // x = +4, facing -x
        AddQuad(Rooms, { -half_v, 0, -2 }, { 0, 0, 4 }, { 0, 3, 0 }, 0.5f);         // the wall's room A side, facing -x
        AddQuad(Rooms, { half_v, 0, -2 }, { 0, 3, 0 }, { 0, 0, 4 }, 0.5f);          // its room B side, facing +x
        constexpr float shift_v = 0.25f;                                          // the wall in the middle of a read cell (0.5 m), not on its edge:
        for (auto& P : Rooms.m_Pos) P.m_X += shift_v;                             // the floor on both sides of it in one cell (the Police closet)
        Rooms.Bake(0.5f, 1.5f);
        Rooms.BuildReadGrid();                                                    // as the renderer has it (the compiler packs it)
        world W;
        const int iRooms = W.Add(Rooms, xmath::fmat4::fromIdentity(), { 0.7f, 0.7f, 0.7f });
        lights L; L.m_Points.push_back({ { -2 + shift_v, 1.5f, 0 }, { 20, 20, 20 }, 12 });
        for (int f = 0; f < 200; ++f) W.Update(L, { -2, 1.5f, 0 }, 1 << 30, 30000 + f);
        const auto& L0 = Rooms.m_Levels[0];
        const auto& E  = W.InstanceLight(iRooms);
        double SumA = 0, SumB = 0, SumBNear = 0; int nA = 0, nB = 0, nBNear = 0;
        for (std::size_t s = 0; s < L0.size(); ++s)
        {
            const float x = L0[s].m_Center.m_X - shift_v;
            if (x < -half_v) { SumA += Lum(E[s]); ++nA; }
            else if (x > half_v) { SumB += Lum(E[s]); ++nB; if (x < 1) { SumBNear += Lum(E[s]); ++nBNear; } }
        }
        const double MeanA = SumA / std::max(nA, 1), MeanB = SumB / std::max(nB, 1), MeanBNear = SumBNear / std::max(nBNear, 1);
        // what a ray hitting room B's floor reads, and the per-pixel read's weight from room A there (its listed surfels, the corner weights)
        double HitB = 0, WFromA = 0, WAll = 0; int nHit = 0;
        for (float x = half_v + 0.002f; x < 4; x += x < 0.1f ? 0.01f : x < 1 ? 0.05f : 0.25f)     // (finely by the wall: its foot, 2 mm on)
            for (float z = -1.75f; z < 2; z += 0.5f)
            {
                const v3 P{ x + shift_v, 0, z }, N{ 0, 1, 0 };
                HitB += Lum(CacheAt(W, P, N)); ++nHit;
                for (auto e : Rooms.ReadCellList(P, N))
                {
                    const float w = Rooms.ReadEntryWeight(e, P, N, N);
                    WAll += w;
                    if (L0[surfel_mesh::ReadEntrySurfel(e)].m_Center.m_X - shift_v < -half_v) WFromA += w;
                }
            }
        HitB /= std::max(nHit, 1);
        std::printf("leaks (room B should be black): room A's light %.3f; room B's surfels %.4f (%.1f%% of A; within 1 m of the wall %.1f%%); a ray hitting B's floor reads %.4f (%.1f%% of A); B's floor per-pixel read weight from room A %.1f%%\n"
                   , MeanA, MeanB, 100 * MeanB / MeanA, 100 * MeanBNear / MeanA, HitB, 100 * HitB / MeanA, 100 * WFromA / std::max(WAll, 1e-9));
        CHECK(MeanA > 0.1, "room A is not lit (%.3f)", MeanA);
        // (before the solver read through the read grid, world::ReadPoint: 5.2% in room B's surfels, 15.2% at a ray's hit on its floor; with
        // the wall across a read cell, before its patches were split there (BuildReadGrid's parts): 2.1% of the per-pixel read's weight)
        CHECK(MeanB < 0.005 * MeanA, "room A's light leaks into room B's surfels: %.2f%%", 100 * MeanB / MeanA);
        CHECK(HitB < 0.005 * MeanA, "a ray hitting room B's floor reads room A's light: %.2f%%", 100 * HitB / MeanA);
        CHECK(WFromA < 0.005 * WAll, "room B's floor reads room A's surfels per pixel (%.2f%% of its weight)", 100 * WFromA / std::max(WAll, 1e-9));
    }

    if (g_nErrors) { std::printf("%d checks failed\n", g_nErrors); return 1; }      // (CHECK counts and goes on: every failure is printed)
    std::printf("PASS\n");
    return 0;
}
