// Surfel GI on the CPU (features/gi/xlionrender_surfel_gi.h), no GPU:
//  - the BVHs (mesh + instances) return the same hits as a brute-force cast
//  - a Cornell box built from instances (the box rotated) converges to a path-traced reference
//  - a rotated instance equals the same geometry pre-transformed (transforms reach surfels, normals and reads)
//  - a city (400 buildings, sun + sky): cost per update, coverage, LOD levels, and a moved building dirtying only its block
//  - surfels saved by the GeomStatic compiler load back as baked, and a receive-only mesh is not in the rays' way
#include "features/gi/xlionrender_surfel_gi.h"
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
        std::printf("city steady: %.1f ms per update; a building moved: %d instances dirty (of %d), updated in %.1f ms\n", Steady / 10, St.m_Dirty, W.InstanceCount(), St.m_Ms);
        CHECK(St.m_Dirty > 1 && St.m_Dirty < 15, "a moved building dirtied %d instances", St.m_Dirty);
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
                if (A.m_OriginFirst != B.m_OriginFirst || A.m_OriginPos.size() != B.m_OriginPos.size()) return false;
                for (std::size_t k = 0; k < A.m_OriginPos.size(); ++k) if ((A.m_OriginPos[k] - B.m_OriginPos[k]).Length() > 0 || (A.m_OriginNrm[k] - B.m_OriginNrm[k]).Length() > 0) return false;
                return true;
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
        const auto Run = [&](bool bLinks, int Rays, int Updates, double& Spread)
        {
            world D;
            D.m_bEmitterLinks = bLinks;
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

    // 6j7. The level by the screen area (setup::m_ProjScale, m_PixelsPerSurfel): the pixels an instance's bounding sphere covers per fine surfel. A small
    //      box holds its parent level where it covers too few, the fine one nearer, and it comes back at 1.25 times the area it left at (no flicker)
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
        const float Radius = 0.5f * std::sqrt(0.12f);                                                           // the bounds' radius
        const float dStar  = std::sqrt(pi_v * Radius * Radius * 1.0e6f / (256.0f * float(Box.m_Levels[0].size())));   // the distance (from the near face + radius) it leaves at
        const auto LevelAt = [&](float x) { W.Update(L, { 0, 0.1f, x * dStar - Radius + 0.1f }, 1 << 20, 20000); return W.Level(i); };
        const int Near = LevelAt(0.5f), Far = LevelAt(2.0f), Between = LevelAt(0.95f), Back = LevelAt(0.5f);
        std::printf("surfel level by screen area (%zu fine surfels, leaves at %.2f m): near %d, far %d, then inside the hysteresis %d, near again %d\n", Box.m_Levels[0].size(), dStar, Near, Far, Between, Back);
        CHECK(Near == 0, "a box covering many pixels is not at the fine level (%d)", Near);
        CHECK(Far == 1, "a box covering few pixels is not at the parent level (%d)", Far);
        CHECK(Between == 1, "the level flickered back inside the hysteresis (%d)", Between);
        CHECK(Back == 0, "the box did not come back to the fine level when near (%d)", Back);
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
