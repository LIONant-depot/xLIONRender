// Surfel GI on the CPU (features/gi/xlionrender_surfel_gi.h), no GPU:
//  - the BVHs (mesh + instances) return the same hits as a brute-force cast
//  - a Cornell box built from instances (the box rotated) converges to a path-traced reference
//  - a rotated instance equals the same geometry pre-transformed (transforms reach surfels, normals and reads)
//  - a city (400 buildings, sun + sky): cost per update, coverage, LOD levels, and a moved building dirtying only its block
//  - surfels saved by the GeomStatic compiler load back as baked, and a receive-only mesh is not in the rays' way
#include "features/gi/xlionrender_surfel_gi.h"
#include <cstdio>
#include <random>
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
                    if ((P - Loaded.m_Levels[0][s].m_Center).Length() < 2 * Loaded.m_Levels[0][s].m_Radius && Loaded.m_Levels[0][s].m_Normal.Dot((Loaded.m_Pos[T[1]] - Loaded.m_Pos[T[0]]).Cross(Loaded.m_Pos[T[2]] - Loaded.m_Pos[T[0]]).NormalizeCopy()) >= 0.5f && std::find(List.begin(), List.end(), std::uint32_t(s)) == List.end()) ++nMissing;
            }
        std::printf("read grid: %zu cells, %zu list entries, at most %d surfels per cell; %d of the surfels reaching %d surface points not listed\n"
                   , std::count_if(Loaded.m_ReadTable.begin(), Loaded.m_ReadTable.end(), [](const auto& E) { return E[1] != 0xFFFFFFFFu; }), Loaded.m_ReadList.size(), MaxList, nMissing, nPoints);
        CHECK(nMissing == 0, "the read grid misses %d surfels that reach a surface point", nMissing);

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
                for (auto s : Room.ReadCellList(P, N))
                {
                    const auto& S = Room.m_Levels[0][s];
                    bCovered = bCovered || (S.m_Normal.Dot(N) >= 0.5f && (P - S.m_Center).Length() < 2 * S.m_Radius);
                }
                nUncovered += !bCovered;
            }
        }
        std::printf("coverage: %d of %d surface points read no surfel\n", nUncovered, nPoints);
        for (const auto& E : Room.Coverage(4).m_Examples) std::printf("  hole %.3f %.3f %.3f N %.2f %.2f %.2f: %d listed, nearest compatible %.3f, any %.3f\n", E.m_P.m_X, E.m_P.m_Y, E.m_P.m_Z, E.m_N.m_X, E.m_N.m_Y, E.m_N.m_Z, E.m_nList, E.m_NearestCompatible, E.m_NearestAny);
        CHECK(nUncovered == 0, "%d of %d surface points read no surfel (black in the debug view)", nUncovered, nPoints);
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
    std::printf("PASS\n");
    return 0;
}
