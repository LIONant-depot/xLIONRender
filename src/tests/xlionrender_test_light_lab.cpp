// The lighting lab (research doc: the three-mode lab; features/lights/xlionrender_light_cells.h). Ground + a grid of spheres
// under many point and spot lights:
//  - the cell lists (CPU): every light whose sphere holds a point is in the list of the point's cell (support)
//  - EXACT vs CELLS: the same image (the lists are complete)
//  - few lights per cell (all in the head): SAMPLED == CELLS
//  - dense overlap: SAMPLED averaged over frames converges to EXACT (unbiased), and is noisy per frame (it does sample)
//  - prints, per mode: GPU scene time and the CPU cell build
#include "xlionrender_test_common.h"
#include <random>

using namespace xlionrender::test;

namespace
{
    bool MakeMesh(xlionrender::instance& Renderer, const xprim_geom::mesh& Prim, xlionrender::mesh_id& Id) noexcept
    {
        std::vector<xlionrender::vertex> Verts(Prim.m_Vertices.size());
        for (std::size_t i = 0; i < Verts.size(); ++i)
        {
            const auto& V = Prim.m_Vertices[i];
            Verts[i] = { { V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }, { V.m_Texcoord.m_X, V.m_Texcoord.m_Y }, 0xffffffffu
                       , { V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }, { V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, V.m_Tangent.m_D } };
        }
        return Renderer.CreateMesh(Id, Verts, Prim.m_Indices) == nullptr;
    }

    std::vector<xlionrender::local_light> MakeLights(int n, float MinRadius, float MaxRadius, unsigned Seed)
    {
        std::mt19937 R(Seed);
        std::uniform_real_distribution<float> X(-6, 6), Y(0.3f, 2.0f), Z(-4, 6), Rad(MinRadius, MaxRadius), C(0.2f, 1.0f), U(0, 1);
        std::vector<xlionrender::local_light> L(n);
        for (int i = 0; i < n; ++i)
        {
            auto& l = L[i];
            l.m_Position  = { X(R), Y(R), Z(R) };
            l.m_Radius    = Rad(R);
            l.m_Color     = { C(R), C(R), C(R) };
            l.m_Intensity = 2.0f;
            l.m_bCastShadows = i < 8;               // a few shadowed ones: the tail weights the shadowed contribution too
            if (U(R) < 0.3f)
            {
                l.m_Type       = xlionrender::local_light::type::SPOT;
                l.m_Direction  = { X(R) * 0.1f, -1.0f, Z(R) * 0.1f };
                l.m_InnerAngle = 0.4f;
                l.m_OuterAngle = 0.7f;
            }
        }
        return L;
    }

    struct stats { double m_MeanRel; double m_Max; };
    stats Compare(const std::vector<float>& A, const std::vector<float>& B)
    {
        double SumDiff = 0, SumRef = 0, Max = 0;
        for (std::size_t i = 0; i < A.size(); ++i) { const double d = std::abs(double(A[i]) - B[i]); SumDiff += d; SumRef += std::abs(B[i]); Max = std::max(Max, d); }
        return { SumDiff / std::max(SumRef, 1e-9), Max };
    }
}

int main()
{
    // The cell lists against brute force (CPU)
    {
        const auto L = MakeLights(300, 1.0f, 5.0f, 7);
        std::vector<xlionrender::light_cells::light> CL;
        for (auto& l : L) CL.push_back({ l.m_Position, l.m_Radius, 1.0f });
        xlionrender::light_cells Cells;
        const xmath::fvec3 Eye{ 0.5f, 3.0f, -2.0f };
        Cells.Build(CL, Eye, 2.0f, 0);
        std::mt19937 R(3);
        std::uniform_real_distribution<float> P(-14, 14);
        constexpr std::uint32_t entries_v = xlionrender::light_cells::levels_v * xlionrender::light_cells::level_cells_v * xlionrender::light_cells::header_uints_v;
        for (int t = 0; t < 2000; ++t)
        {
            const xmath::fvec3 Q{ P(R), P(R) * 0.3f + 1.0f, P(R) };
            int Cell = -1;
            for (int Lv = 0; Lv < xlionrender::light_cells::levels_v && Cell < 0; ++Lv)
            {
                const float S = 2.0f * float(1 << Lv);
                const int   C[3] = { int(std::floor(Q.m_X / S)), int(std::floor(Q.m_Y / S)), int(std::floor(Q.m_Z / S)) };
                bool bIn = true;
                for (int a = 0; a < 3; ++a) bIn &= C[a] >= Cells.m_Origin[Lv][a] && C[a] < Cells.m_Origin[Lv][a] + xlionrender::light_cells::cells_v;
                if (bIn) Cell = Lv * xlionrender::light_cells::level_cells_v + Cells.Slot(C[0], C[1], C[2]);
            }
            if (Cell < 0) continue;
            const std::uint32_t First = Cells.m_Data[Cell * 4], Count = Cells.m_Data[Cell * 4 + 1];
            for (std::uint32_t i = 0; i < L.size(); ++i)
            {
                if ((L[i].m_Position - Q).Length() >= L[i].m_Radius) continue;
                bool bListed = false;
                for (std::uint32_t k = 0; k < Count && !bListed; ++k) bListed = Cells.m_Data[entries_v + 2 * (First + k)] == i;
                CHECK(bListed, "Light %u reaches point %d but is not in its cell's list", i, t);
            }
            for (std::uint32_t k = 1; k < Count; ++k) CHECK(Cells.m_Data[entries_v + 2 * (First + k)] > Cells.m_Data[entries_v + 2 * (First + k - 1)], "A cell's list is not in light order");
        }

        // Boundaries (the invariant: a list may hold extra lights, never miss one that contributes): points just inside each
        // light's radius in 26 directions, just inside spot cones, lights at negative coordinates, a light enclosing the camera,
        // a very narrow cone, a camera far in negative space (other cell windows and slots), every level
        {
            std::vector<xlionrender::local_light> B = MakeLights(64, 0.5f, 9.0f, 21);
            B.push_back({ .m_Position = { 0.5f, 3.0f, -2.0f }, .m_Radius = 40.0f });                                            // encloses the camera
            B.push_back({ .m_Type = xlionrender::local_light::type::SPOT, .m_Position = { -3, 2, -5 }, .m_Direction = { 0.2f, -1, 0.3f }
                        , .m_Radius = 12.0f, .m_InnerAngle = 0.02f, .m_OuterAngle = 0.05f });                                     // very narrow
            for (auto& l : B) if (l.m_Type == xlionrender::local_light::type::SPOT) l.m_Direction = l.m_Direction.NormalizeCopy();
            const auto Contributes = [](const xlionrender::local_light& l, const xmath::fvec3& Q)
            {
                const xmath::fvec3 V = Q - l.m_Position;
                const float d = V.Length();
                if (d >= l.m_Radius) return false;
                return l.m_Type == xlionrender::local_light::type::POINT || (d > 0 && V.Dot(l.m_Direction) / d > std::cos(l.m_OuterAngle));
            };
            for (const xmath::fvec3 Eye2 : { xmath::fvec3{ 0.5f, 3.0f, -2.0f }, xmath::fvec3{ -37.3f, -11.0f, -58.9f } })
            {
                std::vector<xlionrender::light_cells::light> BL;
                for (auto& l : B) BL.push_back({ l.m_Position + (Eye2 - xmath::fvec3{ 0.5f, 3.0f, -2.0f }), l.m_Radius, 1.0f, l.m_Direction
                                               , l.m_Type == xlionrender::local_light::type::SPOT ? l.m_OuterAngle : -1.0f });
                Cells.Build(BL, Eye2, 2.0f, 0);
                const auto CellOf = [&](const xmath::fvec3& Q)
                {
                    for (int Lv = 0; Lv < xlionrender::light_cells::levels_v; ++Lv)
                    {
                        const float S = 2.0f * float(1 << Lv);
                        const int   C[3] = { int(std::floor(Q.m_X / S)), int(std::floor(Q.m_Y / S)), int(std::floor(Q.m_Z / S)) };
                        bool bIn = true;
                        for (int a = 0; a < 3; ++a) bIn &= C[a] >= Cells.m_Origin[Lv][a] && C[a] < Cells.m_Origin[Lv][a] + xlionrender::light_cells::cells_v;
                        if (bIn) return Lv * xlionrender::light_cells::level_cells_v + Cells.Slot(C[0], C[1], C[2]);
                    }
                    return -1;
                };
                int nChecked = 0;
                for (std::uint32_t i = 0; i < BL.size(); ++i)
                {
                    xlionrender::local_light l = B[i];
                    l.m_Position = BL[i].m_Position;
                    std::vector<xmath::fvec3> Points;
                    for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y) for (int z = -1; z <= 1; ++z)
                        if (x || y || z) Points.push_back(l.m_Position + xmath::fvec3(float(x), float(y), float(z)).NormalizeCopy() * (l.m_Radius * 0.9999f));
                    if (l.m_Type == xlionrender::local_light::type::SPOT)
                    {
                        xmath::fvec3 R, U;
                        xlionrender::shadows::FaceBasis(l.m_Direction, R, U);
                        for (int k = 0; k < 16; ++k)
                        {
                            const float A = k * 0.3927f, Edge = l.m_OuterAngle * 0.999f;
                            const xmath::fvec3 Dir = (l.m_Direction * std::cos(Edge) + (R * std::cos(A) + U * std::sin(A)) * std::sin(Edge)).NormalizeCopy();
                            for (float t : { 0.1f, 0.5f, 0.99f }) Points.push_back(l.m_Position + Dir * (l.m_Radius * t));
                        }
                    }
                    for (const auto& Q : Points)
                    {
                        if (!Contributes(l, Q)) continue;
                        const int Cell = CellOf(Q);
                        if (Cell < 0) continue;                                         // outside the cells: the shader loops every light
                        const std::uint32_t First = Cells.m_Data[Cell * 4], Count = Cells.m_Data[Cell * 4 + 1];
                        bool bListed = false;
                        for (std::uint32_t k = 0; k < Count && !bListed; ++k) bListed = Cells.m_Data[entries_v + 2 * (First + k)] == i;
                        CHECK(bListed, "Boundary: light %u reaches (%f %f %f) but is not in its cell's list", i, Q.m_X, Q.m_Y, Q.m_Z);
                        ++nChecked;
                    }
                }
                CHECK(nChecked > 500, "Only %d boundary points checked", nChecked);
            }
        }

        // SAMPLED split: the head + the tail's cumulative probabilities end at 1 and grow
        Cells.Build(CL, Eye, 2.0f, 4);
        int nSplit = 0;
        for (int c = 0; c < xlionrender::light_cells::levels_v * xlionrender::light_cells::level_cells_v; ++c)
        {
            const std::uint32_t First = Cells.m_Data[c * 4], Count = Cells.m_Data[c * 4 + 1], Head = Cells.m_Data[c * 4 + 2];
            CHECK(Head == std::min<std::uint32_t>(Count, 4), "Cell %d: head %u of %u", c, Head, Count);
            if (Count <= Head) continue;
            ++nSplit;
            float Prev = 0;
            for (std::uint32_t k = Head; k < Count; ++k)
            {
                const float Cdf = std::bit_cast<float>(Cells.m_Data[entries_v + 2 * (First + k) + 1]);
                CHECK(Cdf > Prev, "Cell %d: a tail light without probability", c);
                Prev = Cdf;
            }
            CHECK(Prev == 1.0f, "Cell %d: the tail's probabilities end at %f", c, Prev);
        }
        CHECK(nSplit > 100, "Only %d cells have a tail", nSplit);
    }

    app App;
    CHECK(App.Init(), "Init");
    App.m_bQuietTimes = true;
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    xlionrender::mesh_id Sphere, Box;
    CHECK(MakeMesh(Renderer, MakeSphere(), Sphere), "sphere");
    CHECK(MakeMesh(Renderer, xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 }), Box), "box");

    constexpr int size_v = 160;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");

    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.1f });
    Camera.LookAt(xmath::fvec3{ 0, 6.0f, -9.0f }, xmath::fvec3{ 0, 0, 1.0f });

    xlionrender::scene Scene;
    Scene.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ 0, -0.55f, 1 }) * xmath::fmat4::fromScale({ 20.0f, 0.1f, 20.0f }) });
    for (int z = 0; z < 4; ++z) for (int x = 0; x < 5; ++x)
        Scene.m_Draws.push_back({ .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ -4.0f + 2.0f * x, 0.0f, -2.0f + 2.5f * z }), .m_Material = { .m_Metallic = (z % 2) ? 0.8f : 0.0f, .m_Roughness = (x % 2) ? 0.25f : 0.7f } });
    Scene.m_Sun     = { .m_Intensity = 0, .m_bCastShadows = false };
    Scene.m_Ambient = { 0.02f, 0.02f, 0.02f };

    using mode = xlionrender::instance::light_mode;
    const auto RenderMode = [&](mode M, std::vector<float>& Out, int nSettle = 3) -> bool
    {
        Renderer.m_LightMode = M;
        for (int i = 0; i < nSettle; ++i) if (!App.RenderFrame(Scene, View, Camera)) return false;     // shadow maps, timings
        return App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Out);
    };

    // Few lights per cell (small radii): all of a cell's lights fit in the head -> SAMPLED == CELLS == EXACT
    Scene.m_LocalLights = MakeLights(24, 0.8f, 1.6f, 11);
    {
        std::vector<float> Exact, Cells, Sampled;
        CHECK(RenderMode(mode::EXACT, Exact) && RenderMode(mode::CELLS, Cells) && RenderMode(mode::SAMPLED, Sampled), "Render few lights");
        const auto C = Compare(Cells, Exact), S = Compare(Sampled, Cells);
        std::printf("few lights: cells vs exact %.2e (max %.2e), sampled vs cells %.2e (max %.2e)\n", C.m_MeanRel, C.m_Max, S.m_MeanRel, S.m_Max);
        CHECK(C.m_Max < 1e-4, "CELLS differs from EXACT (max %e)", C.m_Max);
        CHECK(S.m_Max < 1e-4, "SAMPLED differs from CELLS with every light in the heads (max %e)", S.m_Max);
    }

    // Dense overlap: 256 lights of radius 3-7
    Scene.m_LocalLights = MakeLights(256, 3.0f, 7.0f, 5);
    std::vector<float> Exact, Cells;
    CHECK(RenderMode(mode::EXACT, Exact), "Render exact");
    const double ExactMs = App.m_LastTimes.m_SceneMs;
    CHECK(RenderMode(mode::CELLS, Cells), "Render cells");
    const double CellsMs = App.m_LastTimes.m_SceneMs, CellsBuild = Renderer.getLightBuildMs();
    const auto C = Compare(Cells, Exact);
    std::printf("dense: cells vs exact %.2e (max %.2e)\n", C.m_MeanRel, C.m_Max);
    CHECK(C.m_Max < 1e-3 && C.m_MeanRel < 1e-5, "CELLS differs from EXACT (mean %e, max %e)", C.m_MeanRel, C.m_Max);

    // SAMPLED: one frame is noisy; the average over frames converges to EXACT
    std::vector<float> Frame, Sum(Exact.size(), 0.0f);
    CHECK(RenderMode(mode::SAMPLED, Frame), "Render sampled");
    const double SampledMs = App.m_LastTimes.m_SceneMs, SampledBuild = Renderer.getLightBuildMs();
    const auto One = Compare(Frame, Exact);
    Dump("lab_exact", Exact); Dump("lab_sampled_1", Frame);
    double Err8 = 0, Err64 = 0;
    for (int f = 1; f <= 64; ++f)
    {
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Frame), "Render sampled %d", f);
        for (std::size_t i = 0; i < Sum.size(); ++i) Sum[i] += Frame[i];
        if (f == 8 || f == 64)
        {
            std::vector<float> Mean(Sum.size());
            for (std::size_t i = 0; i < Sum.size(); ++i) Mean[i] = Sum[i] / float(f);
            (f == 8 ? Err8 : Err64) = Compare(Mean, Exact).m_MeanRel;
            if (f == 64) Dump("lab_sampled_64", Mean);
        }
    }
    std::printf("dense sampled: 1 frame %.3f, mean of 8 %.4f, mean of 64 %.4f (relative mean error)\n", One.m_MeanRel, Err8, Err64);
    CHECK(One.m_MeanRel > 1e-4, "SAMPLED is not sampling (one frame equals EXACT)");
    // Unbiased: the error of a mean of N frames falls as 1/sqrt(N) - 8 -> 64 frames is x0.354 (a bias would flatten it)
    CHECK(Err8 < One.m_MeanRel * 0.5 && Err64 < Err8 * 0.5, "SAMPLED does not converge to EXACT (1: %f, 8: %f, 64: %f)", One.m_MeanRel, Err8, Err64);

    // More stratified tail samples: same estimator, less noise per frame
    double OneBySamples[3] = {};
    for (int si = 0; si < 3; ++si)
    {
        Renderer.m_LightTailSamples = 1 << (2 * si);           // 1, 4, 16
        CHECK(RenderMode(mode::SAMPLED, Frame, 1), "Render sampled x%d", Renderer.m_LightTailSamples);
        OneBySamples[si] = Compare(Frame, Exact).m_MeanRel;
        std::printf("tail samples %2d: 1 frame %.3f, GPU scene %.3f ms\n", Renderer.m_LightTailSamples, OneBySamples[si], App.m_LastTimes.m_SceneMs);
        if (si == 1) Dump("lab_sampled_4", Frame);
    }
    Renderer.m_LightTailSamples = 1;
    CHECK(OneBySamples[1] < OneBySamples[0] && OneBySamples[2] < OneBySamples[1], "More tail samples do not reduce the noise");

    std::printf("GPU scene: exact %.3f ms, cells %.3f ms, sampled %.3f ms (%dx%d); CPU cell build: cells %.3f ms, sampled %.3f ms (%d lights)\n"
               , ExactMs, CellsMs, SampledMs, size_v, size_v, CellsBuild, SampledBuild, int(Scene.m_LocalLights.size()));

    // Benchmark (XLIONRENDER_LAB_BENCH=1; build Release): 1280x720, 256 / 1024 / 4096 lights over a wider area, each mode: GPU
    // scene time (mean of 16 frames), CPU cell build, cell data size, the mode actually used (a full cell buffer falls back)
    if (std::getenv("XLIONRENDER_LAB_BENCH"))
    {
        xlionrender::render_view Big;
        CHECK(!Renderer.CreateView(Big, { .m_Width = 1280, .m_Height = 720 }), "CreateView 1280x720");
        Camera.LookAt(xmath::fvec3{ 0, 10.0f, -24.0f }, xmath::fvec3{ 0, 0, 6.0f });
        for (int n : { 256, 1024, 4096 })
        {
            auto Lights = MakeLights(n, 2.0f, 5.0f, 99);
            for (auto& l : Lights) { l.m_Position.m_X *= 4.0f; l.m_Position.m_Z = l.m_Position.m_Z * 4.0f + 10.0f; }
            Scene.m_LocalLights = Lights;
            for (mode M : { mode::EXACT, mode::CELLS, mode::SAMPLED })
            {
                Renderer.m_LightMode = M;
                double Gpu = 0, Build = 0; int nGpu = 0;
                for (int f = 0; f < 24; ++f)
                {
                    CHECK(App.RenderFrame(Scene, Big, Camera), "Bench render");
                    if (f >= 8) { Gpu += App.m_LastTimes.m_SceneMs; Build += Renderer.getLightBuildMs(); ++nGpu; }
                }
                static const char* Names[] = { "EXACT", "CELLS", "SAMPLED" };
                std::printf("bench %4d lights %-7s: GPU scene %7.3f ms, CPU cell build %6.3f ms, cell data %7.1f KB, used %s\n"
                           , n, Names[int(M)], Gpu / nGpu, Build / nGpu, Renderer.getLightCellUints() * 4 / 1024.0, Names[int(Renderer.getLightModeUsed())]);
            }
        }
        Renderer.m_LightMode = mode::EXACT;
        Renderer.DestroyView(Big);
    }

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
