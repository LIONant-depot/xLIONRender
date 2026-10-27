// The sun's page/caster pairs (documentation/lighting_strategy.md P1c): a field of cluster meshes (many clusters each) under the sun's
// virtual pages, 50 of them moving. The gate's benchmark, warmed: the sun's CPU time and GPU time (the shadow passes) with 50 movers,
// the pages and the clusters drawn into them a frame
#include "xlionrender_test_common.h"
#include "features/shadows/xlionrender_shadow_vsm.h"

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

    // The mesh's triangles in clusters of TrisPerCluster (as the xgeom compilers make them), each with its box; one dequantization
    // entry a cluster (all the same)
    xlionrender::cluster_geom ClusterGeom(xgpu::device& Device, const xprim_geom::mesh& Prim, float Scale, cluster_buffers& B, int TrisPerCluster)
    {
        auto G = B.Geom();
        G.m_Clusters.clear();
        for (std::uint32_t i = 0; i < B.m_nIndices; i += 3u * TrisPerCluster)
        {
            const std::uint32_t n = std::min<std::uint32_t>(3u * TrisPerCluster, B.m_nIndices - i);
            xmath::fbbox Box{ xmath::fvec3(1e30f), xmath::fvec3(-1e30f) };
            for (std::uint32_t k = i; k < i + n; ++k)
            {
                const auto& P = Prim.m_Vertices[Prim.m_Indices[k]].m_Position;
                const xmath::fvec3 V(P.m_X, P.m_Y, P.m_Z);
                Box.AddVerts(std::span{ &V, 1 });
            }
            G.m_Clusters.push_back({ i, n, 0, Box });
        }
        const details::cluster_data D{ { Scale, Scale, Scale, 0 }, { 0, 0, 0, 0 }, { 1, 1, 0, 0 } };
        std::vector<details::cluster_data> All(G.m_Clusters.size(), D);
        B.m_ClusterData = {};
        (void)Device.Create(B.m_ClusterData, { .m_Type = xgpu::buffer::type::STORAGE, .m_EntryByteSize = sizeof(details::cluster_data), .m_EntryCount = int(All.size()), .m_pData = All.data() });
        G.m_Submeshes = { { 0, std::uint32_t(G.m_Clusters.size()), 0 } };
        return G;
    }
}

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    App.m_bQuietTimes = true;
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    static constexpr std::uint32_t s_Frag[] = {
        #include "xlionrender_test_material_frag.h"
    };
    xgpu::shader Frag;
    CHECK(!Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_Frag), std::size(s_Frag) } } }), "material shader");
    const std::uint32_t Texel = 0xffc0c0c0u;
    const auto          Mips  = std::array{ xgpu::texture::setup::mip_size{ sizeof(Texel) } };
    xgpu::texture       Texture;
    CHECK(!Device.Create(Texture, { .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = 1, .m_Height = 1, .m_MipSizes = Mips
                                  , .m_AllFacesData = { reinterpret_cast<const std::byte*>(&Texel), sizeof(Texel) }, .m_isGamma = false }), "material texture");
    xlionrender::material_id Material;
    const auto Slots = std::array<xgpu::texture*, 2>{ nullptr, &Texture };
    CHECK(!Renderer.CreateMaterial(Material, Frag, Slots), "CreateMaterial");

    // A sphere of 64 x 48 (6k triangles) in clusters of 128 triangles (48 clusters)
    const auto Prim = xprim_geom::uvsphere::Generate(64, 48, 1, 0.5f);
    cluster_buffers Buffers;
    CHECK(MakeStaticBuffers(Device, Prim, 0.5f, Buffers), "static geom buffers");
    xlionrender::mesh_id Clusters;
    const auto Geom = ClusterGeom(Device, Prim, 0.5f, Buffers, 128);
    CHECK(!Renderer.CreateClusterGeom(Clusters, Geom), "CreateClusterGeom");
    xlionrender::mesh_id Box;
    CHECK(MakeMesh(Renderer, xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 }), Box), "box");

    constexpr int width_v = 1280, height_v = 720;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = width_v, .m_Height = height_v }), "CreateView");
    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.1f });
    Camera.setViewport({ 0, 0, width_v, height_v });
    Camera.LookAt(xmath::fvec3{ 0, 7.0f, -14.0f }, xmath::fvec3{ 0, 0, 6.0f });

    // A ground, a field of 24 x 24 static spheres (2 m apart, 27k clusters), 50 movers among them
    xlionrender::scene Scene;
    Scene.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ 0, -0.55f, 20 }) * xmath::fmat4::fromScale({ 120.0f, 0.1f, 100.0f }) });
    for (int z = 0; z < 24; ++z) for (int x = 0; x < 24; ++x)
        Scene.m_Draws.push_back({ .m_Mesh = Clusters, .m_L2W = xmath::fmat4::fromTranslation({ -23.0f + 2.0f * x, 0.0f, -4.0f + 2.0f * z }), .m_Materials = { Material } });
    constexpr int movers_v = 50;
    const std::size_t iMovers = Scene.m_Draws.size();
    const auto Mover = [](int k, int f)
    {
        const float A = 0.05f * float(f) + 0.7f * float(k);
        return xmath::fmat4::fromTranslation({ -10.0f + 2.0f * float(k % 10) + 0.6f * std::cos(A), 1.2f, -2.0f + 3.0f * float(k / 10) + 0.6f * std::sin(A) }) * xmath::fmat4::fromScale({ 0.6f, 0.6f, 0.6f });
    };
    for (int k = 0; k < movers_v; ++k) Scene.m_Draws.push_back({ .m_Mesh = Clusters, .m_L2W = Mover(k, 0), .m_Materials = { Material } });
    Scene.m_Sun     = { .m_Direction = { 0.3f, -1.0f, 0.2f }, .m_Intensity = 3, .m_ShadowTileSize = 4.0f };
    Scene.m_Ambient = { 0.05f, 0.05f, 0.05f };

    // The benchmark: warmed (the pages settled, the movers moving), then the means of 120 frames
    const auto Bench = [&](const char* pWhat, bool bMove)
    {
        double Cpu = 0, Gpu = 0, Req = 0, Pages = 0;
        int    nStatic = 0, nMover = 0, nGpu = 0;
        for (int f = 0; f < 60 + 120; ++f)
        {
            if (bMove) for (int k = 0; k < movers_v; ++k) Scene.m_Draws[iMovers + k].m_L2W = Mover(k, f);
            if (!App.RenderFrame(Scene, View, Camera) || !Renderer.validateVSM()) return false;
            if (f < 60) continue;
            const auto& R = Renderer.getVSMReport();
            Cpu += Renderer.getCPUTimes().m_ShadowsMs;
            if (App.m_bGotTimes) { Gpu += App.m_LastTimes.m_ShadowMs; ++nGpu; }
            Req += R.m_RequestGpuMs; Pages += R.m_RenderGpuMs;
            nStatic += R.m_Pages.m_Rendered; nMover += R.m_Pages.m_RenderedMover;
        }
        const auto& P = Renderer.getPairsReport();
        std::printf("%s (%s): sun CPU %.3f ms, GPU: the pages rendered %.3f ms, every shadow pass (+ probes, cull) %.3f ms, request pass %.3f ms; pages rendered a frame: static %.1f, mover %.1f;"
                    " pairs: %d candidates, %d runs, %d pairs, %d dropped, %lld triangles\n"
                   , pWhat, Renderer.m_bShadowGPUPairs ? "GPU pairs" : "CPU", Cpu / 120, Pages / 120, nGpu ? Gpu / nGpu : 0.0, Req / 120, nStatic / 120.0, nMover / 120.0
                   , P.m_nCandidates, P.m_nRuns, P.m_nPairs, P.m_nDropped, (long long)P.m_Triangles);
        return true;
    };

    // The GPU pairs draw what the CPU path draws: every page rendered again by each (the cache off), the same frame - the images agree
    const auto Compare = [&](const char* pWhat)
    {
        std::vector<float> GPU, CPU;
        Renderer.m_bShadowCache = false;
        bool bOk = App.RenderFrame(Scene, View, Camera) && Renderer.validateVSM() && ReadHDR(Device, View, GPU);
        const bool bGPU = Renderer.getPairsReport().m_bGPU;
        Renderer.m_bShadowGPUPairs = false;
        bOk = bOk && App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, CPU);
        Renderer.m_bShadowGPUPairs = true;
        Renderer.m_bShadowCache    = true;
        if (!bOk) return false;
        int nDiff = 0, nShadowed = 0;
        for (std::size_t i = 0; i < CPU.size(); ++i) { nDiff += GPU[i] != CPU[i]; nShadowed += CPU[i] < 0.5f; }
        std::printf("%s: the GPU pairs' image differs from the CPU path's on %d values (%d dark) - the GPU drew the pages: %s\n", pWhat, nDiff, nShadowed, bGPU ? "yes" : "no");
        if (Device.hasDrawIndirectCount() && !bGPU) { std::printf("FAIL: %s: the GPU pairs did not draw the pages\n", pWhat); return false; }
        if (nDiff > int(CPU.size() / 2000)) { std::printf("FAIL: %s: the GPU pairs differ from the CPU path on %d values\n", pWhat, nDiff); return false; }
        return true;
    };
    CHECK(Compare("still"), "compare, still");

    CHECK(Bench("still", false), "the benchmark, still");
    {   // The sun's shadow in the scene pass (the pages' lookups, the contact-hardening's search and filters): its GPU time with and without
        const auto ScenePass = [&]()
        {
            double Sum = 0; int n = 0;
            for (int f = 0; f < 60 + 120; ++f)
            {
                if (!App.RenderFrame(Scene, View, Camera)) return -1.0;
                if (f >= 60 && App.m_bGotTimes) { Sum += App.m_LastTimes.m_SceneMs; ++n; }
            }
            return n ? Sum / n : -1.0;
        };
        const double With = ScenePass();
        Scene.m_Sun.m_bCastShadows = false;
        const double Without = ScenePass();
        Scene.m_Sun.m_bCastShadows = true;
        std::printf("scene pass (%dx%d): %.3f ms with the sun's shadow, %.3f without (the shadow %.3f ms)\n", width_v, height_v, With, Without, With - Without);
        CHECK(With > 0 && Without > 0 && ScenePass() > 0, "the scene pass with and without the shadow");
    }
    CHECK(Bench("50 movers", true), "the benchmark, 50 movers");
    CHECK(Compare("50 movers"), "compare, 50 movers");
    Renderer.m_bShadowGPUPairs = false;
    CHECK(Bench("50 movers", true), "the benchmark, 50 movers, the CPU path");
    Renderer.m_bShadowGPUPairs = true;

    // A point light moving over the field (a torch): its pages render again every frame - the CPU's time for them, the pages a frame
    {
        Scene.m_LocalLights = { { .m_Position = { 0.0f, 2.0f, 4.0f }, .m_Radius = 7.0f, .m_Intensity = 6.0f } };
        double Cpu = 0; int nPages = 0, n = 0;
        for (int f = 0; f < 60 + 120; ++f)
        {
            Scene.m_LocalLights[0].m_Position = { 3.0f * std::cos(0.03f * float(f)), 2.0f, 4.0f + 3.0f * std::sin(0.03f * float(f)) };
            CHECK(App.RenderFrame(Scene, View, Camera) && Renderer.validateLocalPages(), "a moving point light %d", f);
            if (f < 60) continue;
            Cpu += Renderer.getCPUTimes().m_ShadowsMs; nPages += int(Renderer.getLocalPagesRendered().size()); ++n;
        }
        const auto& C = Renderer.getLocalReport().m_Counters;
        std::printf("a moving point light (still movers): shadows CPU %.3f ms, local pages rendered a frame %.1f; its pixels by their level %u of %u\n", Cpu / n, double(nPages) / n, C[1], C[0]);
        CHECK(Compare("a point light"), "compare, a point light's pages");     // (its pages through the pairs against the CPU path)
        Scene.m_LocalLights.clear();
    }
    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
