// Frustum culling (features/culling/xlionrender_frustum.h, the forward plan's scene pass):
//  - CPU: boxes in front, behind, beside, straddling and beyond far, against a camera's frustum
//  - GPU: a field of spheres around the camera: the culled image equals the image with culling off, with fewer draws
//  - GPU cluster culling (features/cull): the field as cluster meshes, culled on the GPU: the CPU path's image
#include "xlionrender_test_common.h"

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

    xmath::fbbox Box(const xmath::fvec3& C, float H) { return xmath::fbbox{ C - xmath::fvec3{ H, H, H }, C + xmath::fvec3{ H, H, H } }; }
}

int main()
{
    // CPU: the camera at the origin looking down +Z
    {
        xgpu::tools::view Camera;
        Camera.setViewport({ 0, 0, 128, 128 });
        Camera.setFov(xmath::radian{ 1.0f });
        Camera.LookAt(xmath::fvec3{ 0, 0, 0 }, xmath::fvec3{ 0, 0, 10 });
        const auto F = xlionrender::frustum::FromW2C(Camera.getW2C());
        CHECK( F.Overlaps(Box({ 0, 0, 10 }, 1)),     "A box in front is visible");
        CHECK(!F.Overlaps(Box({ 0, 0, -10 }, 1)),    "A box behind is culled");
        CHECK(!F.Overlaps(Box({ 50, 0, 10 }, 1)),    "A box far to the side is culled");
        CHECK(!F.Overlaps(Box({ 0, 60, 10 }, 1)),    "A box far above is culled");
        CHECK( F.Overlaps(Box({ 0, 0, 0 }, 1)),      "A box around the camera is visible");
        CHECK( F.Overlaps(Box({ 5.3f, 0, 10 }, 1)),  "A box straddling the side plane is visible");
    }

    app App;
    CHECK(App.Init(), "Init");
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    xlionrender::mesh_id Sphere;
    CHECK(MakeMesh(Renderer, MakeSphere(), Sphere), "sphere");

    constexpr int size_v = 128;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");

    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.0f });
    Camera.LookAt(xmath::fvec3{ 0, 0.5f, 0 }, xmath::fvec3{ 3, 0.3f, 10 });

    xlionrender::scene Scene;
    for (int z = -6; z <= 6; ++z) for (int x = -6; x <= 6; ++x)
        Scene.m_Draws.push_back({ .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ x * 2.0f, 0, z * 2.0f }) });
    Scene.m_Sun     = { .m_Direction = { 0.3f, -1.0f, 0.4f }, .m_Intensity = 3, .m_ShadowTileSize = 1.0f };
    Scene.m_Ambient = { 0.05f, 0.05f, 0.05f };

    std::vector<float> Culled, All;
    Renderer.m_bFrustumCull = false;
    CHECK(App.RenderFrame(Scene, View, Camera) && App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, All), "Render without culling");
    const int nAll = Renderer.getDrawsRendered();
    Renderer.m_bFrustumCull = true;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Culled), "Render with culling");
    const int nCulled = Renderer.getDrawsRendered();
    std::printf("draws: %d of %d with culling\n", nCulled, nAll);
    CHECK(nAll == int(Scene.m_Draws.size()), "Without culling every draw is drawn (%d)", nAll);
    CHECK(nCulled > 0 && nCulled < nAll / 2, "Culling kept %d of %d draws", nCulled, nAll);
    for (std::size_t i = 0; i < All.size(); ++i) CHECK(All[i] == Culled[i], "Culling changed texel %d (%f vs %f)", int(i), All[i], Culled[i]);

    // GPU culling (features/cull): the same field as cluster meshes with a material. The draws' own culling off, every cluster is a candidate
    // and the GPU culls them: the image equals the CPU path's (the draws culled on the CPU, drawn one by one)
    {
        static constexpr std::uint32_t s_Frag[] = {
            #include "xlionrender_test_material_frag.h"
        };
        xgpu::shader Frag;
        CHECK(!Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_Frag), std::size(s_Frag) } } }), "material shader");
        const std::uint32_t Texel = 0xff0080ffu;
        const auto          Mips  = std::array{ xgpu::texture::setup::mip_size{ sizeof(Texel) } };
        xgpu::texture       Texture;
        CHECK(!Device.Create(Texture, { .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = 1, .m_Height = 1, .m_MipSizes = Mips
                                      , .m_AllFacesData = { reinterpret_cast<const std::byte*>(&Texel), sizeof(Texel) }, .m_isGamma = false }), "material texture");
        xlionrender::material_id Material;
        const auto Slots = std::array<xgpu::texture*, 2>{ nullptr, &Texture };
        CHECK(!Renderer.CreateMaterial(Material, Frag, Slots), "CreateMaterial");

        cluster_buffers Buffers;
        CHECK(MakeStaticBuffers(Device, MakeSphere(), 0.5f, Buffers), "static geom buffers");
        xlionrender::mesh_id Clusters;
        CHECK(!Renderer.CreateClusterGeom(Clusters, Buffers.Geom()), "CreateClusterGeom");
        for (auto& D : Scene.m_Draws) { D.m_Mesh = Clusters; D.m_Materials = { Material }; }

        std::vector<float> CPU, GPU;
        Renderer.m_bFrustumCull = true; Renderer.m_bGPUCull = false;
        CHECK(App.RenderFrame(Scene, View, Camera) && App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, CPU), "Render on the CPU path");
        CHECK(Renderer.getCullStats().m_nCandidates == 0, "The CPU path hands nothing to the GPU cull");
        Renderer.m_bFrustumCull = false; Renderer.m_bGPUCull = true;
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, GPU), "Render GPU culled");
        const auto Stats = Renderer.getCullStats();
        std::printf("GPU cull: %d candidates, %d runs\n", Stats.m_nCandidates, Stats.m_nRuns);
        if (Device.hasDrawIndirectCount() && std::getenv("XLION_NO_GPU_CULL") == nullptr)
            CHECK(Stats.m_nCandidates == int(Scene.m_Draws.size()) && Stats.m_nRuns == (std::getenv("XLION_NO_PREPASS") ? 1 : 2), "Every cluster a candidate (%d), one run and its prepass run (%d)", Stats.m_nCandidates, Stats.m_nRuns);
        int nCovered = 0, nDiff = 0;
        for (std::size_t i = 0; i < CPU.size(); ++i)
        {
            nCovered += CPU[i] != 0;
            if (CPU[i] != GPU[i] && nDiff++ < 4) std::printf("texel %d (%d, %d): CPU %f, GPU %f\n", int(i), int(i) % size_v, int(i) / size_v, CPU[i], GPU[i]);
        }
        CHECK(nCovered > size_v * size_v / 10 && nDiff == 0, "GPU culled: %d of %d covered texels differ from the CPU path", nDiff, nCovered);
        // What the GPU kept, read back a few frames later: the spheres in the view (the CPU's draw test keeps a few more: a box against planes)
        for (int k = 0; k < 4; ++k) CHECK(App.RenderFrame(Scene, View, Camera), "Render GPU culled %d", k);
        const auto Kept = Renderer.getCullStats();
        std::printf("GPU cull kept %d of %d (prepass %d); the CPU's draw test %d\n", Kept.m_nDrawn, Kept.m_nCandidates, Kept.m_nPrepassDrawn, nCulled);
        if (Device.hasDrawIndirectCount() && std::getenv("XLION_NO_GPU_CULL") == nullptr)
            CHECK(Kept.m_nDrawn > 0 && Kept.m_nDrawn <= nCulled && Kept.m_nPrepassDrawn == (std::getenv("XLION_NO_PREPASS") ? 0 : Kept.m_nDrawn), "The GPU kept %d (prepass %d), the CPU's draw test %d", Kept.m_nDrawn, Kept.m_nPrepassDrawn, nCulled);

        Renderer.m_bFrustumCull = true;
        Renderer.DestroyMaterial(Material);
        Device.Destroy(std::move(Texture));
    }

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
