// Frustum culling (features/culling/xlionrender_frustum.h, the forward plan's scene pass):
//  - CPU: boxes in front, behind, beside, straddling and beyond far, against a camera's frustum
//  - GPU: a field of spheres around the camera: the culled image equals the image with culling off, with fewer draws
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

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
