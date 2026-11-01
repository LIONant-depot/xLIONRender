// A preview's renderer (instance::setup::Preview(): the editor's preview panels and thumbnail runtimes - a sun pool of 256 slots, a local
// part of 32, no probe atlas) shades a small scene as the renderer's own sizes do. A sphere over a ground slab, a directional light:
//  - the shadow is there with the small pools (a patch of the ground clearly darker, no texel brighter)
//  - settled, the image equals the one a default-sized renderer makes of the same scene
//  - no validation error (the pool's net fits, nothing reads past the 1x1 probe stand-in)
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

    constexpr int size_v = 128;

    // The scene rendered until no page renders; Off: the sun without shadows, On: with
    bool Render(const xlionrender::instance::setup& Setup, std::vector<float>& Off, std::vector<float>& On)
    {
        app App;
        if (!App.Init(Setup)) return false;
        auto& Renderer = App.m_Renderer;
        xlionrender::mesh_id Sphere, Box;
        if (!MakeMesh(Renderer, MakeSphere(), Sphere) || !MakeMesh(Renderer, xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 }), Box)) return false;
        xlionrender::render_view View;
        if (Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v })) return false;

        xgpu::tools::view Camera;
        Camera.setFov(xmath::radian{ 1.0f });
        Camera.LookAt(xmath::fvec3{ 0, 2.5f, -3.0f }, xmath::fvec3{ 0, 0, 0 });
        xlionrender::scene Scene;
        const auto Ground = xmath::fmat4::fromTranslation({ 0, -0.55f, 0 }) * xmath::fmat4::fromScale({ 10.0f, 0.1f, 10.0f });
        Scene.m_Draws   = { { .m_Mesh = Box, .m_L2W = Ground }, { .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ 0, 0.3f, 0 }) } };
        Scene.m_Sun     = { .m_Direction = { 0.3f, -1.0f, 0.2f }, .m_Intensity = 3, .m_ShadowTileSize = 0.5f };
        Scene.m_Ambient = { 0.1f, 0.1f, 0.1f };

        Scene.m_Sun.m_bCastShadows = false;
        bool bOk = App.RenderFrame(Scene, View, Camera) && ReadHDR(App.m_Device, View, Off);
        Scene.m_Sun.m_bCastShadows = true;
        int k = 0;
        for (; bOk && k < 40; ++k)
        {
            bOk = App.RenderFrame(Scene, View, Camera);
            if (k >= 4 && Renderer.getShadowTilesRendered() == 0) break;
        }
        bOk = bOk && k < 40 && ReadHDR(App.m_Device, View, On);
        Renderer.DestroyView(View);
        return bOk;
    }
}

int main()
{
    std::vector<float> Off, On, FullOff, FullOn;
    CHECK(Render(xlionrender::instance::setup::Preview(), Off, On), "Render with the preview's sizes (settled)");
    CHECK(Render({}, FullOff, FullOn), "Render with the renderer's sizes (settled)");

    int nDarker = 0;
    for (std::size_t i = 0; i < On.size(); ++i)
    {
        CHECK(std::isfinite(On[i]), "Non-finite texel %d", int(i));
        CHECK(On[i] <= Off[i] * 1.001f + 1e-4f, "Texel %d got brighter with shadows (%f -> %f)", int(i), Off[i], On[i]);
        if (On[i] < Off[i] * 0.6f) ++nDarker;
    }
    std::printf("shadowed texels with the preview's pools: %d\n", nDarker);
    CHECK(nDarker > 150, "Only %d texels in shadow", nDarker);

    float MaxDiff = 0;
    for (std::size_t i = 0; i < On.size(); ++i) MaxDiff = std::max(MaxDiff, std::abs(On[i] - FullOn[i]));
    std::printf("preview vs renderer sizes: max difference %g\n", MaxDiff);
    CHECK(MaxDiff <= 1e-3f, "The preview's pools shade differently (max %g)", MaxDiff);

    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
