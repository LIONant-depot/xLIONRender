// Point and spot lights and their atlas shadows (features/shadows/xlionrender_shadow_local.h). A sphere over a ground slab, no sun:
//  - a point light lights the scene (brighter than without it)
//  - its shadow is there: with shadows on no texel gets brighter and a patch of the ground gets clearly darker
//  - maps: the first frame renders one per shadowed light (octahedral for a point light), an unchanged frame none (same image)
//  - invalidation: the sphere moves -> only the light whose radius reaches it re-renders its map, and the cached image equals
//    the image with every map re-rendered (m_bShadowCache = false)
//  - a spot light: its map, and its shadow is there too
//  - the per-frame budget: 3 lights, 2 maps per frame -> 2, then 1, then 0, and then the same image as a full rebuild
//  - the buddy allocator the atlas uses: fills up, frees back into one block
//  - no starvation: a small light whose map stays dirty while a big one is dirty every frame gets its update within
//    urgent_age_v frames under a budget of 1
//  - a caster changing material (same mesh, same place) re-renders the maps it reaches
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

    // Texels clearly darker in On than in Off; -1 when one is not finite or got brighter
    int CountDarker(const std::vector<float>& Off, const std::vector<float>& On) noexcept
    {
        int nDarker = 0;
        for (std::size_t i = 0; i < On.size(); ++i)
        {
            if (!std::isfinite(On[i]) || On[i] > Off[i] * 1.001f + 1e-4f)
            {
                std::printf("texel %d: %f -> %f\n", int(i), Off[i], On[i]);
                return -1;
            }
            if (On[i] < Off[i] * 0.6f) ++nDarker;
        }
        return nDarker;
    }
}

int main()
{
    // The atlas allocator (CPU only)
    {
        xlionrender::shadows::buddy B;
        B.Reset(4096, 64);
        std::vector<xlionrender::shadows::buddy::pos> P(16);
        for (auto& Q : P) CHECK(B.Alloc(1024, Q), "16 maps of 1024 fit in 4096");
        xlionrender::shadows::buddy::pos Q;
        CHECK(!B.Alloc(64, Q), "A full atlas takes nothing more");
        for (int i = 0; i < 16; ++i) for (int j = i + 1; j < 16; ++j) CHECK(P[i].m_X != P[j].m_X || P[i].m_Y != P[j].m_Y, "Two maps at the same place");
        for (auto& R : P) B.Free(R, 1024);
        CHECK(B.Alloc(4096, Q) && Q.m_X == 0 && Q.m_Y == 0, "Freed maps merge back into the whole atlas");
    }

    // No starvation (CPU only)
    {
        xlionrender::shadows::local_shadows S;
        S.Init({});
        const std::vector<xlionrender::local_light> L
        { { .m_Position = { 0, 0, 2 },   .m_Radius = 5 }        // big on screen
        , { .m_Position = { 0, 0, 40 },  .m_Radius = 5 } };     // small, far away
        const std::vector<xmath::fbbox> Both{ xmath::fbbox{ xmath::fvec3{ -1, -1, -1 }, xmath::fvec3{ 1, 1, 50 } } };
        std::vector<xlionrender::shadows::local_map>           Maps;
        std::vector<xlionrender::shadows::local_map_to_render> Out;
        S.Update(L, {}, true, { 0, 0, 0 }, 100.0f, 16, Maps, Out);
        CHECK(Out.size() == 2, "Both maps render first (%d)", int(Out.size()));
        int SmallAt = -1;
        for (int f = 1; f <= 2 * xlionrender::shadows::urgent_age_v && SmallAt < 0; ++f)
        {
            S.Update(L, Both, true, { 0, 0, 0 }, 100.0f, 1, Maps, Out);
            CHECK(Out.size() == 1, "A budget of 1 renders 1 map (%d)", int(Out.size()));
            if (Out[0].m_Light == 1) SmallAt = f;
        }
        CHECK(SmallAt > 0 && SmallAt <= xlionrender::shadows::urgent_age_v + 1, "The small light waited %d frames", SmallAt);
    }

    app App;
    CHECK(App.Init(), "Init");
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    xlionrender::mesh_id Sphere, Box;
    CHECK(MakeMesh(Renderer, MakeSphere(), Sphere), "sphere");
    CHECK(MakeMesh(Renderer, xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 }), Box), "box");

    constexpr int size_v = 128;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");

    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.0f });
    Camera.LookAt(xmath::fvec3{ 0, 2.5f, -3.0f }, xmath::fvec3{ 0, 0, 0 });

    xlionrender::scene Scene;
    const auto Ground = xmath::fmat4::fromTranslation({ 0, -0.55f, 0 }) * xmath::fmat4::fromScale({ 10.0f, 0.1f, 10.0f });
    const auto Lifted = [](float X) { return xmath::fmat4::fromTranslation({ X, 0.3f, 0 }); };
    Scene.m_Draws = { { .m_Mesh = Box, .m_L2W = Ground }, { .m_Mesh = Sphere, .m_L2W = Lifted(0) } };
    Scene.m_Sun   = { .m_Intensity = 0, .m_bCastShadows = false };
    Scene.m_Ambient = { 0.05f, 0.05f, 0.05f };

    // The point light that matters, and one far away whose radius reaches nothing that moves
    const xlionrender::local_light Point{ .m_Position = { 0.5f, 1.6f, 0.4f }, .m_Radius = 6.0f, .m_Intensity = 6.0f };
    const xlionrender::local_light Far  { .m_Position = { 20.0f, 1.0f, 0.0f }, .m_Radius = 2.0f, .m_Intensity = 6.0f };

    // The light lights
    std::vector<float> Dark, Off, On;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Dark), "Render without lights");
    Scene.m_LocalLights = { Point, Far };
    Scene.m_LocalLights[0].m_bCastShadows = Scene.m_LocalLights[1].m_bCastShadows = false;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Off), "Render without shadows");
    CHECK(Renderer.getShadowTilesRendered() == 0, "Lights without shadows rendered %d tiles", Renderer.getShadowTilesRendered());
    int nLit = 0;
    for (std::size_t i = 0; i < Off.size(); ++i) nLit += Off[i] > Dark[i] * 1.5f + 0.01f;
    std::printf("texels lit by the point light: %d\n", nLit);
    CHECK(nLit > size_v * size_v / 4, "The point light lit only %d texels", nLit);

    // Its shadow
    Scene.m_LocalLights[0].m_bCastShadows = Scene.m_LocalLights[1].m_bCastShadows = true;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, On), "Render with shadows");
    CHECK(Renderer.getShadowTilesRendered() == 2, "The first frame renders a map per point light (%d)", Renderer.getShadowTilesRendered());
    Dump("point_off", Off); Dump("point_on", On);
    const int nDarker = CountDarker(Off, On);
    std::printf("point shadowed texels: %d\n", nDarker);
    CHECK(nDarker > 150, "Only %d texels in the point light's shadow", nDarker);

    // Caching
    std::vector<float> Again;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Again), "Render again");
    CHECK(Renderer.getShadowTilesRendered() == 0, "An unchanged scene re-rendered %d tiles", Renderer.getShadowTilesRendered());
    for (std::size_t i = 0; i < On.size(); ++i) CHECK(Again[i] == On[i], "The cached frame differs at %d", int(i));

    // The sphere moves: the near light's 6 faces re-render, the far light's do not; cached equals rebuilt
    Scene.m_Draws[1].m_L2W = Lifted(0.4f);
    std::vector<float> Cached, Rebuilt;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Cached), "Render moved (cached)");
    CHECK(Renderer.getShadowTilesRendered() == 1, "A moved caster re-rendered %d maps (expected the near light's)", Renderer.getShadowTilesRendered());
    Renderer.m_bShadowCache = false;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Rebuilt), "Render moved (rebuilt)");
    CHECK(Renderer.getShadowTilesRendered() == 2, "Without the cache every map renders (%d)", Renderer.getShadowTilesRendered());
    Renderer.m_bShadowCache = true;
    for (std::size_t i = 0; i < Cached.size(); ++i) CHECK(Cached[i] == Rebuilt[i], "Cached and rebuilt shadows differ at %d (%f vs %f)", int(i), Cached[i], Rebuilt[i]);
    int nChanged = 0;
    for (std::size_t i = 0; i < Cached.size(); ++i) nChanged += Cached[i] != On[i];
    CHECK(nChanged > 0, "Moving the sphere changed nothing on screen");

    // A caster changing material re-renders the maps it reaches (an alpha-tested material casts another shadow)
    {
        const xlionrender::material_id Mat{ .m_Index = 7, .m_Generation = 1 };     // the key only (the float mesh ignores slots)
        CHECK(App.RenderFrame(Scene, View, Camera) && Renderer.getShadowTilesRendered() == 0, "Settled before the material change");
        Scene.m_Draws[1].m_Materials = { Mat };
        CHECK(App.RenderFrame(Scene, View, Camera), "Render with the new material");
        CHECK(Renderer.getShadowTilesRendered() == 1, "A material change re-rendered %d maps (expected the near light's)", Renderer.getShadowTilesRendered());
        Scene.m_Draws[1].m_Materials.clear();
        CHECK(App.RenderFrame(Scene, View, Camera), "Render back");
    }

    // A spot light pointing down at the sphere: 1 tile, a shadow
    xlionrender::local_light Spot{ .m_Type = xlionrender::local_light::type::SPOT, .m_Position = { 0.3f, 2.0f, 0.2f }, .m_Direction = { -0.1f, -1.0f, -0.1f }
                                 , .m_Radius = 6.0f, .m_InnerAngle = 0.6f, .m_OuterAngle = 0.8f, .m_Intensity = 8.0f, .m_bCastShadows = false };
    Scene.m_LocalLights = { Spot };
    std::vector<float> SpotOff, SpotOn;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, SpotOff), "Render spot without shadows");
    Scene.m_LocalLights[0].m_bCastShadows = true;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, SpotOn), "Render spot with shadows");
    CHECK(Renderer.getShadowTilesRendered() == 1, "A spot light renders 1 tile (%d)", Renderer.getShadowTilesRendered());
    Dump("spot_on", SpotOn);
    const int nSpot = CountDarker(SpotOff, SpotOn);
    std::printf("spot shadowed texels: %d\n", nSpot);
    CHECK(nSpot > 150, "Only %d texels in the spot light's shadow", nSpot);

    // The budget: 3 new shadowed point lights, 2 maps per frame
    Scene.m_LocalLights = { Point, Point, Point };
    Scene.m_LocalLights[1].m_Position = { -0.6f, 1.4f, 0.2f };
    Scene.m_LocalLights[2].m_Position = { 0.1f, 1.8f, -0.5f };
    Renderer.m_ShadowUpdatesPerFrame = 2;
    int nPerFrame[3];
    for (int& n : nPerFrame) { CHECK(App.RenderFrame(Scene, View, Camera), "Render budgeted"); n = Renderer.getShadowTilesRendered(); }
    CHECK(nPerFrame[0] == 2 && nPerFrame[1] == 1 && nPerFrame[2] == 0, "Budgeted maps per frame: %d %d %d (expected 2 1 0)", nPerFrame[0], nPerFrame[1], nPerFrame[2]);
    std::vector<float> Budgeted, Full;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Budgeted), "Render budgeted (settled)");
    Renderer.m_bShadowCache = false;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Full), "Render budgeted (rebuilt)");
    Renderer.m_bShadowCache = true;
    Renderer.m_ShadowUpdatesPerFrame = 16;
    for (std::size_t i = 0; i < Full.size(); ++i) CHECK(Budgeted[i] == Full[i], "Budgeted and rebuilt differ at %d", int(i));

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
