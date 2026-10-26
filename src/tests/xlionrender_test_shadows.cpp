// The directional light's shadow - its virtual pages (features/shadows/xlionrender_shadow_vsm.h; the clipmap this test was written for
// went 2026-10-27). A sphere over a ground slab, a directional light:
//  - the shadow is there: with shadows on no texel gets brighter and a patch of the ground gets clearly darker
//  - caching: once the requested pages are in (a ring late), an unchanged scene renders no page on its next frame
//  - differential: after the sphere moves, the cached image equals the image with every page re-rendered (m_bShadowCache = false),
//    and only some of the pages were re-rendered
//  - the camera moving re-renders only the newly needed pages, not all of them
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
}

int main()
{
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
    Scene.m_Sun   = { .m_Direction = { 0.3f, -1.0f, 0.2f }, .m_Intensity = 3, .m_ShadowTileSize = 0.5f };
    Scene.m_Ambient = { 0.1f, 0.1f, 0.1f };

    // Frames until no page renders (the requests are read back a ring late, the budget spreads them): the number rendered meanwhile
    const auto Settle = [&]() -> int
    {
        int n = 0;
        for (int k = 0; k < 30; ++k)
        {
            if (!App.RenderFrame(Scene, View, Camera)) return -1;
            if (k >= 4 && Renderer.getShadowTilesRendered() == 0) return n;
            n += Renderer.getShadowTilesRendered();
        }
        return -1;
    };

    // Shadows off vs on
    std::vector<float> Off, On;
    Scene.m_Sun.m_bCastShadows = false;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Off), "Render without shadows");
    Scene.m_Sun.m_bCastShadows = true;
    CHECK(App.RenderFrame(Scene, View, Camera), "Render with shadows");
    const int nNet = Renderer.getVSM().getSetup().PinnedSlots();
    CHECK(Renderer.getShadowTilesRendered() >= nNet, "The first shadowed frame renders the coarse net (%d of %d; new casters are movers for a few frames: their pages too)", Renderer.getShadowTilesRendered(), nNet);
    const int nFirst = Settle();
    const int total_tiles_v = Renderer.getVSMReport().m_Pages.m_ResidentFine + Renderer.getVSMReport().m_Pages.m_ResidentPinned;
    std::printf("pages: the net %d, then %d as the requests came; %d resident\n", nNet, nFirst, total_tiles_v);
    CHECK(nFirst > 0, "No page rendered for the requests");
    CHECK(ReadHDR(Device, View, On), "Read the settled frame");

    int nDarker = 0;
    for (std::size_t i = 0; i < On.size(); ++i)
    {
        CHECK(std::isfinite(On[i]), "Non-finite texel %d", int(i));
        CHECK(On[i] <= Off[i] * 1.001f + 1e-4f, "Texel %d got brighter with shadows (%f -> %f)", int(i), Off[i], On[i]);
        if (On[i] < Off[i] * 0.6f) ++nDarker;
    }
    std::printf("shadowed texels: %d\n", nDarker);
    CHECK(nDarker > 150, "Only %d texels in shadow", nDarker);

    // Caching: nothing changed, nothing re-rendered
    std::vector<float> Again;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Again), "Render again");
    CHECK(Renderer.getShadowTilesRendered() == 0, "An unchanged scene re-rendered %d tiles", Renderer.getShadowTilesRendered());
    for (std::size_t i = 0; i < On.size(); ++i) CHECK(Again[i] == On[i], "The cached frame differs at %d", int(i));

    // The sphere moves: some tiles re-render; the cached result equals a full rebuild
    Scene.m_Draws[1].m_L2W = Lifted(0.4f);
    std::vector<float> Cached, Rebuilt;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Cached), "Render moved (cached)");
    const int nMoved = Renderer.getShadowTilesRendered();
    CHECK(Settle() >= 0, "Settle after the move");
    CHECK(ReadHDR(Device, View, Cached), "Read the moved frame, settled");
    std::printf("tiles re-rendered for the moved sphere: %d of %d\n", nMoved, total_tiles_v);
    CHECK(nMoved > 0 && nMoved < total_tiles_v, "A moved caster re-rendered %d of %d tiles", nMoved, total_tiles_v);

    Renderer.m_bShadowCache = false;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Rebuilt), "Render moved (rebuilt)");
    CHECK(Renderer.getShadowTilesRendered() > nMoved, "Without the cache every page renders (%d, the move %d)", Renderer.getShadowTilesRendered(), nMoved);
    Renderer.m_bShadowCache = true;
    for (std::size_t i = 0; i < Cached.size(); ++i) CHECK(Cached[i] == Rebuilt[i], "Cached and rebuilt shadows differ at %d (%f vs %f)", int(i), Cached[i], Rebuilt[i]);
    int nChanged = 0;
    for (std::size_t i = 0; i < Cached.size(); ++i) nChanged += Cached[i] != On[i];
    CHECK(nChanged > 0, "Moving the sphere changed nothing on screen");

    // The camera moves sideways: only the pages newly asked for render
    CHECK(Settle() >= 0, "Settle");
    Camera.LookAt(xmath::fvec3{ 0.6f, 2.5f, -3.0f }, xmath::fvec3{ 0.6f, 0, 0 });
    const int nCamera = Settle();
    std::printf("tiles re-rendered for the moved camera: %d of %d\n", nCamera, total_tiles_v);
    CHECK(nCamera > 0 && nCamera < total_tiles_v, "A camera move re-rendered %d of %d tiles", nCamera, total_tiles_v);

    // The same with instance ids (casters tracked by their persistent slot, not hashed): unchanged -> nothing; moved, gone,
    // re-materialed -> tiles, and the cached image equals a rebuild each time
    {
        const auto CachedEqualsRebuilt = [&](const char* pWhat) -> bool
        {
            std::vector<float> A, B;
            if (!App.RenderFrame(Scene, View, Camera)) return false;
            const int nTiles = Renderer.getShadowTilesRendered();
            if (Settle() < 0 || !ReadHDR(Device, View, A)) return false;
            Renderer.m_bShadowCache = false;
            const bool bOk = App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, B);
            Renderer.m_bShadowCache = true;
            if (!bOk || Settle() < 0) return false;
            std::printf("ids: %s re-rendered %d tiles\n", pWhat, nTiles);
            if (nTiles == 0) { std::printf("FAIL: %s rendered no tile\n", pWhat); return false; }
            for (std::size_t i = 0; i < A.size(); ++i) if (A[i] != B[i]) { std::printf("FAIL: %s: cached and rebuilt differ at %d\n", pWhat, int(i)); return false; }
            return true;
        };
        Scene.m_Draws[0].m_Instance = 1;
        Scene.m_Draws[1].m_Instance = 2;
        CHECK(CachedEqualsRebuilt("new ids"), "Ids: first frame");
        CHECK(Settle() >= 0 && App.RenderFrame(Scene, View, Camera) && Renderer.getShadowTilesRendered() == 0, "Ids: an unchanged scene re-rendered %d tiles", Renderer.getShadowTilesRendered());
        Scene.m_Draws[1].m_L2W = Lifted(-0.3f);
        CHECK(CachedEqualsRebuilt("moved"), "Ids: moved");
        const auto Sphere1 = Scene.m_Draws[1];
        Scene.m_Draws.pop_back();
        CHECK(CachedEqualsRebuilt("gone"), "Ids: gone");
        Scene.m_Draws.push_back(Sphere1);
        CHECK(CachedEqualsRebuilt("back"), "Ids: back");
        Scene.m_Draws[1].m_Material.m_BaseColor = { 0.2f, 0.9f, 0.2f };
        CHECK(CachedEqualsRebuilt("material"), "Ids: material change");
    }

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
