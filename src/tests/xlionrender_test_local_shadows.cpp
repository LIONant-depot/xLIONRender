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
#include "features/shadows/xlionrender_shadow_local_pages.h"

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
    // The virtual pages (features/shadows/xlionrender_shadow_local_pages.h, CPU only): addressing, a page's projection, the pinned pages,
    // requests within the budget, eviction under pressure, a light that moves, a changed caster dirtying its face's pages only
    {
        using pages = xlionrender::shadows::local_pages;
        CHECK(pages::face_entries_v == 341 && pages::LevelBase(pages::levels_v - 1) == 340, "a face's entries");
        for (int i = 0; i < 2 * pages::face_entries_v; ++i) CHECK(pages::Entry(pages::KeyOf(i)) == i, "entry %d round trip", i);

        // A point's page: the face's clip square of that page maps to [-1, 1], the neighbours' do not
        const xlionrender::local_light P{ .m_Position = { 1, 2, 3 }, .m_Radius = 10 };
        for (int k = 0; k < 64; ++k)
        {
            const xmath::fvec3 W{ 1.0f + 4.0f * std::sin(0.7f * k), 2.0f + 3.0f * std::cos(1.3f * k), 3.0f + 5.0f * std::sin(0.31f * k + 1.0f) };
            for (int f = 0; f < 6; ++f)
            {
                const auto F = pages::FaceProjection(P, f);
                const auto C = F * xmath::fvec4(W, 1);
                if (C.m_W <= 0) continue;
                const float u = C.m_X / C.m_W, v = C.m_Y / C.m_W;
                if (std::abs(u) >= 1 || std::abs(v) >= 1) continue;
                for (int L = 0; L < pages::levels_v; ++L)
                {
                    const int n = pages::level0_pages_v >> L;
                    const int x = std::min(int((u * 0.5f + 0.5f) * n), n - 1), y = std::min(int((v * 0.5f + 0.5f) * n), n - 1);
                    const auto Q = pages::PageW2C(P, { f, L, x, y }) * xmath::fvec4(W, 1);
                    CHECK(std::abs(Q.m_X / Q.m_W) <= 1.0001f && std::abs(Q.m_Y / Q.m_W) <= 1.0001f && std::abs(Q.m_Z / Q.m_W - C.m_Z / C.m_W) < 1e-5f, "point %d face %d level %d: outside its page", k, f, L);
                }
            }
        }

        // A wide spot (1.3 rad): its shadow frustum holds a point 1.2 rad off its axis (capped at 1 rad before: unshadowed past 57 degrees)
        {
            const xlionrender::local_light Wide{ .m_Type = xlionrender::local_light::type::SPOT, .m_Position = { 0, 3, 0 }, .m_Direction = { 0, -1, 0 }, .m_Radius = 10, .m_OuterAngle = 1.3f };
            const auto C = pages::FaceProjection(Wide, 0) * xmath::fvec4(xmath::fvec3(0, 3, 0) + xmath::fvec3(std::sin(1.2f), -std::cos(1.2f), 0) * 2.0f, 1);
            CHECK(C.m_W > 0 && std::abs(C.m_X / C.m_W) < 1 && std::abs(C.m_Y / C.m_W) < 1, "a 1.3 rad spot's frustum misses a point 1.2 rad off its axis (%f, %f)", C.m_X / C.m_W, C.m_Y / C.m_W);
        }
        const auto Table = [](const pages& S) { std::vector<std::uint32_t> T; S.BuildTable(T); return T; };
        pages S;
        S.Init({ .m_PoolTexels = 512, .m_Budget = 2 });        // 16 slots
        std::vector<xlionrender::local_light> L{ P, { .m_Type = xlionrender::local_light::type::SPOT, .m_Position = { -5, 2, 0 }, .m_Direction = { 0, -1, 0 }, .m_Radius = 6 } };
        S.Update(L, {}, true, {}, 1);
        CHECK(S.m_Stats.m_RenderedPinned == 7 && S.m_Stats.m_Rendered == 0, "a point's 6 coarsest pages and a spot's 1, pinned (%d)", S.m_Stats.m_RenderedPinned);
        CHECK(S.Validate(Table(S)), "invariant 1, pinned");
        S.Update(L, {}, true, {}, 2);
        CHECK(S.m_ToRender.empty(), "an unchanged frame renders nothing (%d)", int(S.m_ToRender.size()));

        // 5 requests, a budget of 2: coarse first, 2 a frame
        const int f0 = S.SlotOfLight(0) * pages::faces_per_light_v;
        std::vector<pages::key> Req{ { f0, 0, 3, 4 }, { f0, 2, 1, 1 }, { f0, 1, 2, 2 }, { f0, 0, 5, 5 }, { f0 + 1, 3, 0, 1 } };
        S.Update(L, {}, true, Req, 3);
        CHECK(S.m_ToRender.size() == 2 && S.m_ToRender[0].m_Key.m_Level == 3 && S.m_ToRender[1].m_Key.m_Level == 2, "the budget, coarse first");
        S.Update(L, {}, true, Req, 4);
        S.Update(L, {}, true, Req, 5);
        CHECK(S.m_ToRender.size() == 1 && S.m_Stats.m_Resident == 5 && S.Validate(Table(S)), "the requests all resident (%d)", S.m_Stats.m_Resident);

        // Pressure: 16 slots, 7 pinned, 5 resident; 8 new requests: 4 fit, then the least recently requested go, never a pinned one
        std::vector<pages::key> More;
        for (int k = 0; k < 8; ++k) More.push_back({ f0 + 2, 0, k, 0 });
        S.Init({ .m_PoolTexels = 512, .m_Budget = 100 });
        S.Update(L, {}, true, Req, 10);
        S.Update(L, {}, true, More, 11);
        CHECK(S.m_Stats.m_ResidentPinned == 7 && S.m_Stats.m_Resident == 9 && S.m_Stats.m_Rejected == 0, "the older requests evicted for the new (pinned %d, resident %d, rejected %d)", S.m_Stats.m_ResidentPinned, S.m_Stats.m_Resident, S.m_Stats.m_Rejected);
        More.push_back({ f0 + 3, 0, 0, 0 });
        S.Update(L, {}, true, More, 12);
        CHECK(S.m_Stats.m_Rejected == 0 && S.m_Stats.m_ResidentPinned == 7, "one more evicts the last old one (rejected %d)", S.m_Stats.m_Rejected);
        More.push_back({ f0 + 3, 0, 1, 0 });
        S.Update(L, {}, true, More, 13);
        CHECK(S.m_Stats.m_Rejected == 1 && S.m_Stats.m_ResidentPinned == 7 && S.Validate(Table(S)), "a full pool rejects, keeps the pinned (rejected %d)", S.m_Stats.m_Rejected);

        // The point changes (its radius): its pages go, a new generation; requests of the old one are dropped
        const std::vector<std::uint32_t> OldGen{ S.Generation(0), S.Generation(1) };
        L[0].m_Radius += 0.5f;
        S.Update(L, {}, true, {}, 14);
        CHECK(S.m_Stats.m_RenderedPinned == 6 && S.Validate(Table(S)), "a changed point renders its 6 pinned pages again (%d)", S.m_Stats.m_RenderedPinned);
        std::vector<std::uint32_t> Bits(std::size_t(pages::max_faces_v * pages::face_entries_v / 32 + 1), 0u);
        for (const auto& K : Req) Bits[std::size_t(pages::Entry(K)) >> 5] |= 1u << (pages::Entry(K) & 31);
        std::vector<pages::key> Decoded;
        pages::DecodeRequests(Bits, OldGen, S, Decoded);
        int nPoint = 0;
        for (const auto& K : Decoded) nPoint += K.m_Face / pages::faces_per_light_v == f0 / pages::faces_per_light_v;
        CHECK(nPoint == 0, "requests of the changed light's old generation dropped (%d kept)", nPoint);

        // The point moves (a torch): the same slot and generation, its requests kept; its requested pages render again every frame it
        // moves, even past the budget, its pinned ones too; a page it no longer asks for goes (none keeps a render from where it was)
        S.Init({ .m_PoolTexels = 1024, .m_Budget = 0 });
        S.Update(L, {}, true, {}, 30);
        const int fm = S.SlotOfLight(0) * pages::faces_per_light_v;
        const std::uint32_t Gen = S.Generation(S.SlotOfLight(0));
        const std::vector<pages::key> Asked{ { fm + 0, 1, 3, 3 }, { fm + 0, 1, 4, 3 }, { fm + 2, 0, 9, 9 } };
        S.setBudget(100);
        S.Update(L, {}, true, Asked, 31);
        S.setBudget(0);
        const int Resident = S.m_Stats.m_Resident;
        L[0].m_Position.m_X += 0.25f;
        const std::vector<pages::key> Fewer{ Asked[0], Asked[1] };
        S.Update(L, {}, true, Fewer, 32);
        CHECK(S.Generation(S.SlotOfLight(0)) == Gen && S.SlotOfLight(0) * pages::faces_per_light_v == fm, "a moved point lost its slot or generation");
        CHECK(Resident == 3 && S.m_Stats.m_Rendered == 2 && S.m_Stats.m_RenderedPinned == 6 && S.m_Stats.m_Resident == 2 && S.Validate(Table(S))
             , "a moving point: %d requested rendered past a budget of 0 (2), %d pinned (6), %d resident of %d (2: the one not asked for goes)"
             , S.m_Stats.m_Rendered, S.m_Stats.m_RenderedPinned, S.m_Stats.m_Resident, Resident);
        S.Update(L, {}, true, Fewer, 33);
        CHECK(S.m_ToRender.empty(), "a point that stopped renders again (%d)", int(S.m_ToRender.size()));
        L[0].m_Position.m_X -= 0.25f;
        L[1].m_Position.m_X += 0.25f;               // (the spot moves too: each keeps its own slot)
        S.Update(L, {}, true, Fewer, 34);
        CHECK(S.Generation(S.SlotOfLight(0)) == Gen && S.m_Stats.m_RenderedPinned == 7, "two lights moving: pinned pages rendered %d (7)", S.m_Stats.m_RenderedPinned);

        // Handles (lighting_strategy.md sec. 6.1): a light moves while another is put before it in the list - with a handle it is still
        // the same light (its slot, generation and pages follow it); without one, its index changed: a new light
        for (int bHandles = 1; bHandles >= 0; --bHandles)
        {
            auto H = L;
            H[0].m_Handle = bHandles ? 11 : 0; H[1].m_Handle = bHandles ? 12 : 0;
            S.Init({ .m_PoolTexels = 1024, .m_Budget = 100 });
            S.Update(H, {}, true, {}, 40);
            const int Slot = S.SlotOfLight(0);
            const std::uint32_t G0 = S.Generation(Slot);
            auto Moved = H;
            Moved[0].m_Position.m_X += 0.3f;
            Moved.insert(Moved.begin(), xlionrender::local_light{ .m_Position = { 9, 2, 9 }, .m_Radius = 3, .m_Handle = bHandles ? 13u : 0u });
            S.Update(Moved, {}, true, {}, 41);
            const bool bSame = S.SlotOfLight(1) == Slot && S.Generation(Slot) == G0;
            CHECK(bSame == (bHandles != 0), "a moving light put after a new one: %s (handles %d)", bSame ? "the same light" : "a new light", bHandles);
            CHECK(S.Validate(Table(S)), "handles: invariant 1");
        }
        S.Init({ .m_PoolTexels = 512, .m_Budget = 100 });

        // A changed caster below the point (its -Y face) dirties that face's pages, not the +Y face's
        S.Init({ .m_PoolTexels = 1024, .m_Budget = 100 });
        S.Update(L, {}, true, {}, 20);
        const int fp = S.SlotOfLight(0) * pages::faces_per_light_v;
        std::vector<pages::key> Down{ { fp + 3, 2, 1, 1 }, { fp + 2, 2, 1, 1 } };     // -Y, +Y
        S.Update(L, {}, true, Down, 21);
        const xmath::fvec3 Below = L[0].m_Position - xmath::fvec3(0, 2, 0);
        const std::vector<xmath::fbbox> Changed{ xmath::fbbox{ Below - xmath::fvec3(0.2f), Below + xmath::fvec3(0.2f) } };
        S.Update(L, Changed, true, Down, 22);
        int nDown = 0, nUp = 0;
        for (const auto& R : S.m_ToRender) { nDown += R.m_Key.m_Face == fp + 3; nUp += R.m_Key.m_Face == fp + 2; }
        CHECK(nDown >= 1 && nUp == 0, "a caster below dirties the -Y face's pages (%d), not the +Y face's (%d)", nDown, nUp);
        std::printf("local pages (CPU): addressing, projections, pinned, budget, pressure, generations, dirty faces: ok\n");
    }

    // No starvation (CPU only): a light whose page a caster dirties every frame, a budget of 1 - another light's requested page still
    // renders within urgent_age_v frames
    {
        using pages = xlionrender::shadows::local_pages;
        pages S;
        S.Init({ .m_Budget = 1 });
        const std::vector<xlionrender::local_light> L
        { { .m_Position = { 0, 0, 2 },   .m_Radius = 5 }
        , { .m_Position = { 0, 0, 40 },  .m_Radius = 5 } };
        S.Update(L, {}, true, {}, 1);
        const int fa = S.SlotOfLight(0) * pages::faces_per_light_v, fb = S.SlotOfLight(1) * pages::faces_per_light_v;
        const std::vector<pages::key> Req{ { fa + 4, 3, 0, 0 }, { fb + 4, 0, 7, 7 } };     // (the near one's coarse page first: coarse first)
        S.Update(L, {}, true, Req, 2);
        const std::vector<xmath::fbbox> Near{ xmath::fbbox{ xmath::fvec3{ -0.3f, -0.3f, 3.5f }, xmath::fvec3{ 0.3f, 0.3f, 4.0f } } };     // in the near light's +Z face only
        int SmallAt = -1;
        for (int f = 3; f <= 3 + 2 * xlionrender::shadows::urgent_age_v && SmallAt < 0; ++f)
        {
            S.Update(L, Near, true, Req, std::uint64_t(f));
            for (const auto& R : S.m_ToRender) if (!R.m_bPinned && R.m_Light == 1) SmallAt = f - 2;
        }
        CHECK(SmallAt > 0 && SmallAt <= xlionrender::shadows::urgent_age_v + 1, "the far light's page waited %d frames", SmallAt);
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

    // The pages (P1d): the first frame renders each face's pinned coarsest page (a point light: 6); the requests arrive a ring later and the
    // finer pages follow; settled, a frame renders none. Every frame's table holds invariant 1
    const auto Settle = [&](int n) { for (int i = 0; i < n; ++i) if (!App.RenderFrame(Scene, View, Camera) || !Renderer.validateLocalPages()) return false; return true; };
    const auto PagesOf = [&](int Light) { int k = 0; for (const auto& P : Renderer.getLocalPagesRendered()) k += P.m_Light == Light; return k; };
    const auto Pinned  = [&]() { int k = 0; for (const auto& P : Renderer.getLocalPagesRendered()) k += P.m_bPinned; return k; };

    // Its shadow
    Scene.m_LocalLights[0].m_bCastShadows = Scene.m_LocalLights[1].m_bCastShadows = true;
    CHECK(App.RenderFrame(Scene, View, Camera) && Renderer.validateLocalPages(), "Render with shadows");
    CHECK(Pinned() == 12 && Renderer.getShadowTilesRendered() == 12, "The first frame renders the pinned pages, 6 a point light (%d of %d)", Pinned(), Renderer.getShadowTilesRendered());
    CHECK(Settle(8) && ReadHDR(Device, View, On), "Settle with shadows");
    {
        const auto& R = Renderer.getLocalReport();
        const auto& C = R.m_Counters;
        std::printf("settled: pixel-light pairs %u - by their level %u, coarser %u, none %u; levels %u %u %u %u %u; pages resident %d + pinned %d\n"
                   , C[0], C[1], C[2], C[3], C[5], C[6], C[7], C[8], C[9], R.m_Pages.m_Resident, R.m_Pages.m_ResidentPinned);
        CHECK(R.m_bCounters && C[0] > 1000 && C[2] == 0 && C[3] == 0, "not settled: %u pairs, %u by a coarser level, %u by none", C[0], C[2], C[3]);
        CHECK(R.m_Pages.m_Resident > 0, "no requested page resident");
    }
    Dump("point_off", Off); Dump("point_on", On);
    const int nDarker = CountDarker(Off, On);
    std::printf("point shadowed texels: %d\n", nDarker);
    CHECK(nDarker > 150, "Only %d texels in the point light's shadow", nDarker);

    // Caching
    std::vector<float> Again;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Again), "Render again");
    CHECK(Renderer.getShadowTilesRendered() == 0, "An unchanged scene re-rendered %d pages", Renderer.getShadowTilesRendered());
    for (std::size_t i = 0; i < On.size(); ++i) CHECK(Again[i] == On[i], "The cached frame differs at %d", int(i));

    // The point light moves every frame (a torch): it keeps its pages' requests, so its pixels stay shaded by their own level - not by the
    // pinned coarse pages a light that starts again would fall back to
    {
        const auto Home = Scene.m_LocalLights[0].m_Position;
        for (int f = 0; f < 12; ++f)
        {
            Scene.m_LocalLights[0].m_Position = Home + xmath::fvec3(0.02f * float(f), 0.0f, 0.01f * float(f));
            CHECK(App.RenderFrame(Scene, View, Camera) && Renderer.validateLocalPages(), "Render a moving light %d", f);
        }
        const auto& C = Renderer.getLocalReport().m_Counters;
        std::printf("moving light: pixel-light pairs %u - by their level %u, coarser %u, none %u; pages rendered %d\n", C[0], C[1], C[2], C[3], Renderer.getShadowTilesRendered());
        CHECK(C[0] > 1000 && C[1] > C[0] * 9 / 10 && C[3] == 0, "a moving light's pixels: %u of %u by their level, %u by none", C[1], C[0], C[3]);
        Scene.m_LocalLights[0].m_Position = Home;
        CHECK(Settle(8), "Settle back");
    }

    // The sphere moves: the near light's pages that see it render again, the far light's do not; settled, cached equals rebuilt
    Scene.m_Draws[1].m_L2W = Lifted(0.4f);
    std::vector<float> Cached, Rebuilt;
    CHECK(App.RenderFrame(Scene, View, Camera), "Render moved");
    CHECK(PagesOf(0) > 0 && PagesOf(1) == 0, "A moved caster re-rendered %d pages of the near light, %d of the far one", PagesOf(0), PagesOf(1));
    CHECK(Settle(8) && ReadHDR(Device, View, Cached), "Render moved (cached)");
    Renderer.m_bShadowCache = false;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Rebuilt), "Render moved (rebuilt)");
    const int nRebuilt = Renderer.getShadowTilesRendered();
    Renderer.m_bShadowCache = true;
    CHECK(nRebuilt >= 12, "Without the cache every resident page renders (%d)", nRebuilt);
    for (std::size_t i = 0; i < Cached.size(); ++i) CHECK(Cached[i] == Rebuilt[i], "Cached and rebuilt shadows differ at %d (%f vs %f)", int(i), Cached[i], Rebuilt[i]);
    int nChanged = 0;
    for (std::size_t i = 0; i < Cached.size(); ++i) nChanged += Cached[i] != On[i];
    CHECK(nChanged > 0, "Moving the sphere changed nothing on screen");

    // A caster changing material re-renders the pages it reaches (an alpha-tested material casts another shadow)
    {
        const xlionrender::material_id Mat{ .m_Index = 7, .m_Generation = 1 };     // the key only (the float mesh ignores slots)
        CHECK(Settle(4) && Renderer.getShadowTilesRendered() == 0, "Settled before the material change");
        Scene.m_Draws[1].m_Materials = { Mat };
        CHECK(App.RenderFrame(Scene, View, Camera), "Render with the new material");
        CHECK(PagesOf(0) > 0 && PagesOf(1) == 0, "A material change re-rendered %d pages of the near light, %d of the far one", PagesOf(0), PagesOf(1));
        Scene.m_Draws[1].m_Materials.clear();
        CHECK(Settle(4), "Render back");
    }

    // Per-page caster culling draws the same pages as drawing every caster in the light's sphere into every page
    {
        std::vector<float> Faces, Sphere;
        Renderer.m_bShadowCache = false;
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Faces), "Render page-culled");
        Renderer.m_bShadowFaceCull = false;
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Sphere), "Render sphere-culled");
        Renderer.m_bShadowFaceCull = true;
        Renderer.m_bShadowCache    = true;
        for (std::size_t i = 0; i < Faces.size(); ++i) CHECK(Faces[i] == Sphere[i], "Page culling changed the shadows at %d (%f vs %f)", int(i), Faces[i], Sphere[i]);
    }

    // A spot light pointing down at the sphere: 1 pinned page, a shadow
    xlionrender::local_light Spot{ .m_Type = xlionrender::local_light::type::SPOT, .m_Position = { 0.3f, 2.0f, 0.2f }, .m_Direction = { -0.1f, -1.0f, -0.1f }
                                 , .m_Radius = 6.0f, .m_InnerAngle = 0.6f, .m_OuterAngle = 0.8f, .m_Intensity = 8.0f, .m_bCastShadows = false };
    Scene.m_LocalLights = { Spot };
    std::vector<float> SpotOff, SpotOn;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, SpotOff), "Render spot without shadows");
    Scene.m_LocalLights[0].m_bCastShadows = true;
    CHECK(App.RenderFrame(Scene, View, Camera), "Render spot with shadows");
    CHECK(Pinned() == 1 && Renderer.getShadowTilesRendered() == 1, "A spot light renders 1 pinned page first (%d)", Renderer.getShadowTilesRendered());
    CHECK(Settle(8) && ReadHDR(Device, View, SpotOn), "Settle the spot");
    Dump("spot_on", SpotOn);
    const int nSpot = CountDarker(SpotOff, SpotOn);
    std::printf("spot shadowed texels: %d\n", nSpot);
    CHECK(nSpot > 150, "Only %d texels in the spot light's shadow", nSpot);

    // The budget: 3 new shadowed point lights, 2 unpinned pages a frame (the pinned ones always); settled, the same image as a full rebuild
    Scene.m_LocalLights = { Point, Point, Point };
    Scene.m_LocalLights[1].m_Position = { -0.6f, 1.4f, 0.2f };
    Scene.m_LocalLights[2].m_Position = { 0.1f, 1.8f, -0.5f };
    Renderer.m_ShadowUpdatesPerFrame = 2;
    int nMax = 0, nFrames = 0;
    for (; nFrames < 200; ++nFrames)
    {
        CHECK(App.RenderFrame(Scene, View, Camera) && Renderer.validateLocalPages(), "Render budgeted");
        const int n = Renderer.getShadowTilesRendered() - Pinned();
        nMax = std::max(nMax, n);
        if (nFrames > 4 && Renderer.getShadowTilesRendered() == 0) break;
    }
    std::printf("budgeted: settled after %d frames, at most %d unpinned pages a frame\n", nFrames, nMax);
    CHECK(nMax == 2 && nFrames < 200, "Budgeted: at most %d unpinned pages a frame (2), settled after %d frames", nMax, nFrames);
    std::vector<float> Budgeted, Full;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Budgeted), "Render budgeted (settled)");
    Renderer.m_bShadowCache = false;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Full), "Render budgeted (rebuilt)");
    Renderer.m_bShadowCache = true;
    Renderer.m_ShadowUpdatesPerFrame = 64;
    for (std::size_t i = 0; i < Full.size(); ++i) CHECK(Budgeted[i] == Full[i], "Budgeted and rebuilt differ at %d", int(i));

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
