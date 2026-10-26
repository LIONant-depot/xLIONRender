// The lighting references of documentation/lighting_strategy.md P0 (the sampling reference, EXACT against SAMPLED on the same
// visibility, is the light lab's: xlionrender_test_light_lab.cpp):
//  - LightEval (shaders/xlionrender_light_eval.glsl): analytic values of the one falloff and cone the shaders and the cells use
//  - radiometry is not visibility: a colour or intensity change keeps a light's shadow map and scales its light exactly; a move
//    renders the map again
//  - the visibility reference: per pixel, the shadow map's visibility of one light (the image with the light's shadows over the image
//    without) against the GI's rays (gi::world::Occluded, the same geometry). Opaque rigid geometry only: every capture says so
#include "xlionrender_test_common.h"
#include "features/gi/xlionrender_surfel_gi.h"
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

    // The same triangles for the GI's rays (mesh space; the world places them)
    xlionrender::gi::surfel_mesh RayMesh(const xprim_geom::mesh& Prim)
    {
        xlionrender::gi::surfel_mesh M;
        for (const auto& V : Prim.m_Vertices) { M.m_Pos.push_back({ V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }); M.m_Nrm.push_back({ V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }); }
        for (std::size_t i = 0; i + 2 < Prim.m_Indices.size(); i += 3) M.m_Tris.push_back({ int(Prim.m_Indices[i]), int(Prim.m_Indices[i + 1]), int(Prim.m_Indices[i + 2]) });
        M.Bake(0.08f, 0.25f);
        return M;
    }
}

int main()
{
    namespace le = xlionrender::light_eval;

    // LightEval: analytic values
    {
        CHECK(le::LightDistanceFalloff(0.0f, 1.0f) == 1.0f, "falloff at the light: %f", le::LightDistanceFalloff(0.0f, 1.0f));
        CHECK(le::LightDistanceFalloff(1.0f, 1.0f) == 0.0f, "falloff at the radius (no contribution there): %f", le::LightDistanceFalloff(1.0f, 1.0f));
        CHECK(le::LightDistanceFalloff(0.5f, 1.0f) == 0.625f, "falloff at half the radius: %f, not (1 - 1/16) / (1 + 1/2)", le::LightDistanceFalloff(0.5f, 1.0f));
        CHECK(le::LightDistanceFalloff(0.5f, 3.0f) == 1.875f, "a scaled falloff: %f", le::LightDistanceFalloff(0.5f, 3.0f));
        float Prev = 2;
        for (int i = 0; i <= 1024; ++i)
        {
            const float s = float(i) / 1024, f = le::LightDistanceFalloff(s, 1.0f);
            CHECK(f < Prev && f >= 0, "the falloff does not decrease at s = %f (%f after %f)", s, f, Prev);
            Prev = f;
            const float k = 0.37f * float(i + 1), Scaled = le::LightDistanceFalloff(s, k), Expected = k * f;
            CHECK(std::abs(Scaled - Expected) <= 1e-6f * Expected, "the scale is not a plain factor at s = %f: %g vs %g", s, Scaled, Expected);
        }
        const float CosOuter = std::cos(0.7f), CosInner = std::cos(0.4f);
        CHECK(le::LightSpotCone(1.0f, CosOuter, CosInner) == 1.0f && le::LightSpotCone(CosInner, CosOuter, CosInner) == 1.0f, "the cone inside its inner angle is not 1");
        CHECK(le::LightSpotCone(CosOuter, CosOuter, CosInner) == 0.0f && le::LightSpotCone(std::cos(1.2f), CosOuter, CosInner) == 0.0f, "the cone at or past its outer angle is not 0");
        CHECK(std::abs(le::LightSpotCone(0.5f * (CosOuter + CosInner), CosOuter, CosInner) - 0.5f) < 1e-5f, "the cone halfway between its angles is not 0.5");
        std::printf("LightEval: analytic values ok (falloff ends, half radius, monotonic, scale; cone inside, outside, halfway)\n");
    }

    // Radiometry is not visibility: the local shadow cache keeps a map through a colour and intensity change, renders it after a move
    {
        xlionrender::shadows::local_shadows S;
        S.Init({});
        std::vector<xlionrender::local_light> L{ { .m_Position = { 1, 2, 3 }, .m_Radius = 6 } };
        std::vector<xlionrender::shadows::local_map> Maps;
        std::vector<xlionrender::shadows::local_map_to_render> Out;
        const xmath::fvec3 Eye{ 0, 2, -8 };
        S.Update(L, {}, true, Eye, 500, 8, Maps, Out);
        CHECK(Out.size() == 1, "a new light's map is not rendered (%zu)", Out.size());
        S.Update(L, {}, true, Eye, 500, 8, Maps, Out);
        CHECK(Out.empty(), "an unchanged light's map is rendered again");
        L[0].m_Color = { 1, 0.2f, 0.1f }; L[0].m_Intensity = 9;
        S.Update(L, {}, true, Eye, 500, 8, Maps, Out);
        CHECK(Out.empty() && Maps[0].m_Size > 0, "a colour and intensity change rendered the map again (or dropped it)");
        L[0].m_Position.m_X += 0.5f;
        S.Update(L, {}, true, Eye, 500, 8, Maps, Out);
        CHECK(Out.size() == 1, "a moved light kept its old map");
    }

    app App;
    CHECK(App.Init(), "Init");
    App.m_bQuietTimes = true;
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    const auto SpherePrim = MakeSphere();
    const auto BoxPrim    = xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 });
    xlionrender::mesh_id Sphere, Box;
    CHECK(MakeMesh(Renderer, SpherePrim, Sphere), "sphere");
    CHECK(MakeMesh(Renderer, BoxPrim, Box), "box");

    constexpr int size_v = 160;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");
    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.1f });
    Camera.setViewport({ 0, 0, size_v, size_v });
    Camera.LookAt(xmath::fvec3{ 0, 6.0f, -9.0f }, xmath::fvec3{ 0, 0, 1.0f });

    // The light lab's scene: a ground and a grid of spheres, no sun, no ambient (a pixel is its local lights' light only)
    xlionrender::scene Scene;
    Scene.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ 0, -0.55f, 1 }) * xmath::fmat4::fromScale({ 20.0f, 0.1f, 20.0f }) });
    for (int z = 0; z < 4; ++z) for (int x = 0; x < 5; ++x)
        Scene.m_Draws.push_back({ .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ -4.0f + 2.0f * x, 0.0f, -2.0f + 2.5f * z }), .m_Material = { .m_Metallic = (z % 2) ? 0.8f : 0.0f, .m_Roughness = (x % 2) ? 0.25f : 0.7f } });
    Scene.m_Sun     = { .m_Intensity = 0, .m_bCastShadows = false };
    Scene.m_Ambient = { 0, 0, 0 };
    Renderer.m_LightMode = xlionrender::instance::light_mode::EXACT;

    const auto Render = [&](std::vector<float>& Out) -> bool
    {
        for (int i = 0; i < 3; ++i) if (!App.RenderFrame(Scene, View, Camera)) return false;
        return App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Out);
    };

    // Radiometry on the GPU: twice the intensity is twice the light, and no shadow map is rendered for it
    {
        Scene.m_LocalLights = { { .m_Position = { 0.5f, 2.5f, 1.0f }, .m_Radius = 8 }, { .m_Type = xlionrender::local_light::type::SPOT, .m_Position = { 2, 4, -1 }
                              , .m_Direction = xmath::fvec3{ -0.3f, -1, 0.4f }.NormalizeCopy(), .m_Radius = 9, .m_InnerAngle = 0.4f, .m_OuterAngle = 0.7f } };
        std::vector<float> One, Two;
        CHECK(Render(One), "render x1");
        for (auto& l : Scene.m_LocalLights) l.m_Intensity *= 2;
        CHECK(App.RenderFrame(Scene, View, Camera), "render the change");
        CHECK(Renderer.getShadowTilesRendered() == 0, "an intensity change rendered %d shadow maps", Renderer.getShadowTilesRendered());
        CHECK(Render(Two), "render x2");
        double MaxRel = 0; int n = 0;
        for (std::size_t i = 0; i < One.size(); ++i) if (One[i] > 1e-3f) { MaxRel = std::max(MaxRel, std::abs(double(Two[i]) - 2.0 * One[i]) / (2.0 * One[i])); ++n; }
        std::printf("radiometry: x2 intensity, %d lit pixels, max relative error %.2e (half floats: ~1e-3)\n", n, MaxRel);
        CHECK(n > 1000 && MaxRel < 2e-3, "twice the intensity is not twice the light (max relative error %g over %d pixels)", MaxRel, n);
    }

    // The counters (DescribeScene's Lights*): no update budget (forced exhaustion) - a new light's map is never rendered, the light
    // shades unshadowed and is counted (the declared fallback); a moved caster's maps stay sampled while they wait (counted stale: today's
    // behaviour, which P1a's resource states end)
    {
        Scene.m_LocalLights = { { .m_Position = { 0.5f, 2.5f, 1.0f }, .m_Radius = 8 } };
        CHECK(App.RenderFrame(Scene, View, Camera) && App.RenderFrame(Scene, View, Camera), "render with a budget");
        auto L = Renderer.getLightStats();
        CHECK(L.m_nLights == 1 && L.m_nShadowed == 1 && L.m_nShadowFallback == 0 && L.m_nShadowStale == 0 && L.m_nDropped == 0, "counters with a budget: lights %d shadowed %d fallback %d stale %d dropped %d", L.m_nLights, L.m_nShadowed, L.m_nShadowFallback, L.m_nShadowStale, L.m_nDropped);
        const int Budget = Renderer.m_ShadowUpdatesPerFrame;
        Renderer.m_ShadowUpdatesPerFrame = 0;
        Scene.m_Draws[1].m_L2W = xmath::fmat4::fromTranslation({ -4.0f, 0.3f, -2.0f });      // a sphere moves: the map waits
        CHECK(App.RenderFrame(Scene, View, Camera), "render a moved caster");
        L = Renderer.getLightStats();
        CHECK(L.m_nShadowStale == 1 && L.m_nShadowFallback == 0, "a deferred map is not counted stale (stale %d, fallback %d)", L.m_nShadowStale, L.m_nShadowFallback);
        Scene.m_LocalLights.push_back({ .m_Position = { -3.0f, 1.2f, 3.0f }, .m_Radius = 6 });
        CHECK(App.RenderFrame(Scene, View, Camera), "render a new light");
        L = Renderer.getLightStats();
        CHECK(L.m_nShadowFallback == 1, "a shadowed light without a map is not counted as a fallback (%d)", L.m_nShadowFallback);
        std::printf("counters: forced budget exhaustion - stale maps %d, unshadowed fallbacks %d\n", L.m_nShadowStale, L.m_nShadowFallback);
        Renderer.m_ShadowUpdatesPerFrame = Budget;
        Scene.m_Draws[1].m_L2W = xmath::fmat4::fromTranslation({ -4.0f, 0.0f, -2.0f });
    }

    // The visibility reference: the GI's rays on the same triangles
    const auto RaySphere = RayMesh(SpherePrim), RayBox = RayMesh(BoxPrim);      // (before the world: it points at them)
    xlionrender::gi::world World;
    for (std::size_t d = 0; d < Scene.m_Draws.size(); ++d) World.Add(d == 0 ? RayBox : RaySphere, Scene.m_Draws[d].m_L2W, { 0.5f, 0.5f, 0.5f });     // (draw 0: the ground)

    std::printf("reference visibility: opaque rigid geometry: supported; skinned geometry: unsupported; per-texel masked geometry: unsupported\n");
    const std::vector<xlionrender::local_light> Lights =
    {
        { .m_Position = { 0.5f, 2.5f, 1.0f }, .m_Radius = 8 },
        { .m_Position = { -3.0f, 1.2f, 3.0f }, .m_Radius = 6 },
        { .m_Position = { 3.5f, 0.8f, -1.5f }, .m_Radius = 7 },
        { .m_Type = xlionrender::local_light::type::SPOT, .m_Position = { 2, 4, -1 }, .m_Direction = xmath::fvec3{ -0.3f, -1, 0.4f }.NormalizeCopy(), .m_Radius = 9, .m_InnerAngle = 0.4f, .m_OuterAngle = 0.7f },
        { .m_Type = xlionrender::local_light::type::SPOT, .m_Position = { -2, 3, 6 }, .m_Direction = xmath::fvec3{ 0.2f, -0.6f, -1 }.NormalizeCopy(), .m_Radius = 12, .m_InnerAngle = 0.3f, .m_OuterAngle = 0.6f },
    };
    const xmath::fvec3 Eye = Camera.getPosition();
    // A pixel's camera ray: two points of its clip-space line back to the world (the image's row 0 is its top). The NDC axes' signs
    // are not assumed: SX, SY are picked below by matching what the rays see against the first light's lit pixels
    const xmath::fmat4 C2W = Camera.getC2W();
    const xmath::fvec3 Forward = Camera.getForward();
    const auto Ray = [&](int x, int y, int SX, int SY)
    {
        const float nx = SX * (2.0f * (float(x) + 0.5f) / size_v - 1.0f), ny = SY * (1.0f - 2.0f * (float(y) + 0.5f) / size_v);
        const auto At = [&](float z) { const xmath::fvec4 P = C2W * xmath::fvec4{ nx, ny, z, 1.0f }; return xmath::fvec3{ P.m_X, P.m_Y, P.m_Z } * (1.0f / P.m_W); };
        auto D = (At(0.6f) - At(0.2f)).NormalizeCopy();
        return D.Dot(Forward) < 0 ? D * -1.0f : D;
    };
    int nAllLit = 0, nAllDisagree = 0, SX = 0, SY = 0;
    for (std::size_t l = 0; l < Lights.size(); ++l)
    {
        std::vector<float> Shadowed, Unshadowed;
        Scene.m_LocalLights = { Lights[l] };
        CHECK(Render(Shadowed), "render light %zu shadowed", l);
        Scene.m_LocalLights[0].m_bCastShadows = false;
        CHECK(Render(Unshadowed), "render light %zu unshadowed", l);

        if (l == 0)     // the mapping: the axes' signs whose rays light (in radius, facing the light) the pixels the GPU lit
        {
            double Best = -1, Second = -1;
            for (int sx : { 1, -1 }) for (int sy : { 1, -1 })
            {
                int nAgree = 0;
                for (int y = 0; y < size_v; ++y) for (int x = 0; x < size_v; ++x)
                {
                    const auto H = World.Cast(Eye, Ray(x, y, sx, sy));
                    bool bLit = false;
                    if (H.m_Inst >= 0)
                    {
                        const auto P = World.HitPosition(H), D = Lights[0].m_Position - P;
                        bLit = D.Length() < Lights[0].m_Radius && World.HitNormal(H).Dot(D) > 0;
                    }
                    nAgree += bLit == (Unshadowed[std::size_t(y) * size_v + x] > 1e-4f);
                }
                const double A = double(nAgree) / (size_v * size_v);
                if (A > Best) { Second = Best; Best = A; SX = sx; SY = sy; } else Second = std::max(Second, A);
            }
            std::printf("pixel to ray mapping: NDC signs %d %d, lit pixels agree %.2f%% (the next best %.2f%%)\n", SX, SY, 100 * Best, 100 * Second);
            CHECK(Best > 0.99 && Second < Best - 0.02, "no unambiguous pixel to ray mapping (best %.3f, next %.3f)", Best, Second);
        }
        std::vector<float> Disagree(Unshadowed.size(), 0.0f);      // (XLIONRENDER_TEST_DUMP: where they disagree; 1: the rays are dark)
        int nLit = 0, nMiss = 0, nDisagree = 0, nMapDark = 0, nRayDark = 0, nSelf = 0, nGrazing = 0; double SumDiff = 0;
        for (int y = 0; y < size_v; ++y)
        for (int x = 0; x < size_v; ++x)
        {
            const std::size_t i = std::size_t(y) * size_v + x;
            if (Unshadowed[i] <= 1e-4f) continue;                                   // the light does not reach the pixel
            ++nLit;
            const float VMap = std::clamp(Shadowed[i] / Unshadowed[i], 0.0f, 1.0f);
            const auto  H    = World.Cast(Eye, Ray(x, y, SX, SY));
            if (H.m_Inst < 0) { ++nMiss; continue; }
            const auto  P    = World.HitPosition(H), N = World.HitNormal(H).NormalizeCopy();
            const auto  O    = P + N * 0.01f;
            const auto  D    = Lights[l].m_Position - O;
            const float d    = D.Length();
            const float VRay = World.Occluded(O, D * (1.0f / d), d - 2e-3f) ? 0.0f : 1.0f;
            SumDiff += std::abs(VMap - VRay);
            if (std::abs(VMap - VRay) > 0.5f)
            {
                ++nDisagree; (VMap < VRay ? nMapDark : nRayDark)++;
                if (VMap > VRay) { const auto B = World.Cast(O, D * (1.0f / d), d - 2e-3f); nSelf += B.m_Inst == H.m_Inst; nGrazing += N.Dot(D) < 0.2f * d; }
                Disagree[i] = VMap > VRay ? 1.0f : 0.3f;
            }
        }
        std::printf("visibility reference, light %zu (%s): %d lit pixels, %d missed by the rays (mapping), disagree %.2f%% (map dark %d, rays dark %d: by the receiver itself %d, grazing %d), mean |map - rays| %.4f\n"
                   , l, Lights[l].m_Type == xlionrender::local_light::type::SPOT ? "spot" : "point", nLit, nMiss
                   , 100.0 * nDisagree / std::max(nLit, 1), nMapDark, nRayDark, nSelf, nGrazing, SumDiff / std::max(nLit - nMiss, 1));
        Dump(("visref_" + std::to_string(l) + "_disagree").c_str(), Disagree); Dump(("visref_" + std::to_string(l) + "_map").c_str(), Shadowed);
        CHECK(nLit > 500, "light %zu lights only %d pixels", l, nLit);
        CHECK(nMiss * 100 < nLit, "light %zu: %d of %d lit pixels have no ray hit (the pixel to ray mapping is off)", l, nMiss, nLit);
        nAllLit += nLit; nAllDisagree += nDisagree;
    }
    std::printf("visibility reference: the shadow maps disagree with the rays on %.2f%% of %d lit pixels (opaque rigid)\n", 100.0 * nAllDisagree / std::max(nAllLit, 1), nAllLit);
    // The visibility-model error of today's maps (2026-10-26: 6.3%, at the shadows' edges: the maps' texels, PCF and normal offset
    // against exact rays; the rays dark where the maps are lit, mostly). A guard: a change that makes the maps disagree more fails
    CHECK(nAllDisagree * 100 < nAllLit * 8, "the shadow maps disagree with the rays on more than 8%% of the lit pixels");

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
