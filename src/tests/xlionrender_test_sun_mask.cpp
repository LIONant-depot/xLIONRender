// The sun's shadow mask (XLION_SUN_MASK, xLION_Docs shadows_gpu_plan.md S1a; shaders/xlionrender_sun_mask_frag.glsl): the sun's visibility
// computed once per pixel after the G-buffer prepass and read by the materials, against the materials' own lookup. A sphere and a box
// over a ground slab under a low sun (long shadows, a curved receiver, a silhouette against the ground):
//  - mode 2 (the pass runs, the materials skip their lookup): the shadow is gone - the pass and the switch are on the main view's path
//  - mode 3 (the materials read the mask): the image matches the inline one - the mask's plane and footprint come from the depth's
//    neighbours, not the triangle's derivatives, and the visibility has 16 bits: small differences, at silhouettes
//  - no validation error (slot 11's format in every pipeline, the mask in the read layout before the first draw)
#include "xlionrender_test_common.h"
#include <cstdlib>

using namespace xlionrender::test;

namespace
{
    constexpr int size_v = 192;

    void SetEnv(const char* Name, const char* Value) noexcept
    {
#ifdef _WIN32
        _putenv_s(Name, Value);
#else
        if (*Value) setenv(Name, Value, 1); else unsetenv(Name);
#endif
    }

    // The scene settled with XLION_SUN_MASK = Mode (a renderer of its own: the mode is read at Init)
    bool Render(const char* Mode, std::vector<float>& Out)
    {
        SetEnv("XLION_SUN_MASK", Mode);
        app App;
        const bool bInit = App.Init();
        SetEnv("XLION_SUN_MASK", "");
        if (!bInit) return false;
        auto& Renderer = App.m_Renderer;
        // cluster geometry (xgeom_static's path: the G-buffer prepass lays its depth - plain meshes are not in it, they shade inline)
        cluster_buffers SphereB, BoxB;
        xlionrender::mesh_id Sphere, Box;
        if (!MakeStaticBuffers(App.m_Device, MakeSphere(), 0.5f, SphereB) || Renderer.CreateClusterGeom(Sphere, SphereB.Geom())) return false;
        if (!MakeStaticBuffers(App.m_Device, xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 }), 0.5f, BoxB) || Renderer.CreateClusterGeom(Box, BoxB.Geom())) return false;
        xlionrender::render_view View;
        if (Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v })) return false;

        xgpu::tools::view Camera;
        Camera.setFov(xmath::radian{ 1.0f });
        Camera.LookAt(xmath::fvec3{ 0.5f, 2.0f, -3.5f }, xmath::fvec3{ 0, 0, 0 });
        xlionrender::scene Scene;
        const auto Ground = xmath::fmat4::fromTranslation({ 0, -0.55f, 0 }) * xmath::fmat4::fromScale({ 10.0f, 0.1f, 10.0f });
        Scene.m_Draws   = { { .m_Mesh = Box, .m_L2W = Ground }, { .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ -0.4f, 0.3f, 0 }) }
                          , { .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ 0.7f, -0.1f, 0.5f }) * xmath::fmat4::fromScale({ 0.4f, 0.8f, 0.4f }) } };
        Scene.m_Sun     = { .m_Direction = { 0.8f, -0.45f, 0.3f }, .m_Intensity = 3, .m_ShadowTileSize = 0.5f };
        Scene.m_Ambient = { 0.1f, 0.1f, 0.1f };

        bool bOk = true;
        int k = 0;
        for (; bOk && k < 40; ++k)
        {
            bOk = App.RenderFrame(Scene, View, Camera);
            if (k >= 4 && Renderer.getShadowTilesRendered() == 0) break;
        }
        bOk = bOk && k < 40 && App.RenderFrame(Scene, View, Camera) && ReadHDR(App.m_Device, View, Out);
        Renderer.DestroyView(View);
        return bOk;
    }
}

int main()
{
    std::vector<float> Inline, Skip, Mask;
    CHECK(Render("0", Inline), "Render inline (settled)");
    CHECK(Render("2", Skip),   "Render with the mask pass, the materials' lookup off (settled)");
    CHECK(Render("3", Mask),   "Render with the materials reading the mask (settled)");

    // mode 2: the shadowed texels lit again
    int nShadowed = 0, nLitAgain = 0;
    for (std::size_t i = 0; i < Inline.size(); ++i)
        if (Inline[i] < Skip[i] * 0.6f) { ++nShadowed; }
    for (std::size_t i = 0; i < Inline.size(); ++i) if (Skip[i] > Inline[i] * 1.5f) ++nLitAgain;
    std::printf("mode 2: %d texels brighter than inline (the shadow gone)\n", nLitAgain);
    CHECK(nLitAgain > 300, "The mask pass did not run on the main view (%d texels changed)", nLitAgain);

    // mode 3: as inline
    double Sum = 0; float MaxDiff = 0; int nOff = 0;
    for (std::size_t i = 0; i < Inline.size(); ++i)
    {
        const float d = std::abs(Mask[i] - Inline[i]) / std::max(Inline[i], 1e-3f);
        Sum += d; MaxDiff = std::max(MaxDiff, d);
        if (d > 0.05f) ++nOff;
    }
    const double Mean = Sum / double(Inline.size());
    std::printf("mode 3 vs inline: mean relative difference %.5f, max %.3f, texels off by more than 5%%: %d of %d\n", Mean, MaxDiff, nOff, int(Inline.size()));
    CHECK(Sum > 0.0, "The materials never read the mask (identical to inline: its distance check failed everywhere)");
    CHECK(Mean < 0.005, "The mask shades differently on average (%.5f)", Mean);
    CHECK(nOff < int(Inline.size()) / 200, "%d texels differ by more than 5%% (at most 0.5%% of them: silhouettes)", nOff);

    // (whole-page bounds prove almost nothing here: a lit receiver is a caster too - its own surface is in the page, nearer the light
    // across the reach: shadows_gpu_plan.md S1b; the share is printed, its correctness checked)
    // modes 4, 5 (S1b, the early-out on the pages' depth bounds): what it claims lit (5 writes those half lit) is lit in the full
    // evaluation - mode 3's texel equals the unshadowed one (mode 2's: no material lookup) -, and with it the image is mode 3's exactly
    std::vector<float> Early, Claimed;
    CHECK(Render("4", Early),   "Render with the early-out (settled)");
    CHECK(Render("5", Claimed), "Render with the early-out's pixels marked (settled)");
    int nClaimed = 0, nWrong = 0;
    float MaxEarly = 0;
    for (std::size_t i = 0; i < Mask.size(); ++i)
    {
        MaxEarly = std::max(MaxEarly, std::abs(Early[i] - Mask[i]));
        if (std::abs(Claimed[i] - Mask[i]) <= 1e-6f) continue;
        ++nClaimed;
        if (std::abs(Mask[i] - Skip[i]) > 1e-4f * std::max(Skip[i], 1e-3f)) ++nWrong;
    }
    std::printf("modes 4/5: %d of %d texels proven lit (%.1f%%), %d of them not lit in the full evaluation; mode 4 vs 3 max difference %g\n"
               , nClaimed, int(Mask.size()), 100.0 * nClaimed / double(Mask.size()), nWrong, MaxEarly);
    CHECK(nWrong == 0, "%d texels proven lit are shadowed in the full evaluation", nWrong);
    CHECK(MaxEarly <= 1e-6f, "The early-out changed the image (max %g)", MaxEarly);

    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
