// Reflection probes (scene::m_Probes; shaders/xlionrender_environment.glsl):
//  - the captured map is right: red toward the red box (+X), green toward the green one (-X) - faces, orientation, remap
//  - the budget: 1 face per frame -> a capture completes on the 6th frame, not before
//  - invalidation: a static scene does not re-capture; a box moving inside the probe's box does; one far outside does not
//  - shading: a mirror sphere under the probe reflects the red box (more red on that side than without the probe)
//  - light changes: a local light reaching the probe's box re-captures it, one far away does not; a sun change does
//  - the world cells: nested and overlapping probes shade the same image found through the cells (CELLS) as with the full list (EXACT)
//  - residency: 40 probes, only the nearest 32 hold a slot (and are captured); the camera moving to the others swaps them in
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

    // The probe atlas' level 0 texel of a direction (xlionrender_octahedral.glsl / xlionrender_environment_atlas.glsl)
    std::array<float, 3> ProbeTexel(xgpu::device& Device, const xgpu::texture& Atlas, int Slot, xmath::fvec3 D)
    {
        std::vector<std::byte> Raw;
        int W, H;
        if (Device.ReadTexture(Atlas, Raw, W, H)) return { -1, -1, -1 };
        D = D.NormalizeCopy();
        float E[3] = { D.m_X, D.m_Z, -D.m_Y };
        const float L1 = std::abs(E[0]) + std::abs(E[1]) + std::abs(E[2]);
        for (float& e : E) e /= L1;
        float PX = E[0], PY = E[1];
        if (E[2] < 0) { PX = (1 - std::abs(E[1])) * (E[0] >= 0 ? 1.f : -1.f); PY = (1 - std::abs(E[0])) * (E[1] >= 0 ? 1.f : -1.f); }
        const int X = Slot % 4 * 384 + int(2 + (PX * 0.5f + 0.5f) * 252), Y = Slot / 4 * 256 + int(2 + (PY * 0.5f + 0.5f) * 252);
        const auto* p = reinterpret_cast<const std::uint16_t*>(Raw.data()) + (std::size_t(Y) * W + X) * 4;
        return { HalfToFloat(p[0]), HalfToFloat(p[1]), HalfToFloat(p[2]) };
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
    Camera.setFov(xmath::radian{ 0.7f });
    Camera.LookAt(xmath::fvec3{ 0, 1.2f, -4.0f }, xmath::fvec3{ 0, 1, 0 });

    xlionrender::scene Scene;
    const auto Red   = xlionrender::material{ .m_BaseColor = { 0.9f, 0.05f, 0.05f }, .m_Emissive = { 2, 0, 0 } };
    const auto Green = xlionrender::material{ .m_BaseColor = { 0.05f, 0.9f, 0.05f }, .m_Emissive = { 0, 2, 0 } };
    Scene.m_Draws =
    { { .m_Mesh = Box,    .m_L2W = xmath::fmat4::fromTranslation({ 0, -0.5f, 0 }) * xmath::fmat4::fromScale({ 12, 1, 12 }), .m_Material = { .m_BaseColor = { 0.3f, 0.3f, 0.3f } } }
    , { .m_Mesh = Box,    .m_L2W = xmath::fmat4::fromTranslation({  3.5f, 1.5f, 0 }) * xmath::fmat4::fromScale({ 1, 3, 4 }), .m_Material = Red }
    , { .m_Mesh = Box,    .m_L2W = xmath::fmat4::fromTranslation({ -3.5f, 1.5f, 0 }) * xmath::fmat4::fromScale({ 1, 3, 4 }), .m_Material = Green }
    , { .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ 0, 1, 0 }), .m_Material = { .m_BaseColor = { 0.95f, 0.95f, 0.95f }, .m_Metallic = 1, .m_Roughness = 0.05f } }
    };
    Scene.m_Sun     = { .m_Direction = { 0.2f, -1, 0.3f }, .m_Intensity = 2, .m_bCastShadows = false };
    Scene.m_Ambient = { 0.1f, 0.1f, 0.1f };

    // Without a probe: the reference for the shading check
    std::vector<float> NoProbe;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, NoProbe), "Render without a probe");

    // A probe above the sphere, all 6 faces in one frame
    Scene.m_Probes = { { .m_Position = { 0, 2.5f, 0 }, .m_BoxMin = { -6, -1, -6 }, .m_BoxMax = { 6, 6, 6 }, .m_Blend = 0.5f } };
    Renderer.m_ProbeFacesPerFrame = 6;
    CHECK(App.RenderFrame(Scene, View, Camera), "Render the capture");
    CHECK(Renderer.getProbeCaptures() == 1, "One capture (%d)", Renderer.getProbeCaptures());
    const auto PX = ProbeTexel(Device, Renderer.getProbeAtlas(), 0, { 1, -0.2f, 0 });
    const auto NX = ProbeTexel(Device, Renderer.getProbeAtlas(), 0, { -1, -0.2f, 0 });
    std::printf("probe: +X (%.2f %.2f %.2f)  -X (%.2f %.2f %.2f)\n", PX[0], PX[1], PX[2], NX[0], NX[1], NX[2]);
    CHECK(PX[0] > 2 * PX[1] && NX[1] > 2 * NX[0], "The probe does not see red toward +X and green toward -X");

    // Shading: the mirror sphere reflects the red box with the probe
    std::vector<float> WithProbe;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, WithProbe), "Render with the probe");
    double RedGain = 0;
    for (std::size_t i = 0; i < WithProbe.size(); ++i) RedGain += std::max(0.0f, WithProbe[i] - NoProbe[i]);
    std::printf("red gained with the probe: %.2f\n", RedGain);
    CHECK(RedGain > 20, "The mirror sphere does not reflect the probe (%f)", RedGain);
    Dump("probe_without", NoProbe);
    Dump("probe_with", WithProbe);

    // Static: no re-capture
    for (int f = 0; f < 5; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render static");
    CHECK(Renderer.getProbeCaptures() == 1, "A static scene re-captured (%d)", Renderer.getProbeCaptures());

    // A box moving far outside the probe's box: no re-capture
    Scene.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ 50, 1, 0 }) });
    CHECK(App.RenderFrame(Scene, View, Camera), "Render far box");
    Scene.m_Draws.back().m_L2W = xmath::fmat4::fromTranslation({ 51, 1, 0 });
    for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render far box moved");
    CHECK(Renderer.getProbeCaptures() == 1, "A change outside the probe's box re-captured (%d)", Renderer.getProbeCaptures());

    // The red box moves inside the box, 1 face per frame: the capture completes on the 6th frame
    Renderer.m_ProbeFacesPerFrame = 1;
    Scene.m_Draws[1].m_L2W = xmath::fmat4::fromTranslation({ 3.0f, 1.5f, 1.0f }) * xmath::fmat4::fromScale({ 1, 3, 4 });
    int nFrames = 0;
    while (Renderer.getProbeCaptures() == 1 && nFrames < 20) { CHECK(App.RenderFrame(Scene, View, Camera), "Render re-capture"); ++nFrames; }
    std::printf("re-capture at 1 face per frame took %d frames\n", nFrames);
    CHECK(nFrames == 6, "A capture at 1 face per frame took %d frames", nFrames);

    // Light changes (all faces in one frame from here)
    Renderer.m_ProbeFacesPerFrame = 6;
    for (int f = 0; f < 2; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render settle");
    int Before = Renderer.getProbeCaptures();
    Scene.m_LocalLights = { { .m_Position = { 40, 1, 0 }, .m_Radius = 3, .m_Color = { 1, 1, 1 }, .m_Intensity = 4, .m_bCastShadows = false } };
    for (int f = 0; f < 2; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render far light");
    Scene.m_LocalLights[0].m_Intensity = 8;
    for (int f = 0; f < 2; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render far light changed");
    CHECK(Renderer.getProbeCaptures() == Before, "A light that does not reach the probe's box re-captured it (%d)", Renderer.getProbeCaptures() - Before);
    Scene.m_LocalLights[0].m_Position = { 2, 1, 0 };            // moves into the box
    CHECK(App.RenderFrame(Scene, View, Camera), "Render near light");
    CHECK(Renderer.getProbeCaptures() == Before + 1, "A light reaching the probe's box did not re-capture it (%d)", Renderer.getProbeCaptures() - Before);
    Scene.m_LocalLights[0].m_Color = { 1, 0.5f, 0.2f };
    CHECK(App.RenderFrame(Scene, View, Camera), "Render near light color");
    CHECK(Renderer.getProbeCaptures() == Before + 2, "A light color change did not re-capture (%d)", Renderer.getProbeCaptures() - Before);
    Scene.m_Sun.m_Intensity = 3;
    CHECK(App.RenderFrame(Scene, View, Camera), "Render sun change");
    CHECK(Renderer.getProbeCaptures() == Before + 3, "A sun change did not re-capture (%d)", Renderer.getProbeCaptures() - Before);
    Scene.m_LocalLights.clear();

    // The world cells: nested and overlapping probes, CELLS against EXACT
    Scene.m_Probes.push_back({ .m_Position = { -2, 1.5f, 0 }, .m_BoxMin = { -5, -1, -3 }, .m_BoxMax = { 0.5f, 4, 3 }, .m_Blend = 1.0f });
    Scene.m_Probes.push_back({ .m_Position = {  2, 1.5f, 0 }, .m_BoxMin = { -0.5f, -1, -3 }, .m_BoxMax = { 5, 4, 3 }, .m_Blend = 1.0f });
    for (int f = 0; f < 4; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render 3 probes");
    std::vector<float> Cells, Exact;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Cells), "Render cells");
    Renderer.m_LightMode = xlionrender::instance::light_mode::EXACT;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Exact), "Render exact");
    Renderer.m_LightMode = xlionrender::instance::light_mode::CELLS;
    double MaxDiff = 0;
    for (std::size_t i = 0; i < Cells.size(); ++i) MaxDiff = std::max<double>(MaxDiff, std::abs(Cells[i] - Exact[i]));
    std::printf("probes CELLS vs EXACT: max difference %g\n", MaxDiff);
    CHECK(MaxDiff < 1e-3, "Probes through the cells differ from the full list (%g)", MaxDiff);

    // Residency: 40 small probes in a row along +Z, the camera near the first ones
    Scene.m_Probes.clear();
    for (int i = 0; i < 40; ++i)
    {
        const float Z = float(i) * 4.0f;
        Scene.m_Probes.push_back({ .m_Position = { 0, 1.5f, Z }, .m_BoxMin = { -2, -1, Z - 2 }, .m_BoxMax = { 2, 4, Z + 2 }, .m_Blend = 0.5f });
    }
    Before = Renderer.getProbeCaptures();
    for (int f = 0; f < 50; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render 40 probes");
    CHECK(Renderer.getProbeCaptures() - Before == 32, "40 probes: %d captures, the 32 resident expected", Renderer.getProbeCaptures() - Before);
    xgpu::tools::view Far = Camera;
    Far.LookAt(xmath::fvec3{ 0, 1.2f, 39 * 4.0f + 4 }, xmath::fvec3{ 0, 1, 39 * 4.0f });
    Before = Renderer.getProbeCaptures();
    for (int f = 0; f < 50; ++f) CHECK(App.RenderFrame(Scene, View, Far), "Render 40 probes far");
    std::printf("residency: the camera at the far end captured %d probes\n", Renderer.getProbeCaptures() - Before);
    CHECK(Renderer.getProbeCaptures() - Before == 8, "The far end swapped in %d probes, 8 expected", Renderer.getProbeCaptures() - Before);

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
