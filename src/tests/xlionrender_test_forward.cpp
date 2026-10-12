// First xLIONRender slice test: a sphere through plan A (forward, standard PBR, directional light) into HDR, then
// tone mapped. Checks, without golden images:
//  - metamorphic: doubling the light intensity doubles every HDR texel (ambient and emission are zero)
//  - convention : a light from above lights the top of the sphere (xgpu::tools::view's Y-up NDC end to end)
//  - background : texels the sphere does not cover stay exactly the clear color
//  - mesh ids   : a destroyed mesh's id draws nothing, even after its slot is reused
//  - GPU timings arrive; no Vulkan validation messages
#include "xlionrender_test_common.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "dependencies/stb/stb_image_write.h"

using namespace xlionrender::test;

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    xlionrender::mesh_id Sphere;
    {
        const auto Prim = MakeSphere();
        std::vector<xlionrender::vertex> Verts(Prim.m_Vertices.size());
        for (std::size_t i = 0; i < Verts.size(); ++i)
        {
            const auto& V = Prim.m_Vertices[i];
            Verts[i] = { { V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }, { V.m_Texcoord.m_X, V.m_Texcoord.m_Y }, 0xffffffffu
                       , { V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }, { V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, V.m_Tangent.m_D } };
        }
        CHECK(!Renderer.CreateMesh(Sphere, Verts, Prim.m_Indices), "CreateMesh");
    }

    constexpr int size_v = 128;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");

    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.0f });
    Camera.LookAt(xmath::fvec3{ 0, 0, -2.0f }, xmath::fvec3{ 0, 0, 0 });

    xlionrender::scene Scene;
    Scene.m_Draws.push_back({ .m_Mesh = Sphere, .m_Material = { .m_BaseColor = { 0.8f, 0.8f, 0.8f }, .m_Roughness = 0.6f } });
    Scene.m_Light.m_Direction = { 0, -1, 0.3f };          // travels downwards: lights the top
    Scene.m_Ambient         = { 0, 0, 0 };

    // A: intensity 2, B: intensity 4
    std::vector<float> A, B;
    Scene.m_Light.m_Intensity = 2;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, A), "Render A");
    Scene.m_Light.m_Intensity = 4;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, B), "Render B");

    int nLit = 0;
    for (std::size_t i = 0; i < A.size(); ++i)
    {
        CHECK(std::isfinite(A[i]) && std::isfinite(B[i]), "Non-finite texel %d", int(i));
        if (A[i] == 0) { CHECK(B[i] == 0, "Texel %d: 0 at intensity 2 but %f at 4", int(i), B[i]); continue; }
        if (A[i] < 1e-3f) continue;                       // too close to fp16 resolution to compare ratios
        ++nLit;
        CHECK(std::abs(B[i] / A[i] - 2.0f) < 0.01f, "Texel %d: %f -> %f is not x2", int(i), A[i], B[i]);
    }
    CHECK(nLit > size_v * size_v / 20, "Only %d lit texels", nLit);

    // Corners are background; top of the sphere lit, bottom dark
    CHECK(A[0] == 0 && A[size_v * size_v - 1] == 0, "Corners must be background");
    const float Top    = A[(size_v / 2 - size_v / 6) * size_v + size_v / 2];
    const float Bottom = A[(size_v / 2 + size_v / 6) * size_v + size_v / 2];
    CHECK(Top > 4 * Bottom + 1e-4f, "Light from above: top %f should be much brighter than bottom %f (Y convention)", Top, Bottom);

    // Tone-mapped output exists and follows the HDR
    {
        std::vector<std::uint32_t> LDR;
        int W, H;
        CHECK(!Device.ReadTexture(View.m_Output, LDR, W, H) && W == size_v, "Read output");
        stbi_write_png("xlionrender_test_forward.png", W, H, 4, LDR.data(), W * 4);     // for a human look
        const auto Red = [&](int Row) { return LDR[Row * size_v + size_v / 2] & 0xff; };
        CHECK(Red(size_v / 2 - size_v / 6) > Red(size_v / 2 + size_v / 6), "Output top %u should be brighter than bottom %u", Red(size_v / 2 - size_v / 6), Red(size_v / 2 + size_v / 6));
    }

    // Stale ids draw nothing, even after the slot is reused
    {
        const auto Old = Sphere;
        Renderer.DestroyMesh(Sphere);
        const xlionrender::vertex Tri[] = { { {0,0,0} }, { {1,0,0} }, { {0,1,0} } };
        const std::uint32_t       Idx[] = { 0, 1, 2 };
        xlionrender::mesh_id Reused;
        CHECK(!Renderer.CreateMesh(Reused, Tri, Idx), "CreateMesh reuse");
        CHECK(Reused.m_Index == Old.m_Index && Reused.m_Generation != Old.m_Generation, "Slot should be reused with a new generation");

        Scene.m_Draws[0].m_Mesh = Old;
        std::vector<float> C;
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, C), "Render stale");
        for (float v : C) CHECK(v == 0, "A stale mesh id drew something");
    }

    for (int i = 0; i < 4 && !App.m_bGotTimes; ++i) CHECK(App.RenderFrame(Scene, View, Camera), "Render for timings");
    CHECK(App.m_bGotTimes, "No GPU timings");

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
