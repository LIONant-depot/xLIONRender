// The environment probe (scene::m_Sky; shaders/xlionrender_environment*.glsl):
//  - a uniform sky: every texel of every prefiltered level (borders included) is the sky's color - the prefilter is normalized
//    and the octahedral borders fold right
//  - a gradient sky: each level keeps level 0's mean (the prefilter moves light around, it does not add or remove it)
//  - furnace: a white diffuse sphere under a uniform sky of 1 (no sun) shades to ~1
//  - a static sky builds the atlas once; changing it rebuilds
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

    // The atlas' red channel (RGBA16F), and the levels' rects (xlionrender_environment_atlas.glsl)
    bool ReadAtlas(xgpu::device& Device, const xgpu::texture& Atlas, std::vector<float>& Red, int& W)
    {
        std::vector<std::byte> Raw;
        int H;
        if (Device.ReadTexture(Atlas, Raw, W, H)) return false;
        const auto* p = reinterpret_cast<const std::uint16_t*>(Raw.data());
        Red.resize(std::size_t(W) * H);
        for (std::size_t i = 0; i < Red.size(); ++i) Red[i] = HalfToFloat(p[i * 4]);
        return true;
    }
    constexpr int Rects[6][3] = { { 0, 0, 256 }, { 256, 0, 128 }, { 256, 128, 64 }, { 320, 128, 32 }, { 352, 128, 16 }, { 368, 128, 8 } };
}

int main()
{
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
    Camera.setFov(xmath::radian{ 0.6f });
    Camera.LookAt(xmath::fvec3{ 0, 0.3f, -2.5f }, xmath::fvec3{ 0, 0, 0 });

    xlionrender::scene Scene;
    Scene.m_Draws = { { .m_Mesh = Sphere, .m_Material = { .m_BaseColor = { 1, 1, 1 }, .m_Metallic = 0, .m_Roughness = 1.0f, .m_F0 = 0.04f } } };
    Scene.m_Sun   = { .m_Intensity = 0, .m_bCastShadows = false };

    // A uniform sky: every level, every texel = the sky
    Scene.m_Sky = { .m_Zenith = { 1, 1, 1 }, .m_Horizon = { 1, 1, 1 }, .m_Ground = { 1, 1, 1 }, .m_Intensity = 1 };
    CHECK(App.RenderFrame(Scene, View, Camera), "Render uniform sky");
    {
        std::vector<float> Atlas;
        int W;
        CHECK(ReadAtlas(Device, Renderer.getEnvironmentAtlas(), Atlas, W), "Read the atlas");
        for (const auto& R : Rects)
            for (int y = R[1]; y < R[1] + R[2]; ++y)
                for (int x = R[0]; x < R[0] + R[2]; ++x)
                    CHECK(std::abs(Atlas[std::size_t(y) * W + x] - 1.0f) < 0.01f, "Uniform sky: level %d texel (%d,%d) = %f", int(&R - Rects), x, y, Atlas[std::size_t(y) * W + x]);
    }

    // Furnace: a white diffuse sphere under it shades to ~1 (diffuse 1 + a little specular)
    {
        std::vector<float> Img;
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Img), "Render the furnace");
        const float C = Img[std::size_t(size_v / 2) * size_v + size_v / 2];
        std::printf("furnace: the sphere's center %.4f\n", C);
        CHECK(C > 0.9f && C < 1.15f, "Furnace: a white sphere under a sky of 1 shades to %f", C);
    }

    // A gradient sky: each level keeps level 0's mean; a static sky does not rebuild
    Scene.m_Sky = { .m_Zenith = { 0.2f, 0.4f, 1.0f }, .m_Horizon = { 1.0f, 0.9f, 0.8f }, .m_Ground = { 0.1f, 0.1f, 0.1f }, .m_Intensity = 1 };
    CHECK(App.RenderFrame(Scene, View, Camera), "Render gradient sky");
    const int nBuilds = Renderer.getEnvironmentBuilds();
    {
        std::vector<float> Atlas;
        int W;
        CHECK(ReadAtlas(Device, Renderer.getEnvironmentAtlas(), Atlas, W), "Read the atlas");
        double Mean[6] = {};
        for (int L = 0; L < 6; ++L)
        {
            const auto& R = Rects[L];
            double Sum = 0; int n = 0;
            for (int y = R[1] + 2; y < R[1] + R[2] - 2; ++y) for (int x = R[0] + 2; x < R[0] + R[2] - 2; ++x) { Sum += Atlas[std::size_t(y) * W + x]; ++n; }
            Mean[L] = Sum / n;
        }
        std::printf("level means: %.3f %.3f %.3f %.3f %.3f %.3f\n", Mean[0], Mean[1], Mean[2], Mean[3], Mean[4], Mean[5]);
        for (int L = 1; L < 6; ++L) CHECK(std::abs(Mean[L] - Mean[0]) < 0.08 * Mean[0], "Level %d's mean %f is not level 0's %f", L, Mean[L], Mean[0]);
    }
    for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render static sky");
    CHECK(Renderer.getEnvironmentBuilds() == nBuilds, "A static sky rebuilt the atlas (%d -> %d)", nBuilds, Renderer.getEnvironmentBuilds());
    Scene.m_Sky.m_Zenith = { 0.3f, 0.5f, 1.0f };
    CHECK(App.RenderFrame(Scene, View, Camera) && Renderer.getEnvironmentBuilds() == nBuilds + 1, "A sky change did not rebuild the atlas");

    // A cubemap texture instead of the gradient (scene::sky::m_pCubemap), HDR or not: an 8 bit white cube is a uniform sky of 1 (every
    // texel of the atlas 1); a float cube bright overhead lights the sphere's top more than its bottom
    {
        const auto MakeCube = [&](xgpu::texture& T, xgpu::texture::format Fmt, std::size_t Texel, const auto& Fill)
        {
            constexpr int n_v = 16;
            std::vector<std::byte> Data(6 * n_v * n_v * Texel);
            for (int f = 0; f < 6; ++f) for (int i = 0; i < n_v * n_v; ++i) Fill(f, Data.data() + (std::size_t(f) * n_v * n_v + i) * Texel);
            const std::array Mips{ xgpu::texture::setup::mip_size{ std::uint32_t(n_v * n_v * Texel) } };
            return Device.Create(T, { .m_Format = Fmt, .m_AdressModes = { xgpu::texture::address_mode::CLAMP, xgpu::texture::address_mode::CLAMP, xgpu::texture::address_mode::CLAMP }
                                    , .m_Width = n_v, .m_Height = n_v, .m_MipSizes = Mips, .m_AllFacesData = Data, .m_FaceCount = 6, .m_isGamma = false, .m_isCubemap = true }) == nullptr;
        };
        const auto Gradient = Scene.m_Sky;
        xgpu::texture White, Hot;
        CHECK(MakeCube(White, xgpu::texture::format::R8G8B8A8_UNORM, 4, [](int, std::byte* p) { for (int k = 0; k < 4; ++k) p[k] = std::byte{ 255 }; }), "the white cube");
        CHECK(MakeCube(Hot, xgpu::texture::format::R32G32B32A32_SFLOAT, 16, [](int f, std::byte* p) { const float v[4] = { f == 2 ? 4.0f : 0.25f, f == 2 ? 4.0f : 0.25f, f == 2 ? 4.0f : 0.25f, 1 }; std::memcpy(p, v, 16); }), "the hot cube");

        Scene.m_Draws = { { .m_Mesh = Sphere, .m_Material = { .m_BaseColor = { 1, 1, 1 }, .m_Metallic = 0, .m_Roughness = 1.0f, .m_F0 = 0.04f } } };
        Scene.m_Sun   = { .m_Intensity = 0, .m_bCastShadows = false };
        Camera.LookAt(xmath::fvec3{ 0, 0.3f, -2.5f }, xmath::fvec3{ 0, 0, 0 });
        Scene.m_Sky   = { .m_Intensity = 1, .m_pCubemap = &White };
        const int nBefore = Renderer.getEnvironmentBuilds();
        CHECK(App.RenderFrame(Scene, View, Camera) && Renderer.getEnvironmentBuilds() == nBefore + 1, "a cubemap did not build the atlas");
        {
            std::vector<float> Atlas;
            int W;
            CHECK(ReadAtlas(Device, Renderer.getEnvironmentAtlas(), Atlas, W), "Read the atlas");
            float Lo = 1e9f, Hi = 0;
            for (const auto& R : Rects) for (int y = R[1]; y < R[1] + R[2]; ++y) for (int x = R[0]; x < R[0] + R[2]; ++x) { Lo = std::min(Lo, Atlas[std::size_t(y) * W + x]); Hi = std::max(Hi, Atlas[std::size_t(y) * W + x]); }
            std::printf("white cubemap: atlas %.4f .. %.4f\n", Lo, Hi);
            CHECK(Lo > 0.99f && Hi < 1.01f, "a white cubemap is not a uniform sky of 1 (%f .. %f)", Lo, Hi);
        }
        for (int f = 0; f < 5; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render the white cube");     // (its mean is read back after the frames in flight)
        std::printf("white cubemap: mean %.4f\n", Renderer.getEnvironmentMean().m_X);
        CHECK(std::abs(Renderer.getEnvironmentMean().m_X - 1) < 0.01f, "the white cubemap's mean is %f, not 1", Renderer.getEnvironmentMean().m_X);
        Scene.m_Sky.m_pCubemap = &Hot;
        std::vector<float> Img;
        for (int f = 0; f < 2; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render the hot cube");
        CHECK(ReadHDR(Device, View, Img), "Read the hot cube");
        const float Top = Img[std::size_t(size_v / 2 - 30) * size_v + size_v / 2], Bottom = Img[std::size_t(size_v / 2 + 30) * size_v + size_v / 2];
        std::printf("hot cubemap overhead: the sphere's top %.3f, bottom %.3f\n", Top, Bottom);
        // (2x held with uvsphere's old normals, tilted toward the poles; the sphere's true ones at these pixels: 1.27 vs 0.68)
        CHECK(Top > 1.5f * Bottom, "the cubemap's bright top does not light the sphere's top (%f vs %f)", Top, Bottom);
        Scene.m_Sky = Gradient;
        CHECK(App.RenderFrame(Scene, View, Camera), "Render without the cubemaps");
        Device.Destroy(std::move(White));
        Device.Destroy(std::move(Hot));
    }

    // A look (XLIONRENDER_TEST_DUMP): metal spheres from mirror to rough, then dielectrics, under the gradient sky and a sun
    {
        Scene.m_Draws.clear();
        for (int r = 0; r < 2; ++r) for (int i = 0; i < 5; ++i)
            Scene.m_Draws.push_back({ .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ -2.4f + 1.2f * i, r ? -0.6f : 0.6f, 0 })
                                    , .m_Material = { .m_BaseColor = r ? xmath::fvec3{ 0.8f, 0.2f, 0.2f } : xmath::fvec3{ 0.95f, 0.8f, 0.5f }, .m_Metallic = r ? 0.0f : 1.0f, .m_Roughness = 0.05f + 0.2f * i } });
        Scene.m_Sun = { .m_Direction = { -0.4f, -1.0f, 0.6f }, .m_Intensity = 3, .m_bCastShadows = false };
        Camera.LookAt(xmath::fvec3{ 0, 0, -7.0f }, xmath::fvec3{ 0, 0, 0 });
        std::vector<float> Img;
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Img), "Render the look");
        Dump("env_spheres", Img);
    }

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
