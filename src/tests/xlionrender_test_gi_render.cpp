// Surfel GI through the renderer (scene::m_GI, instance::SetMeshGI; the constant material's vertex read,
// shaders/xlionrender_static_geom_body.glsl): a room of cluster meshes - a red wall on the left, a green one on the right, a box,
// one point light - rendered without and with GI:
//  - color bleeding: with GI the floor near the red wall is redder than near the green wall (without GI they match);
//    xgpu::view's screen x runs to world -x here: the green wall (+x) is on the left of the image
//  - GI adds light (the bounces) and nothing breaks (no validation errors)
//  - a moved box re-lights (the world follows the draws) and a removed draw leaves the world
//  - the per-pixel read: matches the vertex read on dense meshes, and on sparse ones (2 triangles a face) stays near the dense image
#include "xlionrender_test_common.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../stb/stb_image_write.h"

using namespace xlionrender::test;

namespace
{
    std::vector<std::array<float, 3>> ReadRGB(xgpu::device& Device, const xlionrender::render_view& View, int& W, int& H)
    {
        std::vector<std::byte> Raw;
        std::vector<std::array<float, 3>> Out;
        if (Device.ReadTexture(View.m_HDR, Raw, W, H)) return Out;
        const auto* p = reinterpret_cast<const std::uint16_t*>(Raw.data());
        Out.resize(std::size_t(W) * H);
        for (std::size_t i = 0; i < Out.size(); ++i) Out[i] = { HalfToFloat(p[i * 4]), HalfToFloat(p[i * 4 + 1]), HalfToFloat(p[i * 4 + 2]) };
        return Out;
    }

    // XLIONRENDER_TEST_DUMP=<folder>: <folder>/<Name>.png (x / (1 + x), gamma 2.2)
    void DumpRGB(const char* pName, const std::vector<std::array<float, 3>>& Img, int W, int H)
    {
        const char* pDir = std::getenv("XLIONRENDER_TEST_DUMP");
        if (pDir == nullptr) return;
        std::vector<unsigned char> Bytes(Img.size() * 3);
        for (std::size_t i = 0; i < Img.size(); ++i)
            for (int k = 0; k < 3; ++k) { const float v = std::max(Img[i][k], 0.0f) * 2; Bytes[i * 3 + k] = (unsigned char)(std::pow(v / (1 + v), 1 / 2.2f) * 255 + 0.5f); }
        stbi_write_png((std::string(pDir) + "/" + pName + ".png").c_str(), W, H, 3, Bytes.data(), W * 3);
    }

    // The mean color of a pixel rectangle (fractions of the image)
    std::array<double, 3> Mean(const std::vector<std::array<float, 3>>& Img, int W, int H, float x0, float y0, float x1, float y1)
    {
        std::array<double, 3> S{}; int n = 0;
        for (int y = int(y0 * H); y < int(y1 * H); ++y) for (int x = int(x0 * W); x < int(x1 * W); ++x) { for (int k = 0; k < 3; ++k) S[k] += Img[std::size_t(y) * W + x][k]; ++n; }
        for (auto& v : S) v /= std::max(n, 1);
        return S;
    }
}

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    // The room's pieces, each its own mesh at its real size (4 m walls, 0.25 m vertex spacing), surfels 0.3 m, parents 0.9 m
    struct piece { xprim_geom::float3 m_Size; xmath::fvec3 m_Pos; xmath::fvec3 m_Color; };
    const piece Pieces[] =
    { { { 4.0f, 0.2f, 4.0f }, { 0, -0.1f, 0 }, { 0.75f, 0.75f, 0.75f } }       // floor
    , { { 4.0f, 0.2f, 4.0f }, { 0, 4.1f, 0 },  { 0.75f, 0.75f, 0.75f } }       // ceiling
    , { { 4.0f, 4.0f, 0.2f }, { 0, 2, 2.1f },  { 0.75f, 0.75f, 0.75f } }       // back
    , { { 0.2f, 4.0f, 4.0f }, { -2.1f, 2, 0 }, { 0.8f, 0.05f, 0.05f } }        // left: red
    , { { 0.2f, 4.0f, 4.0f }, { 2.1f, 2, 0 },  { 0.05f, 0.7f, 0.05f } }        // right: green
    , { { 0.8f, 1.6f, 0.8f }, { -0.6f, 0.8f, 0.6f }, { 0.75f, 0.75f, 0.75f } } // a box
    };
    constexpr int n_v = int(std::size(Pieces));
    std::array<cluster_buffers, n_v>      Buffers;
    std::array<xlionrender::mesh_id, n_v> Meshes;
    for (int i = 0; i < n_v; ++i)
    {
        const auto Prim = xprim_geom::cube::Generate(15, 15, 15, 15, Pieces[i].m_Size);
        CHECK(MakeStaticBuffers(Device, Prim, 2.2f, Buffers[i]), "buffers %d", i);
        CHECK(!Renderer.CreateClusterGeom(Meshes[i], Buffers[i].Geom()), "geom %d", i);
        CHECK(!Renderer.SetMeshGI(Meshes[i], MakeSurfels(Prim, 0.3f, 0.9f)), "surfels %d", i);
    }

    xlionrender::scene Scene;
    for (int i = 0; i < n_v; ++i)
        Scene.m_Draws.push_back({ .m_Mesh = Meshes[i], .m_L2W = xmath::fmat4::fromTranslation(Pieces[i].m_Pos)
                                , .m_Material = { .m_BaseColor = Pieces[i].m_Color, .m_Roughness = 0.9f }, .m_Instance = std::uint64_t(i + 1) });
    Scene.m_Sun.m_Intensity = 0;
    Scene.m_Ambient         = { 0.02f, 0.02f, 0.02f };
    Scene.m_LocalLights     = { { .m_Position = { 0.4f, 3.0f, -0.2f }, .m_Radius = 9, .m_Color = { 1, 1, 1 }, .m_Intensity = 4, .m_bCastShadows = true } };

    constexpr int size_v = 192;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");
    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.3f });
    Camera.LookAt(xmath::fvec3{ 0, 1.6f, -1.9f }, xmath::fvec3{ 0, 0.6f, 1.0f });
    App.m_bQuietTimes = true;

    int W, H;
    for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render without GI");
    const auto Off = ReadRGB(Device, View, W, H);

    Scene.m_GI = { .m_bEnable = true, .m_RayBudget = 1 << 20 };             // every instance every frame
    for (int f = 0; f < 40; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render with GI");
    const auto On = ReadRGB(Device, View, W, H);
    const auto& St = Renderer.getGIStats();
    std::printf("GI: %d instances updated, %d surfels, %d rays, %.2f ms\n", St.m_Updated, St.m_Surfels, St.m_Rays, St.m_Ms);
    DumpRGB("gi_off", Off, W, H);
    DumpRGB("gi_on", On, W, H);

    // Floor near the green wall (bottom left of the image) vs near the red wall (bottom right): red / green ratios
    const auto Ratio = [&](const std::vector<std::array<float, 3>>& Img, float x0, float x1) { const auto m = Mean(Img, W, H, x0, 0.8f, x1, 0.95f); return m[0] / std::max(m[1], 1e-6); };
    const double OffL = Ratio(Off, 0.15f, 0.3f), OffR = Ratio(Off, 0.7f, 0.85f), OnL = Ratio(On, 0.15f, 0.3f), OnR = Ratio(On, 0.7f, 0.85f);
    const auto MOff = Mean(Off, W, H, 0, 0, 1, 1), MOn = Mean(On, W, H, 0, 0, 1, 1);
    std::printf("floor red/green - without GI: left %.2f right %.2f; with GI: left %.2f right %.2f; mean luminance %.3f -> %.3f\n"
               , OffL, OffR, OnL, OnR, 0.2126 * MOff[0] + 0.7152 * MOff[1] + 0.0722 * MOff[2], 0.2126 * MOn[0] + 0.7152 * MOn[1] + 0.0722 * MOn[2]);
    CHECK(std::abs(OffL - OffR) < 0.1, "without GI the floor's two sides differ in color (%.2f vs %.2f)", OffL, OffR);
    CHECK(OnR > OnL * 1.15, "no color bleeding with GI (near red %.2f, near green %.2f)", OnR, OnL);
    CHECK(MOn[1] > MOff[1] * 1.05, "GI adds no light");

    // The per-pixel read (the mesh's read grid, scene::m_GI.m_bPerPixel, on above): on these dense meshes it matches the vertex read; on the
    // same room with 2 triangles a face it stays near the dense image, where the vertex read (the corners only) does not
    const auto Luma = [](const std::array<float, 3>& c) { return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]; };
    const auto Diff = [&](const std::vector<std::array<float, 3>>& A, const std::vector<std::array<float, 3>>& B)
    {
        double S = 0, R = 0;
        for (std::size_t i = 0; i < A.size(); ++i) { S += std::abs(Luma(A[i]) - Luma(B[i])); R += Luma(B[i]); }
        return S / std::max(R, 1e-9);
    };
    {
        Scene.m_GI.m_bPerPixel = false;
        for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render dense, vertex read");
        const auto DenseV = ReadRGB(Device, View, W, H);
        Scene.m_GI.m_bPerPixel = true;

        std::array<cluster_buffers, n_v>      SparseBuffers;
        std::array<xlionrender::mesh_id, n_v> SparseMeshes;
        xlionrender::scene Sparse = Scene;
        for (int i = 0; i < n_v; ++i)
        {
            const auto Prim = xprim_geom::cube::Generate(1, 1, 1, 1, Pieces[i].m_Size);
            CHECK(MakeStaticBuffers(Device, Prim, 2.2f, SparseBuffers[i]), "sparse buffers %d", i);
            CHECK(!Renderer.CreateClusterGeom(SparseMeshes[i], SparseBuffers[i].Geom()), "sparse geom %d", i);
            CHECK(!Renderer.SetMeshGI(SparseMeshes[i], MakeSurfels(Prim, 0.3f, 0.9f)), "sparse surfels %d", i);
            Sparse.m_Draws[i].m_Mesh     = SparseMeshes[i];
            Sparse.m_Draws[i].m_Instance = std::uint64_t(101 + i);
        }
        for (int f = 0; f < 40; ++f) CHECK(App.RenderFrame(Sparse, View, Camera), "Render sparse, pixel read");
        const auto SparseP = ReadRGB(Device, View, W, H);
        Sparse.m_GI.m_bPerPixel = false;
        for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Sparse, View, Camera), "Render sparse, vertex read");
        const auto SparseV = ReadRGB(Device, View, W, H);
        DumpRGB("gi_dense_vertex", DenseV, W, H);
        DumpRGB("gi_sparse_pixel", SparseP, W, H);
        DumpRGB("gi_sparse_vertex", SparseV, W, H);

        const double DenseErr = Diff(DenseV, On), PixelErr = Diff(SparseP, On), VertexErr = Diff(SparseV, On);
        std::printf("per-pixel read vs the dense image: dense vertex read %.1f%%; sparse (2 triangles a face): pixel read %.1f%%, vertex read %.1f%%\n"
                   , 100 * DenseErr, 100 * PixelErr, 100 * VertexErr);
        CHECK(DenseErr < 0.05, "on dense meshes the pixel and the vertex reads differ by %.1f%%", 100 * DenseErr);
        CHECK(PixelErr < 0.04 && PixelErr < 0.75 * VertexErr,     // the dense reads differ by ~2% themselves (vertex spacing vs surfel size)
                  "on sparse meshes the pixel read is %.1f%% off the dense image (the vertex read %.1f%%)", 100 * PixelErr, 100 * VertexErr);
        for (int f = 0; f < 2; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render dense again");
        for (auto& M : SparseMeshes) Renderer.DestroyMesh(M);
    }

    // The GPU's per-pixel lookup agrees with the CPU's read grid: in the SURFELS debug view every pixel of the room (the camera is inside
    // it: every pixel is geometry) has a surfel's color, none is black (a cell the GPU does not find)
    {
        Scene.m_GI.m_Debug = xlionrender::gi_debug::SURFELS;
        for (int f = 0; f < 2; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render the surfels view");
        const auto Ids = ReadRGB(Device, View, W, H);
        Scene.m_GI.m_Debug = xlionrender::gi_debug::NONE;
        DumpRGB("gi_surfels", Ids, W, H);
        int nBlack = 0;
        for (const auto& c : Ids) nBlack += std::max({ c[0], c[1], c[2] }) < 1e-3f;
        std::printf("surfels view: %d of %d pixels black (no surfel found)\n", nBlack, W * H);
        CHECK(nBlack <= W * H / 1000, "the GPU's lookup misses: %d of %d pixels black in the surfels view", nBlack, W * H);
    }

    // The box moves: its instance moves in the world (no new instance); removed: the world drops it
    Scene.m_Draws[5].m_L2W = xmath::fmat4::fromTranslation({ 0.8f, 0.8f, 0.4f });
    for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render moved box");
    Scene.m_Draws.pop_back();
    for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render without the box");

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
