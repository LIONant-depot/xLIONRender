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

    Scene.m_GI = { .m_bEnable = true, .m_RayBudget = 1 << 20, .m_FilterPasses = 2 };    // every instance every frame; the read checks below were tuned at 2 filter passes (more blurs the per-pixel / vertex difference they measure)
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
        // (0.8: with validity from near back faces only - the box edges' surfels valid again - it is 2.3% vs 3.0%; it was 2.0% vs 3.2%)
        CHECK(PixelErr < 0.04 && PixelErr < 0.8 * VertexErr,      // the dense reads differ by ~2% themselves (vertex spacing vs surfel size)
                  "on sparse meshes the pixel read is %.1f%% off the dense image (the vertex read %.1f%%)", 100 * PixelErr, 100 * VertexErr);
        for (int f = 0; f < 2; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render dense again");
        for (auto& M : SparseMeshes) Renderer.DestroyMesh(M);
    }

    // The material capture: each mesh's surfels drawn through its material (here the constant one, from the instance record) and read
    // back: the red wall's surfels have its albedo, the floor's its gray, and nothing emits
    {
        for (int f = 0; f < 200 && Renderer.isGICapturing(); ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render while capturing");
        const auto* pW = Renderer.getGIWorld();
        CHECK(pW && !Renderer.isGICapturing(), "the captures did not finish");
        const auto Mean = [&](const std::vector<xmath::fvec3>& V) { xmath::fvec3 m{ 0, 0, 0 }; for (const auto& v : V) m = m + v; return V.empty() ? m : m * (1.0f / float(V.size())); };
        const auto Red = Mean(pW->SurfelAlbedo(Renderer.getGIId(4))), Floor = Mean(pW->SurfelAlbedo(Renderer.getGIId(1))), Glow = Mean(pW->SurfelEmission(Renderer.getGIId(4)));
        std::printf("captured: red wall %.3f %.3f %.3f (0.8 0.05 0.05), floor %.3f %.3f %.3f (0.75), emission %.3f\n", Red.m_X, Red.m_Y, Red.m_Z, Floor.m_X, Floor.m_Y, Floor.m_Z, Glow.m_X + Glow.m_Y + Glow.m_Z);
        CHECK(std::abs(Red.m_X - 0.8f) < 0.05f && Red.m_Y < 0.1f && std::abs(Floor.m_X - 0.75f) < 0.05f, "the captured albedo is wrong");
        CHECK(Glow.m_X + Glow.m_Y + Glow.m_Z < 0.01f, "something emits");
        // ... and each triangle its own (what a ray hit bounces)
        const auto& Tri = pW->TriangleAlbedo(Renderer.getGIId(4));
        const auto  TRed = Mean(Tri);
        std::printf("captured triangles: red wall %zu, %.3f %.3f %.3f\n", Tri.size(), TRed.m_X, TRed.m_Y, TRed.m_Z);
        CHECK(!Tri.empty() && std::abs(TRed.m_X - 0.8f) < 0.05f && TRed.m_Y < 0.1f, "the triangles' captured albedo is wrong");
    }

    // The red wall's material changes (a material compiled again makes a new material_id): its surfels and triangles are captured again.
    // Changed twice in a row (the first capture still in flight): only the last one's colors stay
    {
        const auto MakeColor = [&](std::uint32_t Texel, xgpu::texture& T, xlionrender::material_id& Id)
        {
            const auto Mips = std::array{ xgpu::texture::setup::mip_size{ sizeof(Texel) } };
            const auto Slots = std::array<xgpu::texture*, 2>{ nullptr, &T };       // slot 0: no shadow map
            return !Device.Create(T, { .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = 1, .m_Height = 1, .m_MipSizes = Mips
                                     , .m_AllFacesData = { reinterpret_cast<const std::byte*>(&Texel), sizeof(Texel) }, .m_isGamma = false })
                && !Renderer.CreateDefaultMaterial(Id, Slots);
        };
        xgpu::texture GreenT, BlueT;
        xlionrender::material_id Green, Blue;
        CHECK(MakeColor(0xff00cc00u, GreenT, Green) && MakeColor(0xffcc0000u, BlueT, Blue), "the green and blue materials");     // RGBA little endian
        const auto* pW = Renderer.getGIWorld();
        const auto Mean = [&](const std::vector<xmath::fvec3>& V) { xmath::fvec3 m{ 0, 0, 0 }; for (const auto& v : V) m = m + v; return V.empty() ? m : m * (1.0f / float(V.size())); };
        float MaxBlue = 0;                                                          // the wall's blue on any frame (the blue capture must never land)
        const auto Recapture = [&](std::vector<xlionrender::material_id> M)
        {
            Scene.m_Draws[3].m_Materials = std::move(M);
            bool bOk = App.RenderFrame(Scene, View, Camera);
            for (int f = 0; f < 200 && Renderer.isGICapturing(); ++f)
            {
                bOk = App.RenderFrame(Scene, View, Camera) && bOk;
                MaxBlue = std::max(MaxBlue, Mean(pW->SurfelAlbedo(Renderer.getGIId(4))).m_Z);
            }
            return bOk && !Renderer.isGICapturing();
        };
        CHECK(Recapture({ Green }), "the green capture did not finish");
        const auto G = Mean(pW->SurfelAlbedo(Renderer.getGIId(4))), GT = Mean(pW->TriangleAlbedo(Renderer.getGIId(4)));
        Scene.m_Draws[3].m_Materials = { Blue };
        CHECK(App.RenderFrame(Scene, View, Camera), "Render blue");               // its capture queued (and maybe recorded): then the material changes back
        MaxBlue = 0;
        CHECK(Recapture({}), "the red capture did not finish");
        CHECK(MaxBlue < 0.1f, "the capture of the material it no longer has landed (blue %.2f)", MaxBlue);
        const auto R = Mean(pW->SurfelAlbedo(Renderer.getGIId(4))), RT = Mean(pW->TriangleAlbedo(Renderer.getGIId(4)));
        std::printf("recaptured: green %.2f %.2f %.2f (triangles %.2f %.2f %.2f); blue then red: %.2f %.2f %.2f (triangles %.2f %.2f %.2f)\n"
                   , G.m_X, G.m_Y, G.m_Z, GT.m_X, GT.m_Y, GT.m_Z, R.m_X, R.m_Y, R.m_Z, RT.m_X, RT.m_Y, RT.m_Z);
        CHECK(G.m_Y > 0.5f && G.m_X < 0.1f && GT.m_Y > 0.5f && GT.m_X < 0.1f, "the new material was not captured again");
        CHECK(std::abs(R.m_X - 0.8f) < 0.05f && R.m_Z < 0.1f && std::abs(RT.m_X - 0.8f) < 0.05f && RT.m_Z < 0.1f, "a capture of a material the wall no longer has stayed");
        Renderer.DestroyMaterial(Green);
        Renderer.DestroyMaterial(Blue);
        Device.Destroy(std::move(GreenT));
        Device.Destroy(std::move(BlueT));
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

    // The SURFACE_COLOR view shows the dominant surfel's captured color (the red wall red, the floor gray)
    {
        const auto View3 = [&](xlionrender::gi_debug D)
        {
            Scene.m_GI.m_Debug = D;
            for (int f = 0; f < 2; ++f) (void)App.RenderFrame(Scene, View, Camera);     // (CHECK returns from main: not in a lambda)
            Scene.m_GI.m_Debug = xlionrender::gi_debug::NONE;
            return ReadRGB(Device, View, W, H);
        };
        const auto Color = View3(xlionrender::gi_debug::SURFACE_COLOR);
        DumpRGB("gi_surface_color", Color, W, H);
        const auto CRed = Mean(Color, W, H, 0.9f, 0.4f, 0.98f, 0.6f), CFloor = Mean(Color, W, H, 0.4f, 0.85f, 0.6f, 0.95f);
        std::printf("surface color: red wall %.2f %.2f %.2f, floor %.2f %.2f %.2f\n", CRed[0], CRed[1], CRed[2], CFloor[0], CFloor[1], CFloor[2]);
        CHECK(CRed[0] > 0.6 && CRed[1] < 0.15 && std::abs(CFloor[0] - 0.75) < 0.08, "the SURFACE_COLOR view does not show the captured colors");

        // The COVERAGE view (the GI query policy, GIQueryResult): which read lights each pixel - green the pixel read, blue the vertex
        // read, magenta none. The room is all supported per pixel; with per-pixel reads off, all by its vertices
        const auto Classes = [&](const std::vector<std::array<float, 3>>& C)
        {
            std::array<int, 4> n{};     // pixel, vertex, unsupported, other
            for (const auto& c : C) n[c[1] > 0.3f && c[0] < 0.1f && c[2] < 0.1f ? 0 : c[2] > 0.5f && c[0] < 0.1f ? 1 : c[0] > 0.5f && c[2] > 0.5f && c[1] < 0.1f ? 2 : 3]++;
            return n;
        };
        const auto Pixel = Classes(View3(xlionrender::gi_debug::COVERAGE));
        Scene.m_GI.m_bPerPixel = false;
        const auto Vertex = Classes(View3(xlionrender::gi_debug::COVERAGE));
        Scene.m_GI.m_bPerPixel = true;
        std::printf("coverage view: per pixel %d pixel / %d vertex / %d unsupported / %d other; per vertex only %d / %d / %d / %d\n"
                   , Pixel[0], Pixel[1], Pixel[2], Pixel[3], Vertex[0], Vertex[1], Vertex[2], Vertex[3]);
        CHECK(Pixel[0] >= W * H * 99 / 100 && Pixel[2] <= W * H / 1000, "the room is not read per pixel everywhere (%d of %d)", Pixel[0], W * H);
        CHECK(Vertex[1] >= W * H * 99 / 100, "per-pixel reads off, the room is not read by its vertices (%d of %d)", Vertex[1], W * H);
    }

    // The box moves: its instance moves in the world (no new instance); removed: the world drops it
    Scene.m_Draws[5].m_L2W = xmath::fmat4::fromTranslation({ 0.8f, 0.8f, 0.4f });
    for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render moved box");
    Scene.m_Draws.pop_back();
    for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "Render without the box");

    // The LIGHT view of a lone small mesh (the GeomStatic editor's preview: one object, surfels bigger than it, a sun and a flat ambient):
    // its irradiance / pi is about the ambient's (a white surface's look), not a saturated white
    {
        const auto Prim = xprim_geom::cube::Generate(4, 4, 4, 4, { 0.4f, 0.1f, 0.1f });
        cluster_buffers B; xlionrender::mesh_id M;
        CHECK(MakeStaticBuffers(Device, Prim, 2.2f, B), "lone buffers");
        CHECK(!Renderer.CreateClusterGeom(M, B.Geom()), "lone geom");
        CHECK(!Renderer.SetMeshGI(M, MakeSurfels(Prim, 0.5f, 1.5f)), "lone surfels");
        xlionrender::scene Lone;
        Lone.m_Draws.push_back({ .m_Mesh = M, .m_Material = { .m_BaseColor = { 0.75f, 0.75f, 0.75f }, .m_Roughness = 0.9f }, .m_Instance = 900 });
        Lone.m_Sun     = { .m_Direction = { 0.3f, -1.0f, 0.4f }, .m_Temperature = 6500, .m_Intensity = 4 };
        Lone.m_Ambient = { 0.7f, 0.7f, 0.7f };
        Lone.m_GI      = { .m_bEnable = true, .m_RayBudget = 1 << 16, .m_Debug = xlionrender::gi_debug::LIGHT };
        xgpu::tools::view Cam;
        Cam.setFov(xmath::radian{ 1.0f });
        Cam.LookAt(xmath::fvec3{ 0, 0.25f, -0.5f }, xmath::fvec3{ 0, 0, 0 });
        for (int f = 0; f < 60; ++f) CHECK(App.RenderFrame(Lone, View, Cam), "Render the lone mesh");
        const auto Img = ReadRGB(Device, View, W, H);
        DumpRGB("gi_lone_light", Img, W, H);
        const auto c = Mean(Img, W, H, 0.45f, 0.45f, 0.55f, 0.55f);
        float Max = 0; for (const auto& p : Img) Max = std::max({ Max, p[0], p[1], p[2] });
        std::printf("lone mesh LIGHT view: middle %.3f %.3f %.3f, max %.3f (the ambient / pi %.3f)\n", c[0], c[1], c[2], Max, 0.7f / 3.14159f);
        CHECK(c[0] < 1.0 && Max < 4.0f, "the lone mesh's LIGHT view is saturated (%.3f, max %.3f)", c[0], Max);

        // No read supports it (its vertices list no surfel, no per-pixel read): the mesh's mean light, not black (a scene lit by its GI
        // would be black there), and COVERAGE says unsupported (magenta)
        auto Unread = MakeSurfels(Prim, 0.5f, 1.5f);
        for (auto& v : Unread.m_VertSurfels) v = { -1, -1, -1, -1 };
        xlionrender::mesh_id M2;
        CHECK(!Renderer.CreateClusterGeom(M2, B.Geom()), "unread geom");
        CHECK(!Renderer.SetMeshGI(M2, std::move(Unread)), "unread surfels");
        Lone.m_Draws[0].m_Mesh = M2;
        Lone.m_GI.m_bPerPixel  = false;
        for (int f = 0; f < 60; ++f) CHECK(App.RenderFrame(Lone, View, Cam), "Render the unread mesh");
        const auto u = Mean(ReadRGB(Device, View, W, H), W, H, 0.45f, 0.45f, 0.55f, 0.55f);
        Lone.m_GI.m_Debug = xlionrender::gi_debug::COVERAGE;
        for (int f = 0; f < 2; ++f) CHECK(App.RenderFrame(Lone, View, Cam), "Render the unread mesh's coverage");
        const auto k = Mean(ReadRGB(Device, View, W, H), W, H, 0.45f, 0.45f, 0.55f, 0.55f);
        std::printf("unsupported everywhere: LIGHT middle %.3f (supported %.3f), coverage %.2f %.2f %.2f (magenta)\n", u[0], c[0], k[0], k[1], k[2]);
        CHECK(u[0] > 0.02f, "a point no read supports is black (%.3f)", u[0]);
        CHECK(k[0] > 0.5f && k[2] > 0.5f && k[1] < 0.1f, "the coverage view does not say unsupported");

        // The mesh made again with other surfels (a GeomStatic compiled again: its mesh destroyed and made again): the world adds the
        // instance again with the new surfels - also when they land at the old ones' address, which the world used to take for the same
        // ones (it then read past the new surfels with the old count)
        {
            Lone.m_GI.m_Debug = xlionrender::gi_debug::LIGHT;
            const int Before = Renderer.getGIId(900);
            Renderer.DestroyMesh(M2);
            CHECK(!Renderer.CreateClusterGeom(M2, B.Geom()), "remade geom");
            CHECK(!Renderer.SetMeshGI(M2, MakeSurfels(Prim, 0.8f, 2.4f)), "remade surfels");
            Lone.m_Draws[0].m_Mesh = M2;
            for (int f = 0; f < 10; ++f) CHECK(App.RenderFrame(Lone, View, Cam), "Render the remade mesh");
            CHECK(Renderer.getGIId(900) != Before, "the world kept the instance of the old surfels");
        }
        Renderer.DestroyMesh(M2);
        Renderer.DestroyMesh(M);
    }

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
