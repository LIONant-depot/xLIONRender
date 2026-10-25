// xGPU's streaming lane (Details/Vulkan/xgpu_vulkan_upload.h), through the renderer's device: textures and buffers created with data are
// copied on the transfer queue from a fixed staging ring - never by waiting the main queue idle - and every frame waits on the GPU for
// them. Run with a 1 MB ring (XGPU_UPLOAD_RING_MB), so 24 textures of 256 KB wrap it several times and one of 4 MB takes a staging
// buffer of its own:
//  - creating them (and a mesh: its buffers) waits the main queue idle 0 times
//  - after the point FlushUploads returns, every texel reads back as written (each texture its own value)
//  - a frame drawn right after creating a texture, without any flush, samples it (the frame waits on the GPU, the acquire runs first)
#include "xlionrender_test_common.h"

using namespace xlionrender::test;

int main()
{
#ifdef _WIN32
    _putenv_s("XGPU_UPLOAD_RING_MB", "1");
#else
    setenv("XGPU_UPLOAD_RING_MB", "1", 1);
#endif
    app App;
    CHECK(App.Init(), "Init");
    auto& Device   = App.m_Device;
    auto& Renderer = App.m_Renderer;

    const auto MakeTexture = [&](xgpu::texture& T, int Side, std::uint8_t Red) -> bool
    {
        std::vector<std::uint32_t> Texels(std::size_t(Side) * Side, 0xff000000u | Red);
        const auto Mips = std::array{ xgpu::texture::setup::mip_size{ std::uint32_t(Texels.size() * 4) } };
        return !Device.Create(T, { .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = Side, .m_Height = Side, .m_MipSizes = Mips
                                 , .m_AllFacesData = { reinterpret_cast<const std::byte*>(Texels.data()), Texels.size() * 4 }, .m_isGamma = false });
    };

    // Uploads only: no main-queue wait
    const auto Waits0 = Device.getMainQueueWaits();
    std::array<xgpu::texture, 24> Small;
    for (int i = 0; i < int(Small.size()); ++i) CHECK(MakeTexture(Small[i], 256, std::uint8_t(10 + 9 * i)), "texture %d", i);
    xgpu::texture Big;
    CHECK(MakeTexture(Big, 1024, 251), "big texture (past half the ring: a staging buffer of its own)");
    xlionrender::mesh_id Mesh;
    {
        const auto Prim = MakeSphere();
        std::vector<xlionrender::vertex> Verts(Prim.m_Vertices.size());
        for (std::size_t i = 0; i < Verts.size(); ++i)
        {
            const auto& V = Prim.m_Vertices[i];
            Verts[i] = { { V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }, { V.m_Texcoord.m_X, V.m_Texcoord.m_Y }, 0xffffffffu
                       , { V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }, { V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, V.m_Tangent.m_D } };
        }
        CHECK(!Renderer.CreateMesh(Mesh, Verts, Prim.m_Indices), "CreateMesh");
    }
    CHECK(Device.getMainQueueWaits() == Waits0, "Creating 25 textures and a mesh waited the main queue idle %llu times", (unsigned long long)(Device.getMainQueueWaits() - Waits0));

    // Their point: once passed, everything reads back as written (6 MB through a 1 MB ring: it wrapped)
    const auto Point = Device.FlushUploads();
    Device.Wait(Point);
    CHECK(Device.isDone(Point), "The uploads' point is done after Wait");
    std::vector<std::uint32_t> Read;
    int W, H;
    for (int i = 0; i < int(Small.size()); ++i)
    {
        CHECK(!Device.ReadTexture(Small[i], Read, W, H) && W == 256 && H == 256, "read back %d", i);
        const std::uint32_t Want = 0xff000000u | std::uint32_t(10 + 9 * i);
        int nOff = 0;
        for (auto t : Read) nOff += t != Want;
        CHECK(nOff == 0, "Texture %d: %d texels of %d are not %08X (the ring's wrap)", i, nOff, W * H, Want);
    }
    CHECK(!Device.ReadTexture(Big, Read, W, H) && W == 1024 && std::ranges::all_of(Read, [](std::uint32_t t) { return t == 0xff0000fbu; }), "The big texture reads back");

    // A frame drawn right after creating a texture samples it: the material reads it in slot 1 (red * 3), no flush in between
    {
        static constexpr std::uint32_t s_Frag[] = {
            #include "xlionrender_test_material_frag.h"
        };
        xgpu::shader Frag;
        CHECK(!Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_Frag), std::size(s_Frag) } } }), "material shader");
        constexpr int size_v = 64;
        xlionrender::render_view View;
        CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");
        xgpu::tools::view Camera;
        Camera.setFov(xmath::radian{ 1.0f });
        Camera.LookAt(xmath::fvec3{ 0, 0, -1.6f }, xmath::fvec3{ 0, 0, 0 });

        cluster_buffers Buffers;                                            // (materials draw on cluster meshes)
        CHECK(MakeStaticBuffers(Device, MakeSphere(), 0.5f, Buffers), "static geom buffers");
        xlionrender::mesh_id Static;
        CHECK(!Renderer.CreateClusterGeom(Static, Buffers.Geom()), "CreateClusterGeom");
        xgpu::texture Fresh;
        CHECK(MakeTexture(Fresh, 4, 128), "fresh texture");
        xlionrender::material_id Material;
        const auto Slots = std::array<xgpu::texture*, 2>{ nullptr, &Fresh };
        CHECK(!Renderer.CreateMaterial(Material, Frag, Slots), "CreateMaterial");
        xlionrender::scene Scene;
        Scene.m_Draws = { { .m_Mesh = Static, .m_Materials = { Material } } };
        std::vector<float> M;
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, M), "Render");
        int n = 0;
        for (float v : M) if (v != 0) { CHECK(std::abs(v - 3.0f * 128 / 255) < 0.01f, "Texel %f, expected %f", v, 3.0f * 128 / 255); ++n; }
        CHECK(n > size_v * size_v / 10, "Only %d texels drawn", n);
        Renderer.DestroyMaterial(Material);
        Renderer.DestroyView(View);
    }

    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
