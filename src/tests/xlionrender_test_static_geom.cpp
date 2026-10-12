// xgeom_static through the new renderer. Differential test: the same sphere drawn
//  (a) as a float mesh through the reference forward path, and
//  (b) quantized into the xgeom_static GPU format (int16 positions per cluster, oct-encoded normal/tangent, uint16
//      UVs and indices, cluster SSBO) and decoded by GeomStaticBasicShader_vert.glsl,
// must give the same HDR image up to quantization. The test does the quantization itself (mirroring the decode in
// shaders/xgeom_static_mb_input_full.vert) so it needs no compiled resources.
#include "xlionrender_test_common.h"

using namespace xlionrender::test;

namespace
{
    // xgeom_static::geom layouts (xgeom_static.plugin/source/xgeom_static.h)
    struct gs_vertex        { std::int16_t m_X, m_Y, m_Z, m_Extra; };
    struct gs_vertex_extras { std::uint16_t m_UV[2]; std::uint8_t m_OctNormal[2]; std::uint8_t m_OctTangent[2]; };
    struct gs_cluster_data  { float m_PosScale[4]; float m_PosTranslation[4]; float m_UVScaleTranslation[4]; };

    void OctEncode(float X, float Y, float Z, float& U, float& V) noexcept
    {
        const float L = std::abs(X) + std::abs(Y) + std::abs(Z);
        X /= L; Y /= L;
        if (Z < 0)
        {
            const float OX = X;
            X = (1 - std::abs(Y))  * (OX >= 0 ? 1.f : -1.f);
            Y = (1 - std::abs(OX)) * (Y  >= 0 ? 1.f : -1.f);
        }
        U = X * 0.5f + 0.5f;
        V = Y * 0.5f + 0.5f;
    }

    std::uint32_t Quantize(float V01, std::uint32_t Max) noexcept { return static_cast<std::uint32_t>(std::lround(std::clamp(V01, 0.f, 1.f) * Max)); }
}

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    const auto Prim = MakeSphere();
    CHECK(Prim.m_Vertices.size() < 65536, "uint16 indices");

    // (a) reference float mesh
    xlionrender::mesh_id Reference;
    {
        std::vector<xlionrender::vertex> Verts(Prim.m_Vertices.size());
        for (std::size_t i = 0; i < Verts.size(); ++i)
        {
            const auto& V = Prim.m_Vertices[i];
            Verts[i] = { { V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }, { V.m_Texcoord.m_X, V.m_Texcoord.m_Y }, 0xffffffffu
                       , { V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }, { V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, V.m_Tangent.m_D } };
        }
        CHECK(!Renderer.CreateMesh(Reference, Verts, Prim.m_Indices), "CreateMesh");
    }

    // (b) the same sphere as one xgeom_static cluster
    const float Scale = 0.5f;                               // positions in [-0.5, 0.5]: cluster scale 0.5, translation 0
    std::vector<gs_vertex>        Pos(Prim.m_Vertices.size());
    std::vector<gs_vertex_extras> Extras(Prim.m_Vertices.size());
    for (std::size_t i = 0; i < Pos.size(); ++i)
    {
        const auto& V = Prim.m_Vertices[i];
        const auto  Q = [&](float P) { return static_cast<std::int16_t>(std::clamp(std::lround((P / Scale + 1.0f) * 32767.5f - 32768.0f), -32768l, 32767l)); };

        float NU, NV, TU, TV;
        OctEncode(V.m_Normal.m_X,  V.m_Normal.m_Y,  V.m_Normal.m_Z,  NU, NV);
        OctEncode(V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, TU, TV);
        const std::uint32_t Nx = Quantize(NU, 4095), Ny = Quantize(NV, 4095), Tx = Quantize(TU, 4095), Ty = Quantize(TV, 2047);
        const std::uint32_t Extra = (Nx & 15) | ((Ny & 15) << 4) | ((Tx & 15) << 8) | ((Ty & 7) << 12) | ((V.m_Tangent.m_D < 0 ? 1u : 0u) << 15);

        Pos[i]    = { Q(V.m_Position.m_X), Q(V.m_Position.m_Y), Q(V.m_Position.m_Z), static_cast<std::int16_t>(static_cast<std::uint16_t>(Extra)) };
        Extras[i] = { { static_cast<std::uint16_t>(Quantize(V.m_Texcoord.m_X, 65535)), static_cast<std::uint16_t>(Quantize(V.m_Texcoord.m_Y, 65535)) }
                    , { static_cast<std::uint8_t>(Nx >> 4), static_cast<std::uint8_t>(Ny >> 4) }
                    , { static_cast<std::uint8_t>(Tx >> 4), static_cast<std::uint8_t>(Ty >> 3) } };
    }
    std::vector<std::uint16_t> Idx16(Prim.m_Indices.begin(), Prim.m_Indices.end());
    const gs_cluster_data Cluster{ { Scale, Scale, Scale, 0 }, { 0, 0, 0, 0 }, { 1, 1, 0, 0 } };

    // Same buffers the xgeom_static loader creates, in its order: index, vertex, extras (contiguous) + the cluster SSBO
    std::array<xgpu::buffer, 3> Streams;
    xgpu::buffer                ClusterData;
    CHECK(!Device.Create(Streams[0],  { .m_Type = xgpu::buffer::type::INDEX,   .m_EntryByteSize = sizeof(std::uint16_t),    .m_EntryCount = int(Idx16.size()),  .m_pData = Idx16.data()  }), "index buffer");
    CHECK(!Device.Create(Streams[1],  { .m_Type = xgpu::buffer::type::VERTEX,  .m_EntryByteSize = sizeof(gs_vertex),        .m_EntryCount = int(Pos.size()),    .m_pData = Pos.data()    }), "vertex buffer");
    CHECK(!Device.Create(Streams[2],  { .m_Type = xgpu::buffer::type::VERTEX,  .m_EntryByteSize = sizeof(gs_vertex_extras), .m_EntryCount = int(Extras.size()), .m_pData = Extras.data() }), "extras buffer");
    CHECK(!Device.Create(ClusterData, { .m_Type = xgpu::buffer::type::STORAGE, .m_EntryByteSize = sizeof(gs_cluster_data),  .m_EntryCount = 1, .m_pData = const_cast<gs_cluster_data*>(&Cluster) }), "cluster buffer");

    xlionrender::mesh_id Static;
    {
        xlionrender::static_geom G{ .m_pStreams = Streams.data(), .m_pClusterData = &ClusterData };
        G.m_Clusters  = { { 0, std::uint32_t(Idx16.size()), 0 } };
        G.m_Submeshes = { { 0, 1, 0 } };
        CHECK(!Renderer.CreateStaticGeom(Static, std::move(G)), "CreateStaticGeom");
    }

    constexpr int size_v = 128;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");

    // Off-axis camera and a raking light so normals, positions and the specular lobe all matter
    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.0f });
    Camera.LookAt(xmath::fvec3{ 0.6f, 0.4f, -1.8f }, xmath::fvec3{ 0, 0, 0 });

    xlionrender::scene Scene;
    Scene.m_Light     = { .m_Direction = { -0.4f, -1.0f, 0.5f }, .m_Intensity = 3 };
    Scene.m_Ambient = { 0.05f, 0.05f, 0.05f };
    const xlionrender::material Mat{ .m_BaseColor = { 0.7f, 0.7f, 0.7f }, .m_Metallic = 0.2f, .m_Roughness = 0.35f };

    std::vector<float> A, B;
    Scene.m_Draws = { { .m_Mesh = Reference, .m_Material = Mat } };
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, A), "Render reference");
    Scene.m_Draws = { { .m_Mesh = Static, .m_Material = Mat } };
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, B), "Render static geom");

    // Interior texels must match within quantization; silhouette texels may flip coverage
    int nCovered = 0, nOff = 0;
    double MaxRel = 0;
    for (std::size_t i = 0; i < A.size(); ++i)
    {
        CHECK(std::isfinite(B[i]), "Non-finite texel %d", int(i));
        if (A[i] == 0 && B[i] == 0) continue;
        ++nCovered;
        if ((A[i] == 0) != (B[i] == 0)) { ++nOff; continue; }
        const double Rel = std::abs(B[i] - A[i]) / std::max(A[i], 0.02f);
        if (Rel > 0.03) ++nOff;
        MaxRel = std::max(MaxRel, Rel);
    }
    std::printf("covered %d, differing %d, max relative %.4f\n", nCovered, nOff, MaxRel);
    CHECK(nCovered > size_v * size_v / 10, "Only %d covered texels", nCovered);
    CHECK(nOff <= nCovered / 50, "%d of %d texels differ beyond quantization", nOff, nCovered);

    // A material (xmaterial layout): its HDR output lands unclamped, the overlay runs, the output alpha is coverage
    {
        static constexpr std::uint32_t s_Frag[] = {
            #include "xlionrender_test_material_frag.h"
        };
        xgpu::shader Frag;
        CHECK(!Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_Frag), std::size(s_Frag) } } }), "material shader");

        const std::uint32_t Texel = 0xff0080ffu;            // R = 1.0, G ~ 0.5, B = 0 (UNORM, little endian RGBA)
        const auto          Mips  = std::array{ xgpu::texture::setup::mip_size{ sizeof(Texel) } };
        xgpu::texture       Texture;
        CHECK(!Device.Create(Texture, { .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = 1, .m_Height = 1, .m_MipSizes = Mips
                                      , .m_AllFacesData = { reinterpret_cast<const std::byte*>(&Texel), sizeof(Texel) }, .m_isGamma = false }), "material texture");

        xlionrender::material_id Material;
        const auto Slots = std::array<xgpu::texture*, 2>{ nullptr, &Texture };      // slot 0: no shadow map
        CHECK(!Renderer.CreateMaterial(Material, Frag, Slots), "CreateMaterial");

        Scene.m_Draws = { { .m_Mesh = Static, .m_Material = Mat, .m_Materials = { Material } } };
        int nOverlay = 0;
        std::vector<float> M;
        CHECK(App.RenderFrame(Scene, View, Camera, [&](xgpu::cmd_buffer&) { ++nOverlay; }) && ReadHDR(Device, View, M), "Render material");
        CHECK(nOverlay == 1, "Overlay ran %d times", nOverlay);

        int nCovered3 = 0;
        for (float v : M) if (v != 0) { CHECK(std::abs(v - 3.0f) < 0.01f, "Material texel %f, expected 3 (texture * 3, HDR)", v); ++nCovered3; }
        CHECK(nCovered3 == nCovered, "Material covered %d texels, the standard path %d", nCovered3, nCovered);

        std::vector<std::uint32_t> LDR;
        int W, H;
        CHECK(!Device.ReadTexture(View.m_Output, LDR, W, H), "Read output");
        CHECK((LDR[0] >> 24) == 0,                                 "Background alpha %u, expected 0", LDR[0] >> 24);
        CHECK((LDR[(size_v / 2) * size_v + size_v / 2] >> 24) == 255, "Geometry alpha %u, expected 255", LDR[(size_v / 2) * size_v + size_v / 2] >> 24);

        // A destroyed material falls back to the standard one
        Renderer.DestroyMaterial(Material);
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, M), "Render destroyed material");
        for (std::size_t i = 0; i < M.size(); ++i) CHECK(std::abs(M[i] - B[i]) <= 1e-3f * std::max(1.0f, B[i]), "Fallback texel %d: %f vs %f", int(i), M[i], B[i]);

        Device.Destroy(std::move(Texture));
    }

    // Stale-id safety does not touch the buffers it references (the caller owns them)
    Renderer.DestroyMesh(Static);
    std::vector<float> C;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, C), "Render after destroy");
    for (float v : C) CHECK(v == 0, "A destroyed static geom drew something");

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
