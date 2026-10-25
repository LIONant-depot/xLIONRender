// xgeom_static through the new renderer. Differential test: the same sphere drawn
//  (a) as a float mesh through the reference forward path, and
//  (b) quantized into the xgeom_static GPU format (int16 positions per cluster, oct-encoded normal/tangent, uint16
//      UVs and indices, cluster SSBO) and decoded by GeomStaticBasicShader_vert.glsl,
// must give the same HDR image up to quantization. The test does the quantization itself (mirroring the decode in
// shaders/xgeom_static_mb_input_full.vert) so it needs no compiled resources.
#include "xlionrender_test_common.h"

using namespace xlionrender::test;


int main()
{
    // Instance ids per entity (xlionrender_scene_renderer.h): entities 1 and 1<<32 had the same 32-bit hash (one slot for both)
    {
        xlionrender::instance_ids Ids;
        Ids.Begin();
        const auto A = Ids.Of(1), B = Ids.Of(1ull << 32);
        CHECK(A && B && A != B && Ids.Of(1) == A, "Two entities get two ids, stable within a frame");
        Ids.Begin();
        CHECK(Ids.Of(1) == A, "An id stays while its entity is drawn");
        Ids.Begin();                                    // B was not drawn for a whole frame: its id is free
        CHECK(Ids.Of(7) == B && Ids.Of(1) == A, "A freed id is reused, the drawn one kept");
    }

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

    // (b) the same sphere as one xgeom_static cluster (positions in [-0.5, 0.5]: cluster scale 0.5)
    cluster_buffers Buffers;
    CHECK(MakeStaticBuffers(Device, Prim, 0.5f, Buffers), "static geom buffers");
    xlionrender::mesh_id Static;
    CHECK(!Renderer.CreateClusterGeom(Static, Buffers.Geom()), "CreateClusterGeom");

    constexpr int size_v = 128;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");

    // Off-axis camera and a raking light so normals, positions and the specular lobe all matter
    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.0f });
    Camera.LookAt(xmath::fvec3{ 0.6f, 0.4f, -1.8f }, xmath::fvec3{ 0, 0, 0 });

    xlionrender::scene Scene;
    Scene.m_Sun     = { .m_Direction = { -0.4f, -1.0f, 0.5f }, .m_Intensity = 3 };
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

    // Materials of one fragment shader and state share their pipelines (instance::m_SharedPipelines): three with their own textures make one
    // rigid pipeline, and each still draws its own texture; another state is another pipeline; the last one destroyed lets it go
    {
        static constexpr std::uint32_t s_Frag[] = {
            #include "xlionrender_test_material_frag.h"
        };
        xgpu::shader Frag;
        CHECK(!Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_Frag), std::size(s_Frag) } } }), "material shader");

        const std::uint32_t Texels[3] = { 0xff0000ffu, 0xff000080u, 0xff000040u };      // red 1, ~0.5, ~0.25 (UNORM, little endian RGBA)
        std::array<xgpu::texture, 3>            Textures;
        std::array<xlionrender::material_id, 3> Materials;
        const int Before = Renderer.getPipelineCount();
        for (int i = 0; i < 3; ++i)
        {
            const auto Mips = std::array{ xgpu::texture::setup::mip_size{ sizeof(Texels[i]) } };
            CHECK(!Device.Create(Textures[i], { .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = 1, .m_Height = 1, .m_MipSizes = Mips
                                              , .m_AllFacesData = { reinterpret_cast<const std::byte*>(&Texels[i]), sizeof(Texels[i]) }, .m_isGamma = false }), "shared material texture");
            const auto Slots = std::array<xgpu::texture*, 2>{ nullptr, &Textures[i] };
            CHECK(!Renderer.CreateMaterial(Materials[i], Frag, Slots), "CreateMaterial %d", i);
        }
        CHECK(Renderer.getPipelineCount() == Before + 1, "Three materials of one shader and state: %d new pipelines, expected 1", Renderer.getPipelineCount() - Before);

        std::vector<float> M;
        for (int i = 0; i < 3; ++i)
        {
            Scene.m_Draws = { { .m_Mesh = Static, .m_Material = Mat, .m_Materials = { Materials[i] } } };
            CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, M), "Render shared material %d", i);
            const float Expected = 3.0f * float(Texels[i] & 0xff) / 255.0f;
            int n = 0;
            for (float v : M) if (v != 0) { CHECK(std::abs(v - Expected) < 0.01f, "Shared material %d: texel %f, expected %f", i, v, Expected); ++n; }
            CHECK(n == nCovered, "Shared material %d covered %d texels, the standard path %d", i, n, nCovered);
        }
        CHECK(Renderer.getPipelineCount() == Before + 1, "Drawing them made %d pipelines, expected 1", Renderer.getPipelineCount() - Before);

        xlionrender::material_id Other;
        const auto Slots = std::array<xgpu::texture*, 2>{ nullptr, &Textures[0] };
        CHECK(!Renderer.CreateMaterial(Other, Frag, Slots, { .m_bDoubleSided = true }), "CreateMaterial double sided");
        CHECK(Renderer.getPipelineCount() == Before + 2, "Another state: %d new pipelines, expected 2", Renderer.getPipelineCount() - Before);

        Renderer.DestroyMaterial(Materials[0]);
        Renderer.DestroyMaterial(Materials[1]);
        CHECK(Renderer.getPipelineCount() == Before + 2, "A pipeline stays while a material uses it");
        Scene.m_Draws = { { .m_Mesh = Static, .m_Material = Mat, .m_Materials = { Materials[2] } } };
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, M), "Render the last shared material");
        for (float v : M) if (v != 0) CHECK(std::abs(v - 3.0f * 64 / 255) < 0.01f, "The last shared material: texel %f", v);
        Renderer.DestroyMaterial(Materials[2]);
        Renderer.DestroyMaterial(Other);
        CHECK(Renderer.getPipelineCount() == Before, "Every material destroyed: %d pipelines left over", Renderer.getPipelineCount() - Before);
        for (auto& T : Textures) Device.Destroy(std::move(T));
    }

    // Material parameters (MaterialParams, set 1 binding 9): the second material's block sits after the first's; the shader finds it by MaterialBase
    {
        static constexpr std::uint32_t s_Frag[] = {
            #include "xlionrender_test_params_frag.h"
        };
        xgpu::shader Frag;
        CHECK(!Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_Frag), std::size(s_Frag) } } }), "params shader");
        const auto  Slots = std::array<xgpu::texture*, 1>{ nullptr };
        const float First[] = { 9, 9, 9, 9, 9, 9, 9, 9 }, Second[] = { 0, 0, 0, 0, 2, 0.5f, 0.25f, 1 };
        xlionrender::material_id MA, MB;
        CHECK(!Renderer.CreateMaterial(MA, Frag, Slots, {}, First) && !Renderer.CreateMaterial(MB, Frag, Slots, {}, Second), "CreateMaterial with parameters");

        Scene.m_Draws = { { .m_Mesh = Static, .m_Material = Mat, .m_Materials = { MB } } };
        std::vector<float> P;
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, P), "Render parameters");
        int nParams = 0;
        for (float v : P) if (v != 0) { CHECK(std::abs(v - 2.0f) < 1e-3f, "Parameter texel %f, expected 2 (the second block's)", v); ++nParams; }
        CHECK(nParams == nCovered, "Parameters covered %d texels, the standard path %d", nParams, nCovered);
        Renderer.DestroyMaterial(MA);
        Renderer.DestroyMaterial(MB);
    }

    // A material's render settings (material_state: xmaterial.plugin documentation/render_settings_plan.md), each its own pipeline. The
    // parameters' shader writes a constant red, so blending, culling and raster show as values and counts of the HDR red
    {
        static constexpr std::uint32_t s_Frag[] = {
            #include "xlionrender_test_params_frag.h"
        };
        static constexpr std::uint32_t s_Cut[] = {
            #include "xlionrender_test_cutout_frag.h"
        };
        xgpu::shader Frag, CutFrag;
        CHECK(!Device.Create(Frag,    { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_Frag), std::size(s_Frag) } } }), "params shader");
        CHECK(!Device.Create(CutFrag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_Cut), std::size(s_Cut) } } }), "cut out shader");
        using st = xlionrender::material_state;
        const auto  Slots   = std::array<xgpu::texture*, 1>{ nullptr };
        const float Quarter[] = { 0, 0, 0, 0, 0.25f, 0, 0, 1 }, Two[] = { 0, 0, 0, 0, 2, 0, 0, 1 };
        xlionrender::material_id Add, AddBoth, Wire, Opaque, NoShadow, Cut;
        CHECK(!Renderer.CreateMaterial(Add,      Frag, Slots, { .m_Blend = st::blend::ADD, .m_bDepthWrite = false }, Quarter), "additive material");
        CHECK(!Renderer.CreateMaterial(AddBoth,  Frag, Slots, { .m_Blend = st::blend::ADD, .m_bDoubleSided = true, .m_bDepthWrite = false }, Quarter), "double sided additive material");
        CHECK(!Renderer.CreateMaterial(Wire,     Frag, Slots, { .m_bWireframe = true }, Two), "wireframe material");
        CHECK(!Renderer.CreateMaterial(Opaque,   Frag, Slots, {}, Two), "opaque material");
        CHECK(!Renderer.CreateMaterial(NoShadow, Frag, Slots, { .m_bCastShadows = false }, Two), "material without shadows");
        CHECK(!Renderer.CreateMaterial(Cut,      CutFrag, Slots, { .m_Blend = st::blend::MASKED }, {}), "masked material");

        std::vector<float> I;
        const auto Count = [&](float V) { int n = 0; for (float v : I) n += std::abs(v - V) < 1e-3f; return n; };
        const auto Draw  = [&](xlionrender::material_id M) { return xlionrender::draw{ .m_Mesh = Static, .m_Material = Mat, .m_Materials = { M } }; };

        Scene.m_Draws = { Draw(Add), Draw(Add) };
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, I), "Render additive");
        CHECK(Count(0.5f) == nCovered, "Additive twice over itself: %d texels at 0.5, %d covered", Count(0.5f), nCovered);

        Scene.m_Draws = { Draw(AddBoth) };
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, I), "Render double sided");
        CHECK(Count(0.5f) > nCovered * 95 / 100, "Double sided: the back faces add too, %d texels at 0.5 of %d", Count(0.5f), nCovered);

        Scene.m_Draws = { Draw(Add), Draw(Opaque) };        // the additive one first: it still lands over the opaque one
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, I), "Render blended after opaque");
        CHECK(Count(2.25f) == nCovered, "Blended after the opaque ones: %d texels at 2.25, %d covered", Count(2.25f), nCovered);

        Scene.m_Draws = { Draw(Wire) };
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, I), "Render wireframe");
        CHECK(Count(2.0f) > 0 && Count(2.0f) < nCovered * 9 / 10, "Wireframe (a dense sphere: gaps between its edges): %d edge texels of %d covered", Count(2.0f), nCovered);

        // Shadows on a ground (the reference path, shadowed by the sun's clipmap): Cast Shadows off and a masked material cut out everywhere
        // leave the ground as with the sun's shadows off
        xlionrender::mesh_id Ground;
        {
            const auto Box = xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 });
            std::vector<xlionrender::vertex> Verts(Box.m_Vertices.size());
            for (std::size_t i = 0; i < Verts.size(); ++i)
            {
                const auto& V = Box.m_Vertices[i];
                Verts[i] = { { V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }, { V.m_Texcoord.m_X, V.m_Texcoord.m_Y }, 0xffffffffu
                           , { V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }, { V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, V.m_Tangent.m_D } };
            }
            CHECK(!Renderer.CreateMesh(Ground, Verts, Box.m_Indices), "ground");
        }
        const auto Shadowed = [&](xlionrender::material_id M, std::vector<float>& On, std::vector<float>& Off) -> bool
        {
            Scene.m_Draws = { { .m_Mesh = Ground, .m_L2W = xmath::fmat4::fromTranslation({ 0, -0.6f, 0 }) * xmath::fmat4::fromScale({ 10.0f, 0.1f, 10.0f }), .m_Material = Mat }, Draw(M) };
            Scene.m_Sun.m_bCastShadows = false;
            const bool bOk = App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Off);
            Scene.m_Sun.m_bCastShadows = true;
            return bOk && App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, On);
        };
        const auto Darker = [](const std::vector<float>& On, const std::vector<float>& Off) { int n = 0; for (std::size_t i = 0; i < On.size(); ++i) n += On[i] < Off[i] * 0.8f; return n; };
        std::vector<float> On, Off;
        CHECK(Shadowed(Opaque, On, Off) && Darker(On, Off) > 50, "An opaque material casts a shadow (%d darker texels)", Darker(On, Off));
        CHECK(Shadowed(NoShadow, On, Off) && Darker(On, Off) == 0, "Cast Shadows off: %d darker texels", Darker(On, Off));
        CHECK(Shadowed(Cut, On, Off) && Darker(On, Off) == 0, "A masked material cut out everywhere casts no shadow: %d darker texels", Darker(On, Off));

        for (auto M : { Add, AddBoth, Wire, Opaque, NoShadow, Cut }) Renderer.DestroyMaterial(M);
        Renderer.DestroyMesh(Ground);
    }

    // Entity parameters (EntityParams, set 1 binding 10, material_contract.md tests 5-6): one material whose Tint (block vec4 0) is red;
    // A and B override it (green, blue), C has no entity values (mask 0) and keeps the material's red. Then B's tint is edited: one slot
    // is written, nothing else changes
    {
        static constexpr std::uint32_t s_Frag[] = {
            #include "xlionrender_test_entity_frag.h"
        };
        xgpu::shader Frag;
        CHECK(!Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_Frag), std::size(s_Frag) } } }), "entity shader");
        const auto  Slots = std::array<xgpu::texture*, 1>{ nullptr };
        const float Block[] = { 1, 0, 0, 1 };
        xlionrender::material_id M;
        CHECK(!Renderer.CreateMaterial(M, Frag, Slots, {}, Block), "CreateMaterial with an entity tint");

        xgpu::tools::view Front;
        Front.setFov(xmath::radian{ 1.0f });
        Front.LookAt(xmath::fvec3{ 0, 0, -4 }, xmath::fvec3{ 0, 0, 0 });
        const auto Tinted = [](float R, float G, float B) { return xlionrender::entity_params{ .m_Mask = xlionrender::entity_params::TINT, .m_Tint = { R, G, B, 1 } }; };
        Scene.m_Draws =
        { { .m_Mesh = Static, .m_L2W = xmath::fmat4::fromTranslation({ -1.2f, 0, 0 }), .m_Material = Mat, .m_Materials = { M }, .m_Instance = 11, .m_EntityParams = Tinted(0, 1, 0) }
        , { .m_Mesh = Static, .m_L2W = xmath::fmat4::fromTranslation({  1.2f, 0, 0 }), .m_Material = Mat, .m_Materials = { M }, .m_Instance = 12, .m_EntityParams = Tinted(0, 0, 1) }
        , { .m_Mesh = Static, .m_L2W = xmath::fmat4::fromIdentity(),                    .m_Material = Mat, .m_Materials = { M }, .m_Instance = 13 }
        };

        // Per texel rgb (ReadHDR has red only); the probes: the middle sphere (C) and the two sides (A and B, in whichever order the camera puts them)
        const auto ReadRGB = [&](std::vector<xmath::fvec3>& Out) -> bool
        {
            std::vector<std::byte> Raw;
            int W, H;
            if (Device.ReadTexture(View.m_HDR, Raw, W, H)) return false;
            const auto* p = reinterpret_cast<const std::uint16_t*>(Raw.data());
            Out.resize(std::size_t(W) * H);
            for (std::size_t i = 0; i < Out.size(); ++i) Out[i] = { HalfToFloat(p[i * 4]), HalfToFloat(p[i * 4 + 1]), HalfToFloat(p[i * 4 + 2]) };
            return true;
        };
        const int  Side = int(0.5f * size_v * (1 + 1.2f / (4 * std::tan(0.5f)))), Mid = size_v / 2;
        const auto At   = [&](const std::vector<xmath::fvec3>& I, int x) { return I[std::size_t(Mid) * size_v + x]; };
        const auto Is   = [](const xmath::fvec3& C, float R, float G, float B) { return std::abs(C.m_X - R) < 1e-3f && std::abs(C.m_Y - G) < 1e-3f && std::abs(C.m_Z - B) < 1e-3f; };

        std::vector<xmath::fvec3> I;
        CHECK(App.RenderFrame(Scene, View, Front) && ReadRGB(I), "Render entity parameters");
        CHECK(Is(At(I, Mid), 1, 0, 0), "C (no entity values) %f %f %f, expected the material's red", At(I, Mid).m_X, At(I, Mid).m_Y, At(I, Mid).m_Z);
        const bool bAIsLeft = Is(At(I, size_v - 1 - Side), 0, 1, 0);
        const int  xA = bAIsLeft ? size_v - 1 - Side : Side, xB = bAIsLeft ? Side : size_v - 1 - Side;
        CHECK(Is(At(I, xA), 0, 1, 0), "A %f %f %f, expected its green", At(I, xA).m_X, At(I, xA).m_Y, At(I, xA).m_Z);
        CHECK(Is(At(I, xB), 0, 0, 1), "B %f %f %f, expected its blue", At(I, xB).m_X, At(I, xB).m_Y, At(I, xB).m_Z);
        int nRed = 0, nGreen = 0, nBlue = 0, nOther = 0;
        for (const auto& C : I) { if (C.m_X == 0 && C.m_Y == 0 && C.m_Z == 0) continue; Is(C, 1, 0, 0) ? ++nRed : Is(C, 0, 1, 0) ? ++nGreen : Is(C, 0, 0, 1) ? ++nBlue : ++nOther; }
        CHECK(nRed > 100 && std::abs(nGreen - nRed) < nRed / 10 && std::abs(nBlue - nRed) < nRed / 10 && nOther == 0, "Covered texels red %d green %d blue %d other %d", nRed, nGreen, nBlue, nOther);

        CHECK(App.RenderFrame(Scene, View, Front) && Renderer.getSlotsWritten() == 0, "Unchanged: %d slots written", Renderer.getSlotsWritten());
        Scene.m_Draws[1].m_EntityParams = Tinted(1, 1, 0);
        CHECK(App.RenderFrame(Scene, View, Front) && ReadRGB(I), "Render after B's tint edit");
        CHECK(Renderer.getSlotsWritten() == 1, "B's tint edit wrote %d slots, expected 1", Renderer.getSlotsWritten());
        CHECK(Is(At(I, xB), 1, 1, 0) && Is(At(I, xA), 0, 1, 0) && Is(At(I, Mid), 1, 0, 0), "After the edit: B yellow, A green, C red");

        Scene.m_Draws[1].m_EntityParams = {};               // the component removed: B reads the material's value again
        CHECK(App.RenderFrame(Scene, View, Front) && ReadRGB(I) && Is(At(I, xB), 1, 0, 0), "B without entity values reads the material's red");
        Renderer.DestroyMaterial(M);
    }

    // The optional COLOR0 stream: a colored and an uncolored copy of the sphere in one frame, the same material reading the vertex color.
    // The colored one decodes its sRGB (255, 128, 0) to linear (1, 0.216, 0); the other has no stream and reads the constant white
    {
        static constexpr std::uint32_t s_Frag[] = {
            #include "xlionrender_test_vertcolor_frag.h"
        };
        xgpu::shader Frag;
        CHECK(!Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_Frag), std::size(s_Frag) } } }), "vertex color shader");
        const auto Slots = std::array<xgpu::texture*, 1>{ nullptr };
        xlionrender::material_id M;
        CHECK(!Renderer.CreateMaterial(M, Frag, Slots), "CreateMaterial reading the vertex color");

        cluster_buffers Colored;
        CHECK(MakeStaticBuffers(Device, Prim, 0.5f, Colored) && Colored.SetColor(Device, Prim.m_Vertices.size(), 0xff0080ffu), "colored static geom buffers");
        xlionrender::mesh_id ColoredId;
        CHECK(!Renderer.CreateClusterGeom(ColoredId, Colored.Geom()), "CreateClusterGeom with colors");

        xgpu::tools::view Front;
        Front.setFov(xmath::radian{ 1.0f });
        Front.LookAt(xmath::fvec3{ 0, 0, -3 }, xmath::fvec3{ 0, 0, 0 });
        Scene.m_Draws =
        { { .m_Mesh = ColoredId, .m_L2W = xmath::fmat4::fromTranslation({ -0.6f, 0, 0 }), .m_Material = Mat, .m_Materials = { M } }
        , { .m_Mesh = Static,    .m_L2W = xmath::fmat4::fromTranslation({  0.6f, 0, 0 }), .m_Material = Mat, .m_Materials = { M } }
        };

        std::vector<std::byte> Raw;
        int W, H;
        CHECK(App.RenderFrame(Scene, View, Front) && !Device.ReadTexture(View.m_HDR, Raw, W, H), "Render vertex colors");
        const auto* p = reinterpret_cast<const std::uint16_t*>(Raw.data());
        int nColored = 0, nWhite = 0, nOther = 0;
        for (std::size_t i = 0; i < std::size_t(W) * H; ++i)
        {
            const float R = HalfToFloat(p[i * 4]), G = HalfToFloat(p[i * 4 + 1]), B = HalfToFloat(p[i * 4 + 2]);
            if (R == 0 && G == 0 && B == 0) continue;
            if      (std::abs(R - 1) < 2e-3f && std::abs(G - 0.2158f) < 2e-3f && B < 2e-3f) ++nColored;
            else if (std::abs(R - 1) < 2e-3f && std::abs(G - 1) < 2e-3f && std::abs(B - 1) < 2e-3f) ++nWhite;
            else ++nOther;
        }
        CHECK(nColored > 100 && std::abs(nColored - nWhite) < nColored / 10 && nOther == 0, "Vertex colors: colored %d white %d other %d", nColored, nWhite, nOther);

        Renderer.DestroyMesh(ColoredId);
        Renderer.DestroyMaterial(M);
        Device.Destroy(std::move(Colored.m_Colors));
    }

    // The optional UV1 stream: a copy of the sphere with a UV1 of (0.25, 0.75) everywhere and one without the stream, in one frame,
    // the material writing UV1 in RG and UV0 in BA: the first reads its stream, the other its UV0
    {
        static constexpr std::uint32_t s_Frag[] = {
            #include "xlionrender_test_uv1_frag.h"
        };
        xgpu::shader Frag;
        CHECK(!Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_Frag), std::size(s_Frag) } } }), "UV1 shader");
        const auto Slots = std::array<xgpu::texture*, 1>{ nullptr };
        xlionrender::material_id M;
        CHECK(!Renderer.CreateMaterial(M, Frag, Slots), "CreateMaterial reading UV1");

        cluster_buffers WithUV1;
        CHECK(MakeStaticBuffers(Device, Prim, 0.5f, WithUV1) && WithUV1.SetUV1(Device, Prim.m_Vertices.size(), 0.25f, 0.75f), "static geom buffers with UV1");
        xlionrender::mesh_id WithUV1Id;
        CHECK(!Renderer.CreateClusterGeom(WithUV1Id, WithUV1.Geom()), "CreateClusterGeom with UV1");

        xgpu::tools::view Front;
        Front.setFov(xmath::radian{ 1.0f });
        Front.LookAt(xmath::fvec3{ 0, 0, -3 }, xmath::fvec3{ 0, 0, 0 });
        Scene.m_Draws =
        { { .m_Mesh = WithUV1Id, .m_L2W = xmath::fmat4::fromTranslation({ -0.6f, 0, 0 }), .m_Material = Mat, .m_Materials = { M } }
        , { .m_Mesh = Static,    .m_L2W = xmath::fmat4::fromTranslation({  0.6f, 0, 0 }), .m_Material = Mat, .m_Materials = { M } }
        };

        std::vector<std::byte> Raw;
        int W, H;
        CHECK(App.RenderFrame(Scene, View, Front) && !Device.ReadTexture(View.m_HDR, Raw, W, H), "Render UV1");
        const auto* p = reinterpret_cast<const std::uint16_t*>(Raw.data());
        int nStream = 0, nUV0 = 0, nOther = 0;
        for (std::size_t i = 0; i < std::size_t(W) * H; ++i)
        {
            const float U1 = HalfToFloat(p[i * 4]), V1 = HalfToFloat(p[i * 4 + 1]), U0 = HalfToFloat(p[i * 4 + 2]), V0 = HalfToFloat(p[i * 4 + 3]);
            if (U1 == 0 && V1 == 0 && U0 == 0 && V0 == 0) continue;
            if      (std::abs(U1 - U0) < 2e-3f && std::abs(V1 - V0) < 2e-3f) ++nUV0;
            else if (std::abs(U1 - 0.25f) < 2e-3f && std::abs(V1 - 0.75f) < 2e-3f) ++nStream;
            else ++nOther;
        }
        CHECK(nStream > 100 && std::abs(nStream - nUV0) < nStream / 10 && nOther == 0, "UV1: from the stream %d, UV0 %d, other %d", nStream, nUV0, nOther);

        Renderer.DestroyMesh(WithUV1Id);
        Renderer.DestroyMaterial(M);
        Device.Destroy(std::move(WithUV1.m_UV1));
    }

    // Stale-id safety does not touch the buffers it references (the caller owns them)
    Renderer.DestroyMesh(Static);
    std::vector<float> C;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, C), "Render after destroy");
    for (float v : C) CHECK(v == 0, "A destroyed static geom drew something");

    // Persistent instances (features/instances): the slot manager on the CPU, then images with instance ids equal to the
    // same draws written every frame (id 0) - first frame, moved, hidden for a frame then back, far from the origin
    {
        xlionrender::instance_slots S;
        S.Init(2);
        const auto M1 = xmath::fmat4::fromTranslation({ 1, 0, 0 }), M2 = xmath::fmat4::fromTranslation({ 2, 0, 0 });
        const auto s1 = S.Acquire(1, M1, 1), s2 = S.Acquire(2, M2, 1);
        CHECK(s1 != s2 && s1 != S.none_v && s2 != S.none_v, "Two slots");
        int nWritten = 0;
        const auto Count = [&](std::uint32_t, const xmath::fmat4&, const xlionrender::material&, const xlionrender::entity_params&) { ++nWritten; };
        CHECK(S.Flush(1, Count) == 1 && S.getPendingCount() == 1, "A budget of 1 writes 1, the other stays pending");
        CHECK(S.Flush(16, Count) == 1 && nWritten == 2, "Then the rest");
        CHECK(S.Acquire(2, M2, 2) == s2 && S.Flush(16, Count) == 0, "An unchanged id keeps its slot, nothing to write");
        CHECK(S.Acquire(2, M1, 2) == s2 && S.Flush(16, Count) == 1, "A moved id is written once");
        CHECK(S.Acquire(2, M1, 2, { .m_BaseColor = { 1, 0, 0 } }) == s2 && S.Flush(16, Count) == 1, "A constant-material edit is written once");
        CHECK(S.Acquire(2, M1, 2, { .m_BaseColor = { 1, 0, 0 } }) == s2 && S.Flush(16, Count) == 0, "The same material again: nothing to write");
        bool bChanged = true;
        CHECK(S.Acquire(2, M1, 2, { .m_BaseColor = { 1, 0, 0 } }, &bChanged, { .m_Mask = 1, .m_Tint = { 0, 1, 0, 1 } }) == s2 && !bChanged && S.getPendingCount() == 1 && S.Flush(16, Count) == 1
             , "An entity-parameter edit is written once, and is not a change for the shadow casters");
        CHECK(S.Acquire(3, M1, 2) == s1, "Full: the least recently seen slot (id 1) is evicted");
        CHECK(S.Acquire(1, M1, 2) == S.none_v, "Full and every slot seen this frame: no slot");
    }
    {
        std::vector<float> Transient, Persistent;
        const auto Compare = [&](const char* pWhat) -> bool
        {
            for (std::size_t i = 0; i < Transient.size(); ++i)
                if (Transient[i] != Persistent[i]) { std::printf("FAIL: %s: texel %d %f vs %f\n", pWhat, int(i), Transient[i], Persistent[i]); return false; }
            return true;
        };
        const auto RenderBoth = [&](const xmath::fmat4& L2W, const char* pWhat) -> bool
        {
            Scene.m_Draws = { { .m_Mesh = Static, .m_L2W = L2W, .m_Material = Mat } };
            if (!App.RenderFrame(Scene, View, Camera) || !ReadHDR(Device, View, Transient)) return false;
            Scene.m_Draws[0].m_Instance = 5;
            if (!App.RenderFrame(Scene, View, Camera) || !ReadHDR(Device, View, Persistent)) return false;
            return Compare(pWhat);
        };
        CHECK(RenderBoth(xmath::fmat4::fromIdentity(), "first frame"), "Instance id: first frame");
        CHECK(RenderBoth(xmath::fmat4::fromTranslation({ 0.1f, 0.05f, 0 }), "moved"), "Instance id: moved");
        Scene.m_Draws.clear();
        CHECK(App.RenderFrame(Scene, View, Camera), "Hidden for a frame");
        CHECK(RenderBoth(xmath::fmat4::fromTranslation({ 0.1f, 0.05f, 0 }), "back"), "Instance id: back after a hidden frame");
        for (int f = 0; f < 4; ++f) CHECK(RenderBoth(xmath::fmat4::fromTranslation({ 0.02f * f, 0, 0 }), "frames"), "Instance id: frame %d", f);

        // Far away: the origin re-snaps (every slot rewritten relative to it)
        const xmath::fvec3 Far{ 5000, 0, -3000 };
        Camera.LookAt(xmath::fvec3{ 0.6f, 0.4f, -1.8f } + Far, Far);
        CHECK(RenderBoth(xmath::fmat4::fromTranslation(Far), "far"), "Instance id: far from the origin");
    }

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
