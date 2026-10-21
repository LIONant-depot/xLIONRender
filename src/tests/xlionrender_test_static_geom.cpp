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
        CHECK(!Renderer.CreateMaterial(MA, Frag, Slots, false, First) && !Renderer.CreateMaterial(MB, Frag, Slots, false, Second), "CreateMaterial with parameters");

        Scene.m_Draws = { { .m_Mesh = Static, .m_Material = Mat, .m_Materials = { MB } } };
        std::vector<float> P;
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, P), "Render parameters");
        int nParams = 0;
        for (float v : P) if (v != 0) { CHECK(std::abs(v - 2.0f) < 1e-3f, "Parameter texel %f, expected 2 (the second block's)", v); ++nParams; }
        CHECK(nParams == nCovered, "Parameters covered %d texels, the standard path %d", nParams, nCovered);
        Renderer.DestroyMaterial(MA);
        Renderer.DestroyMaterial(MB);
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
        CHECK(!Renderer.CreateMaterial(M, Frag, Slots, false, Block), "CreateMaterial with an entity tint");

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
