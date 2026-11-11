// xgeom_skin through the new renderer (the skin vertex path). Metamorphic test: the same sphere
//  (a) in the xgeom_static GPU format, moved by its L2W, and
//  (b) in the xgeom_skin GPU format, moved by its one bone (one influence: weight 1),
// drawn with the same material (the world normal as a color) must give the same image up to quantization. So skinning is applied
// to positions and normals, the skin vertex format decodes, and one material draws both vertex paths.
#include "xlionrender_test_common.h"

using namespace xlionrender::test;

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    const auto Prim = MakeSphere();

    cluster_buffers Rigid, Skin;
    CHECK(MakeStaticBuffers(Device, Prim, 0.5f, Rigid), "static geom buffers");
    CHECK(MakeSkinBuffers  (Device, Prim, 0.5f, Skin),  "skin geom buffers");

    // Bone 0 moves the sphere by Move (and a rotation, so normals have to be skinned too)
    const xmath::fvec3 Move{ 0.15f, -0.1f, 0.05f };
    const xmath::fmat4 BoneMatrix = xmath::fmat4::fromTranslation(Move) * xmath::fmat4::fromRotationY(xmath::radian{ 0.7f });
    xgpu::buffer Bones;
    CHECK(!Device.Create(Bones, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = sizeof(xmath::fmat4), .m_EntryCount = 4 }), "bones");
    (void)Bones.MemoryMap(0, 1, [&](void* p) { std::memcpy(p, &BoneMatrix, sizeof(BoneMatrix)); });

    xlionrender::mesh_id RigidMesh, SkinMesh;
    CHECK(!Renderer.CreateClusterGeom(RigidMesh, Rigid.Geom()), "rigid geom");
    CHECK(!Renderer.CreateClusterGeom(SkinMesh,  Skin.Geom(xlionrender::geom_kind::SKIN, &Bones, 1)), "skin geom");

    static constexpr std::uint32_t s_Frag[] = {
        #include "xlionrender_test_normal_frag.h"
    };
    xgpu::shader             Frag;
    xlionrender::material_id Material;
    CHECK(MakeTestMaterial(Device, Renderer, s_Frag, Frag, Material), "material");

    constexpr int size_v = 128;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");

    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.0f });
    Camera.LookAt(xmath::fvec3{ 0.4f, 0.3f, -1.8f }, xmath::fvec3{ 0, 0, 0 });

    xlionrender::scene Scene;
    std::vector<float> A, B;
    Scene.m_Draws = { { .m_Mesh = RigidMesh, .m_L2W = BoneMatrix, .m_Materials = { Material } } };
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, A), "Render rigid");
    Scene.m_Draws = { { .m_Mesh = SkinMesh, .m_Materials = { Material } } };
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, B), "Render skin");

    int nCovered = 0, nOff = 0;
    double MaxDiff = 0;
    for (std::size_t i = 0; i < A.size(); ++i)
    {
        CHECK(std::isfinite(B[i]), "Non-finite texel %d", int(i));
        if (A[i] == 0 && B[i] == 0) continue;
        ++nCovered;
        if ((A[i] == 0) != (B[i] == 0)) { ++nOff; continue; }
        const double Diff = std::abs(B[i] - A[i]);          // normal * 0.5 + 0.5: a difference of 0.02 is ~2 degrees
        if (Diff > 0.02) ++nOff;
        MaxDiff = std::max(MaxDiff, Diff);
    }
    std::printf("covered %d, differing %d, max difference %.4f\n", nCovered, nOff, MaxDiff);
    CHECK(nCovered > size_v * size_v / 10, "Only %d covered texels", nCovered);
    CHECK(nOff <= nCovered / 50, "%d of %d texels differ beyond quantization", nOff, nCovered);

    // Without its bone the skin sphere is somewhere else: the bones really move it
    const xmath::fmat4 Identity = xmath::fmat4::fromIdentity();
    (void)Bones.MemoryMap(0, 1, [&](void* p) { std::memcpy(p, &Identity, sizeof(Identity)); });
    std::vector<float> C;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, C), "Render skin at rest");
    int nMoved = 0;
    for (std::size_t i = 0; i < C.size(); ++i) nMoved += (C[i] == 0) != (B[i] == 0);
    CHECK(nMoved > nCovered / 10, "Changing the bone moved only %d texels", nMoved);

    // A skin's LOD by its error on the screen (Render): a LOD 1 of half its clusters, error 0.05 - near (about 3 pixels) LOD 0, far (under a pixel) LOD 1
    {
        auto G = Skin.Geom(xlionrender::geom_kind::SKIN, &Bones, 1);
        auto Half = G.m_Submeshes;
        for (auto& S : Half) S.m_nClusters = std::max<std::uint32_t>(1, S.m_nClusters / 2);
        G.m_LODSubmeshes = { G.m_Submeshes, Half };
        G.m_LODErrors    = { std::vector<float>(G.m_Submeshes.size(), 0.0f), std::vector<float>(G.m_Submeshes.size(), 0.05f) };
        G.m_LODIndexEnd.clear();                                // (no residency: every LOD resident)
        xlionrender::mesh_id LODMesh;
        CHECK(!Renderer.CreateClusterGeom(LODMesh, G), "skin geom with LODs");
        const auto Lods = [&](float Z, int& n0, int& n1)
        {
            Scene.m_Draws = { { .m_Mesh = LODMesh, .m_L2W = xmath::fmat4::fromTranslation({ 0, 0, Z }), .m_Materials = { Material } } };
            const bool bOk = App.RenderFrame(Scene, View, Camera);
            n0 = Renderer.getSkinLODReport().m_nSubmeshes[0]; n1 = Renderer.getSkinLODReport().m_nSubmeshes[1];
            return bOk;
        };
        int Near0, Near1, Far0, Far1;
        CHECK(Lods(0.0f, Near0, Near1) && Lods(25.0f, Far0, Far1), "render the skin with LODs");
        std::printf("skin LODs: near LOD 0 %d, LOD 1 %d; far LOD 0 %d, LOD 1 %d\n", Near0, Near1, Far0, Far1);
        CHECK(Near0 > 0 && Near1 == 0, "near, the skin drew LOD 1 (%d submeshes) - its error is 3 pixels there", Near1);
        CHECK(Far1 > 0 && Far0 == 0, "far, the skin drew LOD 0 (%d submeshes) - its error under a pixel there", Far0);
        Renderer.DestroyMesh(LODMesh);
    }

    // A static caster whose resident LOD changes (its buffers gain LOD 0: cluster_geom::Refresh) dirties its cached sun pages: pages drawn while only
    // its coarse LOD was resident kept it - sunlit holes in a building's shadow, Bistro Game 2026-11-07
    {
        auto G = Rigid.Geom();
        auto Half = G.m_Submeshes;
        for (auto& S : Half) S.m_nClusters = std::max<std::uint32_t>(1, S.m_nClusters / 2);
        G.m_LODSubmeshes = { G.m_Submeshes, Half };
        G.m_LODErrors    = { std::vector<float>(G.m_Submeshes.size(), 0.0f), std::vector<float>(G.m_Submeshes.size(), 0.05f) };
        G.m_LODIndexEnd.clear();
        xlionrender::mesh_id Caster;
        CHECK(!Renderer.CreateClusterGeom(Caster, G), "a static cluster caster with two LODs");
        CHECK(Renderer.setMeshResidentLOD(Caster, 1), "its LOD 1 resident only (as loading coarse first)");
        Scene.m_Draws = { { .m_Mesh = Caster, .m_L2W = xmath::fmat4::fromIdentity(), .m_Materials = { Material } } };
        for (int f = 0; f < 40; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "settle %d", f);
        int Before = 0;
        for (int f = 0; f < 4; ++f) { CHECK(App.RenderFrame(Scene, View, Camera), "settled frame %d", f); Before += Renderer.getVSMReport().m_Pages.m_Dirtied; }
        CHECK(Renderer.setMeshResidentLOD(Caster, 0), "LOD 0 arrives");
        int After = 0;
        for (int f = 0; f < 4; ++f) { CHECK(App.RenderFrame(Scene, View, Camera), "after LOD 0 frame %d", f); After += Renderer.getVSMReport().m_Pages.m_Dirtied; }
        std::printf("residency: sun pages dirtied settled %d, after LOD 0 arrived %d\n", Before, After);
        CHECK(Before == 0 && After > 0, "LOD 0 arriving did not dirty the caster's cached sun pages (settled %d, after %d)", Before, After);
        Renderer.DestroyMesh(Caster);
    }

    // A masked skinned mesh goes through its material's cut in the depth prepass, as a rigid one (Ren's hair, 2026-11-11: the plain skinned prepass has no alpha
    // test and culls back faces, so the depth it laid was wrong). Cut out everywhere and in front of a sphere, single and double sided: the image is the sphere alone
    {
        static constexpr std::uint32_t s_CutFrag[] = {
            #include "xlionrender_test_cutout_frag.h"
        };
        xgpu::shader CutFrag;
        CHECK(!Device.Create(CutFrag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(s_CutFrag), std::size(s_CutFrag) } } }), "cut out shader");
        const auto Slots = std::array<xgpu::texture*, 1>{ nullptr };
        const xmath::fmat4 Behind = xmath::fmat4::fromTranslation(xmath::fvec3{ -0.25f, -0.2f, 1.0f });      // (further along the view than the skinned sphere, overlapping it on the screen)
        Scene.m_Sun = { .m_Direction = { -0.4f, -1.0f, 0.5f }, .m_Intensity = 3, .m_bCastShadows = false };      // (lit: the prepass's cut needs the lighting block)
        const auto Compare = [&](bool bDoubleSided) -> int
        {
            xlionrender::material_id Cut;
            CHECK(!Renderer.CreateMaterial(Cut, CutFrag, Slots, { .m_Blend = xlionrender::material_state::blend::MASKED, .m_bDoubleSided = bDoubleSided }), "cut out material");
            std::vector<float> Alone, Masked;
            Scene.m_Draws = { { .m_Mesh = RigidMesh, .m_L2W = Behind, .m_Materials = { Material } } };
            for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "the sphere alone %d", f);
            CHECK(ReadHDR(Device, View, Alone), "read the sphere alone");
            Scene.m_Draws = { { .m_Mesh = RigidMesh, .m_L2W = Behind, .m_Materials = { Material } }, { .m_Mesh = SkinMesh, .m_Materials = { Cut } } };
            for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "behind the cut out skin %d", f);
            CHECK(ReadHDR(Device, View, Masked), "read behind the cut out skin");
            // (the same with the skinned sphere solid: it must cover part of the one behind, or this proves nothing)
            std::vector<float> Solid;
            Scene.m_Draws[1].m_Materials = { Material };
            for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "behind the solid skin %d", f);
            CHECK(ReadHDR(Device, View, Solid), "read behind the solid skin");
            int nHidden = 0;
            for (std::size_t i = 0; i < Alone.size(); ++i) nHidden += std::abs(Alone[i] - Solid[i]) > 0.02f;
            CHECK(nHidden > 100, "the skinned sphere does not overlap the one behind (%d values)", nHidden);
            int nDiff = 0, nCoveredBehind = 0;
            for (std::size_t i = 0; i < Alone.size(); ++i) { nDiff += std::abs(Alone[i] - Masked[i]) > 0.02f; nCoveredBehind += Alone[i] != 0.0f; }
            std::printf("masked skin (%s sided): %d of %d values differ from the sphere alone (%d covered)\n", bDoubleSided ? "double" : "single", nDiff, int(Alone.size()), nCoveredBehind);
            CHECK(nCoveredBehind > 100, "the sphere behind is not on the screen");
            CHECK(nDiff <= int(Alone.size() / 2000), "a skinned mesh cut out everywhere hides or changes what is behind it: %d values", nDiff);
            Scene.m_Draws.clear();
            Renderer.DestroyMaterial(Cut);
            return 0;
        };
        // A masked skinned mesh that draws hides what is behind it, whichever is drawn first and on either side (Ren's hair: the characters behind showed through)
        const auto Occludes = [&](bool bDoubleSided, bool bSkinFirst) -> int
        {
            xlionrender::material_id Mask;
            CHECK(!Renderer.CreateMaterial(Mask, Frag, Slots, { .m_Blend = xlionrender::material_state::blend::MASKED, .m_bDoubleSided = bDoubleSided }), "masked material");
            const xlionrender::draw Far{ .m_Mesh = RigidMesh, .m_L2W = Behind, .m_Materials = { Material } };
            std::vector<float> Want, Got;
            const auto Render = [&](xlionrender::material_id M, std::vector<float>& Out) -> int
            {
                const xlionrender::draw Near{ .m_Mesh = SkinMesh, .m_Materials = { M } };
                if( bSkinFirst ) Scene.m_Draws = { Near, Far }; else Scene.m_Draws = { Far, Near };
                for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, Camera), "frame %d", f);
                CHECK(ReadHDR(Device, View, Out), "read");
                return 0;
            };
            CHECK(Render(Material, Want) == 0, "solid");
            CHECK(Render(Mask, Got) == 0, "masked");
            int nDiff = 0;
            for (std::size_t i = 0; i < Want.size(); ++i) nDiff += std::abs(Want[i] - Got[i]) > 0.02f;
            std::printf("masked skin in front (%s sided, %s first): %d of %d values differ from solid\n", bDoubleSided ? "double" : "single", bSkinFirst ? "skin" : "rigid", nDiff, int(Want.size()));
            CHECK(nDiff <= int(Want.size() / 500), "a masked skinned mesh does not hide what is behind it: %d values", nDiff);
            Scene.m_Draws.clear();
            Renderer.DestroyMaterial(Mask);
            return 0;
        };
        for (const bool b2S : { false, true }) for (const bool bFirst : { false, true }) CHECK(Occludes(b2S, bFirst) == 0, "occlusion");
        CHECK(Compare(false) == 0, "single sided");
        CHECK(Compare(true) == 0, "double sided");
    }

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
