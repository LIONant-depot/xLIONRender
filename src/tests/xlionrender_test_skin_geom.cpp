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

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
