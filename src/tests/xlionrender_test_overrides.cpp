// Material overrides (scene/xlionrender_material_overrides.h, the Mesh's "Override materials"): an entity slot's parameters replaced on its
// material instance. The instance here: the texture-table test material (xlionrender_test_table_body.glsl: HDR = (slot 0's red * P.x + slot 1's
// red * P.y) * 3), its block P = (1, 0), slot 0 red 1.0, slot 1 red 0.5. Its parameters: 100 = P.x, 101 = P.y, 200 / 201 = the textures.
//  - override: an entity with P = (0, 1) draws slot 1 (1.5)
//  - dedup: a second entity with the same overrides shares its material (one override material, one pipeline)
//  - copy on write: the first one's P.y changed while shared: a new material for it, the other keeps 1.5
//  - in place: changed again while it is the only user: the same material, its block rewritten
//  - texture swap: slot 1's texture overridden in place (bindless: an id; else a new set), no pipeline made
//  - released: no override material left
//  - one state: 64 draws on 64 override materials of one pipeline bind it once (they share it; only the texture table is theirs)
//  - many: 9000 override materials, more blocks than a frame's ring region (their copies spread over two frames): the last one draws its value
#include "xlionrender_test_common.h"
#include "scene/xlionrender_material_overrides.h"

using namespace xlionrender::test;

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    static constexpr std::uint32_t s_Table[]         = {
        #include "xlionrender_test_table_frag.h"
    };
    static constexpr std::uint32_t s_TableBindless[] = {
        #include "xlionrender_test_table_bindless_frag.h"
    };
    xgpu::shader Frag;
    const auto Words = Device.isBindless() ? std::span<const std::uint32_t>{ s_TableBindless } : std::span<const std::uint32_t>{ s_Table };
    CHECK(!Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(Words.data()), Words.size() } } }), "table shader");

    const std::uint32_t Texels[3] = { 0xff0000ffu, 0xff000080u, 0xff000040u };      // red 1, ~0.5, ~0.25
    std::array<xgpu::texture, 3> T;
    for (int i = 0; i < 3; ++i)
    {
        const auto Mips = std::array{ xgpu::texture::setup::mip_size{ sizeof(Texels[i]) } };
        CHECK(!Device.Create(T[i], { .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = 1, .m_Height = 1, .m_MipSizes = Mips
                                   , .m_AllFacesData = { reinterpret_cast<const std::byte*>(&Texels[i]), sizeof(Texels[i]) }, .m_isGamma = false }), "texture %d", i);
    }
    const auto Red = [&](int i) { return float(Texels[i] & 0xff) / 255.0f; };

    using mp = xlionrender::material_param;
    const std::array Params{ mp{ .m_ID = 100, .m_Type = mp::type::FLOAT, .m_ByteOffset = 0 }, mp{ .m_ID = 101, .m_Type = mp::type::FLOAT, .m_ByteOffset = 4 }
                           , mp{ .m_ID = 200, .m_Type = mp::type::TEXTURE, .m_Slot = 0 },   mp{ .m_ID = 201, .m_Type = mp::type::TEXTURE, .m_Slot = 1 } };
    const std::array<xgpu::texture*, 3> Textures{ nullptr, &T[0], &T[1] };
    const float Block[] = { 1, 0, 0, 0 };
    const xlionrender::material_source Source{ .m_Key = 7, .m_pFragment = &Frag, .m_Textures = Textures, .m_Block = Block, .m_bTextureTable = true, .m_Params = Params };

    cluster_buffers Buffers;
    CHECK(MakeStaticBuffers(Device, MakeSphere(), 0.5f, Buffers), "static geom buffers");
    xlionrender::mesh_id Static;
    CHECK(!Renderer.CreateClusterGeom(Static, Buffers.Geom()), "CreateClusterGeom");
    constexpr int size_v = 64;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");
    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.0f });
    Camera.LookAt(xmath::fvec3{ 0, 0, -1.6f }, xmath::fvec3{ 0, 0, 0 });
    std::vector<float> M;
    const auto Expect = [&](xlionrender::material_id Id, float Value, const char* pWhat) -> bool
    {
        xlionrender::scene Scene;
        Scene.m_Draws = { { .m_Mesh = Static, .m_Materials = { Id } } };
        if (!App.RenderFrame(Scene, View, Camera) || !ReadHDR(Device, View, M)) return false;
        int n = 0, nOff = 0;
        for (float v : M) if (v != 0) { ++n; nOff += std::abs(v - Value) > 0.01f; }
        std::printf("%s: %d texels, %d off %.3f\n", pWhat, n, nOff, Value);
        return n > size_v * size_v / 10 && nOff == 0;
    };
    const auto Over = [](std::uint64_t ID, float V) { return xlionrender::param_override{ .m_ID = ID, .m_V = { V, 0, 0, 0 } }; };

    xlionrender::material_overrides Overrides;
    xlionrender::material_overrides::slot A, B;
    const int Pipelines0 = Renderer.getPipelineCount(), Materials0 = Renderer.getMaterialCount();

    // Override and dedup
    const std::array PA{ Over(100, 0), Over(101, 1) };
    const auto IdA = Overrides.Acquire(Renderer, Source, PA, A);
    const auto IdB = Overrides.Acquire(Renderer, Source, PA, B);
    CHECK(IdA.isValid() && IdA == IdB && Overrides.Count() == 1, "Two slots with the same overrides share one material (%d made)", Overrides.Count());
    CHECK(Renderer.getPipelineCount() == Pipelines0 + 1, "One pipeline for the override material (%d)", Renderer.getPipelineCount() - Pipelines0);
    CHECK(Expect(IdA, 3 * Red(1), "override"), "The override draws slot 1");

    // Copy on write: A changes while B shares it
    const std::array PA2{ Over(100, 0), Over(101, 0.5f) };
    const auto IdA2 = Overrides.Acquire(Renderer, Source, PA2, A);
    CHECK(IdA2.isValid() && !(IdA2 == IdB) && Overrides.Count() == 2, "Copy on write: A has its own material now (%d entries)", Overrides.Count());
    CHECK(Expect(IdA2, 3 * 0.5f * Red(1), "copy on write: A") && Expect(IdB, 3 * Red(1), "copy on write: B"), "A draws its new value, B its old one");

    // In place: A changes again, alone on its material
    const std::array PA3{ Over(100, 0), Over(101, 0.25f) };
    const auto IdA3 = Overrides.Acquire(Renderer, Source, PA3, A);
    CHECK(IdA3 == IdA2 && Overrides.m_nInPlace == 1, "In place: A keeps its material (%d in place)", Overrides.m_nInPlace);
    CHECK(Expect(IdA3, 3 * 0.25f * Red(1), "in place"), "A draws the value changed in place");

    // Texture swap: slot 1's texture overridden, in place
    auto PA4 = std::array{ Over(100, 0), Over(101, 1), xlionrender::param_override{ .m_ID = 201, .m_pTexture = &T[2], .m_TextureKey = 3 } };
    const auto IdA4 = Overrides.Acquire(Renderer, Source, PA4, A);
    const int nCull = std::getenv("XLION_NO_GPU_CULL") || !Device.hasDrawIndirectCount() ? 0 : 1;      // (drawn: the records path's pipeline as well - GPU culling)
    CHECK(IdA4 == IdA2 && Overrides.m_nInPlace == 2 && Renderer.getPipelineCount() == Pipelines0 + 1 + nCull, "Texture swap in place, no pipeline made");
    CHECK(Expect(IdA4, 3 * Red(2), "texture swap") && Expect(IdB, 3 * Red(1), "texture swap: B"), "A draws the swapped texture, B its own");

    // A back on B's overrides: shares B's material again, its own one goes
    CHECK(Overrides.Acquire(Renderer, Source, PA, A) == IdB && Overrides.Count() == 1, "Back to the shared overrides: one material (%d)", Overrides.Count());

    Overrides.Release(Renderer, A);
    Overrides.Release(Renderer, B);

    // One state: draws on override materials of one pipeline need one bind (not one per material)
    {
        std::vector<xlionrender::material_overrides::slot> S(64);
        xlionrender::scene Scene;
        for (int i = 0; i < 64; ++i)
        {
            const std::array O{ Over(100, 0), Over(101, 1.0f + float(i)) };
            Scene.m_Draws.push_back({ .m_Mesh = Static, .m_Materials = { Overrides.Acquire(Renderer, Source, O, S[i]) } });
        }
        CHECK(Overrides.Count() == 64 && App.RenderFrame(Scene, View, Camera), "64 override materials drawn");
        CHECK(Renderer.getSceneBinds() == 1, "64 override materials of one pipeline: %d scene binds (1 expected)", Renderer.getSceneBinds());
        for (auto& X : S) Overrides.Release(Renderer, X);
    }

    // Many: more parameter blocks than a frame's ring region holds - the rest is copied the frame after, in order
    {
        constexpr int n_v = 9000;       // 2 vec4s each: 18000 > instance::param_ring_v
        std::vector<xlionrender::material_overrides::slot> S(n_v);
        xlionrender::material_id Last;
        for (int i = 0; i < n_v; ++i)
        {
            const std::array O{ Over(100, 0), Over(101, 2.0f + 1e-3f * float(i)) };     // (each its own: no sharing)
            Last = Overrides.Acquire(Renderer, Source, O, S[i]);
            CHECK(Last.isValid(), "override material %d of %d", i, n_v);
        }
        xlionrender::scene Scene;
        Scene.m_Draws = { { .m_Mesh = Static, .m_Materials = { Last } } };
        CHECK(App.RenderFrame(Scene, View, Camera), "frame");
        CHECK(Overrides.Count() == n_v, "%d override materials (%d made)", n_v, Overrides.Count());
        CHECK(Expect(Last, 3 * Red(1) * (2.0f + 1e-3f * float(n_v - 1)), "many: the last one"), "The last of %d override materials draws its own value", n_v);
        for (auto& X : S) Overrides.Release(Renderer, X);
    }
    CHECK(Overrides.Count() == 0 && Renderer.getMaterialCount() == Materials0 && Renderer.getPipelineCount() == Pipelines0, "Released: %d entries, %d materials, %d pipelines left over"
         , Overrides.Count(), Renderer.getMaterialCount() - Materials0, Renderer.getPipelineCount() - Pipelines0);

    Renderer.DestroyView(View);
    for (auto& X : T) Device.Destroy(std::move(X));
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
