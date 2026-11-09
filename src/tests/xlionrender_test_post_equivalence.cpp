// The post chain's equivalence harness (render graph P1c; xLION_Docs/render_graph_design_v3_2026-11-09.md 11.4): two views of the same
// scene and camera, each with its own TAA / exposure / bloom state, view B's chain from view A's lit image (m_pPostInputs), the auto
// exposure's step fixed (m_PostFixedDt) so the two adapt alike. Every frame the TAA images, the exposure, every bloom level and the
// output are read back from both and must be equal, bit for bit; the first that differs is named. (P1c ran it hand-written steps
// against the plan - identical, 5ec9dc2 - before the steps were removed.) First the chain against itself: if two views of it differ,
// the harness is not trusted (the inputs are not the same).
// Then the same with view B's description read from its text form (P2: same output).
// Cases: TAA, bloom, auto exposure on and off; an odd size (805 x 377) and the inventory's 3218 x 1512; a moving camera (reprojection).
//   xlionrender_test_post_equivalence [frames]       (XLION_PE_ONLY=<case index>: one case)
#include "xlionrender_test_common.h"
#include "xlionrender_post_graph_text.h"
#include <filesystem>
#include <cstring>

using namespace xlionrender::test;

struct state { std::vector<std::byte> m_Images[1 + 3 + 6 + 1]; };   // the input (lit image; the depth is not read back - xGPU reads colour only), TAA 0, 1, exposure, the 6 bloom levels, the output
static const char* s_Names[] = { "input HDR", "TAA#0", "TAA#1", "exposure", "bloom#0", "bloom#1", "bloom#2", "bloom#3", "bloom#4", "bloom#5", "output" };

static bool Read( xgpu::device& Device, const xlionrender::render_view& V, state& S )
{
    int W, H;
    const xgpu::texture* T[] = { &V.m_HDR, &V.m_TAA[0], &V.m_TAA[1], &V.m_ExposureTex, &V.m_Bloom[0], &V.m_Bloom[1], &V.m_Bloom[2], &V.m_Bloom[3], &V.m_Bloom[4], &V.m_Bloom[5], &V.m_Output };
    for( int i = 0; i < 11; ++i )
    {
        S.m_Images[i].clear();
        if( !T[i]->m_Private ) continue;        // (TAA never on: not made)
        if( Device.ReadTexture(*T[i], S.m_Images[i], W, H) ) return false;
    }
    return true;
}

// The first image that differs ("" when none), with how many bytes and where the first is
static std::string Compare( const state& A, const state& B )
{
    for( int i = 0; i < 11; ++i )
    {
        const auto& a = A.m_Images[i]; const auto& b = B.m_Images[i];
        if( a.size() != b.size() ) return std::format("{}: {} bytes vs {}", s_Names[i], a.size(), b.size());
        if( a.empty() || !std::memcmp(a.data(), b.data(), a.size()) ) continue;
        std::size_t n = 0, First = 0;
        for( std::size_t k = a.size(); k-- > 0; ) if( a[k] != b[k] ) { ++n; First = k; }
        return std::format("{}: {} of {} bytes differ, the first at byte {}", s_Names[i], n, a.size(), First);
    }
    return {};
}

int main( int argc, char** argv )
{
    const int nFrames = argc > 1 ? std::atoi(argv[1]) : 6;
    app App;
    CHECK(App.Init(), "Init");
    App.m_bQuietTimes = true;
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    xlionrender::mesh_id Sphere;
    {
        const auto Prim = MakeSphere();
        std::vector<xlionrender::vertex> Verts(Prim.m_Vertices.size());
        for( std::size_t i = 0; i < Verts.size(); ++i )
        {
            const auto& V = Prim.m_Vertices[i];
            Verts[i] = { { V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }, { V.m_Texcoord.m_X, V.m_Texcoord.m_Y }, 0xffffffffu
                       , { V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }, { V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, V.m_Tangent.m_D } };
        }
        CHECK(!Renderer.CreateMesh(Sphere, Verts, Prim.m_Indices), "CreateMesh");
    }
    xlionrender::scene Scene;
    Scene.m_Draws.push_back({ .m_Mesh = Sphere, .m_Material = { .m_BaseColor = { 0.8f, 0.8f, 0.8f }, .m_Roughness = 0.3f } });
    Scene.m_Sun.m_Direction = { 0.3f, -1, 0.5f };
    Scene.m_Sun.m_Intensity = 30;                       // (bright: the bloom's threshold passed on the lit side)
    Scene.m_Ambient         = { 0.05f, 0.05f, 0.08f };
    Scene.m_Post.m_BloomThreshold = 0.6f;

    struct test_case { const char* m_pName; bool m_bTAA, m_bBloom, m_bAuto; int m_W, m_H; };
    const test_case Cases[] = {
        { "all on, 805 x 377",          true,  true,  true,  805, 377 },
        { "TAA off",                    false, true,  true,  805, 377 },
        { "bloom off",                  true,  false, true,  805, 377 },
        { "manual exposure",            true,  true,  false, 805, 377 },
        { "all off",                    false, false, false, 805, 377 },
        { "all on, 3218 x 1512",        true,  true,  true,  3218, 1512 },
        { "all on, 64 x 64",            true,  true,  true,  64, 64 },
    };
    // (P2) The description as text: the default written to a file and read back - view B's chain from it
    xlionrender::post::desc FromFile;
    {
        const auto Path = (std::filesystem::temp_directory_path() / "xlionrender_test_post_equivalence.txt").wstring();
        auto D = xlionrender::post::desc::Default(); std::string Problem;
        { xtextfile::stream F; CHECK(!F.Open(false, Path, xtextfile::file_type::TEXT) && !xlionrender::post::Serialize(F, D, Problem), "the description written"); }
        { xtextfile::stream F; CHECK(!F.Open(true, Path, xtextfile::file_type::TEXT) && !xlionrender::post::Serialize(F, FromFile, Problem) && Problem.empty(), "the description read"); }
        std::filesystem::remove(Path);
    }
    const char* pOnly = xlionrender::Env("XLION_PE_ONLY");
    int nFail = 0;
    for( int Control = 1; Control >= 0; --Control )         // 1: hand-written against itself (the harness's own check), 0: against the plan
    for( int c = 0; c < int(std::size(Cases)); ++c )
    {
        if( pOnly && std::atoi(pOnly) != c ) continue;
        const auto& C = Cases[c];
        Scene.m_Post.m_bTAA = C.m_bTAA; Scene.m_Post.m_Bloom = C.m_bBloom ? 0.4f : 0.0f; Scene.m_Post.m_bAutoExposure = C.m_bAuto;
        xlionrender::render_view A, B;
        CHECK(!Renderer.CreateView(A, { .m_Width = C.m_W, .m_Height = C.m_H }) && !Renderer.CreateView(B, { .m_Width = C.m_W, .m_Height = C.m_H }), "CreateView");
        A.m_PostFixedDt = B.m_PostFixedDt = 1.0f / 30.0f;
        B.m_pPostInputs = &A;
        if( !Control ) B.m_PostDesc = FromFile;       // (the scene passes are not the same twice - frame-indexed noise: B's chain starts from A's lit image)
        std::string Why;
        int Frame = 0;
        for( ; Frame < nFrames && Why.empty(); ++Frame )
        {
            xgpu::tools::view Camera;
            Camera.setFov(xmath::radian{ 1.0f });
            Camera.setViewport({ 0, 0, C.m_W, C.m_H });
            Camera.LookAt(xmath::fvec3{ 0.05f * float(Frame), 0.1f, -2.0f }, xmath::fvec3{ 0, 0, 0 });      // (moving: the TAA reprojects)
            state SA, SB;
            CHECK(App.RenderFrame(Scene, A, Camera) && Read(Device, A, SA), "Render A");
            CHECK(App.RenderFrame(Scene, B, Camera) && Read(Device, B, SB), "Render B");
            Why = Compare(SA, SB);
        }
        const auto What = std::format("{} {}", Control ? "the chain against itself" : "the chain against its description from a file", C.m_pName);
        if( Why.empty() ) std::printf("same: %s, %d frames\n", What.c_str(), nFrames);
        else { std::printf("DIFFERENT: %s, frame %d - %s\n", What.c_str(), Frame - 1, Why.c_str()); ++nFail; }
        Renderer.DestroyView(A); Renderer.DestroyView(B);
        if( Control && !Why.empty() ) { std::printf("FAIL: the harness is not trusted (the chain differs from itself)\n"); return 1; }
    }
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    if( nFail ) { std::printf("FAIL: %d cases differ\n", nFail); return 1; }
    std::printf("PASS\n");
    return 0;
}
