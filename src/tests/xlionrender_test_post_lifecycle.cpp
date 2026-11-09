// The post chain's lifecycle under stress (render graph P4; xLION_Docs/render_graph_p4_note_2026-11-09.md): what a view keeps and what
// it starts again when it is resized, its description or settings change, the camera cuts, or two views with different chains render
// in one frame. Checked on read-back state (the adapted exposure, the TAA images) and the Vulkan validation count.
#include "xlionrender_test_common.h"
#include <cmath>

using namespace xlionrender::test;

// (exits: usable in helpers and lambdas, where CHECK's return 1 is not)
#define MUST(COND, ...) do { if (!(COND)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); std::exit(1); } } while(0)

static float ReadExposure( xgpu::device& Device, const xlionrender::render_view& V )
{
    std::vector<std::byte> Raw; int W, H;
    MUST(!Device.ReadTexture(V.m_ExposureTex, Raw, W, H) && Raw.size() >= 2, "read the exposure");
    return HalfToFloat(*reinterpret_cast<const std::uint16_t*>(Raw.data()));
}
static std::vector<std::uint32_t> ReadOutput( xgpu::device& Device, const xlionrender::render_view& V )
{
    std::vector<std::uint32_t> LDR; int W, H;
    MUST(!Device.ReadTexture(V.m_Output, LDR, W, H), "read the output");
    return LDR;
}
static double MeanDiff( const std::vector<std::uint32_t>& A, const std::vector<std::uint32_t>& B )
{
    double S = 0;
    for( std::size_t i = 0; i < A.size(); ++i ) for( int c = 0; c < 24; c += 8 ) S += std::abs(int((A[i] >> c) & 0xff) - int((B[i] >> c) & 0xff));
    return S / double(A.size() * 3);
}

int main()
{
    app App;
    MUST(App.Init(), "Init");
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
        MUST(!Renderer.CreateMesh(Sphere, Verts, Prim.m_Indices), "CreateMesh");
    }
    xlionrender::scene Scene;
    Scene.m_Draws.push_back({ .m_Mesh = Sphere, .m_Material = { .m_BaseColor = { 0.8f, 0.8f, 0.8f }, .m_Roughness = 0.3f } });
    Scene.m_Sun.m_Direction = { 0.3f, -1, 0.5f };
    Scene.m_Sun.m_Intensity = 30;
    Scene.m_Ambient         = { 0.05f, 0.05f, 0.08f };
    Scene.m_Post.m_bAutoExposure = true; Scene.m_Post.m_bTAA = true; Scene.m_Post.m_Bloom = 0.4f; Scene.m_Post.m_BloomThreshold = 0.6f;

    const auto Cam = [](int W, int H, float X)
    {
        xgpu::tools::view C; C.setFov(xmath::radian{ 1.0f }); C.setViewport({ 0, 0, W, H });
        C.LookAt(xmath::fvec3{ X * 0.3f, 0.1f, -0.9f }, xmath::fvec3{ X * 0.3f, 0, 0 });      // (close: the sphere fills most of the view - the metering sees light, not the empty background)
        return C;
    };
    const auto Frame = [&](xlionrender::render_view& V, float X = 0) { auto C = Cam(V.m_Width, V.m_Height, X); MUST(App.RenderFrame(Scene, V, C), "Render"); };

    // 1. The first frame: the exposure snaps to its target (nothing to adapt from)
    xlionrender::render_view V;
    MUST(!Renderer.CreateView(V, { .m_Width = 320, .m_Height = 180 }), "CreateView");
    V.m_PostFixedDt = 0.0f;         // (no time passes: the exposure only snaps, or stays)
    Frame(V);
    const float E1 = ReadExposure(Device, V);
    MUST(E1 > 0.0f, "first frame: the exposure snapped (%f)", E1);
    std::printf("first frame: exposure %f\n", E1);

    // 2. A resize keeps the adapted exposure: the light changes with no time passing (the target moves, the value stays), then a resize
    Scene.m_Sun.m_Intensity = 3;
    Frame(V);
    MUST(ReadExposure(Device, V) == E1, "no time passed: the exposure stays");
    MUST(!Renderer.CreateView(V, { .m_Width = 400, .m_Height = 240 }), "resize");
    Frame(V);
    const float E2 = ReadExposure(Device, V);
    MUST(E2 == E1, "resize: the exposure kept (%f, was %f; a reset would have snapped to the new light)", E2, E1);
    std::printf("resize: exposure kept %f\n", E2);
    // ...and the snap a new view makes, for the negative control: the new light's own target
    {
        xlionrender::render_view Fresh;
        MUST(!Renderer.CreateView(Fresh, { .m_Width = 400, .m_Height = 240 }), "CreateView fresh");
        Fresh.m_PostFixedDt = 0.0f; Frame(Fresh);
        const float EF = ReadExposure(Device, Fresh);
        MUST(std::abs(EF - E1) > 0.1f * E1, "control: a fresh view snaps to a different exposure (%f vs %f) - else case 2 proves nothing", EF, E1);
        std::printf("control: a fresh view snaps to %f\n", EF);
        Renderer.DestroyView(Fresh);
    }
    Scene.m_Sun.m_Intensity = 30;
    V.m_PostFixedDt = 1.0f / 30.0f;

    // 3. The description changes: TAA out (its history is no longer valid), back in (valid after a frame); the bloom's images untouched
    const auto* pBloom0 = V.m_Bloom[0].m_Private.get();
    Frame(V);
    MUST(V.m_bTAAHistory, "TAA on: a history");
    V.m_PostDesc.m_Nodes.erase(V.m_PostDesc.m_Nodes.begin());
    Frame(V);
    MUST(!V.m_bTAAHistory, "TAA out of the description: no history");
    V.m_PostDesc = xlionrender::post::desc::Default();
    Frame(V);
    MUST(V.m_bTAAHistory, "TAA back: a history again");
    // 4. A setting slides: bloom 0.4 -> 0 -> 0.4 - nothing remade
    Scene.m_Post.m_Bloom = 0.0f; Frame(V);
    Scene.m_Post.m_Bloom = 0.4f; Frame(V);
    MUST(V.m_Bloom[0].m_Private.get() == pBloom0, "the bloom's images are the same objects through description and setting changes");

    // 5. A camera cut: the TAA output against the same frame without TAA (a second view, the same lit image), before and after the cut
    {
        xlionrender::render_view NoTAA;
        MUST(!Renderer.CreateView(NoTAA, { .m_Width = V.m_Width, .m_Height = V.m_Height }), "CreateView no TAA");
        NoTAA.m_PostDesc.m_Nodes.erase(NoTAA.m_PostDesc.m_Nodes.begin());
        NoTAA.m_pPostInputs = &V; NoTAA.m_PostFixedDt = V.m_PostFixedDt;
        const auto Both = [&](float X) { Frame(V, X); Frame(NoTAA, X); return MeanDiff(ReadOutput(Device, V), ReadOutput(Device, NoTAA)); };
        double Steady = 0; for( int i = 0; i < 8; ++i ) Steady = Both(0.0f);
        const double Cut = Both(0.6f);
        double After = Cut; for( int i = 0; i < 8; ++i ) After = Both(0.6f);
        std::printf("camera cut: TAA vs none, mean |diff| per channel (0-255): steady %.3f, the cut's frame %.3f, 8 frames later %.3f\n", Steady, Cut, After);
        MUST(After <= Steady * 2.0 + 0.5, "8 frames after a cut the TAA image is back near its steady difference (%.3f vs %.3f)", After, Steady);
        Renderer.DestroyView(NoTAA);
    }

    // 6. Two views, different chains, one frame each: both valid (no validation errors), and they differ where the bloom is
    {
        xlionrender::render_view W2;
        MUST(!Renderer.CreateView(W2, { .m_Width = V.m_Width, .m_Height = V.m_Height }), "CreateView second");
        W2.m_PostDesc.m_Nodes.erase(W2.m_PostDesc.m_Nodes.begin() + 1);      // (no bloom)
        W2.m_pPostInputs = &V; W2.m_PostFixedDt = V.m_PostFixedDt;
        for( int i = 0; i < 3; ++i ) { Frame(V); Frame(W2); }
        const double D = MeanDiff(ReadOutput(Device, V), ReadOutput(Device, W2));
        std::printf("two views (bloom vs none): mean |diff| %.3f\n", D);
        MUST(D > 0.01, "the bloom shows in one view and not the other");
        Renderer.DestroyView(W2);
    }

    Renderer.DestroyView(V);
    MUST(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
