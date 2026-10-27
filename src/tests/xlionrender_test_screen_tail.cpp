// The screen tail (light_mode::SCREEN, documentation/lighting_p3_design.md "The decision"): the night market street of the many-light rig
// (tools/bench_many_lights) at 1024 lights, its image against CELLS' (exact) of the same frame - in a second view, so each keeps its own
// history. Gates: a still camera's per-frame error and its frame-to-frame flicker a fraction of the inline sampled tail's (mode SAMPLED, the
// same resampling, no denoiser); the 16-frame mean near CELLS (the bias of the demodulation); a moving camera, moving lights and a box
// moving across the street (disocclusion) no worse than the inline tail; the targets' memory; no validation error
#include "xlionrender_test_common.h"
#include <random>

using namespace xlionrender::test;

namespace
{
    bool MakeMesh(xlionrender::instance& Renderer, const xprim_geom::mesh& Prim, xlionrender::mesh_id& Id) noexcept
    {
        std::vector<xlionrender::vertex> Verts(Prim.m_Vertices.size());
        for (std::size_t i = 0; i < Verts.size(); ++i)
        {
            const auto& V = Prim.m_Vertices[i];
            Verts[i] = { { V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }, { V.m_Texcoord.m_X, V.m_Texcoord.m_Y }, 0xffffffffu
                       , { V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }, { V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, V.m_Tangent.m_D } };
        }
        return Renderer.CreateMesh(Id, Verts, Prim.m_Indices) == nullptr;
    }

    // The rig's lights (bench_many_lights Lights): string lights, neon, torches (the first 32 shadowed)
    std::vector<xlionrender::local_light> Lights(int n)
    {
        std::mt19937 R(7);
        std::uniform_real_distribution<float> U(0, 1);
        std::vector<xlionrender::local_light> L;
        int nShadowed = 0;
        for (int i = 0; i < n; ++i)
        {
            const float k = U(R);
            xlionrender::local_light l;
            if (k < 0.75f)      { l.m_Position = { -15.0f + 30.0f * U(R), 3.0f + U(R), -15.0f + 30.0f * U(R) }; l.m_Radius = 2.0f + U(R); l.m_Color = { 1.0f, 0.8f + 0.2f * U(R), 0.5f + 0.3f * U(R) }; l.m_Intensity = 0.6f; l.m_bCastShadows = false; }
            else if (k < 0.95f) { l.m_Position = { (U(R) < 0.5f ? -5.5f : 5.5f), 1.5f + 2.0f * U(R), -15.0f + 30.0f * U(R) }; l.m_Radius = 4.0f + 2.0f * U(R); l.m_Color = U(R) < 0.5f ? xmath::fvec3{ 1.0f, 0.2f, 0.6f } : xmath::fvec3{ 0.2f, 0.7f, 1.0f }; l.m_Intensity = 1.5f; l.m_bCastShadows = false; }
            else                { l.m_Position = { -4.0f + 8.0f * U(R), 1.6f, -15.0f + 30.0f * U(R) }; l.m_Radius = 6.0f + 2.0f * U(R); l.m_Color = { 1.0f, 0.6f, 0.3f }; l.m_Intensity = 3.0f; l.m_bCastShadows = nShadowed < xlionrender::max_local_lights_v; nShadowed += l.m_bCastShadows; }
            L.push_back(l);
        }
        return L;
    }

    double RelErr(const std::vector<float>& A, const std::vector<float>& B)
    {
        double d = 0, r = 0;
        for (std::size_t i = 0; i < A.size(); ++i) if ((i & 3) != 3) { d += std::abs(double(A[i]) - B[i]); r += std::abs(double(B[i])); }
        return d / std::max(r, 1e-9);
    }
}

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;
    using mode = xlionrender::instance::light_mode;

    xlionrender::mesh_id Box, ClusterBox;
    CHECK(MakeMesh(Renderer, xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 }), Box), "box");
    cluster_buffers ClusterBuffers;                                             // (a mover: a cluster mesh with an instance id - its transform kept)
    CHECK(MakeStaticBuffers(Device, xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 }), 0.5f, ClusterBuffers) && !Renderer.CreateClusterGeom(ClusterBox, ClusterBuffers.Geom()), "cluster box");

    xlionrender::scene Scene;
    const auto At = [](float X, float Y, float Z, float SX, float SY, float SZ) { return xmath::fmat4::fromTranslation({ X, Y, Z }) * xmath::fmat4::fromScale({ SX, SY, SZ }); };
    Scene.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = At(0, -0.05f, 0, 40, 0.1f, 40), .m_Material = { .m_Roughness = 0.6f } });
    for (int s = -1; s <= 1; s += 2)
        for (int k = 0; k < 6; ++k)
            Scene.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = At(6.0f * float(s), 3.0f, -15.0f + 5.0f * float(k) + 2.5f, 0.6f, 6.0f, 4.6f), .m_Material = { .m_Roughness = 0.8f } });
    for (int k = 0; k < 10; ++k)
    {
        Scene.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = At(-2.5f + 5.0f * float(k % 2), 0.5f, -13.0f + 3.0f * float(k), 1.4f, 1.0f, 1.0f), .m_Material = { .m_Metallic = 0.5f, .m_Roughness = 0.3f } });
        Scene.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = At(-1.0f + 2.0f * float(k % 2), 1.5f, -12.0f + 3.0f * float(k), 0.2f, 3.0f, 0.2f) });
    }
    Scene.m_Sun     = { .m_Intensity = 0.0f, .m_bCastShadows = false };
    Scene.m_Ambient = { 0.01f, 0.01f, 0.012f };
    const auto Base = Lights(1024);
    Scene.m_LocalLights = Base;

    constexpr int W = 640, H = 360;
    xlionrender::render_view View, Ref;
    CHECK(!Renderer.CreateView(View, { .m_Width = W, .m_Height = H }) && !Renderer.CreateView(Ref, { .m_Width = W, .m_Height = H }), "CreateView");
    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.0f });
    Camera.setViewport({ 0, 0, W, H });
    const auto Look = [&](float t) { Camera.LookAt(xmath::fvec3{ 0.4f * std::sin(t), 1.7f, -14.0f + t }, xmath::fvec3{ 0.0f, 1.2f, 10.0f }); };
    Look(0);

    // A frame: the mode under test into View, CELLS into Ref (the same scene and camera); both read back
    std::vector<float> Img, Exact;
    const auto Frame = [&](mode M, bool bRead) -> bool
    {
        Renderer.m_LightMode = M;
        if (!App.RenderFrame(Scene, View, Camera)) return false;
        if (bRead && !ReadHDR(Device, View, Img)) return false;
        Renderer.m_LightMode = mode::CELLS;
        if (!App.RenderFrame(Scene, Ref, Camera)) return false;
        return !bRead || ReadHDR(Device, Ref, Exact);
    };

    // A still camera: per-frame error and flicker, the screen tail against the inline sampled tail
    struct still { double m_Err = 0, m_Flicker = 0, m_MeanErr = 0, m_Ratio = 0; xlionrender::instance::tail_report m_Report; mode m_Used = mode::EXACT; };
    const auto Still = [&](mode M) -> still
    {
        still S;
        for (int f = 0; f < 20; ++f) if (!Frame(M, false)) return { -1, -1, -1 };     // (the shadow pages settle, the history fills)
        std::vector<float> Prev, Mean;
        for (int f = 0; f < 16; ++f)
        {
            if (!Frame(M, true)) return { -1, -1, -1 };
            S.m_Err += RelErr(Img, Exact) / 16;
            if (!Prev.empty()) S.m_Flicker += RelErr(Img, Prev) / 15;
            if (Mean.empty()) Mean.assign(Img.size(), 0.0f);
            for (std::size_t i = 0; i < Img.size(); ++i) Mean[i] += Img[i] / 16.0f;
            Prev = Img;
        }
        S.m_MeanErr = RelErr(Mean, Exact);
        double a = 0, b = 0;
        for (std::size_t i = 0; i < Mean.size(); ++i) if ((i & 3) != 3) { a += Mean[i]; b += Exact[i]; }
        S.m_Ratio = a / std::max(b, 1e-9);
        Renderer.m_LightMode = M;
        if (!App.RenderFrame(Scene, View, Camera)) return { -1, -1, -1 };
        S.m_Report = Renderer.getTailReport(); S.m_Used = Renderer.getLightModeUsed();
        return S;
    };
    // (the denoiser's gates against CELLS, which does not shadow unshadowed lights: the screen trace off - the wall's test below has it)
    Renderer.m_TailTraceLength = 0.0f;
    const auto Inline = Still(mode::SAMPLED), Screen = Still(mode::SCREEN);
    std::printf("still: per-frame error inline %.3f, screen %.3f; flicker inline %.3f, screen %.3f; 16-frame mean's error inline %.3f, screen %.3f; its brightness / CELLS' inline %.3f, screen %.3f\n"
               , Inline.m_Err, Screen.m_Err, Inline.m_Flicker, Screen.m_Flicker, Inline.m_MeanErr, Screen.m_MeanErr, Inline.m_Ratio, Screen.m_Ratio);
    const auto& T = Screen.m_Report;
    std::printf("screen tail %dx%d: G-buffer %.3f ms (%d draws), tail %.3f ms, denoiser %.3f ms; %.1f MB\n", T.m_Width, T.m_Height, T.m_GBufferMs, T.m_nGBufferDraws, T.m_TailMs, T.m_DenoiseMs, T.m_MemoryMB);
    CHECK(T.m_bOn && Screen.m_Used == mode::SCREEN, "the screen tail did not run");
    CHECK(Screen.m_Err >= 0 && Screen.m_Err < 0.5 * Inline.m_Err, "the screen tail's per-frame error %.3f is not under half the inline tail's %.3f", Screen.m_Err, Inline.m_Err);
    CHECK(Screen.m_Flicker < 0.25 * Inline.m_Flicker, "the screen tail flickers %.3f (inline %.3f)", Screen.m_Flicker, Inline.m_Flicker);
    CHECK(Screen.m_MeanErr < 0.1, "the screen tail's mean is %.3f off CELLS (its bias)", Screen.m_MeanErr);
    CHECK(std::abs(Screen.m_Ratio - 1.0) < 0.005, "the screen tail's mean is %.2f%% brighter than CELLS' (its bias: under 0.5%%)", (Screen.m_Ratio - 1.0) * 100.0);

    // Motion: the camera moving (1.5 cm a frame, swaying), the lights moving (a few cm a frame), a box crossing the street (5 cm a frame: what
    // it uncovers has no history). Each: the mean per-frame error over 24 frames, the screen tail's against the inline tail's
    const auto Motion = [&](mode M, int What) -> double
    {
        Look(0);
        Scene.m_LocalLights = Base;
        const auto Draws = Scene.m_Draws;
        if (What == 2) Scene.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = At(-3.0f, 1.0f, -6.0f, 1.0f, 2.0f, 1.0f), .m_Material = { .m_Roughness = 0.5f } });
        for (int f = 0; f < 20; ++f) if (!Frame(M, false)) return -1;
        double E = 0;
        for (int f = 0; f < 24; ++f)
        {
            if (What == 0) Look(0.015f * float(f + 1));
            if (What == 1) for (std::size_t i = 0; i < Scene.m_LocalLights.size(); ++i) Scene.m_LocalLights[i].m_Position = Base[i].m_Position + xmath::fvec3{ 0.3f * std::sin(0.2f * float(f) + float(i)), 0, 0.3f * std::cos(0.2f * float(f) + float(i)) };
            if (What == 2) Scene.m_Draws.back().m_L2W = At(-3.0f + 0.05f * float(f + 1), 1.0f, -6.0f, 1.0f, 2.0f, 1.0f);
            if (!Frame(M, true)) return -1;
            E += RelErr(Img, Exact) / 24;
        }
        Scene.m_Draws = Draws;
        Scene.m_LocalLights = Base;
        Look(0);
        return E;
    };
    const char* Names[] = { "moving camera", "moving lights", "disocclusion (a moving box)" };
    for (int What = 0; What < 3; ++What)
    {
        const double EI = Motion(mode::SAMPLED, What), ES = Motion(mode::SCREEN, What);
        std::printf("%s: per-frame error inline %.3f, screen %.3f\n", Names[What], EI, ES);
        CHECK(ES >= 0 && ES < EI, "%s: the screen tail's error %.3f is not under the inline tail's %.3f", Names[What], ES, EI);
    }

    // An abrupt change: half the lights off at once - the history must not keep them (the clamp to this frame's neighbourhood drops
    // it). The error against CELLS 1, 2, 4 and 8 frames after, the screen tail's against the inline tail's 8 frames after
    {
        const auto Switch = [&](mode M, double* pErr) -> bool
        {
            Scene.m_LocalLights = Base;
            for (int f = 0; f < 24; ++f) if (!Frame(M, false)) return false;
            for (std::size_t i = 0; i < Scene.m_LocalLights.size(); i += 2) Scene.m_LocalLights[i].m_Intensity = 0;
            for (int f = 1; f <= 8; ++f)
            {
                if (!Frame(M, true)) return false;
                if (f == 1 || f == 2 || f == 4 || f == 8) *pErr++ = RelErr(Img, Exact);
            }
            Scene.m_LocalLights = Base;
            return true;
        };
        double EI[4], ES[4];
        CHECK(Switch(mode::SAMPLED, EI) && Switch(mode::SCREEN, ES), "Render the switch");
        std::printf("half the lights off: error 1 / 2 / 4 / 8 frames after, inline %.3f %.3f %.3f %.3f, screen %.3f %.3f %.3f %.3f\n", EI[0], EI[1], EI[2], EI[3], ES[0], ES[1], ES[2], ES[3]);
        CHECK(ES[3] < EI[3], "8 frames after half the lights went off the screen tail's error %.3f is not under the inline tail's %.3f", ES[3], EI[3]);
    }

    // A mover (motion vectors): a box crossing the street fast (12 cm a frame), a cluster mesh with an instance id - its last transform kept,
    // the G-buffer's motion reprojects its texels - against the same box without an id (no motion of its own: the camera's reprojection,
    // its tail drags behind it until the clamp drops it). The error over the 24 frames of its crossing
    {
        const auto Cross = [&](mode M, std::uint64_t Instance) -> double
        {
            Look(0);
            Scene.m_LocalLights = Base;
            const auto Draws = Scene.m_Draws;
            const auto Pose  = [&](int f) { return At(-3.0f + 0.12f * float(f), 1.2f, -8.0f, 1.5f, 2.4f, 1.5f); };
            Scene.m_Draws.push_back({ .m_Mesh = ClusterBox, .m_L2W = Pose(0), .m_Material = { .m_Roughness = 0.4f }, .m_Instance = Instance });
            for (int f = 0; f < 20; ++f) if (!Frame(M, false)) return -1;
            double E = 0;
            for (int f = 0; f < 24; ++f)
            {
                Scene.m_Draws.back().m_L2W = Pose(f + 1);
                if (!Frame(M, true)) return -1;
                E += RelErr(Img, Exact) / 24;
            }
            Scene.m_Draws = Draws;
            return E;
        };
        const double EI = Cross(mode::SAMPLED, 77), EStatic = Cross(mode::SCREEN, 0), EMoving = Cross(mode::SCREEN, 77);
        std::printf("a mover crossing: per-frame error inline %.3f, screen without its motion %.3f, with it %.3f\n", EI, EStatic, EMoving);
        CHECK(EMoving < EStatic && EMoving < EI, "the mover's motion does not help: %.3f with it, %.3f without (inline %.3f)", EMoving, EStatic, EI);
    }

    // The screen trace (P3: "the tail contact-shadowed"): a wall, 32 point lights on one side within reach of the other. The reference: the
    // lights shadowed by their pages (CELLS). Against it, the lights unshadowed in SCREEN - their tail through the wall (a leak) without the
    // trace, contact-shadowed with it (the head's lights, shaded by the materials, still leak: they are the ones pages are for)
    {
        xlionrender::scene Wall;
        Wall.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = At(0, -0.05f, 0, 30, 0.1f, 30), .m_Material = { .m_Roughness = 0.7f } });
        Wall.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = At(0, 1.25f, 0, 0.3f, 2.5f, 14.0f), .m_Material = { .m_Roughness = 0.7f } });
        Wall.m_Sun     = { .m_Intensity = 0.0f, .m_bCastShadows = false };
        Wall.m_Ambient = { 0.01f, 0.01f, 0.012f };
        std::mt19937 Rng(3);
        std::uniform_real_distribution<float> U(0, 1);
        for (int i = 0; i < 32; ++i)
            Wall.m_LocalLights.push_back({ .m_Position = { -0.6f - 1.8f * U(Rng), 0.4f + 1.4f * U(Rng), -6.0f + 12.0f * U(Rng) }, .m_Radius = 5.0f, .m_Color = { 1.0f, 0.85f, 0.6f }, .m_Intensity = 0.8f, .m_bCastShadows = true });
        xgpu::tools::view WallCam;
        WallCam.setFov(xmath::radian{ 1.0f });
        WallCam.setViewport({ 0, 0, W, H });
        WallCam.LookAt(xmath::fvec3{ 5.0f, 1.6f, -7.0f }, xmath::fvec3{ 0.0f, 0.5f, 0.0f });
        std::vector<float> Shadowed, Img2;
        Renderer.m_LightMode = mode::CELLS;
        bool bOk = true;
        for (int f = 0; f < 20 && bOk; ++f) bOk = App.RenderFrame(Wall, Ref, WallCam);
        bOk = bOk && ReadHDR(Device, Ref, Shadowed);
        for (auto& L : Wall.m_LocalLights) L.m_bCastShadows = false;
        const auto Screen = [&](float Trace) -> double
        {
            Renderer.m_LightMode = mode::SCREEN;
            Renderer.m_TailTraceLength = Trace;
            double E = 0;
            for (int f = 0; f < 24 && bOk; ++f)
            {
                bOk = App.RenderFrame(Wall, View, WallCam);
                if (f >= 16) { bOk = bOk && ReadHDR(Device, View, Img2); E += RelErr(Img2, Shadowed) / 8; }
            }
            return E;
        };
        const double ENo = Screen(0.0f), ETrace = Screen(1.0f);
        Renderer.m_TailTraceLength = 1.0f;
        CHECK(bOk, "Render the wall");
        std::printf("a wall, 32 unshadowed lights against their shadowed reference: SCREEN without the screen trace %.3f, with it %.3f\n", ENo, ETrace);
        CHECK(ETrace < 0.85 * ENo, "the screen trace does not shadow the tail through the wall: %.3f with it, %.3f without", ETrace, ENo);
    }

    // The memory at Tomas's Bistro view (3093 x 1493): the targets at half size
    {
        xlionrender::render_view Big;
        CHECK(!Renderer.CreateView(Big, { .m_Width = 3093, .m_Height = 1493 }), "CreateView big");
        Renderer.m_LightMode = mode::SCREEN;
        xgpu::tools::view BigCam = Camera;
        BigCam.setViewport({ 0, 0, 3093, 1493 });
        for (int f = 0; f < 6; ++f) CHECK(App.RenderFrame(Scene, Big, BigCam), "Render big");
        const auto& T = Renderer.getTailReport();
        std::printf("3093x1493: screen tail %dx%d, %.1f MB; G-buffer %.3f ms, tail %.3f ms, denoiser %.3f ms\n", T.m_Width, T.m_Height, T.m_MemoryMB, T.m_GBufferMs, T.m_TailMs, T.m_DenoiseMs);
        CHECK(T.m_MemoryMB < 64.0, "the screen tail's targets take %.1f MB at 3093x1493", T.m_MemoryMB);
        Renderer.DestroyView(Big);
    }

    Renderer.m_LightMode = mode::CELLS;
    Renderer.DestroyView(View);
    Renderer.DestroyView(Ref);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
