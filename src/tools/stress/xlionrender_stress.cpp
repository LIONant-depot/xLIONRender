// xLIONRender stress scene: a procedural kitbashed city (buildings and props from xprim_geom primitives as static cluster
// geometry with persistent instance ids, street lights, the sun) rendered in a window along scripted camera paths, so
// every run is the same. Each scenario ends with a summary (mean, p95, max per measure) on the console, and every frame
// goes to a CSV. Measures, per frame: wall time, the CPU stages of Render, the CPU wait (BeginRendering + PageFlip), the
// GPU passes (shadows, scene, tone map), the draws and lights.
//
//   xlionrender_stress [--scenario flyover|street|moving_lights|all] [--frames N] [--grid N] [--lights N]
//                      [--mode exact|cells|sampled] [--csv FILE] [--validate] [--loop]
//
// The window's title shows the frame rate and the main times, updated every half second.
//
// The research doc's performance baseline comes from here (documentation/xlion_renderer_research.md).
#include "tests/xlionrender_test_common.h"         // the cluster geometry upload (MakeStaticBuffers) and the logs
#include <chrono>
#include <cstring>
#include <fstream>
#include <random>
#include <string>

using namespace xlionrender::test;

namespace
{
    struct options
    {
        std::string     m_Scenario  = "all";
        int             m_Frames    = 600;
        int             m_Grid      = 40;           // city blocks per side
        int             m_Lights    = 1024;
        std::string     m_Mode      = "cells";
        std::string     m_Csv       = "xlionrender_stress.csv";
        bool            m_bValidate = false;
        bool            m_bLoop     = false;
        int             m_Probes    = 0;            // reflection probes along the walked street (captured 1 face per frame)        // repeat the scenarios until the window closes (to watch the title's readout)
    };

    options ParseOptions(int argc, char** argv)
    {
        options O;
        for (int i = 1; i < argc; ++i)
        {
            const std::string A = argv[i];
            const auto Next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
            if      (A == "--scenario") O.m_Scenario = Next();
            else if (A == "--frames")   O.m_Frames   = std::max(1, std::atoi(Next().c_str()));
            else if (A == "--grid")     O.m_Grid     = std::max(1, std::atoi(Next().c_str()));
            else if (A == "--lights")   O.m_Lights   = std::max(0, std::atoi(Next().c_str()));
            else if (A == "--mode")     O.m_Mode     = Next();
            else if (A == "--csv")      O.m_Csv      = Next();
            else if (A == "--validate") O.m_bValidate = true;
            else if (A == "--loop")     O.m_bLoop     = true;
            else if (A == "--probes")   O.m_Probes    = std::max(0, std::atoi(Next().c_str()));
        }
        return O;
    }

    constexpr float block_v  = 8.0f;        // a building's footprint
    constexpr float street_v = 4.0f;        // street width
    constexpr float pitch_v  = block_v + street_v;

    struct city
    {
        std::vector<cluster_buffers>        m_Buffers;          // per primitive kind
        std::vector<xlionrender::mesh_id>   m_Meshes;
        xlionrender::scene                  m_Scene;
        std::vector<xmath::fvec3>           m_LightBase;        // the lights' rest positions (moving_lights)
        float                               m_HalfSize = 0;
    };

    bool BuildCity(xgpu::device& Device, xlionrender::instance& Renderer, const options& O, city& C)
    {
        // Primitive kinds: box, sphere, cylinder, capsule (unit sized; a cluster scale of 1)
        const xprim_geom::mesh Prims[] =
        { xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 })
        , xprim_geom::uvsphere::Generate(16, 16, 1, 0.5f)
        , xprim_geom::cylinder::Generate(1, 16, 1.0f, 0.5f, 0.5f)
        , xprim_geom::capsule::Generate(8, 16, 0.25f, 1.0f)
        };
        C.m_Buffers.resize(std::size(Prims));
        C.m_Meshes.resize(std::size(Prims));
        for (std::size_t k = 0; k < std::size(Prims); ++k)
        {
            if (!MakeStaticBuffers(Device, Prims[k], 1.0f, C.m_Buffers[k])) return false;
            if (Renderer.CreateClusterGeom(C.m_Meshes[k], C.m_Buffers[k].Geom())) return false;
        }

        std::mt19937 R(1234);
        std::uniform_real_distribution<float> U(0, 1);
        const auto Material = [&](float Lum)
        {
            return xlionrender::material{ .m_BaseColor = { Lum * (0.6f + 0.4f * U(R)), Lum * (0.6f + 0.4f * U(R)), Lum * (0.6f + 0.4f * U(R)) }
                                        , .m_Metallic = U(R) < 0.2f ? 0.9f : 0.0f, .m_Roughness = 0.2f + 0.7f * U(R) };
        };

        auto& Draws = C.m_Scene.m_Draws;
        std::uint32_t ID = 1;
        C.m_HalfSize = O.m_Grid * pitch_v * 0.5f;
        Draws.push_back({ .m_Mesh = C.m_Meshes[0], .m_L2W = xmath::fmat4::fromTranslation({ 0, -0.5f, 0 }) * xmath::fmat4::fromScale({ 2 * C.m_HalfSize + 20, 1, 2 * C.m_HalfSize + 20 })
                        , .m_Material = { .m_BaseColor = { 0.25f, 0.25f, 0.27f }, .m_Roughness = 0.9f }, .m_Instance = ID++ });
        for (int bz = 0; bz < O.m_Grid; ++bz)
        for (int bx = 0; bx < O.m_Grid; ++bx)
        {
            const float X = -C.m_HalfSize + (bx + 0.5f) * pitch_v, Z = -C.m_HalfSize + (bz + 0.5f) * pitch_v;
            const float H = 2.0f + 14.0f * U(R) * U(R);
            Draws.push_back({ .m_Mesh = C.m_Meshes[0], .m_L2W = xmath::fmat4::fromTranslation({ X, H * 0.5f, Z }) * xmath::fmat4::fromScale({ block_v * (0.6f + 0.4f * U(R)), H, block_v * (0.6f + 0.4f * U(R)) })
                            , .m_Material = Material(0.8f), .m_Instance = ID++ });
            for (int p = 0; p < 4; ++p)         // props on the block's corners: kitbash
            {
                const int   Kind = 1 + int(U(R) * 3) % 3;
                const float S    = 0.5f + U(R);
                const float PX   = X + ((p & 1) ? 1 : -1) * (block_v * 0.5f + 0.8f), PZ = Z + ((p & 2) ? 1 : -1) * (block_v * 0.5f + 0.8f);
                Draws.push_back({ .m_Mesh = C.m_Meshes[Kind], .m_L2W = xmath::fmat4::fromTranslation({ PX, S * 0.5f, PZ }) * xmath::fmat4::fromScale({ S, S, S })
                                , .m_Material = Material(0.6f), .m_Instance = ID++ });
            }
        }

        // Street lights along the streets (a lattice over the street centerlines), the first 16 shadowed, a quarter spots
        auto& Lights = C.m_Scene.m_LocalLights;
        for (int i = 0; i < O.m_Lights; ++i)
        {
            const bool  bAlongX = (i & 1) != 0;
            const int   Line    = int(U(R) * (O.m_Grid + 1));
            const float Along   = (U(R) * 2 - 1) * C.m_HalfSize;
            const float Across  = -C.m_HalfSize + Line * pitch_v;
            xlionrender::local_light L
            { .m_Position = bAlongX ? xmath::fvec3{ Along, 3.0f, Across } : xmath::fvec3{ Across, 3.0f, Along }
            , .m_Radius = 7.0f, .m_Color = { 1.0f, 0.75f + 0.25f * U(R), 0.5f + 0.5f * U(R) }, .m_Intensity = 3.0f, .m_bCastShadows = i < 16 };
            if (i % 4 == 3) { L.m_Type = xlionrender::local_light::type::SPOT; L.m_Direction = { 0, -1, 0 }; L.m_InnerAngle = 0.5f; L.m_OuterAngle = 0.8f; }
            Lights.push_back(L);
            C.m_LightBase.push_back(L.m_Position);
        }

        C.m_Scene.m_Sun     = { .m_Direction = { 0.4f, -1.0f, 0.3f }, .m_Intensity = 1.5f, .m_ShadowTileSize = 4.0f };
        C.m_Scene.m_Sky     = { .m_Intensity = 0.3f };

        // Probes down the street the "street" scenario walks, each box a block of the street
        const float StreetX = -C.m_HalfSize + std::round(C.m_HalfSize / pitch_v) * pitch_v;
        for (int i = 0; i < std::min(O.m_Probes, xlionrender::max_probes_v); ++i)
        {
            const float Z = -C.m_HalfSize * 0.9f + (i + 0.5f) * (C.m_HalfSize * 1.8f / std::max(O.m_Probes, 1));
            const float H = C.m_HalfSize * 0.9f / std::max(O.m_Probes, 1);
            C.m_Scene.m_Probes.push_back({ .m_Position = { StreetX, 2.0f, Z }, .m_BoxMin = { StreetX - pitch_v * 0.5f, 0.0f, Z - H }, .m_BoxMax = { StreetX + pitch_v * 0.5f, 12.0f, Z + H }, .m_Blend = 1.0f });
        }
        C.m_Scene.m_Ambient = { 0.03f, 0.03f, 0.04f };
        return true;
    }

    // The camera of a scenario at time t (seconds): orbit above the city, or a walk down a street
    void Place(xgpu::tools::view& Camera, const std::string& Scenario, float t, float HalfSize)
    {
        if (Scenario == "flyover")
        {
            const float A = t * 0.15f, Rad = HalfSize * 0.8f;
            Camera.LookAt(xmath::fvec3{ std::cos(A) * Rad, HalfSize * 0.35f, std::sin(A) * Rad }, xmath::fvec3{ 0, 0, 0 });
            return;
        }
        // street / moving_lights: eye height down a street, looking along it, a slow sway
        const float StreetX = -HalfSize + std::round(HalfSize / pitch_v) * pitch_v;
        const float Z       = -HalfSize * 0.9f + std::fmod(t * 6.0f, HalfSize * 1.8f);
        const float Sway    = std::sin(t * 0.7f) * 0.6f;
        Camera.LookAt(xmath::fvec3{ StreetX, 1.7f, Z }, xmath::fvec3{ StreetX + Sway * 10.0f, 1.5f, Z + 10.0f });
    }

    struct frame_sample
    {
        double  m_WallMs, m_RenderMs, m_InstancesMs, m_ShadowsCpuMs, m_CellsMs, m_WaitMs;
        double  m_GpuShadowMs, m_GpuSceneMs, m_GpuToneMapMs;
        int     m_ShadowTiles;
        int     m_DrawsRendered;
    };

    void Summary(const std::string& Name, const std::vector<frame_sample>& S)
    {
        const auto Report = [&](const char* What, auto Get)
        {
            std::vector<double> V;
            for (std::size_t i = S.size() / 10; i < S.size(); ++i) V.push_back(Get(S[i]));     // the first 10% warm up
            if (V.empty()) return;
            std::sort(V.begin(), V.end());
            double Sum = 0;
            for (double x : V) Sum += x;
            std::printf("  %-22s mean %8.3f   p95 %8.3f   max %8.3f\n", What, Sum / V.size(), V[std::min(V.size() - 1, std::size_t(V.size() * 0.95))], V.back());
        };
        std::printf("%s (%d frames):\n", Name.c_str(), int(S.size()));
        Report("frame (wall) ms",       [](const frame_sample& s) { return s.m_WallMs; });
        Report("CPU Render ms",         [](const frame_sample& s) { return s.m_RenderMs; });
        Report("  instances ms",        [](const frame_sample& s) { return s.m_InstancesMs; });
        Report("  shadows ms",          [](const frame_sample& s) { return s.m_ShadowsCpuMs; });
        Report("  light cells ms",      [](const frame_sample& s) { return s.m_CellsMs; });
        Report("CPU wait ms",           [](const frame_sample& s) { return s.m_WaitMs; });
        Report("GPU shadows ms",        [](const frame_sample& s) { return s.m_GpuShadowMs; });
        Report("GPU scene ms",          [](const frame_sample& s) { return s.m_GpuSceneMs; });
        Report("GPU tone map ms",       [](const frame_sample& s) { return s.m_GpuToneMapMs; });
        Report("shadow tiles / maps",   [](const frame_sample& s) { return double(s.m_ShadowTiles); });
        Report("draws drawn",           [](const frame_sample& s) { return double(s.m_DrawsRendered); });
    }
}

int main(int argc, char** argv)
{
    const options O = ParseOptions(argc, argv);

    xgpu::instance          Instance;
    xgpu::device            Device;
    xgpu::window            Window;
    xlionrender::instance   Renderer;
    constexpr int width_v = 1280, height_v = 720;
    if (xgpu::CreateInstance(Instance, { .m_bDebugMode = O.m_bValidate, .m_bEnableRenderDoc = false, .m_pLogErrorFunc = LogError, .m_pLogWarning = LogWarning })) return 1;
    if (Instance.Create(Device))                                                return 1;
    if (Device.Create(Window, { .m_Width = width_v, .m_Height = height_v }))    return 1;
    if (Renderer.Init(Device))                                                  return 1;

    using mode = xlionrender::instance::light_mode;
    Renderer.m_LightMode = O.m_Mode == "exact" ? mode::EXACT : O.m_Mode == "sampled" ? mode::SAMPLED : mode::CELLS;

    city C;
    if (!BuildCity(Device, Renderer, O, C)) { std::printf("FAIL: building the city\n"); return 1; }
    std::printf("city: %d draws, %d lights, mode %s, %dx%d%s\n", int(C.m_Scene.m_Draws.size()), int(C.m_Scene.m_LocalLights.size()), O.m_Mode.c_str()
               , width_v, height_v, O.m_bValidate ? ", validation on" : "");

    xlionrender::render_view View;
    if (Renderer.CreateView(View, { .m_Width = width_v, .m_Height = height_v })) return 1;
    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.0f });

    std::ofstream Csv(O.m_Csv);
    Csv << "scenario,frame,wall_ms,cpu_render_ms,cpu_instances_ms,cpu_shadows_ms,cpu_cells_ms,cpu_wait_ms,gpu_shadow_ms,gpu_scene_ms,gpu_tonemap_ms,shadow_tiles,draws,draws_drawn,lights,mode\n";

    using clock = std::chrono::steady_clock;
    const auto Ms = [](clock::time_point A, clock::time_point B) { return std::chrono::duration<double, std::milli>(B - A).count(); };
    const std::vector<std::string> Scenarios = O.m_Scenario == "all" ? std::vector<std::string>{ "flyover", "street", "moving_lights" } : std::vector<std::string>{ O.m_Scenario };

    xlionrender::instance::gpu_times Gpu;
    do for (const auto& Scenario : Scenarios)
    {
        std::vector<frame_sample> Samples;
        auto Last = clock::now();
        for (int f = 0; f < O.m_Frames; ++f)
        {
            const float t = f / 60.0f;                                     // scripted time: the same path whatever the frame rate
            Place(Camera, Scenario, t, C.m_HalfSize);
            if (Scenario == "moving_lights")
                for (std::size_t i = 0; i < C.m_Scene.m_LocalLights.size(); ++i)
                {
                    const float Ph = float(i) * 0.37f;
                    C.m_Scene.m_LocalLights[i].m_Position = C.m_LightBase[i] + xmath::fvec3{ std::sin(t * 1.3f + Ph) * 2.0f, 0, std::cos(t * 1.1f + Ph) * 2.0f };
                }

            const auto W0 = clock::now();
            if (!Instance.ProcessInputEvents()) return 0;
            while (Window.BeginRendering()) if (!Instance.ProcessInputEvents()) return 0;
            const auto W1 = clock::now();
            Renderer.Render(Window, C.m_Scene, View, Camera);
            {
                auto Cmd = Window.getCmdBuffer();
                Renderer.Blit(Cmd, View);
            }
            const auto P0 = clock::now();
            Window.PageFlip();
            const auto P1 = clock::now();
            Renderer.getGPUTimes(Gpu);                                     // the newest finished frame's (a frame or two behind)

            const auto  Now = clock::now();
            const auto& Cpu = Renderer.getCPUTimes();
            frame_sample S{ Ms(Last, Now), Cpu.m_RenderMs, Cpu.m_InstancesMs, Cpu.m_ShadowsMs, Cpu.m_LightCellsMs, Ms(W0, W1) + Ms(P0, P1)
                          , Gpu.m_ShadowMs, Gpu.m_SceneMs, Gpu.m_ToneMapMs, Renderer.getShadowTilesRendered(), Renderer.getDrawsRendered() };
            Last = Now;
            Samples.push_back(S);
            static const char* Modes[] = { "exact", "cells", "sampled" };

            // The title: a live readout, the mean of the last half second
            {
                static double   TitleMs = 0, TitleCpu = 0, TitleGpu = 0;
                static int      nTitle  = 0;
                TitleMs += S.m_WallMs; TitleCpu += S.m_RenderMs; TitleGpu += S.m_GpuSceneMs + S.m_GpuShadowMs + S.m_GpuToneMapMs; ++nTitle;
                if (TitleMs >= 500.0)
                {
                    char Title[256];
                    std::snprintf(Title, sizeof(Title), "xLIONRender stress - %s - %.0f FPS (%.2f ms) - CPU Render %.2f ms - GPU %.2f ms - %d/%d draws - %d lights %s"
                                 , Scenario.c_str(), 1000.0 * nTitle / TitleMs, TitleMs / nTitle, TitleCpu / nTitle, TitleGpu / nTitle
                                 , S.m_DrawsRendered, int(C.m_Scene.m_Draws.size()), int(C.m_Scene.m_LocalLights.size()), Modes[int(Renderer.getLightModeUsed())]);
                    Window.setTitle(Title);
                    TitleMs = TitleCpu = TitleGpu = 0;
                    nTitle  = 0;
                }
            }
            Csv << Scenario << ',' << f << ',' << S.m_WallMs << ',' << S.m_RenderMs << ',' << S.m_InstancesMs << ',' << S.m_ShadowsCpuMs << ',' << S.m_CellsMs << ','
                << S.m_WaitMs << ',' << S.m_GpuShadowMs << ',' << S.m_GpuSceneMs << ',' << S.m_GpuToneMapMs << ',' << S.m_ShadowTiles << ','
                << C.m_Scene.m_Draws.size() << ',' << S.m_DrawsRendered << ',' << C.m_Scene.m_LocalLights.size() << ',' << Modes[int(Renderer.getLightModeUsed())] << '\n';
        }
        Summary(Scenario, Samples);
        for (std::size_t i = 0; i < C.m_Scene.m_LocalLights.size(); ++i) C.m_Scene.m_LocalLights[i].m_Position = C.m_LightBase[i];
    }
    while (O.m_bLoop);

    std::printf("per frame: %s\n", O.m_Csv.c_str());
    Renderer.DestroyView(View);
    if (O.m_bValidate && g_nErrors) { std::printf("FAIL: %d validation errors\n", g_nErrors); return 1; }
    return 0;
}
