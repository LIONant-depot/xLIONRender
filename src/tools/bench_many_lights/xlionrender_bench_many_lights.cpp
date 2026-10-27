// The dense many-light rig (documentation/lighting_p3_design.md: the evidence its decisions ask for) - a benchmark, not a test.
// A 30 x 30 m market street: a ground, two rows of house walls, stalls and pillars; game-like lights at a density swept from 64 to 4096:
// string lights (most of them: small, dim, unshadowed), neon (medium, coloured, unshadowed) and torches (bright, shadowed: up to 32).
// Per density, mode (CELLS: exact, SAMPLED: the head + the resampled tail) and view size (1920 x 1080, 3093 x 1493: Tomas's Bistro
// view): the GPU scene pass (mean of 60 warmed frames) per mode - CELLS, and SAMPLED with plain tail samples, resampled, resampled with
// visibility (the default) -, the cell build (on the GPU: CELLS' and SAMPLED's, its head/tail split; the CPU's part; the CPU's Build
// for comparison), the cells whose list overflowed (shaded with every light), and SAMPLED's per-frame error against CELLS' image
// (mean |a - b| / mean b).
//
//   xlionrender_bench_many_lights [--densities 64,256,1024,2048,4096] [--frames N] [--big 0|1] [--cpu 1: the modes measured with the CPU's Build]
#include "tests/xlionrender_test_common.h"
#include <random>
#include <sstream>

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

    // The street's lights at a density: 75% string lights along the street (2 - 3 m reach), 20% neon on the walls (4 - 6 m), 5% torches
    // (6 - 8 m, the first 32 of them shadowed)
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
            if (k < 0.75f)          // string lights: strands across the street at 3 - 4 m
            {
                l.m_Position  = { -15.0f + 30.0f * U(R), 3.0f + U(R), -15.0f + 30.0f * U(R) };
                l.m_Radius    = 2.0f + U(R);
                l.m_Color     = { 1.0f, 0.8f + 0.2f * U(R), 0.5f + 0.3f * U(R) };
                l.m_Intensity = 0.6f;
                l.m_bCastShadows = false;
            }
            else if (k < 0.95f)     // neon on the walls
            {
                l.m_Position  = { (U(R) < 0.5f ? -5.5f : 5.5f), 1.5f + 2.0f * U(R), -15.0f + 30.0f * U(R) };
                l.m_Radius    = 4.0f + 2.0f * U(R);
                l.m_Color     = U(R) < 0.5f ? xmath::fvec3{ 1.0f, 0.2f, 0.6f } : xmath::fvec3{ 0.2f, 0.7f, 1.0f };
                l.m_Intensity = 1.5f;
                l.m_bCastShadows = false;
            }
            else                    // torches on the stalls
            {
                l.m_Position  = { -4.0f + 8.0f * U(R), 1.6f, -15.0f + 30.0f * U(R) };
                l.m_Radius    = 6.0f + 2.0f * U(R);
                l.m_Color     = { 1.0f, 0.6f, 0.3f };
                l.m_Intensity = 3.0f;
                l.m_bCastShadows = nShadowed < xlionrender::max_local_lights_v;
                nShadowed += l.m_bCastShadows;
            }
            L.push_back(l);
        }
        return L;
    }
}

int main(int argc, char** argv)
{
    std::vector<int> Densities{ 64, 256, 1024, 2048, 4096 };
    int  nFrames = 60;
    bool bBig    = true;
    bool bCPU    = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string A = argv[i];
        if (A == "--densities" && i + 1 < argc) { Densities.clear(); std::stringstream S(argv[++i]); std::string T; while (std::getline(S, T, ',')) Densities.push_back(std::atoi(T.c_str())); }
        else if (A == "--frames" && i + 1 < argc) nFrames = std::max(1, std::atoi(argv[++i]));
        else if (A == "--big" && i + 1 < argc) bBig = std::atoi(argv[++i]) != 0;
        else if (A == "--cpu" && i + 1 < argc) bCPU = std::atoi(argv[++i]) != 0;
    }

    app App;
    if (!App.Init()) { std::printf("FAIL: Init\n"); return 1; }
    App.m_bQuietTimes = true;
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;
    using mode = xlionrender::instance::light_mode;

    xlionrender::mesh_id Box;
    if (!MakeMesh(Renderer, xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 }), Box)) { std::printf("FAIL: box\n"); return 1; }

    // The street: a ground, house walls each side (6 m apart), stalls and pillars between them
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
    Scene.m_Sun     = { .m_Intensity = 0.0f, .m_bCastShadows = false };       // night: the local lights only
    Scene.m_Ambient = { 0.01f, 0.01f, 0.012f };

    struct size { int W, H; const char* Name; };
    std::vector<size> Sizes{ { 1920, 1080, "1920x1080" } };
    if (bBig) Sizes.push_back({ 3093, 1493, "3093x1493" });

    // The modes: CELLS (exact), SAMPLED with the plain tail samples, resampled (8 candidates), resampled with their visibility (the default)
    struct variant { const char* Name; mode M; int Candidates; bool bVis; };
    const variant Variants[] = { { "cells", mode::CELLS, 0, false }, { "plain", mode::SAMPLED, 0, false }, { "ris", mode::SAMPLED, 8, false }, { "ris+vis", mode::SAMPLED, 8, true } };
    std::printf("many-light rig: %d warmed frames a measure; per mode the GPU scene pass ms (*: the cell buffer was full, the render fell back to EXACT), and SAMPLED's per-frame error against CELLS (mean |a - b| / mean b); the CPU cell build ms\n", nFrames);
    std::printf("%-10s %6s %9s %9s %9s %9s %9s %8s |", "view", "lights", "overflow", "cpu Build", "cpu part", "gpu cells", "gpu split", "ring MB");
    for (const auto& V : Variants) std::printf(" %8s ms", V.Name);
    for (int v = 1; v < 4; ++v) std::printf(" %8s err", Variants[v].Name);
    std::printf("\n");
    for (const auto& Sz : Sizes)
    {
        xlionrender::render_view View;
        if (Renderer.CreateView(View, { .m_Width = Sz.W, .m_Height = Sz.H })) { std::printf("FAIL: CreateView %s\n", Sz.Name); return 1; }
        xgpu::tools::view Camera;
        Camera.setFov(xmath::radian{ 1.0f });
        Camera.setViewport({ 0, 0, Sz.W, Sz.H });
        Camera.LookAt(xmath::fvec3{ 0.0f, 1.7f, -14.0f }, xmath::fvec3{ 0.0f, 1.2f, 10.0f });
        for (int n : Densities)
        {
            Scene.m_LocalLights = Lights(n);
            double Gpu[4] = {}, Err[4] = {}, Build = 0, CpuPart = 0, GpuBuild[4] = {}, RingMB = 0; int Over = 0; bool bFell[4] = {};
            {   // the CPU's Build, for comparison (the GPU builds the cells by default)
                Renderer.m_bLightCellsGPU = false; Renderer.m_LightMode = mode::CELLS;
                for (int f = 0; f < 8; ++f) { if (!App.RenderFrame(Scene, View, Camera)) return 1; if (f >= 4) Build += Renderer.getLightBuildMs() / 4; }
                Renderer.m_bLightCellsGPU = !bCPU;
            }
            std::vector<float> Ref, Fr;
            for (int v = 0; v < 4; ++v)
            {
                Renderer.m_LightMode = Variants[v].M;
                Renderer.m_LightTailCandidates = Variants[v].Candidates;
                Renderer.m_bLightTailVisibility = Variants[v].bVis;
                for (int f = 0; f < 16; ++f) if (!App.RenderFrame(Scene, View, Camera)) return 1;          // (the shadow pages settle, warm)
                if (v == 0 && !ReadHDR(Device, View, Ref)) return 1;
                int nGpu = 0;
                for (int f = 0; f < nFrames; ++f)
                {
                    if (!App.RenderFrame(Scene, View, Camera)) return 1;
                    if (App.m_bGotTimes) { Gpu[v] += App.m_LastTimes.m_SceneMs; ++nGpu; }
                    if (v == 0) CpuPart += Renderer.getLightBuildMs() / nFrames;
                    GpuBuild[v] += Renderer.getLightBuildGpuMs() / nFrames;
                    if (v > 0 && f < 4)            // (the error: a few frames' readbacks - 4.6 M pixels each at the big size)
                    {
                        if (!ReadHDR(Device, View, Fr)) return 1;
                        double d = 0, r = 0;
                        for (std::size_t i = 0; i < Fr.size(); ++i) { d += std::abs(double(Fr[i]) - Ref[i]); r += std::abs(double(Ref[i])); }
                        Err[v] += d / std::max(r, 1e-9) / 4.0;
                    }
                }
                Gpu[v] /= std::max(nGpu, 1);
                bFell[v] = Renderer.getLightModeUsed() != Variants[v].M;       // (the cell buffer full: the whole render EXACT)
                if (v == 0) { Over = Renderer.getLightStats().m_nOverflowCells; RingMB = Renderer.getLightStats().m_CellRingMB; }
            }
            std::printf("%-10s %6d %9d %9.3f %9.3f %9.3f %9.3f %8.1f |", Sz.Name, n, Over, Build, CpuPart, GpuBuild[0], GpuBuild[2], RingMB);
            for (int v = 0; v < 4; ++v) std::printf(" %10.3f%c", Gpu[v], bFell[v] ? '*' : ' ');
            for (int v = 1; v < 4; ++v) std::printf(" %12.3f", Err[v]);
            std::printf("\n");
            std::fflush(stdout);
        }
        Renderer.DestroyView(View);
    }
    Renderer.m_LightTailCandidates = 8;
    Renderer.m_bLightTailVisibility = true;
    Renderer.m_LightMode = mode::CELLS;
    std::printf("PASS\n");
    return 0;
}
