// Material overrides micro-benchmark (documentation/material_overrides_vs_competition.md): 10k entities on one mesh and one material, with
// overrides at 0%, 50% and 99% dedup (10000, 5000 and 100 distinct values), against the same entities with no overrides.
//  - CPU: the first Acquire per slot (makes the materials), the steady per-frame Acquire, a value change (in place, copy on write), a texture change
//  - frame: Render's CPU time and the scene pass's GPU time (mean over the frames)
//  - memory: override materials made, MaterialParams vec4s used
//   Build/test/vs2022/Release/xlionrender_bench_overrides.exe [--entities N] [--frames F]
#include "tests/xlionrender_test_common.h"
#include "scene/xlionrender_material_overrides.h"
#include <chrono>
#include <algorithm>

using namespace xlionrender::test;

namespace
{
    using bench_clock = std::chrono::steady_clock;
    double Us(bench_clock::time_point A, bench_clock::time_point B) { return std::chrono::duration<double, std::micro>(B - A).count(); }
}

int main(int argc, char** argv)
{
    int N = 10000, Frames = 60;
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--entities")) N = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--frames")) Frames = std::atoi(argv[++i]);
    }

    app App;
    CHECK(App.Init(), "Init");
    App.m_bQuietTimes = true;
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;
    std::printf("%s, %d entities, %d frames\n", Device.isBindless() ? "bindless" : "non-bindless", N, Frames);

    static constexpr std::uint32_t s_Table[]         = {
        #include "xlionrender_test_table_frag.h"
    };
    static constexpr std::uint32_t s_TableBindless[] = {
        #include "xlionrender_test_table_bindless_frag.h"
    };
    xgpu::shader Frag;
    const auto Words = Device.isBindless() ? std::span<const std::uint32_t>{ s_TableBindless } : std::span<const std::uint32_t>{ s_Table };
    CHECK(!Device.Create(Frag, { .m_Type = xgpu::shader::type::bit::FRAGMENT, .m_Sharer = xgpu::shader::setup::raw_data{ { reinterpret_cast<const std::int32_t*>(Words.data()), Words.size() } } }), "table shader");

    const std::uint32_t Texels[3] = { 0xff0000ffu, 0xff000080u, 0xff000040u };
    std::array<xgpu::texture, 3> T;
    for (int i = 0; i < 3; ++i)
    {
        const auto Mips = std::array{ xgpu::texture::setup::mip_size{ sizeof(Texels[i]) } };
        CHECK(!Device.Create(T[i], { .m_Format = xgpu::texture::format::R8G8B8A8_UNORM, .m_Width = 1, .m_Height = 1, .m_MipSizes = Mips
                                   , .m_AllFacesData = { reinterpret_cast<const std::byte*>(&Texels[i]), sizeof(Texels[i]) }, .m_isGamma = false }), "texture %d", i);
    }

    using mp = xlionrender::material_param;
    const std::array Params{ mp{ .m_ID = 100, .m_Type = mp::type::FLOAT, .m_ByteOffset = 0 }, mp{ .m_ID = 101, .m_Type = mp::type::FLOAT, .m_ByteOffset = 4 }
                           , mp{ .m_ID = 200, .m_Type = mp::type::TEXTURE, .m_Slot = 0 },   mp{ .m_ID = 201, .m_Type = mp::type::TEXTURE, .m_Slot = 1 } };
    const std::array<xgpu::texture*, 3> Textures{ nullptr, &T[0], &T[1] };
    const float Block[] = { 1, 0, 0, 0 };
    const xlionrender::material_source Source{ .m_Key = 7, .m_pFragment = &Frag, .m_Textures = Textures, .m_Block = Block, .m_bTextureTable = true, .m_Params = Params };
    const auto Over = [](std::uint64_t ID, float V) { return xlionrender::param_override{ .m_ID = ID, .m_V = { V, 0, 0, 0 } }; };

    xlionrender::material_id Base;
    {
        std::vector<float> B(std::begin(Block), std::end(Block));
        CHECK(!Renderer.CreateMaterial(Base, Frag, Textures, {}, B, true), "base material");
    }

    cluster_buffers Buffers;
    CHECK(MakeStaticBuffers(Device, MakeSphere(), 0.4f, Buffers), "static geom buffers");
    xlionrender::mesh_id Mesh;
    CHECK(!Renderer.CreateClusterGeom(Mesh, Buffers.Geom()), "CreateClusterGeom");
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = 256, .m_Height = 256 }), "CreateView");
    const int Side = int(std::ceil(std::sqrt(double(N))));
    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.0f });
    Camera.LookAt(xmath::fvec3{ 0, 0, -1.3f * Side }, xmath::fvec3{ 0, 0, 0 });

    xlionrender::scene Scene;
    Scene.m_Draws.resize(N);
    for (int i = 0; i < N; ++i)
    {
        auto& D = Scene.m_Draws[i];
        D.m_Mesh     = Mesh;
        D.m_L2W      = xmath::fmat4::fromTranslation(xmath::fvec3{ (i % Side - Side / 2) * 1.0f, (i / Side - Side / 2) * 1.0f, 0 });
        D.m_Instance = std::uint64_t(i + 1);
        D.m_Materials = { Base };
    }

    // Frame cost: Render's CPU time, the scene pass's GPU time (the renderer's timestamps arrive a few frames late: the last ones)
    const auto Measure = [&](const char* pWhat, double& CpuMs, double& GpuMs) -> bool
    {
        double Cpu = 0; int nCpu = 0, nGpu = 0; GpuMs = 0;
        for (int f = 0; f < Frames + 4; ++f)
        {
            if (!App.m_Instance.ProcessInputEvents()) return false;
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) return false;
            const auto t0 = bench_clock::now();
            Renderer.Render(App.m_Window, Scene, View, Camera, {});
            const auto t1 = bench_clock::now();
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
            if (f < 4) continue;                // (warm up: instance slots written, pipelines made)
            Cpu += Us(t0, t1) / 1000.0; ++nCpu;
            xlionrender::instance::gpu_times G;
            if (Renderer.getGPUTimes(G)) { GpuMs += G.m_SceneMs; ++nGpu; }
        }
        CpuMs = Cpu / std::max(1, nCpu);
        GpuMs = nGpu ? GpuMs / nGpu : -1;
        std::printf("  frame %-22s CPU %.3f ms  GPU scene %.3f ms\n", pWhat, CpuMs, GpuMs);
        return true;
    };

    const int Mat0 = Renderer.getMaterialCount(), Pipe0 = Renderer.getPipelineCount();
    double BaseCpu = 0, BaseGpu = 0;
    std::printf("no overrides:\n");
    CHECK(Measure("(no overrides)", BaseCpu, BaseGpu), "frames");

    struct row { int Dedup; int Distinct; int Made; int Failed; double CreateUs, SteadyUs, InPlaceUs, CowUs, TexUs, CpuMs, GpuMs; int Vec4s; };
    std::vector<row> Rows;
    for (const int Dedup : { 0, 50, 99, 100 })
    {
        const int Distinct = std::max(1, Dedup == 0 ? N : Dedup == 50 ? N / 2 : Dedup == 99 ? N / 100 : 1);
        row R{ .Dedup = Dedup, .Distinct = Distinct };
        std::printf("%d%% dedup (%d distinct):\n", Dedup, Distinct);

        xlionrender::material_overrides Overrides;
        std::vector<xlionrender::material_overrides::slot> Slots(N);
        const auto ValueOf = [&](int i) { return 1.0f + float(i % Distinct); };

        // First Acquire: makes the materials
        auto t0 = bench_clock::now();
        for (int i = 0; i < N; ++i)
        {
            const std::array O{ Over(100, 0), Over(101, ValueOf(i)) };
            const auto Id = Overrides.Acquire(Renderer, Source, O, Slots[i]);
            R.Failed += !Id.isValid();
            Scene.m_Draws[i].m_Materials = { Id.isValid() ? Id : Base };
        }
        auto t1 = bench_clock::now();
        R.CreateUs = Us(t0, t1) / N;
        R.Made     = Overrides.Count();
        R.Vec4s    = R.Made * 2;            // the block (1 vec4) and the texture ids (2 slots: 1 vec4)

        // Steady: the same overrides again (what a frame asks)
        t0 = bench_clock::now();
        for (int i = 0; i < N; ++i)
        {
            const std::array O{ Over(100, 0), Over(101, ValueOf(i)) };
            (void)Overrides.Acquire(Renderer, Source, O, Slots[i]);
        }
        t1 = bench_clock::now();
        R.SteadyUs = Us(t0, t1) / N;

        CHECK(Measure("(overrides)", R.CpuMs, R.GpuMs), "frames");

        // Value change on a slot alone on its material (in place) and on one sharing it (copy on write, then back: shared again)
        constexpr int changes_v = 1000;
        {
            xlionrender::material_overrides::slot S;
            const std::array O0{ Over(100, 0), Over(101, -1.0f) };
            (void)Overrides.Acquire(Renderer, Source, O0, S);
            t0 = bench_clock::now();
            for (int k = 0; k < changes_v; ++k)
            {
                const std::array O{ Over(100, 0), Over(101, -2.0f - float(k)) };
                (void)Overrides.Acquire(Renderer, Source, O, S);
            }
            t1 = bench_clock::now();
            R.InPlaceUs = Us(t0, t1) / changes_v;

            // Texture change: slot 1's texture swapped in place
            t0 = bench_clock::now();
            for (int k = 0; k < changes_v; ++k)
            {
                const std::array O{ Over(100, 0), Over(101, -2.0f), xlionrender::param_override{ .m_ID = 201, .m_pTexture = &T[1 + (k & 1)], .m_TextureKey = std::uint64_t(2 + (k & 1)) } };
                (void)Overrides.Acquire(Renderer, Source, O, S);
            }
            t1 = bench_clock::now();
            R.TexUs = Us(t0, t1) / changes_v;
            Overrides.Release(Renderer, S);
        }
        if (Distinct < N)
        {
            // Slot 0 shares with slot Distinct: leave to a value of its own (a copy), come back (shares again: the copy goes)
            t0 = bench_clock::now();
            for (int k = 0; k < changes_v / 2; ++k)
            {
                const std::array O1{ Over(100, 0), Over(101, -5.0f) };
                (void)Overrides.Acquire(Renderer, Source, O1, Slots[0]);
                const std::array O2{ Over(100, 0), Over(101, ValueOf(0)) };
                (void)Overrides.Acquire(Renderer, Source, O2, Slots[0]);
            }
            t1 = bench_clock::now();
            R.CowUs = Us(t0, t1) / changes_v;
        }
        else R.CowUs = -1;

        std::printf("  made %d (%d failed), %d MaterialParams vec4s; per slot: create %.2f us, steady %.3f us; change: in place %.2f us, copy on write %.2f us, texture %.2f us\n"
                   , R.Made, R.Failed, R.Vec4s, R.CreateUs, R.SteadyUs, R.InPlaceUs, R.CowUs, R.TexUs);
        Rows.push_back(R);

        for (auto& S : Slots) Overrides.Release(Renderer, S);
        for (auto& D : Scene.m_Draws) D.m_Materials = { Base };
        CHECK(Overrides.Count() == 0 && Renderer.getMaterialCount() == Mat0 && Renderer.getPipelineCount() == Pipe0, "released");
    }

    double BaseCpu2 = 0, BaseGpu2 = 0;
    std::printf("no overrides, again:\n");
    CHECK(Measure("(no overrides)", BaseCpu2, BaseGpu2), "frames");

    std::printf("\n| dedup | distinct | materials made | failed | MaterialParams vec4s | create us/slot | steady us/slot | in place us | copy on write us | texture us | frame CPU ms (none %.3f) | GPU scene ms (none %.3f) |\n", BaseCpu, BaseGpu);
    for (const auto& R : Rows)
        std::printf("| %d%% | %d | %d | %d | %d | %.2f | %.3f | %.2f | %.2f | %.2f | %.3f | %.3f |\n", R.Dedup, R.Distinct, R.Made, R.Failed, R.Vec4s, R.CreateUs, R.SteadyUs, R.InPlaceUs, R.CowUs, R.TexUs, R.CpuMs, R.GpuMs);

    Renderer.DestroyMaterial(Base);
    Renderer.DestroyView(View);
    for (auto& X : T) Device.Destroy(std::move(X));
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    return 0;
}
