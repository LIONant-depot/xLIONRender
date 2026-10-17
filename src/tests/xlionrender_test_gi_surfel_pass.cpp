// Part II, G2 (documentation/gi_dynamic_master_plan.md): a whole surfel pass on the GPU. shaders/xlionrender_gi_surfel_pass_comp.glsl traces
// every surfel of the unit-test scenes (the Cornell box with and without its read grid, a see-through pane under the sun and the sky, an
// emissive sign with the emitter links off) over gi::world::PackGPU's and PackGPUSurfels' data; per surfel its sums (the light, its basis
// shares, the depth moments, the back-face, unsupported and hit counts) must be the CPU's (gi::world::TraceReference). The rays per ms of
// both are printed
#include "xlionrender_test_common.h"

using namespace xlionrender::test;
using namespace xlionrender::gi;

namespace
{
    constexpr std::uint32_t s_PassComp[] = {
        #include "xlionrender_gi_surfel_pass_comp.h"
    };

    template <class T> bool MakeBuffer(xgpu::device& Device, xgpu::buffer& B, const std::vector<T>& Data, std::size_t Count = 0)
    {
        const std::size_t n = std::max<std::size_t>({ Data.size(), Count, 1 });
        if (Device.Create(B, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = int(sizeof(T)), .m_EntryCount = int(n) })) return false;
        if (Data.empty()) return true;
        return B.MemoryMap(0, int(Data.size()), [&](void* p) { std::memcpy(p, Data.data(), Data.size() * sizeof(T)); }) == nullptr;
    }

    void AddQuad(surfel_mesh& M, v3 C, v3 U, v3 V, float Split)        // (xlionrender_test_surfel_gi.cpp's)
    {
        const int nu = std::max(1, int(std::ceil(U.Length() / Split))), nv = std::max(1, int(std::ceil(V.Length() / Split)));
        const v3  N  = U.Cross(V).NormalizeCopy();
        const int Base = int(M.m_Pos.size());
        for (int j = 0; j <= nv; ++j) for (int i = 0; i <= nu; ++i) { M.m_Pos.push_back(C + U * (float(i) / nu) + V * (float(j) / nv)); M.m_Nrm.push_back(N); }
        for (int j = 0; j < nv; ++j) for (int i = 0; i < nu; ++i)
        {
            const int a = Base + j * (nu + 1) + i, b = a + 1, c = a + nu + 1, d = c + 1;
            M.m_Tris.push_back({ a, b, d }); M.m_Tris.push_back({ a, d, c });
        }
    }

    struct result { int m_Items = 0, m_Rays = 0, m_Mismatch = 0, m_CountDiff = 0; double m_Worst = 0, m_CpuRaysPerMs = 0, m_GpuRaysPerMs = 0; double m_Work[4] = {}; };

    // Every surfel of every instance (its level's), Rays each (Copies passes each: enough threads to fill the GPU), against the CPU
    int Run(app& App, xgpu::compute& Pass, world& W, const lights& L, int Rays, unsigned Seed, result& R, int Copies = 1)
    {
        R = {};
        world::gpu_scene G;
        W.PackGPU(G);
        std::vector<std::uint32_t> Data;
        W.PackGPUSurfels(Data, L);
        rng Rng(Seed);
        std::vector<world::gpu_item> Items;
        for (int i = 0; i < W.InstanceCount(); ++i)
            for (int s = 0; s < int(W.InstanceLight(i).size()); ++s) for (int c = 0; c < Copies; ++c)
                Items.push_back({ i, s, 0, Rays, Rng.Next(), Rng.Next(), int(Rng.Next() * 64), 0 });
        R.m_Items = int(Items.size()); R.m_Rays = R.m_Items * Rays;
        std::vector<world::gpu_result> Cpu;
        const auto T0 = std::chrono::steady_clock::now();
        W.TraceReference(Items, L, Cpu);
        R.m_CpuRaysPerMs = double(R.m_Rays) / std::max(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count(), 1e-6);

        xgpu::buffer Nodes, Tri4, Inst, Tlas, DataB, ItemB, OutB;
        CHECK(MakeBuffer(App.m_Device, Nodes, G.m_Nodes) && MakeBuffer(App.m_Device, Tri4, G.m_Tri4) && MakeBuffer(App.m_Device, Inst, G.m_Instances)
              && MakeBuffer(App.m_Device, Tlas, G.m_TlasLeaves) && MakeBuffer(App.m_Device, DataB, Data) && MakeBuffer(App.m_Device, ItemB, Items)
              && MakeBuffer(App.m_Device, OutB, std::vector<world::gpu_result>{}, Items.size()), "buffers");
        xgpu::timestamps Times;
        CHECK(!App.m_Device.Create(Times, { .m_Count = 2 }), "timestamps");
        const struct { int m_nItems, m_TlasRoot, m_PrimOffset, m_Pad; } Push{ R.m_Items, G.m_TlasRoot, G.m_PrimOffset, 0 };
        xgpu::buffer* Bufs[] = { &Nodes, &Tri4, &Inst, &Tlas, &DataB, &ItemB, &OutB };
        double GpuMs = 0;
        for (int Frame = 0; Frame < 4; ++Frame)
        {
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            App.m_Window.WriteTimestamp(Times, 0);
            App.m_Window.Dispatch(Pass, Bufs, &Push, sizeof(Push), world::GPUGroups(Items.size()));
            App.m_Window.WriteTimestamp(Times, 1);
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
            std::vector<std::uint64_t> Ns;
            if (Times.getResults(Ns) && Ns.size() >= 2 && Ns[0] != ~0ull && Ns[1] != ~0ull) GpuMs = double(Ns[1] - Ns[0]) * 1e-6;
        }
        R.m_GpuRaysPerMs = GpuMs > 0 ? double(R.m_Rays) / GpuMs : 0;
        std::vector<world::gpu_result> Gpu(Items.size());
        CHECK(OutB.MemoryMap(0, int(Gpu.size()), [&](void* p) { std::memcpy(Gpu.data(), p, Gpu.size() * sizeof(world::gpu_result)); }) == nullptr, "read the results");

        // Per surfel: the counts equal, the light's sums within 1e-3 of the item's light, the moments within 1e-3 of its largest. A ray
        // grazing an edge may hit another triangle (G1): a few surfels may differ
        for (std::size_t k = 0; k < Items.size(); ++k)
        {
            const auto& C = Cpu[k]; const auto& g = Gpu[k];
            for (int w = 0; w < 4; ++w) R.m_Work[w] += double(g.m_Work[w]) / std::max(R.m_Rays, 1);
            const bool bCounts = g.m_nBack == C.m_nBack && g.m_nBackAll == C.m_nBackAll && g.m_nUnsupported == C.m_nUnsupported && g.m_nHits == C.m_nHits;
            const double Ls = std::max({ double(std::abs(C.m_Sum[0])), double(std::abs(C.m_Sum[1])), double(std::abs(C.m_Sum[2])), 1e-6 });
            double Ms = 1e-6; for (const float m : C.m_Moments) Ms = std::max(Ms, double(std::abs(m)));
            double Err = 0;
            for (int a = 0; a < 3; ++a)  Err = std::max(Err, std::abs(g.m_Sum[a] - C.m_Sum[a]) / Ls);
            for (int a = 0; a < 9; ++a)  Err = std::max(Err, std::abs(g.m_Basis[a] - C.m_Basis[a]) / Ls);
            for (int a = 0; a < 48; ++a) Err = std::max(Err, std::abs(g.m_Moments[a] - C.m_Moments[a]) / Ms);
            R.m_CountDiff += !bCounts;
            if (!bCounts || Err > 1e-3) ++R.m_Mismatch;
            else R.m_Worst = std::max(R.m_Worst, Err);
        }
        return 0;
    }

    int Report(const char* pName, const result& R)
    {
        std::printf("GPU surfel pass, %s: %d surfels, %d rays; %d differ from the CPU (%d by their counts), the worst of the others %.1e; CPU %.0f rays/ms (one thread), GPU %.0f rays/ms; per ray: %.0f BVH nodes, %.2f shadow rays, %.1f read entries, %.1f probes\n",
                    pName, R.m_Items, R.m_Rays, R.m_Mismatch, R.m_CountDiff, R.m_Worst, R.m_CpuRaysPerMs, R.m_GpuRaysPerMs, R.m_Work[0], R.m_Work[1], R.m_Work[2], R.m_Work[3]);
        CHECK(R.m_Items > 0 && R.m_Mismatch * 100 <= R.m_Items, "%s: %d of %d surfels differ from the CPU", pName, R.m_Mismatch, R.m_Items);
        return 0;
    }
}

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    xgpu::shader Shader;
    CHECK(!App.m_Device.Create(Shader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = xgpu::shader::setup::raw_data{ std::span{ (const std::int32_t*)s_PassComp, std::size(s_PassComp) } } }), "the pass shader");
    xgpu::compute Pass;
    CHECK(!App.m_Device.Create(Pass, { .m_pShader = &Shader, .m_nBuffers = 7, .m_PushConstantsSize = 16 }), "the pass compute");

    // (1, 2) The Cornell box of the GI lab (the box an instance, rotated), lit by a point light, after a few updates (its surfels hold
    //        light, validity and moments to read): with its read grid (world::ReadPoint) and without (ReadVertex)
    for (const bool bGrid : { true, false })
    {
        const v3 Wh{ 0.75f, 0.75f, 0.75f }, Rd{ 0.65f, 0.05f, 0.05f }, Gr{ 0.12f, 0.6f, 0.1f };
        std::vector<surfel_mesh> M(7);
        AddQuad(M[0], { -1, -1, -1 }, { 0, 0, 2 }, { 2, 0, 0 }, 0.125f);
        AddQuad(M[1], { -1,  1, -1 }, { 2, 0, 0 }, { 0, 0, 2 }, 0.125f);
        AddQuad(M[2], { -1, -1,  1 }, { 0, 2, 0 }, { 2, 0, 0 }, 0.125f);
        AddQuad(M[3], { -1, -1, -1 }, { 2, 0, 0 }, { 0, 2, 0 }, 0.125f);
        AddQuad(M[4], { -1, -1, -1 }, { 0, 2, 0 }, { 0, 0, 2 }, 0.125f);
        AddQuad(M[5], {  1, -1, -1 }, { 0, 0, 2 }, { 0, 2, 0 }, 0.125f);
        const v3 S{ 0.5f, 0.8f, 0.6f }, Lo = S * -0.5f, X{ S.m_X, 0, 0 }, Y{ 0, S.m_Y, 0 }, Z{ 0, 0, S.m_Z };
        AddQuad(M[6], Lo, X, Z, 0.125f); AddQuad(M[6], Lo + Y, Z, X, 0.125f); AddQuad(M[6], Lo, Z, Y, 0.125f);
        AddQuad(M[6], Lo + X, Y, Z, 0.125f); AddQuad(M[6], Lo, Y, X, 0.125f); AddQuad(M[6], Lo + Z, X, Y, 0.125f);
        for (auto& m : M) { m.Bake(0.3f, 0.75f); if (bGrid) m.BuildReadGrid(); }
        world W;
        const v3 Albedo[] = { Wh, Wh, Wh, Wh, Rd, Gr };
        for (int m = 0; m < 6; ++m) W.Add(M[m], xmath::fmat4::fromIdentity(), Albedo[m]);
        W.Add(M[6], xmath::fmat4::fromTranslation({ -0.35f, -0.6f, 0.1f }) * xmath::fmat4::fromRotationY(xmath::radian{ 0.5f }), Wh);
        lights L;
        L.m_Points = { { { 0.2f, 0.4f, -0.3f }, { 1.5f, 1.5f, 1.5f }, 4.0f } };
        for (int u = 0; u < 4; ++u) W.Update(L, { 0, 0, -0.9f }, 1 << 30, 100 + u);
        result R; if (Run(App, Pass, W, L, 16, 11, R)) return 1;
        if (Report(bGrid ? "Cornell box, read grid" : "Cornell box, vertex reads", R)) return 1;
        if (bGrid) { if (Run(App, Pass, W, L, 16, 14, R, 64)) return 1; if (Report("Cornell box, read grid, 64 passes per surfel (the GPU full)", R)) return 1; }
    }
    // (3) A see-through pane over a floor and an opaque slab beside it, under the sun and a sky by direction: the rays pass the pane, the
    //     sun's shadow rays do not (as the CPU's)
    {
        surfel_mesh Floor, Pane, Slab;
        AddQuad(Floor, { -1, 0, -1 }, { 0, 0, 2 }, { 2, 0, 0 }, 0.25f);
        AddQuad(Pane, { -3, 0.3f, -3 }, { 6, 0, 0 }, { 0, 0, 6 }, 0.5f);
        AddQuad(Slab, { 1.2f, 0, -1 }, { 0, 1, 0 }, { 0, 0, 2 }, 0.25f);
        for (auto* m : { &Floor, &Pane, &Slab }) { m->Bake(0.2f, 0.6f); m->BuildReadGrid(); }
        world W;
        W.Add(Floor, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        const int iPane = W.Add(Pane, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        W.Add(Slab, xmath::fmat4::fromIdentity(), { 0.7f, 0.4f, 0.3f });
        W.SetSeeThrough(iPane, { true });
        lights L;
        L.m_SunDirection = v3{ -0.4f, -1, 0.3f }.NormalizeCopy(); L.m_SunColor = { 2, 2, 1.8f };
        L.m_bSkyOct = true;
        for (int i = 0; i < 16; ++i) L.m_SkyOct[std::size_t(i)] = v3{ 0.2f + 0.05f * float(i % 4), 0.3f, 0.2f + 0.05f * float(i / 4) };
        for (int u = 0; u < 4; ++u) W.Update(L, { 0, 1, 0 }, 1 << 30, 200 + u);
        result R; if (Run(App, Pass, W, L, 16, 12, R)) return 1;
        if (Report("see-through pane, sun and sky", R)) return 1;
    }
    // (4) An emissive sign over the ground in the dark, the emitter links off: its light comes through the rays' hits (SurfaceAt's emission)
    {
        surfel_mesh Ground, Sign;
        AddQuad(Ground, { -2, 0, -2 }, { 0, 0, 4 }, { 4, 0, 0 }, 0.5f);
        AddQuad(Sign, { -0.5f, 1, -0.5f }, { 1, 0, 0 }, { 0, 0, 1 }, 0.1f);
        Ground.Bake(0.3f, 0.9f); Sign.Bake(0.05f, 0.15f);
        world W;
        W.m_bEmitterLinks = false;
        W.Add(Ground, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        const int iS = W.Add(Sign, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        W.SetSurfelMaterial(iS, std::vector<v3>(Sign.m_Levels[0].size(), v3{ 0.3f, 0.6f, 0.3f }), std::vector<v3>(Sign.m_Levels[0].size(), v3{ 20, 20, 20 }));
        lights Dark;
        for (int u = 0; u < 4; ++u) W.Update(Dark, { 0, 1, 0 }, 1 << 30, 300 + u);
        result R; if (Run(App, Pass, W, Dark, 32, 13, R)) return 1;
        if (Report("emissive sign, no links", R)) return 1;
    }
    std::printf("PASS\n");
    return 0;
}
