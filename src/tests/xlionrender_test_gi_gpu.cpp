// Part II, G1 (documentation/gi_dynamic_master_plan.md): the GI's rays on the GPU. xgpu::compute dispatches shaders/xlionrender_gi_cast_comp.glsl
// over the structures gi::world::PackGPU writes; for random rays over a dense mesh and a scene of rotated, scaled instances its nearest hit
// (instance, triangle, distance) must be the CPU's (gi::world::Cast). The rays per ms of both are printed
#include "xlionrender_test_common.h"

using namespace xlionrender::test;
using namespace xlionrender::gi;

namespace
{
    constexpr std::uint32_t s_CastComp[] = {
        #include "xlionrender_gi_cast_comp.h"
    };

    struct gpu_ray { float m_O[4]; float m_D[4]; };
    struct gpu_hit { float m_TUV[4]; int m_Ids[4]; };

    template <class T> bool MakeBuffer(xgpu::device& Device, xgpu::buffer& B, const std::vector<T>& Data, std::size_t Count = 0)
    {
        const std::size_t n = std::max<std::size_t>({ Data.size(), Count, 1 });
        if (Device.Create(B, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = int(sizeof(T)), .m_EntryCount = int(n) })) return false;
        if (Data.empty()) return true;
        return B.MemoryMap(0, int(Data.size()), [&](void* p) { std::memcpy(p, Data.data(), Data.size() * sizeof(T)); }) == nullptr;
    }

    struct result { int m_Rays = 0, m_Hits = 0, m_Mismatch = 0; double m_WorstRel = 0, m_CpuRaysPerMs = 0, m_GpuRaysPerMs = 0; };

    int Run(app& App, xgpu::compute& Cast, const world& W, const v3& Lo, const v3& Hi, int nRays, unsigned Seed, result& R)
    {
        R = {}; R.m_Rays = nRays;
        world::gpu_scene G;
        W.PackGPU(G);
        rng Rng(Seed);
        std::vector<gpu_ray> Batch(static_cast<std::size_t>(nRays));
        for (auto& Ray : Batch)
        {
            const v3 O{ Lo.m_X + (Hi.m_X - Lo.m_X) * Rng.Next(), Lo.m_Y + (Hi.m_Y - Lo.m_Y) * Rng.Next(), Lo.m_Z + (Hi.m_Z - Lo.m_Z) * Rng.Next() };
            v3 D{ Rng.Next() * 2 - 1, Rng.Next() * 2 - 1, Rng.Next() * 2 - 1 };
            D = D.NormalizeCopy();
            const float TMax = Rng.Next() < 0.25f ? 0.5f + 4 * Rng.Next() : 1e30f;     // some short (shadow-ray-like), most unbounded
            Ray = { { O.m_X, O.m_Y, O.m_Z, TMax }, { D.m_X, D.m_Y, D.m_Z, 0 } };
        }
        // the CPU
        std::vector<world::hit> Cpu(Batch.size());
        const auto T0 = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < Batch.size(); ++i)
            Cpu[i] = W.Cast({ Batch[i].m_O[0], Batch[i].m_O[1], Batch[i].m_O[2] }, { Batch[i].m_D[0], Batch[i].m_D[1], Batch[i].m_D[2] }, Batch[i].m_O[3]);
        R.m_CpuRaysPerMs = double(nRays) / std::max(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count(), 1e-6);
        // the GPU: the same rays, one dispatch between two timestamps
        xgpu::buffer Nodes, Tri4, Inst, Tlas, RayB, HitB;
        CHECK(MakeBuffer(App.m_Device, Nodes, G.m_Nodes) && MakeBuffer(App.m_Device, Tri4, G.m_Tri4) && MakeBuffer(App.m_Device, Inst, G.m_Instances)
              && MakeBuffer(App.m_Device, Tlas, G.m_TlasLeaves) && MakeBuffer(App.m_Device, RayB, Batch) && MakeBuffer(App.m_Device, HitB, std::vector<gpu_hit>{}, Batch.size()), "buffers");
        xgpu::timestamps Times;
        CHECK(!App.m_Device.Create(Times, { .m_Count = 2 }), "timestamps");
        const struct { int m_nRays, m_TlasRoot, m_PrimOffset, m_Pad; } Push{ nRays, G.m_TlasRoot, G.m_PrimOffset, 0 };
        xgpu::buffer* Bufs[] = { &Nodes, &Tri4, &Inst, &Tlas, &RayB, &HitB };
        double GpuMs = 0;
        for (int Frame = 0; Frame < 4; ++Frame)              // (the timestamps of a frame are read a frame or more later: the last dispatch's are used)
        {
            while (App.m_Window.BeginRendering()) if (!App.m_Instance.ProcessInputEvents()) break;
            App.m_Window.WriteTimestamp(Times, 0);
            App.m_Window.Dispatch(Cast, Bufs, &Push, sizeof(Push), (nRays + 63) / 64);
            App.m_Window.WriteTimestamp(Times, 1);
            { auto Cmd = App.m_Window.getCmdBuffer(); }
            App.m_Window.PageFlip();
            std::vector<std::uint64_t> Ns;
            if (Times.getResults(Ns) && Ns.size() >= 2 && Ns[0] != ~0ull && Ns[1] != ~0ull) GpuMs = double(Ns[1] - Ns[0]) * 1e-6;
        }
        R.m_GpuRaysPerMs = GpuMs > 0 ? double(nRays) / GpuMs : 0;
        std::vector<gpu_hit> Gpu(Batch.size());
        CHECK(HitB.MemoryMap(0, int(Gpu.size()), [&](void* p) { std::memcpy(Gpu.data(), p, Gpu.size() * sizeof(gpu_hit)); }) == nullptr, "read the hits");
        for (std::size_t i = 0; i < Batch.size(); ++i)
        {
            const auto& C = Cpu[i]; const auto& H = Gpu[i];
            const bool bC = C.m_Inst >= 0, bG = H.m_Ids[0] >= 0;
            R.m_Hits += bC;
            if (bC != bG) { ++R.m_Mismatch; continue; }
            if (!bC) continue;
            const double Rel = std::abs(double(H.m_TUV[0]) - C.m_T) / std::max(double(C.m_T), 1.0);   // relative, 1 m at least: the transform's float rounding (1e-5 m) on short hits
            R.m_WorstRel = std::max(R.m_WorstRel, Rel);
            if (H.m_Ids[0] != C.m_Inst || H.m_Ids[1] != C.m_Tri) { if (Rel > 1e-4) ++R.m_Mismatch; }   // (the same distance on another triangle: a shared edge)
            else if (Rel > 1e-4) ++R.m_Mismatch;
        }
        return 0;
    }
}

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    xgpu::shader Shader;
    CHECK(!App.m_Device.Create(Shader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = xgpu::shader::setup::raw_data{ std::span{ (const std::int32_t*)s_CastComp, std::size(s_CastComp) } } }), "the cast shader");
    xgpu::compute Cast;
    CHECK(!App.m_Device.Create(Cast, { .m_pShader = &Shader, .m_nBuffers = 6, .m_PushConstantsSize = 16 }), "the cast compute");

    // Dense meshes: a 96 x 96 sphere, a rounded cube, a cylinder
    const auto Sphere = MakeSurfels(xprim_geom::uvsphere::Generate(96, 96, 1, 0.5f), 0.3f, 0.9f);
    const auto Box    = MakeSurfels(xprim_geom::rounded_cube::Generate(24, 24, 24, 0.25f), 0.3f, 0.9f);
    const auto Cyl    = MakeSurfels(xprim_geom::cylinder::Generate(16, 64, 1.0f, 0.4f, 0.4f), 0.3f, 0.9f);

    // (1) one mesh, rays from around it
    {
        world W;
        W.Add(Sphere, xmath::fmat4::fromIdentity(), { 0.5f, 0.5f, 0.5f });
        result R; if (Run(App, Cast, W, { -0.7f, -0.7f, -0.7f }, { 0.7f, 0.7f, 0.7f }, 1 << 16, 101, R)) return 1;
        std::printf("GPU rays, one mesh (%zu triangles): %d rays, %d hits; %d differ from the CPU, the worst distance %.1e (relative, 1 m at least); CPU %.0f rays/ms (one thread), GPU %.0f rays/ms\n",
                    Sphere.m_Tris.size(), R.m_Rays, R.m_Hits, R.m_Mismatch, R.m_WorstRel, R.m_CpuRaysPerMs, R.m_GpuRaysPerMs);
        CHECK(R.m_Hits > R.m_Rays / 4 && R.m_Mismatch <= R.m_Rays / 1000, "one mesh: %d of %d rays differ from the CPU", R.m_Mismatch, R.m_Rays);
    }
    // (2) a scene of 400 instances: rotated, scaled (not uniformly), of the three meshes
    {
        world W;
        rng Rng(7);
        const surfel_mesh* Meshes[3] = { &Sphere, &Box, &Cyl };
        for (int z = 0; z < 10; ++z) for (int y = 0; y < 4; ++y) for (int x = 0; x < 10; ++x)
        {
            const auto Rot = xmath::fquat(xmath::fvec3(Rng.Next() * 2 - 1, Rng.Next() * 2 - 1, Rng.Next() * 2 - 1 + 0.01f).NormalizeCopy(), xmath::radian(Rng.Next() * 6.28f));
            const xmath::fvec3 Scale{ 0.5f + Rng.Next(), 0.5f + Rng.Next(), 0.5f + Rng.Next() };
            const auto L2W = xmath::fmat4::fromTranslation({ float(x) * 2.5f, float(y) * 2.5f, float(z) * 2.5f }) * xmath::fmat4::fromRotation(Rot) * xmath::fmat4::fromScale(Scale);
            W.Add(*Meshes[(x + y + z) % 3], L2W, { 0.5f, 0.5f, 0.5f });
        }
        result R; if (Run(App, Cast, W, { -1, -1, -1 }, { 24, 9, 24 }, 1 << 16, 202, R)) return 1;
        std::printf("GPU rays, 400 instances: %d rays, %d hits; %d differ from the CPU, the worst distance %.1e (relative, 1 m at least); CPU %.0f rays/ms (one thread), GPU %.0f rays/ms\n",
                    R.m_Rays, R.m_Hits, R.m_Mismatch, R.m_WorstRel, R.m_CpuRaysPerMs, R.m_GpuRaysPerMs);
        CHECK(R.m_Hits > R.m_Rays / 4 && R.m_Mismatch <= R.m_Rays / 1000, "400 instances: %d of %d rays differ from the CPU", R.m_Mismatch, R.m_Rays);
    }
    std::printf("PASS\n");
    return 0;
}
