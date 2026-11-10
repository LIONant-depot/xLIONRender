// M0, the memory characterization lab (xLION_Docs render_leap_plan_2026-11-10.md section 2b): no editor, no level. Four tests on
// shaders/xlionrender_test_chase_comp.glsl over a permutation that is one cycle (Sattolo: no short cycles inside the working set):
//   1. one dependent chain (one invocation): the exposed latency a step, by working set
//   2. K = 1, 2, 4, 8 independent chains an invocation: what overlapping loads inside one invocation gains
//   3. the same chase at 1 .. 4096 workgroups: underfill, latency hiding and saturation told apart (steps a ns)
//   4. dense addresses (index + 1) against the chase, at the largest group count: the address pattern alone
// GPU timestamps around each dispatch, the median of 5 repeats (warm: the same buffer, run before). A table on stdout; a sanity check
// that the chase visited what it should (the consumed result is nonzero) is the only pass / fail
#include "xlionrender_test_common.h"
#include <chrono>
#include <random>
#include <numeric>

using namespace xlionrender::test;

namespace
{
    constexpr std::uint32_t s_ChaseComp[] = {
        #include "xlionrender_test_chase_comp.h"
    };

    struct lab
    {
        app&            m_App;
        xgpu::compute&  m_C;
        xgpu::buffer&   m_Next;
        xgpu::buffer&   m_Out;
        xgpu::timestamps m_T;

        // GPU ms of one dispatch (median of Reps), after a warm-up
        double Time(std::uint32_t N, std::uint32_t Steps, std::uint32_t K, std::uint32_t Dense, int Groups, int Reps = 5)
        {
            const struct { std::uint32_t m_N, m_Steps, m_K, m_Dense; } Push{ N, Steps, K, Dense };
            xgpu::buffer* Bs[] = { &m_Next, &m_Out };
            std::vector<double> Ms;
            for (int r = 0; r < Reps + 1; ++r)
            {
                for (int Frame = 0; Frame < 3; ++Frame)     // (the timestamps are read a frame or more later: the last frame's)
                {
                    while (m_App.m_Window.BeginRendering()) if (!m_App.m_Instance.ProcessInputEvents()) break;
                    m_App.m_Window.WriteTimestamp(m_T, 0);
                    m_App.m_Window.Dispatch(m_C, Bs, &Push, sizeof(Push), Groups);
                    m_App.m_Window.WriteTimestamp(m_T, 1);
                    { auto Cmd = m_App.m_Window.getCmdBuffer(); }
                    m_App.m_Window.PageFlip();
                }
                std::vector<std::uint64_t> Ns;
                if (r > 0 && m_T.getResults(Ns) && Ns.size() >= 2 && Ns[0] != ~0ull && Ns[1] != ~0ull && Ns[1] > Ns[0]) Ms.push_back(double(Ns[1] - Ns[0]) * 1e-6);
            }
            if (Ms.empty()) return -1;
            std::sort(Ms.begin(), Ms.end());
            return Ms[Ms.size() / 2];
        }
    };
}

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    xgpu::shader Shader;
    CHECK(!App.m_Device.Create(Shader, { .m_Type = xgpu::shader::type::bit::COMPUTE, .m_Sharer = xgpu::shader::setup::raw_data{ std::span{ (const std::int32_t*)s_ChaseComp, std::size(s_ChaseComp) } } }), "the chase shader");
    xgpu::compute C;
    CHECK(!App.m_Device.Create(C, { .m_pShader = &Shader, .m_nBuffers = 2, .m_PushConstantsSize = 16, .m_pName = "mem lab: chase" }), "the chase compute");

    const std::uint32_t MaxN = 1u << 26;            // (256 MB of uints: past every cache)
    const int MaxGroups = 4096;
    xgpu::buffer Next, Out;
    CHECK(!App.m_Device.Create(Out, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = 4, .m_EntryCount = MaxGroups * 64, .m_pName = "mem lab: out" }), "out");
    lab L{ App, C, Next, Out, {} };
    CHECK(!App.m_Device.Create(L.m_T, { .m_Count = 2 }), "timestamps");

    std::printf("MEMLAB (M0): median of 5, GPU timestamps around one dispatch\n");
    for (const std::uint32_t N : { 1u << 14, 1u << 18, 1u << 20, 1u << 22, MaxN })     // 64 KB, 1 MB, 4 MB, 16 MB, 256 MB
    {
        // one cycle over N (Sattolo), uploaded
        std::vector<std::uint32_t> P(N);
        std::iota(P.begin(), P.end(), 0u);
        std::mt19937 Rng(1234u);
        for (std::uint32_t i = N - 1; i > 0; --i) { std::uniform_int_distribution<std::uint32_t> D(0, i - 1); std::swap(P[i], P[D(Rng)]); }
        if (Next.m_Private) App.m_Device.Destroy(std::move(Next));
        // (device memory, uploaded once: a CPU-visible buffer would be read over the bus)
        CHECK(!App.m_Device.Create(Next, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::GPU_READ, .m_EntryByteSize = 4, .m_EntryCount = int(N), .m_pData = P.data(), .m_pName = "mem lab: next" }), "next");
        const double KB = double(N) * 4.0 / 1024.0;

        // 1. one chain, one invocation (one group; the other 63 lanes do the same chain from their own points - one warp's latency)
        const std::uint32_t S1 = 4096;
        const double t1 = L.Time(N, S1, 1, 0, 1);
        std::printf("MEMLAB %8.0f KB  1 chain, 1 group: %.1f ns a step\n", KB, t1 * 1e6 / S1);

        // 2. K chains an invocation, 1 group and 64 groups
        for (const int G : { 1, 64 })
        {
            std::printf("MEMLAB %8.0f KB  K chains, %d groups:", KB, G);
            for (const std::uint32_t K : { 1u, 2u, 4u, 8u })
            {
                const double t = L.Time(N, S1, K, 0, G);
                std::printf("  K%u %.1f ns a step (%.2f loads a ns)", K, t * 1e6 / S1, double(G) * 64.0 * K * S1 / (t * 1e6));
            }
            std::printf("\n");
        }
        // 3. groups sweep, 1 chain
        std::printf("MEMLAB %8.0f KB  groups (1 chain, 1024 steps):", KB);
        for (const int G : { 1, 8, 64, 256, 1024, 4096 })
        {
            const double t = L.Time(N, 1024, 1, 0, G);
            std::printf("  %d: %.3f ms %.2f loads/ns", G, t, double(G) * 64.0 * 1024.0 / (t * 1e6));
        }
        std::printf("\n");
        // 4. dense against the chase, 4096 groups, 1 chain
        const double tc = L.Time(N, 1024, 1, 0, MaxGroups), td = L.Time(N, 1024, 1, 1, MaxGroups);
        std::printf("MEMLAB %8.0f KB  4096 groups: chase %.3f ms, dense %.3f ms (%.1fx)\n", KB, tc, td, tc / std::max(td, 1e-9));
        std::fflush(stdout);
    }
    std::uint32_t Sum = 0;
    CHECK(Out.MemoryMap(0, 64, [&](void* p) { const auto* u = static_cast<const std::uint32_t*>(p); for (int i = 0; i < 64; ++i) Sum |= u[i]; }) == nullptr, "read out");
    CHECK(Sum != 0, "the chase produced nothing");
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
