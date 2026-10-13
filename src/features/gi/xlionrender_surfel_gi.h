#ifndef XLIONRENDER_SURFEL_GI_H
#define XLIONRENDER_SURFEL_GI_H
#pragma once

// Diffuse GI with mesh-owned vertex surfels (research doc: "Owner discussion -> chosen direction", "Experiment 1 results").
// CPU side: the pipeline-style bake of a mesh's surfels and the solver over instances.
//
//  - surfel_mesh: a mesh's surfels (patches of its surface, mesh space: center, normal, radius), one parent level (the
//    hierarchy: LOD), the parent table, per vertex its 4 nearest surfels (indices only: weights are computed at read time
//    from distance and normal), and a BVH over its triangles. Shared by every instance of the mesh.
//  - world: instances (a mesh, a transform with uniform scale, an albedo) each holding its surfels' light at its active
//    level; a BVH over the instances (TLAS). Update() spends a ray budget: dirty instances first (moved, near a moved one,
//    reached by a changed light), then by age over distance. A surfel traces cosine rays; a hit returns its direct light
//    (sun, point lights, shadowed by the same BVHs) + the cache at the hit (one more bounce per update), a miss the sky.
//    L = D + T L, stochastic Jacobi with temporal accumulation (the handoff's §11 operator).
//  - reads (CPU here, the vertex shader later): the hit instance's level; per vertex sum(w E), sum(w) over its 4 surfels
//    (walking the parent table at level 1), interpolated over the triangle and divided (a buried vertex drops out);
//    a surfel whose rays mostly start inside geometry (> 25% back faces) is buried: weight 0 (it follows moving instances).
//
// Point lights use the renderer's falloff (xlionrender_local_lights.glsl): E = color * (1 - s^4) / (1 + 2 s^2) * cos, s = d / radius.
// Ray casts: 4-wide BVHs (binned SAH, collapsed), SIMD box and triangle tests (Embree's layout), persistent workers.
// ponytail: SSE only (scalar elsewhere: NEON next for mobile), every point light shadow-traced per hit (the
// world cells' lists when lights are many), one albedo per instance, no coarse-to-fine solve yet (experiment 1: 3x faster
// convergence after a change - next).

#include "dependencies/xmath/source/xmath.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(_M_X64) || defined(_M_IX86) || defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace xlionrender::gi
{
    using v3 = xmath::fvec3;
    constexpr float pi_v = 3.14159265f;

    template <class F> void ParallelFor(int n, F&& Fn) noexcept
    {
        const int nThreads = std::min<int>(std::max(1u, std::thread::hardware_concurrency()), std::max(n, 1));
        std::atomic<int> Next{ 0 };
        std::vector<std::thread> T;
        for (int t = 1; t < nThreads; ++t) T.emplace_back([&] { for (int i; (i = Next++) < n;) Fn(i); });
        for (int i; (i = Next++) < n;) Fn(i);
        for (auto& t : T) t.join();
    }

    // Persistent workers (spawning threads per update cost ~1 ms): Run(n, Fn) calls Fn(0..n-1) across them and the caller
    class worker_pool
    {
    public:
        explicit worker_pool(int nWorkers = int(std::max(1u, std::thread::hardware_concurrency())) - 1) noexcept
        {
            for (int t = 0; t < nWorkers; ++t) m_Threads.emplace_back([this] { Loop(); });
        }
        ~worker_pool() noexcept
        {
            { std::lock_guard Lk(m_Mutex); m_bQuit = true; }
            m_Go.notify_all();
            for (auto& t : m_Threads) t.join();
        }
        worker_pool(const worker_pool&) = delete;
        worker_pool& operator=(const worker_pool&) = delete;

        template <class F> void Run(int n, F&& Fn) noexcept
        {
            std::function<void(int)> Job = [&](int i) { Fn(i); };
            {
                std::lock_guard Lk(m_Mutex);
                m_pJob = &Job; m_N = n; m_Next = 0; m_Busy = int(m_Threads.size()); ++m_Gen;
            }
            m_Go.notify_all();
            for (int i; (i = m_Next++) < n;) Fn(i);
            std::unique_lock Lk(m_Mutex);
            m_Done.wait(Lk, [&] { return m_Busy == 0; });
            m_pJob = nullptr;
        }

    private:
        void Loop() noexcept
        {
            std::uint64_t Seen = 0;
            for (;;)
            {
                std::function<void(int)>* pJob;
                int n;
                {
                    std::unique_lock Lk(m_Mutex);
                    m_Go.wait(Lk, [&] { return m_bQuit || m_Gen != Seen; });
                    if (m_bQuit) return;
                    Seen = m_Gen; pJob = m_pJob; n = m_N;
                }
                for (int i; (i = m_Next++) < n;) (*pJob)(i);
                { std::lock_guard Lk(m_Mutex); if (--m_Busy == 0) m_Done.notify_one(); }
            }
        }

        std::vector<std::thread>    m_Threads;
        std::mutex                  m_Mutex;
        std::condition_variable     m_Go, m_Done;
        std::function<void(int)>*   m_pJob = nullptr;
        std::atomic<int>            m_Next{ 0 };
        int                         m_N = 0, m_Busy = 0;
        std::uint64_t               m_Gen = 0;
        bool                        m_bQuit = false;
    };

    // PCG (O'Neill): cheap to seed per surfel, unlike std::mt19937's 5 KB state
    struct rng
    {
        std::uint32_t m_State;
        explicit rng(std::uint32_t Seed) noexcept : m_State(Seed * 747796405u + 2891336453u) {}
        float Next() noexcept                                               // [0, 1)
        {
            m_State = m_State * 747796405u + 2891336453u;
            std::uint32_t w = ((m_State >> ((m_State >> 28u) + 4u)) ^ m_State) * 277803737u;
            w = (w >> 22u) ^ w;
            return float(w >> 8) * (1.0f / 16777216.0f);
        }
    };

    inline v3 SafeInverse(const v3& D) noexcept
    {
        const auto I = [](float x) { return 1.0f / (std::abs(x) > 1e-12f ? x : 1e-12f); };
        return { I(D.m_X), I(D.m_Y), I(D.m_Z) };
    }

    // A ray prepared for the 4-wide box test (Embree's node_intersector1: the near plane per axis from the direction's sign)
    struct ray4
    {
        float   m_O[3], m_Inv[3];
        int     m_Near[3];                  // per axis the bounds row of the near plane (lo 0 / hi 1 of that axis)
        ray4(const v3& O, const v3& D) noexcept
        {
            const v3 I = SafeInverse(D);
            for (int a = 0; a < 3; ++a) { m_O[a] = O[a]; m_Inv[a] = I[a]; m_Near[a] = 2 * a + (I[a] < 0 ? 1 : 0); }
        }
    };

    // A 4-wide BVH (Embree's BVH4): per node the bounds of 4 children in rows (lo x, hi x, lo y, hi y, lo z, hi z), tested
    // at once; built as a binary binned-SAH tree (16 bins, Wald 2007 - Jolt's TriangleSplitterBinning) collapsed to 4 wide
    // (open the child with the largest area). Leaves hold up to 4 primitives. A child code >= 0 is a node, < 0 a leaf (~id).
    struct bvh
    {
        struct alignas(16) node
        {
            float   m_B[6][4];
            int     m_Child[4];
        };
        struct leaf { int m_First, m_Count; };

        std::vector<node>   m_Nodes;
        std::vector<leaf>   m_Leaves;
        std::vector<int>    m_Prims;
        int                 m_Root = 0;
        bool                m_bEmpty = true;

        template <class F> void Build(int n, F&& Bounds)                    // Bounds(i, Lo&, Hi&)
        {
            m_Nodes.clear(); m_Leaves.clear(); m_Prims.resize(n);
            m_bEmpty = n == 0;
            if (n == 0) return;
            std::vector<v3> Lo(n), Hi(n), C(n);
            const v3 Pad{ 1e-4f, 1e-4f, 1e-4f };                            // planar primitives: zero-thickness bounds
            for (int i = 0; i < n; ++i) { Bounds(i, Lo[i], Hi[i]); Lo[i] = Lo[i] - Pad; Hi[i] = Hi[i] + Pad; C[i] = (Lo[i] + Hi[i]) * 0.5f; m_Prims[i] = i; }

            // The binary tree
            struct bnode { v3 m_Lo, m_Hi; int m_Left = -1, m_Right = -1, m_First = 0, m_Count = 0; };
            std::vector<bnode> B;
            B.reserve(2 * std::size_t(n));
            const auto Area = [](const v3& lo, const v3& hi) { const v3 d = v3::Max(hi - lo, v3{ 0, 0, 0 }); return d.m_X * d.m_Y + d.m_Y * d.m_Z + d.m_Z * d.m_X; };
            struct item { int m_Node, m_Begin, m_End; };
            std::vector<item> Stack{ { 0, 0, n } };
            B.push_back({});
            constexpr int bins_v = 16;
            while (!Stack.empty())
            {
                const item It = Stack.back(); Stack.pop_back();
                v3 BLo{ 1e30f, 1e30f, 1e30f }, BHi{ -1e30f, -1e30f, -1e30f }, CLo = BLo, CHi = BHi;
                for (int k = It.m_Begin; k < It.m_End; ++k)
                {
                    const int p = m_Prims[k];
                    BLo = v3::Min(BLo, Lo[p]); BHi = v3::Max(BHi, Hi[p]); CLo = v3::Min(CLo, C[p]); CHi = v3::Max(CHi, C[p]);
                }
                B[It.m_Node].m_Lo = BLo; B[It.m_Node].m_Hi = BHi;
                const int Count = It.m_End - It.m_Begin;
                if (Count <= 4) { B[It.m_Node].m_First = It.m_Begin; B[It.m_Node].m_Count = Count; continue; }

                // SAH over 16 bins per axis
                const v3 Ext = CHi - CLo;
                int BestAxis = -1, BestBin = 0;
                float BestCost = 1e30f;
                for (int a = 0; a < 3; ++a)
                {
                    if (Ext[a] <= 1e-6f) continue;
                    struct bin { v3 m_Lo{ 1e30f, 1e30f, 1e30f }, m_Hi{ -1e30f, -1e30f, -1e30f }; int m_N = 0; } Bins[bins_v];
                    const float Scale = bins_v / Ext[a];
                    for (int k = It.m_Begin; k < It.m_End; ++k)
                    {
                        const int p = m_Prims[k];
                        auto& b = Bins[std::min(bins_v - 1, int((C[p][a] - CLo[a]) * Scale))];
                        b.m_Lo = v3::Min(b.m_Lo, Lo[p]); b.m_Hi = v3::Max(b.m_Hi, Hi[p]); ++b.m_N;
                    }
                    float RightArea[bins_v]; int RightN[bins_v];
                    v3 RLo{ 1e30f, 1e30f, 1e30f }, RHi{ -1e30f, -1e30f, -1e30f };
                    int RN = 0;
                    for (int b = bins_v - 1; b > 0; --b)
                    {
                        RLo = v3::Min(RLo, Bins[b].m_Lo); RHi = v3::Max(RHi, Bins[b].m_Hi); RN += Bins[b].m_N;
                        RightArea[b] = Area(RLo, RHi); RightN[b] = RN;
                    }
                    v3 LLo{ 1e30f, 1e30f, 1e30f }, LHi{ -1e30f, -1e30f, -1e30f };
                    int LN = 0;
                    for (int b = 1; b < bins_v; ++b)                         // split before bin b
                    {
                        LLo = v3::Min(LLo, Bins[b - 1].m_Lo); LHi = v3::Max(LHi, Bins[b - 1].m_Hi); LN += Bins[b - 1].m_N;
                        if (LN == 0 || RightN[b] == 0) continue;
                        const float Cost = Area(LLo, LHi) * LN + RightArea[b] * RightN[b];
                        if (Cost < BestCost) { BestCost = Cost; BestAxis = a; BestBin = b; }
                    }
                }
                int Mid;
                if (BestAxis >= 0)
                {
                    const float Scale = bins_v / Ext[BestAxis];
                    Mid = int(std::partition(m_Prims.begin() + It.m_Begin, m_Prims.begin() + It.m_End, [&](int p)
                        { return std::min(bins_v - 1, int((C[p][BestAxis] - CLo[BestAxis]) * Scale)) < BestBin; }) - m_Prims.begin());
                }
                else Mid = (It.m_Begin + It.m_End) / 2;                     // every centroid in one place: halve
                const int L = int(B.size());
                B[It.m_Node].m_Left = L; B[It.m_Node].m_Right = L + 1;
                B.push_back({}); B.push_back({});
                Stack.push_back({ L, It.m_Begin, Mid });
                Stack.push_back({ L + 1, Mid, It.m_End });
            }

            // Collapse to 4 wide
            const auto Collapse = [&](auto&& Self, int b) -> int
            {
                if (B[b].m_Left < 0) { m_Leaves.push_back({ B[b].m_First, B[b].m_Count }); return ~int(m_Leaves.size() - 1); }
                int Kids[4] = { B[b].m_Left, B[b].m_Right }, nKids = 2;
                while (nKids < 4)
                {
                    int Open = -1; float Best = -1;
                    for (int k = 0; k < nKids; ++k)
                        if (B[Kids[k]].m_Left >= 0 && Area(B[Kids[k]].m_Lo, B[Kids[k]].m_Hi) > Best) { Best = Area(B[Kids[k]].m_Lo, B[Kids[k]].m_Hi); Open = k; }
                    if (Open < 0) break;
                    const int o = Kids[Open];
                    Kids[Open] = B[o].m_Left; Kids[nKids++] = B[o].m_Right;
                }
                const int Idx = int(m_Nodes.size());
                m_Nodes.push_back({});
                for (int k = 0; k < 4; ++k)
                {
                    const bool bUsed = k < nKids;
                    const v3 lo = bUsed ? B[Kids[k]].m_Lo : v3{ 1e30f, 1e30f, 1e30f }, hi = bUsed ? B[Kids[k]].m_Hi : v3{ -1e30f, -1e30f, -1e30f };
                    for (int a = 0; a < 3; ++a) { m_Nodes[Idx].m_B[2 * a][k] = lo[a]; m_Nodes[Idx].m_B[2 * a + 1][k] = hi[a]; }
                    m_Nodes[Idx].m_Child[k] = 0;
                }
                for (int k = 0; k < nKids; ++k) { const int c = Self(Self, Kids[k]); m_Nodes[Idx].m_Child[k] = c; }
                return Idx;
            };
            m_Root = Collapse(Collapse, 0);
        }

        // Which of a node's 4 children the ray enters before TMax (bit mask), and where (TNear)
        static int Box4(const node& N, const ray4& R, float TMax, float TNear[4]) noexcept
        {
#if defined(_M_X64) || defined(_M_IX86) || defined(__SSE2__)
            const __m128 ox = _mm_set1_ps(R.m_O[0]), oy = _mm_set1_ps(R.m_O[1]), oz = _mm_set1_ps(R.m_O[2]);
            const __m128 ix = _mm_set1_ps(R.m_Inv[0]), iy = _mm_set1_ps(R.m_Inv[1]), iz = _mm_set1_ps(R.m_Inv[2]);
            const __m128 nx = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(N.m_B[R.m_Near[0]]), ox), ix), fx = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(N.m_B[R.m_Near[0] ^ 1]), ox), ix);
            const __m128 ny = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(N.m_B[R.m_Near[1]]), oy), iy), fy = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(N.m_B[R.m_Near[1] ^ 1]), oy), iy);
            const __m128 nz = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(N.m_B[R.m_Near[2]]), oz), iz), fz = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(N.m_B[R.m_Near[2] ^ 1]), oz), iz);
            const __m128 tn = _mm_max_ps(_mm_max_ps(nx, ny), _mm_max_ps(nz, _mm_setzero_ps()));
            const __m128 tf = _mm_min_ps(_mm_min_ps(fx, fy), _mm_min_ps(fz, _mm_set1_ps(TMax)));
            _mm_storeu_ps(TNear, tn);
            return _mm_movemask_ps(_mm_cmple_ps(tn, tf));
#else
            int Mask = 0;
            for (int k = 0; k < 4; ++k)
            {
                float tn = 0, tf = TMax;
                for (int a = 0; a < 3; ++a)
                {
                    tn = std::max(tn, (N.m_B[R.m_Near[a]][k] - R.m_O[a]) * R.m_Inv[a]);
                    tf = std::min(tf, (N.m_B[R.m_Near[a] ^ 1][k] - R.m_O[a]) * R.m_Inv[a]);
                }
                TNear[k] = tn;
                Mask |= (tn <= tf) << k;
            }
            return Mask;
#endif
        }

        // Leaf(leaf id, TMax&) for every leaf the ray reaches before TMax, nearest first (a hit shortens TMax and culls the rest;
        // TMax < 0 ends the walk - any-hit queries)
        template <class F> void TraverseLeaves(const ray4& R, float& TMax, F&& Leaf) const
        {
            if (m_bEmpty) return;
            struct entry { int m_Code; float m_T; };
            entry Stack[96];
            int   n = 0;
            Stack[n++] = { m_Root, 0 };
            while (n)
            {
                const entry E = Stack[--n];
                if (E.m_T > TMax) continue;
                if (E.m_Code < 0) { Leaf(~E.m_Code, TMax); continue; }
                const node& N = m_Nodes[E.m_Code];
                float TNear[4];
                int Mask = Box4(N, R, TMax, TNear);
                entry Hits[4]; int h = 0;
                for (; Mask; Mask &= Mask - 1)
                {
                    const int k = std::countr_zero(unsigned(Mask));
                    entry e{ N.m_Child[k], TNear[k] };
                    int j = h++;
                    for (; j > 0 && Hits[j - 1].m_T < e.m_T; --j) Hits[j] = Hits[j - 1];   // sorted far .. near
                    Hits[j] = e;
                }
                for (int k = 0; k < h; ++k) Stack[n++] = Hits[k];          // the nearest is pushed last: popped first
            }
        }

        // Hit(prim, TMax&) per primitive of the leaves reached (the TLAS: instances)
        template <class F> void Traverse(const ray4& R, float& TMax, F&& Hit) const
        {
            TraverseLeaves(R, TMax, [&](int l, float& T) { for (int k = 0; k < m_Leaves[l].m_Count && T >= 0; ++k) Hit(m_Prims[m_Leaves[l].m_First + k], T); });
        }
    };

    // 4 triangles of a BVH leaf in rows (Embree's Triangle4v): one SIMD Moller-Trumbore for the leaf. Unused lanes are degenerate.
    struct alignas(16) tri4
    {
        float   m_V0[3][4], m_E1[3][4], m_E2[3][4];
        int     m_Tri[4];

        // The nearest lane hit before TMax (-1 none); Dist, U, V of it
        int Intersect(const v3& O, const v3& D, float TMax, float& Dist, float& U, float& V) const noexcept
        {
#if defined(_M_X64) || defined(_M_IX86) || defined(__SSE2__)
            const __m128 dx = _mm_set1_ps(D.m_X), dy = _mm_set1_ps(D.m_Y), dz = _mm_set1_ps(D.m_Z);
            const __m128 e1x = _mm_load_ps(m_E1[0]), e1y = _mm_load_ps(m_E1[1]), e1z = _mm_load_ps(m_E1[2]);
            const __m128 e2x = _mm_load_ps(m_E2[0]), e2y = _mm_load_ps(m_E2[1]), e2z = _mm_load_ps(m_E2[2]);
            const __m128 px = _mm_sub_ps(_mm_mul_ps(dy, e2z), _mm_mul_ps(dz, e2y));
            const __m128 py = _mm_sub_ps(_mm_mul_ps(dz, e2x), _mm_mul_ps(dx, e2z));
            const __m128 pz = _mm_sub_ps(_mm_mul_ps(dx, e2y), _mm_mul_ps(dy, e2x));
            const __m128 det = _mm_add_ps(_mm_add_ps(_mm_mul_ps(e1x, px), _mm_mul_ps(e1y, py)), _mm_mul_ps(e1z, pz));
            const __m128 adet = _mm_andnot_ps(_mm_set1_ps(-0.0f), det);
            const __m128 inv = _mm_div_ps(_mm_set1_ps(1.0f), det);
            const __m128 tx = _mm_sub_ps(_mm_set1_ps(O.m_X), _mm_load_ps(m_V0[0]));
            const __m128 ty = _mm_sub_ps(_mm_set1_ps(O.m_Y), _mm_load_ps(m_V0[1]));
            const __m128 tz = _mm_sub_ps(_mm_set1_ps(O.m_Z), _mm_load_ps(m_V0[2]));
            const __m128 u = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(tx, px), _mm_mul_ps(ty, py)), _mm_mul_ps(tz, pz)), inv);
            const __m128 qx = _mm_sub_ps(_mm_mul_ps(ty, e1z), _mm_mul_ps(tz, e1y));
            const __m128 qy = _mm_sub_ps(_mm_mul_ps(tz, e1x), _mm_mul_ps(tx, e1z));
            const __m128 qz = _mm_sub_ps(_mm_mul_ps(tx, e1y), _mm_mul_ps(ty, e1x));
            const __m128 v = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(dx, qx), _mm_mul_ps(dy, qy)), _mm_mul_ps(dz, qz)), inv);
            const __m128 t = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(e2x, qx), _mm_mul_ps(e2y, qy)), _mm_mul_ps(e2z, qz)), inv);
            const __m128 zero = _mm_setzero_ps();
            __m128 ok = _mm_cmpgt_ps(adet, _mm_set1_ps(1e-12f));
            ok = _mm_and_ps(ok, _mm_cmpge_ps(u, zero));
            ok = _mm_and_ps(ok, _mm_cmpge_ps(v, zero));
            ok = _mm_and_ps(ok, _mm_cmple_ps(_mm_add_ps(u, v), _mm_set1_ps(1.0f)));
            ok = _mm_and_ps(ok, _mm_cmpgt_ps(t, _mm_set1_ps(1e-5f)));
            ok = _mm_and_ps(ok, _mm_cmplt_ps(t, _mm_set1_ps(TMax)));
            int Mask = _mm_movemask_ps(ok);
            if (!Mask) return -1;
            alignas(16) float T[4], Us[4], Vs[4];
            _mm_store_ps(T, t); _mm_store_ps(Us, u); _mm_store_ps(Vs, v);
#else
            float T[4], Us[4], Vs[4];
            int Mask = 0;
            for (int k = 0; k < 4; ++k)
            {
                const v3 E1{ m_E1[0][k], m_E1[1][k], m_E1[2][k] }, E2{ m_E2[0][k], m_E2[1][k], m_E2[2][k] };
                const v3 P = D.Cross(E2);
                const float det = E1.Dot(P);
                if (std::abs(det) <= 1e-12f) continue;
                const float inv = 1.0f / det;
                const v3 Tv = O - v3{ m_V0[0][k], m_V0[1][k], m_V0[2][k] };
                Us[k] = Tv.Dot(P) * inv;
                const v3 Q = Tv.Cross(E1);
                Vs[k] = D.Dot(Q) * inv;
                T[k] = E2.Dot(Q) * inv;
                if (Us[k] >= 0 && Vs[k] >= 0 && Us[k] + Vs[k] <= 1 && T[k] > 1e-5f && T[k] < TMax) Mask |= 1 << k;
            }
            if (!Mask) return -1;
#endif
            int Best = -1;
            for (; Mask; Mask &= Mask - 1) { const int k = std::countr_zero(unsigned(Mask)); if (Best < 0 || T[k] < T[Best]) Best = k; }
            Dist = T[Best]; U = Us[Best]; V = Vs[Best];
            return Best;
        }
    };

    struct surfel
    {
        v3      m_Center, m_Normal;     // mesh space
        float   m_Radius = 0;
    };

    // Read-time weight of a surfel at a point (mesh space): distance falloff over 2 radii, times the normal match
    inline float SurfelWeight(const surfel& S, const v3& P, const v3& N) noexcept
    {
        const float c = S.m_Normal.Dot(N);
        if (c < 0.5f) return 0;
        const float f = std::max(0.0f, 1.0f - (P - S.m_Center).Length() / (2.0f * S.m_Radius));
        return f * f * c + 1e-6f;
    }

    // A hash grid of points (the bake's neighbor searches)
    struct surfel_grid
    {
        float                                               m_Cell;
        std::unordered_map<std::uint64_t, std::vector<int>> m_Cells;
        static std::uint64_t Key(int x, int y, int z) noexcept { return (std::uint64_t(std::uint32_t(x) & 0x1FFFFF) << 42) | (std::uint64_t(std::uint32_t(y) & 0x1FFFFF) << 21) | (std::uint32_t(z) & 0x1FFFFF); }
        std::array<int, 3> Cell(const v3& P) const noexcept { return { int(std::floor(P.m_X / m_Cell)), int(std::floor(P.m_Y / m_Cell)), int(std::floor(P.m_Z / m_Cell)) }; }
        void Add(const v3& P, int i) { const auto C = Cell(P); m_Cells[Key(C[0], C[1], C[2])].push_back(i); }
        template <class F> void Near(const v3& P, int Ring, F&& Fn) const
        {
            const auto C = Cell(P);
            for (int z = -Ring; z <= Ring; ++z) for (int y = -Ring; y <= Ring; ++y) for (int x = -Ring; x <= Ring; ++x)
                if (auto It = m_Cells.find(Key(C[0] + x, C[1] + y, C[2] + z)); It != m_Cells.end()) for (int i : It->second) Fn(i);
        }
    };

    struct surfel_mesh
    {
        std::vector<v3>                     m_Pos, m_Nrm;           // mesh space; triangles counter-clockwise or not (normals decide)
        std::vector<std::array<int, 3>>     m_Tris;
        std::array<std::vector<surfel>, 2>  m_Levels;               // 0: the surfels, 1: their parents
        std::vector<int>                    m_ParentOf;
        std::vector<std::array<int, 4>>     m_VertSurfels;          // per vertex: its 4 nearest level 0 surfels (-1: none)
        bvh                                 m_Bvh;
        std::vector<tri4>                   m_Tri4;                 // per BVH leaf: its triangles in rows
        v3                                  m_Lo, m_Hi;
        bool                                m_bBounce = true;       // false: receive only (the rays of the GI do not see it)

        // The 4 surfels of a level nearest P with a compatible normal
        std::array<int, 4> Nearest4(int Level, const v3& P, const v3& N) const noexcept
        {
            std::array<std::pair<float, int>, 4> Best;
            Best.fill({ 1e30f, -1 });
            const auto& L = m_Levels[Level];
            for (int s = 0; s < int(L.size()); ++s)
            {
                if (L[s].m_Normal.Dot(N) < 0.5f) continue;
                const float d = (P - L[s].m_Center).Length();
                if (d >= Best[3].first) continue;
                Best[3] = { d, s };
                std::sort(Best.begin(), Best.end());
            }
            return { Best[0].second, Best[1].second, Best[2].second, Best[3].second };
        }

        // The bake (the GeomStatic compiler's job: its surfels are saved in the geometry, see SurfelsFromXGeomStatic):
        //  - candidates sampled on the surface by area (about 4 per Radius^2 of each triangle, at least one per triangle), each with the
        //    interpolated vertex normal: the surfels follow the surface, not the vertices (a big wall gets as many as its area asks for)
        //  - Poisson-disk acceptance through a hash grid of cell Radius: a candidate is kept when no kept surfel with a similar normal
        //    (cos > 0.8) is closer than Radius (dart throwing in a shuffled order: blue-noise spacing)
        //  - per vertex its 4 nearest compatible surfels (cos >= 0.5) from the 3x3x3 cells around it (wider rings if none)
        //  - the parents: the same over the surfels' centers with ParentRadius; each surfel's parent is its nearest compatible one
        //  - the BVH and the 4-triangle rows for the rays
        // Deterministic (fixed seeds): the same mesh bakes the same surfels.
        void Bake(float Radius, float ParentRadius) noexcept
        {

            // Dart throwing over weighted candidates: the kept ones become surfels of radius R
            const auto Poisson = [](const std::vector<v3>& P, const std::vector<v3>& N, float R, std::vector<surfel>& Out, surfel_grid& G)
            {
                Out.clear();
                G = { R, {} };
                std::vector<int> Order(P.size());
                for (std::size_t i = 0; i < Order.size(); ++i) Order[i] = int(i);
                rng Shuffle(0x5EEDu);
                for (std::size_t i = Order.size(); i > 1; --i) std::swap(Order[i - 1], Order[std::size_t(Shuffle.Next() * float(i)) % i]);
                for (int i : Order)
                {
                    bool bTaken = false;
                    G.Near(P[i], 1, [&](int s) { bTaken = bTaken || ((P[i] - Out[s].m_Center).Length() < R && Out[s].m_Normal.Dot(N[i]) > 0.8f); });
                    if (bTaken) continue;
                    G.Add(P[i], int(Out.size()));
                    Out.push_back({ P[i], N[i], R });
                }
            };

            // The nearest compatible surfels of a point (up to 4), widening the search until something is found
            const auto Nearest = [](const std::vector<surfel>& L, const surfel_grid& G, const v3& P, const v3& N, int nMax)
            {
                std::array<std::pair<float, int>, 4> Best;
                Best.fill({ 1e30f, -1 });
                for (int Ring = 1; Ring <= 4 && Best[0].second < 0; ++Ring)
                    G.Near(P, Ring, [&](int s)
                    {
                        if (L[s].m_Normal.Dot(N) < 0.5f) return;
                        const float d = (P - L[s].m_Center).Length();
                        if (d >= Best[nMax - 1].first || std::any_of(Best.begin(), Best.end(), [&](const auto& b) { return b.second == s; })) return;
                        Best[nMax - 1] = { d, s };
                        std::sort(Best.begin(), Best.begin() + nMax);
                    });
                return Best;
            };

            // 1. Candidates on the surface, by area
            std::vector<v3> CP, CN;
            rng Sample(0xBA4Eu);
            const float PerArea = 4.0f / (Radius * Radius);
            for (const auto& T : m_Tris)
            {
                const v3 A = m_Pos[T[0]], B = m_Pos[T[1]], C = m_Pos[T[2]];
                const float Area = 0.5f * (B - A).Cross(C - A).Length();
                const int   n    = std::max(1, int(std::ceil(Area * PerArea)));
                for (int k = 0; k < n; ++k)
                {
                    float u = Sample.Next(), v = Sample.Next();
                    if (u + v > 1) { u = 1 - u; v = 1 - v; }
                    CP.push_back(A + (B - A) * u + (C - A) * v);
                    v3 Nn = m_Nrm[T[0]] * (1 - u - v) + m_Nrm[T[1]] * u + m_Nrm[T[2]] * v;
                    if (Nn.Length() < 1e-6f) Nn = (B - A).Cross(C - A);
                    CN.push_back(Nn.NormalizeCopy());
                }
            }

            // 2. Surfels, 3. vertex indices
            surfel_grid G0, G1;
            Poisson(CP, CN, Radius, m_Levels[0], G0);
            m_VertSurfels.resize(m_Pos.size());
            for (std::size_t v = 0; v < m_Pos.size(); ++v)
            {
                const auto B = Nearest(m_Levels[0], G0, m_Pos[v], m_Nrm[v], 4);
                m_VertSurfels[v] = { B[0].second, B[1].second, B[2].second, B[3].second };
            }

            // 4. Parents
            std::vector<v3> SP(m_Levels[0].size()), SN(m_Levels[0].size());
            for (std::size_t s = 0; s < SP.size(); ++s) { SP[s] = m_Levels[0][s].m_Center; SN[s] = m_Levels[0][s].m_Normal; }
            Poisson(SP, SN, ParentRadius, m_Levels[1], G1);
            m_ParentOf.assign(SP.size(), 0);
            for (std::size_t s = 0; s < SP.size(); ++s)
            {
                const auto B = Nearest(m_Levels[1], G1, SP[s], SN[s], 1);
                m_ParentOf[s] = std::max(B[0].second, 0);
            }

            BuildRays();
        }

        // The rays' structures (the BVH, its 4-triangle rows, the bounds) from m_Pos / m_Tris: the bake's last step, and all a geometry
        // with baked surfels needs at load
        void BuildRays() noexcept
        {
            m_Lo = { 1e30f, 1e30f, 1e30f }; m_Hi = { -1e30f, -1e30f, -1e30f };
            for (const auto& P : m_Pos) { m_Lo = v3::Min(m_Lo, P); m_Hi = v3::Max(m_Hi, P); }
            m_Bvh.Build(int(m_Tris.size()), [&](int t, v3& Lo, v3& Hi)
            {
                const auto& T = m_Tris[t];
                Lo = v3::Min(v3::Min(m_Pos[T[0]], m_Pos[T[1]]), m_Pos[T[2]]);
                Hi = v3::Max(v3::Max(m_Pos[T[0]], m_Pos[T[1]]), m_Pos[T[2]]);
            });
            m_Tri4.assign(m_Bvh.m_Leaves.size(), tri4{});
            for (std::size_t l = 0; l < m_Tri4.size(); ++l)
            {
                auto& Q = m_Tri4[l];
                for (int k = 0; k < 4; ++k)
                {
                    const bool bUsed = k < m_Bvh.m_Leaves[l].m_Count;
                    const int  t     = bUsed ? m_Bvh.m_Prims[m_Bvh.m_Leaves[l].m_First + k] : 0;
                    const v3   A     = m_Pos[m_Tris[t][0]];
                    const v3   E1    = bUsed ? m_Pos[m_Tris[t][1]] - A : v3{ 0, 0, 0 }, E2 = bUsed ? m_Pos[m_Tris[t][2]] - A : v3{ 0, 0, 0 };
                    for (int a = 0; a < 3; ++a) { Q.m_V0[a][k] = A[a]; Q.m_E1[a][k] = E1[a]; Q.m_E2[a][k] = E2[a]; }
                    Q.m_Tri[k] = t;
                }
            }
        }

        // Is anything between O and O + D TMax (mesh space)? Stops at the first hit
        bool Occluded(const v3& O, const v3& D, float TMax) const noexcept
        {
            bool bHit = false;
            m_Bvh.TraverseLeaves(ray4(O, D), TMax, [&](int l, float& TBest)
            {
                float d, u, v;
                if (m_Tri4[l].Intersect(O, D, TBest, d, u, v) >= 0) { bHit = true; TBest = -1; }   // -1: every pending node is past it
            });
            return bHit;
        }

        // The nearest triangle hit before TMax (mesh space, both faces)
        bool Cast(const v3& O, const v3& D, float& TMax, int& Tri, float& U, float& V) const noexcept
        {
            bool bHit = false;
            m_Bvh.TraverseLeaves(ray4(O, D), TMax, [&](int l, float& TBest)
            {
                float d, u, v;
                const int k = m_Tri4[l].Intersect(O, D, TBest, d, u, v);
                if (k >= 0) { TBest = d; Tri = m_Tri4[l].m_Tri[k]; U = u; V = v; bHit = true; }
            });
            return bHit;
        }
    };

    // A compiled xgeom_static's geometry (its CPU copy: positions quantized per cluster, 12-bit oct normals, 16-bit indices per cluster) into a
    // surfel_mesh, in the GPU buffer's vertex order (the vertex shader reads the surfels by gl_VertexIndex). Mirrors getVertexData of
    // shaders/xgeom_static_mb_input_full.vert. Then Bake (the GeomStatic compiler) or SurfelsFromXGeomStatic (the renderer).
    template <class T_GEOM> void FromXGeomStatic(surfel_mesh& M, const T_GEOM& Geom) noexcept
    {
        const auto Verts = Geom.getVertices();
        const auto Extra = Geom.getVertexExtras();
        const auto Index = Geom.getIndices();
        const auto Data  = Geom.getClusterData();
        M.m_Pos.assign(Verts.size(), v3{ 0, 0, 0 });
        M.m_Nrm.assign(Verts.size(), v3{ 0, 1, 0 });
        M.m_Tris.clear();
        const auto OctDecode = [](float x, float y) noexcept
        {
            x = x * 2 - 1; y = y * 2 - 1;
            v3 N{ x, y, 1 - std::abs(x) - std::abs(y) };
            if (N.m_Z < 0) N = v3{ (1 - std::abs(y)) * (x >= 0 ? 1.0f : -1.0f), (1 - std::abs(x)) * (y >= 0 ? 1.0f : -1.0f), N.m_Z };
            return N.NormalizeCopy();
        };
        for (std::size_t c = 0; c < Geom.getClusters().size(); ++c)
        {
            const auto& C = Geom.getClusters()[c];
            const auto& D = Data[c];
            for (std::uint32_t v = C.m_iVertex; v < C.m_iVertex + C.m_nVertices && v < Verts.size(); ++v)
            {
                const auto& V = Verts[v];
                const auto Q = [](std::int16_t q) { return (float(q) + 32768.0f) / 32767.5f - 1.0f; };
                M.m_Pos[v] = { Q(V.m_XPos) * D.m_PosScaleAndWPADDING.m_X + D.m_PosTrasnlationAndWPADDING.m_X
                             , Q(V.m_YPos) * D.m_PosScaleAndWPADDING.m_Y + D.m_PosTrasnlationAndWPADDING.m_Y
                             , Q(V.m_ZPos) * D.m_PosScaleAndWPADDING.m_Z + D.m_PosTrasnlationAndWPADDING.m_Z };
                const std::uint32_t Bits = std::uint16_t(V.m_Extra);
                const std::uint32_t Nx = (std::uint32_t(Extra[v].m_OctNormal[0]) << 4) | (Bits & 0xF), Ny = (std::uint32_t(Extra[v].m_OctNormal[1]) << 4) | ((Bits >> 4) & 0xF);
                M.m_Nrm[v] = OctDecode(float(Nx) / 4095.0f, float(Ny) / 4095.0f);
            }
            for (std::uint32_t i = C.m_iIndex; i + 2 < C.m_iIndex + C.m_nIndices && i + 2 < Index.size(); i += 3)
                M.m_Tris.push_back({ int(C.m_iVertex + Index[i]), int(C.m_iVertex + Index[i + 1]), int(C.m_iVertex + Index[i + 2]) });
        }
    }

    // A compiled xgeom_static's baked surfels (its GeomStatic compiler ran Bake) into a surfel_mesh already filled by FromXGeomStatic, then the
    // rays' structures. False: the geometry has none (compiled with the GI off)
    template <class T_GEOM> bool SurfelsFromXGeomStatic(surfel_mesh& M, const T_GEOM& Geom) noexcept
    {
        const auto Fine = Geom.getGISurfels(), Parents = Geom.getGIParents();
        const auto VS   = Geom.getGIVertexSurfels();
        if (Fine.empty() || VS.size() != 4 * M.m_Pos.size()) return false;
        const auto ToSurfel = [](const auto& S) { return surfel{ v3{ S.m_Center.m_X, S.m_Center.m_Y, S.m_Center.m_Z }, v3{ S.m_Normal.m_X, S.m_Normal.m_Y, S.m_Normal.m_Z }, S.m_Radius }; };
        M.m_Levels[0].clear(); M.m_Levels[1].clear(); M.m_ParentOf.clear();
        for (const auto& S : Fine)    { M.m_Levels[0].push_back(ToSurfel(S)); M.m_ParentOf.push_back(int(S.m_iParent)); }
        for (const auto& S : Parents) M.m_Levels[1].push_back(ToSurfel(S));
        M.m_VertSurfels.resize(M.m_Pos.size());
        for (std::size_t v = 0; v < M.m_Pos.size(); ++v)
            for (int k = 0; k < 4; ++k) M.m_VertSurfels[v][k] = VS[4 * v + k] == 0xFFFF ? -1 : int(VS[4 * v + k]);
        M.m_bBounce = Geom.m_GIMode != decltype(Geom.m_GIMode)::RECEIVE;
        M.BuildRays();
        return true;
    }

    struct lights
    {
        struct point
        {
            v3      m_Position;
            v3      m_Color;            // color x intensity
            float   m_Radius = 1;
        };
        v3                  m_SunDirection{ 0, -1, 0 };     // the direction the light travels
        v3                  m_SunColor{ 0, 0, 0 };
        v3                  m_Sky{ 0, 0, 0 };               // the radiance of rays that leave the scene
        std::vector<point>  m_Points;
    };

    class world
    {
    public:

        struct setup
        {
            int     m_RaysPerSurfel = 8;
            float   m_MinBlend      = 0.2f;     // temporal accumulation: 1 / n after a change, never below this
            float   m_LodDistance   = 40;       // beyond it (eye to bounds) an instance holds its parent level
            float   m_ChangeMargin  = 4;        // a moved instance dirties the instances within this distance of its bounds
        };

        struct stats
        {
            int     m_Updated   = 0;            // instances updated
            int     m_Surfels   = 0;            // surfels traced
            int     m_Rays      = 0;
            int     m_Dirty     = 0;            // instances waiting with a change
            double  m_Ms        = 0;
            double  m_TraceMs   = 0;            // the parallel part (rays); the rest is bookkeeping
        };

        struct hit
        {
            int     m_Inst  = -1, m_Tri = -1;
            float   m_T     = 1e30f, m_U = 0, m_V = 0;
        };

        setup   m_Setup;

        int Add(const surfel_mesh& Mesh, const xmath::fmat4& L2W, const v3& Albedo) noexcept
        {
            m_Instances.push_back({ &Mesh });
            auto& I = m_Instances.back();
            I.m_Albedo = Albedo;
            Place(I, L2W);
            Resize(I);
            m_bTlasDirty = true;
            return int(m_Instances.size()) - 1;
        }

        // The instance stops casting, bouncing and updating (its id is not reused)
        void Remove(int Id) noexcept
        {
            auto& I = m_Instances[Id];
            if (!I.m_bAlive) return;
            DirtyNear(I.m_Lo, I.m_Hi);
            I.m_bAlive = false;
            I.m_E.clear(); I.m_Acc.clear(); I.m_Valid.clear(); I.m_AccValid.clear();
            m_bTlasDirty = true;
        }

        bool IsAlive(int Id) const noexcept { return m_Instances[Id].m_bAlive; }

        void SetTransform(int Id, const xmath::fmat4& L2W) noexcept
        {
            auto& I = m_Instances[Id];
            DirtyNear(I.m_Lo, I.m_Hi);
            Place(I, L2W);
            DirtyNear(I.m_Lo, I.m_Hi);
            m_bTlasDirty = true;
        }

        int             InstanceCount  ( void ) const noexcept { return int(m_Instances.size()); }
        int             Level          ( int Id ) const noexcept { return m_Instances[Id].m_Level; }
        const v3&       Albedo         ( int Id ) const noexcept { return m_Instances[Id].m_Albedo; }
        // What the GPU buffer would hold for an instance: its active level's irradiance and validity per surfel
        const std::vector<v3>&    InstanceLight( int Id ) const noexcept { return m_Instances[Id].m_E; }
        const std::vector<float>& InstanceValid( int Id ) const noexcept { return m_Instances[Id].m_Valid; }

        // The nearest hit before TMax (the instances as they are now: the TLAS is rebuilt after Add / SetTransform; not
        // thread safe across that rebuild - Update rebuilds before its parallel part)
        hit Cast(const v3& O, const v3& D, float TMax = 1e30f) const noexcept
        {
            BuildTlas();
            hit H; H.m_T = TMax;
            m_Tlas.Traverse(ray4(O, D), H.m_T, [&](int t, float& TBest)
            {
                const int i = m_TlasIds[t];
                const auto& I = m_Instances[i];
                int Tri; float U, V;
                // the ray in mesh space: t is the same parameter (the direction is transformed, not normalized)
                if (I.m_pMesh->Cast(I.m_W2L * O, I.m_W2L.TransformDirection(D), TBest, Tri, U, V)) { H.m_Inst = i; H.m_Tri = Tri; H.m_U = U; H.m_V = V; H.m_T = TBest; }
            });
            return H;
        }

        // Is anything between O and O + D TMax? (shadow rays: the first hit ends it)
        bool Occluded(const v3& O, const v3& D, float TMax = 1e30f) const noexcept
        {
            BuildTlas();
            bool bHit = false;
            m_Tlas.Traverse(ray4(O, D), TMax, [&](int t, float& TBest)
            {
                if (bHit) return;
                const auto& I = m_Instances[m_TlasIds[t]];
                if (I.m_pMesh->Occluded(I.m_W2L * O, I.m_W2L.TransformDirection(D), TBest)) { bHit = true; TBest = -1; }
            });
            return bHit;
        }

        v3 HitPosition(const hit& H) const noexcept
        {
            const auto& I = m_Instances[H.m_Inst]; const auto& M = *I.m_pMesh; const auto& T = M.m_Tris[H.m_Tri];
            return I.m_L2W * (M.m_Pos[T[0]] * (1 - H.m_U - H.m_V) + M.m_Pos[T[1]] * H.m_U + M.m_Pos[T[2]] * H.m_V);
        }

        v3 HitNormal(const hit& H) const noexcept
        {
            const auto& I = m_Instances[H.m_Inst]; const auto& M = *I.m_pMesh; const auto& T = M.m_Tris[H.m_Tri];
            return NormalToWorld(I, M.m_Nrm[T[0]] * (1 - H.m_U - H.m_V) + M.m_Nrm[T[1]] * H.m_U + M.m_Nrm[T[2]] * H.m_V);
        }

        // Direct irradiance at a world point with normal N (shadowed)
        v3 Direct(const v3& P, const v3& N, const lights& L) const noexcept
        {
            v3 E{ 0, 0, 0 };
            const v3 O = P + N * 1e-3f;
            const float cs = -N.Dot(L.m_SunDirection);
            if (cs > 0 && (L.m_SunColor.m_X + L.m_SunColor.m_Y + L.m_SunColor.m_Z) > 0 && !Occluded(O, L.m_SunDirection * -1.0f)) E = E + L.m_SunColor * cs;
            if (L.m_Points.size() != m_nLightBvh) const_cast<world*>(this)->BuildLightBvh(L);      // ponytail: Direct before Update (tests)
            float Zero = 0;
            m_LightBvh.Traverse(ray4(P, v3{ 0.577f, 0.577f, 0.577f }), Zero, [&](int li, float&)
            {
                const auto& Pt = L.m_Points[li];
                const v3 D = Pt.m_Position - P;
                const float d = D.Length(), s = d / Pt.m_Radius;
                if (s >= 1 || d < 1e-6f) return;
                const float c = N.Dot(D) / d;
                if (c <= 0) return;
                if (Occluded(O, D * (1.0f / d), d - 2e-3f)) return;
                E = E + Pt.m_Color * ((1 - s * s * s * s) / (1 + 2 * s * s) * c);
            });
            return E;
        }

        // The cache's indirect irradiance at a hit: the hit instance's level through its triangle's vertices
        v3 Indirect(const hit& H) const noexcept
        {
            const auto& I = m_Instances[H.m_Inst]; const auto& M = *I.m_pMesh; const auto& T = M.m_Tris[H.m_Tri];
            const float B[3] = { 1 - H.m_U - H.m_V, H.m_U, H.m_V };
            v3 Sum{ 0, 0, 0 }; float W = 0;
            for (int k = 0; k < 3; ++k)
            {
                float w = 0;
                const v3 E = ReadVertex(I, T[k], w);
                Sum = Sum + E * B[k];
                W += w * B[k];
            }
            return W > 0 ? Sum * (1.0f / W) : v3{ 0, 0, 0 };
        }

        // One update: the light changes, the levels, then the ray budget over the instances that need it most
        stats Update(const lights& L, const v3& Eye, int RayBudget, unsigned Seed) noexcept
        {
            const auto T0 = std::chrono::steady_clock::now();
            stats St;
            BuildTlas();
            TrackLights(L);

            // Levels: the parent level beyond m_LodDistance. Finer: children start from their parent; coarser: a parent is
            // the mean of its children
            for (auto& I : m_Instances)
            {
                if (!I.m_bAlive) continue;
                const int Want = BoxDistance(I, Eye) > m_Setup.m_LodDistance ? 1 : 0;
                if (Want == I.m_Level) continue;
                const auto& M = *I.m_pMesh;
                std::vector<v3> E(M.m_Levels[Want].size(), v3{ 0, 0, 0 });
                std::vector<float> Valid(E.size(), 0);
                if (Want == 0)
                {
                    for (std::size_t s = 0; s < E.size(); ++s) { E[s] = I.m_E[M.m_ParentOf[s]]; Valid[s] = I.m_Valid[M.m_ParentOf[s]]; }
                }
                else
                {
                    std::vector<int> Cnt(E.size(), 0);
                    for (std::size_t s = 0; s < M.m_ParentOf.size(); ++s) { E[M.m_ParentOf[s]] = E[M.m_ParentOf[s]] + I.m_E[s]; Valid[M.m_ParentOf[s]] += I.m_Valid[s]; ++Cnt[M.m_ParentOf[s]]; }
                    for (std::size_t p = 0; p < E.size(); ++p) if (Cnt[p]) { E[p] = E[p] * (1.0f / float(Cnt[p])); Valid[p] /= float(Cnt[p]); }
                }
                I.m_E = std::move(E); I.m_Valid = std::move(Valid); I.m_Level = Want; I.m_Cursor = 0;
                I.m_Acc.assign(I.m_E.size(), v3{ 0, 0, 0 }); I.m_AccValid.assign(I.m_E.size(), 1);
            }

            // Pick: dirty first, then a pass in progress, then age over distance, until the budget is spent. An instance is updated in
            // slices (a cursor over its surfels): a pass over a big mesh spans several updates and the budget holds whatever one mesh
            // has. A pass starts with the blend of its own (1 / n after a change); a change in the middle of a pass starts it again.
            m_Order.clear();
            for (int i = 0; i < int(m_Instances.size()); ++i)
            {
                auto& I = m_Instances[i];
                if (!I.m_bAlive || I.m_E.empty()) continue;
                St.m_Dirty += I.m_bDirty;
                if (I.m_bDirty && I.m_Cursor != 0) I.m_Cursor = 0;
                const float Score = (I.m_bDirty ? 1e6f : 0.0f) + (I.m_Cursor ? 5e5f : 0.0f) + float(I.m_Age + 1) / (1 + BoxDistance(I, Eye) / m_Setup.m_LodDistance);
                m_Order.push_back({ Score, i });
            }
            std::sort(m_Order.begin(), m_Order.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            m_Work.clear();
            int SurfelsLeft = std::max(RayBudget / std::max(m_Setup.m_RaysPerSurfel, 1), 1);
            for (const auto& [Score, i] : m_Order)
            {
                if (SurfelsLeft <= 0) break;
                auto& I = m_Instances[i];
                const int n    = int(I.m_E.size());
                const int Take = std::min(n - I.m_Cursor, SurfelsLeft);
                if (I.m_Cursor == 0)
                {
                    if (I.m_bDirty) { I.m_History = 0; I.m_bDirty = false; }
                    I.m_PassAlpha = std::max(1.0f / float(++I.m_History), m_Setup.m_MinBlend);
                }
                for (int k = 0; k < Take; ++k) m_Work.push_back({ i, I.m_Cursor + k });
                I.m_Cursor += Take;
                if (I.m_Cursor >= n) { I.m_Cursor = 0; I.m_Age = 0; }
                SurfelsLeft -= Take;
                ++St.m_Updated;
                I.m_bPicked = true;
            }
            St.m_Surfels = int(m_Work.size());
            St.m_Rays    = St.m_Surfels * m_Setup.m_RaysPerSurfel;

            const auto T1 = std::chrono::steady_clock::now();
            m_Pool.Run(int(m_Work.size()), [&](int w)
            {
                const auto [i, s] = m_Work[w];
                auto& I = m_Instances[i];
                const auto& S = I.m_pMesh->m_Levels[I.m_Level][s];
                const v3 C = I.m_L2W * S.m_Center, N = NormalToWorld(I, S.m_Normal);
                rng Rng(Seed * 2654435761u + unsigned(w) * 7919u);
                v3 Sum{ 0, 0, 0 };
                int nBack = 0;
                const int R = m_Setup.m_RaysPerSurfel;
                for (int r = 0; r < R; ++r)
                {
                    const v3 D = CosineDir(N, Rng);
                    const hit H = Cast(C + N * 1e-3f, D);
                    if (H.m_Inst < 0) { Sum = Sum + L.m_Sky; continue; }
                    const v3 Nh = HitNormal(H);
                    if (Nh.Dot(D) > 0) { ++nBack; continue; }               // a back face: no light, and maybe buried
                    Sum = Sum + m_Instances[H.m_Inst].m_Albedo * (Direct(HitPosition(H), Nh, L) + Indirect(H)) * (1.0f / pi_v);
                }
                I.m_Acc[s]      = Sum * (pi_v / float(R));                  // E = pi mean(L), cosine sampling
                I.m_AccValid[s] = std::clamp(1.0f - (float(nBack) / float(R) - 0.1f) / 0.15f, 0.0f, 1.0f);
            });

            St.m_TraceMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T1).count();
            for (const auto& [i, s] : m_Work)
            {
                auto& I = m_Instances[i];
                I.m_E[s]     = I.m_E[s] * (1 - I.m_PassAlpha) + I.m_Acc[s] * I.m_PassAlpha;
                I.m_Valid[s] = I.m_Valid[s] * (1 - I.m_PassAlpha) + I.m_AccValid[s] * I.m_PassAlpha;
            }
            for (auto& I : m_Instances)
            {
                if (!I.m_bAlive) continue;
                if (!I.m_bPicked) ++I.m_Age;
                I.m_bPicked = false;
            }
            St.m_Ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();
            return St;
        }

        static v3 CosineDir(const v3& N, rng& Rng) noexcept
        {
            const float r = std::sqrt(Rng.Next()), phi = 2 * pi_v * Rng.Next();
            const v3 T = (std::abs(N.m_X) > 0.5f ? N.Cross(v3{ 0, 1, 0 }) : N.Cross(v3{ 1, 0, 0 })).NormalizeCopy();
            const v3 B = N.Cross(T);
            return (T * (r * std::cos(phi)) + B * (r * std::sin(phi)) + N * std::sqrt(std::max(0.0f, 1 - r * r))).NormalizeCopy();
        }

    private:

        struct instance
        {
            const surfel_mesh*      m_pMesh = nullptr;
            xmath::fmat4            m_L2W, m_W2L;
            v3                      m_Albedo, m_Lo, m_Hi;
            std::vector<v3>         m_E, m_Acc;                 // per surfel of the active level
            std::vector<float>      m_Valid, m_AccValid;
            int                     m_Level     = 0;
            int                     m_Age       = 0;
            int                     m_History   = 0;
            int                     m_Cursor    = 0;            // the next surfel of the pass in progress (0: none)
            float                   m_PassAlpha = 1;            // the blend of the pass in progress
            bool                    m_bDirty    = true;
            bool                    m_bPicked   = false;
            bool                    m_bAlive    = true;
            float                   m_MeanScale = 1;            // the transform's mean axis scale (surfel radii are mesh space)
        };

        void Place(instance& I, const xmath::fmat4& L2W) noexcept
        {
            I.m_L2W = L2W;
            I.m_W2L = L2W.InverseSRT();
            I.m_MeanScale = (L2W.TransformDirection({ 1, 0, 0 }).Length() + L2W.TransformDirection({ 0, 1, 0 }).Length() + L2W.TransformDirection({ 0, 0, 1 }).Length()) / 3;
            const auto& M = *I.m_pMesh;
            I.m_Lo = { 1e30f, 1e30f, 1e30f }; I.m_Hi = { -1e30f, -1e30f, -1e30f };
            for (int c = 0; c < 8; ++c)
            {
                const v3 P = L2W * v3{ (c & 1) ? M.m_Hi.m_X : M.m_Lo.m_X, (c & 2) ? M.m_Hi.m_Y : M.m_Lo.m_Y, (c & 4) ? M.m_Hi.m_Z : M.m_Lo.m_Z };
                I.m_Lo = v3::Min(I.m_Lo, P); I.m_Hi = v3::Max(I.m_Hi, P);
            }
            I.m_bDirty = true;
        }

        void Resize(instance& I) noexcept
        {
            const std::size_t n = I.m_pMesh->m_Levels[I.m_Level].size();
            I.m_E.assign(n, v3{ 0, 0, 0 }); I.m_Acc.assign(n, v3{ 0, 0, 0 });
            I.m_Valid.assign(n, 1); I.m_AccValid.assign(n, 1);
        }

        void DirtyNear(const v3& Lo, const v3& Hi) noexcept
        {
            const v3 M{ m_Setup.m_ChangeMargin, m_Setup.m_ChangeMargin, m_Setup.m_ChangeMargin };
            const v3 A = Lo - M, B = Hi + M;
            for (auto& I : m_Instances)
                if (I.m_bAlive && I.m_Hi.m_X >= A.m_X && I.m_Lo.m_X <= B.m_X && I.m_Hi.m_Y >= A.m_Y && I.m_Lo.m_Y <= B.m_Y && I.m_Hi.m_Z >= A.m_Z && I.m_Lo.m_Z <= B.m_Z)
                    I.m_bDirty = true;
        }

        // A changed point light dirties the instances in its sphere (old and new); a sun / sky change dirties everything
        void TrackLights(const lights& L) noexcept
        {
            const float G[] = { L.m_SunDirection.m_X, L.m_SunDirection.m_Y, L.m_SunDirection.m_Z, L.m_SunColor.m_X, L.m_SunColor.m_Y, L.m_SunColor.m_Z, L.m_Sky.m_X, L.m_Sky.m_Y, L.m_Sky.m_Z };
            if (std::memcmp(G, m_PrevGlobal, sizeof(G)) != 0) { std::memcpy(m_PrevGlobal, G, sizeof(G)); for (auto& I : m_Instances) I.m_bDirty = true; }
            const auto Key = [](const lights::point& P) { return std::array<float, 7>{ P.m_Position.m_X, P.m_Position.m_Y, P.m_Position.m_Z, P.m_Color.m_X, P.m_Color.m_Y, P.m_Color.m_Z, P.m_Radius }; };
            const std::size_t n = std::max(L.m_Points.size(), m_PrevPoints.size());
            for (std::size_t i = 0; i < n; ++i)
            {
                const bool bNew = i < L.m_Points.size(), bOld = i < m_PrevPoints.size();
                if (bNew && bOld && Key(L.m_Points[i]) == m_PrevPoints[i]) continue;
                for (int k = 0; k < 2; ++k)
                {
                    if (k == 0 ? !bOld : !bNew) continue;
                    const auto K = k == 0 ? m_PrevPoints[i] : Key(L.m_Points[i]);
                    const v3 P{ K[0], K[1], K[2] }, R{ K[6], K[6], K[6] };
                    DirtyNear(P - R, P + R);
                }
            }
            bool bMoved = m_PrevPoints.size() != L.m_Points.size();
            m_PrevPoints.resize(L.m_Points.size());
            for (std::size_t i = 0; i < L.m_Points.size(); ++i) { const auto K = Key(L.m_Points[i]); bMoved |= K != m_PrevPoints[i]; m_PrevPoints[i] = K; }
            if (bMoved || L.m_Points.size() != m_nLightBvh) BuildLightBvh(L);
        }

        void BuildLightBvh(const lights& L) noexcept
        {
            m_LightBvh.Build(int(L.m_Points.size()), [&](int i, v3& Lo, v3& Hi)
            {
                const v3 R{ L.m_Points[i].m_Radius, L.m_Points[i].m_Radius, L.m_Points[i].m_Radius };
                Lo = L.m_Points[i].m_Position - R; Hi = L.m_Points[i].m_Position + R;
            });
            m_nLightBvh = L.m_Points.size();
        }

        void BuildTlas() const noexcept
        {
            if (!m_bTlasDirty) return;
            m_TlasIds.clear();
            for (int i = 0; i < int(m_Instances.size()); ++i) if (m_Instances[i].m_bAlive && m_Instances[i].m_pMesh->m_bBounce) m_TlasIds.push_back(i);
            m_Tlas.Build(int(m_TlasIds.size()), [&](int t, v3& Lo, v3& Hi) { Lo = m_Instances[m_TlasIds[t]].m_Lo; Hi = m_Instances[m_TlasIds[t]].m_Hi; });
            m_bTlasDirty = false;
        }

        // A mesh-space normal in world space: the inverse transpose (non-uniform scale bends normals)
        static v3 NormalToWorld(const instance& I, const v3& N) noexcept
        {
            const auto& W = I.m_W2L;
            return v3{ W(0, 0) * N.m_X + W(1, 0) * N.m_Y + W(2, 0) * N.m_Z
                     , W(0, 1) * N.m_X + W(1, 1) * N.m_Y + W(2, 1) * N.m_Z
                     , W(0, 2) * N.m_X + W(1, 2) * N.m_Y + W(2, 2) * N.m_Z }.NormalizeCopy();
        }

        static float BoxDistance(const instance& I, const v3& P) noexcept
        {
            const v3 D = v3::Max(v3::Max(I.m_Lo - P, P - I.m_Hi), v3{ 0, 0, 0 });
            return D.Length();
        }

        // A vertex's weighted sum over its surfels at the instance's level (W: the total weight)
        v3 ReadVertex(const instance& I, int Vertex, float& W) const noexcept
        {
            const auto& M = *I.m_pMesh;
            const auto& Idx = M.m_VertSurfels[Vertex];
            const v3 P = M.m_Pos[Vertex], N = M.m_Nrm[Vertex];
            v3 Sum{ 0, 0, 0 }; W = 0;
            int Seen[4], nSeen = 0;
            for (int k = 0; k < 4; ++k)
            {
                if (Idx[k] < 0) continue;
                int s = Idx[k];
                if (I.m_Level == 1)
                {
                    s = M.m_ParentOf[s];
                    if (std::find(Seen, Seen + nSeen, s) != Seen + nSeen) continue;
                    Seen[nSeen++] = s;
                }
                const auto& S = M.m_Levels[I.m_Level][s];
                const float c = S.m_Normal.Dot(N);
                if (c < 0.5f) continue;
                const float f = std::max(0.0f, 1.0f - I.m_L2W.TransformDirection(P - S.m_Center).Length() / (2.0f * S.m_Radius * I.m_MeanScale));
                const float w = (f * f * c + 1e-6f) * I.m_Valid[s];
                Sum = Sum + I.m_E[s] * w;
                W += w;
            }
            return Sum;
        }

        worker_pool                             m_Pool;
        std::vector<instance>                   m_Instances;
        mutable bvh                             m_Tlas;
        mutable std::vector<int>                m_TlasIds;          // TLAS primitive -> instance (the alive ones)
        bvh                                     m_LightBvh;         // over the point lights' spheres
        std::size_t                             m_nLightBvh = ~std::size_t(0);
        mutable bool                            m_bTlasDirty = true;
        float                                   m_PrevGlobal[9] = { 1e30f };
        std::vector<std::array<float, 7>>       m_PrevPoints;
        std::vector<std::pair<float, int>>      m_Order;
        std::vector<std::pair<int, int>>        m_Work;
    };
}

#endif
