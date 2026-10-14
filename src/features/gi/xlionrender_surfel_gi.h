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
#include <cstdio>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <span>
#include <string>
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

        // The nearest lane hit before TMax (-1 none); Dist, U, V of it. Lanes: the lanes that may be hit (a bit each)
        int Intersect(const v3& O, const v3& D, float TMax, float& Dist, float& U, float& V, int Lanes = 0xF) const noexcept
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
            int Mask = _mm_movemask_ps(ok) & Lanes;
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
            Mask &= Lanes;
            if (!Mask) return -1;
#endif
            int Best = -1;
            for (; Mask; Mask &= Mask - 1) { const int k = std::countr_zero(unsigned(Mask)); if (Best < 0 || T[k] < T[Best]) Best = k; }
            Dist = T[Best]; U = Us[Best]; V = Vs[Best];
            return Best;
        }
    };

    // A surfel's depth moments (Wicked Engine's surfel GI, after DDGI's probe visibility): over its hemisphere, 4 x 4 hemi-octahedral texels
    // in mesh space, the mean and mean squared distance its rays travel (capped at its reach). A point farther than the mean in its
    // direction is probably behind something: the reads fade it (Chebyshev). Mirrored by shaders/xlionrender_surfel_gi.glsl
    constexpr int moment_res_v = 4, moment_texels_v = moment_res_v * moment_res_v;
    inline int MomentTexel(const v3& N, const v3& D) noexcept           // N: the surfel's normal, D: a direction (both mesh space, unit)
    {
        const v3 T = (std::abs(N.m_X) < 0.9f ? v3{ 1, 0, 0 } : v3{ 0, 1, 0 }).Cross(N).NormalizeCopy(), B = N.Cross(T);
        const float x = D.Dot(T), y = D.Dot(B), z = std::max(D.Dot(N), 0.0f);
        const float k = 1.0f / std::max(std::abs(x) + std::abs(y) + z, 1e-6f);
        const float u = (x + y) * k, v = (x - y) * k;                       // the hemi-octahedron's diamond into [-1, 1]^2
        const int   tu = std::clamp(int((u * 0.5f + 0.5f) * moment_res_v), 0, moment_res_v - 1), tv = std::clamp(int((v * 0.5f + 0.5f) * moment_res_v), 0, moment_res_v - 1);
        return tv * moment_res_v + tu;
    }
    using moments = std::array<float, 2 * moment_texels_v>;            // per texel: mean, mean^2 (mesh units)

    // The Half-Life 2 basis (Valve's radiosity normal mapping, Mitchell et al. 2006), on a surfel: 3 directions of its hemisphere, in
    // its tangent frame (the one of MomentTexel), each a color of the light from around it. A surfel sees a hemisphere only: 3 colors
    // cover it where spherical harmonics spend half their coefficients behind the surface. A normal reads them weighted by its
    // squared cosines to them (normalized): the surfel's own normal reads their mean, a normal-mapped one leans toward its side.
    // Mirrored by shaders/xlionrender_surfel_gi.glsl
    using basis = std::array<v3, 3>;
    inline basis BasisDirs(const v3& N) noexcept
    {
        const v3 T = (std::abs(N.m_X) < 0.9f ? v3{ 1, 0, 0 } : v3{ 0, 1, 0 }).Cross(N).NormalizeCopy(), B = N.Cross(T);
        constexpr float a = 0.40824829f, b = 0.70710678f, c = 0.57735027f, d = 0.81649658f;     // 1/sqrt6, 1/sqrt2, 1/sqrt3, sqrt(2/3)
        return { T * -a + B * b + N * c, T * -a + B * -b + N * c, T * d + N * c };
    }
    inline std::array<float, 3> BasisWeights(const basis& Dirs, const v3& n) noexcept
    {
        std::array<float, 3> w;
        float Sum = 0;
        for (int i = 0; i < 3; ++i) { const float c = std::max(Dirs[i].Dot(n), 0.0f); w[i] = c * c; Sum += w[i]; }
        for (auto& x : w) x = Sum > 1e-12f ? x / Sum : 1.0f / 3;
        return w;
    }

    // A surfel's light over time (Wicked Engine's MultiscaleMeanEstimator, MIT): a short-window mean and the variance of the samples
    // around it; the long mean follows the samples at a rate that falls as low as 1/256 while they agree with it (within their own
    // noise: the light settles, no swimming) and rises toward 1 when they move away from it beyond that noise (a light changed: it
    // catches up). Samples far above the noise are clipped (fireflies).
    struct estimator
    {
        v3      m_Mean{ 0, 0, 0 }, m_Short{ 0, 0, 0 }, m_Variance{ 0, 0, 0 };
        float   m_Vbbr          = 1;        // the variance-based blend reduction
        float   m_Inconsistency = 0;

        void Seed(const v3& E) noexcept { m_Mean = m_Short = E; m_Variance = { 0, 0, 0 }; m_Vbbr = 1; m_Inconsistency = 0; }

        void Add(v3 y, float ShortBlend = 0.08f) noexcept
        {
            const auto Luma = [](const v3& c) { return 0.299f * c.m_X + 0.587f * c.m_Y + 0.114f * c.m_Z; };
            const auto Dev  = [&] { return v3{ std::sqrt(std::max(1e-5f, m_Variance.m_X)), std::sqrt(std::max(1e-5f, m_Variance.m_Y)), std::sqrt(std::max(1e-5f, m_Variance.m_Z)) }; };
            {
                const v3 D = Dev(), Hi = m_Short + D * 8 + v3{ 0.1f, 0.1f, 0.1f };
                y = v3::Min(y, Hi);
            }
            const v3 Delta = y - m_Short;
            m_Short = m_Short + (y - m_Short) * ShortBlend;
            const v3 Delta2 = y - m_Short;
            m_Variance = m_Variance + (v3{ Delta.m_X * Delta2.m_X, Delta.m_Y * Delta2.m_Y, Delta.m_Z * Delta2.m_Z } - m_Variance) * (ShortBlend * 0.5f);
            const v3 D = Dev(), Diff = m_Mean - m_Short;
            const float Relative = Luma(v3{ std::abs(Diff.m_X) / D.m_X, std::abs(Diff.m_Y) / D.m_Y, std::abs(Diff.m_Z) / D.m_Z });
            m_Inconsistency += (Relative - m_Inconsistency) * 0.08f;
            const float Vbbr = std::clamp(Luma(v3{ m_Short.m_X / D.m_X, m_Short.m_Y / D.m_Y, m_Short.m_Z / D.m_Z }) * 0.5f, 1.0f / 32, 1.0f);
            const float x = std::clamp(Relative * std::max(0.02f, m_Inconsistency - 0.2f), 0.0f, 1.0f);
            const float CatchUp = std::clamp(x * x * (3 - 2 * x), 1.0f / 256, 1.0f) * m_Vbbr;
            m_Vbbr += (Vbbr - m_Vbbr) * 0.1f;
            m_Mean = m_Mean + (y - m_Mean) * std::clamp(CatchUp, 0.0f, 1.0f);
        }
    };
    constexpr int warmup_passes_v = 16;     // after a change: a running average (1 / n) for this many passes, the estimator after

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
        std::vector<std::array<float, 2>>   m_UV;                   // per vertex (optional): where the surfels sample their material
        std::vector<std::uint16_t>          m_TriMaterial;          // per triangle (optional): its material slot

        // Where each fine surfel samples its material (Bake; its color is the material there, averaged over its footprint): the slot,
        // the UV of its center, the UV radius of its footprint (its radius through its triangle's UV density)
        struct surfel_material { float m_U = 0, m_V = 0, m_UVRadius = 0; std::uint32_t m_iMaterial = 0; };
        std::vector<surfel_material>        m_SurfelMaterial;
        std::array<std::vector<surfel>, 2>  m_Levels;               // 0: the surfels, 1: their parents
        std::vector<int>                    m_ParentOf;
        std::vector<std::array<int, 4>>     m_VertSurfels;          // per vertex: its 4 nearest level 0 surfels (-1: none)
        bvh                                 m_Bvh;
        std::vector<tri4>                   m_Tri4;                 // per BVH leaf: its triangles in rows
        v3                                  m_Lo, m_Hi;
        bool                                m_bBounce = true;       // false: receive only (the rays of the GI do not see it)

        // The per-pixel read (BuildReadGrid): a sparse grid in mesh space, cells of the fine surfels' radius, only the cells the surface
        // crosses. Per cell its patches (the surface in it facing one way in one plane), per patch the fine surfels whose reach (2 radii)
        // touches the cell and that it sees: a point reads its own cell's patch that matches it, no neighbor search. On the GPU an
        // open-addressing hash table (ReadKey / ReadHash, mirrored by shaders/xlionrender_surfel_gi.glsl) of { key low, key high, first,
        // count } into m_ReadList (the patches, then their lists).
        float                                       m_ReadCell = 0;
        std::vector<std::array<std::uint32_t, 4>>   m_ReadTable;    // a power of 2 entries; key high 0xFFFFFFFF: empty
        std::vector<std::uint32_t>                  m_ReadList;
        std::uint32_t                               m_ReadMaxProbe = 0;     // the longest probe of the table (the GPU's lookups go that far, no further)
        std::uint32_t                               m_ReadUnsupported = 0;  // patches whose list is empty (none of their surfels in sight: their reads are unsupported)

        static std::array<int, 3> ReadCellOf(const v3& P, float Cell) noexcept { return { int(std::floor(P.m_X / Cell)), int(std::floor(P.m_Y / Cell)), int(std::floor(P.m_Z / Cell)) }; }
        static std::array<std::uint32_t, 2> ReadKey(const std::array<int, 3>& C) noexcept
        {
            const std::uint64_t K = surfel_grid::Key(C[0], C[1], C[2]);
            return { std::uint32_t(K), std::uint32_t(K >> 32) };
        }
        static std::uint32_t ReadHash(const std::array<int, 3>& C) noexcept { return (std::uint32_t(C[0]) * 73856093u) ^ (std::uint32_t(C[1]) * 19349663u) ^ (std::uint32_t(C[2]) * 83492791u); }

        // The cell of a surface point: lifted a little off its surface first (read_lift_v of a cell along N). A floor at a multiple of the
        // cell size lies on a boundary, and the GPU's interpolated position falls a hair to either side of it: lifted, it does not
        static constexpr float read_lift_v = 0.05f;
        static std::array<int, 3> ReadCellAt(const v3& P, const v3& N, float Cell) noexcept { return ReadCellOf(P + N * (read_lift_v * Cell), Cell); }

        // A list entry: the surfel (low 16 bits) and how much the surface sees it at the 4 corners of the cell in the patch's plane (4 bits
        // each, 0..15): at a point, blended by its place in the cell (as GIReadWeight)
        static std::uint32_t ReadEntrySurfel(std::uint32_t E) noexcept { return E & 0xFFFFu; }
        static std::array<int, 3> ReadAxes(const v3& N) noexcept             // the 2 axes across a plane, then the one along its normal
        {
            const float x = std::abs(N.m_X), y = std::abs(N.m_Y), z = std::abs(N.m_Z);
            return x >= y && x >= z ? std::array<int, 3>{ 1, 2, 0 } : y >= z ? std::array<int, 3>{ 0, 2, 1 } : std::array<int, 3>{ 0, 1, 2 };
        }
        float ReadEntryWeight(std::uint32_t E, const v3& P, const v3& N, const v3& PatchN) const noexcept
        {
            const v3 Lp = P + N * (read_lift_v * m_ReadCell);
            const auto [Au, Av, Aa] = ReadAxes(PatchN);
            const float* pL = &Lp.m_X;
            const float fu = pL[Au] / m_ReadCell - std::floor(pL[Au] / m_ReadCell), fv = pL[Av] / m_ReadCell - std::floor(pL[Av] / m_ReadCell);
            const auto W = [&](int k) { return float((E >> (16 + 4 * k)) & 15u); };
            return ((W(0) * (1 - fu) + W(1) * fu) * (1 - fv) + (W(2) * (1 - fu) + W(3) * fu) * fv) / 15.0f;
        }

        // The fine surfels listed for the surface point P (normal N): its cell's patch that matches it best (facing its way, nearest its
        // plane; as GIReadCell); empty: P is off the surface's cells
        std::span<const std::uint32_t> ReadCellList(const v3& P, const v3& N) const noexcept
        {
            if (m_ReadTable.empty()) return {};
            const auto C = ReadCellAt(P, N, m_ReadCell);
            const auto K = ReadKey(C);
            const std::uint32_t Mask = std::uint32_t(m_ReadTable.size() - 1);
            for (std::uint32_t h = ReadHash(C) & Mask, n = 0; n <= Mask; h = (h + 1) & Mask, ++n)
            {
                const auto& E = m_ReadTable[h];
                if (E[1] == 0xFFFFFFFFu) return {};
                if (E[0] != K[0] || E[1] != K[1]) continue;
                float Best = -1e30f; std::uint32_t Bp = 0xFFFFFFFFu;
                for (std::uint32_t p = 0; p < E[3]; ++p)
                {
                    const std::uint32_t* Pa = &m_ReadList[E[2] + 4 * p];
                    const v3 Np = UnpackOct(Pa[0]);
                    const float c = Np.Dot(N);
                    if (c < 0.5f) continue;
                    const float Score = c - std::abs(P.Dot(Np) - std::bit_cast<float>(Pa[1])) / m_ReadCell;
                    if (Score > Best) { Best = Score; Bp = p; }
                }
                if (Bp == 0xFFFFFFFFu) return {};
                const std::uint32_t* Pa = &m_ReadList[E[2] + 4 * Bp];
                return { m_ReadList.data() + Pa[2], Pa[3] };
            }
            return {};
        }

        // A surface point with something just in front of it (within a quarter of a read cell along its normal: a wall under a door
        // frame): never seen (BuildReadGrid's points, Coverage)
        bool CoveredAt(const v3& P, const v3& N) const noexcept { return Occluded(P + N * (0.02f * m_ReadCell), N, 0.25f * m_ReadCell); }

        // How much of the surface the per-pixel read covers: random points of every triangle, by its area (the interpolated vertex normal, a hair off
        // the surface as the GPU's interpolation is), each read through its cell: uncovered when no listed surfel reaches it with a
        // compatible normal (black in the SURFELS debug view). The first uncovered points are kept, with what is near them
        struct coverage
        {
            struct example { v3 m_P, m_N; int m_nList; float m_NearestCompatible, m_NearestAny; };
            int                     m_nPoints = 0, m_nUncovered = 0, m_nHidden = 0;    // m_nHidden: covered (CoveredAt), never seen: not counted
            std::vector<example>    m_Examples;
        };
        coverage Coverage(int PerTriangle, int nExamples = 8) const noexcept
        {
            coverage Out;
            rng R(0xC0FEu);
            const float PerArea = m_ReadCell > 0 ? 16.0f / (m_ReadCell * m_ReadCell) : 0;     // and 16 per cell of area: long thin strips get theirs
            for (const auto& T : m_Tris)
            {
                const v3 A = m_Pos[T[0]], B = m_Pos[T[1]], C = m_Pos[T[2]];
                const int n = std::min(4096, std::max(PerTriangle, int(0.5f * (B - A).Cross(C - A).Length() * PerArea)));
                for (int k = 0; k < n; ++k, ++Out.m_nPoints)
                {
                    float u = R.Next(), v = R.Next();
                    if (u + v > 1) { u = 1 - u; v = 1 - v; }
                    v3 N = m_Nrm[T[0]] * (1 - u - v) + m_Nrm[T[1]] * u + m_Nrm[T[2]] * v;
                    { const v3 F = (B - A).Cross(C - A); if (F.Length() > 1e-12f) N = (F.Dot(N) < 0 ? F * -1.0f : F).NormalizeCopy(); else N = N.NormalizeCopy(); }     // the face's (as the pixels')
                    const v3 P = A + (B - A) * u + (C - A) * v + v3{ 1e-6f, -1e-6f, 1e-6f } * float((k % 3) - 1);
                    if (CoveredAt(P, N)) { ++Out.m_nHidden; --Out.m_nPoints; continue; }
                    const auto List = ReadCellList(P, N);
                    bool bCovered = false;
                    for (auto e : List)
                    {
                        const auto& S = m_Levels[0][ReadEntrySurfel(e)];
                        bCovered = bCovered || (S.m_Normal.Dot(N) >= 0.5f && (P - S.m_Center).Length() < 2 * S.m_Radius);
                    }
                    if (bCovered) continue;
                    ++Out.m_nUncovered;
                    if (int(Out.m_Examples.size()) >= nExamples) continue;
                    float Nc = 1e30f, Na = 1e30f;
                    for (const auto& S : m_Levels[0]) { const float d = (P - S.m_Center).Length(); Na = std::min(Na, d); if (S.m_Normal.Dot(N) >= 0.5f) Nc = std::min(Nc, d); }
                    Out.m_Examples.push_back({ P, N, int(List.size()), Nc, Na });
                }
            }
            return Out;
        }

        void BuildReadGrid() noexcept
        {
            m_ReadTable.clear(); m_ReadList.clear(); m_ReadMaxProbe = 0; m_ReadUnsupported = 0;
            const auto& L = m_Levels[0];
            if (L.empty()) return;
            m_ReadCell = L[0].m_Radius;
            const float Cell = m_ReadCell, Reach = 2 * L[0].m_Radius;

            // The cells the surface crosses (points on every triangle no further apart than half a cell), and per cell its patches: the
            // pieces of surface in it that face one way in one plane (the blinds and the window frame 10 cm behind them are two patches,
            // whichever way they face), each with up to 4 of its points (the visibility test below)
            struct patch { v3 m_N; float m_D; std::vector<v3> m_Pts, m_Hidden; };     // its points in sight, its covered ones
            struct cell  { std::array<int, 3> m_C; std::vector<patch> m_Patches; };
            std::unordered_map<std::uint64_t, cell> Cells;
            const float PlaneTol = 0.1f * Cell;
            // A point covered by something just in front of it (a wall under a door frame) is never seen: it votes on what the surface sees
            // only where no point in sight is near (it filled a patch's points under the frame: the visible strip beside it saw nothing, an
            // empty list, a black bar). A face covered all over (sealed in) still has its covered points
            const auto Covered = [&](const v3& P, const v3& N) { return CoveredAt(P, N); };
            const auto Room = [&](const std::vector<v3>& Pts, const v3& P)       // up to 4, apart
            {
                if (Pts.size() >= 4) return false;
                for (const auto& Q : Pts) if ((Q - P).Length() < 0.3f * Cell) return false;
                return true;
            };
            const auto Touch = [&](const v3& P, const v3& N)
            {
                const auto C  = ReadCellAt(P, N, Cell);
                auto&      Ce = Cells.try_emplace(surfel_grid::Key(C[0], C[1], C[2]), cell{ C, {} }).first->second;
                patch* pPa = nullptr;
                for (auto& Pa : Ce.m_Patches) if (Pa.m_N.Dot(N) > 0.97f && std::abs(P.Dot(Pa.m_N) - Pa.m_D) < PlaneTol) { pPa = &Pa; break; }
                if (pPa == nullptr) { Ce.m_Patches.push_back({ N, P.Dot(N), {}, {} }); pPa = &Ce.m_Patches.back(); }
                const bool bVis = Room(pPa->m_Pts, P), bHid = pPa->m_Pts.empty() && Room(pPa->m_Hidden, P);     // (no ray when it is kept by neither)
                if (!bVis && !bHid) return;
                if (!Covered(P, N)) { if (bVis) pPa->m_Pts.push_back(P); }
                else if (bHid) pPa->m_Hidden.push_back(P);
            };
            for (const auto& T : m_Tris)
            {
                const v3 A = m_Pos[T[0]], B = m_Pos[T[1]], Cc = m_Pos[T[2]];
                const float Longest = std::max({ (B - A).Length(), (Cc - B).Length(), (A - Cc).Length() });
                const int   k       = std::min(4096, std::max(1, int(std::ceil(Longest / (0.5f * Cell)))));     // every cell it crosses (a 60 m floor: 240)
                for (int i = 0; i <= k; ++i) for (int j = 0; i + j <= k; ++j)
                {
                    const float u = float(i) / k, w = float(j) / k;
                    v3 N = m_Nrm[T[0]] * (1 - u - w) + m_Nrm[T[1]] * u + m_Nrm[T[2]] * w;
                    { const v3 F = (B - A).Cross(Cc - A); if (F.Length() > 1e-12f) N = (F.Dot(N) < 0 ? F * -1.0f : F).NormalizeCopy(); else N = N.NormalizeCopy(); }     // the face's (as the pixels')
                    Touch(A + (B - A) * u + (Cc - A) * w, N);
                }
            }

            // Per patch the surfels whose reach touches its cell's box, facing its way, weighted by how much of it sees them: its points and
            // the same patch's in the 6 neighbor cells (Sees: the bake's test). Without it a pixel of blinds read the surfels of the window frame
            // behind them (dark blotches): the depth moments alone did not stop them (seen from behind, the pixel lies sideways, along the
            // gap). Per patch, not per cell: from the frame's points the frame's surfels are in plain sight. The neighbors' points make the
            // test generous and shared: a surfel leaves a list where the surface around no longer sees it, not at a cell's edge (squares)
            surfel_grid G{ Cell, {} };
            for (int s = 0; s < int(L.size()); ++s) G.Add(L[s].m_Center, s);
            const int Ring = int(std::ceil(Reach / Cell));
            std::vector<const cell*> Order;
            for (const auto& [Key, Ce] : Cells) Order.push_back(&Ce);
            std::vector<std::vector<std::vector<std::uint32_t>>> Lists(Order.size());
            std::atomic<std::uint32_t> Unsupported{ 0 };
            ParallelFor(int(Order.size()), [&](int c)
            {
                const auto& Ce = *Order[c];
                Lists[c].resize(Ce.m_Patches.size());
                const v3 Lo{ Ce.m_C[0] * Cell, Ce.m_C[1] * Cell, Ce.m_C[2] * Cell }, Hi = Lo + v3{ Cell, Cell, Cell };
                std::vector<int> Near;
                G.Near((Lo + Hi) * 0.5f, Ring, [&](int s) { const v3 Q = v3::Max(Lo, v3::Min(Hi, L[s].m_Center)); if ((Q - L[s].m_Center).Length() < Reach) Near.push_back(s); });
                static constexpr int Around[7][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
                for (std::size_t p = 0; p < Ce.m_Patches.size(); ++p)
                {
                    const auto& Pa = Ce.m_Patches[p];
                    std::vector<const v3*> Pts;                                 // the same patch's points around (the 26 neighbor cells too)
                    for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
                    {
                        const auto It = Cells.find(surfel_grid::Key(Ce.m_C[0] + dx, Ce.m_C[1] + dy, Ce.m_C[2] + dz));
                        if (It == Cells.end()) continue;
                        for (const auto& Pb : It->second.m_Patches)
                            if (Pb.m_N.Dot(Pa.m_N) > 0.97f && std::abs(Pb.m_D - Pa.m_D) < PlaneTol) for (const auto& Q : Pb.m_Pts) Pts.push_back(&Q);
                    }
                    if (Pts.empty())                                            // no point of it in sight around: its covered ones (a sealed face)
                        for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
                        {
                            const auto It = Cells.find(surfel_grid::Key(Ce.m_C[0] + dx, Ce.m_C[1] + dy, Ce.m_C[2] + dz));
                            if (It == Cells.end()) continue;
                            for (const auto& Pb : It->second.m_Patches)
                                if (Pb.m_N.Dot(Pa.m_N) > 0.97f && std::abs(Pb.m_D - Pa.m_D) < PlaneTol) for (const auto& Q : Pb.m_Hidden) Pts.push_back(&Q);
                        }
                    // The cell's 4 corners in the patch's plane (ReadAxes: the 2 axes across it), and per corner the patch's points nearest it
                    // (up to 4, within 3/4 of a cell): the same points whichever cell asks, so neighboring cells agree on their shared corners
                    const auto [Au, Av, Aa] = ReadAxes(Pa.m_N);
                    std::array<std::vector<const v3*>, 4> CornerPts;
                    for (int k = 0; k < 4; ++k)
                    {
                        v3 K;
                        float* pK = &K.m_X;
                        pK[Au] = float(Ce.m_C[Au] + (k & 1)) * Cell; pK[Av] = float(Ce.m_C[Av] + (k >> 1)) * Cell;
                        const float* pN = &Pa.m_N.m_X;
                        pK[Aa] = (Pa.m_D - pN[Au] * pK[Au] - pN[Av] * pK[Av]) / pN[Aa];
                        std::vector<std::pair<float, const v3*>> By;
                        for (const v3* Q : Pts) { const float d = (*Q - K).Length(); if (d < 0.75f * Cell) By.push_back({ d, Q }); }
                        std::sort(By.begin(), By.end(), [](const auto& A, const auto& B) { return A.first < B.first || (A.first == B.first && A.second < B.second); });
                        for (std::size_t j = 0; j < By.size() && j < 4; ++j) CornerPts[k].push_back(By[j].second);
                    }
                    // Each facing surfel with how much of the surface around each corner sees it: the fraction of the corner's points in its
                    // sight, cubed (as the depth moments' fade), 4 bits a corner (the read blends them across the cell: ReadEntryWeight). Seen
                    // through the gaps of blinds by a few points: next to nothing; in plain sight: all. A weight, not a cut: no holes, no edges
                    for (int s : Near)
                    {
                        if (L[s].m_Normal.Dot(Pa.m_N) < 0.5f) continue;
                        // A corner with no surface near it knows nothing: it takes the mean of the corners that do (it was: in plain sight -
                        // the surfels the surface does not see leaked in across the cell)
                        std::array<float, 4> Frac{};
                        float Known = 0; int nKnown = 0;
                        for (int k = 0; k < 4; ++k)
                        {
                            if (CornerPts[k].empty()) { Frac[k] = -1; continue; }
                            int nSee = 0;
                            for (const v3* Q : CornerPts[k]) nSee += Sees(*Q, Pa.m_N, L[s]);
                            Frac[k] = float(nSee) / float(CornerPts[k].size());
                            Known += Frac[k]; ++nKnown;
                        }
                        std::uint32_t E = std::uint32_t(s);
                        for (int k = 0; k < 4; ++k)
                        {
                            const float F = Frac[k] >= 0 ? Frac[k] : nKnown ? Known / float(nKnown) : 0.0f;
                            E |= std::uint32_t(std::lround(F * F * F * 15.0f)) << (16 + 4 * k);
                        }
                        Lists[c][p].push_back(E);
                    }
                    // Out of sight everywhere: not listed. None in sight (a face tucked in, or all in reach behind something): an empty list - the
                    // read there is unsupported (GIQueryResult: the vertex read, else no GI), not filled from surfels it does not see
                    std::erase_if(Lists[c][p], [](std::uint32_t E) { return (E >> 16) == 0; });
                    if (Lists[c][p].empty()) Unsupported.fetch_add(1, std::memory_order_relaxed);
                }
            });

            // The table: per cell { key, key, its patches' first word in m_ReadList, how many }; a patch is 4 words { its normal (octahedral,
            // 2 x snorm 16), its plane (float bits), its list's first word, its count }, its list after the cell's patches
            std::uint32_t Size = 1;
            while (Size < 2 * Cells.size()) Size <<= 1;
            m_ReadTable.assign(Size, { 0, 0xFFFFFFFFu, 0, 0 });
            for (std::size_t c = 0; c < Order.size(); ++c)
            {
                const auto& Ce = *Order[c];
                const auto First = std::uint32_t(m_ReadList.size());
                const auto n = std::uint32_t(Ce.m_Patches.size());
                m_ReadList.resize(m_ReadList.size() + 4 * n);
                for (std::uint32_t p = 0; p < n; ++p)
                {
                    const auto& Pa = Ce.m_Patches[p];
                    const std::uint32_t L0 = std::uint32_t(m_ReadList.size());
                    m_ReadList.insert(m_ReadList.end(), Lists[c][p].begin(), Lists[c][p].end());
                    m_ReadList[First + 4 * p + 0] = PackOct(Pa.m_N);
                    m_ReadList[First + 4 * p + 1] = std::bit_cast<std::uint32_t>(Pa.m_D);
                    m_ReadList[First + 4 * p + 2] = L0;
                    m_ReadList[First + 4 * p + 3] = std::uint32_t(Lists[c][p].size());
                }
                const auto K = ReadKey(Ce.m_C);
                std::uint32_t h = ReadHash(Ce.m_C) & (Size - 1), nProbe = 0;
                while (m_ReadTable[h][1] != 0xFFFFFFFFu) { h = (h + 1) & (Size - 1); ++nProbe; }
                m_ReadMaxProbe = std::max(m_ReadMaxProbe, nProbe + 1);
                m_ReadTable[h] = { K[0], K[1], First, n };
            }
            m_ReadUnsupported = Unsupported.load();
        }

        // A unit normal as 2 x snorm 16 octahedral (GLSL unpackSnorm2x16 + GIOctDecode)
        static std::uint32_t PackOct(const v3& N) noexcept
        {
            const float k = 1.0f / (std::abs(N.m_X) + std::abs(N.m_Y) + std::abs(N.m_Z));
            float x = N.m_X * k, y = N.m_Y * k;
            if (N.m_Z < 0) { const float ox = x, oy = y; x = (1 - std::abs(oy)) * (ox >= 0 ? 1.0f : -1.0f); y = (1 - std::abs(ox)) * (oy >= 0 ? 1.0f : -1.0f); }
            const auto S = [](float f) { return std::uint32_t(std::uint16_t(std::int16_t(std::lround(std::clamp(f, -1.0f, 1.0f) * 32767.0f)))); };
            return S(x) | (S(y) << 16);
        }
        static v3 UnpackOct(std::uint32_t W) noexcept
        {
            const float x0 = std::max(float(std::int16_t(W & 0xFFFF)) / 32767.0f, -1.0f), y0 = std::max(float(std::int16_t(W >> 16)) / 32767.0f, -1.0f);
            v3 V{ x0, y0, 1 - std::abs(x0) - std::abs(y0) };
            if (V.m_Z < 0) { V.m_X = (1 - std::abs(y0)) * (x0 >= 0 ? 1.0f : -1.0f); V.m_Y = (1 - std::abs(x0)) * (y0 >= 0 ? 1.0f : -1.0f); }
            return V.NormalizeCopy();
        }

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
        //  - candidates sampled on the surface by area (about 4 per Radius^2 of each triangle, and at least one per half Radius of its longest
        //    edge: a thin strip - a frame, a sill - has little area but its whole length needs surfels), each with the
        //    interpolated vertex normal: the surfels follow the surface, not the vertices (a big wall gets as many as its area asks for)
        //  - Poisson-disk acceptance through a hash grid of cell Radius: a candidate is kept when no kept surfel with a similar normal
        //    (cos > 0.8) is closer than Radius (dart throwing in a shuffled order: blue-noise spacing)
        //  - per vertex its 4 nearest compatible surfels (cos >= 0.5) from the 3x3x3 cells around it (wider rings if none)
        //  - the parents: the same over the surfels' centers with ParentRadius; each surfel's parent is its nearest compatible one
        //  - the BVH and the 4-triangle rows for the rays
        // Deterministic (fixed seeds): the same mesh bakes the same surfels.
        void Bake(float Radius, float ParentRadius) noexcept
        {
            BuildRays();                                            // the vertices' visibility tests cast against the mesh

            // Dart throwing over weighted candidates: the kept ones become surfels of radius R
            // A candidate is taken by a kept surfel only if that one sees it: a door a few cm behind its frame gets surfels of its own
            // (the wall's past the frame's lip do not cover it - read through the lip, they faded in wedges)
            const auto Poisson = [this](const std::vector<v3>& P, const std::vector<v3>& N, float R, std::vector<surfel>& Out, surfel_grid& G, std::vector<int>* pFrom = nullptr)
            {
                Out.clear();
                if (pFrom) pFrom->clear();
                G = { R, {} };
                std::vector<int> Order(P.size());
                for (std::size_t i = 0; i < Order.size(); ++i) Order[i] = int(i);
                rng Shuffle(0x5EEDu);
                for (std::size_t i = Order.size(); i > 1; --i) std::swap(Order[i - 1], Order[std::size_t(Shuffle.Next() * float(i)) % i]);
                for (int i : Order)
                {
                    bool bTaken = false;
                    // ... and only if it lies in the candidate's own plane (a tenth of R off it at most): a window frame 10 cm behind blinds,
                    // tilted so its plane runs through them and seeing them through their gaps, does not take the blinds' surfels
                    G.Near(P[i], 1, [&](int s) { bTaken = bTaken || ((P[i] - Out[s].m_Center).Length() < R && Out[s].m_Normal.Dot(N[i]) > 0.8f
                                                                    && std::abs((Out[s].m_Center - P[i]).Dot(N[i])) < 0.1f * R && Sees(P[i], N[i], Out[s])); });
                    if (bTaken) continue;
                    G.Add(P[i], int(Out.size()));
                    Out.push_back({ P[i], N[i], R });
                    if (pFrom) pFrom->push_back(i);
                }
            };

            // The nearest compatible surfels of a point (up to 4) that the point sees (nothing between them: a surfel behind a wall, or
            // in a gap the point only reaches through solid geometry, is not its surfel), widening the search until something is found
            const auto Nearest = [this](const std::vector<surfel>& L, const surfel_grid& G, const v3& P, const v3& N, int nMax)
            {
                std::array<std::pair<float, int>, 4> Best;
                Best.fill({ 1e30f, -1 });
                for (int Ring = 1; Ring <= 4 && Best[0].second < 0; ++Ring)
                    G.Near(P, Ring, [&](int s)
                    {
                        if (L[s].m_Normal.Dot(N) < 0.5f) return;
                        const float d = (P - L[s].m_Center).Length();
                        if (d >= Best[nMax - 1].first || std::any_of(Best.begin(), Best.end(), [&](const auto& b) { return b.second == s; })) return;
                        if (!Sees(P, N, L[s])) return;
                        Best[nMax - 1] = { d, s };
                        std::sort(Best.begin(), Best.begin() + nMax);
                    });
                return Best;
            };

            // 1. Candidates on the surface, by area
            std::vector<v3> CP, CN;
            std::vector<std::array<float, 3>> CT;                  // per candidate: its triangle, u, v (where its material is)
            rng Sample(0xBA4Eu);
            const float PerArea = 4.0f / (Radius * Radius);
            for (const auto& T : m_Tris)
            {
                const v3 A = m_Pos[T[0]], B = m_Pos[T[1]], C = m_Pos[T[2]];
                const float Area = 0.5f * (B - A).Cross(C - A).Length();
                const float Long = std::max({ (B - A).Length(), (C - B).Length(), (A - C).Length() });
                const int   n    = std::min(1 << 16, std::max({ 1, int(std::ceil(Area * PerArea)), int(std::ceil(Long / (0.5f * Radius))) }));     // by area, and along a thin strip by length
                for (int k = 0; k < n; ++k)
                {
                    float u = Sample.Next(), v = Sample.Next();
                    if (u + v > 1) { u = 1 - u; v = 1 - v; }
                    CP.push_back(A + (B - A) * u + (C - A) * v);
                    // The triangle's own normal (to the smoothed normal's side): the GI's geometry tests use the surface as it is. A box
                    // with averaged corner normals would tilt its surfels near the corners, and the reads step along its triangles
                    const v3 Ns = m_Nrm[T[0]] * (1 - u - v) + m_Nrm[T[1]] * u + m_Nrm[T[2]] * v;
                    v3 Nn = (B - A).Cross(C - A);
                    if (Nn.Length() < 1e-12f) Nn = Ns;
                    else if (Nn.Dot(Ns) < 0) Nn = Nn * -1.0f;
                    CN.push_back(Nn.NormalizeCopy());
                    CT.push_back({ float(&T - m_Tris.data()), u, v });
                }
            }

            // 2. Surfels, 3. vertex indices
            surfel_grid G0, G1;
            std::vector<int> From;
            Poisson(CP, CN, Radius, m_Levels[0], G0, &From);

            // Each surfel's material sample: its triangle's slot, the UV at its center, its radius in UV units (the triangle's UV area
            // over its area, square-rooted: how much texture a mesh unit covers there)
            m_SurfelMaterial.assign(m_Levels[0].size(), surfel_material{});
            for (std::size_t s = 0; s < From.size(); ++s)
            {
                const int   t = int(CT[From[s]][0]);
                const float u = CT[From[s]][1], v = CT[From[s]][2];
                auto&       M = m_SurfelMaterial[s];
                M.m_iMaterial = t < int(m_TriMaterial.size()) ? m_TriMaterial[t] : 0u;
                if (m_UV.size() != m_Pos.size()) continue;
                const auto& T = m_Tris[t];
                const auto& A = m_UV[T[0]]; const auto& B = m_UV[T[1]]; const auto& C = m_UV[T[2]];
                M.m_U = A[0] + (B[0] - A[0]) * u + (C[0] - A[0]) * v;
                M.m_V = A[1] + (B[1] - A[1]) * u + (C[1] - A[1]) * v;
                const float UVArea = 0.5f * std::abs((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1]));
                const float Area   = 0.5f * (m_Pos[T[1]] - m_Pos[T[0]]).Cross(m_Pos[T[2]] - m_Pos[T[0]]).Length();
                M.m_UVRadius = Area > 1e-12f ? Radius * std::sqrt(UVArea / Area) : 0.0f;
            }
            m_VertSurfels.resize(m_Pos.size());
            ParallelFor(int(m_Pos.size()), [&](int v)
            {
                const auto B = Nearest(m_Levels[0], G0, m_Pos[v], m_Nrm[v], 4);
                m_VertSurfels[v] = { B[0].second, B[1].second, B[2].second, B[3].second };
            });

            // 4. Parents
            std::vector<v3> SP(m_Levels[0].size()), SN(m_Levels[0].size());
            for (std::size_t s = 0; s < SP.size(); ++s) { SP[s] = m_Levels[0][s].m_Center; SN[s] = m_Levels[0][s].m_Normal; }
            Poisson(SP, SN, ParentRadius, m_Levels[1], G1);
            m_ParentOf.assign(SP.size(), 0);
            ParallelFor(int(SP.size()), [&](int s)
            {
                const auto B = Nearest(m_Levels[1], G1, SP[s], SN[s], 1);
                m_ParentOf[s] = std::max(B[0].second, 0);
            });

            BuildNeighbors();
        }

        // The spatial filter's neighbors (world::FilteredLight), per level: the surfels within a surfel's reach that face about the same
        // way (cos >= 0.8) and that it sees, each with its geometric weight (a linear falloff over the reach, times the normal match). Built once
        struct neighbors
        {
            std::vector<std::uint32_t>  m_First;        // per surfel + 1: where its neighbors start in m_Index
            std::vector<std::uint32_t>  m_Index;
            std::vector<float>          m_Weight;
        };
        std::array<neighbors, 2>            m_Neighbors;

        // Where a fine surfel's rays start (BuildOrigins): its center, then up to 7 of the mesh's vertices it is the nearest surfel of
        // (within half its radius, facing its way), each with its own normal. A surfel stands for a patch: its rays come from across the patch at
        // the surface's own angles (slats of blinds: from different slats), not all from one point that may sit in one slat's shadow
        std::vector<std::uint32_t>          m_OriginFirst;          // per fine surfel: its first entry (size + 1)
        std::vector<v3>                     m_OriginPos, m_OriginNrm;

        void BuildOrigins() noexcept
        {
            const auto& L = m_Levels[0];
            m_OriginFirst.assign(L.size() + 1, 0); m_OriginPos.clear(); m_OriginNrm.clear();
            std::vector<std::vector<int>> Per(L.size());
            if (m_VertSurfels.size() == m_Pos.size())
                for (std::size_t v = 0; v < m_Pos.size(); ++v)
                {
                    const int s = m_VertSurfels[v][0];
                    if (s < 0 || std::size_t(s) >= L.size()) continue;
                    if ((m_Pos[v] - L[s].m_Center).Length() > 0.5f * L[s].m_Radius || m_Nrm[v].Dot(L[s].m_Normal) < 0.5f) continue;     // its inner half: the patch, not the corners next to it
                    Per[s].push_back(int(v));
                }
            for (std::size_t s = 0; s < L.size(); ++s)
            {
                m_OriginPos.push_back(L[s].m_Center); m_OriginNrm.push_back(L[s].m_Normal);
                const std::size_t n = Per[s].size(), k = std::min<std::size_t>(n, 7);
                for (std::size_t j = 0; j < k; ++j) { const int v = Per[s][j * n / k]; m_OriginPos.push_back(m_Pos[v]); m_OriginNrm.push_back(m_Nrm[v].NormalizeCopy()); }
                m_OriginFirst[s + 1] = std::uint32_t(m_OriginPos.size());
            }
        }

        void BuildNeighbors() noexcept
        {
            BuildOrigins();
            for (int Lv = 0; Lv < 2; ++Lv)
            {
                const auto& L = m_Levels[Lv];
                auto&       Nb = m_Neighbors[Lv];
                Nb = {};
                if (L.empty()) continue;
                surfel_grid G{ 2 * L[0].m_Radius, {} };
                for (int s = 0; s < int(L.size()); ++s) G.Add(L[s].m_Center, s);
                std::vector<std::vector<std::pair<std::uint32_t, float>>> Per(L.size());
                ParallelFor(int(L.size()), [&](int s)
                {
                    G.Near(L[s].m_Center, 1, [&](int n)
                    {
                        if (n == s) return;
                        const float c = L[s].m_Normal.Dot(L[n].m_Normal);
                        if (c < 0.8f) return;
                        const float f = 1.0f - (L[n].m_Center - L[s].m_Center).Length() / (2 * L[s].m_Radius);
                        if (f <= 0 || !(Sees(L[s].m_Center, L[s].m_Normal, L[n]) || SeesLifted(L[s], L[n]))) return;
                        Per[s].push_back({ std::uint32_t(n), f * c });      // linear: a neighbor counts (squared, the read's falloff, it barely did)
                    });
                });
                Nb.m_First.assign(L.size() + 1, 0);
                for (std::size_t s = 0; s < L.size(); ++s)
                {
                    Nb.m_First[s + 1] = Nb.m_First[s] + std::uint32_t(Per[s].size());
                    for (const auto& [n, w] : Per[s]) { Nb.m_Index.push_back(n); Nb.m_Weight.push_back(w); }
                }
            }
        }

        // What BuildReadGrid and BuildNeighbors make (the read grid, the neighbor lists, the rays' origins) as one block of words: the
        // GeomStatic compiler packs it into the resource, the renderer unpacks it at load instead of building it (seconds in Release, half
        // a minute in Debug for the Police Station). Only what the surfels and the geometry decide: the same in both. Unpack checks the
        // block's tag and sizes (false: build them)
        static constexpr std::uint32_t derived_tag_v = 0x31444947u;    // "GID1": a change to the layout below is a new tag
        std::vector<std::uint32_t> PackDerived() const noexcept
        {
            std::vector<std::uint32_t> W{ derived_tag_v, std::bit_cast<std::uint32_t>(m_ReadCell), m_ReadMaxProbe, m_ReadUnsupported
                                        , std::uint32_t(m_ReadTable.size()), std::uint32_t(m_ReadList.size()) };
            for (const auto& E : m_ReadTable) W.insert(W.end(), E.begin(), E.end());
            W.insert(W.end(), m_ReadList.begin(), m_ReadList.end());
            for (const auto& Nb : m_Neighbors)
            {
                W.push_back(std::uint32_t(Nb.m_First.size())); W.push_back(std::uint32_t(Nb.m_Index.size()));
                W.insert(W.end(), Nb.m_First.begin(), Nb.m_First.end());
                W.insert(W.end(), Nb.m_Index.begin(), Nb.m_Index.end());
                for (float w : Nb.m_Weight) W.push_back(std::bit_cast<std::uint32_t>(w));
            }
            W.push_back(std::uint32_t(m_OriginFirst.size())); W.push_back(std::uint32_t(m_OriginPos.size()));
            W.insert(W.end(), m_OriginFirst.begin(), m_OriginFirst.end());
            for (const auto* pV : { &m_OriginPos, &m_OriginNrm }) for (const auto& v : *pV) for (int a = 0; a < 3; ++a) W.push_back(std::bit_cast<std::uint32_t>(v[a]));
            return W;
        }
        bool UnpackDerived(std::span<const std::uint32_t> W) noexcept
        {
            std::size_t i = 0;
            const auto Take = [&](std::size_t n) { const bool b = i + n <= W.size(); const std::size_t at = i; i += n; return b ? W.data() + at : nullptr; };
            const std::uint32_t* p = Take(6);
            if (p == nullptr || p[0] != derived_tag_v) return false;
            const float Cell = std::bit_cast<float>(p[1]);
            const std::uint32_t MaxProbe = p[2], Unsupported = p[3], nTable = p[4], nList = p[5];
            const std::uint32_t* pTable = Take(4 * std::size_t(nTable));
            const std::uint32_t* pList  = Take(nList);
            if (!pTable || !pList) return false;
            std::array<neighbors, 2> Nbs;
            for (auto& Nb : Nbs)
            {
                const std::uint32_t* q = Take(2);
                if (!q) return false;
                const std::uint32_t nFirst = q[0], nIndex = q[1];
                const std::uint32_t *pF = Take(nFirst), *pI = Take(nIndex), *pW = Take(nIndex);
                if (!pF || !pI || !pW) return false;
                Nb.m_First.assign(pF, pF + nFirst); Nb.m_Index.assign(pI, pI + nIndex);
                Nb.m_Weight.resize(nIndex);
                for (std::uint32_t k = 0; k < nIndex; ++k) Nb.m_Weight[k] = std::bit_cast<float>(pW[k]);
            }
            const std::uint32_t* q = Take(2);
            if (!q) return false;
            const std::uint32_t nOFirst = q[0], nO = q[1];
            const std::uint32_t *pOF = Take(nOFirst), *pOP = Take(3 * std::size_t(nO)), *pON = Take(3 * std::size_t(nO));
            if (!pOF || !pOP || !pON || i != W.size()) return false;
            if (Nbs[0].m_First.size() != 0 && Nbs[0].m_First.size() != m_Levels[0].size() + 1) return false;       // made for other surfels
            if (nOFirst != 0 && nOFirst != m_Levels[0].size() + 1) return false;
            m_ReadCell = Cell; m_ReadMaxProbe = MaxProbe; m_ReadUnsupported = Unsupported;
            m_ReadTable.resize(nTable);
            for (std::uint32_t k = 0; k < nTable; ++k) m_ReadTable[k] = { pTable[4 * k], pTable[4 * k + 1], pTable[4 * k + 2], pTable[4 * k + 3] };
            m_ReadList.assign(pList, pList + nList);
            m_Neighbors = std::move(Nbs);
            m_OriginFirst.assign(pOF, pOF + nOFirst);
            const auto V = [](const std::uint32_t* f, std::uint32_t n) { std::vector<v3> Out(n); for (std::uint32_t k = 0; k < n; ++k) Out[k] = { std::bit_cast<float>(f[3 * k]), std::bit_cast<float>(f[3 * k + 1]), std::bit_cast<float>(f[3 * k + 2]) }; return Out; };
            m_OriginPos = V(pOP, nO); m_OriginNrm = V(pON, nO);
            return true;
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

        // Does a point on the surface (its normal) see a surfel: nothing between them, both lifted off their surfaces a little
        // Do two surfels see each other from half a radius off their surfaces? Fine stacked geometry (blinds, louvres) blocks the line
        // along the surface between its surfels, but not the room in front of it; a wall between two floor surfels still blocks it
        bool SeesLifted(const surfel& A, const surfel& B) const noexcept
        {
            const float h = 0.5f * A.m_Radius;
            const v3 P = A.m_Center + A.m_Normal * h, Q = B.m_Center + B.m_Normal * h;
            const auto Clear = [&](const v3& From, const v3& To)
            {
                const v3 D = To - From; const float L = D.Length();
                return L < 1e-6f || !Occluded(From, D * (1.0f / L), L);
            };
            return Clear(A.m_Center + A.m_Normal * (0.02f * A.m_Radius), P) && Clear(B.m_Center + B.m_Normal * (0.02f * B.m_Radius), Q) && Clear(P, Q);
        }

        bool Sees(const v3& P, const v3& N, const surfel& S) const noexcept
        {
            const float e = 0.02f * S.m_Radius;
            const v3 A = P + N * e, D = (S.m_Center + S.m_Normal * e) - A;
            const float L = D.Length();
            return L < 1e-6f || !Occluded(A, D * (1.0f / L), L);
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
        // SeeThrough (per triangle, optional): triangles the rays pass through (transparent materials)
        bool Cast(const v3& O, const v3& D, float& TMax, int& Tri, float& U, float& V, std::span<const std::uint8_t> SeeThrough = {}) const noexcept
        {
            bool bHit = false;
            m_Bvh.TraverseLeaves(ray4(O, D), TMax, [&](int l, float& TBest)
            {
                float d, u, v;
                int Lanes = 0xF;
                if (!SeeThrough.empty())
                    for (int j = 0; j < 4; ++j) { const int t = m_Tri4[l].m_Tri[j]; if (t >= 0 && std::size_t(t) < SeeThrough.size() && SeeThrough[t]) Lanes &= ~(1 << j); }
                const int k = m_Tri4[l].Intersect(O, D, TBest, d, u, v, Lanes);
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
        M.m_UV.assign(Verts.size(), { 0.0f, 0.0f });
        M.m_Tris.clear();
        M.m_TriMaterial.clear();
        std::vector<std::uint16_t> ClusterMaterial(Geom.getClusters().size(), 0);       // a cluster's material: its submesh's
        for (const auto& Sub : Geom.getSubmeshes())
            for (std::uint32_t c = Sub.m_iCluster; c < std::uint32_t(Sub.m_iCluster) + Sub.m_nCluster && c < ClusterMaterial.size(); ++c) ClusterMaterial[c] = Sub.m_iMaterial;
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
                M.m_UV[v]  = { float(Extra[v].m_UV[0]) / 65535.0f * D.m_UVScaleTranslation.m_X + D.m_UVScaleTranslation.m_Z
                             , float(Extra[v].m_UV[1]) / 65535.0f * D.m_UVScaleTranslation.m_Y + D.m_UVScaleTranslation.m_W };
            }
            for (std::uint32_t i = C.m_iIndex; i + 2 < C.m_iIndex + C.m_nIndices && i + 2 < Index.size(); i += 3)
            {
                M.m_Tris.push_back({ int(C.m_iVertex + Index[i]), int(C.m_iVertex + Index[i + 1]), int(C.m_iVertex + Index[i + 2]) });
                M.m_TriMaterial.push_back(ClusterMaterial[c]);
            }
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
        if constexpr (requires { Geom.getGISurfelMaterials(); })
        {
            M.m_SurfelMaterial.clear();
            for (const auto& S : Geom.getGISurfelMaterials()) M.m_SurfelMaterial.push_back({ S.m_U, S.m_V, S.m_UVRadius, S.m_iMaterial });
        }
        M.BuildRays();
        // The read grid, the neighbors, the rays' origins: the compiler's (a geometry compiled before it packed them: built here)
        bool bDerived = false;
        if constexpr (requires { Geom.getGIDerived(); }) bDerived = M.UnpackDerived(Geom.getGIDerived());
        if (!bDerived) M.BuildNeighbors();
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
            float   m_MinBlend      = 0;        // the warm-up's blend (1 / n after a change) never below this (1: every pass replaces)
            float   m_EmissiveScale = 1;        // the surfels' emission (SetSurfelMaterial) times this: self-illumination as the lights it is
            int     m_FilterPasses  = 4;        // the shown light's spatial filter (BasisLight): each pass about one surfel radius wider
            bool    m_bSpreadOrigins = true;    // the fine surfels' rays start across their patch (surfel_mesh::BuildOrigins), not at their center
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
            int     m_nBad      = 0;            // surfels whose light is not a finite, positive value (0: always; a bad one spreads through the bounces)
            int     m_nColored  = 0;            // instances with their surfels' material colors (the GPU capture)
            int     m_nEmissive = 0;            // surfels that emit
            v3      m_MeanAlbedo{ 0, 0, 0 };    // over the colored surfels
            int     m_nLinks    = 0;            // emitter links (RebuildEmitterLinks)
            float   m_MaxEmitted = 0;           // the most emitted light a surfel receives (its links' irradiance, luminance)
            v3      m_MaxEmitter{ 0, 0, 0 };    // the brightest emitter's radiance
            int     m_nUnsupported = 0;         // this update's rays whose hit read no surfel (Indirect unsupported: the policy's light, Unsupported)
            double  m_UploadMs  = 0;            // the renderer's: the shown light (BasisLight, its filter) and the moments into the GPU's buffers
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
            m_bLinksDirty = true;
            return int(m_Instances.size()) - 1;
        }

        // The colors of an instance's surface, per fine surfel (its material sampled where the surfel is: albedo and emitted radiance).
        // The light a ray brings back from the instance is its color there times what lights it, plus what it emits. Empty: the
        // instance's albedo, no emission. Its neighbors re-solve (what they see changed)
        bool m_bEmitterLinks = true;            // false: emitters only through the rays' hits (tests compare the two)
        void SetSurfelMaterial(int Id, std::vector<v3> Albedo, std::vector<v3> Emission) noexcept
        {
            auto& I = m_Instances[Id];
            const std::size_t n = I.m_pMesh->m_Levels[0].size();
            I.m_SurfAlbedo   = Albedo.size() == n ? std::move(Albedo) : std::vector<v3>{};
            I.m_SurfEmission = Emission.size() == n ? std::move(Emission) : std::vector<v3>{};
            m_bLinksDirty = true;
            DirtyNear(I.m_Lo, I.m_Hi);
        }

        // An instance's triangles' colors (the material capture, triangle First on): a ray hit takes its triangle's (SurfaceAt), so a
        // color ends where its triangles do - a surfel's color spreads over its whole footprint (a door frame's brown onto the wall)
        void SetTriangleAlbedo(int Id, std::size_t First, const std::vector<v3>& Albedo) noexcept
        {
            auto& I = m_Instances[Id];
            const std::size_t n = I.m_pMesh->m_Tris.size();
            if (I.m_TriAlbedo.size() != n) I.m_TriAlbedo.assign(n, I.m_Albedo);
            for (std::size_t k = 0; k < Albedo.size() && First + k < n; ++k) I.m_TriAlbedo[First + k] = Albedo[k];
            DirtyNear(I.m_Lo, I.m_Hi);
        }
        const std::vector<v3>& TriangleAlbedo( int Id ) const noexcept { return m_Instances[Id].m_TriAlbedo; }

        // An instance's transparent material slots (SlotSeeThrough[slot]: its material has alpha). The GI leaves them out, as the references
        // do (Lumen, DDGI, Wicked: translucent surfaces neither block nor bounce): rays pass through their triangles, their surfels go dark
        // (validity 0: the reads skip them) - a window lets the sky in, a glass pane does not bounce its own color
        void SetSeeThrough(int Id, const std::vector<bool>& SlotSeeThrough) noexcept
        {
            auto& I = m_Instances[Id];
            const auto& M = *I.m_pMesh;
            const auto Slot = [&](std::size_t s) { return s < SlotSeeThrough.size() && SlotSeeThrough[s]; };
            I.m_SeeThrough.assign(M.m_Tris.size(), 0);
            for (std::size_t t = 0; t < M.m_Tris.size(); ++t) I.m_SeeThrough[t] = Slot(t < M.m_TriMaterial.size() ? M.m_TriMaterial[t] : 0u);
            I.m_SurfSeeThrough.assign(M.m_Levels[0].size(), 0);
            for (std::size_t s = 0; s < I.m_SurfSeeThrough.size(); ++s) I.m_SurfSeeThrough[s] = Slot(s < M.m_SurfelMaterial.size() ? M.m_SurfelMaterial[s].m_iMaterial : 0u);
            if (std::find(I.m_SeeThrough.begin(), I.m_SeeThrough.end(), 1) == I.m_SeeThrough.end()) { I.m_SeeThrough.clear(); I.m_SurfSeeThrough.clear(); }
            DirtyNear(I.m_Lo, I.m_Hi);
        }

        const std::vector<v3>& SurfelAlbedo  ( int Id ) const noexcept { return m_Instances[Id].m_SurfAlbedo; }
        const std::vector<v3>& SurfelEmission( int Id ) const noexcept { return m_Instances[Id].m_SurfEmission; }

        void DirtyAll() noexcept { for (auto& I : m_Instances) if (I.m_bAlive) I.m_bDirty = true; }

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
            m_bLinksDirty = true;           // ponytail: every link is rebuilt when anything moves (scenes with emitters and moving meshes: per instance)
        }

        int             InstanceCount  ( void ) const noexcept { return int(m_Instances.size()); }
        int             Level          ( int Id ) const noexcept { return m_Instances[Id].m_Level; }
        const v3&       Albedo         ( int Id ) const noexcept { return m_Instances[Id].m_Albedo; }
        // What the GPU buffer would hold for an instance: its active level's irradiance and validity per surfel
        const std::vector<v3>&    InstanceLight( int Id ) const noexcept { return m_Instances[Id].m_E; }

        // The light the reads show (the GPU's upload), per basis direction: 3 x the light times its fractions (their mean is the light),
        // blended with the neighbors' (the mesh's neighbor lists, their weights times their validity; itself at weight 1),
        // m_Setup.m_FilterPasses times over. The independent noise of neighboring surfels averages out. Only what is shown: the solver
        // keeps its own light (a filter fed back every update would keep spreading the light)
        // Every frame for every shown instance (the GPU's upload): on plain floats through raw pointers (a Debug build calls every
        // vector and v3 operator: 175 ms a frame for the Police Station's 63k surfels), the surfels in blocks across the workers
        const std::vector<basis>& BasisLight( int Id ) const noexcept
        {
            const auto& I  = m_Instances[Id];
            const auto& Nb = I.m_pMesh->m_Neighbors[I.m_Level];
            const int   n  = int(I.m_E.size());
            auto&       F  = I.m_Filtered;
            F.resize(n);
            {
                const v3* pE = I.m_E.data(); const basis* pFrac = I.m_Frac.data(); basis* pF = F.data();
                for (int s = 0; s < n; ++s)
                    for (int i = 0; i < 3; ++i)
                    {
                        const v3& f = pFrac[s][i];
                        pF[s][i].m_X = pE[s].m_X * f.m_X * 3.0f; pF[s][i].m_Y = pE[s].m_Y * f.m_Y * 3.0f; pF[s][i].m_Z = pE[s].m_Z * f.m_Z * 3.0f;
                    }
            }
            if (Nb.m_First.size() != std::size_t(n) + 1 || m_Setup.m_FilterPasses <= 0) return F;
            auto& In = I.m_FilterIn;
            In.resize(n);
            const std::uint32_t* pFirst = Nb.m_First.data(); const std::uint32_t* pIdx = Nb.m_Index.data();
            const float* pWt = Nb.m_Weight.data(); const float* pValid = I.m_Valid.data();
            constexpr int block_v = 512;
            for (int Pass = 0; Pass < m_Setup.m_FilterPasses; ++Pass)
            {
                In.swap(F);
                const basis* pIn = In.data(); basis* pOut = F.data();
                m_Pool.Run((n + block_v - 1) / block_v, [&](int b)
                {
                    const int s1 = std::min(n, (b + 1) * block_v);
                    for (int s = b * block_v; s < s1; ++s)
                    {
                        const basis& Me = pIn[s];
                        float a0 = Me[0].m_X, a1 = Me[0].m_Y, a2 = Me[0].m_Z, a3 = Me[1].m_X, a4 = Me[1].m_Y, a5 = Me[1].m_Z, a6 = Me[2].m_X, a7 = Me[2].m_Y, a8 = Me[2].m_Z, W = 1;
                        for (std::uint32_t k = pFirst[s], k1 = pFirst[s + 1]; k < k1; ++k)
                        {
                            const std::uint32_t j = pIdx[k];
                            const float w = pWt[k] * pValid[j];
                            const basis& B = pIn[j];
                            a0 += B[0].m_X * w; a1 += B[0].m_Y * w; a2 += B[0].m_Z * w;
                            a3 += B[1].m_X * w; a4 += B[1].m_Y * w; a5 += B[1].m_Z * w;
                            a6 += B[2].m_X * w; a7 += B[2].m_Y * w; a8 += B[2].m_Z * w;
                            W += w;
                        }
                        const float r = 1.0f / W;
                        basis& O = pOut[s];
                        O[0].m_X = a0 * r; O[0].m_Y = a1 * r; O[0].m_Z = a2 * r;
                        O[1].m_X = a3 * r; O[1].m_Y = a4 * r; O[1].m_Z = a5 * r;
                        O[2].m_X = a6 * r; O[2].m_Y = a7 * r; O[2].m_Z = a8 * r;
                    }
                });
            }
            return F;
        }

        // The mean of the shown basis (what a surfel's own normal reads)
        std::vector<v3> FilteredLight( int Id ) const noexcept
        {
            const auto& B = BasisLight(Id);
            std::vector<v3> E(B.size());
            for (std::size_t s = 0; s < B.size(); ++s) E[s] = (B[s][0] + B[s][1] + B[s][2]) * (1.0f / 3);
            return E;
        }

        // Every live instance's surfels of its active level as CSV (the GI baseline, comparisons between runs): world position, normal and
        // radius, the solver's light (unfiltered, InstanceLight), the shown light (FilteredLight), the validity, the captured albedo. The
        // setup on the first lines (# comments). Returns the surfels written (-1: the file could not be made)
        int Dump(const char* pPath) const noexcept
        {
            std::FILE* f = nullptr;
        #ifdef _WIN32
            if (fopen_s(&f, pPath, "w") != 0) f = nullptr;
        #else
            f = std::fopen(pPath, "w");
        #endif
            if (f == nullptr) return -1;
            std::fprintf(f, "# RaysPerSurfel=%d FilterPasses=%d EmissiveScale=%g SpreadOrigins=%d LodDistance=%g\n"
                        , m_Setup.m_RaysPerSurfel, m_Setup.m_FilterPasses, m_Setup.m_EmissiveScale, int(m_Setup.m_bSpreadOrigins), m_Setup.m_LodDistance);
            std::fprintf(f, "inst,level,surfel,px,py,pz,nx,ny,nz,radius,er,eg,eb,fr,fg,fb,valid,ar,ag,ab\n");
            int n = 0;
            for (int Id = 0; Id < InstanceCount(); ++Id)
            {
                const auto& I = m_Instances[Id];
                if (!I.m_bAlive || I.m_pMesh == nullptr) continue;
                const auto& L = I.m_pMesh->m_Levels[I.m_Level];
                const auto  F = FilteredLight(Id);
                for (std::size_t s = 0; s < L.size() && s < I.m_E.size(); ++s, ++n)
                {
                    const v3 P = I.m_L2W * L[s].m_Center, N = I.m_L2W.TransformDirection(L[s].m_Normal).NormalizeCopy();
                    const v3 A = I.m_Level == 0 && s < I.m_SurfAlbedo.size() ? I.m_SurfAlbedo[s] : I.m_Albedo;
                    const v3 Fs = s < F.size() ? F[s] : v3{ 0, 0, 0 };
                    std::fprintf(f, "%d,%d,%zu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f\n"
                                , Id, I.m_Level, s, P.m_X, P.m_Y, P.m_Z, N.m_X, N.m_Y, N.m_Z, L[s].m_Radius * I.m_MeanScale
                                , I.m_E[s].m_X, I.m_E[s].m_Y, I.m_E[s].m_Z, Fs.m_X, Fs.m_Y, Fs.m_Z
                                , s < I.m_Valid.size() ? I.m_Valid[s] : 0.0f, A.m_X, A.m_Y, A.m_Z);
                }
            }
            std::fclose(f);
        #ifdef _WIN32
            if (fopen_s(&f, (std::string(pPath) + ".unsupported.csv").c_str(), "w") != 0) f = nullptr;
        #else
            f = std::fopen((std::string(pPath) + ".unsupported.csv").c_str(), "w");
        #endif
            if (f)                                                  // the last update's unsupported queries (the first per surfel, up to 4096)
            {
                std::fprintf(f, "x,y,z\n");
                for (const auto& P : m_UnsupportedAt) std::fprintf(f, "%.4f,%.4f,%.4f\n", P.m_X, P.m_Y, P.m_Z);
                std::fclose(f);
            }
            return n;
        }

        // What a normal n (mesh space) reads of a surfel's shown light (the CPU twin of the shaders' basis read)
        v3 BasisRead( int Id, int s, const v3& n ) const noexcept
        {
            const auto& I = m_Instances[Id];
            const auto  B = BasisLight(Id)[s];
            const auto  w = BasisWeights(BasisDirs(I.m_pMesh->m_Levels[I.m_Level][s].m_Normal), n);
            return B[0] * w[0] + B[1] * w[1] + B[2] * w[2];
        }
        const std::vector<float>& InstanceValid( int Id ) const noexcept { return m_Instances[Id].m_Valid; }
        const std::vector<moments>& InstanceMoments( int Id ) const noexcept { return m_Instances[Id].m_Moments; }
        // What changes an instance's moments: the last update's surfels (instance, surfel), and a reset of them all (a level switch, a
        // new instance): a copy of them stays current with these two, without reading them all every frame
        std::span<const std::pair<int, int>> LastUpdated( void ) const noexcept { return m_Work; }
        std::uint32_t MomentsReset( int Id ) const noexcept { return m_Instances[Id].m_MomentsReset; }

        // The anti-leak weight of the reads at P (mesh space) for surfel s of instance Id's active level: the plane distance times the
        // depth moments' Chebyshev term (the CPU twin of GIVisibility in shaders/xlionrender_surfel_gi.glsl)
        float Visibility( int Id, int s, const v3& P ) const noexcept
        {
            const auto& I = m_Instances[Id];
            const auto& S = I.m_pMesh->m_Levels[I.m_Level][s];
            const v3    D = P - S.m_Center;
            const float Reach = 2 * S.m_Radius;
            float w = std::clamp(1.0f - std::abs(D.Dot(S.m_Normal)) / (0.7f * Reach), 0.0f, 1.0f);
            if (w <= 0) return w;
            const v3    Dq = D + S.m_Normal * (0.15f * Reach);          // lifted off the surface, as the shaders (GIVisibility)
            const float d  = Dq.Length();
            const auto& M  = I.m_Moments[s];
            const v3    Dd = Dq * (1.0f / d);
            const v3    T  = (std::abs(S.m_Normal.m_X) < 0.9f ? v3{ 1, 0, 0 } : v3{ 0, 1, 0 }).Cross(S.m_Normal).NormalizeCopy(), B = S.m_Normal.Cross(T);
            const float x  = Dd.Dot(T), y = Dd.Dot(B), z = std::max(Dd.Dot(S.m_Normal), 0.0f), k = 1.0f / std::max(std::abs(x) + std::abs(y) + z, 1e-6f);
            const float u  = std::clamp(((x + y) * k * 0.5f + 0.5f) * moment_res_v - 0.5f, 0.0f, float(moment_res_v - 1));
            const float vv = std::clamp(((x - y) * k * 0.5f + 0.5f) * moment_res_v - 0.5f, 0.0f, float(moment_res_v - 1));
            const int   u0 = int(u), v0 = int(vv), u1 = std::min(u0 + 1, moment_res_v - 1), v1 = std::min(v0 + 1, moment_res_v - 1);
            const float fu = u - float(u0), fv = vv - float(v0);
            const auto  At = [&](int tu, int tv, int c) { return M[2 * (tv * moment_res_v + tu) + c]; };
            const auto  Bil = [&](int c) { return (At(u0, v0, c) * (1 - fu) + At(u1, v0, c) * fu) * (1 - fv) + (At(u0, v1, c) * (1 - fu) + At(u1, v1, c) * fu) * fv; };
            const float Mean = Bil(0), Var = std::max(std::abs(Mean * Mean - Bil(1)), 0.0025f * Reach * Reach);     // bilinear, a variance floor: as the shaders
            if (d > Mean) w *= std::pow(Var / (Var + (d - Mean) * (d - Mean) + 1e-12f), 3.0f);
            return w;
        }

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
                if (I.m_pMesh->Cast(I.m_W2L * O, I.m_W2L.TransformDirection(D), TBest, Tri, U, V, I.m_SeeThrough)) { H.m_Inst = i; H.m_Tri = Tri; H.m_U = U; H.m_V = V; H.m_T = TBest; }
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
        // The light a ray's hit point reads (the hit triangle's vertices' surfels, as the GPU's vertex read): the solver's GI query. Unsupported
        // (no valid surfel facing the hit's way at any of its vertices) is not dark: Update applies the policy (Unsupported)
        struct query
        {
            v3      m_E{ 0, 0, 0 };
            float   m_Confidence = 0;       // the read's total weight (a convention, not an error bound)
            bool    m_bSupported = false;
        };
        query Indirect(const hit& H) const noexcept
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
            if (W <= 0) return {};
            return { Sum * (1.0f / W), W, true };
        }
        // The policy for an unsupported query: the hit instance's mean light (its valid surfels', weighted by validity; MeanLight), not black and
        // not dropped - the ray still counts as one of the surfel's rays, the same denominator. Counted, the first places kept (UnsupportedAt)
        v3 Unsupported(const hit& H) const noexcept { return m_Instances[H.m_Inst].m_MeanE; }
        const v3& MeanLight(int Id) const noexcept { return m_Instances[Id].m_MeanE; }      // the same for the GPU's reads (the renderer uploads it)
        const std::vector<v3>& UnsupportedAt() const noexcept { return m_UnsupportedAt; }

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
                m_bEmitDirty = true;
                I.m_Acc.assign(I.m_E.size(), v3{ 0, 0, 0 }); I.m_AccValid.assign(I.m_E.size(), 1);
                ResetMoments(I);
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

            if (m_bLinksDirty) { if (m_bEmitterLinks) RebuildEmitterLinks(); else { m_Emitters.clear(); for (auto& I : m_Instances) { I.m_Links.clear(); I.m_LinkFirst.clear(); } } m_bLinksDirty = false; m_bEmitDirty = true; }
            // The emitters' light: made of the links, the emissions (SetSurfelMaterial: the links again) and the scale; a level switch
            // changes its layout. Nothing else: not every update
            if (m_bEmitDirty || m_EmitScale != m_Setup.m_EmissiveScale) { UpdateEmittedLight(); m_bEmitDirty = false; m_EmitScale = m_Setup.m_EmissiveScale; }

            // The policy's value per instance (Unsupported): this update's mean light of its valid surfels; none valid: the world's
            {
                v3 WSum{ 0, 0, 0 }; float WW = 0;
                for (auto& I : m_Instances)
                {
                    v3 S{ 0, 0, 0 }; float W = 0;
                    if (I.m_bAlive) for (std::size_t s = 0; s < I.m_E.size(); ++s) { S = S + I.m_E[s] * I.m_Valid[s]; W += I.m_Valid[s]; }
                    I.m_MeanE = W > 0 ? S * (1.0f / W) : v3{ -1, -1, -1 };
                    WSum = WSum + S; WW += W;
                }
                const v3 World = WW > 0 ? WSum * (1.0f / WW) : v3{ 0, 0, 0 };
                for (auto& I : m_Instances) if (I.m_MeanE.m_X < 0) I.m_MeanE = World;
            }
            m_WorkUnsupported.assign(m_Work.size(), 0);
            m_WorkUnsupportedAt.resize(m_Work.size());

            const auto T1 = std::chrono::steady_clock::now();
            m_Pool.Run(int(m_Work.size()), [&](int w)
            {
                const auto [i, s] = m_Work[w];
                auto& I = m_Instances[i];
                const auto& S = I.m_pMesh->m_Levels[I.m_Level][s];
                const v3 C = I.m_L2W * S.m_Center, N = NormalToWorld(I, S.m_Normal);
                rng Rng(Seed * 2654435761u + unsigned(w) * 7919u);
                v3 Sum{ 0, 0, 0 };
                int nBack = 0, nUnsupported = 0;
                const int R = m_Setup.m_RaysPerSurfel;
                auto& MA = I.m_MomAcc[s];
                MA.fill(0);
                auto& BA = I.m_BasisAcc[s];
                BA.fill(v3{ 0, 0, 0 });
                const basis Dirs = BasisDirs(S.m_Normal);
                std::array<float, 3> Bw{};
                const float Reach = 2 * S.m_Radius;                         // mesh units
                const float Off1 = Rng.Next(), Off2 = Rng.Next();         // this pass' rotation of the stratified set
                const auto& Mo = *I.m_pMesh;
                const bool  bSpread = m_Setup.m_bSpreadOrigins && I.m_Level == 0 && Mo.m_OriginFirst.size() == Mo.m_Levels[0].size() + 1;
                const std::uint32_t o0 = bSpread ? Mo.m_OriginFirst[s] : 0, nO = bSpread ? Mo.m_OriginFirst[s + 1] - o0 : 0;
                const int   Shift = int(Rng.Next() * 64.0f);               // the pass' pairing of origins and direction strata
                for (int r = 0; r < R; ++r)
                {
                    v3 Cr = C, Nr = N;                                      // the ray's origin: across the surfel's patch (BuildOrigins)
                    if (nO > 1) { const std::uint32_t k = o0 + std::uint32_t(r + Shift) % nO; Cr = I.m_L2W * Mo.m_OriginPos[k]; Nr = NormalToWorld(I, Mo.m_OriginNrm[k]); }
                    const v3 D = StratifiedCosineDir(Nr, r, R, Off1, Off2);
                    const hit H = Cast(Cr + Nr * 1e-3f, D);
                    {
                        v3 Dm = I.m_W2L.TransformDirection(D);              // mesh space: the reads compare in mesh units
                        const float Lm = Dm.Length();
                        Dm = Dm * (1.0f / Lm);
                        // the depth moments are of the surfel's center (the reads compare a point with it): where the ray ends, seen from it
                        const v3    Pe = I.m_W2L * (Cr + D * (H.m_Inst < 0 ? Reach / Lm : H.m_T)) - S.m_Center;
                        const float Le = Pe.Length();
                        const float d  = std::min(Le, Reach);
                        const int   t  = MomentTexel(S.m_Normal, Le > 1e-6f ? Pe * (1.0f / Le) : Dm);
                        MA[3 * t] += d; MA[3 * t + 1] += d * d; MA[3 * t + 2] += 1;
                        Bw = BasisWeights(Dirs, Dm);                        // the ray's share of each basis direction
                    }
                    if (H.m_Inst < 0) { Sum = Sum + L.m_Sky; for (int i = 0; i < 3; ++i) BA[i] = BA[i] + L.m_Sky * Bw[i]; continue; }
                    const v3 Nh = HitNormal(H);
                    if (Nh.Dot(D) > 0) { ++nBack; continue; }               // a back face: no light, and maybe buried
                    v3 Albedo, Emission;
                    SurfaceAt(H, Albedo, Emission);
                    const query Q  = Indirect(H);
                    if (!Q.m_bSupported) { if (nUnsupported++ == 0) m_WorkUnsupportedAt[w] = HitPosition(H); }
                    const v3 In = Direct(HitPosition(H), Nh, L) + (Q.m_bSupported ? Q.m_E : Unsupported(H));
                    const v3 Lr = v3{ Albedo.m_X * In.m_X, Albedo.m_Y * In.m_Y, Albedo.m_Z * In.m_Z } * (1.0f / pi_v) + (m_Emitters.empty() ? Emission : v3{ 0, 0, 0 });     // with emitters the links bring their light
                    Sum = Sum + Lr;
                    for (int i = 0; i < 3; ++i) BA[i] = BA[i] + Lr * Bw[i];
                }
                if (s < I.m_EmitB.size())                                   // the emitters it sees (the links): deterministic, no rays
                    for (int i = 0; i < 3; ++i) { const v3 e = I.m_EmitB[s][i] * (float(R) / (3.0f * pi_v)); Sum = Sum + e; BA[i] = BA[i] + e; }     // the basis' mean is the irradiance
                I.m_Acc[s]      = Sum * (pi_v / float(R));                  // E = pi mean(L), cosine sampling
                I.m_AccValid[s] = std::clamp(1.0f - (float(nBack) / float(R) - 0.1f) / 0.15f, 0.0f, 1.0f);
                m_WorkUnsupported[w] = nUnsupported;
            });
            m_UnsupportedAt.clear();
            for (std::size_t w = 0; w < m_Work.size(); ++w)
            {
                St.m_nUnsupported += m_WorkUnsupported[w];
                if (m_WorkUnsupported[w] && m_UnsupportedAt.size() < 4096) m_UnsupportedAt.push_back(m_WorkUnsupportedAt[w]);
            }

            St.m_TraceMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T1).count();
            for (const auto& [i, s] : m_Work)
            {
                auto& I = m_Instances[i];
                // The light: a running average while warming up (after a change), then the estimator (it settles, and still follows a
                // change). The validity and the depth moments: their own slow blends (they weigh the reads: noise there swims too)
                auto& Es = I.m_Est[s];
                if (I.m_History <= warmup_passes_v || I.m_PassAlpha >= 1)
                {
                    I.m_E[s] = I.m_E[s] * (1 - I.m_PassAlpha) + I.m_Acc[s] * I.m_PassAlpha;
                    Es.Seed(I.m_E[s]);
                }
                else
                {
                    Es.Add(I.m_Acc[s]);
                    I.m_E[s] = Es.m_Mean;
                }
                const float av = std::max(I.m_PassAlpha, 0.05f);
                I.m_Valid[s] = I.m_Valid[s] * (1 - av) + I.m_AccValid[s] * av;
                if (I.m_Level == 0 && s < I.m_SurfSeeThrough.size() && I.m_SurfSeeThrough[s]) I.m_Valid[s] = 0;     // on a transparent material
                {   // the basis: what fraction of the light came from around each direction (per channel), slowly (8 rays split 3 ways)
                    const auto& BA = I.m_BasisAcc[s];
                    const v3 Tot = BA[0] + BA[1] + BA[2];
                    const float af = std::max(I.m_PassAlpha, 0.1f);
                    for (int i = 0; i < 3; ++i)
                    {
                        const v3 F{ Tot.m_X > 1e-9f ? BA[i].m_X / Tot.m_X : 1.0f / 3, Tot.m_Y > 1e-9f ? BA[i].m_Y / Tot.m_Y : 1.0f / 3, Tot.m_Z > 1e-9f ? BA[i].m_Z / Tot.m_Z : 1.0f / 3 };
                        I.m_Frac[s][i] = I.m_Frac[s][i] + (F - I.m_Frac[s][i]) * af;
                    }
                }
                for (int t = 0; t < moment_texels_v; ++t)
                {
                    const auto& MA = I.m_MomAcc[s];
                    if (MA[3 * t + 2] == 0) continue;
                    auto& n = I.m_MomCount[s][t];
                    if (n < 0xFFFF) ++n;
                    const float a = std::max(1.0f / float(n), 0.03f);
                    auto& Mo = I.m_Moments[s];
                    Mo[2 * t]     += (MA[3 * t] / MA[3 * t + 2] - Mo[2 * t]) * a;
                    Mo[2 * t + 1] += (MA[3 * t + 1] / MA[3 * t + 2] - Mo[2 * t + 1]) * a;
                }
                if (I.m_Valid[s] < 1e-6f) I.m_Valid[s] = 0;         // buried: 0, not a decay into denormals (the reads divide by sums of these weights)
            }
            for (auto& I : m_Instances)
            {
                if (!I.m_bAlive) continue;
                if (!I.m_bPicked) ++I.m_Age;
                I.m_bPicked = false;
            }
            St.m_Ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();
            for (const auto& I : m_Instances)
                for (std::size_t s = 0; s < I.m_E.size(); ++s)
                    St.m_nBad += !(std::isfinite(I.m_E[s].m_X + I.m_E[s].m_Y + I.m_E[s].m_Z + I.m_Valid[s]) && I.m_E[s].m_X >= 0 && I.m_E[s].m_Y >= 0 && I.m_E[s].m_Z >= 0);
            std::size_t nAlbedo = 0;
            for (const auto& I : m_Instances)
            {
                if (!I.m_bAlive || I.m_SurfAlbedo.empty()) continue;
                ++St.m_nColored;
                for (const auto& a : I.m_SurfAlbedo) St.m_MeanAlbedo = St.m_MeanAlbedo + a;
                nAlbedo += I.m_SurfAlbedo.size();
                for (const auto& e : I.m_SurfEmission) St.m_nEmissive += (e.m_X + e.m_Y + e.m_Z) > 0.01f;
            }
            if (nAlbedo) St.m_MeanAlbedo = St.m_MeanAlbedo * (1.0f / float(nAlbedo));
            for (const auto& I : m_Instances)
            {
                St.m_nLinks += int(I.m_Links.size());
                for (const auto& B : I.m_EmitB) St.m_MaxEmitted = std::max(St.m_MaxEmitted, (B[0].m_X + B[1].m_X + B[2].m_X + B[0].m_Y + B[1].m_Y + B[2].m_Y) / 6.0f);
            }
            for (const auto& [i, e] : m_Emitters)
            {
                const v3 Le = m_Instances[i].m_SurfEmission[e] * m_Setup.m_EmissiveScale;
                if (Le.m_X + Le.m_Y + Le.m_Z > St.m_MaxEmitter.m_X + St.m_MaxEmitter.m_Y + St.m_MaxEmitter.m_Z) St.m_MaxEmitter = Le;
            }
            return St;
        }

        // Ray i of a pass of R, cosine distributed and stratified: the radial coordinate in R strata, the angle along the golden ratio, both
        // shifted by the pass' random offsets (Cranley-Patterson). A pass covers the hemisphere evenly (DDGI's rotated spherical Fibonacci
        // rays, for the same reason): 8 independent random rays cluster and miss the gaps between slats, and the surfel's light stays off
        static v3 StratifiedCosineDir(const v3& N, int i, int R, float Off1, float Off2) noexcept
        {
            const auto Frac = [](float x) { return x - std::floor(x); };
            const float u1 = Frac((float(i) + 0.5f) / float(R) + Off1), u2 = Frac(float(i) * 0.61803398875f + Off2);
            const float r = std::sqrt(u1), phi = 2 * pi_v * u2;
            const v3 T = (std::abs(N.m_X) > 0.5f ? N.Cross(v3{ 0, 1, 0 }) : N.Cross(v3{ 1, 0, 0 })).NormalizeCopy();
            const v3 B = N.Cross(T);
            return (T * (r * std::cos(phi)) + B * (r * std::sin(phi)) + N * std::sqrt(std::max(0.0f, 1 - r * r))).NormalizeCopy();
        }

        static v3 CosineDir(const v3& N, rng& Rng) noexcept
        {
            const float r = std::sqrt(Rng.Next()), phi = 2 * pi_v * Rng.Next();
            const v3 T = (std::abs(N.m_X) > 0.5f ? N.Cross(v3{ 0, 1, 0 }) : N.Cross(v3{ 1, 0, 0 })).NormalizeCopy();
            const v3 B = N.Cross(T);
            return (T * (r * std::cos(phi)) + B * (r * std::sin(phi)) + N * std::sqrt(std::max(0.0f, 1 - r * r))).NormalizeCopy();
        }

    private:

        // A surfel's view of an emitter (RebuildEmitterLinks): the emitter (m_Emitters), its weight (area x both cosines / distance^2:
        // the irradiance per unit of its radiance) and the shares of the surfel's 3 basis directions
        struct emit_link { std::uint32_t m_Emitter; float m_W; std::array<float, 3> m_Bw; };

        struct instance
        {
            const surfel_mesh*      m_pMesh = nullptr;
            xmath::fmat4            m_L2W, m_W2L;
            v3                      m_Albedo, m_Lo, m_Hi;
            std::vector<v3>         m_E, m_Acc;                 // per surfel of the active level
            std::vector<float>      m_Valid, m_AccValid;
            std::vector<moments>    m_Moments;                  // per surfel of the active level (MomentTexel)
            std::vector<v3>         m_SurfAlbedo, m_SurfEmission;     // per fine surfel (SetSurfelMaterial; empty: m_Albedo, none)
            std::vector<v3>         m_TriAlbedo;                      // per triangle (SetTriangleAlbedo; empty: the surfels')
            std::vector<std::uint8_t> m_SeeThrough, m_SurfSeeThrough;   // per triangle, per fine surfel: transparent (SetSeeThrough; empty: none)
            std::vector<std::uint32_t> m_LinkFirst;              // per fine surfel + 1: its emitter links (RebuildEmitterLinks)
            std::vector<emit_link>  m_Links;
            std::vector<basis>      m_EmitB;                    // per surfel of the active level: the emitters' irradiance this update, per basis direction
            std::vector<basis>      m_Frac;                     // per surfel: the fraction of its light from around each basis direction (per channel)
            std::vector<basis>      m_BasisAcc;                 // this pass: the light per basis direction
            mutable std::vector<basis> m_Filtered, m_FilterIn;  // BasisLight's (the result, the pass before)
            std::vector<std::array<float, 3 * moment_texels_v>> m_MomAcc;     // this pass: per texel the sum of d, of d^2, the count
            std::vector<std::array<std::uint16_t, moment_texels_v>> m_MomCount; // per texel the samples it has had (its blend: 1 / n, then slow)
            std::vector<estimator>  m_Est;                      // per surfel of the active level: its light over time
            int                     m_Level     = 0;
            int                     m_Age       = 0;
            int                     m_History   = 0;
            int                     m_Cursor    = 0;            // the next surfel of the pass in progress (0: none)
            float                   m_PassAlpha = 1;            // the blend of the pass in progress
            bool                    m_bDirty    = true;
            bool                    m_bPicked   = false;
            bool                    m_bAlive    = true;
            float                   m_MeanScale = 1;            // the transform's mean axis scale (surfel radii are mesh space)
            v3                      m_MeanE{ 0, 0, 0 };                 // this update's mean light of its valid surfels: an unsupported query's value (Unsupported)
            std::uint32_t           m_MomentsReset = 0;         // ResetMoments' count (MomentsReset)
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
            ResetMoments(I);
        }

        // Fully open (every direction travels the whole reach): a new surfel reads at full weight until its rays say otherwise
        static void ResetMoments(instance& I) noexcept
        {
            ++I.m_MomentsReset;
            const auto& L = I.m_pMesh->m_Levels[I.m_Level];
            I.m_Moments.resize(L.size());
            I.m_MomAcc.assign(L.size(), {});
            I.m_MomCount.assign(L.size(), {});
            const v3 Third{ 1.0f / 3, 1.0f / 3, 1.0f / 3 };
            I.m_Frac.assign(L.size(), basis{ Third, Third, Third });
            I.m_BasisAcc.assign(L.size(), basis{});
            I.m_Est.resize(L.size());
            for (std::size_t s = 0; s < L.size(); ++s) I.m_Est[s].Seed(I.m_E[s]);
            for (std::size_t s = 0; s < L.size(); ++s)
            {
                const float R = 2 * L[s].m_Radius;
                for (int t = 0; t < moment_texels_v; ++t) { I.m_Moments[s][2 * t] = R; I.m_Moments[s][2 * t + 1] = R * R; }
            }
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
        // The emitter lists (next-event estimation, precomputed): every fine surfel of every instance finds the emissive surfels it sees
        // (an emitter's light reaches it as its radiance x its area x both cosines / distance^2) and keeps the strongest that nothing
        // hides (a shadow ray each), once - not by waiting for 8 random rays to find a small sign. Rebuilt when emitters or placements
        // change
        static constexpr int max_links_v = 32;                 // enough that neighbors keep about the same emitters (no steps where their sets differ)
        void RebuildEmitterLinks() noexcept
        {
            m_Emitters.clear();
            for (int i = 0; i < int(m_Instances.size()); ++i)
            {
                const auto& I = m_Instances[i];
                if (!I.m_bAlive) continue;
                for (std::size_t e = 0; e < I.m_SurfEmission.size(); ++e)
                    if (I.m_SurfEmission[e].m_X + I.m_SurfEmission[e].m_Y + I.m_SurfEmission[e].m_Z > 1e-3f) m_Emitters.push_back({ i, int(e) });
            }
            struct emitter { v3 m_C, m_N; float m_A; };
            std::vector<emitter> Em(m_Emitters.size());
            for (std::size_t k = 0; k < Em.size(); ++k)
            {
                const auto& I = m_Instances[m_Emitters[k].first];
                const auto& S = I.m_pMesh->m_Levels[0][m_Emitters[k].second];
                const float r = S.m_Radius * I.m_MeanScale;
                Em[k] = { I.m_L2W * S.m_Center, NormalToWorld(I, S.m_Normal), 1.47f * r * r };     // its share of the surface: surfels baked r apart cover ~1.47 r^2 each
            }
            for (int i = 0; i < int(m_Instances.size()); ++i)
            {
                auto& I = m_Instances[i];
                I.m_LinkFirst.clear(); I.m_Links.clear();
                if (!I.m_bAlive || Em.empty()) continue;
                const auto& L = I.m_pMesh->m_Levels[0];
                std::vector<std::vector<emit_link>> Per(L.size());
                ParallelFor(int(L.size()), [&](int s)
                {
                    const v3 C = I.m_L2W * L[s].m_Center, N = NormalToWorld(I, L[s].m_Normal);
                    std::vector<std::pair<float, std::uint32_t>> Cand;
                    for (std::uint32_t k = 0; k < Em.size(); ++k)
                    {
                        if (m_Emitters[k].first == i && m_Emitters[k].second == s) continue;
                        const v3 D = Em[k].m_C - C;
                        const float d2 = D.Dot(D);
                        if (d2 < 1e-10f) continue;
                        const v3 Dn = D * (1.0f / std::sqrt(d2));
                        const float cr = N.Dot(Dn), ce = -Em[k].m_N.Dot(Dn);
                        if (cr <= 0 || ce <= 0) continue;
                        Cand.push_back({ Em[k].m_A * cr * ce / (d2 + Em[k].m_A), k });       // a small disk, not a point: soft up close (no hotspot next to an emitter)
                    }
                    std::sort(Cand.begin(), Cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                    const basis Dirs = BasisDirs(L[s].m_Normal);
                    for (const auto& [w, k] : Cand)
                    {
                        if (int(Per[s].size()) >= max_links_v || w < 1e-5f) break;
                        const v3 D = Em[k].m_C - C;
                        const float d = D.Length();
                        if (Occluded(C + N * 1e-3f, D * (1.0f / d), d * 0.98f)) continue;      // stops short of the emitter's own surface
                        v3 Dm = I.m_W2L.TransformDirection(D * (1.0f / d));
                        Dm = Dm * (1.0f / std::max(Dm.Length(), 1e-12f));
                        Per[s].push_back({ k, w, BasisWeights(Dirs, Dm) });
                    }
                });
                I.m_LinkFirst.assign(L.size() + 1, 0);
                for (std::size_t s = 0; s < L.size(); ++s)
                {
                    I.m_LinkFirst[s + 1] = I.m_LinkFirst[s] + std::uint32_t(Per[s].size());
                    I.m_Links.insert(I.m_Links.end(), Per[s].begin(), Per[s].end());
                }
            }
        }

        // This update's emitted light reaching each surfel (its links x the emitters' current radiance), per basis direction, at the
        // instance's active level (a parent: the mean of its children)
        void UpdateEmittedLight() noexcept
        {
            for (auto& I : m_Instances)
            {
                I.m_EmitB.clear();
                if (!I.m_bAlive || I.m_Links.empty()) continue;
                const auto& M = *I.m_pMesh;
                std::vector<basis> Fine(M.m_Levels[0].size(), basis{});
                for (std::size_t s = 0; s + 1 < I.m_LinkFirst.size(); ++s)
                    for (std::uint32_t k = I.m_LinkFirst[s]; k < I.m_LinkFirst[s + 1]; ++k)
                    {
                        const auto& Lk = I.m_Links[k];
                        const auto& Em = m_Emitters[Lk.m_Emitter];
                        const v3    Le = m_Instances[Em.first].m_SurfEmission[Em.second] * (Lk.m_W * m_Setup.m_EmissiveScale);
                        for (int i = 0; i < 3; ++i) Fine[s][i] = Fine[s][i] + Le * (3.0f * Lk.m_Bw[i]);     // the basis' mean is the irradiance
                    }
                if (I.m_Level == 0) { I.m_EmitB = std::move(Fine); continue; }
                I.m_EmitB.assign(M.m_Levels[1].size(), basis{});
                std::vector<int> Cnt(I.m_EmitB.size(), 0);
                for (std::size_t s = 0; s < Fine.size(); ++s)
                {
                    const int p = M.m_ParentOf[s];
                    for (int i = 0; i < 3; ++i) I.m_EmitB[p][i] = I.m_EmitB[p][i] + Fine[s][i];
                    ++Cnt[p];
                }
                for (std::size_t p = 0; p < Cnt.size(); ++p) if (Cnt[p]) for (int i = 0; i < 3; ++i) I.m_EmitB[p][i] = I.m_EmitB[p][i] * (1.0f / float(Cnt[p]));
            }
        }

        // The surface's colors at a ray hit: the hit triangle's albedo (SetTriangleAlbedo), else its fine surfels' (the hit triangle's
        // vertices' surfels, the reads' weights); the emission the surfels'. The instance's albedo and no emission when it has none
        void SurfaceAt(const hit& H, v3& Albedo, v3& Emission) const noexcept
        {
            const auto& I = m_Instances[H.m_Inst];
            const bool  bTri = std::size_t(H.m_Tri) < I.m_TriAlbedo.size();
            Albedo = bTri ? I.m_TriAlbedo[H.m_Tri] : I.m_Albedo; Emission = { 0, 0, 0 };
            if (I.m_SurfAlbedo.empty() && I.m_SurfEmission.empty()) return;
            const auto& M = *I.m_pMesh; const auto& T = M.m_Tris[H.m_Tri];
            const float Bc[3] = { 1 - H.m_U - H.m_V, H.m_U, H.m_V };
            const v3 P = M.m_Pos[T[0]] * Bc[0] + M.m_Pos[T[1]] * Bc[1] + M.m_Pos[T[2]] * Bc[2];
            v3 SA{ 0, 0, 0 }, SE{ 0, 0, 0 }; float W = 0;
            for (int k = 0; k < 3; ++k)
                for (int s : M.m_VertSurfels[T[k]])
                {
                    if (s < 0) continue;
                    const auto& S = M.m_Levels[0][s];
                    const float c = S.m_Normal.Dot(M.m_Nrm[T[k]]);
                    if (c < 0.5f) continue;
                    const float f = std::max(0.0f, 1.0f - (P - S.m_Center).Length() / (2.0f * S.m_Radius));
                    const float w = (f * f * c + 1e-6f) * Bc[k];
                    if (!I.m_SurfAlbedo.empty())   SA = SA + I.m_SurfAlbedo[s] * w;
                    if (!I.m_SurfEmission.empty()) SE = SE + I.m_SurfEmission[s] * w;
                    W += w;
                }
            if (W <= 0) return;
            if (!I.m_SurfAlbedo.empty() && !bTri) Albedo = SA * (1.0f / W);
            Emission = SE * (m_Setup.m_EmissiveScale / W);
        }

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

        bool                                    m_bLinksDirty = false;
        bool                                    m_bEmitDirty = true;        // UpdateEmittedLight's inputs changed (Update)
        float                                   m_EmitScale = -1;           // the scale it was made with
        std::vector<std::pair<int, int>>        m_Emitters;         // (instance, fine surfel) of every surfel that emits
        mutable worker_pool                     m_Pool;                 // (BasisLight, const, runs on it too)
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
        std::vector<int>                        m_WorkUnsupported;      // per work item: its unsupported queries this update
        std::vector<v3>                         m_WorkUnsupportedAt;    // per work item: the first one's hit point
        std::vector<v3>                         m_UnsupportedAt;        // the last update's (UnsupportedAt)
    };
}

#endif
