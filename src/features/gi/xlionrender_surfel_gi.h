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
#include "dependencies/xscheduler/source/xscheduler.h"
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
#include <unordered_set>
#include <span>
#include <string>
#include <deque>
#include <vector>
#if defined(_M_X64) || defined(_M_IX86) || defined(__SSE2__)
#include <xmmintrin.h>
#endif

namespace xlionrender::gi
{
    using v3 = xmath::fvec3;
    constexpr float pi_v = 3.14159265f;

    // The process' scheduler (xscheduler). A module that has not started its own (LIONRender.dll and its copies: xscheduler's globals and
    // its thread_local worker ids are per module, so the exe's cannot be shared) starts it on first use, from the thread that updates the GI
    inline xscheduler::system& Scheduler() noexcept
    {
        auto& S = xscheduler::g_System;
        if (S.getWorkerCount() == 0) S.Init();
        return S;
    }

    // Fn(0..n-1) across the scheduler's workers; the caller works too, and returns when all are done
    template <class F> void ParallelFor(int n, F&& Fn) noexcept
    {
        if (n <= 0) return;
        auto& S = Scheduler();
        const int nJobs = std::min(n, std::max(S.getWorkerCount(), 1));
        if (nJobs == 1) { for (int i = 0; i < n; ++i) Fn(i); return; }
        std::atomic<int> Next{ 0 };
        xscheduler::task_group G({ "GI" }, S);
        for (int j = 1; j < nJobs; ++j) G.Submit([&] { for (int i; (i = Next++) < n;) Fn(i); });
        for (int i; (i = Next++) < n;) Fn(i);
        G.join();
    }
    inline int ParallelWidth() noexcept { return std::max(Scheduler().getWorkerCount(), 1); }     // the workers ParallelFor spreads over

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
    enum class role : std::uint8_t { ASSET = 0, RECEIVER = 1, FULL = 2 };     // Phase 5: scene::gi_role (an instance's transport, over its mesh's m_bBounce)
    constexpr int warmup_passes_v = 16;     // after a change: the estimator's first samples are averaged (1 / n), its multiscale blend after
    constexpr int fallback_cap_v  = 32;     // the most list entries an unsupported query walks (surfel_mesh::SortReadLists: the nearest first)
    constexpr int running_cap_v   = 256;    // the multiscale blend never below 1 / n of the samples since the change, up to this (estimator::Add)

    // A color in 32 bits (RGB9E5: three 9-bit mantissas, a shared 5-bit exponent; VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, as Godot's probes): the
    // shown light the pixels read (gi_dynamic_master_plan.md 7a, the compression survey). Steps of 2^-9 of the largest channel; 0 to 65408;
    // negative and NaN are 0. Every step is exact in float (powers of two, +0.5, floor), so the shaders' (xlionrender_gi_filter_comp.glsl,
    // xlionrender_surfel_gi.glsl) give the same bits
    inline std::uint32_t PackRGB9E5(const v3& c) noexcept
    {
        const auto Clamp = [](float x) { return x > 0 ? std::min(x, 65408.0f) : 0.0f; };       // (NaN > 0 is false)
        const float r = Clamp(c.m_X), g = Clamp(c.m_Y), b = Clamp(c.m_Z), m = std::max(r, std::max(g, b));
        const int Floor = m > 0 ? int((std::bit_cast<std::uint32_t>(m) >> 23) & 0xFFu) - 127 : -16;     // floor(log2(m)) (a denormal: very small)
        int e = std::max(-16, Floor) + 16;                                                    // the shared exponent (bias 15) + 1
        float s = std::bit_cast<float>(std::uint32_t(127 - (e - 24)) << 23);                  // 2^-(e - 15 - 9)
        if (std::floor(m * s + 0.5f) >= 512.0f) { ++e; s *= 0.5f; }
        const auto M = [&](float x) { return std::uint32_t(std::floor(x * s + 0.5f)); };
        return M(r) | (M(g) << 9) | (M(b) << 18) | (std::uint32_t(e) << 27);
    }
    inline v3 UnpackRGB9E5(std::uint32_t p) noexcept
    {
        const float s = std::bit_cast<float>(std::uint32_t(127 + int(p >> 27) - 24) << 23);
        return v3{ float(p & 511u) * s, float((p >> 9) & 511u) * s, float((p >> 18) & 511u) * s };
    }

    // A surfel's light over time: one estimator for both regimes (documentation/gi_dynamic_master_plan.md, Phase 2b). After a change
    // (Restart) the first warmup_passes_v samples are averaged exactly (1 / n: the fastest unbiased start), then the multiscale blend
    // (GIBS's MultiscaleMeanEstimator) takes over from that mean
    struct estimator
    {
        v3              m_Mean{ 0, 0, 0 }, m_Short{ 0, 0, 0 }, m_Variance{ 0, 0, 0 };
        float           m_Vbbr          = 1;        // the variance-based blend reduction
        float           m_Inconsistency = 0;
        std::uint16_t   m_N             = 0;        // samples since the last Restart (counts to warmup_passes_v); its confidence
        std::int8_t     m_Streak        = 0;        // Lower: the sign of the last sample's large difference to the mean (0: none)
        std::uint16_t   m_nTaken        = 0;        // (tests) the changes it took (two large differences: their mean replaced it)
        bool            m_bWatch        = false;    // Lower: watching for a real change (two large differences the same way)
        bool            m_bTook         = false;    // a change taken since the Lower: what the next one differs from follows the mean
        v3              m_Last{ 0, 0, 0 };          // the last sample, while watching
        v3              m_Ref{ 0, 0, 0 };           // the light when the change came (what a real change differs from)
        float           m_PrevLuma = 0, m_DiffSq = 0;   // while watching: the last sample's luminance, the sum of the squared steps between
        std::uint16_t   m_nDiff = 0;                    // samples that did not deviate, and their count (the samples' noise: a change is beyond it)

        // A value carried over (a level switch, a new layout): it continues as settled light
        void Seed(const v3& E) noexcept { m_Mean = m_Short = E; m_Variance = { 0, 0, 0 }; m_Vbbr = 1; m_Inconsistency = 0; m_N = warmup_passes_v; }
        void Restart() noexcept { m_N = 0; }        // nothing to keep (a new surfel): the next sample replaces it

        // Phase 4: a change affects it by a (0..1): its confidence drops to (1 - a) of the warm-up, one sample at least, so a single
        // noisy pass never replaces its light (at most half of it); what it had is the warm start. Until it is confident again it
        // watches for a real change: two samples in a row that differ from the light it had by half of it, the same way, replace it with
        // their own mean (two passes of evidence, as production DDGI rejects history on a large change)
        void Lower(float a) noexcept
        {
            const int k = std::max(1, int((1.0f - std::clamp(a, 0.0f, 1.0f)) * float(warmup_passes_v)));
            if (m_N > k) m_N = std::uint16_t(k);
            if (m_N > 0) { m_bWatch = true; m_bTook = false; m_Streak = 0; m_Ref = m_Mean; m_DiffSq = 0; m_nDiff = 0; m_PrevLuma = -1; }
        }

        // MinBlend: the warm-up's blend never below it (1: every sample replaces, no history)
        void Add(v3 y, float MinBlend = 0, float ShortBlend = 0.08f) noexcept
        {
            if (MinBlend >= 1) { Seed(y); return; }
            if (m_bWatch)
            {
                const auto Luma = [](const v3& c) { return 0.299f * c.m_X + 0.587f * c.m_Y + 0.114f * c.m_Z; };
                // a change differs from the light by half of it and by three times the samples' noise (from the steps between those
                // that did not deviate: a step is about sqrt(2) noise for noise, small for a ramp, one jump for a change of level)
                const float Ly = Luma(y), D = Ly - Luma(m_Ref);
                const float Noise = m_nDiff >= 3 ? std::sqrt(m_DiffSq / (2.0f * float(m_nDiff))) : 0.0f;
                const std::int8_t Sign = std::abs(D) > std::max(0.5f * std::max(Luma(m_Ref), 1e-3f), 3.0f * Noise) ? (D > 0 ? 1 : -1) : 0;
                if (Sign == 0 && m_PrevLuma >= 0) { m_DiffSq += (Ly - m_PrevLuma) * (Ly - m_PrevLuma); ++m_nDiff; }
                m_PrevLuma = Ly;
                // two in a row the same way that agree with each other (a new level, not noise: a noisy surfel's samples rarely agree).
                // Taken, what the next one differs from is the warm-up's mean from there (it follows a ramp, take by take). It used to
                // stay the light before the change: a noisy surfel took a "change" every few passes, its light two samples' mean for
                // ever - the flicker (Police: 2/3 of its surfels)
                if (Sign != 0 && Sign == m_Streak && std::abs(Luma(y) - Luma(m_Last)) < 0.5f * std::abs(D) && (!m_bTook || m_N >= 4))     // (after a take, 4 samples of its mean first)
                {
                    m_Mean = m_Short = (m_Last + y) * 0.5f; m_Variance = { 0, 0, 0 }; m_N = 2; m_Streak = 0; m_Last = y; ++m_nTaken;
                    m_bTook = true; m_Ref = m_Mean;
                    return;
                }
                m_Streak = Sign; m_Last = y;
            }
            if (m_N < warmup_passes_v)
            {
                ++m_N;
                const float a = std::max(1.0f / float(m_N), MinBlend);
                m_Mean = m_Mean + (y - m_Mean) * a;
                // the multiscale blend starts from no variance (as before the merge): the light is still building up (each bounce
                // lights the next), and a large variance measured from 8-ray samples would make it slow to follow
                m_Short = m_Mean; m_Variance = { 0, 0, 0 }; m_Vbbr = 1; m_Inconsistency = 0;
                if (m_bTook) m_Ref = m_Mean;
                if (m_N >= warmup_passes_v) m_bWatch = false;
                return;
            }
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
            // The running mean under the blend (gi_dynamic_master_plan.md 7a): the samples keep counting past the warm-up, up to
            // running_cap_v, and the mean follows each by 1 / n at least. The multiscale blend alone fell to 1 / 256 x Vbbr (as low as
            // 1 / 8192) while the bounces still built up below the samples' noise: the light froze short of its level (the room 8 % off
            // after 200 updates, 5.8 % with this; the flicker test's limits hold)
            if (m_N < running_cap_v) ++m_N;
            m_Mean = m_Mean + (y - m_Mean) * std::clamp(std::max(CatchUp, 1.0f / float(m_N)), 0.0f, 1.0f);
        }
    };
    constexpr int link_unit_first_v = 16;   // the link build's first chunk of receiver surfels, before their cost is measured
    constexpr int dead_retrace_v  = 8;      // a buried surfel (validity 0) is traced one pass in this many (Update)
    // A surfel far from the eye is traced less often: its footprint on the screen is about (radius / distance)^2 pixels, so a far one
    // is a few pixels and a near one is many. The period by distance in radii; off screen or behind the eye it still bounces light onto what is seen, so it is only
    // slower, never dropped: every surfel is traced at least once in dead_retrace_v passes
    constexpr float far_tier_radii_v[3] = { 12.0f, 24.0f, 48.0f };      // beyond each: the period doubles (1, 2, 4, 8)

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
        // Leak protection (the GeomStatic descriptor's GI LeakProtection): the read grid's patches split where the surface is cut (a wall
        // across a cell: BuildReadGrid), and the solver's rays read through it (world::ReadPoint), not the hit triangle's vertices
        bool                                        m_bLeakProtection = true;
        // Defects (geometry the GI cannot serve: a thin face beside other geometry - a strip of ceiling between a beam and a wall - that no
        // surfel serves). BuildReadGrid lists the seen pieces of surface whose read lists nothing (a point and its normal each); the compiler
        // can bake a surfel for each (Bake's Forced: PatchDefects). The fine surfels from m_FirstPatch on are those (0xFFFFFFFF: none)
        std::vector<std::pair<v3, v3>>              m_EmptyParts;
        std::uint32_t                               m_FirstPatch = 0xFFFFFFFFu;
        // The last Bake's stages, ms (the compiler's autopsy): candidates, fine darts, materials, vertex surfels, parents, neighbours
        std::array<double, 6>                       m_BakeMs{};
        // Bake's progress, 0..1, after each of its stages (the compiler's progress bar; the stages weighed as the Bistro's took: the
        // vertex surfels, the parents and the neighbours most). Empty: none
        std::function<void(float)>                  m_OnBakeProgress;
        // ... inside a stage too (its loops in slices: BakeTick, SlicedFor): the bar moves every few seconds, not once a minute
        float                                       m_BakeLo = 0, m_BakeHi = 1;
        mutable float                               m_BakeLast = -1;
        void BakeTick(float f) const noexcept
        {
            if (!m_OnBakeProgress) return;
            const float v = m_BakeLo + (m_BakeHi - m_BakeLo) * std::clamp(f, 0.0f, 1.0f);
            if (v - m_BakeLast < 0.005f) return;
            m_BakeLast = v;
            m_OnBakeProgress(v);
        }
        // ParallelFor over [0, n) in 20 slices, a tick after each (f0..f1 of the stage)
        template <class F> void SlicedFor(int n, F&& Fn, float f0 = 0, float f1 = 1) const noexcept
        {
            constexpr int slices_v = 20;
            for (int k = 0; k < slices_v; ++k)
            {
                const int a = int(std::int64_t(n) * k / slices_v), b = int(std::int64_t(n) * (k + 1) / slices_v);
                if (b > a) ParallelFor(b - a, [&](int i) { Fn(a + i); });
                BakeTick(f0 + (f1 - f0) * float(k + 1) / float(slices_v));
            }
        }

        static std::array<int, 3> ReadCellOf(const v3& P, float Cell) noexcept { return { int(std::floor(P.m_X / Cell)), int(std::floor(P.m_Y / Cell)), int(std::floor(P.m_Z / Cell)) }; }
        static std::array<std::uint32_t, 2> ReadKey(const std::array<int, 3>& C) noexcept
        {
            const std::uint64_t K = surfel_grid::Key(C[0], C[1], C[2]);
            return { std::uint32_t(K), std::uint32_t(K >> 32) };
        }
        // The fallback's bound (GI review, experiment A): every patch's list nearest first, from its cell's center, so an unsupported query's
        // walk (world::Unsupported, the pass's Unsupported, SurfelGILocalMean) can stop after fallback_cap_v entries and still take the
        // nearest. Called after the grid is built or unpacked; the reads that sum a whole list are unchanged (the order of a sum)
        void SortReadLists() noexcept
        {
            if (std::getenv("XLION_GI_GRIDSTATS"))     // measurement: each loaded or built read grid (GI review)
            {
                std::size_t nCells = 0, nPatch = 0, MaxPatch = 0, nEnt = 0, MaxEnt = 0, nPast32 = 0, nPast16 = 0;
                for (const auto& E : m_ReadTable) { if (E[1] == 0xFFFFFFFFu) continue; ++nCells; nPatch += E[3]; MaxPatch = std::max<std::size_t>(MaxPatch, E[3]);
                    for (std::uint32_t p = 0; p < E[3]; ++p) { const std::size_t n = m_ReadList[E[2] + 4 * p + 3] & 0xFFFFu; nEnt += n; MaxEnt = std::max(MaxEnt, n); nPast32 += n > 32 ? n - 32 : 0; nPast16 += n > 16 ? n - 16 : 0; } }
                std::printf("GIGRID cells %zu (table %zu), patches/cell mean %.1f max %zu, entries/patch mean %.1f max %zu (past 32: %zu, past 16: %zu of %zu), cell %.3f, surfels %zu\n", nCells, m_ReadTable.size(),
                            nCells ? double(nPatch) / nCells : 0.0, MaxPatch, nPatch ? double(nEnt) / nPatch : 0.0, MaxEnt, nPast32, nPast16, nEnt, m_ReadCell, m_Levels.empty() ? 0 : m_Levels[0].size()); std::fflush(stdout);
            }
            const auto Sx = [](std::uint64_t v) { const int i = int(v & 0x1FFFFF); return i >= 0x100000 ? i - 0x200000 : i; };
            for (const auto& E : m_ReadTable)
            {
                if (E[1] == 0xFFFFFFFFu) continue;
                const std::uint64_t K = std::uint64_t(E[0]) | (std::uint64_t(E[1]) << 32);
                const v3 Cc{ (float(Sx(K >> 42)) + 0.5f) * m_ReadCell, (float(Sx(K >> 21)) + 0.5f) * m_ReadCell, (float(Sx(K)) + 0.5f) * m_ReadCell };
                for (std::uint32_t p = 0; p < E[3]; ++p)
                {
                    const std::uint32_t First = m_ReadList[E[2] + 4 * p + 2], n = m_ReadList[E[2] + 4 * p + 3] & 0xFFFFu;
                    if (n < 2 || First + n > m_ReadList.size()) continue;
                    const auto D2 = [&](std::uint32_t e) { const v3 d = m_Levels[0][e & 0xFFFFu].m_Center - Cc; return d.Dot(d); };
                    std::stable_sort(m_ReadList.begin() + First, m_ReadList.begin() + First + n, [&](std::uint32_t a, std::uint32_t b) { return D2(a) < D2(b); });
                }
            }
            RefreshPackedReadGrid();
        }
        static std::uint32_t ReadHash(const std::array<int, 3>& C) noexcept { return (std::uint32_t(C[0]) * 73856093u) ^ (std::uint32_t(C[1]) * 19349663u) ^ (std::uint32_t(C[2]) * 83492791u); }

        // The cell of a surface point: lifted a little off its surface first (read_lift_v of a cell along N). A floor at a multiple of the
        // cell size lies on a boundary, and the GPU's interpolated position falls a hair to either side of it: lifted, it does not
        static constexpr float read_lift_v = 0.05f;
        static std::array<int, 3> ReadCellAt(const v3& P, const v3& N, float Cell) noexcept { return ReadCellOf(P + N * (read_lift_v * Cell), Cell); }

        // A list entry: the surfel (low 16 bits) and how much the surface sees it at the 4 corners of the cell in the patch's plane (4 bits
        // each, 0..15): at a point, blended by its place in the cell (as GIReadWeight)
        static std::uint32_t ReadEntrySurfel(std::uint32_t E) noexcept { return E & 0xFFFFu; }

        // The read grid as the GPU and the resource keep it (PackReadGrid): a part's list of 1..fallback_cap_v entries keeps the entries with
        // corner weights as they are, then its all-seen ones (every corner 15: only the surfel to say) two to a word, 16 bits each; its count
        // (header word 3, low 16 bits) is then read_packed_v | the first count | the second << 7. Longer lists stay as they are (a count
        // below read_packed_v). The Bistro: 61% of its lists' entries all-seen, its sections' lists 143 MB -> about 97
        static constexpr std::uint32_t read_packed_v = 0x8000u;
        static std::uint32_t ReadListCount(std::uint32_t Code) noexcept { Code &= 0xFFFFu; return (Code & read_packed_v) ? (Code & 0x7Fu) + ((Code >> 7) & 0x7Fu) : Code; }
        static void PackReadGrid(bool bPack, const std::vector<std::array<std::uint32_t, 4>>& Table, const std::vector<std::uint32_t>& List,
                                 std::vector<std::array<std::uint32_t, 4>>& OutTable, std::vector<std::uint32_t>& OutList) noexcept
        {
            OutTable = Table; OutList.clear(); OutList.reserve(List.size());
            std::vector<std::uint32_t> Halves, Order;           // (the cells in the lists' order: unpacked again, the same layout)
            for (std::uint32_t k = 0; k < OutTable.size(); ++k) if (OutTable[k][1] != 0xFFFFFFFFu) Order.push_back(k);
            std::sort(Order.begin(), Order.end(), [&](std::uint32_t x, std::uint32_t y) { return Table[x][2] < Table[y][2]; });
            for (const std::uint32_t k : Order)
            {
                auto& E = OutTable[k];
                if (E[1] == 0xFFFFFFFFu || std::size_t(E[2]) + 4 * std::size_t(E[3]) > List.size()) continue;
                const std::uint32_t H0 = E[2], At = std::uint32_t(OutList.size());
                OutList.insert(OutList.end(), List.begin() + H0, List.begin() + H0 + 4 * E[3]);
                E[2] = At;
                for (std::uint32_t p = 0; p < E[3]; ++p)
                {
                    const std::uint32_t First = List[H0 + 4 * p + 2], Code = List[H0 + 4 * p + 3] & 0xFFFFu, Mask = List[H0 + 4 * p + 3] & 0xFFFF0000u;
                    const std::uint32_t L0 = std::uint32_t(OutList.size());
                    std::uint32_t NewCode = Code;
                    if (!bPack && (Code & read_packed_v))
                    {
                        const std::uint32_t nF = Code & 0x7Fu, nH = (Code >> 7) & 0x7Fu;
                        OutList.insert(OutList.end(), List.begin() + First, List.begin() + First + nF);
                        for (std::uint32_t h = 0; h < nH; ++h) { const std::uint32_t w = List[First + nF + h / 2]; OutList.push_back(((h & 1) ? (w >> 16) : (w & 0xFFFFu)) | 0xFFFF0000u); }
                        NewCode = nF + nH;
                    }
                    else if (bPack && Code >= 1 && Code <= std::uint32_t(fallback_cap_v))
                    {
                        Halves.clear();
                        for (std::uint32_t k = 0; k < Code; ++k) { const std::uint32_t e = List[First + k]; if ((e >> 16) == 0xFFFFu) Halves.push_back(e & 0xFFFFu); else OutList.push_back(e); }
                        const std::uint32_t nF = Code - std::uint32_t(Halves.size());
                        for (std::size_t h = 0; h < Halves.size(); h += 2) OutList.push_back(Halves[h] | (h + 1 < Halves.size() ? Halves[h + 1] << 16 : 0u));
                        NewCode = read_packed_v | nF | (std::uint32_t(Halves.size()) << 7);
                    }
                    else OutList.insert(OutList.end(), List.begin() + First, List.begin() + First + Code);
                    OutList[At + 4 * p + 2] = L0;
                    OutList[At + 4 * p + 3] = Mask | NewCode;
                }
            }
        }
        // The packed form of the lists (the GPU's display and trace buffers): made again whenever the lists change (SortReadLists, and the
        // chunks' remap). ponytail: a second copy on the CPU (the Bistro: ~97 MB more); the CPU reads could walk the packed form instead
        std::vector<std::array<std::uint32_t, 4>>   m_PackedReadTable;
        std::vector<std::uint32_t>                  m_PackedReadList;
        void RefreshPackedReadGrid() noexcept { PackReadGrid(true, m_ReadTable, m_ReadList, m_PackedReadTable, m_PackedReadList); }
        // Experiment (XLION_GI_SPARSEGRID=<ratio>): the read grid only where the triangles are coarse - a triangle whose longest edge passes
        // Ratio x the surfels' radius keeps the cells it crosses (and their neighbors); the others' cells are emptied (no patch: the pixel
        // reads its triangle's vertex surfels, as off the grid). The vertex read is the grid's within 2% on dense meshes (gi_render)
        void SparsifyReadGrid(float Ratio) noexcept
        {
            if (m_ReadTable.empty() || m_Levels[0].empty()) return;
            const float Cell = m_ReadCell, Long = Ratio * m_Levels[0][0].m_Radius;
            std::unordered_set<std::uint64_t> Keep;
            for (const auto& T : m_Tris)
            {
                const v3 A = m_Pos[T[0]], B = m_Pos[T[1]], C = m_Pos[T[2]];
                const float Edge = std::max({ (B - A).Length(), (C - B).Length(), (A - C).Length() });
                if (Edge <= Long) continue;
                v3 N = (B - A).Cross(C - A); const float l = N.Length(); if (l <= 0) continue; N = N * (1.0f / l);
                const int n = std::max(1, int(std::ceil(Edge / (0.5f * Cell))));
                for (int i = 0; i <= n; ++i) for (int j = 0; i + j <= n; ++j)
                {
                    const v3 P = A + (B - A) * (float(i) / n) + (C - A) * (float(j) / n);
                    const auto Ck = ReadCellAt(P, N, Cell);
                    for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz)
                        Keep.insert(surfel_grid::Key(Ck[0] + dx, Ck[1] + dy, Ck[2] + dz));
                }
            }
            std::vector<std::uint32_t> List;
            List.reserve(m_ReadList.size() / 4);
            for (auto& E : m_ReadTable)
            {
                if (E[1] == 0xFFFFFFFFu) continue;
                if (!Keep.contains(std::uint64_t(E[0]) | (std::uint64_t(E[1]) << 32))) { E[2] = 0; E[3] = 0; continue; }
                const std::uint32_t At = std::uint32_t(List.size());
                List.insert(List.end(), m_ReadList.begin() + E[2], m_ReadList.begin() + E[2] + 4 * E[3]);
                for (std::uint32_t p = 0; p < E[3]; ++p)
                {
                    const std::uint32_t First = m_ReadList[E[2] + 4 * p + 2], n = m_ReadList[E[2] + 4 * p + 3] & 0xFFFFu;
                    List[At + 4 * p + 2] = std::uint32_t(List.size());
                    List.insert(List.end(), m_ReadList.begin() + First, m_ReadList.begin() + First + n);
                }
                E[2] = At;
            }
            if (std::getenv("XLION_GI_GRIDSTATS")) std::printf("GISPARSE ratio %.1f: lists %zu -> %zu words (%.1f%%)\n", Ratio, m_ReadList.size(), List.size(), 100.0 * double(List.size()) / std::max<double>(1, double(m_ReadList.size()))), std::fflush(stdout);
            m_ReadList = std::move(List);
            RefreshPackedReadGrid();
        }
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
        std::span<const std::uint32_t> ReadCellList(const v3& P, const v3& N, v3* pPatchN = nullptr) const noexcept
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
                // The part whose sub-cells hold the point first (a wall across the cell: the point's own side), then facing, then nearest.
                // Every ray's hit: plain floats (UnpackOct, CellUV and MaskDistance inline; a Debug build calls every v3 operator)
                float Best = -1e30f; std::uint32_t Bp = 0xFFFFFFFFu;
                const float iCell = 1.0f / m_ReadCell, Nx = N.m_X, Ny = N.m_Y, Nz = N.m_Z;
                const float Lp[3] = { (P.m_X + Nx * (read_lift_v * m_ReadCell)) * iCell, (P.m_Y + Ny * (read_lift_v * m_ReadCell)) * iCell, (P.m_Z + Nz * (read_lift_v * m_ReadCell)) * iCell };
                const float Fr[3] = { Lp[0] - std::floor(Lp[0]), Lp[1] - std::floor(Lp[1]), Lp[2] - std::floor(Lp[2]) };
                const std::uint32_t* pList = m_ReadList.data() + E[2];
                for (std::uint32_t p = 0; p < E[3]; ++p)
                {
                    const std::uint32_t* Pa = pList + 4 * p;
                    float x = std::max(float(std::int16_t(Pa[0] & 0xFFFF)) / 32767.0f, -1.0f), y = std::max(float(std::int16_t(Pa[0] >> 16)) / 32767.0f, -1.0f);
                    float z = 1 - std::abs(x) - std::abs(y);
                    if (z < 0) { const float ox = x; x = (1 - std::abs(y)) * (ox >= 0 ? 1.0f : -1.0f); y = (1 - std::abs(ox)) * (y >= 0 ? 1.0f : -1.0f); }
                    const float il = 1.0f / std::sqrt(x * x + y * y + z * z); x *= il; y *= il; z *= il;
                    const float c = x * Nx + y * Ny + z * Nz;
                    if (c < 0.5f) continue;
                    const float Upper = c - std::abs(P.m_X * x + P.m_Y * y + P.m_Z * z - std::bit_cast<float>(Pa[1])) * iCell + 2.0f;     // (its score inside its sub-cells)
                    if (Upper <= Best) continue;                                                     // (it cannot win: GIReadCell the same)
                    const float ax = std::abs(x), ay = std::abs(y), az = std::abs(z);
                    const int   Au = ax >= ay && ax >= az ? 1 : 0, Av = ax >= ay && ax >= az ? 2 : ay >= az ? 2 : 1;      // ReadAxes
                    const float uv[2] = { Fr[Au], Fr[Av] };
                    const std::uint32_t Msk = Pa[3] >> 16;
                    const int su = std::min(int(uv[0] * sub_v), sub_v - 1), sv = std::min(int(uv[1] * sub_v), sub_v - 1);
                    const float D = (Msk >> (sv * sub_v + su)) & 1u ? 0.0f : MaskDistance(Msk, { uv[0], uv[1] });
                    const float Score = Upper - 2.0f * std::min(1.0f, D);
                    if (Score > Best) { Best = Score; Bp = p; }
                }
                if (Bp == 0xFFFFFFFFu) return {};
                const std::uint32_t* Pa = &m_ReadList[E[2] + 4 * Bp];
                if (pPatchN) *pPatchN = UnpackOct(Pa[0]);
                return { m_ReadList.data() + Pa[2], Pa[3] & 0xFFFFu };
            }
            return {};
        }

        // A surface point never seen (BuildReadGrid's points, Coverage): something just in front of it (within a quarter of a read cell
        // along its normal: a wall under a door frame), or enclosed - inside a hollow wall (two faces, no bottom: the floor runs on inside
        // it). Inside, a short ray along the surface meets a face from behind; outside it never does (a wall's foot is met from the front)
        bool CoveredAt(const v3& P, const v3& N) const noexcept
        {
            const v3 O = P + N * (0.02f * m_ReadCell);
            if (Occluded(O, N, 0.25f * m_ReadCell)) return true;
            const v3 T = (std::abs(N.m_X) < 0.9f ? v3{ 1, 0, 0 } : v3{ 0, 1, 0 }).Cross(N).NormalizeCopy(), B = N.Cross(T);
            for (const v3& D : { T, T * -1.0f, B, B * -1.0f })
            {
                float Len = 0.25f * m_ReadCell, U, V; int Tri;
                if (!Cast(O, D, Len, Tri, U, V)) continue;
                const auto& Tr = m_Tris[Tri];
                if ((m_Pos[Tr[1]] - m_Pos[Tr[0]]).Cross(m_Pos[Tr[2]] - m_Pos[Tr[0]]).Dot(D) > 0) return true;     // a back face: inside
            }
            return false;
        }

        // How much of the surface the per-pixel read covers: random points of every triangle, by its area (the interpolated vertex normal, a hair off
        // the surface as the GPU's interpolation is), each read through its cell: uncovered when no listed surfel reaches it with a
        // compatible normal (black in the SURFELS debug view). The first uncovered points are kept, with what is near them
        struct coverage
        {
            struct example { v3 m_P, m_N; int m_nList; float m_NearestCompatible, m_NearestAny; };
            int                     m_nPoints = 0, m_nUncovered = 0, m_nHidden = 0;    // m_nHidden: covered (CoveredAt), never seen: not counted
            double                  m_ListSum = 0;  int m_ListMax = 0;                  // the lists' lengths (each read's cost: the solver reads one per ray hit)
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
                    Out.m_ListSum += double(List.size()); Out.m_ListMax = std::max(Out.m_ListMax, int(List.size()));
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

        // A patch's sub-cells (sub_v x sub_v across the cell, in the patch's plane: ReadAxes) and the one a surface point (its normal, as
        // the reads lift it) falls in: its bit in a patch's mask (shaders/xlionrender_surfel_gi.glsl GIReadCell)
        static constexpr int sub_v = 4;
        static std::array<float, 2> CellUV(const v3& P, const v3& N, const v3& PatchN, float Cell) noexcept     // where in its cell, 0..1
        {
            const v3 Lp = (P + N * (read_lift_v * Cell)) * (1.0f / Cell);
            const auto [Au, Av, Aa] = ReadAxes(PatchN);
            const float* pL = &Lp.m_X;
            return { pL[Au] - std::floor(pL[Au]), pL[Av] - std::floor(pL[Av]) };
        }
        static int SubCell(const v3& P, const v3& N, const v3& PatchN, float Cell) noexcept
        {
            const auto uv = CellUV(P, N, PatchN, Cell);
            return std::clamp(int(uv[1] * sub_v), 0, sub_v - 1) * sub_v + std::clamp(int(uv[0] * sub_v), 0, sub_v - 1);
        }
        // How far (sub-cells) a point (CellUV) is from a part's sub-cells (its mask): 0 inside one. The part a point reads is the nearest
        // one: where no part holds its sub-cell (the strip by a wall's foot no sample fell in) its own side is still the nearer
        static float MaskDistance(std::uint32_t Mask, const std::array<float, 2>& uv) noexcept
        {
            const int su = std::clamp(int(uv[0] * sub_v), 0, sub_v - 1), sv = std::clamp(int(uv[1] * sub_v), 0, sub_v - 1);
            if (Mask & (1u << (sv * sub_v + su))) return 0;             // inside one (the usual case)
            float Best = 1e30f;
            for (int b = 0; b < sub_v * sub_v; ++b)
            {
                if (!(Mask & (1u << b))) continue;
                const float u0 = float(b % sub_v) / sub_v, v0 = float(b / sub_v) / sub_v, e = 1.0f / sub_v;
                const float du = std::max({ u0 - uv[0], uv[0] - (u0 + e), 0.0f }), dv = std::max({ v0 - uv[1], uv[1] - (v0 + e), 0.0f });
                Best = std::min(Best, std::max(du, dv) * sub_v);
            }
            return Best;
        }

        // The read grid's cell: the fine surfels' radius, halved while its cells hold more than read_patches_v patches on average (at most
        // read_halvings_v times). Each pixel scores its cell's patches (GIReadCell): a crumpled surface (the Golem, 2 m surfels on a 4 m
        // model) put 300 of them in each of its 24 cells, 30 ms a frame maximized. The lists stay the surfels whose reach touches the cell
        static constexpr float read_patches_v  = 8;
        static constexpr int   read_halvings_v = 4;
        // A part's list: its nearest read_list_cap_v seen surfels from its cell's center (the per-pixel read walks the 32 nearest:
        // gi_pixel_cap_v). The Bistro's layered surface listed thousands a part: 5 GB of lists, 10 min of rays, a 19 GB peak
        static constexpr std::size_t read_list_cap_v = 32;
        // ... and the facing surfels a part asks (its rays: up to 16 each), nearest first: a part of a dense Bistro section asked thousands, hidden
        // behind each other (121 s for its lists)
        static constexpr int read_tests_v = 256;
        std::size_t m_ReadListCap = read_list_cap_v;        // (tests: ~0 builds the unbounded lists, the reference)
        static constexpr int near_tests_v = 256;            // Bake's nearest seen surfels of a vertex / a surfel's parent: the rays it may cast
        float ReadPatchesPerCell() const noexcept
        {
            std::size_t nCells = 0, nPatches = 0;
            for (const auto& E : m_ReadTable) if (E[1] != 0xFFFFFFFFu) { ++nCells; nPatches += E[3]; }
            return nCells ? float(nPatches) / float(nCells) : 0.0f;
        }
        void BuildReadGrid() noexcept
        {
            m_ReadTable.clear(); m_ReadList.clear(); m_PackedReadTable.clear(); m_PackedReadList.clear(); m_ReadMaxProbe = 0; m_ReadUnsupported = 0;
            if (m_Levels[0].empty()) return;
            // The cell: halved while the surface's patches alone (BuildReadGridAt's first step: no lists, no rays) are too many per cell
            float Cell = m_Levels[0][0].m_Radius;
            const auto T0 = std::chrono::steady_clock::now();
            for (int k = 0; k < read_halvings_v && BuildReadGridAt(Cell, true) > read_patches_v; ++k) Cell *= 0.5f;
            m_ReadGridMs = {}; m_ReadGridMs[0] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();
            BuildReadGridAt(Cell);
        }
        std::array<double, 4> m_ReadGridMs{};       // the last BuildReadGrid's stages, ms (measurement): the cell's probes, the samples and their patches, the parts, the lists
        // bPatches: only the cells' patches (their mean per cell returned; the grid is left empty)
        float BuildReadGridAt(float CellSize, bool bPatches = false) noexcept
        {
            m_ReadTable.clear(); m_ReadList.clear(); m_PackedReadTable.clear(); m_PackedReadList.clear(); m_ReadMaxProbe = 0; m_ReadUnsupported = 0;
            const auto& L = m_Levels[0];
            if (L.empty()) return 0;
            m_ReadCell = CellSize;
            const float Cell = m_ReadCell, Reach = 2 * L[0].m_Radius;
            const auto TGrid0 = std::chrono::steady_clock::now();

            // The cells the surface crosses (points on every triangle an eighth of a cell apart), and per cell its patches: the pieces of
            // surface in it that face one way in one plane (the blinds and the window frame 10 cm behind them are two patches, whichever
            // way they face). Per patch, per sub-cell, a point of it in sight - or a covered one (CoveredAt: under a door frame, never seen)
            // where none is: a face covered all over (sealed in) still has points
            struct patch
            {
                v3 m_N; float m_D;
                std::array<v3, sub_v * sub_v> m_Vis{}, m_Hid{};
                std::uint16_t m_VisMask = 0, m_HidMask = 0;
            };
            // A patch's connected piece: its sub-cells joined where the surface between them is not cut (a wall across the cell cuts the
            // floor in two: each side reads only what it sees; one list for both was the light of one room on the other's floor)
            struct part  { v3 m_N; float m_D; std::uint16_t m_Mask; std::vector<v3> m_Pts; bool m_bSeen = true; };
            struct cell  { std::array<int, 3> m_C; std::vector<patch> m_Patches; std::vector<part> m_Parts; };
            std::unordered_map<std::uint64_t, cell> Cells;
            const float PlaneTol = 0.1f * Cell;
            // A point in sight also stands for the cells next to its own when it is within a sample's spacing of their edge (their sub-cell at
            // that edge): a cell the surface only grazes - a strip of floor at a wall's foot thinner than the spacing, the rest of the cell
            // under the wall - has no sample of its own in sight, and its patch was covered all over (an empty list: a purple line at the seam)
            const float Edge = 1.0f / (2 * sub_v);                         // the samples' spacing, in cells
            const auto PatchIn = [&](const std::array<int, 3>& C, const v3& P, const v3& N) -> patch&
            {
                auto& Ce = Cells.try_emplace(surfel_grid::Key(C[0], C[1], C[2]), cell{ C, {}, {} }).first->second;
                for (auto& Pa : Ce.m_Patches) if (Pa.m_N.Dot(N) > 0.97f && std::abs(P.Dot(Pa.m_N) - Pa.m_D) < PlaneTol) return Pa;
                Ce.m_Patches.push_back({});
                auto& Pa = Ce.m_Patches.back(); Pa.m_N = N; Pa.m_D = P.Dot(N);
                return Pa;
            };
            const auto Touch = [&](const v3& P, const v3& N, bool bCovered)      // (bCovered: CoveredAt(P, N), asked before)
            {
                const auto C  = ReadCellAt(P, N, Cell);
                patch& Pa = PatchIn(C, P, N);
                const int b = SubCell(P, N, Pa.m_N, Cell);
                const auto Bit = std::uint16_t(1u << b);
                if (Pa.m_VisMask & Bit) return;                            // (it has its point in sight)
                if (bCovered) { if (!(Pa.m_HidMask & Bit)) { Pa.m_HidMask |= Bit; Pa.m_Hid[b] = P; } return; }
                Pa.m_VisMask |= Bit; Pa.m_Vis[b] = P;
                // ... and the neighbor cells across the edges it is near (in the patch's plane: ReadAxes)
                const auto [Au, Av, Aa] = ReadAxes(Pa.m_N);
                const v3 Lp = (P + N * (read_lift_v * Cell)) * (1.0f / Cell);
                const float* pL = &Lp.m_X;
                const float fu = pL[Au] - std::floor(pL[Au]), fv = pL[Av] - std::floor(pL[Av]);
                const int du = fu < Edge ? -1 : fu > 1 - Edge ? 1 : 0, dv = fv < Edge ? -1 : fv > 1 - Edge ? 1 : 0;
                for (int k = 1; k < 4; ++k)
                {
                    const int ou = (k & 1) ? du : 0, ov = (k & 2) ? dv : 0;
                    if ((k & 1 && !du) || (k & 2 && !dv)) continue;
                    auto Cn = C; Cn[Au] += ou; Cn[Av] += ov;
                    patch& Pn = PatchIn(Cn, P, N);
                    const int su = ou < 0 ? sub_v - 1 : ou > 0 ? 0 : std::clamp(int(fu * sub_v), 0, sub_v - 1);
                    const int sv = ov < 0 ? sub_v - 1 : ov > 0 ? 0 : std::clamp(int(fv * sub_v), 0, sub_v - 1);
                    const int bn = sv * sub_v + su;
                    if (!(Pn.m_VisMask & (1u << bn))) { Pn.m_VisMask |= std::uint16_t(1u << bn); Pn.m_Vis[bn] = P; }
                }
            };
            // The samples in order, whether each is covered (CoveredAt's rays) asked in parallel first, then touched in order: the same grid as
            // asking each when touched (a pure test), the rays no longer one at a time (a Bistro section's grid: minutes)
            std::vector<std::pair<v3, v3>> Samples;
            for (const auto& T : m_Tris)
            {
                const v3 A = m_Pos[T[0]], B = m_Pos[T[1]], Cc = m_Pos[T[2]];
                const float Longest = std::max({ (B - A).Length(), (Cc - B).Length(), (A - Cc).Length() });
                const int   k       = std::min(4096, std::max(1, int(std::ceil(Longest / (Cell / (2 * sub_v))))));     // every sub-cell it crosses
                for (int i = 0; i <= k; ++i) for (int j = 0; i + j <= k; ++j)
                {
                    const float u = float(i) / k, w = float(j) / k;
                    v3 N = m_Nrm[T[0]] * (1 - u - w) + m_Nrm[T[1]] * u + m_Nrm[T[2]] * w;
                    { const v3 F = (B - A).Cross(Cc - A); if (F.Length() > 1e-12f) N = (F.Dot(N) < 0 ? F * -1.0f : F).NormalizeCopy(); else N = N.NormalizeCopy(); }     // the face's (as the pixels')
                    Samples.push_back({ A + (B - A) * u + (Cc - A) * w, N });
                }
            }
            std::vector<char> Covered(Samples.size(), 0);
            constexpr int block_v = 4096;
            // (the patches alone need no rays: what is covered only marks their sub-cells)
            if (!bPatches) ParallelFor(int((Samples.size() + block_v - 1) / block_v), [&](int b)
            {
                for (std::size_t s = std::size_t(b) * block_v; s < std::min(Samples.size(), std::size_t(b + 1) * block_v); ++s) Covered[s] = CoveredAt(Samples[s].first, Samples[s].second);
            });
            for (std::size_t s = 0; s < Samples.size(); ++s) Touch(Samples[s].first, Samples[s].second, Covered[s] != 0);
            auto TLap = std::chrono::steady_clock::now();
            const auto Lap = [&](int k) { const auto T = std::chrono::steady_clock::now(); if (!bPatches) m_ReadGridMs[std::size_t(k)] += std::chrono::duration<double, std::milli>(T - TLap).count(); TLap = T; };
            if (!bPatches) m_ReadGridMs[1] = std::chrono::duration<double, std::milli>(TLap - TGrid0).count();

            if (bPatches)
            {
                std::size_t n = 0;
                for (const auto& [Key, Ce] : Cells) n += Ce.m_Patches.size();
                return Cells.empty() ? 0.0f : float(n) / float(Cells.size());
            }

            // Two points of one surface are joined when the line between them, lifted by a layer (SameLayer's: a tenth of a cell), is clear
            const float Lift = 0.1f * Cell;
            const auto Linked = [&](const v3& P, const v3& Q, const v3& N)
            {
                const v3 A = P + N * Lift, D = (Q + N * Lift) - A;
                const float Len = D.Length();
                return Len < 1e-6f || !Occluded(A, D * (1.0f / Len), Len);
            };
            std::vector<cell*> Order;
            for (auto& [Key, Ce] : Cells) Order.push_back(&Ce);
            ParallelFor(int(Order.size()), [&](int c)
            {
                auto& Ce = *Order[c];
                for (const auto& Pa : Ce.m_Patches)
                {
                    const auto PartOf = [&](std::uint16_t Mask, const std::array<v3, sub_v * sub_v>& Pts)
                    {
                        part R{ Pa.m_N, Pa.m_D, Mask, {} };
                        for (int b = 0; b < sub_v * sub_v; ++b) if (Mask & (1u << b)) R.m_Pts.push_back(Pts[b]);
                        return R;
                    };
                    if (Pa.m_VisMask == 0) { Ce.m_Parts.push_back(PartOf(Pa.m_HidMask, Pa.m_Hid)); Ce.m_Parts.back().m_bSeen = false; continue; }      // covered all over
                    if (!m_bLeakProtection) { auto R = PartOf(Pa.m_VisMask, Pa.m_Vis); R.m_Mask = 0xFFFF; Ce.m_Parts.push_back(std::move(R)); continue; }
                    std::uint16_t Left = Pa.m_VisMask;
                    while (Left)
                    {
                        std::uint16_t Mask = 0;
                        int Stack[sub_v * sub_v], n = 0;
                        const int b0 = std::countr_zero(unsigned(Left));
                        Stack[n++] = b0; Left &= std::uint16_t(~(1u << b0)); Mask |= std::uint16_t(1u << b0);
                        while (n)
                        {
                            const int b = Stack[--n], su = b % sub_v, sv = b / sub_v;
                            const int Next[4][2] = { { su - 1, sv }, { su + 1, sv }, { su, sv - 1 }, { su, sv + 1 } };
                            for (const auto& Nx : Next)
                            {
                                if (Nx[0] < 0 || Nx[0] >= sub_v || Nx[1] < 0 || Nx[1] >= sub_v) continue;
                                const int nb = Nx[1] * sub_v + Nx[0];
                                if (!(Left & (1u << nb)) || !Linked(Pa.m_Vis[b], Pa.m_Vis[nb], Pa.m_N)) continue;
                                Left &= std::uint16_t(~(1u << nb)); Mask |= std::uint16_t(1u << nb); Stack[n++] = nb;
                            }
                        }
                        Ce.m_Parts.push_back(PartOf(Mask, Pa.m_Vis));
                    }
                }
            });

            // Per part the surfels whose reach touches its cell's box, facing its way, weighted by how much of it sees them: its points and
            // the points of the same surface around it (the 26 neighbor cells' parts in its plane that it is joined to: their nearest points
            // Linked). Without it a pixel of blinds read the surfels of the window frame behind them (dark blotches). The neighbors' points
            // make the test generous and shared: a surfel leaves a list where the surface around no longer sees it, not at a cell's edge
            Lap(2);
            // The surfels by cells of Reach + Cell: one ring around a read cell's center holds all within Reach of its box (per axis it needs
            // Cell / 2 + Reach). A grid of the read cell's size needed Reach / Cell rings: a refined cell (R / 16) walked 65^3 grid cells a
            // read cell - minutes for a dense Bistro section
            surfel_grid G{ Reach + Cell, {} };
            for (int s = 0; s < int(L.size()); ++s) G.Add(L[s].m_Center, s);
            const int Ring = 1;
            std::vector<std::vector<std::vector<std::uint32_t>>> Lists(Order.size());
            std::atomic<std::uint32_t> Unsupported{ 0 };
            // Buried surfels (something within a tenth of a read cell in front of them: ivy pressed on a wall) are listed after the others
            // (only where the list has room): they see back faces (their validity 0, their light none), and a list's nearest slots taken
            // by them left the surface around them reading next to nothing - the Bistro's dark and bright cell blocks behind its ivy,
            // 2026-10-24. Not left out: one under a box's edge still serves the floor beside it
            std::vector<char> Buried(L.size(), 0);
            ParallelFor(int((L.size() + 1023) / 1024), [&](int b) { for (std::size_t k = std::size_t(b) * 1024; k < std::min(L.size(), std::size_t(b + 1) * 1024); ++k) Buried[k] = Occluded(L[k].m_Center + L[k].m_Normal * (0.02f * m_ReadCell), L[k].m_Normal, 0.1f * m_ReadCell); });
            ParallelFor(int(Order.size()), [&](int c)
            {
                const auto& Ce = *Order[c];
                Lists[c].resize(Ce.m_Parts.size());
                const v3 Lo{ Ce.m_C[0] * Cell, Ce.m_C[1] * Cell, Ce.m_C[2] * Cell }, Hi = Lo + v3{ Cell, Cell, Cell };
                std::vector<int> Near;
                G.Near((Lo + Hi) * 0.5f, Ring, [&](int s) { const v3 Q = v3::Max(Lo, v3::Min(Hi, L[s].m_Center)); if ((Q - L[s].m_Center).Length() < Reach) Near.push_back(s); });
                // Nearest first from the cell's center (SortReadLists' order and center): each list stops at m_ReadListCap seen surfels, the
                // farther ones never tested
                {
                    const v3 Cc{ (float(Ce.m_C[0]) + 0.5f) * Cell, (float(Ce.m_C[1]) + 0.5f) * Cell, (float(Ce.m_C[2]) + 0.5f) * Cell };
                    const auto D2 = [&](int s) { const v3 d = L[s].m_Center - Cc; return d.Dot(d); };
                    std::stable_sort(Near.begin(), Near.end(), [&](int a, int b) { return D2(a) < D2(b); });
                }
                for (std::size_t p = 0; p < Ce.m_Parts.size(); ++p)
                {
                    const auto& Pa = Ce.m_Parts[p];
                    std::vector<const v3*> Pts;
                    for (const auto& Q : Pa.m_Pts) Pts.push_back(&Q);
                    for (int dz = -1; dz <= 1; ++dz) for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
                    {
                        const auto It = Cells.find(surfel_grid::Key(Ce.m_C[0] + dx, Ce.m_C[1] + dy, Ce.m_C[2] + dz));
                        if (It == Cells.end()) continue;
                        for (const auto& Pb : It->second.m_Parts)
                        {
                            if (&Pb == &Pa || Pb.m_N.Dot(Pa.m_N) <= 0.97f || std::abs(Pb.m_D - Pa.m_D) >= PlaneTol || Pb.m_Pts.empty()) continue;
                            const v3* pA = nullptr; const v3* pB = nullptr; float Best = 1e30f;
                            for (const auto& A : Pa.m_Pts) for (const auto& B : Pb.m_Pts) { const float d = (A - B).Length(); if (d < Best) { Best = d; pA = &A; pB = &B; } }
                            if (!m_bLeakProtection || Linked(*pA, *pB, Pa.m_N)) for (const auto& Q : Pb.m_Pts) Pts.push_back(&Q);
                        }
                    }
                    // The cell's 4 corners in the patch's plane (ReadAxes: the 2 axes across it), and per corner the points nearest it (up to 4,
                    // within 3/4 of a cell): the same points whichever cell asks, so neighboring cells agree on their shared corners
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
                    int nTested = 0;
                    for (int Pass = 0; Pass < 2 && Lists[c][p].size() < m_ReadListCap; ++Pass)     // (the buried last: Buried)
                    for (const int s : Near)
                    {
                        if (L[s].m_Normal.Dot(Pa.m_N) < 0.5f || (Buried[std::size_t(s)] != 0) != (Pass == 1)) continue;
                        if (++nTested > read_tests_v) break;           // (its nearest read_tests_v facing ones asked: past them a part is enclosed)
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
                        // Out of sight everywhere: not listed. None in sight (a face tucked in, or all in reach behind something): an empty list -
                        // the read there is unsupported (GIQueryResult: the vertex read, else the mean light), not filled from surfels it does not see
                        if ((E >> 16) == 0) continue;
                        Lists[c][p].push_back(E);
                        if (Lists[c][p].size() >= m_ReadListCap) break;
                    }
                    if (Lists[c][p].empty()) Unsupported.fetch_add(1, std::memory_order_relaxed);
                }
            });

            // The table: per cell { key, key, its patches' first word in m_ReadList, how many }; a patch (a part) is 4 words { its normal
            // (octahedral, 2 x snorm 16), its plane (float bits), its list's first word, its count | its sub-cells' mask << 16 }, its list
            // after the cell's patches
            std::uint32_t Size = 1;
            while (Size < 2 * Cells.size()) Size <<= 1;
            m_ReadTable.assign(Size, { 0, 0xFFFFFFFFu, 0, 0 });
            for (std::size_t c = 0; c < Order.size(); ++c)
            {
                const auto& Ce = *Order[c];
                const auto First = std::uint32_t(m_ReadList.size());
                const auto n = std::uint32_t(Ce.m_Parts.size());
                m_ReadList.resize(m_ReadList.size() + 4 * n);
                for (std::uint32_t p = 0; p < n; ++p)
                {
                    const auto& Pa = Ce.m_Parts[p];
                    const std::uint32_t L0 = std::uint32_t(m_ReadList.size());
                    m_ReadList.insert(m_ReadList.end(), Lists[c][p].begin(), Lists[c][p].end());
                    m_ReadList[First + 4 * p + 0] = PackOct(Pa.m_N);
                    m_ReadList[First + 4 * p + 1] = std::bit_cast<std::uint32_t>(Pa.m_D);
                    m_ReadList[First + 4 * p + 2] = L0;
                    m_ReadList[First + 4 * p + 3] = std::min<std::uint32_t>(std::uint32_t(Lists[c][p].size()), read_packed_v - 1) | (std::uint32_t(Pa.m_Mask) << 16);
                }
                const auto K = ReadKey(Ce.m_C);
                std::uint32_t h = ReadHash(Ce.m_C) & (Size - 1), nProbe = 0;
                while (m_ReadTable[h][1] != 0xFFFFFFFFu) { h = (h + 1) & (Size - 1); ++nProbe; }
                m_ReadMaxProbe = std::max(m_ReadMaxProbe, nProbe + 1);
                m_ReadTable[h] = { K[0], K[1], First, n };
            }
            m_ReadUnsupported = Unsupported.load();
            m_EmptyParts.clear();
            for (std::size_t c = 0; c < Order.size(); ++c)
                for (std::size_t p = 0; p < Order[c]->m_Parts.size(); ++p)
                {
                    const auto& Pa = Order[c]->m_Parts[p];
                    if (Lists[c][p].empty() && Pa.m_bSeen && !Pa.m_Pts.empty()) m_EmptyParts.push_back({ Pa.m_Pts[Pa.m_Pts.size() / 2], Pa.m_N });
                }
            SortReadLists();
            Lap(3);
            return ReadPatchesPerCell();
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

        // Bake's candidates: about 4 per Radius^2 of each triangle and at least one per half Radius of its longest edge, each with its
        // triangle's normal (to the smoothed normal's side); per candidate its triangle, u, v when asked (pCT)
        void Candidates(float Radius, std::vector<v3>& CP, std::vector<v3>& CN, std::vector<std::array<float, 3>>* pCT) const noexcept
        {
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
                    if (pCT) pCT->push_back({ float(&T - m_Tris.data()), u, v });
                }
            }
        }

        // How many surfels Bake(Radius) keeps (its step 2 alone: its candidates in its order, its acceptance, the rays it needs built once),
        // stopping past Stop (Stop + 1): over a limit it ends as soon as the limit is passed. A surface whose area alone asks for more (its
        // R-disks: the visibility test only ever keeps more) returns that before any candidate is made (a mesh in cm: gigabytes of them).
        // bRays false: without the visibility test (no BVH, no rays: much cheaper), a lower bound (that test only ever keeps more)
        std::size_t CountSurfels(float Radius, std::size_t Stop = ~std::size_t(0), bool bRays = true) noexcept
        {
            double Area = 0;
            for (const auto& T : m_Tris) Area += 0.5 * double((m_Pos[T[1]] - m_Pos[T[0]]).Cross(m_Pos[T[2]] - m_Pos[T[0]]).Length());
            if (const double Disks = Area / (3.14159265 * double(Radius) * double(Radius)); Disks > double(Stop)) return std::size_t(Disks);

            if (bRays && m_Tri4.empty()) BuildRays();
            std::vector<v3> CP, CN;
            Candidates(Radius, CP, CN, nullptr);
            std::vector<int> Order(CP.size());
            for (std::size_t i = 0; i < Order.size(); ++i) Order[i] = int(i);
            rng Shuffle(0x5EEDu);
            for (std::size_t i = Order.size(); i > 1; --i) std::swap(Order[i - 1], Order[std::size_t(Shuffle.Next() * float(i)) % i]);
            surfel_grid G{ Radius, {} };
            std::vector<surfel> Kept;
            for (int i : Order)
            {
                bool bTaken = false;
                G.Near(CP[i], 1, [&](int s) { bTaken = bTaken || ((CP[i] - Kept[s].m_Center).Length() < Radius && Kept[s].m_Normal.Dot(CN[i]) > 0.8f
                                                                && std::abs((Kept[s].m_Center - CP[i]).Dot(CN[i])) < 0.1f * Radius && (!bRays || Sees(CP[i], CN[i], Kept[s]))); });
                if (bTaken) continue;
                G.Add(CP[i], int(Kept.size()));
                Kept.push_back({ CP[i], CN[i], Radius });
                if (Kept.size() > Stop) break;
            }
            return Kept.size();
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
        // Past MaxSurfels surfels it stops (m_Levels[0] has MaxSurfels + 1 of them, nothing else is made): the caller drops the GI.
        void Bake(float Radius, float ParentRadius, std::span<const std::pair<v3, v3>> Forced = {}, std::size_t MaxSurfels = ~std::size_t(0)) noexcept
        {
            BuildRays();                                            // the vertices' visibility tests cast against the mesh

            // Dart throwing over weighted candidates: the kept ones become surfels of radius R
            // A candidate is taken by a kept surfel only if that one sees it: a door a few cm behind its frame gets surfels of its own
            // (the wall's past the frame's lip do not cover it - read through the lip, they faded in wedges)
            const auto Poisson = [this](const std::vector<v3>& P, const std::vector<v3>& N, float R, std::vector<surfel>& Out, surfel_grid& G, std::vector<int>* pFrom = nullptr, std::size_t Max = ~std::size_t(0))
            {
                Out.clear();
                if (pFrom) pFrom->clear();
                G = { R, {} };
                std::vector<int> Order(P.size());
                for (std::size_t i = 0; i < Order.size(); ++i) Order[i] = int(i);
                rng Shuffle(0x5EEDu);
                for (std::size_t i = Order.size(); i > 1; --i) std::swap(Order[i - 1], Order[std::size_t(Shuffle.Next() * float(i)) % i]);
                // ... and only if it lies in the candidate's own plane (a tenth of R off it at most): a window frame 10 cm behind blinds,
                // tilted so its plane runs through them and seeing them through their gaps, does not take the blinds' surfels
                const auto Takes = [&](int i, int s) { return (P[i] - Out[s].m_Center).Length() < R && Out[s].m_Normal.Dot(N[i]) > 0.8f
                                                              && std::abs((Out[s].m_Center - P[i]).Dot(N[i])) < 0.1f * R && Sees(P[i], N[i], Out[s]); };
                // In batches of the order (the same surfels as one at a time: taken = taken by a surfel kept before it, whichever pass looks):
                // the surfels kept before the batch looked at in parallel (their rays: the bake's cost, single-threaded it was 16 min on the
                // Bistro), then the batch in order against the ones it keeps itself
                constexpr std::size_t batch_v = 8192;
                std::vector<std::uint8_t> Prior;
                for (std::size_t b0 = 0; b0 < Order.size(); b0 += batch_v)
                {
                    const std::size_t b1 = std::min(Order.size(), b0 + batch_v);
                    BakeTick(0.9f * float(b0) / float(Order.size()));          // (the parents' stage: their nearest after, the last tenth)
                    const int nBefore = int(Out.size());
                    Prior.assign(b1 - b0, 0);
                    ParallelFor(int(b1 - b0), [&](int k)
                    {
                        const int i = Order[b0 + std::size_t(k)];
                        bool bTaken = false;
                        G.Near(P[i], 1, [&](int s) { bTaken = bTaken || Takes(i, s); });
                        Prior[std::size_t(k)] = bTaken;
                    });
                    for (std::size_t k = 0; k < b1 - b0; ++k)
                    {
                        if (Prior[k]) continue;
                        const int i = Order[b0 + k];
                        bool bTaken = false;
                        G.Near(P[i], 1, [&](int s) { bTaken = bTaken || (s >= nBefore && Takes(i, s)); });
                        if (bTaken) continue;
                        G.Add(P[i], int(Out.size()));
                        Out.push_back({ P[i], N[i], R });
                        if (pFrom) pFrom->push_back(i);
                        if (Out.size() > Max) return;
                    }
                }
            };

            // The nearest compatible surfels of a point (up to 4) that the point sees (nothing between them: a surfel behind a wall, or
            // in a gap the point only reaches through solid geometry, is not its surfel), widening the search until something is found
            const auto Nearest = [this](const std::vector<surfel>& L, const surfel_grid& G, const v3& P, const v3& N, int nMax)
            {
                std::array<std::pair<float, int>, 4> Best;
                Best.fill({ 1e30f, -1 });
                // Nearest first ((distance, surfel): the order Best keeps), a ray each until nMax are seen: the same ones as testing all of
                // them; a wider ring tests only what the narrower did not. At most near_tests_v rays a point: one that sees none of its
                // nearest is enclosed (the Bistro's hidden vertices tested every compatible surfel 4 m around: 11 of its 17 minutes of bake)
                std::vector<std::pair<float, int>> By;
                std::vector<int> Tested;
                int nRays = 0;
                for (int Ring = 1; Ring <= 4 && Best[0].second < 0 && nRays < near_tests_v; ++Ring)
                {
                    By.clear();
                    G.Near(P, Ring, [&](int s) { if (L[s].m_Normal.Dot(N) >= 0.5f) By.push_back({ (P - L[s].m_Center).Length(), s }); });
                    std::sort(By.begin(), By.end());
                    std::sort(Tested.begin(), Tested.end());
                    int n = 0;
                    for (const auto& [d, s] : By)
                    {
                        if (std::binary_search(Tested.begin(), Tested.end(), s)) continue;
                        if (nRays++ >= near_tests_v) break;
                        Tested.push_back(s);
                        if (!Sees(P, N, L[s])) continue;
                        Best[std::size_t(n)] = { d, s };
                        if (++n == nMax) break;
                    }
                }
                return Best;
            };

            m_BakeMs = {};
            auto TLap = std::chrono::steady_clock::now();
            static constexpr float Done[6] = { 0.01f, 0.11f, 0.12f, 0.40f, 0.67f, 1.0f };     // (after each stage)
            m_BakeLo = 0; m_BakeHi = Done[0]; m_BakeLast = -1;
            const auto Lap = [&](int k)
            {
                const auto T = std::chrono::steady_clock::now(); m_BakeMs[std::size_t(k)] += std::chrono::duration<double, std::milli>(T - TLap).count(); TLap = T;
                if (m_OnBakeProgress) { m_OnBakeProgress(Done[std::size_t(k)]); m_BakeLast = Done[std::size_t(k)]; }
                m_BakeLo = Done[std::size_t(k)]; m_BakeHi = k < 5 ? Done[std::size_t(k) + 1] : 1.0f;      // (the next stage's range)
            };

            // 1. Candidates on the surface, by area
            std::vector<v3> CP, CN;
            std::vector<std::array<float, 3>> CT;                  // per candidate: its triangle, u, v (where its material is)
            Candidates(Radius, CP, CN, &CT);
            Lap(0);

            // 2. Surfels, 3. vertex indices
            surfel_grid G0, G1;
            std::vector<int> From;
            Poisson(CP, CN, Radius, m_Levels[0], G0, &From, MaxSurfels);
            Lap(1);
            if (m_Levels[0].size() > MaxSurfels) return;
            // The patches (PatchDefects: Forced, from BuildReadGrid's m_EmptyParts): a surfel at each, after the others (one where one already
            // stands for it: the same rule as the darts'), its material where it lies (the triangle under it)
            m_FirstPatch = 0xFFFFFFFFu;
            const std::size_t FirstPatch = m_Levels[0].size();
            for (const auto& [P, N] : Forced)
            {
                bool bTaken = false;
                for (std::size_t s = FirstPatch; s < m_Levels[0].size() && !bTaken; ++s)
                {
                    const auto& S = m_Levels[0][s];
                    bTaken = (S.m_Center - P).Length() < Radius && S.m_Normal.Dot(N) > 0.8f && std::abs((S.m_Center - P).Dot(N)) < 0.1f * Radius && Sees(P, N, S);
                }
                if (bTaken) continue;
                float T = 0.04f * Radius, U = 0, V = 0; int Tri = 0;
                if (!Cast(P + N * (0.02f * Radius), N * -1.0f, T, Tri, U, V)) { Tri = 0; U = V = 0; }
                CP.push_back(P); CN.push_back(N); CT.push_back({ float(Tri), U, V });
                From.push_back(int(CP.size() - 1));
                G0.Add(P, int(m_Levels[0].size()));
                m_Levels[0].push_back({ P, N, Radius });
            }
            if (m_Levels[0].size() > FirstPatch) m_FirstPatch = std::uint32_t(FirstPatch);

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
            Lap(2);
            m_VertSurfels.resize(m_Pos.size());
            SlicedFor(int(m_Pos.size()), [&](int v)
            {
                const auto B = Nearest(m_Levels[0], G0, m_Pos[v], m_Nrm[v], 4);
                m_VertSurfels[v] = { B[0].second, B[1].second, B[2].second, B[3].second };
            });
            Lap(3);

            // 4. Parents
            std::vector<v3> SP(m_Levels[0].size()), SN(m_Levels[0].size());
            for (std::size_t s = 0; s < SP.size(); ++s) { SP[s] = m_Levels[0][s].m_Center; SN[s] = m_Levels[0][s].m_Normal; }
            Poisson(SP, SN, ParentRadius, m_Levels[1], G1);
            m_ParentOf.assign(SP.size(), 0);
            SlicedFor(int(SP.size()), [&](int s)
            {
                const auto B = Nearest(m_Levels[1], G1, SP[s], SN[s], 1);
                m_ParentOf[s] = std::max(B[0].second, 0);
            }, 0.9f, 1.0f);
            Lap(4);

            BuildNeighbors();
            Lap(5);
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
        // Origin 0 is the surfel's own center and normal (implicit: 68% of Police's surfels have no other); the others its entries, each a
        // vertex (its position, its normal normalized). Compressed (GI working set): 4 B an origin, was a position and a normal (24 B; Police
        // 6.6 MB -> 1.0 MB), the same origins
        std::vector<std::uint32_t>          m_OriginFirst;          // per fine surfel: its first entry (size + 1)
        std::vector<std::uint32_t>          m_OriginVert;
        // A fine surfel's origin k (0: its center; k: its entry o0 + k - 1, o0 = m_OriginFirst[s]), mesh space
        void Origin(std::uint32_t o0, std::uint32_t k, int s, v3& P, v3& N) const noexcept
        {
            if (k == 0) { P = m_Levels[0][std::size_t(s)].m_Center; N = m_Levels[0][std::size_t(s)].m_Normal; return; }
            const std::uint32_t v = m_OriginVert[o0 + k - 1];
            P = m_Pos[v]; N = m_Nrm[v].NormalizeCopy();
        }

        void BuildOrigins() noexcept
        {
            const auto& L = m_Levels[0];
            m_OriginFirst.assign(L.size() + 1, 0); m_OriginVert.clear();
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
                const std::size_t n = Per[s].size(), k = std::min<std::size_t>(n, 7);
                for (std::size_t j = 0; j < k; ++j) m_OriginVert.push_back(std::uint32_t(Per[s][j * n / k]));
                m_OriginFirst[s + 1] = std::uint32_t(m_OriginVert.size());
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
                SlicedFor(int(L.size()), [&](int s)
                {
                    G.Near(L[s].m_Center, 1, [&](int n)
                    {
                        if (n == s) return;
                        const float c = L[s].m_Normal.Dot(L[n].m_Normal);
                        if (c < 0.8f) return;
                        const float f = 1.0f - (L[n].m_Center - L[s].m_Center).Length() / (2 * L[s].m_Radius);
                        if (f <= 0 || !SameLayer(L[s], L[n])) return;
                        Per[s].push_back({ std::uint32_t(n), f * c });      // linear: a neighbor counts (squared, the read's falloff, it barely did)
                    });
                }, Lv == 0 ? 0.1f : 0.9f, Lv == 0 ? 0.9f : 1.0f);
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
        static constexpr std::uint32_t derived_tag_v = 0x36444947u;    // "GID6": the lists packed (PackReadGrid). GID5: as they are, read: unpacked
        static constexpr std::uint32_t derived_tag5_v = 0x35444947u;   // (GID4, never released: the center as an entry; GID3: the origins as
                                                                        // positions and normals, read: made again;
                                                                        // GID2: no first patch; GID1: no sub-cell masks, no flags): a change to the layout below is a new tag
        static constexpr std::uint32_t derived_tag3_v = 0x33444947u;
        std::vector<std::uint32_t> PackDerived() const noexcept
        {
            std::vector<std::array<std::uint32_t, 4>> PTable; std::vector<std::uint32_t> PList;
            PackReadGrid(true, m_ReadTable, m_ReadList, PTable, PList);        // (made here: the compiler's chunks change the lists after their sort)
            std::vector<std::uint32_t> W{ derived_tag_v, std::uint32_t(m_bLeakProtection), m_FirstPatch, std::bit_cast<std::uint32_t>(m_ReadCell), m_ReadMaxProbe, m_ReadUnsupported
                                        , std::uint32_t(PTable.size()), std::uint32_t(PList.size()) };
            for (const auto& E : PTable) W.insert(W.end(), E.begin(), E.end());
            W.insert(W.end(), PList.begin(), PList.end());
            for (const auto& Nb : m_Neighbors)
            {
                W.push_back(std::uint32_t(Nb.m_First.size())); W.push_back(std::uint32_t(Nb.m_Index.size()));
                W.insert(W.end(), Nb.m_First.begin(), Nb.m_First.end());
                W.insert(W.end(), Nb.m_Index.begin(), Nb.m_Index.end());
                for (float w : Nb.m_Weight) W.push_back(std::bit_cast<std::uint32_t>(w));
            }
            W.push_back(std::uint32_t(m_OriginFirst.size())); W.push_back(std::uint32_t(m_OriginVert.size()));
            W.insert(W.end(), m_OriginFirst.begin(), m_OriginFirst.end());
            W.insert(W.end(), m_OriginVert.begin(), m_OriginVert.end());
            return W;
        }
        bool UnpackDerived(std::span<const std::uint32_t> W) noexcept
        {
            std::size_t i = 0;
            const auto Take = [&](std::size_t n) { const bool b = i + n <= W.size(); const std::size_t at = i; i += n; return b ? W.data() + at : nullptr; };
            const std::uint32_t* p = Take(8);
            if (p == nullptr || (p[0] != derived_tag_v && p[0] != derived_tag5_v && p[0] != derived_tag3_v)) return false;
            const bool bPacked = p[0] == derived_tag_v;
            const bool bTag3 = p[0] == derived_tag3_v;
            const bool bLeak = (p[1] & 1u) != 0;
            const std::uint32_t FirstPatch = p[2];
            const float Cell = std::bit_cast<float>(p[3]);
            const std::uint32_t MaxProbe = p[4], Unsupported = p[5], nTable = p[6], nList = p[7];
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
            const std::uint32_t *pOF = Take(nOFirst), *pOV = Take(bTag3 ? 6 * std::size_t(nO) : std::size_t(nO));
            if (!pOF || !pOV || i != W.size()) return false;
            if (!bTag3) for (std::uint32_t k = 0; k < nO; ++k) if (pOV[k] >= m_Pos.size()) return false;     // (made for other vertices)
            if (Nbs[0].m_First.size() != 0 && Nbs[0].m_First.size() != m_Levels[0].size() + 1) return false;       // made for other surfels
            if (nOFirst != 0 && nOFirst != m_Levels[0].size() + 1) return false;
            m_ReadCell = Cell; m_ReadMaxProbe = MaxProbe; m_ReadUnsupported = Unsupported; m_bLeakProtection = bLeak; m_FirstPatch = FirstPatch;
            m_ReadTable.resize(nTable);
            for (std::uint32_t k = 0; k < nTable; ++k) m_ReadTable[k] = { pTable[4 * k], pTable[4 * k + 1], pTable[4 * k + 2], pTable[4 * k + 3] };
            m_ReadList.assign(pList, pList + nList);
            if (bPacked) { auto T = m_ReadTable; auto L = m_ReadList; PackReadGrid(false, T, L, m_ReadTable, m_ReadList); }
            m_Neighbors = std::move(Nbs);
            if (bTag3) BuildOrigins();                                  // (a GID3 resource: its origins made again from the vertex surfels, the same ones)
            else { m_OriginFirst.assign(pOF, pOF + nOFirst); m_OriginVert.assign(pOV, pOV + nO); }
            SortReadLists();
            if (const char* pS = std::getenv("XLION_GI_SPARSEGRID")) SparsifyReadGrid(float(std::atof(pS)));     // (experiment)
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

        // The display filter's neighbors (BuildNeighbors): one surface layer, not merely in sight of each other. A layer is layer_v of a
        // radius thick: each center within it of the other's plane (blinds' slats, a centimeter of relief, are one layer; the window
        // frame 10 cm behind them is another), and nothing between them higher than it (the line between them lifted by it is clear:
        // the slats' edges do not stop it, a baseboard or a wall does). It was: in sight from half a radius off the surface - over
        // everything lower than that, into the layer behind
        static constexpr float layer_v = 0.1f;
        bool SameLayer(const surfel& A, const surfel& B) const noexcept
        {
            const float h = layer_v * A.m_Radius;
            const v3 D = B.m_Center - A.m_Center;
            if (std::abs(D.Dot(A.m_Normal)) > h || std::abs(D.Dot(B.m_Normal)) > h) return false;
            return Sees(A.m_Center, A.m_Normal, B) || SeesLifted(A, B, h);
        }

        // Do two surfels see each other from h off their surfaces (the risers up to there clear too)?
        bool SeesLifted(const surfel& A, const surfel& B, float h) const noexcept
        {
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

        // GI chunks (xlionrender_surfel_gi_chunks.h): a chunk's bake-time visibility is the whole geometry's (its builders' rays see every
        // triangle, not only the chunk's); cleared after the build, so the runtime world's rays stay per instance
        const surfel_mesh*                  m_pOccluder = nullptr;

        // Is anything between O and O + D TMax (mesh space)? Stops at the first hit
        bool Occluded(const v3& O, const v3& D, float TMax) const noexcept
        {
            if (m_pOccluder) return m_pOccluder->Occluded(O, D, TMax);
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
    // Phase 7: a geometry's arrays copied (what FromXGeomStatic and SurfelsFromXGeomStatic read), so its surfels can be made on a worker
    // while the geometry itself may be released: the same accessors as the geometry
    template <class T_GEOM> struct geom_gi_copy
    {
        template <class T_SPAN> using elem_t = std::remove_cvref_t<decltype(std::declval<T_SPAN>()[0])>;
        using surfel_t   = std::remove_cvref_t<decltype(std::declval<const T_GEOM&>().getGISurfels()[0])>;
        using material_t = std::remove_cvref_t<decltype(std::declval<const T_GEOM&>().getGISurfelMaterials()[0])>;
        using vertex_t   = elem_t<decltype(std::declval<const T_GEOM&>().getVertices())>;
        using extra_t    = elem_t<decltype(std::declval<const T_GEOM&>().getVertexExtras())>;
        using index_t    = elem_t<decltype(std::declval<const T_GEOM&>().getIndices())>;
        using cluster_t  = elem_t<decltype(std::declval<const T_GEOM&>().getClusters())>;
        using cdata_t    = elem_t<decltype(std::declval<const T_GEOM&>().getClusterData())>;
        using submesh_t  = elem_t<decltype(std::declval<const T_GEOM&>().getSubmeshes())>;
        template <class T> static std::vector<std::remove_cvref_t<decltype(*std::declval<T>().begin())>> Copy(const T& S) { return { S.begin(), S.end() }; }
        explicit geom_gi_copy(const T_GEOM& G) noexcept
            : m_Vertices(Copy(G.getVertices())), m_Extras(Copy(G.getVertexExtras())), m_Indices(Copy(G.getIndices()))
            , m_Clusters(Copy(G.getClusters())), m_ClusterData(Copy(G.getClusterData())), m_Submeshes(Copy(G.getSubmeshes()))
            , m_Fine(G.getGISurfels().begin(), G.getGISurfels().end()), m_Parents(G.getGIParents().begin(), G.getGIParents().end())
            , m_VS(G.getGIVertexSurfels().begin(), G.getGIVertexSurfels().end())
            , m_Materials(G.getGISurfelMaterials().begin(), G.getGISurfelMaterials().end())
            , m_Derived(G.getGIDerived().begin(), G.getGIDerived().end()), m_GIMode(G.m_GIMode)
        {
            if constexpr (requires { G.getGISections(); G.getGIClusterSections(); G.getGIGhosts(); })      // (xgeom_static version 7: the GI in sections)
            {
                for (const auto& S : G.getGISections()) m_Sections.push_back({ S.m_iFine, S.m_nFine, S.m_nOwned, S.m_iParent, S.m_nParents, S.m_iDerived, S.m_nDerived, S.m_iGhost, S.m_nGhosts });
                m_ClusterSections.assign(G.getGIClusterSections().begin(), G.getGIClusterSections().end());
                for (const auto& Gh : G.getGIGhosts()) m_Ghosts.push_back({ Gh.m_Section, Gh.m_Local });
            }
        }
        struct section_t { std::uint32_t m_iFine, m_nFine, m_nOwned, m_iParent, m_nParents, m_iDerived, m_nDerived, m_iGhost, m_nGhosts; };
        struct ghost_t   { std::uint32_t m_Section, m_Local; };
        std::vector<section_t>      m_Sections;
        std::vector<std::uint16_t>  m_ClusterSections;
        std::vector<ghost_t>        m_Ghosts;
        std::span<const section_t>     getGISections       () const noexcept { return m_Sections; }
        std::span<const std::uint16_t> getGIClusterSections() const noexcept { return m_ClusterSections; }
        std::span<const ghost_t>       getGIGhosts         () const noexcept { return m_Ghosts; }
        std::vector<vertex_t>       m_Vertices;
        std::vector<extra_t>        m_Extras;
        std::vector<index_t>        m_Indices;
        std::vector<cluster_t>      m_Clusters;
        std::vector<cdata_t>        m_ClusterData;
        std::vector<submesh_t>      m_Submeshes;
        std::vector<surfel_t>       m_Fine, m_Parents;
        std::vector<std::uint16_t>  m_VS;
        std::vector<material_t>     m_Materials;
        std::vector<std::uint32_t>  m_Derived;
        decltype(T_GEOM::m_GIMode)  m_GIMode;
        std::span<const vertex_t>      getVertices         () const noexcept { return m_Vertices; }
        std::span<const extra_t>       getVertexExtras     () const noexcept { return m_Extras; }
        std::span<const index_t>       getIndices          () const noexcept { return m_Indices; }
        std::span<const cluster_t>     getClusters         () const noexcept { return m_Clusters; }
        std::span<const cdata_t>       getClusterData      () const noexcept { return m_ClusterData; }
        std::span<const submesh_t>     getSubmeshes        () const noexcept { return m_Submeshes; }
        std::span<const surfel_t>      getGISurfels        () const noexcept { return m_Fine; }
        std::span<const surfel_t>      getGIParents        () const noexcept { return m_Parents; }
        std::span<const std::uint16_t> getGIVertexSurfels  () const noexcept { return m_VS; }
        std::span<const material_t>    getGISurfelMaterials() const noexcept { return m_Materials; }
        std::span<const std::uint32_t> getGIDerived        () const noexcept { return m_Derived; }
    };

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
        if (std::getenv("XLION_GI_REGRID")) M.BuildReadGrid();      // measurement: the grid rebuilt at load (BuildReadGrid's cell), no recompile
        return true;
    }

    struct lights
    {
        struct point
        {
            v3      m_Position;
            v3      m_Color;            // color x intensity
            float   m_Radius = 1;
            v3      m_Dir{ 0, 0, 0 };   // a spot: where it points (unit)
            float   m_CosOuter = -2;    // a spot: no light outside this cone (< -1: a point light, all around)
            float   m_CosInner = -2;    //   full light inside this one (the renderer's falloff: smoothstep outer -> inner)
        };
        v3                  m_SunDirection{ 0, -1, 0 };     // the direction the light travels
        v3                  m_SunColor{ 0, 0, 0 };
        v3                  m_Sky{ 0, 0, 0 };               // the radiance of rays that leave the scene (its mean)
        std::array<v3, 16>  m_SkyOct{};                     // or by direction (m_bSkyOct): a 4 x 4 octahedral map of the sky's coarsest prefiltered level (xlionrender_octahedral.glsl)
        bool                m_bSkyOct = false;
        std::vector<point>  m_Points;

        // The radiance of a ray that leaves the scene along D: the sky above is not the ground below (a uniform mean lit the shade of an
        // arcade with the average of a bright sky and a dark ground)
        v3 SkyAt(const v3& D) const noexcept
        {
            if (!m_bSkyOct) return m_Sky;
            const float x0 = D.m_X, y0 = D.m_Z, z0 = -D.m_Y;                    // OctEncode: -Y is the center
            const float k  = 1.0f / std::max(std::abs(x0) + std::abs(y0) + std::abs(z0), 1e-12f);
            const float x = x0 * k, y = y0 * k, z = z0 * k;
            const float u = z >= 0 ? x : (1 - std::abs(y)) * (x >= 0 ? 1.0f : -1.0f);
            const float v = z >= 0 ? y : (1 - std::abs(x)) * (y >= 0 ? 1.0f : -1.0f);
            const float fx = std::clamp((u * 0.5f + 0.5f) * 4 - 0.5f, 0.0f, 3.0f), fy = std::clamp((v * 0.5f + 0.5f) * 4 - 0.5f, 0.0f, 3.0f);
            const int   ix = std::min(int(fx), 2), iy = std::min(int(fy), 2);
            const float tx = fx - float(ix), ty = fy - float(iy);
            const auto  T  = [&](int i, int j) { return m_SkyOct[std::size_t(j * 4 + i)]; };
            return (T(ix, iy) * (1 - tx) + T(ix + 1, iy) * tx) * (1 - ty) + (T(ix, iy + 1) * (1 - tx) + T(ix + 1, iy + 1) * tx) * ty;
        }
    };

    class world
    {
    public:

        struct setup
        {
            int     m_RaysPerSurfel = 8;
            float   m_MinBlend      = 0;        // the warm-up's blend (1 / n after a change) never below this (1: every pass replaces)
            float   m_TraceAllowanceMs = 0;     // Phase 1 (gi_dynamic_master_plan.md): the trace's main-thread time an update may use (0: no limit, every picked surfel is traced in full this update, as before)
            float   m_UnitTargetUs  = 250;      // the size of a unit of the trace (rule 4): a surfel's rays are traced in chunks of about this many microseconds
            float   m_FairShare     = 0.2f;
            bool    m_bAdaptiveRays = true;     // Phase 6: a settled surfel asks for rays by its uncertainty (RequestedRays); false: m_RaysPerSurfel each
            float   m_RayUncertainty = 3.0f;    // RequestedRays: the rays = m_RaysPerSurfel x relative deviation x this (a quarter to all of them)
            float   m_LinkShare     = 0.3f;     // Phase 2: the part of the allowance the emitter links may use while the trace has work too     // the share of an update's surfels kept for the instance waiting longest (fairness: whatever its score)
            bool    m_bDistanceTiers = true;    // a surfel far from the eye (in its radii) is traced every 2nd, 4th, 8th pass (Update)
            bool    m_bAblateHitRead = false;   // measurement only (GI review, experiment A): the GPU pass's hits read a constant instead of the cache (header word 11 bit 4)
            bool    m_bAblateFreeze  = false;   // measurement only: the GPU commit keeps its ranges' stats but leaves the light as it is (header word 11 bit 8)
            bool    m_bAblateStats   = false;   // measurement only, with the freeze: the pass's work counters summed per range (bit 16; gpu_trace::Collect prints them)
            bool    m_bAblateFallback = false;  // measurement only: an unsupported hit takes its instance's mean instead of walking its cell (bit 32)
            float   m_Saturation    = 1;       // the shown light's saturation (BasisLight, after the filter; luminance kept): the filter and the coarse surfels average colors toward grey
            float   m_EmissiveScale = 1;        // the surfels' emission (SetSurfelMaterial) times this: self-illumination as the lights it is
            int     m_FilterPasses  = 4;
            float   m_ShowShare     = 0.2f;     // Phase 6b: the part of the allowance kept for the shown light while it has a backlog
            float   m_OverheadMs    = 0;        // Phase 6b: the GI work of the frame outside Update (the renderer's upload, by its last frames): taken from the allowance        // the shown light's spatial filter (BasisLight): each pass about one surfel radius wider
            bool    m_bGPUTrace     = false;    // G3 (gi_dynamic_master_plan.md): Update picks and commits, the GPU traces (gpu_trace: GPUWork, DeliverGPU)
            bool    m_bGPUShown     = false;    // G4: the shown light is the GPU's (gpu_trace::Show): Update makes none (UpdateShown, BasisLight)
            bool    m_bEmitterSampling = false; // G5 (gi_dynamic_master_plan.md): the emitters' light by light sampling in each pass (SampleEmitters: 8
                                                // samples, one shadow ray each), not the links (not built); the rays' hits leave the emission out
            bool    m_bGPUOwned     = false;    // G4c: the light lives on the GPU only (with m_bGPUTrace, m_bGPUShown): the pick hands out ranges
                                                // (GPURanges), the GPU's expand makes the surfels' passes, the means come back (SetGPUMeans); the
                                                // surfels' arrays here are stale (gpu_trace::ReadBack fills them)
            bool    m_bSpreadOrigins = true;    // the fine surfels' rays start across their patch (surfel_mesh::BuildOrigins), not at their center
            float   m_LodDistance   = 40;       // the priority's distance scale: an instance this far (eye to bounds) scores half as much (Update)
            float   m_ProjScale     = 0;        // the view's pixels per unit at distance 1 (height / 2 tan(fov / 2)): with it the level follows the area an instance covers on the screen (0: only the distance)
            bool    m_bCube         = true;     // an instance that covers few pixels for its parents traces 6 of them, one per axis direction, and blends the others from them (an ambient cube)
            float   m_PixelsPerSurfel = 256;    // the level rule: a level's surfels must each cover this many pixels where the instance is nearest the eye (about 16 x 16), else coarser
            int     m_RelightPeriod = 120;      // Phase 6: RelightAll at most once in this many updates
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
            int     m_nLinks    = 0;            // emitter links (PumpLinks)
            float   m_MaxEmitted = 0;           // the most emitted light a surfel receives (its links' irradiance, luminance)
            v3      m_MaxEmitter{ 0, 0, 0 };    // the brightest emitter's radiance
            int     m_nUnsupported = 0;         // this update's rays whose hit read no surfel (Indirect unsupported: the policy's light, Unsupported)
            double  m_UploadMs  = 0;            // the renderer's: the shown light (BasisLight, its filter) and the moments into the GPU's buffers

            // Phase 0 instrumentation (documentation/gi_dynamic_master_plan.md): where an update's main-thread time goes, stage by stage. Measuring
            // only: nothing here changes what the solver does
            double  m_TlasLightsMs  = 0;        // the instances' TLAS and the light tracking (TrackLights)
            double  m_LevelsMs      = 0;        // the level / ambient cube switches
            double  m_PickMs        = 0;        // ordering the instances and picking this update's surfels
            double  m_LinksMs       = 0;        // the emitters' links (PumpLinks: bounded by its share of the allowance)
            double  m_EmitMs        = 0;        // the emitted light from the links (UpdateEmittedLight)
            double  m_MeanMs        = 0;        // every instance's mean light (a pass over every surfel)
            double  m_CommitMs      = 0;        // the traced results into the estimators, validity, basis and moments
            double  m_CubeMs        = 0;        // the ambient cubes' blend (ExpandCube)
            double  m_StatsMs       = 0;        // these statistics' own pass over every surfel (nBad, albedo, links...)
            double  m_TotalMs       = 0;        // the whole Update, the statistics included (m_Ms leaves them out)
            double  m_LongestSurfelUs = 0;      // the slowest surfel's trace (its rays and every hit's direct light and read): the longest unit of the trace
            long long m_Hits        = 0;        // rays that hit a lit surface: each one evaluates the direct light (the sun's shadow ray, the point lights in reach)
            int     m_nLinkRebuilds = 0;        // instances whose links were published this update (PumpLinks)
            int     m_LinkPending   = 0;        // instances waiting for (or in) a link build after this update
            int     m_LinkSurfels   = 0;        // receiver surfels whose links were built this update
            int     m_nEmitters     = 0;
            int     m_Pending       = 0;        // Phase 1: surfels picked and not finished (their rays continue next update)
            int     m_PendingRays   = 0;        // their rays still to trace
            int     m_ChunkRays     = 0;        // the rays of a unit this update (the time governor's choice)
            float   m_RayUsEst      = 0;        // the governor's conservative estimate of a ray's cost (us)
            int     m_Rounds        = 0;        // chunk rounds this update
            double  m_CastMs        = 0;        // the trace's worker time split (summed over the workers): casting the rays,
            double  m_DirectMs      = 0;        //   the direct light at the hits (shadow rays),
            double  m_ReadMs        = 0;        //   and the reads of the hit's light (Indirect, Unsupported)
            int     m_FairSurfels   = 0;        // surfels picked by the fairness share
            int     m_UrgentSurfels = 0;        // Phase 4: surfels picked from the urgent lists (a change affected them)
            double  m_PreMs     = 0;            // Phase 6b: what the frame spent on the GI before the update (events, the renderer's overhead)
            double  m_OverAllowMs = 0;          // Phase 6b: how far the update's stages went past the allowance (one unit at most; < 0: under it)
            double  m_TraceCpuMs = 0;           // Phase 6b batch accounting: the trace's units' time on the workers, summed (the main thread: m_TraceMs)
            double  m_LinkCpuMs  = 0;           // ... the link chunks' surfels, summed (the main thread: m_LinksMs)
            double  m_LinkLongestUs = 0;        // ... the slowest link surfel
            int     m_TokenUnits = 0;           // units started past the allowance (the token: one at most an update)
            int     m_nCubes        = 0;        // instances held as an ambient cube, and their parents blended this update (ExpandCube)
            int     m_CubeParents   = 0;
            int     m_nMoves        = 0;        // SetTransform calls since the last update
            double  m_MovesMs       = 0;        // their time (AffectNear, Place)
            int     m_nMarkedDirty  = 0;        // instances they marked dirty (AffectNear: each mover's old and new bounds, plus the margin)
            int     m_nHistoryResets = 0;       // instances whose history went to 0 this update (dirty: their next pass replaces their light)
            std::size_t m_LinkBytes  = 0;       // memory: the emitter links
            std::size_t m_LightBytes = 0;       // memory: the per-instance light state (light, estimators, validity, basis, moments, accumulators)
            std::size_t m_FilterBytes = 0;      // memory: the display filter's buffers
            // the renderer's, filled after Update (instance::updateGI)
            double  m_FilterMs      = 0;        // the display filter (BasisLight) of every shown instance
            double  m_RingCopyMs    = 0;        // copying the shown light into the frame's ring region
            double  m_MomentsMs     = 0;        // packing and copying the moments
            std::size_t m_UploadBytes = 0;      // bytes written into the frame's GI regions
            int     m_nDropped      = 0;        // shown instances left without GI: the frame's region was full
            double  m_UpdateGIMs    = 0;        // the renderer's whole updateGI (instances, Update, upload)
            double  m_GPUTraceMs    = 0;        // G3: the last batch's GPU time (gpu_trace), 0: the CPU traces
            int     m_GPUPacks      = 0;        // G3: the times the GPU's scene was packed in full
            double  m_GPUCpuMs      = 0;        // G3: the main thread's part of it (gpu_trace Collect and Submit: the packing, the uploads)
            int     m_GPUSliceRays  = 0;        // G3: the last slice's rays
            int     m_nScreenDraws  = 0;        // the half-resolution GI read's draws this frame (the renderer's: scene::gi_settings::m_bHalfRes)
            double  m_TimeMs        = 0;        // the renderer's time since its first GI frame (the frame log's clock: convergence over time)
            float   m_MeanLum       = 0;        // the luminance of the world's mean light (its valid surfels'): how far it has converged
            double  m_CaptureMs     = 0;        // the material capture (processGICapture)
            double  m_LoadMs        = 0;        // meshes' GI made ready this frame (scene_renderer::MeshOf: decoding, the rays' BVH)
        };

        struct hit
        {
            int     m_Inst  = -1, m_Tri = -1;
            float   m_T     = 1e30f, m_U = 0, m_V = 0;
        };

        setup   m_Setup;

        int Add(const surfel_mesh& Mesh, const xmath::fmat4& L2W, const v3& Albedo) noexcept
        {
            ++m_GPUGen;                                    // G3: the GPU's copy of the scene is packed again (GPUGen)
            m_Instances.push_back({ &Mesh });
            auto& I = m_Instances.back();
            I.m_Albedo = Albedo;
            Place(I, L2W);
            Resize(I);
            m_bTlasDirty = true;
            const int Id = int(m_Instances.size()) - 1;
            MarkLinks(Id);                                  // its own links
            MarkLinksCrossing(m_Instances[Id].m_Lo, m_Instances[Id].m_Hi);      // it may hide emitters from others
            return Id;
        }

        // The colors of an instance's surface, per fine surfel (its material sampled where the surfel is: albedo and emitted radiance).
        // The light a ray brings back from the instance is its color there times what lights it, plus what it emits. Empty: the
        // instance's albedo, no emission. Its neighbors re-solve (what they see changed)
        bool m_bEmitterLinks = true;            // false: emitters only through the rays' hits (tests compare the two)
        void SetSurfelMaterial(int Id, std::vector<v3> Albedo, std::vector<v3> Emission) noexcept
        {
            ++m_Instances[std::size_t(Id)].m_MatGen;      // (G4c: the GPU's copy updated in place: GPUMaterial)
            auto& I = m_Instances[Id];
            const std::size_t n = I.m_pMesh->m_Levels[0].size();
            I.m_SurfAlbedo   = Albedo.size() == n ? std::move(Albedo) : std::vector<v3>{};
            I.m_SurfEmissionPoint = Emission.size() == n ? std::move(Emission) : std::vector<v3>{};
            ResolveSurfEmission(Id);
            EmittersChanged(Id);
            AffectNear(I.m_Lo, I.m_Hi, Id);
        }

        // An instance's triangles' emission (the material capture, triangle First on). A surfel's emission is the mean over its patch, so
        // the solver's area factor (1.47 r^2) gives the patch's power: a light smaller than a surfel is not lost (a point sample reads
        // 0 or the full emission) and a surfel is not over-lit by it
        void SetTriangleEmission(int Id, std::size_t First, const std::vector<v3>& Emission) noexcept
        {
            ++m_Instances[std::size_t(Id)].m_MatGen;      // (G4c: the GPU's copy updated in place: GPUMaterial)
            auto& I = m_Instances[Id];
            const auto& M = *I.m_pMesh;
            const std::size_t n = M.m_Tris.size();
            if (First == 0 || I.m_SurfEmitPower.size() != M.m_Levels[0].size())    // a capture's first batch: the power starts again
            {
                I.m_SurfEmitPower.assign(M.m_Levels[0].size(), v3{ 0, 0, 0 });
                I.m_EmitUnassigned = v3{ 0, 0, 0 }; I.m_nEmitUnassigned = 0; I.m_bTriEmission = false;
            }
            AccumulateTriangleEmission(I, First, Emission);    // this batch's triangles only: the work of a batch, never the whole mesh at once
            if (First + Emission.size() < n) return;            // not the last batch
            I.m_bTriEmission = true;
            ResolveSurfEmission(Id);
            EmittersChanged(Id);
            AffectNear(I.m_Lo, I.m_Hi, Id);
        }

        // The emitted power of triangles First.. into the surfels that represent them (world::instance::m_SurfEmitPower). A triangle's
        // candidates are its vertices' surfels (surfel_mesh::m_VertSurfels: at most 12, deduplicated), those facing its way (cos >= 0.5);
        // its power (emission x area) is split over them by weights that are >= 0 and sum to 1 (the distance falloff at its centroid; equal
        // when none reaches it), so the power is conserved whatever the tessellation and never added in full to more than one surfel. A
        // triangle with no candidate is counted (m_EmitUnassigned), not searched for: the work is bounded by 12 a triangle. A large triangle
        // (below) is sampled with a capped search instead, so a big emissive panel lights the surfels over it, not only its corners'
        static void AccumulateTriangleEmission(auto& I, std::size_t First, const std::vector<v3>& Emission) noexcept
        {
            const auto& M = *I.m_pMesh;
            const auto& L = M.m_Levels[0];
            const bool bVert = M.m_VertSurfels.size() == M.m_Pos.size();
            if (L.empty()) return;
            float Rmin = L[0].m_Radius;
            for (const auto& S : L) Rmin = std::min(Rmin, S.m_Radius);
            // A triangle larger than its vertices' surfels can represent (more than 4 Rmin^2: the bake's candidate density) is sampled
            // instead: points over it, each to the nearest surfel facing its way in a grid of cells Rmin, the search capped at
            // emit_search_cap_v entries a point (the grid is made for the batch only when it has such a triangle)
            constexpr int emit_search_cap_v = 64, emit_samples_cap_v = 4096;
            surfel_grid G{ Rmin, {} };
            bool bGrid = false;
            for (std::size_t k = 0; k < Emission.size() && First + k < M.m_Tris.size(); ++k)
            {
                const v3& Le = Emission[k];
                if (Le.m_X + Le.m_Y + Le.m_Z <= 1e-6f) continue;
                const auto& T = M.m_Tris[First + k];
                const v3 A = M.m_Pos[T[0]], B = M.m_Pos[T[1]], C = M.m_Pos[T[2]];
                const v3 F = (B - A).Cross(C - A);
                const float Len = F.Length();
                if (Len < 2e-12f) continue;
                const v3 Nt = F * (1.0f / Len), Ct = (A + B + C) * (1.0f / 3.0f);
                const v3 Power = Le * (0.5f * Len);
                const float Area = 0.5f * Len;
                if (Area > 4.0f * Rmin * Rmin)
                {
                    if (!bGrid) { for (int q = 0; q < int(L.size()); ++q) G.Add(L[q].m_Center, q); bGrid = true; }
                    const int nS = std::clamp(int(std::ceil(Area * 4.0f / (Rmin * Rmin))), 1, emit_samples_cap_v);
                    const v3 Share = Power * (1.0f / float(nS));
                    for (int j = 0; j < nS; ++j)
                    {
                        const auto Frac = [](float x) { return x - std::floor(x); };
                        float u = Frac((float(j) + 0.5f) * 0.7548776662f), v = Frac((float(j) + 0.5f) * 0.5698402910f);
                        if (u + v > 1) { u = 1 - u; v = 1 - v; }
                        const v3 P = A + (B - A) * u + (C - A) * v;
                        float Best = 1e30f; int iBest = -1, nSeen = 0;
                        G.Near(P, 1, [&](int q)
                        {
                            if (nSeen >= emit_search_cap_v) return;
                            ++nSeen; ++I.m_EmitResolveWork;
                            if (L[q].m_Normal.Dot(Nt) < 0.5f) return;
                            const float d = (P - L[q].m_Center).Length();
                            if (d < Best) { Best = d; iBest = q; }
                        });
                        if (iBest >= 0) I.m_SurfEmitPower[iBest] = I.m_SurfEmitPower[iBest] + Share;
                        else { I.m_EmitUnassigned = I.m_EmitUnassigned + Share; ++I.m_nEmitUnassigned; }
                    }
                    continue;
                }
                int Cand[12]; float Wt[12]; int nC = 0;
                if (bVert)
                    for (int v = 0; v < 3; ++v)
                        for (const int s : M.m_VertSurfels[T[v]])
                        {
                            ++I.m_EmitResolveWork;
                            if (s < 0 || L[s].m_Normal.Dot(Nt) < 0.5f) continue;
                            bool bSeen = false;
                            for (int j = 0; j < nC; ++j) bSeen = bSeen || Cand[j] == s;
                            if (bSeen) continue;
                            Cand[nC] = s;
                            Wt[nC++] = std::max(0.0f, 1.0f - (Ct - L[s].m_Center).Length() / (2.0f * L[s].m_Radius));
                        }
                if (nC == 0) { I.m_EmitUnassigned = I.m_EmitUnassigned + Power; ++I.m_nEmitUnassigned; continue; }
                float Sum = 0;
                for (int j = 0; j < nC; ++j) Sum += Wt[j];
                for (int j = 0; j < nC; ++j)
                {
                    const float w = Sum > 1e-12f ? Wt[j] / Sum : 1.0f / float(nC);
                    I.m_SurfEmitPower[Cand[j]] = I.m_SurfEmitPower[Cand[j]] + Power * w;
                }
            }
        }

        // The surfels' emission: the radiance whose power over the surfel's nominal area (1.47 r^2, RebuildEmitterList) is the power the
        // triangles gave it (AccumulateTriangleEmission). Without the triangles' emission (or the surfels' own capture): the surfels' sample
        // at their center. O(surfels): the triangles were done batch by batch as they came
        void ResolveSurfEmission(int Id) noexcept
        {
            ++m_Instances[std::size_t(Id)].m_MatGen;      // (G4c: the GPU's copy updated in place: GPUMaterial)
            auto& I = m_Instances[Id];
            I.m_SurfEmission = I.m_SurfEmissionPoint;
            const auto& L = I.m_pMesh->m_Levels[0];
            if (!I.m_bTriEmission || L.empty() || I.m_SurfEmissionPoint.size() != L.size() || I.m_SurfEmitPower.size() != L.size()) return;
            for (std::size_t s = 0; s < L.size(); ++s) I.m_SurfEmission[s] = I.m_SurfEmitPower[s] * (1.0f / (1.47f * L[s].m_Radius * L[s].m_Radius));
        }

        // An instance's triangles' colors (the material capture, triangle First on): a ray hit takes its triangle's (SurfaceAt), so a
        // color ends where its triangles do - a surfel's color spreads over its whole footprint (a door frame's brown onto the wall)
        void SetTriangleAlbedo(int Id, std::size_t First, const std::vector<v3>& Albedo) noexcept
        {
            ++m_Instances[std::size_t(Id)].m_MatGen;      // (G4c: the GPU's copy updated in place: GPUMaterial)
            auto& I = m_Instances[Id];
            const std::size_t n = I.m_pMesh->m_Tris.size();
            if (I.m_TriAlbedo.size() != n) I.m_TriAlbedo.assign(n, I.m_Albedo);
            for (std::size_t k = 0; k < Albedo.size() && First + k < n; ++k) I.m_TriAlbedo[First + k] = Albedo[k];
            AffectNear(I.m_Lo, I.m_Hi, Id);
        }
        const std::vector<v3>& TriangleAlbedo( int Id ) const noexcept { return m_Instances[Id].m_TriAlbedo; }
        const std::vector<v3>& SurfaceEmission( int Id ) const noexcept { return m_Instances[Id].m_SurfEmission; }       // per fine surfel (tests)
        v3          EmissionUnassigned( int Id ) const noexcept { return m_Instances[Id].m_EmitUnassigned; }      // (tests) the triangles' power no surfel represents
        std::size_t EmissionResolveWork( int Id ) const noexcept { return m_Instances[Id].m_EmitResolveWork; }    // (tests) AccumulateTriangleEmission's candidates examined

        // An instance's transparent material slots (SlotSeeThrough[slot]: its material has alpha). The GI leaves them out, as the references
        // do (Lumen, DDGI, Wicked: translucent surfaces neither block nor bounce): rays pass through their triangles, their surfels go dark
        // (validity 0: the reads skip them) - a window lets the sky in, a glass pane does not bounce its own color
        void SetSeeThrough(int Id, const std::vector<bool>& SlotSeeThrough) noexcept
        {
            ++m_Instances[std::size_t(Id)].m_MatGen;      // (G4c: the GPU's copy updated in place: GPUMaterial)
            auto& I = m_Instances[Id];
            const auto& M = *I.m_pMesh;
            const auto Slot = [&](std::size_t s) { return s < SlotSeeThrough.size() && SlotSeeThrough[s]; };
            I.m_SeeThrough.assign(M.m_Tris.size(), 0);
            for (std::size_t t = 0; t < M.m_Tris.size(); ++t) I.m_SeeThrough[t] = Slot(t < M.m_TriMaterial.size() ? M.m_TriMaterial[t] : 0u);
            I.m_SurfSeeThrough.assign(M.m_Levels[0].size(), 0);
            for (std::size_t s = 0; s < I.m_SurfSeeThrough.size(); ++s) I.m_SurfSeeThrough[s] = Slot(s < M.m_SurfelMaterial.size() ? M.m_SurfelMaterial[s].m_iMaterial : 0u);
            if (std::find(I.m_SeeThrough.begin(), I.m_SeeThrough.end(), 1) == I.m_SeeThrough.end()) { I.m_SeeThrough.clear(); I.m_SurfSeeThrough.clear(); }
            AffectNear(I.m_Lo, I.m_Hi, Id);
        }

        // 7a (gi_dynamic_master_plan.md): an instance whose material is not captured yet (the host's capture queued, SetSurfelMaterial and
        // SetTriangleAlbedo / SetTriangleEmission to come). It stays in the ray scene (it occludes) with a provisional lower-energy
        // response: its hits bring no light (no albedo, no emission) and it has no emitters, where its default albedo (a material's base
        // color factor: white for most textured ones) would make it a bright reflector. When the last of the initial ones is captured,
        // every instance's light continues as a warm start with its confidence lowered (DirtyAll: estimator::Lower at its next pass, which
        // is after the batch in flight) - the light made from provisional materials is not kept as settled. Later instances only relight
        // their neighbors, as any material change does
        void SetProvisional(int Id, bool bProvisional) noexcept
        {
            auto& I = m_Instances[std::size_t(Id)];
            if (I.m_bProvisional == bProvisional) return;
            I.m_bProvisional = bProvisional;
            m_nProvisional += bProvisional ? 1 : -1;
            ++I.m_MatGen;
            EmittersChanged(Id);
            AffectNear(I.m_Lo, I.m_Hi, Id);
            if (!bProvisional && m_nProvisional == 0 && !m_bBooted) { m_bBooted = true; DirtyAll(); }
        }
        bool IsProvisional(int Id) const noexcept { return m_Instances[std::size_t(Id)].m_bProvisional; }

        const std::vector<v3>& SurfelAlbedo  ( int Id ) const noexcept { return m_Instances[Id].m_SurfAlbedo; }
        const std::vector<v3>& SurfelEmission( int Id ) const noexcept { return m_Instances[Id].m_SurfEmission; }

        void DirtyAll() noexcept { for (auto& I : m_Instances) if (I.m_bAlive) MarkDirty(I, 1.0f); }

        // Phase 6: the explicit "relight everything" (a command): every instance traced again under the allowance (sliced, never at once),
        // each surfel's light the warm start (estimator::Lower). At most once in m_RelightPeriod updates: a call sooner is kept and done
        // when the period has passed (false: deferred)
        bool RelightAll() noexcept
        {
            if (m_nUpdates < m_RelightAt) { m_bRelightDeferred = true; return false; }
            DirtyAll();
            m_RelightAt = m_nUpdates + m_Setup.m_RelightPeriod; m_bRelightDeferred = false;
            return true;
        }

        // The angle between two placements' rotations (radians; their scale left out)
        static float RotationAngle(const xmath::fmat4& A, const xmath::fmat4& B) noexcept
        {
            float Tr = 0;
            for (int c = 0; c < 3; ++c)
            {
                const v3 a = v3{ A(0, c), A(1, c), A(2, c) }.NormalizeCopy(), b = v3{ B(0, c), B(1, c), B(2, c) }.NormalizeCopy();
                Tr += a.Dot(b);
            }
            return std::acos(std::clamp((Tr - 1.0f) * 0.5f, -1.0f, 1.0f));
        }

        // The instance stops casting, bouncing and updating (its id is not reused)
        void Remove(int Id) noexcept
        {
            ++m_GPUGen;
            auto& I = m_Instances[Id];
            if (!I.m_bAlive) return;
            if (I.m_bProvisional) SetProvisional(Id, false);     // (7a: the count of those not captured yet)
            I.m_bAlive = false;
            if (Transports(I)) AffectNear(I.m_Lo, I.m_Hi);
            I.m_E.clear(); I.m_Acc.clear(); I.m_Valid.clear(); I.m_AccValid.clear();
            I.m_Links.clear(); I.m_LinkFirst.clear(); I.m_EmitB.clear(); I.m_bLinkHull = false; I.m_SurfLinkLo.clear(); I.m_SurfLinkHi.clear();
            m_bTlasDirty = true;
            if (!Transports(I)) I.m_EmitSurfs.clear();                                                      // (it was in nobody's light)
            else if (!I.m_EmitSurfs.empty()) { I.m_EmitSurfs.clear(); m_bEmittersDirty = true; MarkAllLinks(); }   // its light is gone for everyone
            else MarkLinksCrossing(I.m_Lo, I.m_Hi);                                                          // what it hid is seen again
        }

        bool IsAlive(int Id) const noexcept { return m_Instances[Id].m_bAlive; }

        // Phase 5: an instance's transport role, over its mesh's (ASSET: the mesh's m_bBounce). A change of what it does for the others
        // (blocking, bouncing, emitting) is a change for them: the rays' scene, the surfels near it, the links and the emitters
        void SetRole(int Id, role Role) noexcept
        {
            auto& I = m_Instances[Id];
            if (I.m_Role == Role) return;
            const bool bWas = Transports(I);
            I.m_Role = Role;
            if (!I.m_bAlive || bWas == Transports(I)) return;
            m_bTlasDirty = true;
            AffectNear(I.m_Lo, I.m_Hi, Id);
            MarkLinksCrossing(I.m_Lo, I.m_Hi);
            if (!I.m_EmitSurfs.empty()) { m_bEmittersDirty = true; MarkAllLinks(); }
        }
        role GetRole(int Id) const noexcept { return m_Instances[Id].m_Role; }

        void SetTransform(int Id, const xmath::fmat4& L2W) noexcept
        {
            ++m_GPUXformGen;                                // (G4c: the GPU's TLAS and records again, not the scene: PackGPUTlas)
            const auto T0 = std::chrono::steady_clock::now();
            auto& I = m_Instances[Id];
            const v3 OldLo = I.m_Lo, OldHi = I.m_Hi;
            const xmath::fmat4 Old = I.m_L2W;
            Place(I, L2W);
            // Phase 4: it keeps its layout and starts from its light, its confidence lowered by half for a move, more by its rotation (the
            // basis turns with it: its directions are no longer what they saw; the angle weighs it, a heuristic, not an error bound)
            const float SelfA = std::max(0.5f, std::clamp(RotationAngle(Old, L2W) / (0.5f * pi_v), 0.0f, 1.0f));
            if (Transports(I))
            {
                m_MoveMarked += AffectNear(OldLo, OldHi, Id, SelfA);
                m_MoveMarked += AffectNear(I.m_Lo, I.m_Hi, Id, SelfA);
                m_bTlasDirty = true;
            }
            else MarkDirty(I, SelfA);                       // a receiver (Phase 5): only its own light; the others' and the rays' scene are as they were
            // The links (Phase 2): an emitter that moves changes what everyone sees of it; an occluder, only the links whose segments
            // cross its old or new box; a receiver, its own. They are rebuilt in bounded chunks (PumpLinks), the published ones stay in use
            if (!Transports(I)) MarkLinks(Id);                                         // its own links only
            else if (!I.m_EmitSurfs.empty()) { m_bEmittersDirty = true; MarkAllLinks(); }
            else { MarkLinks(Id); MarkLinksCrossing(OldLo, OldHi); MarkLinksCrossing(I.m_Lo, I.m_Hi); }
            ++m_nMovesAcc;
            m_MovesMsAcc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();
        }

        int             InstanceCount  ( void ) const noexcept { return int(m_Instances.size()); }
        int             Level          ( int Id ) const noexcept { return m_Instances[Id].m_Level; }
        bool            IsCube         ( int Id ) const noexcept { return m_Instances[Id].m_bCube; }
        const std::vector<int>& CubeSet( int Id ) const noexcept { return m_Instances[Id].m_CubeSet; }
        // GI chunks (xlionrender_surfel_gi_chunks.h): Id's level-0 surfels from nOwned on are ghosts of Owners' (instance, surfel) - copied
        // from them after each commit, never traced, never emitters. Equal levels only (plan v3: mixed levels gated)
        void SetGhosts(int Id, std::uint32_t nOwned, std::vector<std::array<int, 2>> Owners) noexcept
        {
            auto& I = m_Instances[std::size_t(Id)];
            I.m_nOwned = nOwned; I.m_GhostOwner = std::move(Owners);
            m_bEmittersDirty = true;
            ++m_GPUGen;                                         // (gpu_trace: its ghost list, packed again)
        }
        std::uint32_t OwnedSurfels(int Id) const noexcept { const auto& I = m_Instances[std::size_t(Id)]; return std::min<std::uint32_t>(I.m_nOwned, std::uint32_t(I.m_E.size())); }
        const std::vector<std::array<int, 2>>& GhostOwners(int Id) const noexcept { return m_Instances[std::size_t(Id)].m_GhostOwner; }      // (gpu_trace: its ghost pass)
        // The ghosts' published set from their owners (the commit's epoch: before the means and the filter read it)
        void SyncGhosts() noexcept
        {
            for (auto& I : m_Instances)
            {
                if (!I.m_bAlive || I.m_GhostOwner.empty() || I.m_Level != 0) continue;
                for (std::size_t k = 0; k < I.m_GhostOwner.size(); ++k)
                {
                    const std::size_t g = std::size_t(I.m_nOwned) + k;
                    const auto& [oi, os] = I.m_GhostOwner[k];
                    if (g >= I.m_E.size() || oi < 0 || std::size_t(oi) >= m_Instances.size()) continue;
                    const auto& O = m_Instances[std::size_t(oi)];
                    if (!O.m_bAlive || O.m_Level != 0 || std::size_t(os) >= O.m_E.size()) continue;      // (mixed levels: gated - the ghost keeps its last set)
                    I.m_E[g] = O.m_E[std::size_t(os)]; I.m_Valid[g] = O.m_Valid[std::size_t(os)];
                    if (g < I.m_Moments.size() && std::size_t(os) < O.m_Moments.size()) I.m_Moments[g] = O.m_Moments[std::size_t(os)];
                    if (g < I.m_Frac.size() && std::size_t(os) < O.m_Frac.size()) I.m_Frac[g] = O.m_Frac[std::size_t(os)];
                    I.m_FilterDirty.push_back(int(g));
                }
                I.m_bMeanDirty = true;
            }
        }
        int             LinkRebuilds   ( void ) const noexcept { return m_nLinkRebuilds; }
        int             LinkCount      ( int Id ) const noexcept { return int(m_Instances[Id].m_Links.size()); }
        std::uint32_t   LastPass       ( int Id, int s ) const noexcept { return m_Instances[Id].m_LastPass[s]; }       // the update that last finished it
        std::array<float, 2> RayOffsets( int Id, int s ) const noexcept { return m_Instances[Id].m_LastOff[s]; }        // its last pass' ray offsets
        std::uint32_t   UpdateCount    ( void ) const noexcept { return m_nUpdates; }
        int             Confidence     ( int Id, int s ) const noexcept { return m_Instances[Id].m_Est[s].m_N; }      // its samples since a restart, lowered by a change (estimator::Lower)
        int             UrgentCount    ( int Id ) const noexcept { return int(m_Instances[Id].m_Urgent.size()); }
        // (tests) the published links of an instance as one number: two builds of the same scene give the same
        std::uint64_t   LinksDigest    ( int Id ) const noexcept
        {
            std::uint64_t h = 1469598103934665603ull;
            const auto Mix = [&](std::uint64_t v) { h = (h ^ v) * 1099511628211ull; };
            for (const auto f : m_Instances[Id].m_LinkFirst) Mix(f);
            for (const auto& L : m_Instances[Id].m_Links) { Mix(L.m_EmInst); Mix(L.m_EmSurf); Mix(std::bit_cast<std::uint32_t>(L.m_W)); }
            return h;
        }
        const v3&       Albedo         ( int Id ) const noexcept { return m_Instances[Id].m_Albedo; }
        // What the GPU buffer would hold for an instance: its active level's irradiance and validity per surfel
        const std::vector<v3>&    InstanceLight( int Id ) const noexcept { return m_Instances[Id].m_E; }
        // The emitters' light reaching an instance through its links this update (the sum over its surfels of the basis' mean, red)
        double                    InstanceEmitted( int Id ) const noexcept { double e = 0; for (const auto& B : m_Instances[Id].m_EmitB) e += (B[0].m_X + B[1].m_X + B[2].m_X) / 3.0; return e; }

        // The light the reads show (the GPU's upload), per basis direction: 3 x the light times its fractions (their mean is the light),
        // blended with the neighbors' (the mesh's neighbor lists, their weights times their validity; itself at weight 1),
        // m_Setup.m_FilterPasses times over, then saturated. The independent noise of neighboring surfels averages out. Only what is
        // shown: the solver keeps its own light (a filter fed back every update would keep spreading the light). Kept by UpdateShown
        const std::vector<basis>& BasisLight( int Id ) const noexcept
        {
            ShownSettings();
            const auto& I = m_Instances[Id];
            if (I.m_Shown.size() != I.m_E.size() || I.m_bFilterFull || I.m_FullPass >= 0) UpdateShown(Id);
            return I.m_Shown;
        }
        // What the last Update changed of an instance's shown light (the renderer uploads only that): all of it, or these surfels
        bool                      ShownFull   ( int Id ) const noexcept { return m_Instances[Id].m_bShownFull; }
        // (tests) the whole filter of an instance's current inputs, made apart (its kept passes and its shown light are not touched)
        std::vector<basis>        FilterOf    ( int Id ) const noexcept
        {
            const auto& I = m_Instances[Id];
            world Copy; Copy.m_Setup = m_Setup;
            Copy.m_Instances.push_back(I);
            Copy.m_Instances[0].m_bFilterFull = true; Copy.m_Instances[0].m_FullPass = -1; Copy.m_Instances[0].m_Shown.clear();
            Copy.UpdateShown(0);
            return Copy.m_Instances[0].m_Shown;
        }
        void                      ShowAllAgain( void ) const noexcept { for (int i = 0; i < int(m_Instances.size()); ++i) if (m_Instances[i].m_bAlive) { m_Instances[i].m_bFilterFull = true; UpdateShown(i); } }   // (tests: the full filter, to compare)
        const std::vector<int>&   ShownChanged( int Id ) const noexcept { return m_Instances[Id].m_ShownChanged; }

        // The shown light kept up to date (Phase 3; Phase 6b: under the update's deadline, resumable). The passes are kept; the surfels
        // whose light, fractions or validity changed (m_FilterDirty: the ones finished) are pass 0's, and each further pass is made again
        // for the surfels that read any made again by the pass before (the reverse neighbor lists): the dependency halo grows one ring a
        // pass. That work goes in chunks of 64 dirty surfels until the deadline. A level switch, a new layout or a new setting makes all of
        // it again through a cursor (pass by pass, in blocks), also until the deadline: its pass 0 is shown at once (unfiltered) when the
        // layout changed, the filtered values replace it as the passes complete. bToken: this stage may start one chunk past the deadline
        // (the update's one-unit overshoot, given to the stage that waited longest). Returns whether it did any work.
        //
        // Publication (what the renderer uploads of BasisLight): the unit is one surfel's shown basis, written whole. Neighboring surfels
        // may be of different revisions while a backlog is worked off (a surfel is made from the passes as they are when it is made); once
        // the inputs stop changing, every surfel reaches the whole filter of the final inputs (test 6j12, 6j16)
        bool UpdateShown(int Id, double DeadlineMs = 0, const std::chrono::steady_clock::time_point* pT0 = nullptr, bool* pToken = nullptr) const noexcept
        {
            const auto& I = m_Instances[Id];
            const int   n  = int(I.m_E.size());
            const auto& Nb = I.m_pMesh->m_Neighbors[I.m_Level];
            const bool  bNb = Nb.m_First.size() == std::size_t(n) + 1;
            const int   nP  = bNb ? std::clamp(m_Setup.m_FilterPasses, 0, 16) : 0;
            I.m_ShownChanged.clear(); I.m_bShownFull = false;
            const auto Late = [&] { return pT0 && DeadlineMs > 0 && std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - *pT0).count() >= DeadlineMs; };
            const auto Admit = [&](bool bDidAny) { if (!Late()) return true; if (!bDidAny && pToken && *pToken) { *pToken = false; return true; } return false; };
            const auto Pass0 = [&](int s)
            {
                for (int i = 0; i < 3; ++i)
                {
                    const v3& f = I.m_Frac[s][i];
                    I.m_Pass[0][s][i] = v3{ I.m_E[s].m_X * f.m_X * 3.0f, I.m_E[s].m_Y * f.m_Y * 3.0f, I.m_E[s].m_Z * f.m_Z * 3.0f };
                }
            };
            const auto PassAt = [&](int p, int s)      // (plain floats through raw pointers: a Debug build calls every v3 operator)
            {
                const basis* pIn = I.m_Pass[p - 1].data();
                const basis& Me = pIn[s];
                float a0 = Me[0].m_X, a1 = Me[0].m_Y, a2 = Me[0].m_Z, a3 = Me[1].m_X, a4 = Me[1].m_Y, a5 = Me[1].m_Z, a6 = Me[2].m_X, a7 = Me[2].m_Y, a8 = Me[2].m_Z, W = 1;
                for (std::uint32_t k = Nb.m_First[s], k1 = Nb.m_First[s + 1]; k < k1; ++k)
                {
                    const std::uint32_t j = Nb.m_Index[k];
                    const float w = Nb.m_Weight[k] * I.m_Valid[j];
                    const basis& B = pIn[j];
                    a0 += B[0].m_X * w; a1 += B[0].m_Y * w; a2 += B[0].m_Z * w;
                    a3 += B[1].m_X * w; a4 += B[1].m_Y * w; a5 += B[1].m_Z * w;
                    a6 += B[2].m_X * w; a7 += B[2].m_Y * w; a8 += B[2].m_Z * w;
                    W += w;
                }
                const float r = 1.0f / W;
                basis& O = I.m_Pass[p][s];
                O[0].m_X = a0 * r; O[0].m_Y = a1 * r; O[0].m_Z = a2 * r;
                O[1].m_X = a3 * r; O[1].m_Y = a4 * r; O[1].m_Z = a5 * r;
                O[2].m_X = a6 * r; O[2].m_Y = a7 * r; O[2].m_Z = a8 * r;
            };
            const auto Show = [&](int s, int p) { I.m_Shown[s] = I.m_Pass[p][s]; SaturateOne(I.m_Shown[s]); };
            const auto Blocks = [&](std::size_t Count, auto&& Fn)     // Fn(k) for k in [0, Count) across the workers, in blocks
            {
                constexpr std::size_t block_v = 256;
                ParallelFor(int((Count + block_v - 1) / block_v), [&](int b) { for (std::size_t k = std::size_t(b) * block_v; k < std::min(Count, std::size_t(b + 1) * block_v); ++k) Fn(k); });
            };

            // a new layout or setting: the cursor starts (pass 0 shown at once when the layout changed: the renderer needs a value per surfel)
            const bool bLayout = I.m_Shown.size() != std::size_t(n) || I.m_Pass.size() != std::size_t(nP) + 1 || I.m_Pass[0].size() != std::size_t(n);
            if (I.m_bFilterFull || bLayout)
            {
                if (bLayout)
                {
                    I.m_Pass.assign(std::size_t(nP) + 1, std::vector<basis>(std::size_t(n)));
                    I.m_Shown.resize(std::size_t(n)); I.m_Mark.assign(std::size_t(n), 0);
                    Blocks(std::size_t(n), [&](std::size_t k) { Pass0(int(k)); Show(int(k), 0); });
                    I.m_bShownFull = true;
                }
                I.m_RevFirst.clear(); I.m_RevIndex.clear();
                if (bNb)        // who reads each surfel (the neighbor lists need not be symmetric: the radii may differ)
                {
                    I.m_RevFirst.assign(std::size_t(n) + 1, 0);
                    for (const auto j : Nb.m_Index) ++I.m_RevFirst[j + 1];
                    for (int s = 0; s < n; ++s) I.m_RevFirst[s + 1] += I.m_RevFirst[s];
                    I.m_RevIndex.resize(Nb.m_Index.size());
                    std::vector<std::uint32_t> At(I.m_RevFirst.begin(), I.m_RevFirst.end() - 1);
                    for (int s = 0; s < n; ++s) for (std::uint32_t k = Nb.m_First[s]; k < Nb.m_First[s + 1]; ++k) I.m_RevIndex[At[Nb.m_Index[k]]++] = std::uint32_t(s);
                }
                I.m_bFilterFull = false;
                I.m_FullPass = bLayout ? 1 : 0; I.m_FullAt = 0;
            }
            // a quarter of it or more dirty (a GPU batch: thousands a frame): the whole filter is less work than the dirty chunks' halos (each
            // chunk makes its own halo again: a floor all dirty made millions of surfels per update)
            // (checked again when a whole filter ends: what was dirtied while it ran)
            bool bDid = false;
            for (int Round = 0; Round < 2; ++Round)
            {
            if (I.m_FullPass < 0 && I.m_FilterDirty.size() * 4 > std::size_t(n)) { I.m_FilterDirty.clear(); I.m_FullPass = 0; I.m_FullAt = 0; }
            // the whole filter, resumable: pass by pass (pass nP + 1: shown), blocks of 4096 surfels
            if (I.m_FullPass >= 0)
            {
                constexpr int full_block_v = 4096;
                while (I.m_FullPass >= 0 && Admit(bDid))
                {
                    const int p = I.m_FullPass, s0 = I.m_FullAt, s1 = std::min(n, s0 + full_block_v);
                    if (p == 0)           Blocks(std::size_t(s1 - s0), [&](std::size_t k) { Pass0(s0 + int(k)); });
                    else if (p <= nP)     Blocks(std::size_t(s1 - s0), [&](std::size_t k) { PassAt(p, s0 + int(k)); });
                    else                { Blocks(std::size_t(s1 - s0), [&](std::size_t k) { Show(s0 + int(k), nP); }); for (int s = s0; s < s1; ++s) I.m_ShownChanged.push_back(s); }
                    bDid = true;
                    I.m_FullAt = s1;
                    if (s1 >= n) { I.m_FullAt = 0; if (++I.m_FullPass > nP + 1) I.m_FullPass = -1; }
                }
                if (I.m_FullPass >= 0) return bDid;          // the dirty surfels wait for the whole filter (they are made again after it)
            }
            }
            if (I.m_FilterDirty.empty()) return bDid;
            // the halo work in chunks of 64 dirty surfels until the deadline; the rest waits for the next update
            std::vector<int> Changed;
            std::size_t Taken = 0;
            while (Taken < I.m_FilterDirty.size() && Admit(bDid))
            {
                std::vector<int> Chunk;
                const std::size_t End = std::min(I.m_FilterDirty.size(), Taken + 64);
                for (; Taken < End; ++Taken) if (const int x = I.m_FilterDirty[Taken]; x < n && !I.m_Mark[x]) { I.m_Mark[x] = 1; Chunk.push_back(x); }
                Blocks(Chunk.size(), [&](std::size_t k) { Pass0(Chunk[k]); });
                for (int p = 1; p <= nP; ++p)
                {
                    const std::size_t k0 = Chunk.size();
                    for (std::size_t c = 0; c < k0; ++c)
                        for (std::uint32_t k = I.m_RevFirst[Chunk[c]]; k < I.m_RevFirst[Chunk[c] + 1]; ++k)
                            if (const auto r = I.m_RevIndex[k]; !I.m_Mark[r]) { I.m_Mark[r] = 1; Chunk.push_back(int(r)); }
                    Blocks(Chunk.size(), [&](std::size_t k) { PassAt(p, Chunk[k]); });
                }
                Blocks(Chunk.size(), [&](std::size_t k) { Show(Chunk[k], nP); });
                for (const int x : Chunk) I.m_Mark[x] = 0;
                Changed.insert(Changed.end(), Chunk.begin(), Chunk.end());
                bDid = true;
            }
            I.m_FilterDirty.erase(I.m_FilterDirty.begin(), I.m_FilterDirty.begin() + std::ptrdiff_t(Taken));
            I.m_ShownChanged.insert(I.m_ShownChanged.end(), Changed.begin(), Changed.end());
            return bDid;
        }

        // A new setting of what is shown (the passes, the saturation): all of it is made again
        void ShownSettings() const noexcept
        {
            if (m_ShownPasses == m_Setup.m_FilterPasses && m_ShownSat == m_Setup.m_Saturation) return;
            for (const auto& I : m_Instances) I.m_bFilterFull = true;
            m_ShownPasses = m_Setup.m_FilterPasses; m_ShownSat = m_Setup.m_Saturation;
        }

        // A color of the shown basis moved from its luminance by m_Setup.m_Saturation (the luminance kept; what the filter and the
        // surfels' size average away). Only what is shown (the solver's light is not touched: its bounces would compound it)
        void SaturateOne(basis& B) const noexcept
        {
            const float Sa = m_Setup.m_Saturation;
            if (Sa == 1.0f) return;
            for (int i = 0; i < 3; ++i)
            {
                v3& C = B[i];
                const float Y = 0.2126f * C.m_X + 0.7152f * C.m_Y + 0.0722f * C.m_Z;
                C.m_X = std::max(Y + (C.m_X - Y) * Sa, 0.0f); C.m_Y = std::max(Y + (C.m_Y - Y) * Sa, 0.0f); C.m_Z = std::max(Y + (C.m_Z - Y) * Sa, 0.0f);
            }
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
            std::fprintf(f, "inst,level,surfel,px,py,pz,nx,ny,nz,radius,er,eg,eb,fr,fg,fb,valid,ar,ag,ab,xr,xg,xb,n,watch,short,dev,parent\n");      // x: the captured emission (before EmissiveScale); parent: its level 1 surfel (level 1: itself)
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
                    const v3 X  = I.m_Level == 0 && s < I.m_SurfEmission.size() ? I.m_SurfEmission[s] : v3{ 0, 0, 0 };
                    std::fprintf(f, "%d,%d,%zu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.4f,%.4f,%.4f,%.4f,%.6g,%.6g,%.6g,%d,%d,%.6g,%.6g,%d\n"
                                , Id, I.m_Level + (I.m_bCube ? 1 : 0), s, P.m_X, P.m_Y, P.m_Z, N.m_X, N.m_Y, N.m_Z, L[s].m_Radius * I.m_MeanScale
                                , I.m_E[s].m_X, I.m_E[s].m_Y, I.m_E[s].m_Z, Fs.m_X, Fs.m_Y, Fs.m_Z
                                , s < I.m_Valid.size() ? I.m_Valid[s] : 0.0f, A.m_X, A.m_Y, A.m_Z, X.m_X, X.m_Y, X.m_Z
                                , s < I.m_Est.size() ? int(I.m_Est[s].m_N) : -1, s < I.m_Est.size() ? int(I.m_Est[s].m_bWatch) : 0     // (its estimator: samples, watching, short mean, deviation: luminance)
                                , s < I.m_Est.size() ? Lum(I.m_Est[s].m_Short) : 0.0f, s < I.m_Est.size() ? std::sqrt(std::max(0.0f, Lum(I.m_Est[s].m_Variance))) : 0.0f
                                , I.m_Level == 0 && s < I.m_pMesh->m_ParentOf.size() ? I.m_pMesh->m_ParentOf[s] : int(s));
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
            // (every hit of every ray reads it: plain floats, a Debug build calls every v3 operator)
            const auto& I = m_Instances[Id];
            const auto& S = I.m_pMesh->m_Levels[I.m_Level][s];
            const float Nx = S.m_Normal.m_X, Ny = S.m_Normal.m_Y, Nz = S.m_Normal.m_Z;
            const float Dx = P.m_X - S.m_Center.m_X, Dy = P.m_Y - S.m_Center.m_Y, Dz = P.m_Z - S.m_Center.m_Z;
            const float Reach = 2 * S.m_Radius;
            float w = std::clamp(1.0f - std::abs(Dx * Nx + Dy * Ny + Dz * Nz) / (0.7f * Reach), 0.0f, 1.0f);
            if (w <= 0) return w;
            const float Qx = Dx + Nx * (0.15f * Reach), Qy = Dy + Ny * (0.15f * Reach), Qz = Dz + Nz * (0.15f * Reach);   // lifted off the surface, as the shaders (GIVisibility)
            const float d  = std::sqrt(Qx * Qx + Qy * Qy + Qz * Qz), id = 1.0f / d;
            const float Ux = Qx * id, Uy = Qy * id, Uz = Qz * id;
            // T = (|Nx| < 0.9 ? X : Y) x N, normalized; B = N x T
            float Tx, Ty, Tz;
            if (std::abs(Nx) < 0.9f) { Tx = 0; Ty = -Nz; Tz = Ny; } else { Tx = Nz; Ty = 0; Tz = -Nx; }
            const float it = 1.0f / std::sqrt(Tx * Tx + Ty * Ty + Tz * Tz); Tx *= it; Ty *= it; Tz *= it;
            const float Bx = Ny * Tz - Nz * Ty, By = Nz * Tx - Nx * Tz, Bz = Nx * Ty - Ny * Tx;
            const float x  = Ux * Tx + Uy * Ty + Uz * Tz, y = Ux * Bx + Uy * By + Uz * Bz, z = std::max(Ux * Nx + Uy * Ny + Uz * Nz, 0.0f);
            const float k  = 1.0f / std::max(std::abs(x) + std::abs(y) + z, 1e-6f);
            const float u  = std::clamp(((x + y) * k * 0.5f + 0.5f) * moment_res_v - 0.5f, 0.0f, float(moment_res_v - 1));
            const float vv = std::clamp(((x - y) * k * 0.5f + 0.5f) * moment_res_v - 0.5f, 0.0f, float(moment_res_v - 1));
            const int   u0 = int(u), v0 = int(vv), u1 = std::min(u0 + 1, moment_res_v - 1), v1 = std::min(v0 + 1, moment_res_v - 1);
            const float fu = u - float(u0), fv = vv - float(v0);
            const float* M = I.m_Moments[s].data();
            const float a00 = (1 - fu) * (1 - fv), a10 = fu * (1 - fv), a01 = (1 - fu) * fv, a11 = fu * fv;
            const int   t00 = 2 * (v0 * moment_res_v + u0), t10 = 2 * (v0 * moment_res_v + u1), t01 = 2 * (v1 * moment_res_v + u0), t11 = 2 * (v1 * moment_res_v + u1);
            const float Mean = M[t00] * a00 + M[t10] * a10 + M[t01] * a01 + M[t11] * a11;
            const float M2   = M[t00 + 1] * a00 + M[t10 + 1] * a10 + M[t01 + 1] * a01 + M[t11 + 1] * a11;
            const float Var  = std::max(std::abs(Mean * Mean - M2), 0.0025f * Reach * Reach);     // bilinear, a variance floor: as the shaders
            if (d > Mean) { const float r = Var / (Var + (d - Mean) * (d - Mean) + 1e-12f); w *= r * r * r; }
            return w;
        }

        // Part II, G1 (documentation/gi_dynamic_master_plan.md): the rays' structures for the GPU (shaders/xlionrender_gi_bvh.glsl, the same
        // walk as Cast). Every mesh's BVH4 nodes and the TLAS's in one array, their child codes global (>= 0 a node, < 0 ~ a leaf: for a mesh
        // its 4-triangle row in m_Tri4, for the TLAS its entry in m_TlasLeaves); per TLAS primitive its instance (the world-to-mesh rows,
        // its mesh's root code, its world id). The node and row layouts are bvh::node and tri4 as they are (std430: 112 and 160 bytes).
        // ponytail: see-through triangles are not left out yet (G2)
        struct gpu_scene
        {
            struct instance { float m_W2L[3][4]; int m_Root = 0, m_Inst = -1, m_Pad0 = 0, m_Pad1 = 0; };
            std::vector<bvh::node>  m_Nodes;
            std::vector<tri4>       m_Tri4;
            std::vector<instance>   m_Instances;
            std::vector<int>        m_TlasLeaves;       // per TLAS leaf its first and count in the primitives after them (m_PrimOffset on)
            int                     m_TlasRoot   = 0;
            int                     m_PrimOffset = 0;
            bool                    m_bEmpty     = true;
            int                     m_nMeshNodes = 0;   // G4c: the meshes' nodes come first, the TLAS's after them (PackGPUTlas)
            std::unordered_map<const surfel_mesh*, int> m_MeshRoots;     // each mesh's root code
        };
        void PackGPU(gpu_scene& Out) const noexcept
        {
            BuildTlas();
            Out = {};
            Out.m_bEmpty = m_Tlas.m_bEmpty;
            if (Out.m_bEmpty) return;
            for (const int i : m_TlasIds)             // the meshes' BVHs first (G4c: a move changes the TLAS only, at the end: PackGPUTlas)
            {
                const auto* pM = m_Instances[std::size_t(i)].m_pMesh;
                auto [It, bNew] = Out.m_MeshRoots.try_emplace(pM, 0);
                if (!bNew) continue;
                It->second = CopyBvh(Out, pM->m_Bvh, int(Out.m_Nodes.size()), int(Out.m_Tri4.size()));
                Out.m_Tri4.insert(Out.m_Tri4.end(), pM->m_Tri4.begin(), pM->m_Tri4.end());
            }
            Out.m_nMeshNodes = int(Out.m_Nodes.size());
            PackGPUTlas(Out);
        }
        // G4c: the TLAS again (its nodes after the meshes', the instances' rows, its leaves): the instances moved, the meshes are the same
        // (false: one is not in Out, PackGPU)
        bool PackGPUTlas(gpu_scene& Out) const noexcept
        {
            BuildTlas();
            Out.m_bEmpty = m_Tlas.m_bEmpty;
            if (Out.m_bEmpty) return true;
            Out.m_Nodes.resize(std::size_t(Out.m_nMeshNodes));
            Out.m_Instances.clear(); Out.m_TlasLeaves.clear();
            Out.m_TlasRoot = CopyBvh(Out, m_Tlas, Out.m_nMeshNodes, 0);
            for (const int i : m_TlasIds)
            {
                const auto& I = m_Instances[std::size_t(i)];
                const auto It = Out.m_MeshRoots.find(I.m_pMesh);
                if (It == Out.m_MeshRoots.end()) return false;
                gpu_scene::instance G;
                for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) G.m_W2L[r][c] = I.m_W2L(r, c);
                G.m_Root = It->second; G.m_Inst = i;
                Out.m_Instances.push_back(G);
            }
            for (const auto& L : m_Tlas.m_Leaves) { Out.m_TlasLeaves.push_back(L.m_First); Out.m_TlasLeaves.push_back(L.m_Count); }
            Out.m_PrimOffset = int(Out.m_TlasLeaves.size());
            Out.m_TlasLeaves.insert(Out.m_TlasLeaves.end(), m_Tlas.m_Prims.begin(), m_Tlas.m_Prims.end());
            return true;
        }
        static int CopyBvh(gpu_scene& Out, const bvh& B, int NodeBase, int LeafBase) noexcept       // its nodes, the child codes made global
        {
            for (auto N : B.m_Nodes)
            {
                for (int k = 0; k < 4; ++k) N.m_Child[k] = N.m_Child[k] >= 0 ? N.m_Child[k] + NodeBase : ~(~N.m_Child[k] + LeafBase);
                Out.m_Nodes.push_back(N);
            }
            return B.m_Root >= 0 ? B.m_Root + NodeBase : ~(~B.m_Root + LeafBase);
        }
        std::uint64_t GPUXformGen() const noexcept { return m_GPUXformGen; }      // G4c: the instances' placements' version (PackGPUTlas)
        // G4c: an instance's placement words as the GPU's record has them (15 mean scale, 16..27 L2W rows, 28..39 W2L rows)
        void GPUPlacement(int Id, std::uint32_t (&W)[25]) const noexcept
        {
            const auto& I = m_Instances[std::size_t(Id)];
            W[0] = std::bit_cast<std::uint32_t>(I.m_MeanScale);
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) { W[1 + 4 * r + c] = std::bit_cast<std::uint32_t>(I.m_L2W(r, c)); W[13 + 4 * r + c] = std::bit_cast<std::uint32_t>(I.m_W2L(r, c)); }
        }

        // Part II, G2: everything a surfel pass reads besides the rays' structures, for shaders/xlionrender_gi_surfel_pass_comp.glsl, as
        // one array of words (floats as their bits). Per mesh a header (gpu_mesh_v words) of offsets into the array; per world instance a
        // record (gpu_inst_v words); the lights; the setup. Layouts (word offsets):
        //   mesh:     0 nVerts 1 pos 2 nrm 3 tris 4 vertSurfels (2 per uint, 0xFFFF none) 5 level0 (7 floats: center, normal, radius)
        //             6 n0 7 level1 8 n1 9 parentOf 10 readTable (4 per entry) 11 its size 12 readList 13 readCell 14 originFirst
        //             15 originPos 16 originNrm 17 leak protection 18..20 level 0's neighbor lists (first, index, weight: the shown
        //             light's filter; none) 21..23 level 1's 24, 25 per parent its children (first: n1 + 1, index: the re-level)
        //   instance: 0 mesh 1 level 2 E 3 valid 4 moments (32 per surfel) 5..7 mean light 8..10 albedo 11 triAlbedo 12 surfAlbedo
        //             13 surfEmission 14 see-through bits 15 meanScale 16..27 L2W rows 28..39 W2L rows 40 alive   (~0u: none)
        //             G4 (the commit on the GPU): 41 fractions (9 per surfel) 42 estimators (gpu_est_v: PackEst) 43 moment counts (16)
        //             44 emitted light from the links (9, zeros: none) 45 transparent surfels' bits (level 0; none) 46 its first surfel
        //             in the shown light's global order (gpu_trace writes it) 47 its material's flags (GPUMaterial: 11..14 and 45 are
        //             always there, the flags say which hold its material)
        //   header:   0 instances 1 nInstances 2 lights 3 nPoints 4 sky oct (0/1) 5 emissive scale 6 spread origins 7 no emitters
        //             8 the warm-up's least blend (m_MinBlend) 9 warmup_passes_v 10 rays per surfel 11 flags (1 adaptive rays, 2 tiers)
        //             12 the rays' uncertainty factor 13 light sampling (G5: 1, 0) 14 the emitter table (GPUEmitters) 15 its count
        //             (5, 7, 8, 10..13: PackGPUSetup)
        //   lights:   sun direction 3, sun color 3, sky 3, sky oct 48, then per point position 3, color 3, radius
        // ponytail: the whole scene every call (G3 keeps it resident and uploads changes)
        static constexpr std::uint32_t gpu_mesh_v = 26, gpu_inst_v = 48, gpu_none_v = 0xFFFFFFFFu, gpu_header_v = 16, gpu_est_v = 24;
        static constexpr int gpu_lanes_v = 8, gpu_group_v = 64;    // the pass: invocations per item, per group (a dispatch: GPUGroups)
        static constexpr int GPUGroups(std::size_t nItems) noexcept { return int((nItems * gpu_lanes_v + gpu_group_v - 1) / gpu_group_v); }
        struct gpu_item { int m_Inst = 0, m_Surf = 0, m_Level = 0, m_Rays = 0; float m_Off1 = 0, m_Off2 = 0; int m_Shift = 0, m_Pad = 0; };
        struct gpu_result                       // per item: what TraceChunk accumulates, before it finishes the pass
        {
            float   m_Sum[3];
            float   m_Basis[9];
            float   m_Moments[3 * moment_texels_v];
            int     m_nBack, m_nBackAll, m_nUnsupported, m_nHits;
            float   m_UnsupportedAt[3], m_Pad;
            int     m_Work[4];                  // (diagnostics) the GPU's work for it: BVH nodes popped, shadow rays, read list entries, read hash probes
        };
        void PackGPUSurfels(std::vector<std::uint32_t>& D, const lights& L) const noexcept
        {
            // (whole blocks through raw pointers: the Debug build's checked push_back per word took 290 ms for Police's 11.6M words)
            D.assign(gpu_header_v, 0);
            const auto F = [](float f) { return std::bit_cast<std::uint32_t>(f); };
            const auto Grow = [&](std::size_t n) { const std::size_t At = D.size(); D.resize(At + n); return At; };
            const auto Words = [&](const void* p, std::size_t n) { const std::size_t At = Grow(n); if (n) std::memcpy(D.data() + At, p, n * 4); return std::uint32_t(At); };
            const auto V3s = [&](const v3* p, std::size_t n) { const std::size_t At = Grow(3 * n); std::uint32_t* o = D.data() + At; for (std::size_t k = 0; k < n; ++k) std::memcpy(o + 3 * k, &p[k].m_X, 12); return std::uint32_t(At); };
            const auto PushV = [&](const v3& v) { V3s(&v, 1); };
            std::unordered_map<const surfel_mesh*, std::uint32_t> Meshes;
            const auto MeshOf = [&](const surfel_mesh& M) -> std::uint32_t
            {
                if (auto It = Meshes.find(&M); It != Meshes.end()) return It->second;
                std::uint32_t H[gpu_mesh_v]{};
                H[0] = std::uint32_t(M.m_Pos.size());
                H[1] = V3s(M.m_Pos.data(), M.m_Pos.size());
                H[2] = V3s(M.m_Nrm.data(), M.m_Nrm.size());
                static_assert(sizeof(M.m_Tris[0]) == 12 && sizeof(M.m_ReadTable[0]) == 16);
                H[3] = Words(M.m_Tris.data(), 3 * M.m_Tris.size());
                {
                    const std::size_t nV = M.m_Pos.size(), At = Grow(2 * nV);
                    std::uint32_t* o = D.data() + At;
                    const auto* pVS = M.m_VertSurfels.data();
                    for (std::size_t v = 0; v < nV; ++v)
                    {
                        std::uint32_t U[4] = { 0xFFFFu, 0xFFFFu, 0xFFFFu, 0xFFFFu };
                        if (v < M.m_VertSurfels.size()) for (int k = 0; k < 4; ++k) if (pVS[v][std::size_t(k)] >= 0) U[k] = std::uint32_t(pVS[v][std::size_t(k)]);
                        o[2 * v] = U[0] | (U[1] << 16); o[2 * v + 1] = U[2] | (U[3] << 16);
                    }
                    H[4] = std::uint32_t(At);
                }
                for (int Lv = 0; Lv < 2; ++Lv)
                {
                    const auto& Lev = M.m_Levels[Lv];
                    const std::size_t At = Grow(7 * Lev.size());
                    std::uint32_t* o = D.data() + At;
                    for (std::size_t k = 0; k < Lev.size(); ++k) { std::memcpy(o + 7 * k, &Lev[k].m_Center.m_X, 12); std::memcpy(o + 7 * k + 3, &Lev[k].m_Normal.m_X, 12); o[7 * k + 6] = F(Lev[k].m_Radius); }
                    H[5 + 2 * Lv] = std::uint32_t(At); H[6 + 2 * Lv] = std::uint32_t(Lev.size());
                }
                H[9]  = Words(M.m_ParentOf.data(), M.m_ParentOf.size());
                H[10] = Words(M.m_PackedReadTable.data(), 4 * M.m_PackedReadTable.size());     // (the packed lists: PackReadGrid)
                H[11] = std::uint32_t(M.m_PackedReadTable.size());
                H[12] = Words(M.m_PackedReadList.data(), M.m_PackedReadList.size());
                H[13] = F(M.m_ReadCell);
                H[14] = M.m_OriginFirst.empty() ? gpu_none_v : Words(M.m_OriginFirst.data(), M.m_OriginFirst.size());
                H[15] = Words(M.m_OriginVert.data(), M.m_OriginVert.size());     // (per origin a vertex of H[1] / H[2], or the surfel's own: GIOrigin)
                H[16] = gpu_none_v;
                H[17] = M.m_bLeakProtection ? 1u : 0u;
                for (int Lv = 0; Lv < 2; ++Lv)
                {
                    const auto& Nb = M.m_Neighbors[Lv];
                    if (Nb.m_First.size() != M.m_Levels[Lv].size() + 1) { H[18 + 3 * Lv] = H[19 + 3 * Lv] = H[20 + 3 * Lv] = gpu_none_v; continue; }
                    H[18 + 3 * Lv] = Words(Nb.m_First.data(), Nb.m_First.size());
                    H[19 + 3 * Lv] = Words(Nb.m_Index.data(), Nb.m_Index.size());
                    H[20 + 3 * Lv] = Words(Nb.m_Weight.data(), Nb.m_Weight.size());
                }
                {   // per parent its children (the parents' order of m_ParentOf)
                    const std::size_t n1 = M.m_Levels[1].size();
                    std::vector<std::uint32_t> First(n1 + 1, 0);
                    for (const int p : M.m_ParentOf) if (std::size_t(p) < n1) ++First[std::size_t(p) + 1];
                    for (std::size_t p = 0; p < n1; ++p) First[p + 1] += First[p];
                    std::vector<std::uint32_t> Index(First.back()), At(First.begin(), First.end() - 1);
                    for (std::size_t c = 0; c < M.m_ParentOf.size(); ++c) if (std::size_t(M.m_ParentOf[c]) < n1) Index[At[std::size_t(M.m_ParentOf[c])]++] = std::uint32_t(c);
                    H[24] = Words(First.data(), First.size());
                    H[25] = Words(Index.data(), Index.size());
                }
                const std::uint32_t At = Words(H, gpu_mesh_v);
                Meshes[&M] = At;
                return At;
            };
            std::vector<std::uint32_t> Recs(m_Instances.size() * gpu_inst_v, gpu_none_v);
            for (std::size_t i = 0; i < m_Instances.size(); ++i)
            {
                const auto& I = m_Instances[i];
                std::uint32_t* R = &Recs[i * gpu_inst_v];
                R[40] = I.m_bAlive ? 1u : 0u;
                if (!I.m_bAlive) continue;
                R[0] = MeshOf(*I.m_pMesh);
                R[1] = std::uint32_t(I.m_Level);
                const std::size_t nS = I.m_E.size();
                // room for both levels in every per-surfel array (the finer one's count): a level switch re-levels in place (gpu_trace::Relevel)
                const std::size_t nRoom = std::max({ nS, I.m_pMesh->m_Levels[0].size(), I.m_pMesh->m_Levels[1].size() });
                R[2] = V3s(I.m_E.data(), nS); Grow(3 * (nRoom - nS));
                R[3] = Words(I.m_Valid.data(), I.m_Valid.size()); Grow(nRoom - I.m_Valid.size());
                static_assert(sizeof(moments) == 4 * 2 * moment_texels_v);
                R[4] = Words(I.m_Moments.data(), I.m_Moments.size() * 2 * moment_texels_v); Grow((nRoom - I.m_Moments.size()) * 2 * moment_texels_v);
                R[5] = F(I.m_MeanE.m_X); R[6] = F(I.m_MeanE.m_Y); R[7] = F(I.m_MeanE.m_Z);
                R[8] = F(I.m_Albedo.m_X); R[9] = F(I.m_Albedo.m_Y); R[10] = F(I.m_Albedo.m_Z);
                {   // its material, always there (changed in place: GPUMaterial)
                    std::vector<std::uint32_t> A[5];
                    R[47] = GPUMaterial(int(i), A);
                    const std::uint32_t Word[5] = { 11, 12, 13, 14, 45 };
                    for (int k = 0; k < 5; ++k) R[Word[k]] = Words(A[k].data(), A[k].size());
                }
                R[15] = F(I.m_MeanScale);
                {
                    const std::size_t At = Grow(9 * nRoom);
                    std::uint32_t* o = D.data() + At;
                    const float Third = 1.0f / 3;
                    for (std::size_t k = 0; k < nS; ++k)
                        for (int c = 0; c < 3; ++c)
                            if (k < I.m_Frac.size()) std::memcpy(o + 9 * k + 3 * std::size_t(c), &I.m_Frac[k][std::size_t(c)].m_X, 12);
                            else for (int q = 0; q < 3; ++q) o[9 * k + 3 * std::size_t(c) + std::size_t(q)] = F(Third);
                    R[41] = std::uint32_t(At);
                }
                {
                    const std::size_t At = Grow(nRoom * gpu_est_v);
                    for (std::size_t k = 0; k < nS; ++k)
                    {
                        if (k < I.m_Est.size()) { PackEst(I.m_Est[k], D.data() + At + k * gpu_est_v); continue; }
                        estimator Es; Es.Seed(k < I.m_E.size() ? I.m_E[k] : v3{ 0, 0, 0 }); Es.Restart();     // (GPU-owned: no CPU estimators: a new
                                                                    // instance's, as Resize makes them; one packed before keeps its own: the migrate)
                        PackEst(Es, D.data() + At + k * gpu_est_v);
                    }
                    R[42] = std::uint32_t(At);
                }
                {
                    const std::size_t At = Grow(nRoom * moment_texels_v);
                    std::uint32_t* o = D.data() + At;
                    for (std::size_t k = 0; k < nS && k < I.m_MomCount.size(); ++k) for (int t = 0; t < moment_texels_v; ++t) o[k * moment_texels_v + std::size_t(t)] = I.m_MomCount[k][std::size_t(t)];
                    R[43] = std::uint32_t(At);
                }
                {
                    const std::size_t At = Grow(nRoom * 9);
                    std::uint32_t* o = D.data() + At;
                    for (std::size_t k = 0; k < nS && k < I.m_EmitB.size(); ++k) for (int c = 0; c < 3; ++c) std::memcpy(o + 9 * k + 3 * std::size_t(c), &I.m_EmitB[k][std::size_t(c)].m_X, 12);
                    R[44] = std::uint32_t(At);
                }
                for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) { R[16 + 4 * r + c] = F(I.m_L2W(r, c)); R[28 + 4 * r + c] = F(I.m_W2L(r, c)); }
            }
            D[0] = std::uint32_t(D.size()); D[1] = std::uint32_t(m_Instances.size());
            D.insert(D.end(), Recs.begin(), Recs.end());
            D[2] = std::uint32_t(D.size()); D[3] = std::uint32_t(L.m_Points.size());
            PushV(L.m_SunDirection); PushV(L.m_SunColor); PushV(L.m_Sky);
            for (const auto& s : L.m_SkyOct) PushV(s);
            for (const auto& P : L.m_Points) { PushV(P.m_Position); PushV(P.m_Color); D.push_back(F(P.m_Radius)); PushV(P.m_Dir); D.push_back(F(P.m_CosOuter)); D.push_back(F(P.m_CosInner)); }     // (gpu_light_words_v)
            {   // G5: the emitter table, with room (gpu_trace writes it every slice; past its room: packed again)
                std::vector<std::uint32_t> Em;
                GPUEmitters(Em);
                D[14] = std::uint32_t(D.size()); D[15] = std::uint32_t(Em.size() / gpu_emitter_v);
                D.insert(D.end(), Em.begin(), Em.end());
                D.resize(D.size() + std::size_t(GPUEmitterRoom(D[15]) - D[15]) * gpu_emitter_v, 0u);
            }
            D[4] = L.m_bSkyOct ? 1u : 0u;
            D[6] = m_Setup.m_bSpreadOrigins ? 1u : 0u;
            D[9] = std::uint32_t(warmup_passes_v);
            PackGPUSetup(D.data());
        }
        // G4c: an instance's material as the GPU has it (record words 11 triangle albedo 3 a triangle, 12 surfel albedo 3 a fine surfel, 13 surfel
        // emission, 14 the triangles' see-through bits, 45 the fine surfels'): always their full size (zeros: none), the flags (bits 0..4)
        // which hold its material, bit 5 whether it has links (m_EmitB)
        std::uint32_t GPUMaterial(int Id, std::vector<std::uint32_t> (&A)[5]) const noexcept
        {
            const auto& I = m_Instances[std::size_t(Id)];
            const std::size_t nT = I.m_pMesh->m_Tris.size(), nF = I.m_pMesh->m_Levels[0].size();
            const auto F = [](float f) { return std::bit_cast<std::uint32_t>(f); };
            const auto V = [&](std::vector<std::uint32_t>& Out, const std::vector<v3>& In, std::size_t n) { Out.assign(3 * n, 0u); std::uint32_t* o = Out.data(); for (std::size_t k = 0; k < In.size() && k < n; ++k) std::memcpy(o + 3 * k, &In[k].m_X, 12); };
            const auto B = [&](std::vector<std::uint32_t>& Out, const std::vector<std::uint8_t>& In, std::size_t n) { Out.assign((n + 31) / 32, 0u); std::uint32_t* o = Out.data(); const std::uint8_t* p = In.data(); for (std::size_t k = 0; k < In.size() && k < n; ++k) if (p[k]) o[k / 32] |= 1u << (k % 32); };
            V(A[0], I.m_TriAlbedo, nT); V(A[1], I.m_SurfAlbedo, nF); V(A[2], I.m_SurfEmission, nF);
            B(A[3], I.m_SeeThrough, nT); B(A[4], I.m_SurfSeeThrough, nF);
            for (auto& a : A) if (a.empty()) a.push_back(0u);
            return (I.m_TriAlbedo.empty() ? 0u : 1u) | (I.m_SurfAlbedo.empty() ? 0u : 2u) | (I.m_SurfEmission.empty() ? 0u : 4u) | (I.m_SeeThrough.empty() ? 0u : 8u) | (I.m_SurfSeeThrough.empty() ? 0u : 16u)
                 | (I.m_EmitB.empty() ? 0u : 32u)      // (32: it has links: its rays' hits leave the emission out, the links bring it)
                 | (I.m_bProvisional ? 64u : 0u);      // (64: 7a, not captured yet: its hits bring nothing)
        }
        std::uint32_t MatGen(int Id) const noexcept { return m_Instances[std::size_t(Id)].m_MatGen; }
        const xmath::fmat4& InstanceL2W(int Id) const noexcept { return m_Instances[std::size_t(Id)].m_L2W; }            // (gpu_trace: the links' local rebuild)
        std::pair<v3, v3>   InstanceBounds(int Id) const noexcept { return { m_Instances[std::size_t(Id)].m_Lo, m_Instances[std::size_t(Id)].m_Hi }; }
        const surfel_mesh&  InstanceMesh(int Id) const noexcept { return *m_Instances[std::size_t(Id)].m_pMesh; }

        // G5: the emitter list as the GPU's light sampling reads it: per emitter (gpu_emitter_v words) its world center 3, normal 3, area,
        // radiance now 3 (the emissive scale in it), instance, surfel. Room: what the table is packed with for Count of them
        static constexpr std::uint32_t gpu_emitter_v = 12;
        std::size_t EmitterCount() const noexcept { return m_Em.size(); }
        static constexpr std::uint32_t GPUEmitterRoom(std::uint32_t Count) noexcept { return std::max<std::uint32_t>(1024, 2 * Count); }
        void GPUEmitters(std::vector<std::uint32_t>& Out) const noexcept
        {
            Out.clear(); Out.reserve(m_Em.size() * gpu_emitter_v);
            const auto F = [](float f) { return std::bit_cast<std::uint32_t>(f); };
            for (std::size_t k = 0; k < m_Em.size(); ++k)
            {
                const auto& E = m_Em[k]; const v3 Le = EmitterRadiance(k);
                const std::uint32_t W[gpu_emitter_v] = { F(E.m_C.m_X), F(E.m_C.m_Y), F(E.m_C.m_Z), F(E.m_N.m_X), F(E.m_N.m_Y), F(E.m_N.m_Z), F(E.m_A)
                                                       , F(Le.m_X), F(Le.m_Y), F(Le.m_Z), std::uint32_t(m_Emitters[k].first), std::uint32_t(m_Emitters[k].second) };
                Out.insert(Out.end(), W, W + gpu_emitter_v);
            }
        }

        // The setup the GPU's commit and expand read (header words 5, 7, 8, 10..12: the GPU's copy is refreshed every slice)
        void PackGPUSetup(std::uint32_t* D) const noexcept
        {
            D[5]  = std::bit_cast<std::uint32_t>(m_Setup.m_EmissiveScale);
            D[7]  = m_Emitters.empty() ? 1u : 0u;
            D[8]  = std::bit_cast<std::uint32_t>(m_Setup.m_MinBlend);
            D[10] = std::uint32_t(std::max(m_Setup.m_RaysPerSurfel, 1));
            D[11] = (m_Setup.m_bAdaptiveRays ? 1u : 0u) | (m_Setup.m_bDistanceTiers ? 2u : 0u) | (m_Setup.m_bAblateHitRead ? 4u : 0u) | (m_Setup.m_bAblateFreeze ? 8u : 0u) | (m_Setup.m_bAblateStats ? 16u : 0u) | (m_Setup.m_bAblateFallback ? 32u : 0u);
            D[12] = std::bit_cast<std::uint32_t>(m_Setup.m_RayUncertainty);
            D[13] = m_Setup.m_bEmitterSampling ? 1u : 0u;
        }

        // G4: an estimator as gpu_est_v words (the GPU's commit, shaders/xlionrender_gi_commit_comp.glsl): mean 3, short 3, variance 3, vbbr,
        // inconsistency, n, streak, taken, watch (bit 0) and took (bit 1), last 3, ref 3, the noise's last luminance, sum, count
        static void PackEst(const estimator& E, std::uint32_t* p) noexcept
        {
            const auto F = [](float f) { return std::bit_cast<std::uint32_t>(f); };
            const v3* V[] = { &E.m_Mean, &E.m_Short, &E.m_Variance };
            for (int k = 0; k < 3; ++k) { p[3 * k] = F(V[k]->m_X); p[3 * k + 1] = F(V[k]->m_Y); p[3 * k + 2] = F(V[k]->m_Z); }
            p[9] = F(E.m_Vbbr); p[10] = F(E.m_Inconsistency); p[11] = E.m_N; p[12] = std::uint32_t(std::int32_t(E.m_Streak)); p[13] = E.m_nTaken; p[14] = (E.m_bWatch ? 1u : 0u) | (E.m_bTook ? 2u : 0u);
            p[15] = F(E.m_Last.m_X); p[16] = F(E.m_Last.m_Y); p[17] = F(E.m_Last.m_Z); p[18] = F(E.m_Ref.m_X); p[19] = F(E.m_Ref.m_Y); p[20] = F(E.m_Ref.m_Z);
            p[21] = F(E.m_PrevLuma); p[22] = F(E.m_DiffSq); p[23] = E.m_nDiff;
        }
        static void UnpackEst(const std::uint32_t* p, estimator& E) noexcept
        {
            const auto F = [](std::uint32_t u) { return std::bit_cast<float>(u); };
            v3* V[] = { &E.m_Mean, &E.m_Short, &E.m_Variance };
            for (int k = 0; k < 3; ++k) *V[k] = { F(p[3 * k]), F(p[3 * k + 1]), F(p[3 * k + 2]) };
            E.m_Vbbr = F(p[9]); E.m_Inconsistency = F(p[10]); E.m_N = std::uint16_t(p[11]); E.m_Streak = std::int8_t(std::int32_t(p[12])); E.m_nTaken = std::uint16_t(p[13]); E.m_bWatch = (p[14] & 1u) != 0; E.m_bTook = (p[14] & 2u) != 0;
            E.m_Last = { F(p[15]), F(p[16]), F(p[17]) }; E.m_Ref = { F(p[18]), F(p[19]), F(p[20]) };
            E.m_PrevLuma = F(p[21]); E.m_DiffSq = F(p[22]); E.m_nDiff = std::uint16_t(p[23]);
        }

        // G4: a change to the estimators the world made (estimator::Lower: at a dirty instance's pass, by ApplyAffect), for the GPU's
        // copy (shaders/xlionrender_gi_ops_comp.glsl): both apply it, the same way. Surf -1: all of the instance's; bMomCount: its moment
        // counts start again
        struct gpu_op { int m_Inst = 0, m_Surf = -1; float m_A = 1; int m_bMomCount = 0; };

        // (tests) The CPU's surfel pass for the same items (TraceChunk, all rays at once), its sums before the pass is finished. Items whose
        // instance has emitter links get their finishing light in m_Basis (the scenes compared have none)
        void TraceReference(std::span<const gpu_item> Items, const lights& L, std::vector<gpu_result>& Out) noexcept
        {
            BuildTlas();
            if (L.m_Points.size() != m_nLightBvh) BuildLightBvh(L);
            Out.assign(Items.size(), gpu_result{});
            for (std::size_t k = 0; k < Items.size(); ++k)
            {
                const auto& G = Items[k];
                work_item W; W.m_Inst = G.m_Inst; W.m_Surf = G.m_Surf; W.m_Level = G.m_Level; W.m_Rays = G.m_Rays; W.m_Off1 = G.m_Off1; W.m_Off2 = G.m_Off2; W.m_Shift = G.m_Shift;
                EnsureSolver(m_Instances[G.m_Inst]);
                TraceChunk(W, 0, W.m_Rays, L);
                const auto& I = m_Instances[G.m_Inst];
                auto& R = Out[k];
                for (int a = 0; a < 3; ++a) R.m_Sum[a] = W.m_Sum[a];
                for (int i = 0; i < 3; ++i) for (int a = 0; a < 3; ++a) R.m_Basis[3 * i + a] = I.m_BasisAcc[G.m_Surf][i][a];
                for (int t = 0; t < 3 * moment_texels_v; ++t) R.m_Moments[t] = I.m_MomAcc[G.m_Surf][t];
                R.m_nBack = W.m_nBack; R.m_nBackAll = W.m_nBackAll; R.m_nUnsupported = W.m_nUnsupported; R.m_nHits = W.m_nHits;
                for (int a = 0; a < 3; ++a) R.m_UnsupportedAt[a] = W.m_UnsupportedAt[a];
            }
        }

        // The nearest hit before TMax (the instances as they are now: the TLAS is rebuilt after Add / SetTransform; not
        // thread safe across that rebuild - Update rebuilds before its parallel part)
        // G3 (gi_dynamic_master_plan.md, gpu_trace): the picked items not yet handed to the GPU, handed now, in m_Pending's order
        std::span<const gpu_item> GPUWork() noexcept
        {
            m_GPUOut.clear();
            for (auto& W : m_Pending)
                if (!W.m_bGPU && W.m_RaysDone < W.m_Rays) { W.m_bGPU = true; m_GPUOut.push_back({ W.m_Inst, W.m_Surf, W.m_Level, W.m_Rays, W.m_Off1, W.m_Off2, W.m_Shift, 0 }); }
            return m_GPUOut;
        }
        // Their sums (Items: what GPUWork handed, or a slice of it, in order; R per item) into the passes, finished: the next Update commits them. An item dropped since
        // (Update: its instance removed, its level changed) is skipped: m_Pending keeps its order, what is left of Items is in it in order
        void DeliverGPU(std::span<const gpu_item> Items, std::span<const gpu_result> R) noexcept
        {
            std::size_t j = 0;
            for (std::size_t k = 0; k < Items.size() && j < m_Pending.size(); ++k)
            {
                while (j < m_Pending.size() && m_Pending[j].m_bGPU && m_Pending[j].m_RaysDone >= m_Pending[j].m_Rays) ++j;     // (delivered by an earlier slice)
                if (j >= m_Pending.size()) break;
                auto& W = m_Pending[j];
                if (!W.m_bGPU || W.m_Inst != Items[k].m_Inst || W.m_Surf != Items[k].m_Surf || W.m_Level != Items[k].m_Level) continue;
                ++j;
                const auto& G = R[k];
                auto& I = m_Instances[W.m_Inst];
                EnsureSolver(I);                                        // (the CPU commits it: its pass' accumulators)
                W.m_Sum = { G.m_Sum[0], G.m_Sum[1], G.m_Sum[2] };
                for (int i = 0; i < 3; ++i) I.m_BasisAcc[W.m_Surf][i] = { G.m_Basis[3 * i], G.m_Basis[3 * i + 1], G.m_Basis[3 * i + 2] };
                std::copy_n(G.m_Moments, 3 * moment_texels_v, I.m_MomAcc[W.m_Surf].begin());
                W.m_nBack = G.m_nBack; W.m_nBackAll = G.m_nBackAll; W.m_nUnsupported = G.m_nUnsupported; W.m_nHits = G.m_nHits;
                W.m_UnsupportedAt = { G.m_UnsupportedAt[0], G.m_UnsupportedAt[1], G.m_UnsupportedAt[2] };
                W.m_RaysDone = W.m_Rays;
                m_GPURays += W.m_Rays;
                FinishPass(W);
            }
        }
        std::uint64_t GPUGen() const noexcept { return m_GPUGen; }
        // every instance's material generation summed: a capture landing changes it (the GPU's links, built from the materials, are made again)
        std::uint64_t MaterialGenSum() const noexcept { std::uint64_t n = 0; for (const auto& I : m_Instances) n += I.m_MatGen; return n; }
        std::uint64_t GPULevelGen() const noexcept { return m_GPULevelGen; }     // an instance changed level (the arrays have room for both: PackGPUSurfels)

        // G4c: a range of a surfel pass (GPURanges): the GPU's expand serves its surfels (shaders/xlionrender_gi_expand_comp.glsl). Flags:
        // gpu_range_urgent_v (a change's surfel: no tier), gpu_range_settled_v (its instance is settled: the tiers apply). m_Start: its
        // first item in the batch's global order (the sum of the counts before it)
        static constexpr int gpu_range_urgent_v = 1, gpu_range_settled_v = 2;
        struct gpu_range { int m_Inst = 0, m_Level = 0, m_First = 0, m_Count = 0, m_Flags = 0, m_History = 0; std::uint32_t m_Seed = 0; int m_Start = 0, m_Rays = 0; };   // (m_Rays: the estimate, the slices')
        // what the GPU found for a range (its items' atomics): the rays, the surfels traced, unsupported, hits, the first unsupported hit
        struct gpu_range_stats { int m_Rays = 0, m_Traced = 0, m_nUnsupported = 0, m_nHits = 0; float m_UnsupportedAt[3] = {}; int m_bAt = 0; };
        std::span<const gpu_range> GPURanges() noexcept
        {
            m_GPURangesOut.clear();
            int Start = 0;
            for (auto& W : m_Pending)
                if (!W.m_bGPU && W.m_RaysDone < W.m_Rays)
                {
                    W.m_bGPU = true;
                    m_GPURangesOut.push_back({ W.m_Inst, W.m_Level, W.m_Surf, W.m_Count, W.m_Flags, W.m_History, W.m_Seed, Start, W.m_Rays });
                    Start += W.m_Count;
                }
            return m_GPURangesOut;
        }
        // The ranges traced and committed (Ranges: a slice of what GPURanges handed, in order; St per range): finished, Update counts them.
        // Each instance learns what a step of its cursor costs (the budget's estimate of its next ranges)
        void DeliverGPURanges(std::span<const gpu_range> Ranges, std::span<const gpu_range_stats> St) noexcept
        {
            std::size_t j = 0;
            for (std::size_t k = 0; k < Ranges.size() && j < m_Pending.size(); ++k)
            {
                while (j < m_Pending.size() && m_Pending[j].m_bGPU && m_Pending[j].m_RaysDone >= m_Pending[j].m_Rays) ++j;
                if (j >= m_Pending.size()) break;
                auto& W = m_Pending[j];
                if (!W.m_bGPU || W.m_Inst != Ranges[k].m_Inst || W.m_Surf != Ranges[k].m_First || W.m_Level != Ranges[k].m_Level || W.m_Count != Ranges[k].m_Count) continue;
                ++j;
                const auto& G = St[k];
                auto& I = m_Instances[W.m_Inst];
                if (!(W.m_Flags & gpu_range_urgent_v) && W.m_Count > 0)
                {
                    const float r = float(G.m_Rays) / float(W.m_Count);
                    I.m_RaysPerStep = I.m_RaysPerStep > 0 ? I.m_RaysPerStep + (r - I.m_RaysPerStep) * 0.25f : r;
                }
                W.m_nUnsupported = G.m_nUnsupported; W.m_nHits = G.m_nHits; W.m_Traced = G.m_Traced;
                W.m_UnsupportedAt = { G.m_UnsupportedAt[0], G.m_UnsupportedAt[1], G.m_UnsupportedAt[2] };
                W.m_RaysDone = W.m_Rays; W.m_bCommitted = true;
                m_GPURays += G.m_Rays;
            }
        }
        // The instances' means of their valid surfels' light, from the GPU (per instance 5 floats: the sum of E V, of V, the bad ones)
        void SetGPUMeans(std::span<const float> M) noexcept
        {
            for (std::size_t i = 0; i < m_Instances.size() && 5 * i + 4 < M.size(); ++i)
            {
                auto& I = m_Instances[i];
                if (!I.m_bAlive) continue;
                I.m_MeanS = { M[5 * i], M[5 * i + 1], M[5 * i + 2] }; I.m_MeanW = M[5 * i + 3]; I.m_nBad = int(M[5 * i + 4]);
                I.m_bMeanDirty = false; I.m_nMeanDeltas = 0;
            }
        }
        // (G4c: gpu_trace::ReadBack) the GPU's state of an instance's surfels into this world's arrays: E, validity, moments, fractions,
        // estimators, moment counts (the layouts of PackGPUSurfels)
        void SetGPUState(int Id, const std::uint32_t* E, const std::uint32_t* V, const std::uint32_t* Mo, const std::uint32_t* Fr, const std::uint32_t* Es, const std::uint32_t* Mc) noexcept
        {
            auto& I = m_Instances[std::size_t(Id)];
            EnsureSolver(I);                                            // (the estimators and counts read back: GPU-owned, not kept otherwise)
            const auto F = [](std::uint32_t u) { return std::bit_cast<float>(u); };
            for (std::size_t s = 0; s < I.m_E.size(); ++s)
            {
                I.m_E[s] = { F(E[3 * s]), F(E[3 * s + 1]), F(E[3 * s + 2]) }; I.m_Valid[s] = F(V[s]);
                for (int t = 0; t < 2 * moment_texels_v; ++t) I.m_Moments[s][std::size_t(t)] = F(Mo[2 * moment_texels_v * s + std::size_t(t)]);
                for (int k = 0; k < 3; ++k) I.m_Frac[s][k] = { F(Fr[9 * s + 3 * k]), F(Fr[9 * s + 3 * k + 1]), F(Fr[9 * s + 3 * k + 2]) };
                UnpackEst(Es + gpu_est_v * s, I.m_Est[s]);
                for (int t = 0; t < moment_texels_v; ++t) I.m_MomCount[s][std::size_t(t)] = std::uint16_t(Mc[moment_texels_v * s + std::size_t(t)]);
            }
        }

        // G4: the estimator changes since the last call (gpu_trace applies them to the GPU's copy)
        std::vector<gpu_op> TakeGPUOps() noexcept { return std::exchange(m_GPUOps, {}); }
        std::uint32_t EmitGen(int Id) const noexcept { return m_Instances[Id].m_EmitGen; }       // its emitted light's version (m_EmitB)
        const std::vector<basis>& EmittedLight(int Id) const noexcept { return m_Instances[Id].m_EmitB; }
        const std::vector<basis>& BasisFractions(int Id) const noexcept { return m_Instances[Id].m_Frac; }
        const estimator&          Estimator(int Id, int s) const noexcept { return m_Instances[Id].m_Est[std::size_t(s)]; }
        const std::vector<std::array<std::uint16_t, moment_texels_v>>& MomentCounts(int Id) const noexcept { return m_Instances[Id].m_MomCount; }

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
                float Cone = 1;
                if (Pt.m_CosOuter > -1.5f)          // a spot: its cone (xlionrender_local_lights.glsl)
                {
                    const float t = std::clamp((-D.Dot(Pt.m_Dir) / d - Pt.m_CosOuter) / std::max(Pt.m_CosInner - Pt.m_CosOuter, 1e-6f), 0.0f, 1.0f);
                    Cone = t * t * (3 - 2 * t);
                    if (Cone <= 0) return;
                }
                if (Occluded(O, D * (1.0f / d), d - 2e-3f)) return;
                E = E + Pt.m_Color * ((1 - s * s * s * s) / (1 + 2 * s * s) * c * Cone);
            });
            return E;
        }

        // Experiment D, step 1 (GI review: compiled transport), the CPU prototype. Each surfel's irradiance as the solver estimates it is
        // linear in the others' once the reads' weights are fixed (validity, moments: as they are now): E_s = D_s + sum_j T_sj E_j, with
        // D_s the sky, the direct light and the emission its rays bring and T_sj the albedo-weighted share of surfel j in the reads at its
        // rays' hits (the solver's own cosine rays, origins and reads). CompileTransport casts Rays a surfel once; SolveTransport iterates
        // E = D + T E with no ray at all. Level 0 of every instance; unsupported hits are counted and dropped (their share is reported)
        struct transport
        {
            std::vector<std::uint32_t>  m_InstFirst;     // per instance its first row (+1: the total)
            std::vector<std::uint32_t>  m_First;         // per row its first link (+1)
            std::vector<std::uint32_t>  m_Col;           // per link: the row it reads
            std::vector<v3>             m_W;             // per link: its weight (albedo x read share / rays)
            std::vector<v3>             m_D;             // per row: the part no surfel carries (sky, direct, emission)
            // Relightable (CompileTransport bRelight; the GPU's links always): no light in D; each row keeps its rays' hits (the GPU: 8
            // clusters of them) and its first bounce of the lights, F = sum albedo x Direct(hit), is made again from them (RelightTransport,
            // into m_Dir): E = D + F + T E. Exact at any light (the GPU: as close as its clusters)
            struct hit_pt { v3 m_P, m_N, m_A; };     // (world space; m_A: albedo / rays)
            std::vector<std::uint32_t>  m_HitFirst;  // per row its first hit (+1)
            std::vector<hit_pt>         m_Hits;
            std::vector<v3>             m_Dir;
            std::size_t                 m_nRays = 0, m_nUnsupported = 0, m_nDropped = 0, m_nBack = 0, m_nSky = 0, m_nFar = 0;
            // TruncateTransport: per row its aggregate links (the tail) to groups (a surfel's parent: its level-1 surfel); per row its group
            std::vector<std::uint32_t>  m_AFirst, m_AGroup, m_Group, m_GroupSize;
            std::vector<v3>             m_AW;     // (far: a hit on a level-1 instance, dropped)
            double                      m_CompileMs = 0;
        };
        transport CompileTransport(const lights& L, int Rays, unsigned Seed = 1, bool bRelight = false) noexcept
        {
            const auto T0 = std::chrono::steady_clock::now();
            BuildTlas();
            transport T;
            T.m_InstFirst.push_back(0);
            for (const auto& I : m_Instances) T.m_InstFirst.push_back(T.m_InstFirst.back() + std::uint32_t(I.m_bAlive ? I.m_pMesh->m_Levels[0].size() : 0));
            T.m_First.push_back(0);
            if (bRelight) T.m_HitFirst.push_back(0);
            std::vector<std::pair<int, float>> Cap;
            std::unordered_map<std::uint32_t, v3> Row;
            rng R(Seed);
            for (std::size_t i = 0; i < m_Instances.size(); ++i)
            {
                const auto& I = m_Instances[i];
                if (!I.m_bAlive) continue;
                const auto& Mo = *I.m_pMesh;
                for (std::size_t s = 0; s < Mo.m_Levels[0].size(); ++s)
                {
                    const auto& S = Mo.m_Levels[0][s];
                    const bool bSpread = Mo.m_OriginFirst.size() == Mo.m_Levels[0].size() + 1;
                    const std::uint32_t o0 = bSpread ? Mo.m_OriginFirst[s] : 0, nO = bSpread ? Mo.m_OriginFirst[s + 1] - o0 + 1 : 0;     // (+1: its center)
                    v3 D{ 0, 0, 0 };
                    Row.clear();
                    const float Off1 = R.Next(), Off2 = R.Next();
                    for (int r = 0; r < Rays; ++r)
                    {
                        v3 Cr = I.m_L2W * S.m_Center, Nr = NormalToWorld(I, S.m_Normal);
                        if (nO > 1) { v3 Po, No; Mo.Origin(o0, std::uint32_t(r) % nO, int(s), Po, No); Cr = I.m_L2W * Po; Nr = NormalToWorld(I, No); }
                        const v3 Dir = StratifiedCosineDir(Nr, r, Rays, Off1, Off2);
                        const hit H = Cast(Cr + Nr * 1e-3f, Dir);
                        ++T.m_nRays;
                        if (H.m_Inst < 0) { D = D + L.SkyAt(Dir) * (pi_v / float(Rays)); ++T.m_nSky; continue; }
                        const v3 Nh = HitNormal(H);
                        if (Nh.Dot(Dir) > 0) { ++T.m_nBack; continue; }
                        if (m_Instances[std::size_t(H.m_Inst)].m_Level != 0) { ++T.m_nFar; continue; }      // (the prototype's rows are level 0)
                        v3 Albedo, Emission;
                        SurfaceAt(H, Albedo, Emission);
                        const v3 Em = I.m_EmitB.empty() && !m_Setup.m_bEmitterSampling ? Emission : v3{ 0, 0, 0 };
                        D = D + Em * (pi_v / float(Rays));
                        if (bRelight) T.m_Hits.push_back({ HitPosition(H), Nh, Albedo * (1.0f / float(Rays)) });
                        else
                        {
                            const v3 Dl = Direct(HitPosition(H), Nh, L);
                            D = D + v3{ Albedo.m_X * Dl.m_X, Albedo.m_Y * Dl.m_Y, Albedo.m_Z * Dl.m_Z } * (1.0f / float(Rays));
                        }
                        Cap.clear();
                        m_pReadCapture = &Cap;
                        const query Q = Indirect(H);
                        m_pReadCapture = nullptr;
                        if (!Q.m_bSupported || Cap.empty())         // the solver's fallback (Unsupported), its weights; none: dropped (counted)
                        {
                            ++T.m_nUnsupported;
                            Cap.clear(); m_pReadCapture = &Cap; (void)Unsupported(H); m_pReadCapture = nullptr;
                            if (Cap.empty()) { ++T.m_nDropped; continue; }
                        }
                        const std::uint32_t Base = T.m_InstFirst[std::size_t(H.m_Inst)];
                        for (const auto& [j, w] : Cap) Row[Base + std::uint32_t(j)] = Row[Base + std::uint32_t(j)] + Albedo * (w / float(Rays));
                    }
                    // the emitters' light the pass adds besides its rays (FinishPass, SampleEmitters): the links', or the sampling's expectation
                    // (every emitter in sight, by its weight: what a lane's pick averages to)
                    if (m_Setup.m_bEmitterSampling)
                    {
                        const v3 C = I.m_L2W * S.m_Center, N = NormalToWorld(I, S.m_Normal);
                        for (std::size_t k = 0; k < m_Em.size(); ++k)
                        {
                            if (m_Emitters[k].first == int(i) && m_Emitters[k].second == int(s)) continue;
                            float w;
                            if (!EmitterWeight(k, C, N, w)) continue;
                            const v3 Dk = m_Em[k].m_C - C; const float d = Dk.Length();
                            if (Occluded(C + N * 1e-3f, Dk * (1.0f / d), d * 0.98f)) continue;
                            D = D + EmitterRadiance(k) * w;
                        }
                    }
                    else if (s < I.m_EmitB.size()) D = D + (I.m_EmitB[s][0] + I.m_EmitB[s][1] + I.m_EmitB[s][2]) * (1.0f / 3.0f);
                    T.m_D.push_back(D);
                    if (bRelight) T.m_HitFirst.push_back(std::uint32_t(T.m_Hits.size()));
                    for (const auto& [j, w] : Row) { T.m_Col.push_back(j); T.m_W.push_back(w); }
                    T.m_First.push_back(std::uint32_t(T.m_Col.size()));
                }
            }
            T.m_CompileMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();
            if (bRelight) RelightTransport(T, L);
            return T;
        }
        // A relightable transport's first bounce of the lights (m_Dir): each row's hits lit (the GPU's links mode 1, a shadow ray a light)
        void RelightTransport(transport& T, const lights& L) const noexcept
        {
            T.m_Dir.assign(T.m_D.size(), v3{ 0, 0, 0 });
            for (std::size_t r = 0; r < T.m_Dir.size() && r + 1 < T.m_HitFirst.size(); ++r)
                for (std::uint32_t h = T.m_HitFirst[r]; h < T.m_HitFirst[r + 1]; ++h)
                {
                    const auto& H = T.m_Hits[h];
                    const v3 Dl = Direct(H.m_P, H.m_N, L);
                    T.m_Dir[r] = T.m_Dir[r] + v3{ H.m_A.m_X * Dl.m_X, H.m_A.m_Y * Dl.m_Y, H.m_A.m_Z * Dl.m_Z };
                }
        }
        // Iterations of E = D + T E (Jacobi), from E (empty: from zero)
        static void SolveTransport(const transport& T, std::vector<v3>& E, int Iterations) noexcept
        {
            const std::size_t n = T.m_D.size();
            if (E.size() != n) E.assign(n, v3{ 0, 0, 0 });
            std::vector<v3> Next(n), Mean(T.m_GroupSize.size());
            const bool bTail = !T.m_AFirst.empty(), bDir = T.m_Dir.size() == n;
            for (int it = 0; it < Iterations; ++it)
            {
                const auto& Src = E;
                if (bTail)
                {
                    std::fill(Mean.begin(), Mean.end(), v3{ 0, 0, 0 });
                    for (std::size_t r = 0; r < n; ++r) Mean[T.m_Group[r]] = Mean[T.m_Group[r]] + Src[r];
                    for (std::size_t g = 0; g < Mean.size(); ++g) Mean[g] = Mean[g] * (1.0f / float(std::max<std::uint32_t>(T.m_GroupSize[g], 1)));
                }
                for (std::size_t r = 0; r < n; ++r)
                {
                    v3 Sum = bDir ? T.m_D[r] + T.m_Dir[r] : T.m_D[r];
                    for (std::uint32_t k = T.m_First[r]; k < T.m_First[r + 1]; ++k) { const v3& w = T.m_W[k]; const v3& e = Src[T.m_Col[k]]; Sum = Sum + v3{ w.m_X * e.m_X, w.m_Y * e.m_Y, w.m_Z * e.m_Z }; }
                    if (bTail) for (std::uint32_t k = T.m_AFirst[r]; k < T.m_AFirst[r + 1]; ++k) { const v3& w = T.m_AW[k]; const v3& m = Mean[T.m_AGroup[k]]; Sum = Sum + v3{ w.m_X * m.m_X, w.m_Y * m.m_Y, w.m_Z * m.m_Z }; }
                    Next[r] = Sum;
                }
                E.swap(Next);
            }
        }
        // Each row its K heaviest links (by luminance); the rest read the mean of their surfel's parent (level 1), one aggregate link per
        // parent: an explicit tail, not dropped and not renormalized away
        transport TruncateTransport(const transport& T, int K, bool bInstances = false) const noexcept     // (bInstances: the tail to whole instances)
        {
            transport O;
            O.m_InstFirst = T.m_InstFirst; O.m_D = T.m_D; O.m_Dir = T.m_Dir; O.m_HitFirst = T.m_HitFirst; O.m_Hits = T.m_Hits; O.m_nRays = T.m_nRays; O.m_nUnsupported = T.m_nUnsupported; O.m_nDropped = T.m_nDropped;
            O.m_nBack = T.m_nBack; O.m_nSky = T.m_nSky; O.m_nFar = T.m_nFar; O.m_CompileMs = T.m_CompileMs;
            const std::size_t n = T.m_D.size();
            O.m_Group.resize(n);
            for (std::size_t i = 0, Base = 0; i < m_Instances.size(); ++i)
            {
                const std::size_t r0 = T.m_InstFirst[i], r1 = T.m_InstFirst[i + 1];
                if (r1 == r0) continue;
                const auto& Pa = m_Instances[i].m_pMesh->m_ParentOf;
                std::uint32_t nP = 1;
                for (std::size_t r = r0; r < r1; ++r) { const std::uint32_t p = bInstances ? 0u : (r - r0) < Pa.size() ? std::uint32_t(std::max(Pa[r - r0], 0)) : 0u; O.m_Group[r] = std::uint32_t(Base) + p; nP = std::max(nP, p + 1); }
                Base += nP;
                O.m_GroupSize.resize(Base, 0);
            }
            for (std::size_t r = 0; r < n; ++r) ++O.m_GroupSize[O.m_Group[r]];
            const auto Lum = [](const v3& e) { return 0.2126f * e.m_X + 0.7152f * e.m_Y + 0.0722f * e.m_Z; };
            O.m_First.push_back(0); O.m_AFirst.push_back(0);
            std::vector<std::uint32_t> Ord; std::unordered_map<std::uint32_t, v3> Tail;
            for (std::size_t r = 0; r < n; ++r)
            {
                Ord.clear(); Tail.clear();
                for (std::uint32_t k = T.m_First[r]; k < T.m_First[r + 1]; ++k) Ord.push_back(k);
                if (int(Ord.size()) > K) std::nth_element(Ord.begin(), Ord.begin() + K, Ord.end(), [&](std::uint32_t a, std::uint32_t b) { return Lum(T.m_W[a]) > Lum(T.m_W[b]); });
                for (std::size_t q = 0; q < Ord.size(); ++q)
                    if (int(q) < K) { O.m_Col.push_back(T.m_Col[Ord[q]]); O.m_W.push_back(T.m_W[Ord[q]]); }
                    else Tail[O.m_Group[T.m_Col[Ord[q]]]] = Tail[O.m_Group[T.m_Col[Ord[q]]]] + T.m_W[Ord[q]];
                for (const auto& [g, w] : Tail) { O.m_AGroup.push_back(g); O.m_AW.push_back(w); }
                O.m_First.push_back(std::uint32_t(O.m_Col.size())); O.m_AFirst.push_back(std::uint32_t(O.m_AGroup.size()));
            }
            return O;
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
            if (M.m_bLeakProtection && !M.m_ReadTable.empty()) return ReadPoint(H.m_Inst, M.m_Pos[T[0]] * B[0] + M.m_Pos[T[1]] * B[1] + M.m_Pos[T[2]] * B[2]
                                                       , (M.m_Pos[T[1]] - M.m_Pos[T[0]]).Cross(M.m_Pos[T[2]] - M.m_Pos[T[0]]).NormalizeCopy());
            v3 Sum{ 0, 0, 0 }; float W = 0;
            const std::size_t Cap0 = m_pReadCapture ? m_pReadCapture->size() : 0;
            for (int k = 0; k < 3; ++k)
            {
                float w = 0;
                m_CaptureScale = B[k];
                const v3 E = ReadVertex(I, T[k], w);
                Sum = Sum + E * B[k];
                W += w * B[k];
            }
            if (W <= 0) return {};
            if (m_pReadCapture) for (std::size_t k = Cap0; k < m_pReadCapture->size(); ++k) (*m_pReadCapture)[k].second /= W;     // (normalized: the read is their mean)
            return { Sum * (1.0f / W), W, true };
        }
        // The surfels a point of an instance's surface (mesh space, its face's normal) reads: the read grid's patch for it and the weights of
        // the GPU's per-pixel read (shaders/xlionrender_surfel_gi.glsl SurfelGIList: falloff, normal, corner weights, validity, depth
        // moments) - the one support the solver's rays and the pixels share. The solver reads the unfiltered light (the pixels the filtered)
        query ReadPoint(int Id, const v3& P, const v3& N) const noexcept
        {
            // Every ray's hit: plain floats and raw pointers (a Debug build calls every v3 operator), the cheap terms first, the corner blend
            // once (it is the same for every entry: ReadEntryWeight's), the moments' test last and only for what is still in
            const auto& I = m_Instances[Id]; const auto& M = *I.m_pMesh;
            v3 PatchN = N;
            const auto List = M.ReadCellList(P, N, &PatchN);
            if (List.empty()) return {};
            const float Cell = M.m_ReadCell;
            const float Lp[3] = { (P.m_X + N.m_X * (surfel_mesh::read_lift_v * Cell)) / Cell, (P.m_Y + N.m_Y * (surfel_mesh::read_lift_v * Cell)) / Cell, (P.m_Z + N.m_Z * (surfel_mesh::read_lift_v * Cell)) / Cell };
            const auto [Au, Av, Aa] = surfel_mesh::ReadAxes(PatchN);
            const float fu = Lp[Au] - std::floor(Lp[Au]), fv = Lp[Av] - std::floor(Lp[Av]);
            const float k0 = (1 - fu) * (1 - fv) / 15.0f, k1 = fu * (1 - fv) / 15.0f, k2 = (1 - fu) * fv / 15.0f, k3 = fu * fv / 15.0f;
            const surfel* pL = M.m_Levels[0].data();
            const int*    pParent = M.m_ParentOf.data();
            const float*  pValid  = I.m_Valid.data();
            const v3*     pE      = I.m_E.data();
            const int     nLight  = int(I.m_E.size());
            const bool    bParent = I.m_Level == 1;
            const float   Px = P.m_X, Py = P.m_Y, Pz = P.m_Z, Nx = N.m_X, Ny = N.m_Y, Nz = N.m_Z;
            float Sx = 0, Sy = 0, Sz = 0, W = 0, Rx = 0, Ry = 0, Rz = 0, RW = 0;     // R: without the moments' test (the second tier)
            for (const std::uint32_t e : List)
            {
                const int s = int(e & 0xFFFFu);
                const surfel& S = pL[s];
                const float c = S.m_Normal.m_X * Nx + S.m_Normal.m_Y * Ny + S.m_Normal.m_Z * Nz;
                if (c < 0.5f) continue;
                const float dx = Px - S.m_Center.m_X, dy = Py - S.m_Center.m_Y, dz = Pz - S.m_Center.m_Z;
                const float f = 1.0f - std::sqrt(dx * dx + dy * dy + dz * dz) / (2 * S.m_Radius);
                if (f <= 0) continue;
                const int sl = bParent ? pParent[s] : s;
                if (sl >= nLight) continue;
                const float t  = std::min((c - 0.5f) * 4.0f, 1.0f), cn = t * t * (3 - 2 * t);     // smoothstep(0.5, 0.75, c)
                const float Corner = float((e >> 16) & 15u) * k0 + float((e >> 20) & 15u) * k1 + float((e >> 24) & 15u) * k2 + float((e >> 28) & 15u) * k3;
                const float w0 = f * f * c * cn * Corner * pValid[sl];
                if (w0 <= 0) continue;
                Rx += pE[sl].m_X * w0; Ry += pE[sl].m_Y * w0; Rz += pE[sl].m_Z * w0; RW += w0;
                if (m_pReadCapture) m_CapT2.push_back({ sl, w0 });
                const float w = w0 * Visibility(Id, sl, P);
                if (w <= 0) continue;
                Sx += pE[sl].m_X * w; Sy += pE[sl].m_Y * w; Sz += pE[sl].m_Z * w; W += w;
                if (m_pReadCapture) m_CapT1.push_back({ sl, w });
            }
            if (m_pReadCapture)     // (CompileTransport: the tier used, normalized)
            {
                const bool b1 = W > 1e-6f; const float Wt = b1 ? W : RW;
                if (Wt > 1e-6f) for (const auto& [j, w] : b1 ? m_CapT1 : m_CapT2) m_pReadCapture->push_back({ j, w / Wt });
                m_CapT1.clear(); m_CapT2.clear();
            }
            // The moments' test rejects every surfel (next to a wall's foot: their rays met the wall just past the point): its own surfels
            // without it (SurfelGIQuery's second tier) - the corner weights still keep the other side of a wall out
            if (W <= 1e-6f) { Sx = Rx; Sy = Ry; Sz = Rz; W = RW; }
            if (W <= 1e-6f) return {};
            const float r = 1.0f / W;
            return { v3{ Sx * r, Sy * r, Sz * r }, W, true };
        }

        // The policy for an unsupported query: the hit instance's mean light (its valid surfels', weighted by validity; MeanLight), not black and
        // not dropped - the ray still counts as one of the surfel's rays, the same denominator. Counted, the first places kept (UnsupportedAt)
        v3 Unsupported(const hit& H, int* pWalked = nullptr) const noexcept     // (pWalked, tests: the list entries walked)
        {
            // Local first (SurfelGILocalMean's twin), bounded: the patch of the hit's cell that matches its surface (ReadCellList: never the
            // whole cell, which mixes layers), its nearest fallback_cap_v surfels, by reach and validity, the current light; else the mean
            const auto& I = m_Instances[H.m_Inst]; const auto& M = *I.m_pMesh;
            if (M.m_ReadTable.empty()) return I.m_MeanE;
            const auto& T = M.m_Tris[H.m_Tri];
            const float B[3] = { 1 - H.m_U - H.m_V, H.m_U, H.m_V };
            const v3 P = M.m_Pos[T[0]] * B[0] + M.m_Pos[T[1]] * B[1] + M.m_Pos[T[2]] * B[2];
            const v3 N = (M.m_Pos[T[1]] - M.m_Pos[T[0]]).Cross(M.m_Pos[T[2]] - M.m_Pos[T[0]]).NormalizeCopy();
            const auto List = M.ReadCellList(P, N);
            float Sx = 0, Sy = 0, Sz = 0, W = 0;
            for (std::size_t i = 0; i < std::min<std::size_t>(List.size(), fallback_cap_v); ++i)
            {
                if (pWalked) ++*pWalked;
                const int s = int(List[i] & 0xFFFFu);
                const auto& S = M.m_Levels[0][s];
                const float f = std::max(0.0f, 1.0f - (P - S.m_Center).Length() / (2 * S.m_Radius));
                const int sl = I.m_Level == 1 ? M.m_ParentOf[s] : s;
                if (std::size_t(sl) >= I.m_E.size()) continue;
                const float w = f * I.m_Valid[sl];
                if (w <= 0) continue;
                Sx += I.m_E[sl].m_X * w; Sy += I.m_E[sl].m_Y * w; Sz += I.m_E[sl].m_Z * w; W += w;
                if (m_pReadCapture) m_pReadCapture->push_back({ sl, w });
            }
            if (m_pReadCapture && W > 1e-6f) for (auto& [j, w] : *m_pReadCapture) w /= W;     // (CompileTransport: normalized)
            if (W > 1e-6f) return v3{ Sx / W, Sy / W, Sz / W };
            return I.m_MeanE;
        }
        const v3& MeanLight(int Id) const noexcept { return m_Instances[Id].m_MeanE; }
        std::size_t InstanceSurfels0(int Id) const noexcept { return m_Instances[std::size_t(Id)].m_pMesh->m_Levels[0].size(); }     // (its level-0 surfels: the transport's rows)      // the same for the GPU's reads (the renderer uploads it)
        const std::vector<v3>& UnsupportedAt() const noexcept { return m_UnsupportedAt; }

        // The 6 parents an ambient cube traces: for each axis direction (+-x, +-y, +-z) the parent whose normal faces it best (within 60 degrees)
        static std::vector<int> CubeSetOf(const surfel_mesh& M) noexcept
        {
            std::vector<int> Set;
            const v3 Axes[6] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
            for (const v3& A : Axes)
            {
                int Best = -1; float BestD = 0.5f;
                for (int p = 0; p < int(M.m_Levels[1].size()); ++p) { const float d = M.m_Levels[1][p].m_Normal.Dot(A); if (d > BestD) { BestD = d; Best = p; } }
                if (Best >= 0 && std::find(Set.begin(), Set.end(), Best) == Set.end()) Set.push_back(Best);
            }
            return Set;
        }

        // The parents an ambient cube did not trace: the traced ones' light blended by the normals (the squared cosine, as the HL2 basis does),
        // and their directional fractions with it (not the validity: a parent buried in the solid stays buried, it does not shine). A cube's light at any point is what the 6 directions bring, by the way it faces
        void ExpandCube(int Id) noexcept
        {
            auto& I = m_Instances[Id];
            const auto& L = I.m_pMesh->m_Levels[1];
            std::vector<char> InSet(L.size(), 0);
            for (int k : I.m_CubeSet) InSet[std::size_t(k)] = 1;
            for (std::size_t p = 0; p < L.size(); ++p)
            {
                if (InSet[p]) continue;
                v3 E{ 0, 0, 0 }; float W = 0; basis F{};
                for (int k : I.m_CubeSet)
                {
                    const float c = std::max(L[p].m_Normal.Dot(L[std::size_t(k)].m_Normal), 0.0f), w = c * c * (I.m_Valid[std::size_t(k)] + 1e-3f);
                    if (w <= 0) continue;
                    E = E + I.m_E[std::size_t(k)] * w; W += w;
                    for (int i = 0; i < 3; ++i) F[i] = F[i] + I.m_Frac[std::size_t(k)][i] * w;
                }
                if (W <= 0)     // facing none of them: the nearest in direction
                {
                    int Best = I.m_CubeSet[0]; float BestD = -2;
                    for (int k : I.m_CubeSet) { const float d = L[p].m_Normal.Dot(L[std::size_t(k)].m_Normal); if (d > BestD) { BestD = d; Best = k; } }
                    I.m_E[p] = I.m_E[std::size_t(Best)]; I.m_Frac[p] = I.m_Frac[std::size_t(Best)];
                    continue;
                }
                const float r = 1.0f / W;
                I.m_E[p] = E * r;
                for (int i = 0; i < 3; ++i) I.m_Frac[p][i] = F[i] * r;
            }
        }

        // A pending surfel (Phase 1): its pass, traced in chunks across updates. The pass' random offsets are drawn when it is picked; its
        // sums live here until all its rays are in
        struct work_item
        {
            int     m_Inst = 0, m_Surf = 0, m_Level = 0;
            int     m_Rays = 0, m_RaysDone = 0;
            float   m_Off1 = 0, m_Off2 = 0;
            int     m_Shift = 0;
            bool    m_bGPU = false;                                    // G3: handed to the GPU (GPUWork)
            bool    m_bCommitted = false;                              // G4c: the GPU traced and committed it (DeliverGPURanges): Update counts it
            int     m_Count = 1, m_Flags = 0, m_History = 0, m_Traced = 0;     // G4c: a range of m_Count surfels from m_Surf (flags: gpu_range_*), its instance's
            unsigned m_Seed = 0;                                       //      history and the update's seed (the GPU's expand: the tiers, the offsets)
            v3      m_Sum{ 0, 0, 0 };
            int     m_nBack = 0, m_nBackAll = 0, m_nUnsupported = 0, m_nHits = 0;
            v3      m_UnsupportedAt{ 0, 0, 0 };
            float   m_UnitUs = 0;   int m_UnitRays = 0;           // this update's unit (instrumentation, the governor)
            float   m_CastUs = 0, m_DirectUs = 0, m_ReadUs = 0;
        };

        // The policy's value per instance (Unsupported): its mean light of its valid surfels now; none valid: the world's
        void UpdateMeans() noexcept
        {
            v3 WSum{ 0, 0, 0 }; float WW = 0;
            for (auto& I : m_Instances)
            {
                if (!I.m_bAlive) { I.m_MeanS = v3{ 0, 0, 0 }; I.m_MeanW = 0; I.m_nBad = 0; I.m_bMeanDirty = false; }
                else if (I.m_bMeanDirty && !m_Setup.m_bGPUOwned)     // (Phase 3: only an instance whose light changed; G4c: the GPU's, SetGPUMeans)
                {
                    v3 S{ 0, 0, 0 }; float W = 0; int nBad = 0;
                    for (std::size_t s = 0; s < I.m_E.size(); ++s) { S = S + I.m_E[s] * I.m_Valid[s]; W += I.m_Valid[s]; nBad += BadSurfel(I.m_E[s], I.m_Valid[s]); }
                    I.m_MeanS = S; I.m_MeanW = W; I.m_nBad = nBad; I.m_nMeanDeltas = 0; I.m_bMeanDirty = false;
                }
                I.m_MeanE = I.m_MeanW > 0 ? I.m_MeanS * (1.0f / I.m_MeanW) : v3{ -1, -1, -1 };
                WSum = WSum + I.m_MeanS; WW += I.m_MeanW;
            }
            const v3 World = WW > 0 ? WSum * (1.0f / WW) : v3{ 0, 0, 0 };
            for (auto& I : m_Instances) if (I.m_MeanE.m_X < 0) I.m_MeanE = World;
            m_MeanLum = 0.2126f * World.m_X + 0.7152f * World.m_Y + 0.0722f * World.m_Z;
        }

        // G4c: an instance's share of the pick as ranges: its urgent surfels (one each, the full rays), then the pass from its cursor, as many
        // steps as the budget's estimate allows (what a step cost, else the full rays), split around the urgent ones (a cube: its set's
        // surfels one each). The GPU's expand applies the tiers and the adaptive rays to each surfel
        void PickRanges(int i, int Cap, bool bSettled, unsigned Seed, int& Took, int& TookRays, stats& St) noexcept
        {
            auto& I = m_Instances[std::size_t(i)];
            const int R = std::max(m_Setup.m_RaysPerSurfel, 1);
            const auto Range = [&](int First, int Count, int Flags, float Rays)
            {
                work_item W;
                W.m_Inst = i; W.m_Surf = First; W.m_Count = Count; W.m_Level = I.m_Level; W.m_Flags = Flags | (bSettled ? gpu_range_settled_v : 0);
                W.m_History = I.m_History; W.m_Seed = Seed; W.m_Rays = std::max(1, int(std::ceil(Rays)));
                m_Pending.push_back(W);
            };
            std::vector<int> Urgent;
            while (!I.m_Urgent.empty() && TookRays < Cap)
            {
                const int Surf = I.m_Urgent.back(); I.m_Urgent.pop_back();
                if (std::size_t(Surf) >= I.m_E.size() || I.m_UrgentMark[Surf] == 2) continue;
                I.m_UrgentMark[Surf] = 2; Urgent.push_back(Surf);
                Range(Surf, 1, gpu_range_urgent_v, float(R)); ++Took; TookRays += R; ++St.m_UrgentSurfels;
            }
            const int n = I.m_bCube ? int(I.m_CubeSet.size()) : I.m_Level == 0 ? int(std::min<std::size_t>(I.m_nOwned, I.m_E.size())) : int(I.m_E.size());     // (GI chunks: no ghost)
            const float Step = I.m_RaysPerStep > 0 ? I.m_RaysPerStep : float(R);
            const int Steps = TookRays >= Cap ? 0 : std::min(n - I.m_Cursor, std::max(1, int(std::ceil(float(Cap - TookRays) / Step))));
            const int c0 = I.m_Cursor, c1 = c0 + Steps;
            if (I.m_bCube) { for (int c = c0; c < c1; ++c) if (I.m_UrgentMark[I.m_CubeSet[std::size_t(c)]] != 2) Range(I.m_CubeSet[std::size_t(c)], 1, 0, Step); }
            else
            {
                std::sort(Urgent.begin(), Urgent.end());
                int a = c0;
                for (const int u : Urgent) if (u >= a && u < c1) { if (u > a) Range(a, u - a, 0, Step * float(u - a)); a = u + 1; }
                if (a < c1) Range(a, c1 - a, 0, Step * float(c1 - a));
            }
            I.m_Cursor = c1; Took += Steps; TookRays += int(std::ceil(Step * float(Steps)));
            for (const int u : Urgent) I.m_UrgentMark[u] = 0;
        }

        // A pass' random offsets (the rotation of its stratified set, the pairing of its origins and strata) from the update's seed and the
        // surfel: a hash, the same on the GPU (shaders/xlionrender_gi_expand_comp.glsl), so its pick and the CPU's draw the same rays
        static std::uint32_t Hash32(std::uint32_t x) noexcept { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
        static void PassOffsets(unsigned Seed, int Inst, int Surf, float& Off1, float& Off2, int& Shift) noexcept
        {
            const std::uint32_t h1 = Hash32(Seed * 0x9E3779B9u ^ std::uint32_t(Inst) * 0x85EBCA6Bu ^ std::uint32_t(Surf) * 0xC2B2AE35u), h2 = Hash32(h1), h3 = Hash32(h2);
            Off1 = float(h1 >> 8) * (1.0f / 16777216.0f); Off2 = float(h2 >> 8) * (1.0f / 16777216.0f); Shift = int(h3 >> 26);
        }

        // G5: an emitter's weight for a receiver at C facing N (the links' small disk: A cos cos / (d^2 + A); false: it faces away, or too
        // little) and its radiance now
        bool EmitterWeight(std::size_t k, const v3& C, const v3& N, float& w) const noexcept
        {
            const v3 D = m_Em[k].m_C - C;
            const float d2 = D.Dot(D);
            if (d2 < 1e-10f) return false;
            const v3 Dn = D * (1.0f / std::sqrt(d2));
            const float cr = N.Dot(Dn), ce = -m_Em[k].m_N.Dot(Dn);
            if (cr <= 0 || ce <= 0) return false;
            w = m_Em[k].m_A * cr * ce / (d2 + m_Em[k].m_A);
            return w >= 1e-5f;
        }
        v3 EmitterRadiance(std::size_t k) const noexcept
        {
            const auto& [i, e] = m_Emitters[k];
            const auto& SE = m_Instances[std::size_t(i)].m_SurfEmission;
            return std::size_t(e) < SE.size() ? SE[std::size_t(e)] * m_Setup.m_EmissiveScale : v3{ 0, 0, 0 };
        }
        static float Lum(const v3& c) noexcept { return 0.2126f * c.m_X + 0.7152f * c.m_Y + 0.0722f * c.m_Z; }

        // G5: the emitters' light of a pass by light sampling (the GPU's pass draws the same: shaders/xlionrender_gi_surfel_pass_comp.glsl).
        // gpu_lanes_v samples: each an emitter picked by weighted reservoir sampling (weight w lum(L), its random numbers hashes of the
        // pass' offsets, the sample, the emitter), its shadow ray, L Q / lum(L) when it is seen (Q: the weights' sum). Their mean is what
        // the links' sum was (FinishPass: the basis split by the direction to the emitter), added to the pass' sums
        void SampleEmitters(work_item& W) noexcept
        {
            auto& I = m_Instances[std::size_t(W.m_Inst)];
            const int s = W.m_Surf;
            const auto& S = I.m_pMesh->m_Levels[I.m_Level][std::size_t(s)];
            const v3 C = I.m_L2W * S.m_Center, N = NormalToWorld(I, S.m_Normal);
            const basis Dirs = BasisDirs(S.m_Normal);
            const std::uint32_t Base = Hash32(std::bit_cast<std::uint32_t>(W.m_Off1) ^ Hash32(std::bit_cast<std::uint32_t>(W.m_Off2)));
            basis E{};
            for (int l = 0; l < gpu_lanes_v; ++l)
            {
                const std::uint32_t Seed = Hash32(Base ^ std::uint32_t(l + 1) * 0x9E3779B9u);
                float Q = 0; int Pick = -1;
                for (std::size_t k = 0; k < m_Em.size(); ++k)
                {
                    if (I.m_Level == 0 && m_Emitters[k].first == W.m_Inst && m_Emitters[k].second == s) continue;
                    float w;
                    if (!EmitterWeight(k, C, N, w)) continue;
                    const float q = w * Lum(EmitterRadiance(k));
                    if (q <= 0) continue;
                    Q += q;
                    const float u = float(Hash32(Seed ^ std::uint32_t(k) * 0x85EBCA6Bu) >> 8) * (1.0f / 16777216.0f);
                    if (u * Q < q) Pick = int(k);
                }
                if (Pick < 0) continue;
                const v3 D = m_Em[std::size_t(Pick)].m_C - C;
                const float d = D.Length();
                if (Occluded(C + N * 1e-3f, D * (1.0f / d), d * 0.98f)) continue;      // stops short of the emitter's own surface (as the links)
                const v3 Le = EmitterRadiance(std::size_t(Pick)), c = Le * (Q / Lum(Le));
                v3 Dm = I.m_W2L.TransformDirection(D * (1.0f / d));
                Dm = Dm * (1.0f / std::max(Dm.Length(), 1e-12f));
                const auto Bw = BasisWeights(Dirs, Dm);
                for (int i = 0; i < 3; ++i) E[i] = E[i] + c * (3.0f * Bw[i] / float(gpu_lanes_v));
            }
            const float f = float(W.m_Rays) / (3.0f * pi_v);       // (as FinishPass adds the links': the pass' sums are of R rays)
            auto& BA = I.m_BasisAcc[std::size_t(s)];
            for (int i = 0; i < 3; ++i) { const v3 e = E[i] * f; W.m_Sum = W.m_Sum + e; BA[i] = BA[i] + e; }
        }

        // A pass whose rays are all in (TraceChunk's last chunk, DeliverGPU): the emitted light from its links, the irradiance E = pi mean(L),
        // its validity
        void FinishPass(const work_item& W) noexcept
        {
            auto& I = m_Instances[W.m_Inst];
            const int s = W.m_Surf, R = W.m_Rays;
            auto& BA = I.m_BasisAcc[s];
            v3 Sum = W.m_Sum;
            if (s < I.m_EmitB.size())                                   // the emitters it sees (the links): deterministic, no rays
                for (int i = 0; i < 3; ++i) { const v3 e = I.m_EmitB[s][i] * (float(R) / (3.0f * pi_v)); Sum = Sum + e; BA[i] = BA[i] + e; }     // the basis' mean is the irradiance
            I.m_Acc[s]      = Sum * (pi_v / float(R));                  // E = pi mean(L), cosine sampling
            // Buried -> validity 0, by either sign: back faces near it (within its reach: inside a wall or a frame they are centimeters
            // away; full validity up to 20% of its rays, none from 50%), or back faces all around at any distance (inside a solid object;
            // full up to 80%, none from 95%). A floor under a ceiling made to face out sees back faces metres away, in some directions
            // only: it keeps its light (it was: any back face, 10%..25% - 2 rays of 8 - and 28% of the Police Station read nothing)
            const float Near = std::clamp(1.0f - (float(W.m_nBack) / float(R) - 0.2f) / 0.3f, 0.0f, 1.0f);
            const float All  = std::clamp(1.0f - (float(W.m_nBackAll) / float(R) - 0.8f) / 0.15f, 0.0f, 1.0f);
            I.m_AccValid[s] = std::min(Near, All);
        }

        // Rays [r0, r1) of a pending surfel's pass (the first chunk clears its accumulators, the last one finishes the pass: the emitted light
        // from its links, the irradiance E = pi mean(L), its validity). The rays are the same whatever the chunks: ray r of R of the pass
        void TraceChunk(work_item& W, int r0, int r1, const lights& L) noexcept
        {
            using clock = std::chrono::steady_clock;
            const auto Us = [](clock::time_point A, clock::time_point B) { return std::chrono::duration<float, std::micro>(B - A).count(); };
            const auto Tw = clock::now();
            const int s = W.m_Surf;
            auto& I = m_Instances[W.m_Inst];
            const auto& S = I.m_pMesh->m_Levels[I.m_Level][s];
            const v3 C = I.m_L2W * S.m_Center, N = NormalToWorld(I, S.m_Normal);
            const int R = W.m_Rays;
            auto& MA = I.m_MomAcc[s];
            auto& BA = I.m_BasisAcc[s];
            if (r0 == 0) { MA.fill(0); BA.fill(v3{ 0, 0, 0 }); W.m_Sum = v3{ 0, 0, 0 }; W.m_nBack = W.m_nBackAll = W.m_nUnsupported = W.m_nHits = 0; }
            const basis Dirs = BasisDirs(S.m_Normal);
            std::array<float, 3> Bw{};
            const float Reach = 2 * S.m_Radius;                         // mesh units
            const float Off1 = W.m_Off1, Off2 = W.m_Off2;               // this pass' rotation of the stratified set
            const auto& Mo = *I.m_pMesh;
            const bool  bSpread = m_Setup.m_bSpreadOrigins && I.m_Level == 0 && Mo.m_OriginFirst.size() == Mo.m_Levels[0].size() + 1;
            const std::uint32_t o0 = bSpread ? Mo.m_OriginFirst[s] : 0, nO = bSpread ? Mo.m_OriginFirst[s + 1] - o0 + 1 : 0;     // (+1: its center)
            const int   Shift = W.m_Shift;                              // the pass' pairing of origins and direction strata
            v3 Sum = W.m_Sum;
            int nBack = W.m_nBack, nBackAll = W.m_nBackAll, nUnsupported = W.m_nUnsupported, nHits = W.m_nHits;
            for (int r = r0; r < r1; ++r)
            {
                v3 Cr = C, Nr = N;                                      // the ray's origin: across the surfel's patch (BuildOrigins)
                if (nO > 1) { v3 Po, No; Mo.Origin(o0, std::uint32_t(r + Shift) % nO, s, Po, No); Cr = I.m_L2W * Po; Nr = NormalToWorld(I, No); }
                const v3 D = StratifiedCosineDir(Nr, r, R, Off1, Off2);
                const auto Tc = clock::now();
                const hit H = Cast(Cr + Nr * 1e-3f, D);
                W.m_CastUs += Us(Tc, clock::now());
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
                if (H.m_Inst < 0) { const v3 Sky = L.SkyAt(D); Sum = Sum + Sky; for (int i = 0; i < 3; ++i) BA[i] = BA[i] + Sky * Bw[i]; continue; }
                const v3 Nh = HitNormal(H);
                // A back face: no light. Within the surfel's reach it says the surfel may be buried (inside a wall or a frame, back faces are
                // centimeters away); farther it is single-sided geometry seen from behind (a ceiling made to face out, metres up): not buried
                if (Nh.Dot(D) > 0) { nBack += H.m_T < Reach * I.m_MeanScale; ++nBackAll; continue; }
                v3 Albedo, Emission;
                SurfaceAt(H, Albedo, Emission);
                const auto Tr = clock::now();
                const query Q  = Indirect(H);
                W.m_ReadUs += Us(Tr, clock::now());
                if (!Q.m_bSupported) { if (nUnsupported++ == 0) W.m_UnsupportedAt = HitPosition(H); }
                ++nHits;
                const auto Td = clock::now();
                const v3 Dl = Direct(HitPosition(H), Nh, L);
                W.m_DirectUs += Us(Td, clock::now());
                const auto Tu = clock::now();
                const v3 In = Dl + (Q.m_bSupported ? Q.m_E : Unsupported(H));
                W.m_ReadUs += Us(Tu, clock::now());
                const v3 Lr = v3{ Albedo.m_X * In.m_X, Albedo.m_Y * In.m_Y, Albedo.m_Z * In.m_Z } * (1.0f / pi_v) + (I.m_EmitB.empty() && !m_Setup.m_bEmitterSampling ? Emission : v3{ 0, 0, 0 });     // with its links (or the sampling) they bring the emitters' light; without (none, or not built yet: a minute for Police) the hits do
                Sum = Sum + Lr;
                for (int i = 0; i < 3; ++i) BA[i] = BA[i] + Lr * Bw[i];
            }
            W.m_Sum = Sum; W.m_nBack = nBack; W.m_nBackAll = nBackAll; W.m_nUnsupported = nUnsupported; W.m_nHits = nHits;
            W.m_RaysDone = r1;
            if (r1 >= R && m_Setup.m_bEmitterSampling) SampleEmitters(W);
            if (r1 >= R) FinishPass(W);
            const float u = Us(Tw, clock::now());
            W.m_UnitUs = std::max(W.m_UnitUs, u);
            W.m_UnitRays = r1 - r0;
        }

        // One update: the light changes, the levels, then the ray budget over the instances that need it most
        stats Update(const lights& L, const v3& Eye, int RayBudget, unsigned Seed) noexcept
        {
            const auto T0 = std::chrono::steady_clock::now();
            stats St;
            auto TPrev = T0;
            const auto Lap = [&](double& Out) { const auto T = std::chrono::steady_clock::now(); Out += std::chrono::duration<double, std::milli>(T - TPrev).count(); TPrev = T; };
            St.m_nMoves = m_nMovesAcc; St.m_MovesMs = m_MovesMsAcc; St.m_nMarkedDirty = m_MoveMarked;
            // Phase 6b: the one allowance of the update. What this frame already spent on the GI before it (the events since the last
            // update: moves, their marks; the renderer's own work: m_OverheadMs) and everything the update has done so far are taken
            // from it; each stage gets what is left (Left), the trace leaving the shown light its reserve while that has a backlog. Only
            // the stage that waited longest may start one unit past it (the token): the update stays within the allowance plus one unit
            const double AllowMs = m_Setup.m_TraceAllowanceMs;
            const double PreMs   = m_MovesMsAcc + std::max(double(m_Setup.m_OverheadMs), 0.0);
            const auto   Left    = [&] { return AllowMs - PreMs - std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count(); };
            const auto   Grant   = [](double Ms) { return std::max(Ms, 1e-6); };      // (0 would mean "no limit" to a stage)
            int Token = -1;                                                         // 0 links, 1 trace, 2 shown, 3 events
            if (AllowMs > 0) for (int k = 0, Best = 0; k < 4; ++k) if (m_Starve[k] > Best) { Best = m_Starve[k]; Token = k; }
            St.m_PreMs = PreMs;
            {   // the events' work that waits for buckets (a change near an instance whose buckets are not made yet)
                const bool bWork = m_bPendingAffects;
                bool bTok = Token == 3;
                const bool bDid = PumpAffects(AllowMs > 0 ? Grant(AllowMs - PreMs) : 0.0, T0, &bTok);
                if (AllowMs > 0) m_Starve[3] = bWork && !bDid ? m_Starve[3] + 1 : 0;
            }
            m_nMovesAcc = 0; m_MovesMsAcc = 0; m_MoveMarked = 0;
            if (!m_Setup.m_bGPUOwned) for (auto& I : m_Instances) { if (I.m_bAlive) EnsureSolver(I); }     // (the CPU solves again: AllocSolver)
            else if (!m_bSolverFreed) for (auto& I : m_Instances) FreeSolver(I);          // (GPU-owned since: the instances added before it)
            m_bSolverFreed = m_Setup.m_bGPUOwned;            BuildTlas();
            TrackLights(L);
            Lap(St.m_TlasLightsMs);

            // Levels (Phase 2b: one rule): the finest level whose surfels, at the instance's point nearest the eye, each cover at least
            // m_PixelsPerSurfel pixels (a disk of the level's radius: pi (r / d)^2 ProjScale^2); the ambient cube (6 of the parents traced)
            // when not even the parents do. Coming back finer needs 1.25 times that (no flicker at the edge). Far, small and large instances
            // alike: the eye inside a building is at distance 0 from it, its surfels cover the view. Finer: children start from their
            // parent; coarser: a parent is the mean of its children
            for (auto& I : m_Instances)
            {
                if (!I.m_bAlive) continue;
                const auto& Mh   = *I.m_pMesh;
                const float Dist = BoxDistance(I, Eye);
                const auto  Px   = [&](int Lv) { const float r = Mh.m_Levels[Lv][0].m_Radius * I.m_MeanScale, d = std::max(Dist, r); return pi_v * (r / d) * (r / d) * m_Setup.m_ProjScale * m_Setup.m_ProjScale; };
                int  Want = 0;
                bool bCubeWant = false;
                if (m_Setup.m_ProjScale > 0 && !Mh.m_Levels[0].empty() && !Mh.m_Levels[1].empty())
                {
                    if (Px(0) < m_Setup.m_PixelsPerSurfel * (I.m_Level == 1 ? 1.25f : 1.0f)) Want = 1;
                    bCubeWant = Want == 1 && m_Setup.m_bCube && Mh.m_Levels[1].size() > 12 && Px(1) < m_Setup.m_PixelsPerSurfel * (I.m_bCube ? 1.25f : 1.0f);
                }
                const auto& M = *I.m_pMesh;
                if (Want != I.m_Level)
                {
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
                I.m_E = std::move(E); I.m_Valid = std::move(Valid); I.m_Level = Want; I.m_Cursor = 0; ++m_GPULevelGen;     // (the GPU re-levels in place: gpu_trace::Relevel)
                I.m_bEmitRelevel = true;
                ResetMoments(I);                                // (the solver's arrays: AllocSolver, CPU-solved only)
                }
                // An ambient cube: its parents cover too few pixels: 6 of them traced, the rest blended (the instance is still at level 1 for everything that reads it)
                bCubeWant = bCubeWant && I.m_Level == 1;
                std::vector<int> Set;
                if (bCubeWant && !I.m_bCube) Set = CubeSetOf(M);
                if (bCubeWant && !I.m_bCube && !Set.empty()) { I.m_bCube = true; I.m_CubeSet = std::move(Set); I.m_Cursor = 0; I.m_bEmitDirty = true; }
                else if (!bCubeWant && I.m_bCube) { I.m_bCube = false; I.m_CubeSet.clear(); I.m_Cursor = 0; I.m_bDirty = true; I.m_bEmitDirty = true; }       // the parents trace again: from the start
            }

            Lap(St.m_LevelsMs);
            // Pick: by Priority (dirty first, then a pass in progress, then age over distance), until the budget is spent. An instance is updated in
            // slices (a cursor over its surfels): a pass over a big mesh spans several updates and the budget holds whatever one mesh
            // has. A pass starts with the blend of its own (1 / n after a change); a change in the middle of a pass starts it again.
            // Items picked by an earlier update and not finished keep going; one whose instance was removed or changed level (its arrays were
            // made again) is dropped. A pick is made only when nothing is pending: one surfel is never in flight twice
            std::erase_if(m_Pending, [&](const work_item& W) { const auto& I = m_Instances[W.m_Inst]; return !I.m_bAlive || I.m_Level != W.m_Level || std::size_t(W.m_Surf) >= I.m_E.size(); });
            m_Work.clear();
            // (G3: the items the GPU delivered are finished, not yet committed: the update that commits them picks the next ones)
            if (std::none_of(m_Pending.begin(), m_Pending.end(), [](const work_item& W) { return W.m_RaysDone < W.m_Rays; }))
            {
                m_Order.clear();
                for (int i = 0; i < int(m_Instances.size()); ++i)
                {
                    auto& I = m_Instances[i];
                    if (!I.m_bAlive || I.m_E.empty()) continue;
                    St.m_Dirty += I.m_bDirty;
                    if (I.m_bDirty && I.m_Cursor != 0) I.m_Cursor = 0;
                    m_Order.push_back({ Priority(I, Eye), i });
                }
                std::sort(m_Order.begin(), m_Order.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                m_Work.clear();
                int RaysLeft = std::max(RayBudget, 1);          // (Phase 6: in rays, each surfel what it asks for: RequestedRays)
                // Fairness (Phase 1): the instance that has waited longest gets m_FairShare of the update first, whatever its score; the rest
                // goes by score. An instance is visited once per pick (two visits would trace a surfel twice in one pass)
                int Fair = -1, FairAge = -1;
                if (m_Order.size() > 1 && m_Setup.m_FairShare > 0)
                    for (const auto& [Score, i] : m_Order) if (m_Instances[i].m_Age > FairAge) { FairAge = m_Instances[i].m_Age; Fair = i; }
                if (Fair >= 0 && Fair != m_Order.front().second)
                {
                    auto It = std::find_if(m_Order.begin(), m_Order.end(), [&](const auto& o) { return o.second == Fair; });
                    std::rotate(m_Order.begin(), It, It + 1);
                }
                else Fair = -1;
                for (const auto& [Score, i] : m_Order)
                {
                    if (RaysLeft <= 0) break;
                    auto& I = m_Instances[i];
                    const int Cap = i == Fair ? std::max(1, int(float(RaysLeft) * m_Setup.m_FairShare)) : RaysLeft;
                    const int n    = I.m_bCube ? int(I.m_CubeSet.size()) : I.m_Level == 0 ? int(std::min<std::size_t>(I.m_nOwned, I.m_E.size())) : int(I.m_E.size());     // (GI chunks: its ghosts are never traced)
                    if (I.m_Cursor == 0)
                    {
                        if (I.m_bDirty) { I.m_History = 0; I.m_bDirty = false; ++St.m_nHistoryResets; for (auto& Es : I.m_Est) Es.Lower(I.m_DirtyA); if (m_Setup.m_bGPUTrace) m_GPUOps.push_back({ i, -1, I.m_DirtyA, 0 }); I.m_DirtyA = 1; }
                        ++I.m_History;
                    }
                    const bool bSettled = I.m_History > warmup_passes_v;
                    int Took = 0, TookRays = 0;
                    if (m_Setup.m_bGPUOwned)        // G4c: ranges, sized by what a step costs; the GPU's expand serves them surfel by surfel
                    {
                        PickRanges(i, Cap, bSettled, Seed, Took, TookRays, St);
                        if (I.m_Cursor >= n) { I.m_Cursor = 0; I.m_Age = 0; }
                        RaysLeft -= TookRays;
                        if (i == Fair) St.m_FairSurfels = Took;
                        ++St.m_Updated;
                        I.m_bPicked = true;
                        continue;
                    }
                    // Phase 4: the surfels a change affected first (its urgent list), then the pass (which skips them this pick)
                    const std::size_t UrgentAt = m_Work.size();
                    while (!I.m_Urgent.empty() && TookRays < Cap)
                    {
                        const int Surf = I.m_Urgent.back(); I.m_Urgent.pop_back();
                        if (std::size_t(Surf) >= I.m_E.size()) continue;
                        I.m_UrgentMark[Surf] = 2;
                        m_Work.push_back({ i, Surf }); ++Took; TookRays += RequestedRays(I, Surf); ++St.m_UrgentSurfels;
                    }
                    const std::size_t UrgentEnd = m_Work.size();
                    for (; I.m_Cursor < n && TookRays < Cap; ++I.m_Cursor)
                    {
                        const int Surf = I.m_bCube ? I.m_CubeSet[I.m_Cursor] : I.m_Cursor;
                        if (I.m_UrgentMark[Surf] == 2) continue;
                        if (bSettled)
                        {
                            const int Period = TracePeriod(I, Surf, Eye);
                            if (Period > 1 && ((I.m_History + Surf) % Period) != 0) continue;     // staggered: a period's passes together trace every one
                        }
                        m_Work.push_back({ i, Surf });
                        ++Took; TookRays += RequestedRays(I, Surf);
                    }
                    for (std::size_t w = UrgentAt; w < UrgentEnd; ++w) I.m_UrgentMark[m_Work[w].second] = 0;
                    if (I.m_Cursor >= n) { I.m_Cursor = 0; I.m_Age = 0; }
                    RaysLeft -= TookRays;
                    if (i == Fair) St.m_FairSurfels = Took;
                    ++St.m_Updated;
                    I.m_bPicked = true;
                }
                for (std::size_t w = 0; w < m_Work.size(); ++w)
                {
                    const auto [pi, ps] = m_Work[w];
                    work_item W;
                    W.m_Inst = pi; W.m_Surf = ps; W.m_Level = m_Instances[pi].m_Level; W.m_Rays = RequestedRays(m_Instances[pi], ps);
                    PassOffsets(Seed, pi, ps, W.m_Off1, W.m_Off2, W.m_Shift);     // the pass' random offsets (the GPU's pick draws the same)
                    if (auto& PI = m_Instances[pi]; std::size_t(ps) < PI.m_LastOff.size()) PI.m_LastOff[ps] = { W.m_Off1, W.m_Off2 };
                    m_Pending.push_back(W);
                }
                m_Work.clear();      // from here m_Work lists the surfels finished this update (LastUpdated)
            }
            Lap(St.m_PickMs);

            ++m_nUpdates;
            if (m_bRelightDeferred && m_nUpdates >= m_RelightAt) RelightAll();
            bool bShowBacklog = false;
            for (const auto& I : m_Instances) if (I.m_bAlive && (!I.m_FilterDirty.empty() || I.m_FullPass >= 0 || I.m_bFilterFull)) { bShowBacklog = true; break; }
            const double ShowReserve = AllowMs > 0 && bShowBacklog ? AllowMs * double(m_Setup.m_ShowShare) : 0.0;
            const bool   bLinkWork   = m_LinkCur.m_Inst >= 0 || !m_LinkQueue.empty();
            const int    LinkSurfels0 = St.m_LinkSurfels;
            const double LinkUsedMs = PumpLinks(AllowMs > 0 ? Grant((Left() - ShowReserve) * (m_Pending.empty() || m_Setup.m_bGPUTrace ? 1.0 : double(m_Setup.m_LinkShare))) : 0.0, St, Token == 0);   // the trace waits for nothing (or the GPU traces): all of it
            (void)LinkUsedMs;
            if (AllowMs > 0) m_Starve[0] = bLinkWork && St.m_LinkSurfels == LinkSurfels0 ? m_Starve[0] + 1 : 0;
            Lap(St.m_LinksMs);
            // The emitters' light: made of the links, the emissions (SetSurfelMaterial: the links again) and the scale; a level switch
            // changes its layout. Nothing else: not every update
            if (m_EmitScale != m_Setup.m_EmissiveScale) { m_bEmitDirty = true; m_EmitScale = m_Setup.m_EmissiveScale; }
            UpdateEmittedLight(m_bEmitDirty); m_bEmitDirty = false;
            Lap(St.m_EmitMs);

            UpdateMeans();
            Lap(St.m_MeanMs);

            // The trace (Phase 1): the pending surfels' rays in chunks (units of about m_UnitTargetUs), round after round, until every one is
            // finished or the allowance is spent. A unit is not started when the conservative estimate says it would not fit; one that is
            // started finishes: the overshoot is one unit. Without an allowance a chunk is all of a surfel's rays and one round traces them all
            // (the solver as it was: the same rays, the same seeds). A surfel is committed when all its rays are in; until then it is pending
            const auto T1      = std::chrono::steady_clock::now();
            const double Allow = AllowMs > 0 ? Grant(Left() - ShowReserve) : 0.0;     // what is left of the one allowance, the shown light's reserve kept
            bool bTraceToken = Token == 1;
            const auto Deadline = T1 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double, std::milli>(Allow));
            // conservative: the slowest ray cost seen lately (it decays), at least the mean plus twice the mean deviation. The cost of a ray
            // has a long tail (a ray into dense geometry): sized by the mean, units of 8 rays reached 1.1 ms in the Police level
            const float RayUs  = std::max(m_RayUsHi, m_RayUsMean + 2.0f * m_RayUsDev);
            const int   Chunk  = Allow > 0 ? std::clamp(int(m_Setup.m_UnitTargetUs / std::max(RayUs, 1.0f)), 1, std::max(m_Setup.m_RaysPerSurfel, 1)) : std::max(m_Setup.m_RaysPerSurfel, 1);
            St.m_ChunkRays = Chunk; St.m_RayUsEst = RayUs;
            std::vector<int> Live;
            std::atomic<int> RaysDone{ 0 };
            if (!m_Setup.m_bGPUTrace) for (;;)                         // (G3: the GPU traces them: GPUWork, DeliverGPU)
            {
                Live.clear();
                for (int k = 0; k < int(m_Pending.size()); ++k) if (m_Pending[k].m_RaysDone < m_Pending[k].m_Rays) Live.push_back(k);
                if (Live.empty()) break;
                if (Allow > 0 && std::chrono::steady_clock::now() >= Deadline) break;
                const int Before = RaysDone.load();
                ++St.m_Rounds;
                const bool bForce = bTraceToken && St.m_Rays == 0 && RaysDone.load() == 0;   // the token: one unit past the deadline, the first item's
                St.m_TokenUnits += bForce;
                bTraceToken = false;
                ParallelFor(int(Live.size()), [&](int k)
                {
                    auto& W = m_Pending[Live[k]];
                    const int r0 = W.m_RaysDone, r1 = std::min(W.m_Rays, r0 + Chunk);
                    if (bForce && k == 0) { TraceChunk(W, r0, r1, L); RaysDone += r1 - r0; return; }
                    if (Allow > 0 && std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double, std::micro>(double(RayUs) * (r1 - r0))) > Deadline) return;
                    TraceChunk(W, r0, r1, L);
                    RaysDone += r1 - r0;
                });
                if (Allow <= 0 || RaysDone.load() == Before) break;      // one round traces everything without an allowance; no unit fits any more
            }
            St.m_Rays = RaysDone.load() + std::exchange(m_GPURays, 0);
            if (AllowMs > 0) m_Starve[1] = !m_Pending.empty() && St.m_Rays == 0 ? m_Starve[1] + 1 : 0;
            m_UnsupportedAt.clear();
            // the governor's estimate of a ray's cost, from this update's units (the mean and the mean deviation, slowly)
            m_RayUsHi *= 0.98f;
            for (const auto& W : m_Pending)
                if (W.m_UnitRays > 0)
                {
                    const float u = W.m_UnitUs / float(W.m_UnitRays);
                    m_RayUsHi = std::max(m_RayUsHi, u);
                    m_RayUsDev  += (std::abs(u - m_RayUsMean) - m_RayUsDev) * 0.05f;
                    m_RayUsMean += (u - m_RayUsMean) * 0.05f;
                    St.m_LongestSurfelUs = std::max(St.m_LongestSurfelUs, double(W.m_UnitUs));
                }
            // the finished surfels: their results into the solver's state
            for (const auto& W : m_Pending)
            {
                St.m_CastMs += W.m_CastUs * 1e-3; St.m_DirectMs += W.m_DirectUs * 1e-3; St.m_ReadMs += W.m_ReadUs * 1e-3;
                if (W.m_RaysDone < W.m_Rays) continue;
                if (m_Setup.m_bGPUOwned && W.m_bCommitted)      // G4c: a range: its surfels stay on the GPU
                {
                    St.m_Surfels += W.m_Traced; St.m_nUnsupported += W.m_nUnsupported; St.m_Hits += W.m_nHits;
                    if (W.m_nUnsupported && m_UnsupportedAt.size() < 4096) m_UnsupportedAt.push_back(W.m_UnsupportedAt);
                    continue;
                }
                m_Work.push_back({ W.m_Inst, W.m_Surf });
                St.m_nUnsupported += W.m_nUnsupported;
                if (W.m_nUnsupported && m_UnsupportedAt.size() < 4096) m_UnsupportedAt.push_back(W.m_UnsupportedAt);
                St.m_Hits += W.m_nHits;
            }
            for (auto& W : m_Pending) { St.m_TraceCpuMs += W.m_UnitUs * 1e-3; W.m_UnitUs = 0; W.m_UnitRays = 0; W.m_CastUs = W.m_DirectUs = W.m_ReadUs = 0; }
            std::erase_if(m_Pending, [](const work_item& W) { return W.m_RaysDone >= W.m_Rays; });
            St.m_Surfels += int(m_Work.size());
            St.m_Pending = int(m_Pending.size());
            for (const auto& W : m_Pending) St.m_PendingRays += W.m_Rays - W.m_RaysDone;

            St.m_TraceMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T1).count();
            TPrev = std::chrono::steady_clock::now();
            for (const auto& [i, s] : m_Work)       // the surfels finished this update
            {
                auto& I = m_Instances[i];
                EnsureSolver(I);
                // The light: the estimator (it warms up after a change, settles, and still follows one). The validity and the depth
                // moments: their own slow blends (they weigh the reads: noise there swims too)
                auto& Es = I.m_Est[s];
                const float Conf = std::max(1.0f / float(Es.m_N + 1), m_Setup.m_MinBlend);     // the blends of its fractions and validity: by its confidence
                if (std::size_t(s) < I.m_LastPass.size()) I.m_LastPass[s] = m_nUpdates;
                I.m_FilterDirty.push_back(s);
                I.m_MeanS = I.m_MeanS - I.m_E[s] * I.m_Valid[s]; I.m_MeanW -= I.m_Valid[s];     // its share of the mean, out (in again below)
                I.m_nBad -= BadSurfel(I.m_E[s], I.m_Valid[s]);
                Es.Add(I.m_Acc[s], m_Setup.m_MinBlend);
                I.m_E[s] = Es.m_Mean;
                const float av = std::max(Conf, 0.05f);
                I.m_Valid[s] = I.m_Valid[s] * (1 - av) + I.m_AccValid[s] * av;
                if (I.m_Level == 0 && s < I.m_SurfSeeThrough.size() && I.m_SurfSeeThrough[s]) I.m_Valid[s] = 0;     // on a transparent material
                {   // the basis: what fraction of the light came from around each direction (per channel), slowly (8 rays split 3 ways)
                    const auto& BA = I.m_BasisAcc[s];
                    const v3 Tot = BA[0] + BA[1] + BA[2];
                    const float af = std::max(Conf, 0.1f);
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
                I.m_MeanS = I.m_MeanS + I.m_E[s] * I.m_Valid[s]; I.m_MeanW += I.m_Valid[s];     // its new share of the mean
                I.m_nBad += BadSurfel(I.m_E[s], I.m_Valid[s]);
                if (++I.m_nMeanDeltas > I.m_E.size()) I.m_bMeanDirty = true;  // summed again exactly now and then (float drift)
            }

            if (!m_Setup.m_bGPUOwned) SyncGhosts();          // (GI chunks: the ghosts take their owners' committed set; the GPU's: its own pass)
            for (int i = 0; i < int(m_Instances.size()); ++i)
                if (m_Instances[i].m_bAlive && m_Instances[i].m_bCube)
                {
                    if (!m_Setup.m_bGPUOwned) ExpandCube(i);           // (G4c: the GPU's, gpu_trace)
                    m_Instances[i].m_bFilterFull = true; m_Instances[i].m_bMeanDirty = true;   // (36 parents or so)
                    ++St.m_nCubes; St.m_CubeParents += int(m_Instances[i].m_pMesh->m_Levels[1].size());
                }
            if (m_Setup.m_bGPUTrace) UpdateMeans();      // (G3: the batch dispatched after this update reads the light it committed, as the CPU's next trace would)
            Lap(St.m_CubeMs);
            ShownSettings();
            {
                bool bShowToken = Token == 2, bShowDid = false;
                const double ShowDeadline = AllowMs > 0 ? Grant(AllowMs - PreMs) : 0.0;     // what is left of the allowance, measured from T0 (none left: Grant's
                                                                                            // tiny one, not 0 = no limit: the GPU's part pre-charged it all and the filter ran unbounded)
                for (int i = 0; i < int(m_Instances.size()); ++i)
                    if (m_Setup.m_bGPUShown) { m_Instances[i].m_ShownChanged.clear(); m_Instances[i].m_bShownFull = false; m_Instances[i].m_FilterDirty.clear(); }     // (G4: the GPU's)
                    else if (m_Instances[i].m_bAlive) bShowDid |= UpdateShown(i, ShowDeadline, AllowMs > 0 ? &T0 : nullptr, &bShowToken);
                    else { m_Instances[i].m_ShownChanged.clear(); m_Instances[i].m_bShownFull = false; }
                if (AllowMs > 0) m_Starve[2] = bShowBacklog && !bShowDid ? m_Starve[2] + 1 : 0;
                St.m_OverAllowMs = AllowMs > 0 ? -Left() : 0.0;                  // (before the bookkeeping that follows: how far past it the stages went)
            }
            Lap(St.m_FilterMs);
            for (auto& I : m_Instances)
            {
                if (!I.m_bAlive) continue;
                if (!I.m_bPicked) ++I.m_Age;
                I.m_bPicked = false;
            }
            St.m_Ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();
            // The diagnostics, from what each instance keeps where it changes (Phase 3: no loop over every surfel every update): its
            // bad surfels and its mean (the finished surfels' deltas), its colors (its material), its emitted light (UpdateEmittedLight)
            std::size_t nAlbedo = 0;
            for (const auto& I : m_Instances)
            {
                if (!I.m_bAlive) continue;
                St.m_nBad += I.m_nBad;
                St.m_nLinks += int(I.m_Links.size());
                St.m_MaxEmitted = std::max(St.m_MaxEmitted, I.m_MaxEmitted);
                if (I.m_nAlbedo == 0) continue;
                ++St.m_nColored;
                St.m_MeanAlbedo = St.m_MeanAlbedo + I.m_AlbedoSum;
                nAlbedo += I.m_nAlbedo;
                St.m_nEmissive += I.m_nEmissiveSurf;
            }
            if (nAlbedo) St.m_MeanAlbedo = St.m_MeanAlbedo * (1.0f / float(nAlbedo));
            for (const auto& [i, e] : m_Emitters)
            {
                const v3 Le = m_Instances[i].m_SurfEmission[e] * m_Setup.m_EmissiveScale;
                if (Le.m_X + Le.m_Y + Le.m_Z > St.m_MaxEmitter.m_X + St.m_MaxEmitter.m_Y + St.m_MaxEmitter.m_Z) St.m_MaxEmitter = Le;
            }
            // memory by category (rule 5 of the plan)
            m_nEmitters = int(m_Emitters.size());
            St.m_nEmitters = m_nEmitters;
            for (const auto& I : m_Instances)
            {
                St.m_LinkBytes  += I.m_Links.capacity() * sizeof(I.m_Links[0]) + I.m_LinkFirst.capacity() * sizeof(std::uint32_t);
                St.m_LightBytes += (I.m_E.capacity() + I.m_Acc.capacity() + I.m_SurfAlbedo.capacity() + I.m_SurfEmission.capacity()) * sizeof(v3)
                                 + (I.m_Valid.capacity() + I.m_AccValid.capacity()) * sizeof(float)
                                 + I.m_Est.capacity() * sizeof(estimator) + (I.m_Frac.capacity() + I.m_BasisAcc.capacity() + I.m_EmitB.capacity()) * sizeof(basis)
                                 + I.m_Moments.capacity() * sizeof(moments) + I.m_MomAcc.capacity() * sizeof(I.m_MomAcc[0]) + I.m_MomCount.capacity() * sizeof(I.m_MomCount[0]);
                for (const auto& P : I.m_Pass) St.m_FilterBytes += P.capacity() * sizeof(basis);
                St.m_FilterBytes += I.m_Shown.capacity() * sizeof(basis) + (I.m_RevFirst.capacity() + I.m_RevIndex.capacity()) * sizeof(std::uint32_t);
            }
            Lap(St.m_StatsMs);
            St.m_MeanLum = m_MeanLum;
            St.m_TotalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();
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

        // A surfel's view of an emitter (PumpLinks): the emitter (m_Emitters), its weight (area x both cosines / distance^2:
        // the irradiance per unit of its radiance) and the shares of the surfel's 3 basis directions
        struct emit_link { std::uint32_t m_EmInst, m_EmSurf; float m_W; std::array<float, 3> m_Bw; };     // the emitter by (instance, fine surfel): stable while the emitter list changes

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
            std::vector<v3>         m_SurfEmitPower, m_SurfEmissionPoint; // per fine surfel: the power its triangles emit (SetTriangleEmission, batch by batch), and its emission sampled at its center (SetSurfelMaterial)
            v3                      m_EmitUnassigned{ 0, 0, 0 };      // the emitted power of triangles no surfel represents (none of their vertices' surfels faces their way)
            int                     m_nEmitUnassigned = 0;
            std::size_t             m_EmitResolveWork = 0;            // (tests) candidates examined by AccumulateTriangleEmission: at most 12 a small emissive triangle, 64 a sample of a large one
            bool                    m_bTriEmission = false;           // every batch of the triangles' emission is in: m_SurfEmission is made from m_SurfEmitPower (ResolveSurfEmission)
            std::vector<std::uint8_t> m_SeeThrough, m_SurfSeeThrough;   // per triangle, per fine surfel: transparent (SetSeeThrough; empty: none)
            std::vector<std::uint32_t> m_LinkFirst;              // per fine surfel + 1: its emitter links (PumpLinks)
            std::vector<emit_link>  m_Links;
            std::vector<int>        m_EmitSurfs;                // the fine surfels that emit (EmittersChanged)
            v3                      m_LinkLo{ 0, 0, 0 }, m_LinkHi{ 0, 0, 0 };   // the published links' hull: the union of the surfels' (m_SurfLinkLo/Hi)
            bool                    m_bLinkHull   = false;
            bool                    m_bLinkQueued = false;      // in m_LinkQueue
            bool                    m_bLinkDirtyAll = false;    // every surfel's links are built again (MarkLinks)
            std::vector<std::uint8_t> m_LinkDirty;              // per fine surfel: its links are built again (MarkLinksCrossing)
            std::vector<v3>         m_SurfLinkLo, m_SurfLinkHi; // per fine surfel: it and the emitters it examined (what an occluder must cross)
            bool                    m_bEmitDirty  = false;      // m_EmitB is made again from all its links (UpdateEmittedLight)
            bool                    m_bEmitRelevel = false;     // m_EmitB is made again from m_EmitFine (its level changed)
            std::vector<basis>      m_EmitFine, m_EmitSum;      // per fine surfel its emitted light; per parent the sum of its children's
            std::vector<int>        m_ParentCnt;                // per parent its children
            std::vector<basis>      m_EmitB;                    // per surfel of the active level: the emitters' irradiance this update, per basis direction
            std::vector<basis>      m_Frac;                     // per surfel: the fraction of its light from around each basis direction (per channel)
            std::vector<basis>      m_BasisAcc;                 // this pass: the light per basis direction
            mutable std::vector<std::vector<basis>> m_Pass;     // UpdateShown: per filter pass (0: unfiltered) per surfel
            mutable std::vector<basis> m_Shown;                 // BasisLight: the last pass, saturated
            mutable std::vector<int>   m_FilterDirty;           // surfels whose light, fractions or validity changed since UpdateShown
            mutable std::vector<int>   m_ShownChanged;          // what the last UpdateShown made again (the renderer's upload)
            mutable std::vector<std::uint8_t> m_Mark;           // (UpdateShown's scratch)
            mutable std::vector<std::uint32_t> m_RevFirst, m_RevIndex;   // per surfel: who reads it (the neighbor lists reversed)
            mutable bool               m_bFilterFull = true;    // UpdateShown makes all of it again
            mutable bool               m_bShownFull  = false;   // the last UpdateShown made all of it
            mutable int                m_FullPass = -1;         // UpdateShown's whole filter in progress: the pass (nP + 1: shown), -1: none
            mutable int                m_FullAt   = 0;          // ... and its cursor
            bool                       m_bMeanDirty  = true;    // m_MeanS / m_MeanW are made again (its light changed)
            v3                         m_MeanS{ 0, 0, 0 };      // the sum of its valid surfels' light, and of their validity
            float                      m_MeanW = 0;
            std::size_t                m_nMeanDeltas = 0;       // finished surfels since m_MeanS was summed exactly
            int                        m_nBad = 0;              // its surfels with a light that is not one (BadSurfel)
            v3                         m_AlbedoSum{ 0, 0, 0 };  // its material's colors (EmittersChanged: the diagnostics)
            std::size_t                m_nAlbedo = 0;
            int                        m_nEmissiveSurf = 0;
            float                      m_MaxEmitted = 0;        // its brightest emitted light (UpdateEmittedLight)
            std::vector<std::array<float, 3 * moment_texels_v>> m_MomAcc;     // this pass: per texel the sum of d, of d^2, the count
            std::vector<std::array<std::uint16_t, moment_texels_v>> m_MomCount; // per texel the samples it has had (its blend: 1 / n, then slow)
            std::vector<estimator>  m_Est;                      // per surfel of the active level: its light over time
            std::uint32_t           m_EmitGen   = 0;            // G4: m_EmitB's version (the GPU's copy)
            std::uint32_t           m_MatGen    = 0;            // G4c: its material's version (albedo, emission, see-through: GPUMaterial)
            bool                    m_bProvisional = false;     // 7a: its material not captured yet (SetProvisional): its hits bring nothing
            float                   m_RaysPerStep = 0;          // G4c: the rays a step of its cursor cost on the GPU (its ranges' mean; 0: not known)
            int                     m_Level     = 0;
            bool                    m_bCube     = false;        // level 1 as an ambient cube: only m_CubeSet is traced, the other parents blend from them (ExpandCube)
            std::uint32_t           m_nOwned    = 0xFFFFFFFFu;  // GI chunks: its level-0 surfels from here on are ghosts (SetGhosts); all: none
            std::vector<std::array<int, 2>> m_GhostOwner;      //   per ghost: its owner's instance and surfel
            std::vector<int>        m_CubeSet;                  // the parents traced: for each axis direction the one facing it best
            int                     m_Age       = 0;
            int                     m_History   = 0;
            int                     m_Cursor    = 0;            // the next surfel of the pass in progress (0: none)
            float                   m_DirtyA    = 1;            // MarkDirty: how much the change affects it (estimator::Lower at its next pass)
            role                    m_Role      = role::ASSET;  // Phase 5: its transport (SetRole, Transports)
            std::vector<int>        m_Urgent;                   // Phase 4: surfels a change affected, traced first (AffectNear)
            mutable bool            m_bBuckets = false;         // BuildBuckets: its surfels in world cells (what a change near it looks at)
            mutable float           m_BucketCell = 1;
            mutable v3              m_BucketLo{ 0, 0, 0 };
            mutable int             m_BucketDim[3] = { 1, 1, 1 };
            mutable std::vector<float> m_BucketPos;             // per surfel: its world center
            mutable std::vector<int>   m_BucketCellOf;          // per surfel: its cell (BuildBuckets' slices)
            mutable std::size_t        m_BucketAt = 0;          // BuildBuckets: the next surfel of the slices
            std::vector<std::array<v3, 3>> m_PendingAffect;     // Phase 6b: changes near it waiting for its buckets (Lo, Hi, reach in x)
            mutable std::vector<std::uint32_t> m_BucketFirst, m_BucketIndex;
            std::vector<std::uint8_t> m_UrgentMark;             // per surfel: 1 in m_Urgent, 2 picked from it this pick
            std::vector<std::uint32_t> m_LastPass;              // per surfel: the update that last finished it
            std::vector<std::array<float, 2>> m_LastOff;        // per surfel: its last pass' ray offsets (tests: the ray sequence advances)
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
            I.m_bBuckets = false; I.m_BucketAt = 0;
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
            I.m_E.assign(n, v3{ 0, 0, 0 });
            I.m_Valid.assign(n, 1);
            ResetMoments(I);
            for (auto& Es : I.m_Est) Es.Restart();                      // new: the first sample is its light
        }

        // The CPU solver's state per surfel (the pass' accumulators, the estimators, the moments' counts): only where the CPU solves. GPU-owned,
        // the GPU keeps its own (gpu_trace); these arrays were ~250 B a surfel at the fine count, never read (GI working set, job 2). Made
        // when the CPU solves again, or reads the GPU's state back (SetGPUState), from the light it has then
        void AllocSolver(instance& I) noexcept
        {
            const std::size_t n = I.m_E.size();            I.m_Acc.assign(n, v3{ 0, 0, 0 }); I.m_AccValid.assign(n, 1);
            I.m_MomAcc.assign(n, {});
            I.m_MomCount.assign(n, {});
            I.m_BasisAcc.assign(n, basis{});
            I.m_Est.assign(n, estimator{});                             // (all new, as the GPU's re-level: G4c)
            for (std::size_t s = 0; s < n; ++s) I.m_Est[s].Seed(I.m_E[s]);
        }
        void EnsureSolver(instance& I) noexcept { if (I.m_Est.size() != I.m_E.size() || I.m_Acc.size() != I.m_E.size()) AllocSolver(I); }
        static void FreeSolver(instance& I) noexcept
        {
            const auto Free = [](auto& V) { V.clear(); V.shrink_to_fit(); };     // (= {} keeps the capacity)
            Free(I.m_Acc); Free(I.m_AccValid); Free(I.m_MomAcc); Free(I.m_MomCount); Free(I.m_BasisAcc); Free(I.m_Est);
        }

        // Fully open (every direction travels the whole reach): a new surfel reads at full weight until its rays say otherwise
        void ResetMoments(instance& I) noexcept      // (a new layout: its shown light and its mean are made again too)
        {
            I.m_bFilterFull = true; I.m_bMeanDirty = true;
            ++I.m_MomentsReset;
            I.m_Urgent.clear(); I.m_UrgentMark.assign(I.m_pMesh->m_Levels[I.m_Level].size(), 0); I.m_bBuckets = false; I.m_BucketAt = 0;
            I.m_LastPass.assign(I.m_pMesh->m_Levels[I.m_Level].size(), 0); I.m_LastOff.assign(I.m_pMesh->m_Levels[I.m_Level].size(), { 0.0f, 0.0f });
            const auto& L = I.m_pMesh->m_Levels[I.m_Level];
            I.m_Moments.resize(L.size());
            const v3 Third{ 1.0f / 3, 1.0f / 3, 1.0f / 3 };
            I.m_Frac.assign(L.size(), basis{ Third, Third, Third });
            if (m_Setup.m_bGPUOwned) FreeSolver(I); else AllocSolver(I);
            for (std::size_t s = 0; s < L.size(); ++s)
            {
                const float R = 2 * L[s].m_Radius;
                for (int t = 0; t < moment_texels_v; ++t) { I.m_Moments[s][2 * t] = R; I.m_Moments[s][2 * t + 1] = R * R; }
            }
        }

        // Phase 4: a change in a box (a moved, added or removed instance, a changed material or point light) affects the surfels around
        // it by their distance to it: 1 at the box, 0 at its reach (twice its diagonal, at least half a meter, at most m_ChangeMargin: a
        // small object changes the light near it, a large one farther). An affected surfel's confidence drops (estimator::Lower), its
        // moments refresh when it is affected by half or more, and it is traced first (its instance's urgent list). Self (an instance
        // that changed itself: moved, recolored) is affected as a whole instead (MarkDirty). Returns the instances affected
        int AffectNear(const v3& Lo, const v3& Hi, int Self = -1, float SelfA = 1.0f) noexcept
        {
            const float Mg = std::clamp(2.0f * (Hi - Lo).Length(), std::min(0.5f, m_Setup.m_ChangeMargin), std::max(m_Setup.m_ChangeMargin, 1e-3f));
            const v3 M{ Mg, Mg, Mg };
            const v3 A = Lo - M, B = Hi + M;
            int nNew = 0;       // (instrumentation)
            for (int i = 0; i < int(m_Instances.size()); ++i)
            {
                auto& I = m_Instances[i];
                if (i == Self || !I.m_bAlive || I.m_E.empty() || !(I.m_Hi.m_X >= A.m_X && I.m_Lo.m_X <= B.m_X && I.m_Hi.m_Y >= A.m_Y && I.m_Lo.m_Y <= B.m_Y && I.m_Hi.m_Z >= A.m_Z && I.m_Lo.m_Z <= B.m_Z)) continue;
                if (I.m_bDirty) continue;                               // its whole pass is coming anyway
                // Phase 6b: its buckets not made yet (the first change near it, or it moved): the box waits, the buckets are made in
                // slices of the update's allowance (PumpAffects), then it is applied
                if (!I.m_bBuckets) { I.m_PendingAffect.push_back({ Lo, Hi, v3{ Mg, 0, 0 } }); m_bPendingAffects = true; ++nNew; continue; }
                const bool bAny = ApplyAffect(I, Lo, Hi, Mg);
                nNew += bAny;
            }
            if (Self >= 0) MarkDirty(m_Instances[Self], SelfA);
            return nNew;
        }

        // The surfels of I within Mg of the box: lowered by their distance, their moments refreshed by half or more, queued (urgent)
        bool ApplyAffect(instance& I, const v3& Lo, const v3& Hi, float Mg) noexcept
        {
            const v3 M{ Mg, Mg, Mg };
            const v3 A = Lo - M, B = Hi + M;
            bool bAny = false;
            const float* W = I.m_BucketPos.data();
            const int c0[3] = { BucketOf(I, 0, A.m_X), BucketOf(I, 1, A.m_Y), BucketOf(I, 2, A.m_Z) }, c1[3] = { BucketOf(I, 0, B.m_X), BucketOf(I, 1, B.m_Y), BucketOf(I, 2, B.m_Z) };
            for (int z = c0[2]; z <= c1[2]; ++z) for (int y = c0[1]; y <= c1[1]; ++y) for (int x = c0[0]; x <= c1[0]; ++x)
            {
                const int Cell = (z * I.m_BucketDim[1] + y) * I.m_BucketDim[0] + x;
                for (std::uint32_t k = I.m_BucketFirst[Cell]; k < I.m_BucketFirst[Cell + 1]; ++k)
                {
                    const std::uint32_t s = I.m_BucketIndex[k];
                    if (s >= I.m_E.size()) continue;
                    const float* C = W + 3 * std::size_t(s);       // (plain floats: a Debug build calls every v3 operator)
                    const float dx = std::max(std::max(Lo.m_X - C[0], C[0] - Hi.m_X), 0.0f), dy = std::max(std::max(Lo.m_Y - C[1], C[1] - Hi.m_Y), 0.0f), dz = std::max(std::max(Lo.m_Z - C[2], C[2] - Hi.m_Z), 0.0f);
                    const float a = 1.0f - std::sqrt(dx * dx + dy * dy + dz * dz) / Mg;
                    if (a <= 0) continue;
                    if (s < I.m_Est.size()) { I.m_Est[s].Lower(a); if (a >= 0.5f) I.m_MomCount[s] = {}; }     // (GPU-owned: the GPU's, the op below)
                    if (m_Setup.m_bGPUTrace) m_GPUOps.push_back({ int(&I - m_Instances.data()), int(s), a, a >= 0.5f ? 1 : 0 });
                    if (!I.m_UrgentMark[s]) { I.m_UrgentMark[s] = 1; I.m_Urgent.push_back(int(s)); }
                    bAny = true;
                }
            }
            return bAny;
        }

        // Phase 6b: the boxes waiting for buckets (AffectNear), their buckets made in slices until the deadline (pToken: one slice past it)
        bool PumpAffects(double DeadlineMs, const std::chrono::steady_clock::time_point& T0, bool* pToken) noexcept
        {
            if (!m_bPendingAffects) return false;
            bool bDid = false, bLeft = false;
            const auto Late = [&] { return DeadlineMs > 0 && std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count() >= DeadlineMs; };
            for (auto& I : m_Instances)
            {
                if (I.m_PendingAffect.empty()) continue;
                if (!I.m_bAlive) { I.m_PendingAffect.clear(); continue; }
                while (!I.m_bBuckets)
                {
                    if (Late() && !(pToken && *pToken && !bDid)) break;
                    if (Late() && pToken) *pToken = false;
                    BuildBuckets(I, 2048);
                    bDid = true;
                }
                if (!I.m_bBuckets) { bLeft = true; continue; }
                for (const auto& [Lo, Hi, Mg] : I.m_PendingAffect) ApplyAffect(I, Lo, Hi, Mg.m_X);
                I.m_PendingAffect.clear();
            }
            m_bPendingAffects = bLeft;
            return bDid;
        }

        // An instance's surfels (its active level) by where they are in the world, in cells of m_ChangeMargin: what a change near it
        // looks at (AffectNear). Made again after it moves or switches level (m_bBuckets false). Phase 6b: in slices of Slice surfels
        // (0: all of it now); the cells are counted and filled when the last slice is in
        void BuildBuckets(instance& I, int Slice = 0) const noexcept
        {
            if (I.m_bBuckets) return;
            const auto& L = I.m_pMesh->m_Levels[I.m_Level];
            const float Cell = std::max(m_Setup.m_ChangeMargin, 1e-3f);
            if (I.m_BucketAt == 0)
            {
                I.m_BucketCell = Cell; I.m_BucketLo = I.m_Lo;
                for (int a = 0; a < 3; ++a) I.m_BucketDim[a] = std::clamp(int((I.m_Hi[a] - I.m_Lo[a]) / Cell) + 1, 1, 256);
                I.m_BucketPos.resize(3 * L.size());
                I.m_BucketCellOf.resize(L.size());
            }
            const int nCells = I.m_BucketDim[0] * I.m_BucketDim[1] * I.m_BucketDim[2];
            auto& CellOf = I.m_BucketCellOf;
            float T[12];                                        // the placement as plain floats (a Debug build calls every v3 and matrix operator)
            for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) T[r * 4 + c] = I.m_L2W(r, c);
            const std::size_t End = Slice > 0 ? std::min(L.size(), I.m_BucketAt + std::size_t(Slice)) : L.size();
            for (std::size_t s = I.m_BucketAt; s < End; ++s)
            {
                const float x = L[s].m_Center.m_X, y = L[s].m_Center.m_Y, z = L[s].m_Center.m_Z;
                float* C = &I.m_BucketPos[3 * s];
                for (int r = 0; r < 3; ++r) C[r] = T[r * 4] * x + T[r * 4 + 1] * y + T[r * 4 + 2] * z + T[r * 4 + 3];
                CellOf[s] = (BucketOf(I, 2, C[2]) * I.m_BucketDim[1] + BucketOf(I, 1, C[1])) * I.m_BucketDim[0] + BucketOf(I, 0, C[0]);
            }
            I.m_BucketAt = End;
            if (End < L.size()) return;
            I.m_BucketAt = 0;
            I.m_BucketFirst.assign(std::size_t(nCells) + 1, 0);
            for (std::size_t s = 0; s < L.size(); ++s) ++I.m_BucketFirst[CellOf[s] + 1];
            for (int c = 0; c < nCells; ++c) I.m_BucketFirst[c + 1] += I.m_BucketFirst[c];
            I.m_BucketIndex.resize(L.size());
            std::vector<std::uint32_t> At(I.m_BucketFirst.begin(), I.m_BucketFirst.end() - 1);
            for (std::size_t s = 0; s < L.size(); ++s) I.m_BucketIndex[At[CellOf[s]]++] = std::uint32_t(s);
            I.m_bBuckets = true;
        }
        static int BucketOf(const instance& I, int Axis, float v) noexcept
        {
            return std::clamp(int((v - I.m_BucketLo[Axis]) / I.m_BucketCell), 0, I.m_BucketDim[Axis] - 1);
        }

        // A whole instance changed (by a: its confidence at the next pass, estimator::Lower): it is traced again from the start
        static void MarkDirty(instance& I, float a) noexcept
        {
            I.m_DirtyA = I.m_bDirty ? std::max(I.m_DirtyA, a) : a;
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
                    AffectNear(P - R, P + R);
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
            for (int i = 0; i < int(m_Instances.size()); ++i) if (m_Instances[i].m_bAlive && Transports(m_Instances[i])) m_TlasIds.push_back(i);
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

        // The one priority (documentation/gi_dynamic_master_plan.md, Phase 2b): which instance is served first, from one queue under the
        // one allowance. Its terms: a change (dirty) first, then the surfels a change affected (Phase 4's urgent list), then a pass in
        // progress (finish what was started), then age over distance
        // (m_LodDistance: the distance at which it counts half). Later terms (Phase 5's transport roles, Phase 6's relighting) are added
        // here, never as a pass or a queue of their own. Fairness stays outside it (Update: the oldest instance's share, whatever this says)
        float Priority(const instance& I, const v3& Eye) const noexcept
        {
            return (I.m_bDirty ? 1e6f : 0.0f) + (I.m_Urgent.empty() ? 0.0f : 7e5f) + (I.m_Cursor ? 5e5f : 0.0f) + float(I.m_Age + 1) / (1 + BoxDistance(I, Eye) / m_Setup.m_LodDistance);
        }

        // Phase 6 (section 5d: rays requested, the allowance granting): how many rays a surfel asks for. All of them while it warms up
        // or watches a change (estimator::Lower); settled, by its uncertainty (the relative deviation of its samples: a quiet surfel needs
        // fewer), a quarter at least. The samples' deviation shrinks with their ray count, so fewer rays raise it again: the loop settles
        // instead of starving a surfel. m_bAdaptiveRays false: always m_RaysPerSurfel
        int RequestedRays(const instance& I, int Surf) const noexcept
        {
            const int R = std::max(m_Setup.m_RaysPerSurfel, 1);
            if (!m_Setup.m_bAdaptiveRays || std::size_t(Surf) >= I.m_Est.size()) return R;
            const auto& Es = I.m_Est[Surf];
            if (Es.m_N < warmup_passes_v || Es.m_bWatch) return R;
            const float Mean = 0.299f * Es.m_Mean.m_X + 0.587f * Es.m_Mean.m_Y + 0.114f * Es.m_Mean.m_Z;
            const float Dev  = std::sqrt(std::max(0.299f * Es.m_Variance.m_X + 0.587f * Es.m_Variance.m_Y + 0.114f * Es.m_Variance.m_Z, 0.0f));
            const float u    = Mean > 1e-4f ? Dev / Mean : 1.0f;
            return std::clamp(int(std::ceil(float(R) * u * m_Setup.m_RayUncertainty)), std::max(1, R / 4), R);
        }

        // Within a settled instance's pass, how often a surfel is served (1: every pass, n: one pass in n, staggered). A buried surfel
        // (validity 0: inside a solid, on a see-through material) one pass in dead_retrace_v: the reads give it no weight, and its validity
        // is geometry (a change dirties the instance: every pass again, and it comes back). A far one by its distance in its radii (its
        // footprint on the screen; m_bDistanceTiers): every 2nd, 4th, 8th pass, never dropped
        int TracePeriod(const instance& I, int Surf, const v3& Eye) const noexcept
        {
            if (I.m_Valid[Surf] < 0.01f) return dead_retrace_v;
            if (!m_Setup.m_bDistanceTiers) return 1;
            const surfel& S = I.m_pMesh->m_Levels[I.m_Level][Surf];
            const float   d = (I.m_L2W * S.m_Center - Eye).Length() / (S.m_Radius * std::max(I.m_MeanScale, 1e-9f));
            return d < far_tier_radii_v[0] ? 1 : d < far_tier_radii_v[1] ? 2 : d < far_tier_radii_v[2] ? 4 : 8;
        }

        static bool Transports(const instance& I) noexcept           // it blocks, bounces and emits for the others
        {
            return I.m_Role == role::FULL || (I.m_Role == role::ASSET && I.m_pMesh->m_bBounce);
        }

        static int BadSurfel(const v3& E, float Valid) noexcept      // not a light: not finite, or negative
        {
            return !(std::isfinite(E.m_X + E.m_Y + E.m_Z + Valid) && E.m_X >= 0 && E.m_Y >= 0 && E.m_Z >= 0);
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
        static constexpr int max_links_v    = 32;              // enough that neighbors keep about the same emitters (no steps where their sets differ)
        static constexpr int max_examined_v = 64;              // the shadow rays of one receiver surfel at most (the link work's unit is bounded: rule 4)

        // Phase 2 (documentation/gi_dynamic_master_plan.md): the links are maintained per receiver surfel. A change marks what it
        // concerns: a receiver that moves, all its surfels; an emitter that moves or appears, every receiver; an occluder, only the
        // surfels whose examined segments (each surfel's box with the emitters it examined) cross its old or new box. A queue of
        // instances builds the marked surfels in chunks within the links' share of the allowance, beside the published links, and
        // publishes the instance when its list is done. A mark while an instance is being built queues it again (repeated edits
        // coalesce, a drag still publishes as it goes); after the last change every marked surfel was built after it, so the links
        // equal a fresh build of the final scene
        struct emitter_pt { v3 m_C, m_N; float m_A; };
        // A candidate: the listed surfels are built (into one flat buffer: no allocation per surfel), then the instance's links are
        // flattened with the published ones of the other surfels into staging storage, a slice per update, and swapped in
        struct link_build
        {
            static constexpr std::uint8_t not_built_v = 0xFF;
            int                                  m_Inst  = -1;
            int                                  m_Next  = 0;      // the next entry of m_List
            std::vector<int>                     m_List;           // the receiver surfels to build
            std::vector<emit_link>               m_Flat;           // the built surfels' links
            std::vector<std::uint32_t>           m_Off;            // per receiver surfel: its first in m_Flat
            std::vector<std::uint8_t>            m_Cnt;            // per receiver surfel: its count (not_built_v: keeps the published)
            std::vector<v3>                      m_Lo, m_Hi;       // per receiver surfel: it and the emitters it examined
            int                                  m_FlatNext = -1;  // >= 0: flattening, the next surfel
            std::vector<std::uint32_t>           m_StFirst;        // the staging storage
            std::vector<emit_link>               m_StLinks;
            std::vector<basis>                   m_StEmit;         // per receiver surfel its emitted light (made in the slices too)
            v3                                   m_HullLo{ 0, 0, 0 }, m_HullHi{ 0, 0, 0 };
        };

        static bool Overlap(const v3& aLo, const v3& aHi, const v3& bLo, const v3& bHi) noexcept
        {
            return aLo.m_X <= bHi.m_X && bLo.m_X <= aHi.m_X && aLo.m_Y <= bHi.m_Y && bLo.m_Y <= aHi.m_Y && aLo.m_Z <= bHi.m_Z && bLo.m_Z <= aHi.m_Z;
        }

        void QueueLinks(int Id) noexcept
        {
            auto& I = m_Instances[Id];
            if (!I.m_bLinkQueued) { I.m_bLinkQueued = true; m_LinkQueue.push_back(Id); }
        }
        void MarkLinks(int Id) noexcept                         // every surfel of the receiver
        {
            if (!m_bEmitterLinks || !m_Instances[Id].m_bAlive) return;
            m_Instances[Id].m_bLinkDirtyAll = true;
            QueueLinks(Id);
        }
        void MarkAllLinks() noexcept { for (int i = 0; i < int(m_Instances.size()); ++i) MarkLinks(i); }
        void MarkLinksCrossing(const v3& Lo, const v3& Hi) noexcept      // an occluder's box: the surfels whose examined segments may cross it
        {
            if (!m_bEmitterLinks) return;
            for (int i = 0; i < int(m_Instances.size()); ++i)
            {
                auto& I = m_Instances[i];
                if (!I.m_bAlive || !I.m_bLinkHull || I.m_bLinkDirtyAll || !Overlap(I.m_LinkLo, I.m_LinkHi, Lo, Hi)) continue;
                bool bAny = false;
                const v3* pLo = I.m_SurfLinkLo.data(); const v3* pHi = I.m_SurfLinkHi.data();      // (plain floats: a Debug build calls every v3 operator)
                const float lx = Lo.m_X, ly = Lo.m_Y, lz = Lo.m_Z, hx = Hi.m_X, hy = Hi.m_Y, hz = Hi.m_Z;
                for (std::size_t s = 0; s < I.m_SurfLinkLo.size(); ++s)
                    if (!I.m_LinkDirty[s] && pLo[s].m_X <= hx && lx <= pHi[s].m_X && pLo[s].m_Y <= hy && ly <= pHi[s].m_Y && pLo[s].m_Z <= hz && lz <= pHi[s].m_Z) { I.m_LinkDirty[s] = 1; bAny = true; }
                if (bAny) QueueLinks(i);
            }
        }

        // An instance's material changed: which of its surfels emit. Only a change of that set changes links (a link's weight is
        // geometric); a change of how much they emit changes the emitted light only
        void EmittersChanged(int Id) noexcept
        {
            auto& I = m_Instances[Id];
            I.m_AlbedoSum = v3{ 0, 0, 0 }; I.m_nAlbedo = I.m_SurfAlbedo.size(); I.m_nEmissiveSurf = 0;       // (the diagnostics)
            for (const auto& a : I.m_SurfAlbedo) I.m_AlbedoSum = I.m_AlbedoSum + a;
            for (const auto& e : I.m_SurfEmission) I.m_nEmissiveSurf += (e.m_X + e.m_Y + e.m_Z) > 0.01f;
            std::vector<int> Now;
            for (int e = 0; e < int(I.m_SurfEmission.size()) && !I.m_bProvisional; ++e)      // (7a: none while provisional)
                if (I.m_SurfEmission[e].m_X + I.m_SurfEmission[e].m_Y + I.m_SurfEmission[e].m_Z > 1e-3f) Now.push_back(e);
            m_bEmitDirty = true;
            if (Now == I.m_EmitSurfs) return;
            I.m_EmitSurfs = std::move(Now);
            m_bEmittersDirty = true;
            MarkAllLinks();
        }

        void RebuildEmitterList() noexcept
        {
            m_Emitters.clear(); m_Em.clear();
            for (int i = 0; i < int(m_Instances.size()); ++i)
            {
                const auto& I = m_Instances[i];
                if (!I.m_bAlive || !Transports(I)) continue;
                for (const int e : I.m_EmitSurfs)
                {
                    if (std::uint32_t(e) >= I.m_nOwned) continue;      // (a ghost: its owner emits)
                    const auto& S = I.m_pMesh->m_Levels[0][e];
                    const float r = S.m_Radius * I.m_MeanScale;
                    m_Emitters.push_back({ i, e });
                    m_Em.push_back({ I.m_L2W * S.m_Center, NormalToWorld(I, S.m_Normal), 1.47f * r * r });     // its share of the surface: surfels baked r apart cover ~1.47 r^2 each
                }
            }
        }

        // Entries [First, First + n) of the build's list: every emissive surfel each receiver surfel sees, the strongest that nothing hides
        // (a shadow ray each, max_examined_v at most), once - not by waiting for 8 random rays to find a small sign. Returns the slowest
        // surfel's time (us)
        float BuildLinkChunk(link_build& B, int First, int n) noexcept
        {
            const auto& I = m_Instances[B.m_Inst];
            const auto& L = I.m_pMesh->m_Levels[0];
            std::vector<float> Us(std::size_t(n), 0.0f);
            std::vector<emit_link>    Scratch(std::size_t(n) * max_links_v);
            std::vector<std::uint8_t> Count(std::size_t(n), 0);
            ParallelFor(n, [&](int j)
            {
                const auto TS = std::chrono::steady_clock::now();
                const int s = B.m_List[std::size_t(First + j)];
                const v3 C = I.m_L2W * L[s].m_Center, N = NormalToWorld(I, L[s].m_Normal);
                v3 Lo = C, Hi = C;
                emit_link* const pOut = &Scratch[std::size_t(j) * max_links_v];
                int nOut = 0;
                std::vector<std::pair<float, std::uint32_t>> Cand;
                for (std::uint32_t k = 0; k < m_Em.size(); ++k)
                {
                    if (m_Emitters[k].first == B.m_Inst && m_Emitters[k].second == s) continue;
                    const v3 D = m_Em[k].m_C - C;
                    const float d2 = D.Dot(D);
                    if (d2 < 1e-10f) continue;
                    const v3 Dn = D * (1.0f / std::sqrt(d2));
                    const float cr = N.Dot(Dn), ce = -m_Em[k].m_N.Dot(Dn);
                    if (cr <= 0 || ce <= 0) continue;
                    const float w = m_Em[k].m_A * cr * ce / (d2 + m_Em[k].m_A);     // a small disk, not a point: soft up close (no hotspot next to an emitter)
                    if (w >= 1e-5f) Cand.push_back({ w, k });
                }
                const auto Better = [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
                if (Cand.size() > std::size_t(max_examined_v)) { std::nth_element(Cand.begin(), Cand.begin() + max_examined_v, Cand.end(), Better); Cand.resize(max_examined_v); }
                std::sort(Cand.begin(), Cand.end(), Better);
                const basis Dirs = BasisDirs(L[s].m_Normal);
                for (const auto& [w, k] : Cand)
                {
                    if (nOut >= max_links_v) break;
                    const v3 E = m_Em[k].m_C;
                    Lo = v3{ std::min(Lo.m_X, E.m_X), std::min(Lo.m_Y, E.m_Y), std::min(Lo.m_Z, E.m_Z) };   // examined: an occluder leaving it uncovers it
                    Hi = v3{ std::max(Hi.m_X, E.m_X), std::max(Hi.m_Y, E.m_Y), std::max(Hi.m_Z, E.m_Z) };
                    const v3 D = E - C;
                    const float d = D.Length();
                    if (Occluded(C + N * 1e-3f, D * (1.0f / d), d * 0.98f)) continue;      // stops short of the emitter's own surface
                    v3 Dm = I.m_W2L.TransformDirection(D * (1.0f / d));
                    Dm = Dm * (1.0f / std::max(Dm.Length(), 1e-12f));
                    pOut[nOut++] = { std::uint32_t(m_Emitters[k].first), std::uint32_t(m_Emitters[k].second), w, BasisWeights(Dirs, Dm) };
                }
                B.m_Lo[s] = Lo; B.m_Hi[s] = Hi; Count[std::size_t(j)] = std::uint8_t(nOut);
                Us[j] = float(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - TS).count());
            });
            for (int j = 0; j < n; ++j)
            {
                const int s = B.m_List[std::size_t(First + j)];
                B.m_Off[s] = std::uint32_t(B.m_Flat.size()); B.m_Cnt[s] = Count[std::size_t(j)];
                B.m_Flat.insert(B.m_Flat.end(), Scratch.begin() + std::ptrdiff_t(j) * max_links_v, Scratch.begin() + std::ptrdiff_t(j) * max_links_v + Count[std::size_t(j)]);
            }
            m_LastChunkCpuUs = 0; for (const float u : Us) m_LastChunkCpuUs += u;
            return n ? *std::max_element(Us.begin(), Us.end()) : 0.0f;
        }

        // The publication, a slice of surfels per call until done (true): the built surfels' links, the others' published ones, into the
        // staging storage; then swapped in. The emitted light is made again for the built surfels only (UpdateEmittedLight)
        bool FlattenLinks(link_build& B, double AllowMs, const std::chrono::steady_clock::time_point& T0) noexcept
        {
            auto& I = m_Instances[B.m_Inst];
            const int n = int(B.m_Cnt.size());
            const bool bOld = I.m_LinkFirst.size() == std::size_t(n) + 1;
            if (B.m_FlatNext < 0)
            {
                B.m_FlatNext = 0;
                B.m_StFirst.assign(std::size_t(n) + 1, 0);
                B.m_StLinks.clear(); B.m_StLinks.reserve(B.m_Flat.size() + (bOld ? I.m_Links.size() : 0));
                B.m_StEmit.assign(std::size_t(n), basis{});
                B.m_HullLo = I.m_Lo; B.m_HullHi = I.m_Hi;
                if (I.m_SurfLinkLo.size() != std::size_t(n)) { I.m_SurfLinkLo.assign(std::size_t(n), I.m_Lo); I.m_SurfLinkHi.assign(std::size_t(n), I.m_Hi); }
            }
            constexpr int slice_v = 512;
            while (B.m_FlatNext < n)
            {
                const int End = std::min(n, B.m_FlatNext + slice_v);
                for (int s = B.m_FlatNext; s < End; ++s)
                {
                    if (B.m_Cnt[s] != link_build::not_built_v)
                    {
                        B.m_StLinks.insert(B.m_StLinks.end(), B.m_Flat.begin() + B.m_Off[s], B.m_Flat.begin() + B.m_Off[s] + B.m_Cnt[s]);
                        I.m_SurfLinkLo[s] = B.m_Lo[s]; I.m_SurfLinkHi[s] = B.m_Hi[s];
                        B.m_StEmit[s] = EmittedBy(B.m_Flat.data() + B.m_Off[s], B.m_Cnt[s]);
                    }
                    else if (bOld)
                    {
                        B.m_StLinks.insert(B.m_StLinks.end(), I.m_Links.begin() + I.m_LinkFirst[s], I.m_Links.begin() + I.m_LinkFirst[s + 1]);
                        B.m_StEmit[s] = I.m_EmitFine.size() == std::size_t(n) ? I.m_EmitFine[s] : EmittedAt(I, std::size_t(s));
                    }
                    B.m_StFirst[std::size_t(s) + 1] = std::uint32_t(B.m_StLinks.size());
                    const v3& a = I.m_SurfLinkLo[s]; const v3& b = I.m_SurfLinkHi[s];
                    B.m_HullLo = v3{ std::min(B.m_HullLo.m_X, a.m_X), std::min(B.m_HullLo.m_Y, a.m_Y), std::min(B.m_HullLo.m_Z, a.m_Z) };
                    B.m_HullHi = v3{ std::max(B.m_HullHi.m_X, b.m_X), std::max(B.m_HullHi.m_Y, b.m_Y), std::max(B.m_HullHi.m_Z, b.m_Z) };
                }
                B.m_FlatNext = End;
                if (AllowMs > 0 && B.m_FlatNext < n && std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count() >= AllowMs) return false;
            }
            // swapped in; the emitted light: per built surfel its change into its parent's sum (O(built)); or re-derived (O(surfels), once)
            const bool bDeltas = bOld && int(B.m_List.size()) < n && I.m_EmitFine.size() == std::size_t(n) && !I.m_bEmitDirty && !I.m_bEmitRelevel
                              && (I.m_Level == 0 ? I.m_EmitB.size() == std::size_t(n) : I.m_EmitSum.size() == I.m_pMesh->m_Levels[1].size() && I.m_EmitB.size() == I.m_EmitSum.size());
            if (bDeltas)
                for (const int x : B.m_List)
                {
                    const basis& New = B.m_StEmit[x];
                    I.m_MaxEmitted = std::max(I.m_MaxEmitted, EmittedLuma(New));       // ponytail: a decrease shows at the next re-derive
                    ++I.m_EmitGen;
                    if (I.m_Level == 0) I.m_EmitB[x] = New;
                    else
                    {
                        const int p = I.m_pMesh->m_ParentOf[x];
                        for (int i = 0; i < 3; ++i) I.m_EmitSum[p][i] = I.m_EmitSum[p][i] + (New[i] - I.m_EmitFine[x][i]);
                        const float c = 1.0f / float(std::max<int>(I.m_ParentCnt[p], 1));
                        for (int i = 0; i < 3; ++i) I.m_EmitB[p][i] = I.m_EmitSum[p][i] * c;
                    }
                }
            else I.m_bEmitRelevel = true;
            I.m_EmitFine.swap(B.m_StEmit);
            I.m_Links.swap(B.m_StLinks);
            if (I.m_Links.empty()) I.m_LinkFirst.clear(); else I.m_LinkFirst.swap(B.m_StFirst);
            I.m_LinkLo = B.m_HullLo; I.m_LinkHi = B.m_HullHi; I.m_bLinkHull = true;
            return true;
        }

        // The link work of an update: the queued receivers' marked surfels in chunks until the queue is empty or AllowMs is spent (0: no
        // limit, everything now). A chunk is sized by the slowest recent surfel; one started finishes (the overshoot is one chunk), and
        // every update makes progress (at least one surfel per worker). Returns the time it used
        double PumpLinks(double AllowMs, stats& St, bool bToken = false) noexcept
        {
            const auto T0 = std::chrono::steady_clock::now();
            const auto Elapsed = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count(); };
            if (!m_bEmitterLinks || m_Setup.m_bEmitterSampling)
            {
                if ((!m_Setup.m_bEmitterSampling && !m_Emitters.empty()) || !m_LinkQueue.empty() || m_LinkCur.m_Inst >= 0 || m_bLinksExist)
                {
                    m_Emitters.clear(); m_Em.clear(); m_LinkQueue.clear(); m_LinkCur = {}; m_bLinksExist = false;
                    for (auto& I : m_Instances) { I.m_Links.clear(); I.m_LinkFirst.clear(); I.m_bLinkQueued = false; I.m_bLinkHull = false; I.m_bEmitDirty = true; }
                }
                if (m_Setup.m_bEmitterSampling) RebuildEmitterList();      // (G5: small; it follows the emitters that moved)
                return 0;
            }
            m_bLinksExist = true;
            if (m_bEmittersDirty) { RebuildEmitterList(); m_bEmittersDirty = false; }
            for (;;)
            {
                if (m_LinkCur.m_Inst < 0)
                {
                    while (!m_LinkQueue.empty() && m_LinkCur.m_Inst < 0)
                    {
                        const int Id = m_LinkQueue.front(); m_LinkQueue.pop_front();
                        auto& I = m_Instances[Id];
                        I.m_bLinkQueued = false;
                        if (!I.m_bAlive) continue;
                        const int n = int(I.m_pMesh->m_Levels[0].size());
                        auto& B = m_LinkCur;
                        B.m_Inst = Id; B.m_Next = 0; B.m_List.clear();
                        if (I.m_LinkDirty.size() != std::size_t(n)) { I.m_LinkDirty.assign(std::size_t(n), 0); I.m_bLinkDirtyAll = true; }
                        for (int s = 0; s < n; ++s) if (I.m_bLinkDirtyAll || I.m_LinkDirty[s]) B.m_List.push_back(s);
                        std::fill(I.m_LinkDirty.begin(), I.m_LinkDirty.end(), std::uint8_t(0)); I.m_bLinkDirtyAll = false;
                        B.m_Flat.clear(); B.m_Off.assign(std::size_t(n), 0); B.m_Cnt.assign(std::size_t(n), link_build::not_built_v); B.m_FlatNext = -1;
                        B.m_Lo.assign(std::size_t(n), v3{ 0, 0, 0 }); B.m_Hi.assign(std::size_t(n), v3{ 0, 0, 0 });
                    }
                    if (m_LinkCur.m_Inst < 0) break;
                }
                auto& B = m_LinkCur;
                if (!m_Instances[B.m_Inst].m_bAlive) { B = {}; continue; }
                const int Left = int(B.m_List.size()) - B.m_Next;
                if (Left > 0 && AllowMs > 0 && Elapsed() >= AllowMs && !bToken) break;
                if (Left > 0)
                {
                    const double Used = Elapsed();
                    if (AllowMs > 0 && Used >= AllowMs && !bToken) break;
                    int Chunk = Left;
                    if (AllowMs > 0)
                    {
                        // the workers take a surfel each: a chunk of n costs about n / workers of the slowest surfel lately
                        int Fit = m_LinkUsHi > 0 ? int((AllowMs - Used) * 1000.0 * double(ParallelWidth()) / double(m_LinkUsHi)) : link_unit_first_v;   // (the time, then the workers: not truncated to whole surfels first)
                        if (Fit < 1)
                        {
                            if (!bToken) break;                                     // it would not fit: next update
                            bToken = false; Fit = 1; ++St.m_TokenUnits;             // the token (the stage that waited longest): one surfel past it
                        }
                        Chunk = std::min({ Fit, Left, ParallelWidth() });       // one round at most (a surfel per worker): admitted again before the next, the overshoot one surfel's time
                    }
                    Chunk = std::min(Chunk, 4096);                                  // (the chunk's scratch: 32 links a surfel)
                    const float SlowUs = BuildLinkChunk(B, B.m_Next, Chunk);
                    St.m_LinkCpuMs += m_LastChunkCpuUs * 1e-3; St.m_LinkLongestUs = std::max(St.m_LinkLongestUs, double(SlowUs));
                    m_LinkUsHi = std::max(m_LinkUsHi * 0.98f, SlowUs);              // conservative: the slowest surfel lately (it decays)
                    B.m_Next += Chunk; St.m_LinkSurfels += Chunk;
                }
                if (B.m_Next >= int(B.m_List.size()))
                {
                    if (!FlattenLinks(B, AllowMs, T0)) break;                         // the rest of the publication next update
                    ++St.m_nLinkRebuilds; ++m_nLinkRebuilds;
                    B = {};
                }
            }
            St.m_LinkPending = int(m_LinkQueue.size()) + (m_LinkCur.m_Inst >= 0 ? 1 : 0);
            return Elapsed();
        }

        // This update's emitted light reaching each surfel (its links x the emitters' current radiance), per basis direction, at the
        // instance's active level (a parent: the mean of its children)
        static float EmittedLuma(const basis& B) noexcept { return (B[0].m_X + B[1].m_X + B[2].m_X + B[0].m_Y + B[1].m_Y + B[2].m_Y) / 6.0f; }
        basis EmittedAt(const instance& I, std::size_t s) const noexcept     // a fine surfel's emitted light: its links x the emitters' radiance now
        {
            if (s + 1 >= I.m_LinkFirst.size()) return basis{};
            return EmittedBy(I.m_Links.data() + I.m_LinkFirst[s], int(I.m_LinkFirst[s + 1] - I.m_LinkFirst[s]));
        }
        basis EmittedBy(const emit_link* pLinks, int n) const noexcept
        {
            basis B{};
            for (int k = 0; k < n; ++k)
            {
                const auto& Lk = pLinks[k];
                const auto& EI = m_Instances[Lk.m_EmInst];
                if (!EI.m_bAlive || Lk.m_EmSurf >= EI.m_SurfEmission.size()) continue;      // (a published link to what is gone: until its rebuild)
                const v3 Le = EI.m_SurfEmission[Lk.m_EmSurf] * (Lk.m_W * m_Setup.m_EmissiveScale);
                for (int i = 0; i < 3; ++i) B[i] = B[i] + Le * (3.0f * Lk.m_Bw[i]);   // the basis' mean is the irradiance
            }
            return B;
        }

        // This update's emitted light reaching each surfel, per basis direction, at the instance's active level (a parent: the mean of its
        // children). Made again from the links when the emissions changed (bAll, m_bEmitDirty: O(links)); from the kept fine values when
        // the level switched or links were published (FlattenLinks makes the fine values in its slices)
        void UpdateEmittedLight(bool bAll) noexcept
        {
            for (auto& I : m_Instances)
            {
                const auto& M = *I.m_pMesh;
                const std::size_t nFine = M.m_Levels[0].size();
                if (!I.m_bAlive || I.m_Links.empty())
                {
                    if (!I.m_EmitB.empty() || !I.m_EmitFine.empty()) { I.m_EmitB.clear(); I.m_EmitFine.clear(); I.m_EmitSum.clear(); ++I.m_EmitGen; }
                    I.m_MaxEmitted = 0;
                    I.m_bEmitDirty = I.m_bEmitRelevel = false;
                    continue;
                }
                const bool bParents = I.m_Level != 0;
                if (bAll || I.m_bEmitDirty || I.m_EmitFine.size() != nFine)
                {
                    I.m_EmitFine.resize(nFine);
                    for (std::size_t s = 0; s < nFine; ++s) I.m_EmitFine[s] = EmittedAt(I, s);
                    I.m_bEmitRelevel = true;
                }
                I.m_bEmitDirty = false;
                if (!I.m_bEmitRelevel) continue;
                I.m_bEmitRelevel = false;
                ++I.m_EmitGen;
                if (!bParents) { I.m_EmitB = I.m_EmitFine; I.m_MaxEmitted = 0; for (const auto& B : I.m_EmitB) I.m_MaxEmitted = std::max(I.m_MaxEmitted, EmittedLuma(B)); continue; }
                const std::size_t nP = M.m_Levels[1].size();
                I.m_EmitSum.assign(nP, basis{}); I.m_ParentCnt.assign(nP, 0);
                for (std::size_t s = 0; s < nFine; ++s)
                {
                    const int p = M.m_ParentOf[s];
                    for (int i = 0; i < 3; ++i) I.m_EmitSum[p][i] = I.m_EmitSum[p][i] + I.m_EmitFine[s][i];
                    ++I.m_ParentCnt[p];
                }
                I.m_EmitB.resize(nP);
                for (std::size_t p = 0; p < nP; ++p) { const float c = 1.0f / float(std::max<int>(I.m_ParentCnt[p], 1)); for (int i = 0; i < 3; ++i) I.m_EmitB[p][i] = I.m_EmitSum[p][i] * c; }
                I.m_MaxEmitted = 0;
                for (const auto& B : I.m_EmitB) I.m_MaxEmitted = std::max(I.m_MaxEmitted, EmittedLuma(B));
            }
        }

        // The surface's colors at a ray hit: the hit triangle's albedo (SetTriangleAlbedo), else its fine surfels' (the hit triangle's
        // vertices' surfels, the reads' weights); the emission the surfels'. The instance's albedo and no emission when it has none
        void SurfaceAt(const hit& H, v3& Albedo, v3& Emission) const noexcept
        {
            const auto& I = m_Instances[H.m_Inst];
            const bool  bTri = std::size_t(H.m_Tri) < I.m_TriAlbedo.size();
            Albedo = bTri ? I.m_TriAlbedo[H.m_Tri] : I.m_Albedo; Emission = { 0, 0, 0 };
            if (I.m_bProvisional) { Albedo = { 0, 0, 0 }; return; }     // (7a: not captured yet)
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
                if (m_pReadCapture && w > 0) m_pReadCapture->push_back({ s, w * m_CaptureScale });     // (CompileTransport)
            }
            return Sum;
        }

        mutable std::vector<std::pair<int, float>>*  m_pReadCapture = nullptr;     // CompileTransport: the reads report their surfels' weights here
        mutable float                           m_CaptureScale = 1;         //   (a vertex's share of the triangle read)
        mutable std::vector<std::pair<int, float>>   m_CapT1, m_CapT2;          //   (ReadPoint's two tiers)
        bool                                    m_bEmittersDirty = false;   // m_Emitters / m_Em are made again (PumpLinks)
        std::deque<int>                         m_LinkQueue;                // receivers waiting for a link build (MarkLinks)
        link_build                              m_LinkCur;                  // the one being built
        std::vector<emitter_pt>                 m_Em;                       // per m_Emitters: world center, normal, area
        float                                   m_LastChunkCpuUs = 0;       // BuildLinkChunk: its surfels' time, summed (batch accounting)
        float                                   m_LinkUsHi = 0;             // a link build's cost of one receiver surfel, the slowest lately
        int                                     m_nMovesAcc = 0, m_MoveMarked = 0;     // (instrumentation) SetTransform since the last update
        double                                  m_MovesMsAcc = 0;
        std::vector<work_item>                  m_Pending;              // Phase 1: the surfels picked and not finished
        std::uint64_t                           m_GPUGen = 1;           // G3: the structure's version (what PackGPU and PackGPUSurfels write)
        std::uint64_t                           m_GPULevelGen = 1;      // the instances' levels' version (gpu_trace::Relevel: no repack)
        std::uint64_t                           m_GPUXformGen = 0;      // G4c: the placements' version (PackGPUTlas, the records' rows)
        std::vector<gpu_item>                   m_GPUOut;               // G3: GPUWork's list
        std::vector<gpu_range>                  m_GPURangesOut;         // G4c: GPURanges' list
        int                                     m_GPURays = 0;          // G3: the rays delivered since the last update
        float                                   m_MeanLum = 0;          // UpdateMeans: the world's mean light's luminance (stats)
        bool                                    m_bLinksExist = false;  // G5: the link pump ran (with sampling turned on: its links cleared once)
        std::vector<gpu_op>                     m_GPUOps;               // G4: the estimator changes for the GPU's copy (TakeGPUOps)
        float                                   m_RayUsMean = 50, m_RayUsDev = 50, m_RayUsHi = 0;     // the governor's running cost of a ray (us), and the slowest seen lately
        int                                     m_nEmitters = 0;
        unsigned                                m_nUpdates = 0;
        int                                     m_nLinkRebuilds = 0;        // (tests)
        unsigned                                m_RelightAt = 0;            // RelightAll: the update from which it may run again
        int                                     m_Starve[4] = { 0, 0, 0, 0 };   // Phase 6b: per stage (links, trace, shown, events) the updates it had work and did none: the token's
        bool                                    m_bPendingAffects = false;  // some instance has changes waiting for its buckets (PumpAffects)
        bool                                    m_bRelightDeferred = false; // a RelightAll that came too soon (Update runs it when it may)
        int                                     m_nProvisional = 0;         // 7a: instances whose material is not captured yet (SetProvisional)
        bool                                    m_bBooted = false;          // 7a: the initial captures were all done once (the scene-wide transition ran)
        mutable int                             m_ShownPasses = -1;         // the settings the shown light was made with (ShownSettings)
        mutable float                           m_ShownSat    = -1;
        bool                                    m_bEmitDirty = true;        // UpdateEmittedLight's inputs changed (Update)
        float                                   m_EmitScale = -1;           // the scale it was made with
        std::vector<std::pair<int, int>>        m_Emitters;         // (instance, fine surfel) of every surfel that emits
        std::vector<instance>                   m_Instances;
        bool                                    m_bSolverFreed = false;     // GPU-owned at the last Update: the CPU solver's arrays freed (AllocSolver)
        mutable bvh                             m_Tlas;
        mutable std::vector<int>                m_TlasIds;          // TLAS primitive -> instance (the alive ones)
        bvh                                     m_LightBvh;         // over the point lights' spheres
        std::size_t                             m_nLightBvh = ~std::size_t(0);
        mutable bool                            m_bTlasDirty = true;
        float                                   m_PrevGlobal[9] = { 1e30f };
        std::vector<std::array<float, 7>>       m_PrevPoints;
        std::vector<std::pair<float, int>>      m_Order;
        std::vector<std::pair<int, int>>        m_Work;
        std::vector<v3>                         m_UnsupportedAt;        // the last update's (UnsupportedAt)
    };
}

#endif
