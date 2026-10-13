// GI lab, experiment 1 (research doc: "Owner discussion -> chosen direction: mesh-owned vertex surfels"). CPU only, no GPU:
//  - bake: quads split to the sample density, surfels per mesh (patches of the surface: center, normal, radius, area),
//    each vertex keeps up to 4 surfel indices; weights are computed at read time from distance and normal (never stored)
//  - solve: L = D + T L on the surfels, stochastically (cosine rays from each surfel, a hit returns its direct light plus the
//    cache's value there: one more bounce per iteration), temporal accumulation
//  - read: through the vertices (barycentric of the vertex values, the GPU path) and per point (the per-pixel fallback)
//  - reference: a CPU path tracer at fixed test points (indirect irradiance)
// Scenes: a Cornell box with a box (the box moves: convergence after a change), and a sealed room next to a lit one through
// a 2 cm wall (leak test: the dark side must stay dark).
//  - buried surfels (> 25% of their rays start inside geometry, e.g. floor under a box) weigh 0; the vertex read interpolates
//    (sum w E, sum w) and divides per pixel, so a buried vertex drops out (the GPU path: two varyings)
//  - hierarchy: one parent level (greedy cover of the surfels, a parent table); a light switch-on solved fine only vs
//    coarse-to-fine at the same ray budget; the parent level read through the same vertex indices (LOD)
// Results: research doc, "Experiment 1 results". --quick: fewer rays (the checks hold in both).
// ponytail: brute-force ray casts (no BVH: a few thousand triangles, bounds culling); one parent level, not a full hierarchy.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../../stb/stb_image_write.h"

namespace
{
    struct v3 { float x = 0, y = 0, z = 0; };
    v3    operator+(v3 a, v3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
    v3    operator-(v3 a, v3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
    v3    operator*(v3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
    v3    operator*(v3 a, v3 b) { return { a.x * b.x, a.y * b.y, a.z * b.z }; }
    float Dot(v3 a, v3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    v3    Cross(v3 a, v3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
    float Length(v3 a) { return std::sqrt(Dot(a, a)); }
    v3    Normalize(v3 a) { return a * (1.0f / Length(a)); }
    float Lum(v3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }
    constexpr float pi_v = 3.14159265f;

    struct surfel
    {
        v3    m_Center, m_Normal;
        float m_Radius = 0;
        v3    m_E;                  // indirect irradiance (the cache)
        v3    m_Acc;                // this iteration's estimate
        float m_Valid = 1;          // 0: buried (its rays start inside other geometry - e.g. floor under a box): reads skip it
        float m_AccValid = 1;
    };

    struct mesh
    {
        std::vector<v3>                     m_Pos, m_Nrm;
        std::vector<std::array<int, 3>>     m_Tris;
        v3                                  m_Albedo;
        std::vector<surfel>                 m_Surfels;
        std::vector<std::array<int, 4>>     m_VertSurfels;      // per vertex: up to 4 surfel indices (-1: none)
        v3                                  m_Lo{ 1e30f, 1e30f, 1e30f }, m_Hi{ -1e30f, -1e30f, -1e30f };   // bounds (ray culling)
        std::vector<surfel>                 m_Coarse;           // the parent level (the hierarchy's next level: LOD and solver)
        std::vector<int>                    m_ParentOf;         // per surfel: its parent in m_Coarse
    };

    bool g_bValidity = true;        // buried surfels are skipped by the reads (the experiment toggles it)

    // Read-time weight of a surfel at point P with normal N (shared by the vertex and the per-point read)
    float SurfelWeight(const surfel& S, v3 P, v3 N)
    {
        const float c = Dot(S.m_Normal, N);
        if (c < 0.5f) return 0;
        const float d = Length(P - S.m_Center) / (2.0f * S.m_Radius);
        const float f = std::max(0.0f, 1.0f - d);
        return f * f * c + 1e-6f;
    }

    bool g_bHomogeneous = true;     // the vertex read interpolates (sum w E, sum w) and divides per pixel (false: normalized per vertex)

    // The weighted sum of a point's surfels; W: the total weight (0: every candidate buried or out of reach)
    bool g_bReadCoarse = false;     // reads use the parent level (walking m_ParentOf from the vertex's finest indices)

    v3 ReadSurfels(const mesh& M, const int* pIdx, int n, v3 P, v3 N, float* pW = nullptr)
    {
        v3 Sum; float W = 0;
        int Seen[4]; int nSeen = 0;
        for (int k = 0; k < n; ++k)
        {
            if (pIdx[k] < 0) continue;
            int Idx = pIdx[k];
            if (g_bReadCoarse)
            {
                Idx = M.m_ParentOf[Idx];
                if (std::find(Seen, Seen + nSeen, Idx) != Seen + nSeen) continue;     // two children of one parent
                Seen[nSeen++] = Idx;
            }
            const auto& S = g_bReadCoarse ? M.m_Coarse[Idx] : M.m_Surfels[Idx];
            const float w = SurfelWeight(S, P, N) * (g_bValidity ? S.m_Valid : 1.0f);
            Sum = Sum + S.m_E * w; W += w;
        }
        if (pW) { *pW = W; return Sum; }
        return W > 0 ? Sum * (1.0f / W) : v3{};
    }

    // The 4 surfels nearest P with a compatible normal
    std::array<int, 4> Nearest4(const mesh& M, v3 P, v3 N)
    {
        std::array<std::pair<float, int>, 4> Best;
        Best.fill({ 1e30f, -1 });
        for (int s = 0; s < int(M.m_Surfels.size()); ++s)
        {
            if (Dot(M.m_Surfels[s].m_Normal, N) < 0.5f) continue;
            const float d = Length(P - M.m_Surfels[s].m_Center);
            if (d >= Best[3].first) continue;
            Best[3] = { d, s };
            std::sort(Best.begin(), Best.end());
        }
        return { Best[0].second, Best[1].second, Best[2].second, Best[3].second };
    }

    struct scene
    {
        std::vector<mesh>   m_Meshes;
        v3                  m_LightPos;
        v3                  m_LightPower;       // intensity (W/sr-like): E = I cos / d^2

        // A quad (corner, two edges; normal = Normalize(Cross(U, V))), split into n x n cells of at most MaxEdge
        void AddQuad(v3 C, v3 U, v3 V, v3 Albedo, float MaxEdge)
        {
            mesh M;
            M.m_Albedo = Albedo;
            const int nu = MaxEdge > 0 ? std::max(1, int(std::ceil(Length(U) / MaxEdge))) : 1;
            const int nv = MaxEdge > 0 ? std::max(1, int(std::ceil(Length(V) / MaxEdge))) : 1;
            const v3  N  = Normalize(Cross(U, V));
            for (int j = 0; j <= nv; ++j)
            for (int i = 0; i <= nu; ++i)
            {
                const v3 P = C + U * (float(i) / nu) + V * (float(j) / nv);
                M.m_Pos.push_back(P); M.m_Nrm.push_back(N);
                M.m_Lo = { std::min(M.m_Lo.x, P.x), std::min(M.m_Lo.y, P.y), std::min(M.m_Lo.z, P.z) };
                M.m_Hi = { std::max(M.m_Hi.x, P.x), std::max(M.m_Hi.y, P.y), std::max(M.m_Hi.z, P.z) };
            }
            for (int j = 0; j < nv; ++j)
            for (int i = 0; i < nu; ++i)
            {
                const int a = j * (nu + 1) + i, b = a + 1, c = a + nu + 1, d = c + 1;
                M.m_Tris.push_back({ a, b, d });
                M.m_Tris.push_back({ a, d, c });
            }
            m_Meshes.push_back(std::move(M));
        }

        // An axis box (outward faces) from Lo to Hi
        void AddBox(v3 Lo, v3 Hi, v3 Albedo, float MaxEdge)
        {
            const v3 D = Hi - Lo;
            const v3 X{ D.x, 0, 0 }, Y{ 0, D.y, 0 }, Z{ 0, 0, D.z };
            AddQuad(Lo, Z, X, Albedo, MaxEdge);             // -Y (Cross(Z, X) = +Y... flipped below)
            m_Meshes.back().m_Nrm.assign(m_Meshes.back().m_Nrm.size(), v3{ 0, -1, 0 });
            for (auto& t : m_Meshes.back().m_Tris) std::swap(t[1], t[2]);
            AddQuad(Lo + Y, X, Z, Albedo, MaxEdge);         // +Y: Cross(X, Z) = -Y -> fix
            m_Meshes.back().m_Nrm.assign(m_Meshes.back().m_Nrm.size(), v3{ 0, 1, 0 });
            AddQuad(Lo, Y, Z, Albedo, MaxEdge);             // -X: Cross(Y, Z) = +X -> fix
            m_Meshes.back().m_Nrm.assign(m_Meshes.back().m_Nrm.size(), v3{ -1, 0, 0 });
            AddQuad(Lo + X, Z, Y, Albedo, MaxEdge);         // +X
            m_Meshes.back().m_Nrm.assign(m_Meshes.back().m_Nrm.size(), v3{ 1, 0, 0 });
            AddQuad(Lo, X, Y, Albedo, MaxEdge);             // -Z: Cross(X, Y) = +Z -> fix
            m_Meshes.back().m_Nrm.assign(m_Meshes.back().m_Nrm.size(), v3{ 0, 0, -1 });
            AddQuad(Lo + Z, Y, X, Albedo, MaxEdge);         // +Z
            m_Meshes.back().m_Nrm.assign(m_Meshes.back().m_Nrm.size(), v3{ 0, 0, 1 });
        }

        struct hit { int m_Mesh = -1, m_Tri = -1; float m_T = 1e30f, m_U = 0, m_V = 0; };

        // Moller-Trumbore, both faces (shadow and bounce rays see every surface)
        hit Cast(v3 O, v3 D, float TMax = 1e30f) const
        {
            hit H; H.m_T = TMax;
            const v3 Inv{ 1.0f / (std::abs(D.x) > 1e-12f ? D.x : 1e-12f), 1.0f / (std::abs(D.y) > 1e-12f ? D.y : 1e-12f), 1.0f / (std::abs(D.z) > 1e-12f ? D.z : 1e-12f) };
            for (int m = 0; m < int(m_Meshes.size()); ++m)
            {
                const auto& M = m_Meshes[m];
                {   // slab test against the mesh's bounds (padded: planar meshes have a zero-thickness box)
                    const v3 Lo = M.m_Lo - v3{ 1e-3f, 1e-3f, 1e-3f }, Hi = M.m_Hi + v3{ 1e-3f, 1e-3f, 1e-3f };
                    const v3 A = (Lo - O) * Inv, B = (Hi - O) * Inv;
                    const float t0 = std::max({ std::min(A.x, B.x), std::min(A.y, B.y), std::min(A.z, B.z) });
                    const float t1 = std::min({ std::max(A.x, B.x), std::max(A.y, B.y), std::max(A.z, B.z) });
                    if (t1 < std::max(t0, 0.0f) || t0 > H.m_T) continue;
                }
                for (int t = 0; t < int(M.m_Tris.size()); ++t)
                {
                    const v3 A = M.m_Pos[M.m_Tris[t][0]], B = M.m_Pos[M.m_Tris[t][1]], C = M.m_Pos[M.m_Tris[t][2]];
                    const v3 E1 = B - A, E2 = C - A, P = Cross(D, E2);
                    const float det = Dot(E1, P);
                    if (std::abs(det) < 1e-9f) continue;
                    const float inv = 1.0f / det;
                    const v3 T = O - A;
                    const float u = Dot(T, P) * inv;
                    if (u < 0 || u > 1) continue;
                    const v3 Q = Cross(T, E1);
                    const float v = Dot(D, Q) * inv;
                    if (v < 0 || u + v > 1) continue;
                    const float d = Dot(E2, Q) * inv;
                    if (d > 1e-4f && d < H.m_T) H = { m, t, d, u, v };
                }
            }
            return H;
        }

        v3 HitPos(const hit& H) const
        {
            const auto& M = m_Meshes[H.m_Mesh]; const auto& T = M.m_Tris[H.m_Tri];
            return M.m_Pos[T[0]] * (1 - H.m_U - H.m_V) + M.m_Pos[T[1]] * H.m_U + M.m_Pos[T[2]] * H.m_V;
        }

        v3 Direct(v3 P, v3 N) const
        {
            const v3 L = m_LightPos - P;
            const float d = Length(L), c = Dot(N, L) / d;
            if (c <= 0) return {};
            if (Cast(P + N * 1e-3f, L * (1.0f / d), d - 1e-3f).m_Mesh >= 0) return {};
            return m_LightPower * (c / (d * d));
        }

        // The cache's indirect irradiance at a hit: the GPU path (vertex values interpolated) or per point
        v3 CacheAt(const hit& H, bool bPerPoint) const
        {
            const auto& M = m_Meshes[H.m_Mesh]; const auto& T = M.m_Tris[H.m_Tri];
            const v3 P = HitPos(H), N = M.m_Nrm[T[0]];
            if (bPerPoint) { const auto I = Nearest4(M, P, N); return ReadSurfels(M, I.data(), 4, P, N); }
            v3 Sum; float W = 0;
            const float B[3] = { 1 - H.m_U - H.m_V, H.m_U, H.m_V };
            for (int k = 0; k < 3; ++k)
            {
                // the GPU path: two varyings (sum w E, sum w) interpolated, divided per pixel - a buried vertex (w 0) drops out
                float w = 0;
                const v3 E = ReadSurfels(M, M.m_VertSurfels[T[k]].data(), 4, M.m_Pos[T[k]], N, &w);
                if (g_bHomogeneous) { Sum = Sum + E * B[k]; W += w * B[k]; }
                else                { Sum = Sum + (w > 0 ? E * (1.0f / w) : v3{}) * B[k]; W += B[k]; }
            }
            return W > 0 ? Sum * (1.0f / W) : v3{};
        }

        // Bake: surfels of radius R per mesh (greedy cover of the vertices, center = mean of its vertices), then per vertex the
        // 4 nearest compatible surfels
        void Bake(float R, float CoarseR = 0)
        {
            for (auto& M : m_Meshes)
            {
                M.m_Surfels.clear();
                for (std::size_t v = 0; v < M.m_Pos.size(); ++v)
                {
                    bool bCovered = false;
                    for (const auto& S : M.m_Surfels) bCovered |= Length(M.m_Pos[v] - S.m_Center) < R && Dot(S.m_Normal, M.m_Nrm[v]) > 0.8f;
                    if (!bCovered) M.m_Surfels.push_back({ M.m_Pos[v], M.m_Nrm[v], R });
                }
                std::vector<v3> Sum(M.m_Surfels.size()); std::vector<int> Cnt(M.m_Surfels.size(), 0);
                for (std::size_t v = 0; v < M.m_Pos.size(); ++v)
                {
                    const int s = Nearest4(M, M.m_Pos[v], M.m_Nrm[v])[0];
                    Sum[s] = Sum[s] + M.m_Pos[v]; ++Cnt[s];
                }
                for (std::size_t s = 0; s < M.m_Surfels.size(); ++s) if (Cnt[s]) M.m_Surfels[s].m_Center = Sum[s] * (1.0f / Cnt[s]);
                M.m_VertSurfels.resize(M.m_Pos.size());
                for (std::size_t v = 0; v < M.m_Pos.size(); ++v) M.m_VertSurfels[v] = Nearest4(M, M.m_Pos[v], M.m_Nrm[v]);

                // The parent level: the same greedy cover over the surfels' centers, radius CoarseR
                M.m_Coarse.clear();
                M.m_ParentOf.assign(M.m_Surfels.size(), 0);
                if (CoarseR <= 0) continue;
                for (const auto& S : M.m_Surfels)
                {
                    bool bCovered = false;
                    for (const auto& C : M.m_Coarse) bCovered |= Length(S.m_Center - C.m_Center) < CoarseR && Dot(C.m_Normal, S.m_Normal) > 0.8f;
                    if (!bCovered) M.m_Coarse.push_back({ S.m_Center, S.m_Normal, CoarseR });
                }
                std::vector<v3> CSum(M.m_Coarse.size()); std::vector<int> CCnt(M.m_Coarse.size(), 0);
                for (std::size_t s = 0; s < M.m_Surfels.size(); ++s)
                {
                    float Best = 1e30f;
                    for (std::size_t c = 0; c < M.m_Coarse.size(); ++c)
                    {
                        const float d = Length(M.m_Surfels[s].m_Center - M.m_Coarse[c].m_Center);
                        if (Dot(M.m_Coarse[c].m_Normal, M.m_Surfels[s].m_Normal) > 0.5f && d < Best) { Best = d; M.m_ParentOf[s] = int(c); }
                    }
                    CSum[M.m_ParentOf[s]] = CSum[M.m_ParentOf[s]] + M.m_Surfels[s].m_Center; ++CCnt[M.m_ParentOf[s]];
                }
                for (std::size_t c = 0; c < M.m_Coarse.size(); ++c) if (CCnt[c]) M.m_Coarse[c].m_Center = CSum[c] * (1.0f / CCnt[c]);
            }
        }

        // The fine level starts from its parents (going finer: no black pop, no cold start)
        void PushDown()
        {
            for (auto& M : m_Meshes)
                for (std::size_t s = 0; s < M.m_Surfels.size(); ++s) { M.m_Surfels[s].m_E = M.m_Coarse[M.m_ParentOf[s]].m_E; M.m_Surfels[s].m_Valid = M.m_Coarse[M.m_ParentOf[s]].m_Valid; }
        }
        int CoarseCount() const { int n = 0; for (const auto& M : m_Meshes) n += int(M.m_Coarse.size()); return n; }

        int SurfelCount() const { int n = 0; for (const auto& M : m_Meshes) n += int(M.m_Surfels.size()); return n; }

        static v3 CosineDir(v3 N, std::mt19937& Rng)
        {
            std::uniform_real_distribution<float> U(0, 1);
            const float r = std::sqrt(U(Rng)), phi = 2 * pi_v * U(Rng);
            const v3 T = Normalize(std::abs(N.x) > 0.5f ? Cross(N, v3{ 0, 1, 0 }) : Cross(N, v3{ 1, 0, 0 }));
            const v3 B = Cross(N, T);
            return Normalize(T * (r * std::cos(phi)) + B * (r * std::sin(phi)) + N * std::sqrt(std::max(0.0f, 1 - r * r)));
        }

        // One solver iteration: Rays per surfel; Alpha the temporal blend (1: replace - plain Jacobi)
        // bCoarse: update the parent level instead of the surfels (where hits read from is g_bReadCoarse)
        void Iterate(int Rays, float Alpha, bool bPerPoint, unsigned Seed, bool bCoarse = false)
        {
            const auto Level = [&](mesh& M) -> std::vector<surfel>& { return bCoarse ? M.m_Coarse : M.m_Surfels; };
            std::vector<std::pair<int, int>> Work;
            for (int m = 0; m < int(m_Meshes.size()); ++m) for (int s = 0; s < int(Level(m_Meshes[m]).size()); ++s) Work.push_back({ m, s });
            ParallelFor(int(Work.size()), [&](int i)
            {
                std::mt19937 Rng(Seed * 7919u + unsigned(i));
                auto& S = Level(m_Meshes[Work[i].first])[Work[i].second];
                v3 Sum;
                int nBack = 0;
                for (int r = 0; r < Rays; ++r)
                {
                    const v3 D = CosineDir(S.m_Normal, Rng);
                    const hit H = Cast(S.m_Center + S.m_Normal * 1e-3f, D);
                    if (H.m_Mesh < 0) continue;                         // closed scenes: no sky
                    const auto& M = m_Meshes[H.m_Mesh];
                    const v3 N = M.m_Nrm[M.m_Tris[H.m_Tri][0]];
                    if (Dot(N, D) > 0) { ++nBack; continue; }            // a back face: no light leaves it (and the surfel may be buried)
                    Sum = Sum + M.m_Albedo * (Direct(HitPos(H), N) + CacheAt(H, bPerPoint)) * (1.0f / pi_v);
                }
                S.m_Acc = Sum * (pi_v / Rays);                           // E = pi * mean(L) with cosine sampling
                S.m_AccValid = std::clamp(1.0f - (float(nBack) / Rays - 0.1f) / 0.15f, 0.0f, 1.0f);    // > 25% back faces: buried
            });
            for (auto& M : m_Meshes) for (auto& S : Level(M)) { S.m_E = S.m_E * (1 - Alpha) + S.m_Acc * Alpha; S.m_Valid = S.m_Valid * (1 - Alpha) + S.m_AccValid * Alpha; }
        }

        // Path-traced indirect irradiance at P (normal N): the reference
        v3 Reference(v3 P, v3 N, int Rays, int Depth, std::mt19937& Rng) const
        {
            v3 Sum;
            for (int r = 0; r < Rays; ++r)
            {
                v3 O = P, Nn = N, Throughput{ 1, 1, 1 };
                for (int d = 0; d < Depth; ++d)
                {
                    const v3 D = CosineDir(Nn, Rng);
                    const hit H = Cast(O + Nn * 1e-3f, D);
                    if (H.m_Mesh < 0) break;
                    const auto& M = m_Meshes[H.m_Mesh];
                    const v3 Nh = M.m_Nrm[M.m_Tris[H.m_Tri][0]];
                    if (Dot(Nh, D) > 0) break;
                    O = HitPos(H); Nn = Nh;
                    Throughput = Throughput * M.m_Albedo;                // albedo/pi * pi (cosine pdf)
                    Sum = Sum + Throughput * Direct(O, Nn) * (1.0f / pi_v) * pi_v;
                }
            }
            return Sum * (1.0f / Rays);
        }

        template <class F> static void ParallelFor(int n, F&& Fn)
        {
            const int nThreads = std::max(1u, std::thread::hardware_concurrency());
            std::atomic<int> Next{ 0 };
            std::vector<std::thread> T;
            for (int t = 0; t < nThreads; ++t) T.emplace_back([&] { for (int i; (i = Next++) < n;) Fn(i); });
            for (auto& t : T) t.join();
        }
    };

    struct test_point { int m_Mesh; v3 m_P, m_N; std::string m_Name; };

    // The indirect irradiance the cache gives at a test point: through the vertices of the triangle under it, or per point
    v3 CacheRead(const scene& S, const test_point& T, bool bPerPoint)
    {
        const auto H = S.Cast(T.m_P + T.m_N * 0.05f, T.m_N * -1.0f, 0.1f);
        if (H.m_Mesh != T.m_Mesh) return { -1, -1, -1 };
        return S.CacheAt(H, bPerPoint);
    }

    // Cornell box (1 x 1 x 1 room centered at the origin, inward faces) + a box; MaxEdge: the split density (0: as authored)
    scene Cornell(float MaxEdge, v3 BoxLo, v3 BoxHi)
    {
        scene S;
        const v3 W{ 0.75f, 0.75f, 0.75f }, R{ 0.65f, 0.05f, 0.05f }, G{ 0.12f, 0.6f, 0.1f };
        S.AddQuad({ -1, -1, -1 }, { 0, 0, 2 }, { 2, 0, 0 }, W, MaxEdge);     // floor (+Y)
        S.AddQuad({ -1,  1, -1 }, { 2, 0, 0 }, { 0, 0, 2 }, W, MaxEdge);     // ceiling (-Y)
        S.AddQuad({ -1, -1,  1 }, { 0, 2, 0 }, { 2, 0, 0 }, W, MaxEdge);     // back (-Z)
        S.AddQuad({ -1, -1, -1 }, { 2, 0, 0 }, { 0, 2, 0 }, W, MaxEdge);     // front (+Z, closes the box)
        S.AddQuad({ -1, -1, -1 }, { 0, 2, 0 }, { 0, 0, 2 }, R, MaxEdge);     // left (+X)
        S.AddQuad({  1, -1, -1 }, { 0, 0, 2 }, { 0, 2, 0 }, G, MaxEdge);     // right (-X)
        S.AddBox(BoxLo, BoxHi, W, MaxEdge);
        S.m_LightPos   = { 0.2f, 0.4f, -0.3f };
        S.m_LightPower = { 1, 1, 1 };
        return S;
    }

    // Two rooms (x in [-1, 0] lit, [0, 1] sealed) sharing a 2 cm wall
    scene TwoRooms(float MaxEdge)
    {
        scene S;
        const v3 W{ 0.75f, 0.75f, 0.75f };
        for (float x0 : { -1.0f, 0.01f })
        {
            const float x1 = x0 + 0.99f;
            S.AddQuad({ x0, -1, -1 }, { 0, 0, 2 }, { x1 - x0, 0, 0 }, W, MaxEdge);   // floor
            S.AddQuad({ x0,  1, -1 }, { x1 - x0, 0, 0 }, { 0, 0, 2 }, W, MaxEdge);   // ceiling
            S.AddQuad({ x0, -1,  1 }, { 0, 2, 0 }, { x1 - x0, 0, 0 }, W, MaxEdge);   // back
            S.AddQuad({ x0, -1, -1 }, { x1 - x0, 0, 0 }, { 0, 2, 0 }, W, MaxEdge);   // front
            S.AddQuad({ x0, -1, -1 }, { 0, 2, 0 }, { 0, 0, 2 }, W, MaxEdge);         // -x side (+X)
            S.AddQuad({ x1, -1, -1 }, { 0, 0, 2 }, { 0, 2, 0 }, W, MaxEdge);         // +x side (-X)
        }
        S.m_LightPos   = { -0.5f, 0.5f, 0 };
        S.m_LightPower = { 1, 1, 1 };
        return S;
    }

    int g_nErrors = 0;
    #define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); ++g_nErrors; } } while (0)

    struct error { double m_Mean = 0, m_Max = 0; std::string m_Worst; };
    error Compare(const scene& S, const std::vector<test_point>& Pts, const std::vector<v3>& Ref, bool bPerPoint)
    {
        error E; double RefMean = 0;
        for (const auto& r : Ref) RefMean += Lum(r);
        RefMean /= Ref.size();
        for (std::size_t i = 0; i < Pts.size(); ++i)
        {
            const double d = std::abs(Lum(CacheRead(S, Pts[i], bPerPoint)) - Lum(Ref[i])) / RefMean;    // relative to the scene's mean
            E.m_Mean += d / Pts.size(); if (d > E.m_Max) { E.m_Max = d; E.m_Worst = Pts[i].m_Name + " (" + std::to_string(Pts[i].m_P.x).substr(0, 5) + ", " + std::to_string(Pts[i].m_P.y).substr(0, 5) + ", " + std::to_string(Pts[i].m_P.z).substr(0, 5) + ")"; }
        }
        return E;
    }
}

namespace
{
    // A view of the box from inside the (open-looking) front: per pixel the primary hit, shaded albedo/pi (direct + indirect).
    // Indirect: Mode 0 none, 1 path traced (the reference), 2 the cache through the vertices
    void RenderPanel(const scene& S, int Mode, int Size, unsigned char* pOut, int Stride, int Paths)
    {
        const v3 Eye{ 0.8f, 0.3f, -0.9f };                                      // the front-right corner, looking at the box's base
        const v3 Fwd = Normalize(v3{ -0.35f, -0.85f, 0.1f } - Eye);
        const v3 Right = Normalize(Cross(v3{ 0, 1, 0 }, Fwd)), Up = Cross(Fwd, Right);
        const float Tan = std::tan(0.5f * 72.0f * pi_v / 180.0f);
        scene::ParallelFor(Size, [&](int y)
        {
            std::mt19937 Rng(9001 + y);
            for (int x = 0; x < Size; ++x)
            {
                const v3 D = Normalize(Right * (((x + 0.5f) / Size * 2 - 1) * Tan) + Up * ((1 - (y + 0.5f) / Size * 2) * Tan) + Fwd);
                const auto H = S.Cast(Eye, D);
                v3 C;
                if (H.m_Mesh >= 0)
                {
                    const auto& M = S.m_Meshes[H.m_Mesh];
                    const v3 N = M.m_Nrm[M.m_Tris[H.m_Tri][0]];
                    if (Dot(N, D) < 0)
                    {
                        const v3 P = S.HitPos(H);
                        v3 E = S.Direct(P, N);
                        if (Mode == 1) E = E + S.Reference(P, N, Paths, 6, Rng);
                        if (Mode == 2) E = E + S.CacheAt(H, false);
                        C = M.m_Albedo * E * (1.0f / pi_v);
                    }
                }
                unsigned char* o = pOut + (std::size_t(y) * Stride + x) * 3;
                const float c[3] = { C.x, C.y, C.z };
                for (int k = 0; k < 3; ++k)
                {
                    const float v = c[k] * 1.5f;                                    // exposure, then Reinhard and gamma
                    o[k] = (unsigned char)std::clamp(std::pow(v / (1 + v), 1 / 2.2f) * 255.0f + 0.5f, 0.0f, 255.0f);
                }
            }
        });
    }

    // --images <file.png>: a 3 x 2 sheet - direct only | path-traced reference | vertex surfels (0.125 m vertices, 0.3 m surfels)
    //                                       as authored (no split) | plain read (no buried test, no homogeneous read) | parent level (LOD)
    int Images(const char* pFile, v3 BoxLo, v3 BoxHi)
    {
        constexpr int size_v = 320;
        std::vector<unsigned char> Img(size_t(size_v) * 3 * size_v * 2 * 3, 0);
        const int Stride = size_v * 3;
        const auto Panel = [&](int cx, int cy) { return Img.data() + (std::size_t(cy) * size_v * Stride + std::size_t(cx) * size_v) * 3; };

        const scene Coarse = Cornell(0, BoxLo, BoxHi);
        RenderPanel(Coarse, 0, size_v, Panel(0, 0), Stride, 0);
        RenderPanel(Coarse, 1, size_v, Panel(1, 0), Stride, 256);

        scene S = Cornell(0.125f, BoxLo, BoxHi);
        S.Bake(0.3f, 0.75f);
        for (int it = 0; it < 6; ++it) S.Iterate(128, 1.0f, false, it);
        for (int k = 2; k <= 12; ++k) S.Iterate(128, 1.0f / k, false, 50 + k);
        RenderPanel(S, 2, size_v, Panel(2, 0), Stride, 0);

        scene A = Cornell(0, BoxLo, BoxHi);                                     // as authored: 2 triangles per wall
        A.Bake(2.0f);
        for (int it = 0; it < 6; ++it) A.Iterate(128, 1.0f, false, it);
        for (int k = 2; k <= 12; ++k) A.Iterate(128, 1.0f / k, false, 50 + k);
        RenderPanel(A, 2, size_v, Panel(0, 1), Stride, 0);

        g_bValidity = false; g_bHomogeneous = false;
        RenderPanel(S, 2, size_v, Panel(1, 1), Stride, 0);
        g_bValidity = true; g_bHomogeneous = true;

        scene L = S;
        g_bReadCoarse = true;
        for (int it = 0; it < 6; ++it) L.Iterate(128, 1.0f, false, 900 + it, true);
        for (int k = 2; k <= 12; ++k) L.Iterate(128, 1.0f / k, false, 950 + k, true);
        RenderPanel(L, 2, size_v, Panel(2, 1), Stride, 0);
        g_bReadCoarse = false;

        if (!stbi_write_png(pFile, size_v * 3, size_v * 2, 3, Img.data(), Stride * 3)) { std::printf("could not write %s\n", pFile); return 1; }
        std::printf("wrote %s\n", pFile);
        return 0;
    }
}

int main(int argc, char** argv)
{
    if (argc > 2 && std::strcmp(argv[1], "--images") == 0) return Images(argv[2], { -0.6f, -1, -0.2f }, { -0.1f, -0.2f, 0.4f });
    const bool bQuick = argc > 1 && std::strcmp(argv[1], "--quick") == 0;
    const int  RefRays = bQuick ? 256 : 2048;
    const auto Now = [] { return std::chrono::steady_clock::now(); };
    const auto Ms  = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };

    // Test points: floor grid, back wall grid, the contact zone next to the box
    const v3 BoxLo{ -0.6f, -1, -0.2f }, BoxHi{ -0.1f, -0.2f, 0.4f };
    std::vector<test_point> Pts;
    for (int j = 0; j < 4; ++j) for (int i = 0; i < 4; ++i)
    {
        const float a = -0.8f + 1.6f * (i + 0.37f) / 4, b = -0.8f + 1.6f * (j + 0.61f) / 4;
        if (!(a > BoxLo.x - 0.02f && a < BoxHi.x + 0.02f && b > BoxLo.z - 0.02f && b < BoxHi.z + 0.02f)) Pts.push_back({ 0, { a, -1, b }, { 0, 1, 0 }, "floor" });
        Pts.push_back({ 2, { a, b, 1 }, { 0, 0, -1 }, "back" });
    }
    Pts.push_back({ 0, { BoxHi.x + 0.05f, -1, 0.1f }, { 0, 1, 0 }, "contact" });

    // Reference (the scene's geometry does not depend on the split; the coarse one traces fastest)
    auto T0 = Now();
    const scene RefScene = Cornell(0, BoxLo, BoxHi);
    std::vector<v3> Ref(Pts.size());
    scene::ParallelFor(int(Pts.size()), [&](int i) { std::mt19937 Rng(1234 + i); Ref[i] = RefScene.Reference(Pts[i].m_P, Pts[i].m_N, RefRays, 6, Rng); });
    std::printf("reference: %zu points, %d paths each, %.0f ms\n", Pts.size(), RefRays, Ms(T0, Now()));

    // Density sweep: sample spacing (split size = surfel radius) vs error, vertex read and per-point read
    std::printf("%-12s %8s %8s %12s %12s %12s %12s %10s\n", "split/radius", "verts", "surfels", "vtx mean", "vtx max", "pt mean", "pt max", "solve ms");
    error Best{ 1e9, 1e9 }, Dense, Clustered;
    const std::pair<float, float> Sweep[] = { { 2.0f, 2.0f }, { 0.5f, 0.5f }, { 0.25f, 0.25f }, { 0.125f, 0.125f }, { 0.125f, 0.3f } };
    for (const auto& [Spacing, Radius] : Sweep)
    {
        scene S = Cornell(Spacing, BoxLo, BoxHi);
        S.Bake(Radius);
        int nVerts = 0; for (const auto& M : S.m_Meshes) nVerts += int(M.m_Pos.size());
        T0 = Now();
        for (int it = 0; it < 6; ++it) S.Iterate(bQuick ? 64 : 256, 1.0f, false, it);       // 6 bounces of Jacobi
        for (int k = 2; k <= 12; ++k) S.Iterate(bQuick ? 64 : 256, 1.0f / k, false, 50 + k);  // then average the noise out
        const double SolveMs = Ms(T0, Now());
        g_bValidity = false; g_bHomogeneous = false;
        const error NoValid = Compare(S, Pts, Ref, false);
        g_bValidity = true;
        const error NoHomog = Compare(S, Pts, Ref, false);
        g_bHomogeneous = true;
        const error V = Compare(S, Pts, Ref, false), P = Compare(S, Pts, Ref, true);
        std::printf("%5.3f/%-6.3f %8d %8d %11.1f%% %11.1f%% %11.1f%% %11.1f%% %10.0f\n", Spacing, Radius, nVerts, S.SurfelCount(), V.m_Mean * 100, V.m_Max * 100, P.m_Mean * 100, P.m_Max * 100, SolveMs);
        std::printf("             worst: %s | plain read: %.1f%% / %.1f%% at %s | + buried test: %.1f%% / %.1f%%\n", V.m_Worst.c_str()
                   , NoValid.m_Mean * 100, NoValid.m_Max * 100, NoValid.m_Worst.c_str(), NoHomog.m_Mean * 100, NoHomog.m_Max * 100);
        if (Spacing == 0.25f && Radius == 0.25f) Best = V;
        if (Spacing == 0.125f) (Radius == 0.125f ? Dense : Clustered) = V;
    }
    CHECK(Best.m_Mean < 0.25, "vertex surfels at 0.25 spacing: mean error %.1f%% of the scene's mean", Best.m_Mean * 100);
    CHECK(Clustered.m_Mean < 0.06 && Clustered.m_Max < 0.25, "dense vertices, clustered surfels: %.1f%% / %.1f%%", Clustered.m_Mean * 100, Clustered.m_Max * 100);
    CHECK(Dense.m_Max < 0.25, "the contact (buried surfels + homogeneous read): max %.1f%%", Dense.m_Max * 100);

    // Leak: the sealed room's side of the 2 cm wall must stay dark
    {
        scene S = TwoRooms(0.25f);
        S.Bake(0.25f);
        for (int it = 0; it < 6; ++it) S.Iterate(bQuick ? 64 : 256, 1.0f, false, 100 + it);
        double Lit = 0, Dark = 0; int nLit = 0, nDark = 0;
        for (const auto& M : S.m_Meshes) for (const auto& Sf : M.m_Surfels)
        {
            if (Sf.m_Center.x < 0) { Lit += Lum(Sf.m_E); ++nLit; } else { Dark += Lum(Sf.m_E); ++nDark; }
        }
        std::printf("leak: lit room mean indirect %.4f, sealed room %.6f\n", Lit / nLit, Dark / nDark);
        CHECK(Dark == 0, "light leaked into the sealed room (%g)", Dark / nDark);
    }

    // Change response: the box moves; temporal accumulation (8 rays per surfel per frame) - frames until the error settles
    {
        const v3 Moved{ 0.5f, 0, -0.3f };
        scene S = Cornell(0.25f, BoxLo, BoxHi);
        S.Bake(0.25f);
        for (int it = 0; it < 6; ++it) S.Iterate(64, 1.0f, false, 200 + it);
        scene After = Cornell(0.25f, BoxLo + Moved, BoxHi + Moved);           // same topology: carry the cache over
        After.Bake(0.25f);
        for (std::size_t m = 0; m < S.m_Meshes.size(); ++m)
            for (std::size_t s = 0; s < std::min(S.m_Meshes[m].m_Surfels.size(), After.m_Meshes[m].m_Surfels.size()); ++s)
                After.m_Meshes[m].m_Surfels[s].m_E = S.m_Meshes[m].m_Surfels[s].m_E;
        std::vector<test_point> Pts2;
        for (const auto& p : Pts) if (p.m_Name != "contact" && p.m_Mesh != 0) Pts2.push_back(p);            // the back wall
        std::vector<v3> Ref2(Pts2.size());
        {
            scene Conv = Cornell(0.25f, BoxLo + Moved, BoxHi + Moved);
            Conv.Bake(0.25f);
            for (int it = 0; it < 6; ++it) Conv.Iterate(64, 1.0f, false, 400 + it);
            for (int k = 2; k <= 12; ++k) Conv.Iterate(64, 1.0f / k, false, 450 + k);
            for (std::size_t i = 0; i < Pts2.size(); ++i) Ref2[i] = CacheRead(Conv, Pts2[i], false);
        }
        std::printf("before the move, back wall vs the moved scene's converged cache: %.1f%%\n", Compare(After, Pts2, Ref2, false).m_Mean * 100);
        std::printf("after the move (8 rays/surfel/frame, blend 0.2), back wall vs converged:");
        double FrameMs = 0;
        for (int f = 1; f <= 32; ++f)
        {
            T0 = Now();
            After.Iterate(8, 0.2f, false, 300 + f);
            FrameMs += Ms(T0, Now());
            if (f == 1 || f == 2 || f == 4 || f == 8 || f == 16 || f == 32) std::printf(" f%d %.1f%%", f, Compare(After, Pts2, Ref2, false).m_Mean * 100);
        }
        std::printf("  (%.2f ms/frame, %d surfels)\n", FrameMs / 32, After.SurfelCount());
    }

    // Hierarchy: a light switches on; fine only vs coarse-to-fine at the same ray budget per frame. Coarse-to-fine: the parent
    // level takes 4 bounces per frame (cheap: few surfels), the fine level gathers from it (its hits read the parents: every
    // bounce at once - hierarchical radiosity) and starts from it (PushDown)
    {
        scene S = Cornell(0.125f, BoxLo, BoxHi);
        S.Bake(0.3f, 0.75f);
        const int nFine = S.SurfelCount(), nCoarse = S.CoarseCount();
        std::vector<v3> Truth(Pts.size());
        {
            scene Conv = S;
            for (int it = 0; it < 6; ++it) Conv.Iterate(64, 1.0f, false, 500 + it);
            for (int k = 2; k <= 12; ++k) Conv.Iterate(64, 1.0f / k, false, 550 + k);
            for (std::size_t i = 0; i < Pts.size(); ++i) Truth[i] = CacheRead(Conv, Pts[i], false);
        }
        const int FineRays = 8;                                                     // budget: nFine x 8 rays per frame
        const int CoarseRays = std::max(1, nFine * FineRays / 2 / (nCoarse * 4));   // half of it: 4 coarse bounces
        std::printf("light switch-on (%d surfels, %d parents; budget %d rays/frame):\n", nFine, nCoarse, nFine * FineRays);
        double First[2] = {};
        for (int Mode = 0; Mode < 2; ++Mode)
        {
            scene A = S;                                                            // all dark, the light just came on
            std::printf("  %-15s", Mode == 0 ? "fine only" : "coarse-to-fine");
            for (int f = 1; f <= 32; ++f)
            {
                const float Alpha = std::max(1.0f / f, 0.2f);                       // history reset on the change, then a 0.2 blend
                if (Mode == 0 || f > 8) A.Iterate(FineRays, Alpha, false, 600 + f);     // settled (8 frames after the change): fine reads fine
                else
                {
                    g_bReadCoarse = true;
                    for (int b = 0; b < 4; ++b) A.Iterate(CoarseRays, b == 0 ? Alpha : 1.0f, false, 700 + f * 8 + b, true);
                    if (f == 1) A.PushDown();
                    A.Iterate(FineRays / 2, Alpha, false, 800 + f);                  // hits read the parents
                    g_bReadCoarse = false;
                }
                if (f == 1 || f == 2 || f == 4 || f == 8 || f == 16 || f == 32)
                {
                    const double e = Compare(A, Pts, Truth, false).m_Mean;
                    if (f == 1) First[Mode] = e;
                    std::printf(" f%d %5.1f%%", f, e * 100);
                }
            }
            std::printf("\n");
        }
        // LOD: the parent level read through the same vertex indices (a far instance) vs the converged fine level
        scene L = S;
        g_bReadCoarse = true;
        for (int it = 0; it < 6; ++it) L.Iterate(64, 1.0f, false, 950 + it, true);
        const error Coarse = Compare(L, Pts, Truth, false);
        g_bReadCoarse = false;
        std::printf("  LOD: the parent level (%d surfels) read through the vertices: %.1f%% mean vs the fine level\n", nCoarse, Coarse.m_Mean * 100);
        CHECK(First[1] < 0.5 * First[0], "coarse-to-fine is not faster after a light switch-on (frame 1: %.1f%% vs %.1f%%)", First[1] * 100, First[0] * 100);
        CHECK(Coarse.m_Mean < 0.12, "the parent level as a LOD: %.1f%%", Coarse.m_Mean * 100);
    }

    if (g_nErrors) return 1;
    std::printf("PASS\n");
    return 0;
}
