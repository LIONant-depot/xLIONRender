#ifndef XLIONRENDER_WORLD_SDF_BAKE_H
#define XLIONRENDER_WORLD_SDF_BAKE_H
// The geometry's distance field, baked by the geometry's compiler (reflections' empty space skipping): per level, sparse bricks of 8^3 cells around the
// triangles. A cell holds its center's true distance to the nearest triangle (point to triangle, not rays), one byte, rounded down, over the range of
// range_cells_v cells. A brick that isn't there is one whose every cell is at least the range from any triangle: absent means far, not unknown, because the
// bricks come from the triangles themselves. So at a point p in a cell, cell_value - half the cell's diagonal is a lower bound of the distance: LowerBound.
// The running of the bricks is the caller's (T_PARALLEL_FOR: count, function of index), so that a compiler gives it its scheduler
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace xlionrender::gi::sdf_bake
{
    inline constexpr int brick_cells_v = 8;
    inline constexpr int range_cells_v = 4;
    using v3f = std::array<float, 3>;

    struct level
    {
        float                       m_Cell = 0;
        std::array<int, 3>          m_Org{ 0, 0, 0 }, m_Dim{ 0, 0, 0 };      // the page table's first brick and its size, in bricks
        std::vector<std::uint32_t>  m_Table;                                 // a brick's index in m_Bytes / 512; absent_v: none
        std::vector<std::uint8_t>   m_Bytes;
        static constexpr std::uint32_t absent_v = 0xFFFFFFFFu;

        // The stored distance at a cell (the range where no brick is)
        float CellDistance(const std::array<int, 3>& c) const noexcept
        {
            std::array<int, 3> b, l;
            for (int a = 0; a < 3; ++a)
            {
                const int fb = c[a] >= 0 ? c[a] / brick_cells_v : -((-c[a] + brick_cells_v - 1) / brick_cells_v);
                b[a] = fb - m_Org[a]; l[a] = c[a] - fb * brick_cells_v;
                if (b[a] < 0 || b[a] >= m_Dim[a]) return float(range_cells_v) * m_Cell;
            }
            const std::uint32_t i = m_Table[std::size_t(b[2]) * m_Dim[1] * m_Dim[0] + std::size_t(b[1]) * m_Dim[0] + b[0]];
            if (i == absent_v) return float(range_cells_v) * m_Cell;
            const std::uint8_t v = m_Bytes[std::size_t(i) * 512 + std::size_t(l[2]) * 64 + std::size_t(l[1]) * 8 + l[0]];
            return float(v) / 255.0f * float(range_cells_v) * m_Cell;
        }
        // A safe distance from p to any triangle (0 when it can't say): the distance stored at p's cell minus the farthest p can be from its center
        float LowerBound(const v3f& p) const noexcept
        {
            const std::array<int, 3> c{ int(std::floor(p[0] / m_Cell)), int(std::floor(p[1] / m_Cell)), int(std::floor(p[2] / m_Cell)) };
            return std::max(0.0f, CellDistance(c) - 0.8661f * m_Cell);
        }
    };

    inline float Dot(const v3f& a, const v3f& b) noexcept { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
    inline v3f   Sub(const v3f& a, const v3f& b) noexcept { return { a[0] - b[0], a[1] - b[1], a[2] - b[2] }; }

    // The squared distance from a point to a triangle (Ericson, Real-Time Collision Detection 5.1.5)
    inline float PointTriangle2(const v3f& p, const v3f& a, const v3f& b, const v3f& c) noexcept
    {
        const v3f ab = Sub(b, a), ac = Sub(c, a), ap = Sub(p, a);
        const float d1 = Dot(ab, ap), d2 = Dot(ac, ap);
        const auto Len2 = [](const v3f& v) { return Dot(v, v); };
        if (d1 <= 0 && d2 <= 0) return Len2(ap);
        const v3f bp = Sub(p, b);
        const float d3 = Dot(ab, bp), d4 = Dot(ac, bp);
        if (d3 >= 0 && d4 <= d3) return Len2(bp);
        const float vc = d1 * d4 - d3 * d2;
        const auto Seg = [&](const v3f& o, const v3f& e, float t) { const v3f q{ o[0] + e[0] * t - p[0], o[1] + e[1] * t - p[1], o[2] + e[2] * t - p[2] }; return Len2(q); };
        if (vc <= 0 && d1 >= 0 && d3 <= 0) return Seg(a, ab, d1 / std::max(d1 - d3, 1e-20f));
        const v3f cp = Sub(p, c);
        const float d5 = Dot(ab, cp), d6 = Dot(ac, cp);
        if (d6 >= 0 && d5 <= d6) return Len2(cp);
        const float vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) return Seg(a, ac, d2 / std::max(d2 - d6, 1e-20f));
        const float va = d3 * d6 - d5 * d4;
        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) return Seg(b, Sub(c, b), (d4 - d3) / std::max((d4 - d3) + (d5 - d6), 1e-20f));
        const float den = 1.0f / std::max(va + vb + vc, 1e-20f), v = vb * den, w = vc * den;
        const v3f q{ a[0] + ab[0] * v + ac[0] * w - p[0], a[1] + ab[1] * v + ac[1] * w - p[1], a[2] + ab[2] * v + ac[2] * w - p[2] };
        return Len2(q);
    }

    template <class T_PARALLEL_FOR>
    level BakeLevel(std::span<const v3f> Pos, std::span<const std::array<std::uint32_t, 3>> Tris, float Cell, T_PARALLEL_FOR&& ParallelFor)
    {
        level L; L.m_Cell = Cell;
        const float Brick = Cell * brick_cells_v, Range = Cell * range_cells_v;
        struct key_hash { std::size_t operator()(std::uint64_t k) const noexcept { return std::size_t(k * 0x9E3779B97F4A7C15ull); } };
        const auto Key = [](int x, int y, int z) { return (std::uint64_t(std::uint32_t(x + (1 << 20))) << 42) | (std::uint64_t(std::uint32_t(y + (1 << 20))) << 21) | std::uint64_t(std::uint32_t(z + (1 << 20))); };

        // Each triangle, in the bricks its box and the range touch
        std::unordered_map<std::uint64_t, std::vector<std::uint32_t>, key_hash> Lists;
        std::array<int, 3> Lo{ 1 << 30, 1 << 30, 1 << 30 }, Hi{ -(1 << 30), -(1 << 30), -(1 << 30) };
        for (std::size_t t = 0; t < Tris.size(); ++t)
        {
            std::array<int, 3> b0, b1;
            for (int a = 0; a < 3; ++a)
            {
                const float mn = std::min({ Pos[Tris[t][0]][a], Pos[Tris[t][1]][a], Pos[Tris[t][2]][a] }), mx = std::max({ Pos[Tris[t][0]][a], Pos[Tris[t][1]][a], Pos[Tris[t][2]][a] });
                b0[a] = int(std::floor((mn - Range) / Brick)); b1[a] = int(std::floor((mx + Range) / Brick));
            }
            for (int z = b0[2]; z <= b1[2]; ++z) for (int y = b0[1]; y <= b1[1]; ++y) for (int x = b0[0]; x <= b1[0]; ++x) Lists[Key(x, y, z)].push_back(std::uint32_t(t));
            for (int a = 0; a < 3; ++a) { Lo[a] = std::min(Lo[a], b0[a]); Hi[a] = std::max(Hi[a], b1[a]); }
        }
        if (Lists.empty()) return L;

        struct job { int x, y, z; const std::vector<std::uint32_t>* pList; };
        std::vector<job> Jobs; Jobs.reserve(Lists.size());
        for (const auto& [k, v] : Lists) Jobs.push_back({ int((k >> 42) & 0x1FFFFF) - (1 << 20), int((k >> 21) & 0x1FFFFF) - (1 << 20), int(k & 0x1FFFFF) - (1 << 20), &v });

        // A brick's cells: the nearest of its triangles. The ones nothing comes near are dropped: absent means far
        std::vector<std::vector<std::uint8_t>> Out(Jobs.size());
        ParallelFor(Jobs.size(), [&](std::size_t j)
        {
            const job& J = Jobs[j];
            std::vector<std::uint8_t> B(512);
            bool bNear = false;
            for (int z = 0; z < brick_cells_v; ++z) for (int y = 0; y < brick_cells_v; ++y) for (int x = 0; x < brick_cells_v; ++x)
            {
                const v3f p{ (float(J.x * brick_cells_v + x) + 0.5f) * Cell, (float(J.y * brick_cells_v + y) + 0.5f) * Cell, (float(J.z * brick_cells_v + z) + 0.5f) * Cell };
                float Best = Range * Range;
                for (const std::uint32_t t : *J.pList) Best = std::min(Best, PointTriangle2(p, Pos[Tris[t][0]], Pos[Tris[t][1]], Pos[Tris[t][2]]));
                bNear |= Best < Range * Range;
                B[std::size_t(z) * 64 + std::size_t(y) * 8 + x] = std::uint8_t(std::floor(std::sqrt(Best) / Range * 255.0f));     // (down: never above the truth)
            }
            if (bNear) Out[j] = std::move(B);
        });

        for (int a = 0; a < 3; ++a) { L.m_Org[a] = Lo[a]; L.m_Dim[a] = Hi[a] - Lo[a] + 1; }
        L.m_Table.assign(std::size_t(L.m_Dim[0]) * L.m_Dim[1] * L.m_Dim[2], level::absent_v);
        for (std::size_t j = 0; j < Jobs.size(); ++j)
        {
            if (Out[j].empty()) continue;
            const auto& J = Jobs[j];
            L.m_Table[std::size_t(J.z - Lo[2]) * L.m_Dim[1] * L.m_Dim[0] + std::size_t(J.y - Lo[1]) * L.m_Dim[0] + (J.x - Lo[0])] = std::uint32_t(L.m_Bytes.size() / 512);
            L.m_Bytes.insert(L.m_Bytes.end(), Out[j].begin(), Out[j].end());
        }
        return L;
    }

    // A self-check: at random points, the lower bound of each level never above the true distance (false: a point where it was)
    inline bool Check(const level& L, std::span<const v3f> Pos, std::span<const std::array<std::uint32_t, 3>> Tris, std::span<const v3f> Points) noexcept
    {
        for (const auto& p : Points)
        {
            float Best = 1e30f;
            for (const auto& T : Tris) Best = std::min(Best, PointTriangle2(p, Pos[T[0]], Pos[T[1]], Pos[T[2]]));
            if (L.LowerBound(p) > std::sqrt(Best) + 1e-4f) return false;
        }
        return true;
    }
}
#endif
