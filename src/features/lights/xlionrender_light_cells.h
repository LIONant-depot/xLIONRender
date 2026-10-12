#ifndef XLIONRENDER_LIGHT_CELLS_H
#define XLIONRENDER_LIGHT_CELLS_H
#pragma once

// World-cell light lists - the lighting lab (research doc: "Taking STB further", the three-mode lab). A camera-centered clipmap of
// world cells ("little world": levels of cells_v^3 world-anchored cells, slot = cell index mod cells_v, level L's cells 2^L the
// size of level 0's). Each cell lists the local lights whose sphere reaches it, in light order, so shading a point with its cell's
// list equals shading it with every light (mode CELLS == mode EXACT). Mode SAMPLED splits each list: the head (the cell's
// head_v most important lights, kept in light order) is shaded exactly, the tail by one light J drawn with its probability
// p(J): L = sum(head) + f(J) / p(J). Every tail light keeps p > 0 (a floor), head and tail never share a light.
//
// Layout (uints, shaders/xlionrender_local_lights.glsl reads the same): per cell (level-major, then slot) a header
// { first entry, count, head count, 0 }, then entries { light index, floatBitsToUint(tail cumulative probability) }.
// ponytail: rebuilt from scratch every render on the CPU (the lab measures that cost); dirty cells, the visibility term and
// a compute builder come after the lab says the cells earn their place.

#include "dependencies/xmath/source/xmath.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xlionrender
{
    struct light_cells
    {
        static constexpr int levels_v = 5;                                      // level 4: 32 unit cells, a 512 unit window (aerial views stay inside)
        static constexpr int cells_v  = 16;                                     // per level and axis
        static constexpr int level_cells_v = cells_v * cells_v * cells_v;
        static constexpr int header_uints_v = 4;

        struct light
        {
            xmath::fvec3    m_Position;
            float           m_Radius;
            float           m_Power;            // luminance x intensity: the importance estimate's scale
            xmath::fvec3    m_Direction{ 0, -1, 0 };    // spot: where it points (normalized)
            float           m_OuterAngle = -1;  // spot: the cone's outer half angle (radians); < 0: a point light
        };

        // Cells of level 0 are CellSize wide. HeadCount: the exact head of mode SAMPLED (0: no split - mode CELLS).
        void Build(const std::vector<light>& Lights, const xmath::fvec3& Eye, float CellSize, int HeadCount) noexcept
        {
            m_CellSize = CellSize;
            for (int L = 0; L < levels_v; ++L)
            {
                const float S = CellSize * float(1 << L);
                for (int a = 0; a < 3; ++a) m_Origin[L][a] = int(std::floor(Eye[a] / S)) - cells_v / 2;
            }

            // Pass 1: count per cell, pass 2: fill (light order kept)
            constexpr int nCells = levels_v * level_cells_v;
            std::vector<std::uint32_t> Count(nCells, 0);
            const auto ForEachCell = [&](const light& Lt, auto&& Fn)
            {
                for (int L = 0; L < levels_v; ++L)
                {
                    const float S = CellSize * float(1 << L);
                    int Lo[3], Hi[3];
                    bool bEmpty = false;
                    for (int a = 0; a < 3; ++a)
                    {
                        Lo[a] = std::max(int(std::floor((Lt.m_Position[a] - Lt.m_Radius) / S)), m_Origin[L][a]);
                        Hi[a] = std::min(int(std::floor((Lt.m_Position[a] + Lt.m_Radius) / S)), m_Origin[L][a] + cells_v - 1);
                        bEmpty |= Lo[a] > Hi[a];
                    }
                    if (bEmpty) continue;
                    for (int z = Lo[2]; z <= Hi[2]; ++z)
                    for (int y = Lo[1]; y <= Hi[1]; ++y)
                    for (int x = Lo[0]; x <= Hi[0]; ++x)
                    {
                        // the sphere against the cell's box (a light listed only where it can reach)
                        const int   C[3] = { x, y, z };
                        float D2 = 0;
                        for (int a = 0; a < 3; ++a)
                        {
                            const float Min = C[a] * S, Max = Min + S, V = Lt.m_Position[a];
                            if (V < Min) D2 += (Min - V) * (Min - V); else if (V > Max) D2 += (V - Max) * (V - Max);
                        }
                        if (D2 > Lt.m_Radius * Lt.m_Radius) continue;
                        if (Lt.m_OuterAngle >= 0 && OutsideCone(Lt, C, S)) continue;
                        Fn(L * level_cells_v + Slot(x, y, z), std::sqrt(D2));
                    }
                }
            };
            for (const auto& Lt : Lights) ForEachCell(Lt, [&](int Cell, float) { ++Count[Cell]; });

            std::uint32_t nEntries = 0;
            m_Data.assign(nCells * header_uints_v, 0);
            for (int c = 0; c < nCells; ++c) { m_Data[c * header_uints_v] = nEntries; m_Data[c * header_uints_v + 1] = Count[c]; nEntries += Count[c]; }

            const std::uint32_t EntryBase = nCells * header_uints_v;
            m_Data.resize(EntryBase + 2 * nEntries, 0);
            std::vector<std::uint32_t> Fill(nCells, 0);
            std::vector<float>         Importance(nEntries, 0.0f);
            for (std::uint32_t i = 0; i < Lights.size(); ++i)
                ForEachCell(Lights[i], [&](int Cell, float Dist)
                {
                    const std::uint32_t e = m_Data[Cell * header_uints_v] + Fill[Cell]++;
                    m_Data[EntryBase + 2 * e] = i;
                    const float s = Dist / Lights[i].m_Radius;                                         // the shader's falloff at the
                    Importance[e] = Lights[i].m_Power * (1.0f - s * s * s * s) / (1.0f + 2.0f * s * s);   // cell's nearest point: an upper bound
                });

            if (HeadCount > 0) for (int c = 0; c < nCells; ++c) SplitCell(c, HeadCount, Importance);
        }

        // A spot light's cone misses the cell's bounding sphere: the light is exactly 0 there (the shader's smoothstep is 0 at
        // and beyond the outer angle), so leaving it out keeps the lists complete
        static bool OutsideCone(const light& Lt, const int C[3], float S) noexcept
        {
            const xmath::fvec3 Center{ (C[0] + 0.5f) * S, (C[1] + 0.5f) * S, (C[2] + 0.5f) * S };
            const xmath::fvec3 V = Center - Lt.m_Position;
            const float d = V.Length(), rc = S * 0.8660254f;
            if (d <= rc) return false;
            const float Angle = std::acos(std::clamp(V.Dot(Lt.m_Direction) / d, -1.0f, 1.0f));
            return Angle - std::asin(rc / d) > Lt.m_OuterAngle;
        }

        int Slot(int x, int y, int z) const noexcept
        {
            const auto M = [](int v) { return ((v % cells_v) + cells_v) % cells_v; };
            return (M(z) * cells_v + M(y)) * cells_v + M(x);
        }

        std::vector<std::uint32_t>              m_Data;
        std::array<std::array<int, 3>, levels_v> m_Origin{};        // per level: the window's first cell index
        float                                   m_CellSize = 1;

    private:
        // Head: the HeadCount most important, back in light order; tail: the rest, with cumulative probabilities (floored)
        void SplitCell(int Cell, int HeadCount, const std::vector<float>& Importance) noexcept
        {
            const std::uint32_t First = m_Data[Cell * header_uints_v], n = m_Data[Cell * header_uints_v + 1];
            const std::uint32_t EntryBase = levels_v * level_cells_v * header_uints_v;
            if (n == 0) return;
            if (int(n) <= HeadCount) { m_Data[Cell * header_uints_v + 2] = n; return; }

            struct item { std::uint32_t m_Light; float m_I; };
            std::vector<item> Items(n);
            for (std::uint32_t k = 0; k < n; ++k) Items[k] = { m_Data[EntryBase + 2 * (First + k)], Importance[First + k] };
            std::stable_sort(Items.begin(), Items.end(), [](const item& a, const item& b) { return a.m_I > b.m_I; });
            std::sort(Items.begin(), Items.begin() + HeadCount, [](const item& a, const item& b) { return a.m_Light < b.m_Light; });
            std::sort(Items.begin() + HeadCount, Items.end(),   [](const item& a, const item& b) { return a.m_Light < b.m_Light; });

            float MaxTail = 0;
            for (std::uint32_t k = HeadCount; k < n; ++k) MaxTail = std::max(MaxTail, Items[k].m_I);
            const float Floor = std::max(MaxTail * 0.01f, 1e-12f);     // nonzero support for every tail light
            float Sum = 0;
            for (std::uint32_t k = HeadCount; k < n; ++k) Sum += std::max(Items[k].m_I, Floor);

            float Cumulative = 0;
            for (std::uint32_t k = 0; k < n; ++k)
            {
                m_Data[EntryBase + 2 * (First + k)] = Items[k].m_Light;
                if (k < std::uint32_t(HeadCount)) { m_Data[EntryBase + 2 * (First + k) + 1] = 0; continue; }
                Cumulative += std::max(Items[k].m_I, Floor) / Sum;
                m_Data[EntryBase + 2 * (First + k) + 1] = std::bit_cast<std::uint32_t>(k + 1 == n ? 1.0f : Cumulative);
            }
            m_Data[Cell * header_uints_v + 2] = std::uint32_t(HeadCount);
        }
    };
}

#endif
