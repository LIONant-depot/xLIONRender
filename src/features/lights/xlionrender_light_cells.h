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
// Layout (uints, shaders/xlionrender_local_lights.glsl reads the same): per cell (level-major, then slot) one packed header
// uint - bits 0-19 the first entry, 20-27 the count, 28-31 the head count; a count of overflow_v (255) means "more than fits":
// the shader shades that cell with every light (exact, never a dropped light) - then the entries { light index,
// floatBitsToUint(tail cumulative probability) }.
// ponytail: rebuilt from scratch every render on the CPU (the lab measures that cost); dirty cells, the visibility term and
// a compute builder come after the lab says the cells earn their place.

#include "dependencies/xmath/source/xmath.h"
#include <algorithm>
#include <cassert>
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
        static constexpr int header_uints_v = 1;
        static constexpr std::uint32_t overflow_v  = 255;                       // a packed count: the cell's list does not fit
        static constexpr std::uint32_t max_head_v  = 15;
        static constexpr std::uint32_t max_entry_v = (1u << 20) - 1;
        static constexpr std::uint32_t entry_base_v = levels_v * level_cells_v * header_uints_v;     // the entries' first uint

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
            m_First.resize(nCells);
            m_Head.assign(nCells, 0);
            m_Count = Count;
            for (int c = 0; c < nCells; ++c) { m_First[c] = nEntries; nEntries += Count[c]; }
            m_Data.assign(nCells * header_uints_v, 0);

            const std::uint32_t EntryBase = entry_base_v;
            m_Data.resize(EntryBase + 2 * nEntries, 0);
            std::vector<std::uint32_t> Fill(nCells, 0);
            std::vector<float>         Importance(nEntries, 0.0f);
            for (std::uint32_t i = 0; i < Lights.size(); ++i)
                ForEachCell(Lights[i], [&](int Cell, float Dist)
                {
                    const std::uint32_t e = m_First[Cell] + Fill[Cell]++;
                    m_Data[EntryBase + 2 * e] = i;
                    const float s = Dist / Lights[i].m_Radius;                                         // the shader's falloff at the
                    Importance[e] = Lights[i].m_Power * (1.0f - s * s * s * s) / (1.0f + 2.0f * s * s);   // cell's nearest point: an upper bound
                });

            if (HeadCount > 0) for (int c = 0; c < nCells; ++c) SplitCell(c, std::min<int>(HeadCount, max_head_v), Importance);

            // The packed headers (a list too long for the count, or entries past 20 bits: the cell is shaded exactly)
            for (int c = 0; c < nCells; ++c)
            {
                const bool          bOver = m_Count[c] >= overflow_v || m_First[c] + m_Count[c] > max_entry_v;
                const std::uint32_t n     = bOver ? overflow_v : m_Count[c];
                m_Data[c] = (bOver ? 0 : m_First[c]) | (n << 20) | (std::min(m_Head[c], max_head_v) << 28);
            }
        }

        // Boxes (the reflection probes' influence boxes, world) listed per cell of Build's window, appended to m_Data: per cell
        // (level-major, then slot) a packed header uint - bits 0-23 the first entry (a uint index into m_Data), 24-31 the count -,
        // then the entries (box indices, in the given order: the shader keeps it). Returns the headers' first uint in m_Data.
        std::uint32_t AddBoxes(const std::vector<xmath::fbbox>& Boxes) noexcept
        {
            constexpr int nCells = levels_v * level_cells_v;
            assert(Boxes.size() < 256);                                         // a cell's count is 8 bits
            const auto Base = std::uint32_t(m_Data.size());
            const auto ForEachCell = [&](const xmath::fbbox& B, auto&& Fn)
            {
                for (int L = 0; L < levels_v; ++L)
                {
                    const float S = m_CellSize * float(1 << L);
                    int Lo[3], Hi[3];
                    bool bEmpty = false;
                    for (int a = 0; a < 3; ++a)
                    {
                        Lo[a] = std::max(int(std::floor(B.m_Min[a] / S)), m_Origin[L][a]);
                        Hi[a] = std::min(int(std::floor(B.m_Max[a] / S)), m_Origin[L][a] + cells_v - 1);
                        bEmpty |= Lo[a] > Hi[a];
                    }
                    if (bEmpty) continue;
                    for (int z = Lo[2]; z <= Hi[2]; ++z)
                    for (int y = Lo[1]; y <= Hi[1]; ++y)
                    for (int x = Lo[0]; x <= Hi[0]; ++x) Fn(L * level_cells_v + Slot(x, y, z));
                }
            };
            std::vector<std::uint32_t> Count(nCells, 0);
            for (const auto& B : Boxes) ForEachCell(B, [&](int Cell) { ++Count[Cell]; });

            m_Data.resize(Base + nCells);
            std::uint32_t e = Base + nCells;
            for (int c = 0; c < nCells; ++c) { m_Data[Base + c] = e | (Count[c] << 24); e += Count[c]; }
            m_Data.resize(e);
            std::fill(Count.begin(), Count.end(), 0u);
            for (std::uint32_t i = 0; i < Boxes.size(); ++i)
                ForEachCell(Boxes[i], [&](int Cell) { m_Data[(m_Data[Base + Cell] & 0xFFFFFFu) + Count[Cell]++] = i; });
            return Base;
        }

        // A cell's unpacked header (the CPU side: tests, diagnostics)
        std::uint32_t First(int Cell) const noexcept { return m_First[Cell]; }
        std::uint32_t Count(int Cell) const noexcept { return m_Count[Cell]; }
        std::uint32_t Head (int Cell) const noexcept { return m_Head[Cell]; }

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

        std::vector<std::uint32_t>              m_Data;             // what the shader reads: packed headers, then entries
        std::vector<std::uint32_t>              m_First, m_Count, m_Head;   // the headers unpacked
        std::array<std::array<int, 3>, levels_v> m_Origin{};        // per level: the window's first cell index
        float                                   m_CellSize = 1;

    private:
        // Head: the HeadCount most important, back in light order; tail: the rest, with cumulative probabilities (floored)
        void SplitCell(int Cell, int HeadCount, const std::vector<float>& Importance) noexcept
        {
            const std::uint32_t First = m_First[Cell], n = m_Count[Cell];
            const std::uint32_t EntryBase = entry_base_v;
            if (n == 0) return;
            if (int(n) <= HeadCount) { m_Head[Cell] = n; return; }

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
            m_Head[Cell] = std::uint32_t(HeadCount);
        }
    };
}

#endif
