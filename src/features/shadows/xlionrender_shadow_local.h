#ifndef XLIONRENDER_SHADOW_LOCAL_H
#define XLIONRENDER_SHADOW_LOCAL_H
#pragma once

// Point and spot light shadows (research doc: shadows phase 3). One shadow map per light in the local part of the atlas (16 bits of
// linear distance to the light, divided by its radius):
//  - a point light's map is octahedral: the whole sphere of directions in one square tile (down at the center, the seams up);
//    a spot light's is its perspective frustum. Both have a 2 texel border (the octahedral one folded over its edges) for PCF.
//  - Maps are rendered in two steps: the casters' depth into a scratch D32F (a point light's 6 cube faces at half the map size,
//    a spot's frustum at the map size), then a remap pass writes the map: cube -> octahedral, hyperbolic depth -> linear distance.
//  - Map sizes are powers of two following the light's size on screen (with hysteresis), allocated by a quadtree buddy allocator.
//  - Caching as the clipmap: a map re-renders when a changed caster's bounds touch the light's sphere. A light that moves or
//    changes size gets a new map. Re-renders go through a per-frame budget (lights without a map first, then maps waiting
//    for urgent_age_v frames or more - the longest first -, then by screen size); past it a map keeps its last render (an
//    allowed approximation, not a cache hit). A light whose map was never rendered is unshadowed.
//
// The shaders (shaders/xlionrender_local_lights.glsl, xlionrender_shadow_remap_frag.glsl) share the face basis, the octahedral
// mapping (xlionrender_octahedral.glsl) and the border with this file.
// ponytail: the remap takes the nearest cube texel (not the min of its footprint), and a light that does not fit gets no shadow
// (an LRU / shrinking the far lights when the atlas fills up).

#include "scene/xlionrender_scene.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xlionrender::shadows
{
    // Per platform sizes (instance::Init)
    struct setup
    {
        int     m_ClipmapTileSize   = 512;          // texels per clipmap tile; the clipmap is 8x8 tiles
        int     m_LocalAtlasSize    = 4096;         // the local lights' square part of the atlas
        int     m_MinLocalSize      = 64;           // a light's shadow map size range
        int     m_MaxLocalSize      = 1024;
        int     m_ScratchSize       = 2048;         // the D32F the maps are rendered in before the remap
    };

    constexpr int local_border_v = 2;
    constexpr int urgent_age_v   = 4;           // frames a dirty map waits before it goes ahead of bigger ones

    // A quadtree buddy allocator of power-of-two squares in a square
    struct buddy
    {
        struct pos { int m_X, m_Y; };

        void Reset(int Size, int Min) noexcept
        {
            m_Min = Min;
            m_Top = std::countr_zero(unsigned(Size / Min));
            m_Free.assign(m_Top + 1, {});
            m_Free[m_Top].push_back({ 0, 0 });
        }

        bool Alloc(int Size, pos& Out) noexcept
        {
            const int L = Level(Size);
            int k = L;
            while (k <= m_Top && m_Free[k].empty()) ++k;
            if (k > m_Top) return false;
            Out = m_Free[k].back();
            m_Free[k].pop_back();
            while (k > L)
            {
                --k;
                const int H = m_Min << k;
                m_Free[k].push_back({ Out.m_X + H, Out.m_Y });
                m_Free[k].push_back({ Out.m_X, Out.m_Y + H });
                m_Free[k].push_back({ Out.m_X + H, Out.m_Y + H });
            }
            return true;
        }

        void Free(pos P, int Size) noexcept
        {
            int L = Level(Size);
            while (L < m_Top)
            {
                const int  S = m_Min << L, PX = P.m_X / (2 * S) * (2 * S), PY = P.m_Y / (2 * S) * (2 * S);
                const auto IsSibling = [&](const pos& Q) { return (Q.m_X == PX || Q.m_X == PX + S) && (Q.m_Y == PY || Q.m_Y == PY + S); };
                auto& F = m_Free[L];
                if (std::count_if(F.begin(), F.end(), IsSibling) != 3) break;
                std::erase_if(F, IsSibling);
                P = { PX, PY };
                ++L;
            }
            m_Free[L].push_back(P);
        }

    private:
        int Level(int Size) const noexcept { return std::countr_zero(unsigned(Size / m_Min)); }

        std::vector<std::vector<pos>>   m_Free;
        int                             m_Min = 1, m_Top = 0;
    };

    // The basis of a face looking along F (the shaders build the same one)
    inline void FaceBasis(const xmath::fvec3& F, xmath::fvec3& R, xmath::fvec3& U) noexcept
    {
        const xmath::fvec3 Up = std::abs(F.m_Y) < 0.99f ? xmath::fvec3(0, 1, 0) : xmath::fvec3(1, 0, 0);
        R = xmath::fvec3::Cross(Up, F).NormalizeCopy();
        U = xmath::fvec3::Cross(F, R);
    }

    // Cube faces in this order: +X -X +Y -Y +Z -Z (the remap shader picks them the same way)
    inline xmath::fvec3 CubeFaceAxis(int Face) noexcept
    {
        constexpr float A[6][3] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
        return { A[Face][0], A[Face][1], A[Face][2] };
    }

    inline float LocalNear(float Radius) noexcept { return std::max(Radius * 0.01f, 0.005f); }
    inline float SpotTanHalf(const local_light& L) noexcept { return std::tan(std::min(L.m_OuterAngle, 1.0f)); }

    // World -> clip of a face: perspective along F with a half field of view of atan(TanHalf), depth [Near, Far] -> [0, 1]
    inline xmath::fmat4 FaceW2C(const xmath::fvec3& Pos, const xmath::fvec3& F, float TanHalf, float Near, float Far) noexcept
    {
        xmath::fvec3 R, U;
        FaceBasis(F, R, U);
        const float A = Far / (Far - Near), B = -Far * Near / (Far - Near);
        xmath::fmat4 M = xmath::fmat4::fromZero();
        const xmath::fvec3 Rows[2] = { R * (1.0f / TanHalf), U * (1.0f / TanHalf) };
        for (int c = 0; c < 3; ++c) { M(0, c) = Rows[0][c]; M(1, c) = Rows[1][c]; M(2, c) = F[c] * A; M(3, c) = F[c]; }
        M(0, 3) = -Rows[0].Dot(Pos);
        M(1, 3) = -Rows[1].Dot(Pos);
        M(2, 3) = -F.Dot(Pos) * A + B;
        M(3, 3) = -F.Dot(Pos);
        return M;
    }

    inline bool Overlaps(const xmath::fbbox& Box, const xmath::fvec3& Center, float Radius) noexcept
    {
        float D2 = 0;
        for (int i = 0; i < 3; ++i)
        {
            const float V = Center[i], Lo = Box.m_Min[i], Hi = Box.m_Max[i];
            if (V < Lo) D2 += (Lo - V) * (Lo - V);
            else if (V > Hi) D2 += (V - Hi) * (V - Hi);
        }
        return D2 <= Radius * Radius;
    }

    // A light's map in the local part of the atlas (texels, relative to that part); m_Size 0 = none
    struct local_map
    {
        int     m_X = 0, m_Y = 0, m_Size = 0;
    };

    struct local_map_to_render
    {
        int         m_Light;                // index in the scene's local lights
        local_map   m_Map;
    };

    struct local_shadows
    {
        void Init(const setup& Setup) noexcept
        {
            m_Setup = Setup;
            m_Atlas.Reset(Setup.m_LocalAtlasSize, Setup.m_MinLocalSize);
            m_Allocs.clear();
        }

        // Maps[i]: local light i's map to sample (m_Size 0: unshadowed). Out: the maps to render this frame, at most Budget of them
        // (bCache = false: all, no budget). Eye, ProjScale (pixels per unit at distance 1): for the maps' sizes and priorities.
        void Update(const std::vector<local_light>& Lights, const std::vector<xmath::fbbox>& Changed, bool bCache, const xmath::fvec3& Eye, float ProjScale
                   , int Budget, std::vector<local_map>& Maps, std::vector<local_map_to_render>& Out) noexcept
        {
            Out.clear();
            ++m_Frame;
            const int nLights = std::min(int(Lights.size()), max_local_lights_v);
            Maps.assign(Lights.size(), {});
            for (auto& A : m_Allocs) { A.m_bUsed = false; A.m_iLight = -1; }

            // Keep the maps of the lights that did not move or change size; mark the ones a changed caster touched
            std::vector<float> Pixels(nLights, 0.0f);
            std::vector<int>   Wanted(nLights, 0);
            for (int i = 0; i < nLights; ++i)
            {
                const auto& L = Lights[i];
                if (!L.m_bCastShadows || L.m_Radius <= 0) continue;
                const float Dist = (L.m_Position - Eye).Length();
                Pixels[i] = Dist <= L.m_Radius ? 1e9f : 2.0f * L.m_Radius / Dist * ProjScale;

                const std::uint64_t Key = LightKey(L);
                alloc* pA = nullptr;
                for (auto& A : m_Allocs) if (!A.m_bUsed && A.m_Key == Key) { pA = &A; break; }
                if (pA && (Pixels[i] > pA->m_Size * 1.4f || Pixels[i] < pA->m_Size * 0.35f) && SizeFor(Pixels[i]) != pA->m_Size) pA = nullptr;   // hysteresis
                if (pA == nullptr) { Wanted[i] = SizeFor(Pixels[i]); continue; }

                pA->m_bUsed  = true;
                pA->m_iLight = i;
                bool bDirty = !bCache;
                for (std::size_t c = 0; !bDirty && c < Changed.size(); ++c) bDirty = Overlaps(Changed[c], L.m_Position, L.m_Radius);
                if (bDirty && !pA->m_bDirty) { pA->m_bDirty = true; pA->m_DirtyFrame = m_Frame; }
            }

            // Maps of lights that are gone, moved or resized go back; new maps for the others, the biggest on screen first
            std::erase_if(m_Allocs, [&](const alloc& A) { if (!A.m_bUsed) m_Atlas.Free({ A.m_X, A.m_Y }, A.m_Size); return !A.m_bUsed; });
            std::vector<int> New;
            for (int i = 0; i < nLights; ++i) if (Wanted[i]) New.push_back(i);
            std::sort(New.begin(), New.end(), [&](int a, int b) { return Pixels[a] > Pixels[b]; });
            for (int i : New)
            {
                buddy::pos P;
                if (!m_Atlas.Alloc(Wanted[i], P)) continue;
                m_Allocs.push_back({ LightKey(Lights[i]), P.m_X, P.m_Y, Wanted[i], i, true, false, true, m_Frame });
            }

            // Within the budget: maps never rendered first, then the ones waiting urgent_age_v frames or more (the longest
            // first), then by size on screen
            std::vector<alloc*> Dirty;
            for (auto& A : m_Allocs) if (A.m_bDirty) Dirty.push_back(&A);
            const auto Age = [&](const alloc* a) { return m_Frame - a->m_DirtyFrame; };
            std::sort(Dirty.begin(), Dirty.end(), [&](const alloc* a, const alloc* b)
            {
                if (a->m_bValid != b->m_bValid) return !a->m_bValid;
                const bool aUrgent = Age(a) >= urgent_age_v, bUrgent = Age(b) >= urgent_age_v;
                if (aUrgent != bUrgent) return aUrgent;
                if (aUrgent && Age(a) != Age(b)) return Age(a) > Age(b);
                return Pixels[a->m_iLight] > Pixels[b->m_iLight];
            });
            if (bCache && int(Dirty.size()) > Budget) Dirty.resize(std::max(Budget, 0));
            for (auto* pA : Dirty)
            {
                pA->m_bDirty = false;
                pA->m_bValid = true;
                Out.push_back({ pA->m_iLight, { pA->m_X, pA->m_Y, pA->m_Size } });
            }

            for (const auto& A : m_Allocs) if (A.m_bValid) Maps[A.m_iLight] = { A.m_X, A.m_Y, A.m_Size };
        }

    private:
        struct alloc
        {
            std::uint64_t   m_Key;
            int             m_X, m_Y, m_Size;
            int             m_iLight;
            bool            m_bUsed, m_bValid, m_bDirty;
            std::uint64_t   m_DirtyFrame;       // when it became dirty (waiting age)
        };

        int SizeFor(float Pixels) const noexcept
        {
            const float P = std::clamp(Pixels, float(m_Setup.m_MinLocalSize), float(m_Setup.m_MaxLocalSize));
            return int(std::bit_ceil(unsigned(P)));
        }

        static std::uint64_t LightKey(const local_light& L) noexcept
        {
            const bool  bSpot = L.m_Type == local_light::type::SPOT;
            const float V[] = { float(L.m_Type), L.m_Position.m_X, L.m_Position.m_Y, L.m_Position.m_Z, L.m_Radius
                              , bSpot ? L.m_Direction.m_X : 0, bSpot ? L.m_Direction.m_Y : 0, bSpot ? L.m_Direction.m_Z : 0, bSpot ? L.m_OuterAngle : 0 };
            std::uint64_t H = 1469598103934665603ull;
            for (auto* p = reinterpret_cast<const std::uint8_t*>(V); p != reinterpret_cast<const std::uint8_t*>(V + std::size(V)); ++p) H = (H ^ *p) * 1099511628211ull;
            return H;
        }

        setup               m_Setup;
        std::uint64_t       m_Frame = 0;
        buddy               m_Atlas;
        std::vector<alloc>  m_Allocs;
    };
}

#endif
