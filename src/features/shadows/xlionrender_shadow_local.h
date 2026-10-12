#ifndef XLIONRENDER_SHADOW_LOCAL_H
#define XLIONRENDER_SHADOW_LOCAL_H
#pragma once

// Point and spot light shadows: tiles of the shadow atlas's right half (the clipmap has the left), 512^2 each. A point light takes 6
// consecutive tiles (one per cube face, perspective 90 degrees), a spot light 1 (its cone). A light keeps its tiles while it exists and
// does not move; they re-render when a caster changes inside its radius (conservative: all its faces). A light that moves is a new light
// (new tiles). No room left: that light has no shadow this frame.
//
// The shader (shaders/xlionrender_local_lights.glsl) finds a face and its texels with the same basis and projection as FaceW2C.
// ponytail: fixed 512 tiles and all faces of a light at once; tiles sized by screen importance and per-face invalidation when lights
// get numerous (Godot sizes its atlas quadrants that way).

#include "scene/xlionrender_scene.h"
#include "features/shadows/xlionrender_shadow_clipmap.h"
#include <cmath>
#include <vector>

namespace xlionrender::shadows
{
    constexpr int local_column_v    = atlas_tiles_v;                        // the local lights' half starts at this atlas column (in tiles)
    constexpr int local_tiles_v     = atlas_tiles_v * atlas_tiles_v;        // tiles in that half (8 x 8)
    constexpr int atlas_width_v     = atlas_size_v * 2;                     // the atlas: the clipmap's half + the local lights' half

    // The atlas texel origin of a local tile
    inline int LocalTileX(int Tile) noexcept { return (local_column_v + Tile % atlas_tiles_v) * tile_size_v; }
    inline int LocalTileY(int Tile) noexcept { return (Tile / atlas_tiles_v) * tile_size_v; }

    // The basis of a face looking along F (the shader builds the same one)
    inline void FaceBasis(const xmath::fvec3& F, xmath::fvec3& R, xmath::fvec3& U) noexcept
    {
        const xmath::fvec3 Up = std::abs(F.m_Y) < 0.99f ? xmath::fvec3(0, 1, 0) : xmath::fvec3(1, 0, 0);
        R = xmath::fvec3::Cross(Up, F).NormalizeCopy();
        U = xmath::fvec3::Cross(F, R);
    }

    inline xmath::fvec3 CubeFaceAxis(int Face) noexcept
    {
        constexpr float A[6][3] = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };
        return { A[Face][0], A[Face][1], A[Face][2] };
    }

    inline float LocalNear(float Radius) noexcept { return std::max(Radius * 0.01f, 0.005f); }

    // World -> clip of a face: perspective along F with a half field of view of atan(TanHalf), depth [Near, Radius] -> [0, 1]
    inline xmath::fmat4 FaceW2C(const xmath::fvec3& Pos, const xmath::fvec3& F, float TanHalf, float Near, float Far) noexcept
    {
        xmath::fvec3 R, U;
        FaceBasis(F, R, U);
        const float A = Far / (Far - Near), B = -Far * Near / (Far - Near);
        xmath::fmat4 M = xmath::fmat4::fromZero();
        const xmath::fvec3 Rows[3] = { R * (1.0f / TanHalf), U * (1.0f / TanHalf), F };
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

    struct local_tile_to_render
    {
        int             m_Tile;         // in the local half of the atlas
        int             m_Light;        // index in the scene's local lights
        xmath::fmat4    m_W2C;
    };

    struct local_shadows
    {
        // FirstTile[i]: the first atlas tile of local light i, -1 when it has none. Out: the tiles to render this frame.
        void Update(const std::vector<local_light>& Lights, const std::vector<xmath::fbbox>& Changed, bool bCache, std::vector<int>& FirstTile, std::vector<local_tile_to_render>& Out) noexcept
        {
            Out.clear();
            FirstTile.assign(Lights.size(), -1);
            for (auto& A : m_Allocs) A.m_bUsed = false;

            std::vector<bool> bRender(Lights.size(), false);
            for (std::size_t i = 0; i < Lights.size() && int(i) < max_local_lights_v; ++i)
            {
                const auto& L = Lights[i];
                if (!L.m_bCastShadows) continue;

                const std::uint64_t Key = LightKey(L);
                alloc* pA = nullptr;
                for (auto& A : m_Allocs) if (!A.m_bUsed && A.m_Key == Key) { pA = &A; break; }
                if (pA == nullptr) continue;            // allocated below, after every reused light kept its tiles
                pA->m_bUsed  = true;
                FirstTile[i] = pA->m_First;
                bRender[i]   = !bCache;
                for (std::size_t c = 0; !bRender[i] && c < Changed.size(); ++c) bRender[i] = Overlaps(Changed[c], L.m_Position, L.m_Radius);
            }

            // Tiles of lights that are gone or moved go back; new lights take free runs
            std::erase_if(m_Allocs, [&](const alloc& A) { if (!A.m_bUsed) for (int t = 0; t < A.m_Count; ++t) m_TileUsed[A.m_First + t] = false; return !A.m_bUsed; });
            for (std::size_t i = 0; i < Lights.size() && int(i) < max_local_lights_v; ++i)
            {
                const auto& L = Lights[i];
                if (!L.m_bCastShadows || FirstTile[i] >= 0) continue;
                const int Count = L.m_Type == local_light::type::POINT ? 6 : 1;
                const int First = FindRun(Count);
                if (First < 0) continue;                // ponytail: no room, no shadow (sized tiles / an LRU when lights get numerous)
                for (int t = 0; t < Count; ++t) m_TileUsed[First + t] = true;
                m_Allocs.push_back({ LightKey(L), First, Count, true });
                FirstTile[i] = First;
                bRender[i]   = true;
            }

            for (std::size_t i = 0; i < Lights.size(); ++i)
            {
                if (!bRender[i] || FirstTile[i] < 0) continue;
                const auto& L    = Lights[i];
                const float Near = LocalNear(L.m_Radius);
                if (L.m_Type == local_light::type::POINT)
                    for (int f = 0; f < 6; ++f) Out.push_back({ FirstTile[i] + f, int(i), FaceW2C(L.m_Position, CubeFaceAxis(f), 1.0f, Near, L.m_Radius) });
                else
                    Out.push_back({ FirstTile[i], int(i), FaceW2C(L.m_Position, L.m_Direction.NormalizeCopy(), std::tan(std::min(L.m_OuterAngle, 1.0f)), Near, L.m_Radius) });
            }
        }

    private:
        struct alloc
        {
            std::uint64_t   m_Key;
            int             m_First, m_Count;
            bool            m_bUsed;
        };

        static std::uint64_t LightKey(const local_light& L) noexcept
        {
            const float V[] = { float(L.m_Type), L.m_Position.m_X, L.m_Position.m_Y, L.m_Position.m_Z, L.m_Radius
                              , L.m_Type == local_light::type::SPOT ? L.m_Direction.m_X : 0, L.m_Type == local_light::type::SPOT ? L.m_Direction.m_Y : 0
                              , L.m_Type == local_light::type::SPOT ? L.m_Direction.m_Z : 0, L.m_Type == local_light::type::SPOT ? L.m_OuterAngle : 0 };
            std::uint64_t H = 1469598103934665603ull;
            for (auto* p = reinterpret_cast<const std::uint8_t*>(V); p != reinterpret_cast<const std::uint8_t*>(V + std::size(V)); ++p) H = (H ^ *p) * 1099511628211ull;
            return H;
        }

        int FindRun(int Count) const noexcept
        {
            for (int First = 0; First + Count <= local_tiles_v; ++First)
            {
                int t = 0;
                while (t < Count && !m_TileUsed[First + t]) ++t;
                if (t == Count) return First;
            }
            return -1;
        }

        std::vector<alloc>                   m_Allocs;
        std::array<bool, local_tiles_v>      m_TileUsed{};
    };
}

#endif
