#ifndef XLIONRENDER_SHADOW_CLIPMAP_H
#define XLIONRENDER_SHADOW_CLIPMAP_H
#pragma once

// The directional light's shadows: a clipmap of world-anchored, cached tiles in the shadow atlas (research doc: shadows).
//
//  - Light space: a fixed rotation looking along the light. Its xy plane is cut into square tiles; level L's tiles are 2^L times
//    the size of level 0's. Each level keeps the clipmap_tiles_v x clipmap_tiles_v tiles around the camera.
//  - A tile is anchored in the world (its integer xy index in light space), so it stays valid while the camera moves; its atlas
//    slot is its index mod clipmap_tiles_v (toroidal), so no allocator and no page table: the shader computes the same slot.
//  - Depth: each level has a depth range around the camera with headroom, snapped to a coarse grid (it changes - and its tiles
//    invalidate - only when the camera leaves the margin). Casters nearer to the light than the range still cast: the shadow
//    pipelines clamp depth ("pancaking", as Unreal's virtual shadow maps).
//  - Invalidation (conservative, research doc Track A): the renderer passes the world bounds of every caster that appeared,
//    disappeared or changed (old and new; skinned casters every frame); every tile under them re-renders. A new light direction
//    invalidates everything.
//
// ponytail: the tiles touched by a moving caster re-render whole (all casters); Unreal keeps static and dynamic depth apart
// and re-draws only the movers over a copy of the static tile - add when moving characters make tiles re-render all the time.

#include "dependencies/xmath/source/xmath.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xlionrender::shadows
{
    constexpr int atlas_tiles_v         = 8;                            // the clipmap's part of the atlas: 8x8 tiles (setup::m_ClipmapTileSize texels each)
    constexpr int clipmap_levels_v      = 4;
    constexpr int clipmap_tiles_v       = 4;                            // tiles per level side; a level's block in the atlas is 4x4
    constexpr float depth_scale_v       = 2.0f;                         // a level's depth range: +-depth_scale_v x its extent, around the camera

    static_assert(clipmap_levels_v * clipmap_tiles_v * clipmap_tiles_v <= atlas_tiles_v * atlas_tiles_v);

    // Where level L's 4x4 block sits in the atlas (in tiles)
    inline int LevelAtlasX(int Level) noexcept { return (Level % (atlas_tiles_v / clipmap_tiles_v)) * clipmap_tiles_v; }
    inline int LevelAtlasY(int Level) noexcept { return (Level / (atlas_tiles_v / clipmap_tiles_v)) * clipmap_tiles_v; }
    inline int PositiveMod(int A, int B)  noexcept { return ((A % B) + B) % B; }

    // A light-space axis aligned box (x, y across the light, z along it)
    struct lbox
    {
        float m_Min[3] = {  1e30f,  1e30f,  1e30f };
        float m_Max[3] = { -1e30f, -1e30f, -1e30f };
        void Add(const xmath::fvec3& P) noexcept { const float V[3] = { P.m_X, P.m_Y, P.m_Z }; for (int i = 0; i < 3; ++i) { m_Min[i] = std::min(m_Min[i], V[i]); m_Max[i] = std::max(m_Max[i], V[i]); } }
    };

    struct tile_to_render
    {
        int             m_Level, m_X, m_Y;          // the tile's world index (light space)
        int             m_AtlasX, m_AtlasY;         // its slot in the atlas (in tiles)
        xmath::fmat4    m_W2C;                      // world -> the tile's clip space
        float           m_MinX, m_MinY, m_Size;     // its light-space rect (for culling the casters)
    };

    struct clipmap
    {
        struct level
        {
            int     m_OriginX = 0, m_OriginY = 0;    // the first tile index of the 4x4 window around the camera
            float   m_TileSize = 1;                  // world units per tile side
            float   m_DepthCenter = 0;               // light-space z the depth range is centered on (snapped)
        };

        struct slot
        {
            bool    m_bValid = false;
            int     m_X = 0, m_Y = 0;               // the world tile it holds
            float   m_DepthCenter = 0;              // with which depth range
        };

        // Called once per frame. Returns (in Out) the tiles that must be rendered: the ones that came into the window, whose depth
        // range moved, or that a changed caster (Changed: world bounds) touched. bCache = false re-renders every tile (the reference
        // to compare against).
        void Update(const xmath::fvec3& LightDir, float Level0TileSize, const xmath::fvec3& Camera, const std::vector<xmath::fbbox>& ChangedWorld, bool bCache, std::vector<tile_to_render>& Out) noexcept
        {
            Out.clear();

            // Light basis; a new direction (or scale) invalidates everything
            const xmath::fvec3 Z = LightDir.NormalizeCopy();
            if (!bCache || !m_bHasBasis || Z.Dot(m_Z) < 0.99999f || Level0TileSize != m_Level0TileSize)
            {
                const xmath::fvec3 Up = std::abs(Z.m_Y) < 0.99f ? xmath::fvec3(0, 1, 0) : xmath::fvec3(1, 0, 0);
                m_X = xmath::fvec3::Cross(Up, Z).NormalizeCopy();
                m_Y = xmath::fvec3::Cross(Z, m_X);
                m_Z = Z;
                m_Level0TileSize = Level0TileSize;
                m_bHasBasis = true;
                for (auto& S : m_Slots) S.m_bValid = false;
            }

            std::vector<lbox> Changed;
            Changed.reserve(ChangedWorld.size());
            for (auto& B : ChangedWorld) Changed.push_back(Bounds(B));

            const xmath::fvec3 C = ToLight(Camera);
            for (int L = 0; L < clipmap_levels_v; ++L)
            {
                auto& Lv = m_Levels[L];
                Lv.m_TileSize    = Level0TileSize * float(1 << L);
                Lv.m_OriginX     = int(std::floor(C.m_X / Lv.m_TileSize)) - clipmap_tiles_v / 2;
                Lv.m_OriginY     = int(std::floor(C.m_Y / Lv.m_TileSize)) - clipmap_tiles_v / 2;

                // Re-center the depth range only when the camera leaves its middle half
                const float Half = DepthHalfRange(L);
                if (std::abs(C.m_Z - Lv.m_DepthCenter) > Half * 0.5f || !m_bDepthSet[L])
                {
                    Lv.m_DepthCenter = std::round(C.m_Z / (Half * 0.5f)) * (Half * 0.5f);
                    m_bDepthSet[L] = true;
                }

                for (int ty = 0; ty < clipmap_tiles_v; ++ty)
                for (int tx = 0; tx < clipmap_tiles_v; ++tx)
                {
                    const int X = Lv.m_OriginX + tx, Y = Lv.m_OriginY + ty;
                    const int AX = LevelAtlasX(L) + PositiveMod(X, clipmap_tiles_v), AY = LevelAtlasY(L) + PositiveMod(Y, clipmap_tiles_v);
                    auto& S = m_Slots[AY * atlas_tiles_v + AX];

                    const float MinX = X * Lv.m_TileSize, MinY = Y * Lv.m_TileSize;
                    bool bRender = !bCache || !S.m_bValid || S.m_X != X || S.m_Y != Y || S.m_DepthCenter != Lv.m_DepthCenter;
                    for (std::size_t i = 0; !bRender && i < Changed.size(); ++i) bRender = Overlaps(Changed[i], MinX, MinY, Lv.m_TileSize);
                    if (!bRender) continue;

                    S = { true, X, Y, Lv.m_DepthCenter };
                    Out.push_back({ L, X, Y, AX, AY, TileW2C(L, MinX, MinY), MinX, MinY, Lv.m_TileSize });
                }
            }
        }

        // Every tile renders on the next Update: what changed while no Update ran (the sun not casting) never reached the tiles
        void Invalidate(void) noexcept { m_bHasBasis = false; }

        xmath::fvec3 ToLight(const xmath::fvec3& P) const noexcept { return { P.Dot(m_X), P.Dot(m_Y), P.Dot(m_Z) }; }

        // Light-space bounds of a world box
        lbox Bounds(const xmath::fbbox& World) const noexcept
        {
            lbox B;
            for (int i = 0; i < 8; ++i)
                B.Add(ToLight({ (i & 1) ? World.m_Max.m_X : World.m_Min.m_X, (i & 2) ? World.m_Max.m_Y : World.m_Min.m_Y, (i & 4) ? World.m_Max.m_Z : World.m_Min.m_Z }));
            return B;
        }

        static bool Overlaps(const lbox& B, float MinX, float MinY, float Size) noexcept
        {
            return B.m_Max[0] >= MinX && B.m_Min[0] <= MinX + Size && B.m_Max[1] >= MinY && B.m_Min[1] <= MinY + Size;
        }

        float DepthHalfRange(int Level) const noexcept { return m_Level0TileSize * float(1 << Level) * clipmap_tiles_v * depth_scale_v; }

        // World -> clip of one tile: x, y over the tile's light-space rect to [-1,1], z over the level's depth range to [0,1]
        xmath::fmat4 TileW2C(int Level, float MinX, float MinY) const noexcept
        {
            const auto& Lv   = m_Levels[Level];
            const float Half = DepthHalfRange(Level);
            const float SX = 2.0f / Lv.m_TileSize, SZ = 1.0f / (2.0f * Half);
            const float ZMin = Lv.m_DepthCenter - Half;
            xmath::fmat4 M = xmath::fmat4::fromZero();
            M(0, 0) = m_X.m_X * SX; M(0, 1) = m_X.m_Y * SX; M(0, 2) = m_X.m_Z * SX; M(0, 3) = -MinX * SX - 1.0f;
            M(1, 0) = m_Y.m_X * SX; M(1, 1) = m_Y.m_Y * SX; M(1, 2) = m_Y.m_Z * SX; M(1, 3) = -MinY * SX - 1.0f;
            M(2, 0) = m_Z.m_X * SZ; M(2, 1) = m_Z.m_Y * SZ; M(2, 2) = m_Z.m_Z * SZ; M(2, 3) = -ZMin * SZ;
            M(3, 3) = 1.0f;
            return M;
        }

        // Rows of the world -> light rotation (for the shader)
        xmath::fvec3                                     m_X{ 1, 0, 0 }, m_Y{ 0, 1, 0 }, m_Z{ 0, 0, 1 };
        std::array<level, clipmap_levels_v>              m_Levels{};

    private:
        std::array<slot, atlas_tiles_v * atlas_tiles_v>  m_Slots{};
        std::array<bool, clipmap_levels_v>               m_bDepthSet{};
        float                                            m_Level0TileSize = 0;
        bool                                             m_bHasBasis = false;
    };
}

#endif
