#ifndef XLIONRENDER_SHADOW_CLIPMAP_H
#define XLIONRENDER_SHADOW_CLIPMAP_H
#pragma once

// The sun's shadows were a clipmap of world-anchored tiles (4 levels of 4 x 4 tiles around the camera) until the virtual pages replaced
// it (features/shadows/xlionrender_shadow_vsm.h; documentation/lighting_strategy.md P1a-P1e; the clipmap went 2026-10-27). What is left:
// the atlas sizes the pages' pool and the scene's sun tile still use, and the light-space box the pages cull their casters with.

#include "dependencies/xmath/source/xmath.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xlionrender::shadows
{
    constexpr int atlas_tiles_v         = 8;            // the sun's part of the atlas, per clipmap width: 8 x 8 tiles of setup::m_ClipmapTileSize texels
    constexpr int clipmap_levels_v      = 4;            // the scene's sun tile (scene::sun_light::m_ShadowTileSize) is sized as the clipmap's level 0
    constexpr int clipmap_tiles_v       = 4;            // tile was (level 0's texel x 4 tiles x 2^3 levels: the reach the pages' levels must cover)

    inline int PositiveMod(int A, int B) noexcept { return ((A % B) + B) % B; }

    // A light-space axis aligned box (x, y across the light, z along it)
    struct lbox
    {
        float m_Min[3] = {  1e30f,  1e30f,  1e30f };
        float m_Max[3] = { -1e30f, -1e30f, -1e30f };
        void Add(const xmath::fvec3& P) noexcept { const float V[3] = { P.m_X, P.m_Y, P.m_Z }; for (int i = 0; i < 3; ++i) { m_Min[i] = std::min(m_Min[i], V[i]); m_Max[i] = std::max(m_Max[i], V[i]); } }
    };

    // A light-space box against a square rect of light space (a page)
    inline bool RectOverlaps(const lbox& B, float MinX, float MinY, float Size) noexcept
    {
        return B.m_Max[0] >= MinX && B.m_Min[0] <= MinX + Size && B.m_Max[1] >= MinY && B.m_Min[1] <= MinY + Size;
    }
}

#endif
