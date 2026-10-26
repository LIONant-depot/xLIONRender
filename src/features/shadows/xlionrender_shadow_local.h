#ifndef XLIONRENDER_SHADOW_LOCAL_H
#define XLIONRENDER_SHADOW_LOCAL_H
#pragma once

// Point and spot light shadows: the face projections and helpers the local lights' virtual pages share
// (features/shadows/xlionrender_shadow_local_pages.h: the pages, the pool and the table; shaders/xlionrender_local_lights.glsl and
// xlionrender_local_request_comp.glsl build the same face basis and projection). The octahedral maps in a buddy atlas, rendered as
// cube faces into a scratch and remapped, were retired by P1d (documentation/lighting_strategy.md).

#include "scene/xlionrender_scene.h"
#include <algorithm>
#include <array>
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

    constexpr int urgent_age_v   = 4;           // frames a page waits before it goes ahead of coarser ones (local_pages)

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
}

#endif
