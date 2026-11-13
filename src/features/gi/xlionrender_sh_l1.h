#ifndef XLIONRENDER_SH_L1_H
#define XLIONRENDER_SH_L1_H
#pragma once
// Spherical harmonics, order 1 (bands 0 and 1: 4 coefficients a colour channel), real and orthonormal - the movers' irradiance probes
// (dynamic_objects_gi_plan_2026-11-13_v3.md, milestone A). The radiometric contract:
//  - a probe stores INCIDENT RADIANCE projected on the basis, Monte Carlo over K uniform-sphere rays: L_lm = (4 pi / K) sum_k L(w_k) Y_lm(w_k)
//  - irradiance for a normal n is the clamped-cosine convolution: E(n) = sum_lm A_l L_lm Y_lm(n), A_0 = pi, A_1 = 2 pi / 3
//  - the coefficients live in a fixed WORLD basis: a probe or its object turning changes where the probe is, never its coefficients
// xlionrender_sh_l1.glsl is the GPU copy (same constants); xlionrender_test_sh_l1.cpp checks both conventions on the CPU.
#include <array>
#include <cmath>

namespace xlionrender::sh
{
    inline constexpr float y00_v = 0.282094792f;        // 1 / (2 sqrt(pi))
    inline constexpr float y1_v  = 0.488602512f;        // sqrt(3 / (4 pi))
    inline constexpr float pi_v  = 3.14159265f;
    inline constexpr float a0_v  = pi_v;                // the clamped cosine's band 0 factor
    inline constexpr float a1_v  = 2.0f * pi_v / 3.0f;  // band 1

    // The basis at a unit direction (x, y, z): order Y00, Y1-1 (y), Y10 (z), Y11 (x)
    inline std::array<float, 4> Basis( float x, float y, float z ) noexcept { return { y00_v, y1_v * y, y1_v * z, y1_v * x }; }

    // One colour channel's coefficients
    struct l1 { std::array<float, 4> m_C{}; };

    // A sample of incident radiance L from direction (x, y, z), one of K uniform-sphere samples: adds (4 pi / K) L Y(w)
    inline void AddSample( l1& S, float L, float x, float y, float z, int K ) noexcept
    {
        const auto B = Basis(x, y, z); const float w = 4.0f * pi_v / float(K);
        for( int i = 0; i < 4; ++i ) S.m_C[std::size_t(i)] += w * L * B[std::size_t(i)];
    }

    // Irradiance for the unit normal (nx, ny, nz); never negative (an L1 ringing below zero clamped once, at the end)
    inline float Irradiance( const l1& S, float nx, float ny, float nz ) noexcept
    {
        const auto B = Basis(nx, ny, nz);
        const float E = a0_v * S.m_C[0] * B[0] + a1_v * (S.m_C[1] * B[1] + S.m_C[2] * B[2] + S.m_C[3] * B[3]);
        return E > 0.0f ? E : 0.0f;
    }
}
#endif
