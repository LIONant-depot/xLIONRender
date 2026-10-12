#ifndef XLIONRENDER_FRUSTUM_H
#define XLIONRENDER_FRUSTUM_H
#pragma once

// A view frustum from a world -> clip matrix (Gribb & Hartmann), in the xgpu::view conventions: -w <= x, y <= w and
// 0 <= z <= w (depth 0 at near, 1 at far). Overlaps is conservative: a box it rejects is entirely outside one plane;
// a box near a corner may pass and be drawn for nothing.

#include "dependencies/xmath/source/xmath.h"
#include <array>

namespace xlionrender
{
    struct frustum
    {
        std::array<xmath::fvec4, 6> m_Planes;       // xyz normal (pointing inside), w distance: inside when dot(n, p) + w >= 0

        static frustum FromW2C(const xmath::fmat4& M) noexcept
        {
            const auto Row = [&](int r) { return xmath::fvec4(M(r, 0), M(r, 1), M(r, 2), M(r, 3)); };
            const xmath::fvec4 R0 = Row(0), R1 = Row(1), R2 = Row(2), R3 = Row(3);
            return { { R3 + R0, R3 - R0, R3 + R1, R3 - R1, R2, R3 - R2 } };   // left, right, bottom, top, near, far
        }

        // bNear = false: no near plane - for shadow pancaking, where casters in front of the near plane still cast (depth clamp)
        bool Overlaps(const xmath::fbbox& B, bool bNear = true) const noexcept
        {
            for (int i = 0; i < 6; ++i)
            {
                if (i == 4 && !bNear) continue;
                const auto& P = m_Planes[i];
                // the box's corner furthest along the plane's normal
                const float X = P.m_X >= 0 ? B.m_Max.m_X : B.m_Min.m_X;
                const float Y = P.m_Y >= 0 ? B.m_Max.m_Y : B.m_Min.m_Y;
                const float Z = P.m_Z >= 0 ? B.m_Max.m_Z : B.m_Min.m_Z;
                if (P.m_X * X + P.m_Y * Y + P.m_Z * Z + P.m_W < 0) return false;
            }
            return true;
        }
    };
}

#endif
