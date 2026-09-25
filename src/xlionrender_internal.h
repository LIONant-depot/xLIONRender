#ifndef XLIONRENDER_INTERNAL_H
#define XLIONRENDER_INTERNAL_H
#pragma once

// Declares the one seam between the ECS-facing system (xlionrender_system.h) and the DLL's own
// renderer singleton (defined in xlionrender_api.cpp). Not exported - no dllexport/dllimport macro -
// both sides of this call are always compiled into the same DLL (LIONRender.dll), so ordinary internal
// linkage is enough. "No one submits shapes except internally to this DLL" is literally what this is.
#include "xlionrender_primitive.h"
#include "dependencies/xmath/source/xmath.h"
#include <cstdint>

namespace xlionrender
{
    // EntityValue (xecs::component::entity::m_Value) - not touched by SubmitInternal itself, just
    // carried through to the renderer's draw list so Draw can find the selected item again later.
    void SubmitInternal(shape Shape, const xmath::fmat4& L2W, const xmath::fvec3& Color, std::uint64_t EntityValue) noexcept;

    // The live render system (set in its OnCreate, cleared in OnDestroy) - Draw asks it to collect.
    struct system;
    void SetActiveSystemInternal(system* pSystem) noexcept;
}

#endif // XLIONRENDER_INTERNAL_H
