#ifndef XLIONRENDER_API_H
#define XLIONRENDER_API_H
#pragma once

// The only two functions xLION.exe calls on this DLL directly (ordinary compile-time import-lib
// linking, not GetProcAddress - these have a fixed, always-present signature, unlike the ECS
// component/system set the self-registration entry points in xlionrender_plugin_entry.cpp handle).
// Init/Draw are the host's own turn: Init sets up the device once; Draw fires once per frame, from
// the host's own render callback, after the ECS's turn (system::OnUpdate, compiled into this same
// DLL) already collected what to draw via the renderer's internal Submit - see
// xlionrender_renderer.h's own comment for that split. Named XLIONRENDER_API (not XECS_API - that
// name is specific to the ECS's own cross-DLL symbols, this is a different, render-specific boundary).
#include "dependencies/xGPU/source/xgpu.h"
#include "dependencies/xmath/source/xmath.h"
#include <cstdint>
#include <limits>

#if defined(XLIONRENDER_BUILD_SHARED)
    #if defined(XLIONRENDER_EXPORTS)
        #define XLIONRENDER_API __declspec(dllexport)
    #else
        #define XLIONRENDER_API __declspec(dllimport)
    #endif
#else
    #define XLIONRENDER_API
#endif

namespace xlionrender
{
    XLIONRENDER_API bool Init (xgpu::device& Device) noexcept;

    // ViewportW/H (pixels) size the selected entity's outline width in screen space - see
    // xlionrender_renderer.h's own comment on the outline pass.
    XLIONRENDER_API void Draw (xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH) noexcept;

    // CPU ray-pick against every rendered entity's rigid_body AABB (xeditor_tools::picking, shared
    // with xskeleton.plugin's own bone picking) - closest hit wins. MaxT caps the ray so a closer
    // ground/grid hit can occlude entities behind the floor without a GPU ID buffer. Returns
    // xecs::component::entity::invalid_entity_v (0xFFFFFFFFFFFFFFFF) as raw m_Value on a miss, so the
    // host never needs to include xecs.h just to call this; it already keys its own scene entity maps
    // (m_RuntimeToLocal) by this same raw value.
    XLIONRENDER_API std::uint64_t Pick(const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT = std::numeric_limits<float>::max()) noexcept;

    // The entity (raw xecs::component::entity::m_Value, or invalid_entity_v for none) to draw with an
    // outline this frame - the host calls this once per frame with its own current selection.
    XLIONRENDER_API void SetSelectedEntity(std::uint64_t EntityValue) noexcept;
}

#endif // XLIONRENDER_API_H
