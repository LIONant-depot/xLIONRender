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
    // pWorld: the xecs::game_mgr::instance whose entities to draw (each open Level has its own).
    XLIONRENDER_API void Draw (const void* pWorld, xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH) noexcept;

    // CPU ray-pick against every rendered entity's transform box (xeditor_tools::picking, shared
    // with xskeleton.plugin's own bone picking) - closest hit wins. MaxT caps the ray so a closer
    // ground/grid hit can occlude entities behind the floor without a GPU ID buffer. Returns
    // xecs::component::entity::invalid_entity_v (0xFFFFFFFFFFFFFFFF) as raw m_Value on a miss, so the
    // host never needs to include xecs.h just to call this; it already keys its own scene entity maps
    // (m_RuntimeToLocal) by this same raw value.
    XLIONRENDER_API std::uint64_t Pick(const void* pWorld, const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT = std::numeric_limits<float>::max()) noexcept;

    // The entity (raw xecs::component::entity::m_Value, or invalid_entity_v for none) to draw with an
    // outline this frame - the host calls this once per frame with its own current selection.
    XLIONRENDER_API void SetSelectedEntity(std::uint64_t EntityValue) noexcept;

    // The same four calls as a pure virtual interface that a COPY of this DLL hands out (xlionrender_editor.cpp, factory CreateEditorName): the editor calls the copy that belongs to the copy of the core
    // its world is in, found by the name of that module (GetProcAddress), instead of importing these functions - an import is bound to the one DLL the exe was linked with. Everything that crosses is
    // the host's (xgpu, xmath) or a world address / raw entity value, never an xecs type. Release() frees it.
    struct xRenderEditor
    {
        static constexpr std::uint32_t kVersion = 2;
        virtual std::uint32_t Version() const noexcept = 0;
        virtual void          Release() noexcept = 0;
        virtual bool          Init(xgpu::device& Device) noexcept = 0;
        virtual void          Draw(const void* pWorld, xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH) noexcept = 0;
        virtual std::uint64_t Pick(const void* pWorld, const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT) noexcept = 0;
        virtual void          SetSelectedEntity(std::uint64_t EntityValue) noexcept = 0;
        // What the layout of the Text of an entity is (lines, glyphs, the box of the text, or why there is none yet: the font is not loaded), as lines of text written to pOut (at most Capacity
        // bytes, zero ended). Returns the length written, or -1 when the entity has no Text. Version 2.
        virtual int           DescribeText(const void* pWorld, std::uint64_t EntityValue, char* pOut, int Capacity) noexcept = 0;
        // What the last draw of the Texts did (labels and glyphs sent, draw calls made, labels dropped for lack of room), written like DescribeText. Returns the length written. Version 2.
        virtual int           DescribeTextDraw(char* pOut, int Capacity) noexcept = 0;
    };

    constexpr const char* kCreateEditorName = "XLionRender_CreateEditor";
    using pfn_create_editor = xRenderEditor* (*)() noexcept;
}

#endif // XLIONRENDER_API_H
