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
#include "dependencies/xGPU/source/xGPU.h"
#include "dependencies/xmath/source/xmath.h"
#include "xlionrender_view.h"
#include "dependencies/xGPU/source/Tools/xgpu_view.h"
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
    XLIONRENDER_API void Draw (const void* pWorld, xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH, view View = view::SCENE, const roles* pRoles = nullptr) noexcept;

    // CPU ray-pick against every rendered entity's transform box (xeditor_tools::picking, shared
    // with xskeleton.plugin's own bone picking) - closest hit wins. MaxT caps the ray so a closer
    // ground/grid hit can occlude entities behind the floor without a GPU ID buffer. Returns
    // xecs::component::entity::invalid_entity_v (0xFFFFFFFFFFFFFFFF) as raw m_Value on a miss, so the
    // host never needs to include xecs.h just to call this; it already keys its own scene entity maps
    // (m_RuntimeToLocal) by this same raw value.
    XLIONRENDER_API std::uint64_t Pick(const void* pWorld, const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT = std::numeric_limits<float>::max(), view View = view::SCENE, const roles* pRoles = nullptr) noexcept;

    // The entity (raw xecs::component::entity::m_Value, or invalid_entity_v for none) to draw with an
    // outline this frame - the host calls this once per frame with its own current selection.
    XLIONRENDER_API void SetSelectedEntity(std::uint64_t EntityValue) noexcept;

    // The same four calls as a pure virtual interface that a COPY of this DLL hands out (xlionrender_editor.cpp, factory CreateEditorName): the editor calls the copy that belongs to the copy of the core
    // its world is in, found by the name of that module (GetProcAddress), instead of importing these functions - an import is bound to the one DLL the exe was linked with. Everything that crosses is
    // the host's (xgpu, xmath) or a world address / raw entity value, never an xecs type. Release() frees it.
    struct xRenderEditor
    {
        static constexpr std::uint32_t kVersion = 7;                // 7: SetViewOptions (the editor's GI view, not saved); 6: DumpGI (the GI's surfels to a file); 5: RenderScene, BlitScene, DescribeScene (the world's Meshes and lights through the forward renderer); 4: DrawRoles, PickRoles, DescribeRoleDraw (editing in context: context scenes drawn faded, never picked); 3: Draw and Pick say which view asks
        virtual std::uint32_t Version() const noexcept = 0;
        virtual void          Release() noexcept = 0;
        virtual bool          Init(xgpu::device& Device) noexcept = 0;
        virtual void          Draw(const void* pWorld, xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH, view View) noexcept = 0;
        virtual std::uint64_t Pick(const void* pWorld, const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT, view View) noexcept = 0;
        virtual void          SetSelectedEntity(std::uint64_t EntityValue) noexcept = 0;
        // What the layout of the Text of an entity is (lines, glyphs, the box of the text, or why there is none yet: the font is not loaded), as lines of text written to pOut (at most Capacity
        // bytes, zero ended). Returns the length written, or -1 when the entity has no Text. Version 2.
        virtual int           DescribeText(const void* pWorld, std::uint64_t EntityValue, char* pOut, int Capacity) noexcept = 0;
        // What the last draw of the Texts did (labels and glyphs sent, draw calls made, labels dropped for lack of room), written like DescribeText. Returns the length written. Version 2.
        virtual int           DescribeTextDraw(char* pOut, int Capacity) noexcept = 0;
        // Draw and Pick with the roles of the entities of the world in this editing session (xlionrender::roles: the context is drawn first and faded, never picked; the hidden are not drawn nor picked). The host
        // owns the arrays for the call. Draw and Pick above are these with no roles. Version 4.
        virtual void          DrawRoles(const void* pWorld, xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH, view View, const roles& Roles) noexcept = 0;
        virtual std::uint64_t PickRoles(const void* pWorld, const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT, view View, const roles& Roles) noexcept = 0;
        // What the last draw did with the roles (items of the context, items of the document, whether the fade was drawn), written like DescribeTextDraw. Returns the length written. Version 4.
        virtual int           DescribeRoleDraw(char* pOut, int Capacity) noexcept = 0;
        // The world's Meshes, lights and Environment through the forward renderer (xlionrender::instance: shadows, probes, GI) into the world's own
        // W x H view, with pOverlay drawn over the tone-mapped image with the scene's depth (the host's grid). Records render passes: call while the
        // frame is built, before the UI is rendered; then BlitScene from a render callback of the viewport. Version 5.
        using overlay_fn = void (*)(xgpu::cmd_buffer& CmdBuffer, void* pUser) noexcept;
        virtual void          RenderScene(const void* pWorld, xgpu::window& Window, xgpu::tools::view& Camera, int W, int H, view View, overlay_fn pOverlay, void* pUser, const roles* pRoles) noexcept = 0;
        virtual void          BlitScene(const void* pWorld, xgpu::cmd_buffer& CmdBuffer) noexcept = 0;
        // What the last RenderScene of the world had (Meshes, drawn, waiting for their resources, material slots on the constant material, lights), written like DescribeTextDraw. Version 5.
        virtual int           DescribeScene(const void* pWorld, char* pOut, int Capacity) noexcept = 0;
        // The GI's surfels of the world's last RenderScene to a CSV file (positions, the solver's light, the shown light, validity, albedo). Returns the surfels written, -1: no GI or no file. Version 6.
        virtual int           DumpGI(const void* pWorld, const char* pPath) noexcept = 0;
        // The world's viewing choices from now on (view_options: the GI view of an editor's viewport, on top of its Environment). Version 7.
        virtual void          SetViewOptions(const void* pWorld, const view_options& Options) noexcept = 0;
    };

    constexpr const char* kCreateEditorName = "XLionRender_CreateEditor";
    using pfn_create_editor = xRenderEditor* (*)() noexcept;
}

#endif // XLIONRENDER_API_H
