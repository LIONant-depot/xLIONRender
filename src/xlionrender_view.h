#ifndef XLIONRENDER_VIEW_H
#define XLIONRENDER_VIEW_H
#pragma once

#include <algorithm>
#include <cstdint>

namespace xlionrender
{
    // The view that asks (Draw, Pick). Every view shows the game: an entity with the runtime tags (xlioncore::no_render_tag) is in none of them, and an entity with an exclusive disable tag (the game's or the
    // editor's) is in none either (no system sees it). What differs is the editor's own state: the SCENE view (the editor's viewport) leaves out what the editor hid (xecs::editor::no_render_tag), the GAME view
    // shows the game as it is.
    enum class view : std::uint8_t { SCENE, GAME };

    // An editor's viewing choices for a world's RenderScene, on top of its Environment's GI settings: what the viewport shows, never
    // saved (the Level is not changed). Version 7.
    struct view_options
    {
        std::int8_t m_GIDebug     = -1;         // -1: the Environment's GIDebug; else a scene::gi_debug (0 NONE, 1 SURFELS, 2 LIGHT, 4 SURFACE_COLOR, 5 COVERAGE)
        bool        m_bNoGI       = false;      // drawn without GI, whatever the Environment says
        bool        m_bNoPerPixel = false;      // the vertex read only (the Environment's GIPerPixel off)
        bool        m_bNoMoments  = false;      // the reads without the depth-moment visibility test (the Environment's GIMoments off)
        float       m_GITraceAllowanceMs = -1;  // < 0: the scene's own; else this frame's grant of the host's shared allowance (version 9)
        // Huge meshes (version 10): the Meshes are not drawn, only their boxes (xRenderEditor::NotDrawnBoxes); or, with a frame budget (> 0, ms), only as many of their
        // clusters as the frame time allows, the biggest on the screen first, the others as their boxes (the allowance shrinks while the frames are slower than the budget
        // and grows back as they are faster)
        bool        m_bBoxesOnly    = false;
        float       m_FrameBudgetMs = 0;
        // GPU cluster culling (version 12, documentation/gpu_cluster_culling.md): off (the CPU path), its occlusion off (the frustum only),
        // frozen (the last cull's lists drawn from wherever the camera goes: what was culled shows as missing)
        bool        m_bNoGPUCull      = false;
        bool        m_bNoGPUOcclusion = false;
        bool        m_bFreezeCull     = false;
        bool operator==(const view_options&) const noexcept = default;
    };

    // A box a render did not draw the inside of (view_options::m_bBoxesOnly, m_FrameBudgetMs): a Mesh or one of its clusters, as the unit cube (-0.5 .. 0.5) placed in the
    // world (the rows of the affine matrix), and the entity it is of (raw value). Version 10.
    struct wire_box
    {
        float           m_Rows[3][4];
        std::uint64_t   m_Entity;
    };

    // The roles of the entities of a world in an editing session (documentation/Editors/prefabs_plan.md, 3.7 and phase 7). The host knows which scene an entity is in, the render does not: it is told which
    // entities (their raw runtime values, sorted ascending) are CONTEXT - drawn first, faded by a full screen quad, never picked, so that what is not being edited is there to see and in the way of nothing -
    // and which are HIDDEN (the instance that is being edited in context: its prefab is the document, drawn in its place). Everything else is the document. No roles (null): every entity is the document.
    struct roles
    {
        const std::uint64_t*    m_pContext  = nullptr;
        int                     m_nContext  = 0;
        const std::uint64_t*    m_pHidden   = nullptr;
        int                     m_nHidden   = 0;
        float                   m_FadeAlpha = 0.65f;                    // how much of the background color is put over the context (0: not faded)
        float                   m_FadeColor[3] = { 0.15f, 0.15f, 0.15f };

        bool IsContext(std::uint64_t Entity) const noexcept { return m_nContext > 0 && std::binary_search(m_pContext, m_pContext + m_nContext, Entity); }
        bool IsHidden (std::uint64_t Entity) const noexcept { return m_nHidden  > 0 && std::binary_search(m_pHidden,  m_pHidden  + m_nHidden,  Entity); }
    };
}

#endif
