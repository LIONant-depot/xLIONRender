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
