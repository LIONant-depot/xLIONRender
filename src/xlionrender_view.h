#ifndef XLIONRENDER_VIEW_H
#define XLIONRENDER_VIEW_H
#pragma once

#include <cstdint>

namespace xlionrender
{
    // The view that asks (Draw, Pick). Every view shows the game: an entity with the runtime tags (xlioncore::no_render_tag) is in none of them, and an entity with an exclusive disable tag (the game's or the
    // editor's) is in none either (no system sees it). What differs is the editor's own state: the SCENE view (the editor's viewport) leaves out what the editor hid (xecs::editor::no_render_tag), the GAME view
    // shows the game as it is.
    enum class view : std::uint8_t { SCENE, GAME };
}

#endif
