#ifndef XLIONRENDER_TEXT_H
#define XLIONRENDER_TEXT_H
#pragma once

// The Text component: flat text drawn from the atlas of a Font resource, placed in the 3D world by the Transform of its entity. Registered and consumed entirely inside LIONRender.dll (see
// xlionrender_plugin_entry.cpp), like the Primitive.
//
// What belongs where:
//   the Font resource   the atlas, the glyph metrics, the kerning, and how the atlas is encoded (MTSDF, SDF or BITMAP): the component never says
//   this component      what to write and how: the text, the font, the size, the color, the alignment and wrapping, how it is placed in the world
//   the Transform       where it is (position, rotation, scale)
//   the renderer        the laid-out glyphs (xlionrender_text_layout.h) and what is sent to the GPU: derived data, rebuilt when the text, the font, the size, the wrap or the spacing change
//
// Units: Size is the size of one em. In WORLD mode an em is Size world units before the entity's scale; in SCREEN mode it is Size pixels of the viewport and the entity's scale is ignored (only
// its position places the text). WrapWidth and the layout use the same units as Size.
#include "dependencies/xECSV2/src/xecs.h"
#include "dependencies/xmath/source/xmath.h"
#include "plugins/xscript_module.plugin/source/Runtime/xscript_registration.h"

#include <string>

namespace xlionrender
{
    // The Font resource: the type guid is the one of the resource plugin (xfont.plugin), derived the same way every plugin derives its own - the render DLL does not include the plugin's loader.
    inline constexpr auto font_type_guid_v = xresource::type_guid(xresource::guid_generator::Instance64FromString("font"));
    using font_ref = xresource::def_guid<font_type_guid_v>;

    enum class text_halign      : std::uint8_t { LEFT, CENTER, RIGHT };
    enum class text_vanchor     : std::uint8_t { TOP, MIDDLE, BOTTOM, BASELINE };
    enum class text_orientation : std::uint8_t { ENTITY, CAMERA, UPRIGHT_CAMERA };
    enum class text_size_mode   : std::uint8_t { WORLD, SCREEN };
    enum class text_depth_mode  : std::uint8_t { DEPTH_TESTED, OVERLAY };

    inline constexpr auto text_halign_list_v = std::array
    { xproperty::settings::enum_item{ "Left",   text_halign::LEFT   }
    , xproperty::settings::enum_item{ "Center", text_halign::CENTER }
    , xproperty::settings::enum_item{ "Right",  text_halign::RIGHT  }
    };

    inline constexpr auto text_vanchor_list_v = std::array
    { xproperty::settings::enum_item{ "Top",      text_vanchor::TOP      }
    , xproperty::settings::enum_item{ "Middle",   text_vanchor::MIDDLE   }
    , xproperty::settings::enum_item{ "Bottom",   text_vanchor::BOTTOM   }
    , xproperty::settings::enum_item{ "Baseline", text_vanchor::BASELINE }
    };

    inline constexpr auto text_orientation_list_v = std::array
    { xproperty::settings::enum_item{ "Entity",         text_orientation::ENTITY         }
    , xproperty::settings::enum_item{ "Camera",         text_orientation::CAMERA         }
    , xproperty::settings::enum_item{ "Upright Camera", text_orientation::UPRIGHT_CAMERA }
    };

    inline constexpr auto text_size_mode_list_v = std::array
    { xproperty::settings::enum_item{ "World",  text_size_mode::WORLD  }
    , xproperty::settings::enum_item{ "Screen", text_size_mode::SCREEN }
    };

    inline constexpr auto text_depth_mode_list_v = std::array
    { xproperty::settings::enum_item{ "Depth Tested", text_depth_mode::DEPTH_TESTED }
    , xproperty::settings::enum_item{ "Overlay",      text_depth_mode::OVERLAY      }
    };

    struct text
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "Text" };

        font_ref            m_Font;
        std::wstring        m_Text          = L"Text";
        float               m_Size          = 0.25f;
        xmath::fvec3        m_Color         = xmath::fvec3::fromOne();
        float               m_Opacity       = 1.0f;
        text_halign         m_HAlign        = text_halign::CENTER;
        text_vanchor        m_VAnchor       = text_vanchor::MIDDLE;
        float               m_WrapWidth     = 0.0f;
        float               m_LineSpacing   = 1.0f;
        text_orientation    m_Orientation   = text_orientation::ENTITY;
        text_size_mode      m_SizeMode      = text_size_mode::WORLD;
        text_depth_mode     m_DepthMode     = text_depth_mode::DEPTH_TESTED;
        bool                m_bSort         = false;

        XPROPERTY_DEF
        ( "Text", text
        , obj_member<"Font",        &text::m_Font,                                              member_help<"The Font resource the text is drawn from. How its atlas is encoded (MTSDF, SDF or bitmap) is the font's own: changing the font never changes this component.">>
        , obj_member<"Text",        &text::m_Text,          member_flags<flags::MULTILINE>,        member_help<"What is written: Unicode text, with line breaks (Enter in the box). A character the font does not have is drawn as the font's replacement glyph.">>
        , obj_member<"Size",        &text::m_Size,                                              member_help<"The size of one em. World mode: world units before the entity's scale. Screen mode: pixels of the viewport (the entity's scale is ignored).">>
        , obj_member<"Color",       &text::m_Color>
        , obj_member<"Opacity",     &text::m_Opacity,                                           member_help<"0 is invisible, 1 is solid.">>
        , obj_member<"HAlign",      &text::m_HAlign,        member_enum_span<text_halign_list_v>,  member_help<"How the lines sit in the width of the text. The origin of the entity is at the left edge, the middle or the right edge of the text to match.">>
        , obj_member<"VAnchor",     &text::m_VAnchor,       member_enum_span<text_vanchor_list_v>, member_help<"Which part of the text block is at the origin of the entity: its top, its middle, its bottom, or the baseline of its first line.">>
        , obj_member<"WrapWidth",   &text::m_WrapWidth,                                         member_help<"Lines are broken at spaces to fit this width (in the units of Size). 0 breaks only at the line breaks of the text.">>
        , obj_member<"LineSpacing", &text::m_LineSpacing,                                       member_help<"A multiplier of the line height of the font: 1 is what the font asks for.">>
        , obj_member<"Orientation", &text::m_Orientation,   member_enum_span<text_orientation_list_v>, member_help<"Entity: it faces where the entity faces. Camera: it always faces the camera. Upright Camera: it faces the camera but stays upright (turns around the vertical axis only).">>
        , obj_member<"SizeMode",    &text::m_SizeMode,      member_enum_span<text_size_mode_list_v>, member_help<"World: the text has a size in the world and gets smaller with distance. Screen: the text has the same size in pixels wherever it is.">>
        , obj_member<"DepthMode",   &text::m_DepthMode,     member_enum_span<text_depth_mode_list_v>, member_help<"Depth Tested: things in front of the text hide it. Overlay: it is always drawn over the scene.">>
        , obj_member<"Sort",        &text::m_bSort,                                             member_help<"Sort this text with the other transparent things by its distance to the camera. Off by default: most text is in front of something solid and does not need it.">>
        )
    };
    // After the Primitive (50): Transform stays at the top of the inspector.
    XSCRIPT_REGISTER_COMPONENT(text, "Rendering", 55)
}

#endif // XLIONRENDER_TEXT_H
