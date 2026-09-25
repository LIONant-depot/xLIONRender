#ifndef XLIONRENDER_PRIMITIVE_H
#define XLIONRENDER_PRIMITIVE_H
#pragma once

// The render component: registered and consumed entirely inside LIONRender.dll (see
// xlionrender_plugin_entry.cpp) - xLION.exe never includes this header. Deliberately carries no
// position/rotation of its own - the system queries it together with xlioncore::physics::rigid_body,
// which already has both.
#include "dependencies/xECSV2/src/xecs.h"
#include "dependencies/xmath/source/xmath.h"

namespace xlionrender
{
    enum class shape : std::uint8_t { CUBE, CAPSULE, SPHERE, CYLINDER };

    inline constexpr auto shape_list_v = std::array
    { xproperty::settings::enum_item{ "Cube",    shape::CUBE }
    , xproperty::settings::enum_item{ "Capsule", shape::CAPSULE }
    , xproperty::settings::enum_item{ "Sphere",  shape::SPHERE }
    , xproperty::settings::enum_item{ "Cylinder",shape::CYLINDER }
    };

    struct primitive
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "Primitive" };

        shape        m_Shape = shape::CUBE;
        xmath::fvec3 m_Color = xmath::fvec3::fromOne();

        XPROPERTY_DEF
        ( "Primitive", primitive
        , obj_member<"Shape", &primitive::m_Shape, member_enum_span<shape_list_v>>
        , obj_member<"Color", &primitive::m_Color>
        )
    };
    XPROPERTY_REG(primitive)
}

#endif // XLIONRENDER_PRIMITIVE_H
