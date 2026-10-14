#ifndef XLIONRENDER_SCENE_COMPONENTS_H
#define XLIONRENDER_SCENE_COMPONENTS_H
#pragma once

// The components a Level draws with through xlionrender::instance (the forward renderer: shadows, probes, GI): a Mesh, the lights, the
// Environment. Registered and consumed entirely inside LIONRender.dll, like the Primitive and the Text (xlionrender_plugin_entry.cpp).
//
// What belongs where:
//   the resources      the geometry, its default materials, the textures (GeomStatic, MaterialInstance, Material, Texture)
//   these components   which resource is drawn and how a light shines; the Transform says where (and for the sun and the spots, where to)
//   the renderer       the snapshot of the frame (xlionrender::scene) and everything derived from it (xlionrender_scene_renderer.h)
//
// A light points along its entity's -Y: with no rotation the sun shines straight down and a spot looks at the floor; rotate the entity to aim it.
#include "dependencies/xECSV2/src/xecs.h"
#include "dependencies/xmath/source/xmath.h"
#include "scene/xlionrender_scene.h"         // gi_debug
#include "plugins/xscript_module.plugin/source/Runtime/xscript_registration.h"

#include <vector>

namespace xlionrender
{
    // The resources, by the type guids of their plugins (derived the same way every plugin derives its own: no plugin header here)
    inline constexpr auto geom_static_type_guid_v       = xresource::type_guid(xresource::guid_generator::Instance64FromString("GeomStatic"));
    inline constexpr auto material_instance_type_guid_v = xresource::type_guid(xresource::guid_generator::Instance64FromString("MaterialInstance"));
    inline constexpr auto material_type_guid_v          = xresource::type_guid(xresource::guid_generator::Instance64FromString("Material"));
    inline constexpr auto texture_type_guid_v           = xresource::type_guid(xresource::guid_generator::Instance64FromString("texture"));
    using geom_static_ref       = xresource::def_guid<geom_static_type_guid_v>;
    using material_instance_ref = xresource::def_guid<material_instance_type_guid_v>;

    // A static geometry drawn with its materials
    struct mesh
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "Mesh" };

        geom_static_ref                     m_Geom;
        std::vector<material_instance_ref>  m_Materials;    // per material slot of the geometry; an empty entry (or a missing slot): the geometry's own

        XPROPERTY_DEF
        ( "Mesh", mesh
        , obj_member<"Geom",      &mesh::m_Geom,      member_help<"The GeomStatic resource to draw">>
        , obj_member<"Materials", &mesh::m_Materials, member_help<"Material instances per material slot of the geometry, replacing the geometry's own (leave an entry empty to keep it)">>
        )
    };

    // The directional light: it shines along the entity's -Y; one per level (the first one found)
    struct light_sun
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "LightSun" };

        xmath::fvec3    m_Color         = xmath::fvec3::fromOne();
        float           m_Intensity     = 3.0f;
        bool            m_bCastShadows  = true;
        float           m_ShadowTileSize = 4.0f;           // world size of a level-0 shadow tile; the clipmap levels double it

        XPROPERTY_DEF
        ( "LightSun", light_sun
        , obj_member<"Color",          &light_sun::m_Color>
        , obj_member<"Intensity",      &light_sun::m_Intensity>
        , obj_member<"CastShadows",    &light_sun::m_bCastShadows>
        , obj_member<"ShadowTileSize", &light_sun::m_ShadowTileSize, member_help<"World size of the finest shadow tile: smaller is sharper near the camera">>
        )
    };

    // A point light at the entity's position
    struct light_point
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "LightPoint" };

        xmath::fvec3    m_Color         = xmath::fvec3::fromOne();
        float           m_Intensity     = 4.0f;
        float           m_Radius        = 8.0f;
        bool            m_bCastShadows  = true;

        XPROPERTY_DEF
        ( "LightPoint", light_point
        , obj_member<"Color",       &light_point::m_Color>
        , obj_member<"Intensity",   &light_point::m_Intensity>
        , obj_member<"Radius",      &light_point::m_Radius, member_help<"No light at and beyond this distance">>
        , obj_member<"CastShadows", &light_point::m_bCastShadows>
        )
    };

    // A spot light at the entity's position, looking along its -Y
    struct light_spot
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "LightSpot" };

        xmath::fvec3    m_Color         = xmath::fvec3::fromOne();
        float           m_Intensity     = 4.0f;
        float           m_Radius        = 10.0f;
        float           m_InnerAngle    = 0.4f;            // radians from the axis: full light inside
        float           m_OuterAngle    = 0.6f;            // radians: no light outside (at most ~1)
        bool            m_bCastShadows  = true;

        XPROPERTY_DEF
        ( "LightSpot", light_spot
        , obj_member<"Color",       &light_spot::m_Color>
        , obj_member<"Intensity",   &light_spot::m_Intensity>
        , obj_member<"Radius",      &light_spot::m_Radius>
        , obj_member<"InnerAngle",  &light_spot::m_InnerAngle, member_help<"Radians from the axis inside which the light is full">>
        , obj_member<"OuterAngle",  &light_spot::m_OuterAngle, member_help<"Radians from the axis beyond which there is no light (at most about 1)">>
        , obj_member<"CastShadows", &light_spot::m_bCastShadows>
        )
    };

    static constexpr auto gi_debug_v = std::array
    { xproperty::settings::enum_item("NONE",    gi_debug::NONE)
    , xproperty::settings::enum_item("SURFELS", gi_debug::SURFELS)
    , xproperty::settings::enum_item("LIGHT",   gi_debug::LIGHT)
    , xproperty::settings::enum_item("SURFACE_COLOR", gi_debug::SURFACE_COLOR)
    };

    // The level's sky, ambient and diffuse GI (one per level: the first one found; without one: a dim flat ambient)
    struct environment
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "Environment" };

        xmath::fvec3    m_SkyZenith     { 0.20f, 0.40f, 0.85f };
        xmath::fvec3    m_SkyHorizon    { 0.70f, 0.78f, 0.90f };
        xmath::fvec3    m_SkyGround     { 0.22f, 0.20f, 0.18f };
        float           m_SkyIntensity  = 0.3f;            // 0: no sky (the flat ambient lights instead)
        xmath::fvec3    m_Ambient       { 0.03f, 0.03f, 0.04f };
        bool            m_bGI           = false;           // surfel GI: the meshes with surfels (baked by their GeomStatic compiler) are lit by it
        int             m_GIRayBudget   = 16384;
        bool            m_bGIPerPixel   = true;            // near, every pixel reads the surfels (no extra vertices needed); off: the vertices only
        gi_debug        m_GIDebug       = gi_debug::NONE;
        float           m_GIEmissiveScale = 100.0f;         // self-illumination as lights: their emission x this, for the GI
        int             m_GIRaysPerSurfel = 8;              // rays per surfel update
        int             m_GIFilterPasses  = 4;              // the GI light's spatial filter passes (smoother with more)
        bool            m_bGIMoments      = true;           // the reads' depth-moment visibility test

        XPROPERTY_DEF
        ( "Environment", environment
        , obj_member<"SkyZenith",    &environment::m_SkyZenith>
        , obj_member<"SkyHorizon",   &environment::m_SkyHorizon>
        , obj_member<"SkyGround",    &environment::m_SkyGround>
        , obj_member<"SkyIntensity", &environment::m_SkyIntensity, member_help<"0 turns the sky off: the flat ambient lights the scene instead">>
        , obj_member<"Ambient",      &environment::m_Ambient>
        , obj_member<"GI",           &environment::m_bGI,         member_help<"Diffuse global illumination (surfels): bounce light from every mesh that has surfels">>
        , obj_member<"GIRayBudget",  &environment::m_GIRayBudget, member_help<"Rays traced per frame for the GI: more converges faster, costs more CPU">>
        , obj_member<"GIRaysPerSurfel", &environment::m_GIRaysPerSurfel, member_help<"Rays each surfel casts per update, spread evenly over its hemisphere: more is less noise (blotches on fine geometry), but the ray budget then updates fewer surfels per frame (raise GIRayBudget with it)">>
        , obj_member<"GIMoments", &environment::m_bGIMoments, member_help<"The GI reads' visibility test (each surfel's depth moments: a surfel hidden from the point is faded). Off: only the plane test - more light leaks, but no blotches from it (a diagnostic)">>
        , obj_member<"GIFilterPasses", &environment::m_GIFilterPasses, member_help<"How many times the GI light is blended with the neighboring surfels' before it is shown: more is smoother (blotches fade, contrast too); 0: none. The solve itself is not filtered">>
        , obj_member<"GIEmissiveScale", &environment::m_GIEmissiveScale, member_help<"Self-illuminated materials light the scene through the GI with their emission times this (0: they light nothing; their own look does not change)">>
        , obj_member<"GIDebug",      &environment::m_GIDebug, member_enum_span<gi_debug_v>, member_help<"SURFELS: every pixel in the color of the surfel it reads most (black: no surfel reaches it); LIGHT: the GI light alone; SURFACE_COLOR: the surface color of the surfel each pixel reads most (the material capture: what the GI bounces)">>
        , obj_member<"GIPerPixel",   &environment::m_bGIPerPixel, member_help<"Near meshes read the GI per pixel (a mesh needs no extra vertices for it); off: per vertex only (cheaper, needs dense meshes)">>
        )
    };

    // After the Primitive (50) and the Text
    XSCRIPT_REGISTER_COMPONENT(mesh,        "Rendering", 52)
    XSCRIPT_REGISTER_COMPONENT(light_sun,   "Rendering", 53)
    XSCRIPT_REGISTER_COMPONENT(light_point, "Rendering", 54)
    XSCRIPT_REGISTER_COMPONENT(light_spot,  "Rendering", 55)
    XSCRIPT_REGISTER_COMPONENT(environment, "Rendering", 56)
}

#endif
