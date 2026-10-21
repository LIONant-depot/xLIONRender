#ifndef XLIONRENDER_SCENE_COMPONENTS_H
#define XLIONRENDER_SCENE_COMPONENTS_H
#pragma once

// The components a Level draws with through xlionrender::instance (the forward renderer: shadows, probes, GI): a Mesh and its
// MeshMaterial, the lights, the Environment. Registered and consumed entirely inside LIONRender.dll, like the Primitive and the Text (xlionrender_plugin_entry.cpp).
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

    static constexpr auto gi_role_v = std::array
    { xproperty::settings::enum_item("ASSET",    gi_role::ASSET)
    , xproperty::settings::enum_item("RECEIVER", gi_role::RECEIVER)
    , xproperty::settings::enum_item("FULL",     gi_role::FULL)
    };

    // A static geometry drawn with its materials (which ones, and the entity's own values for them: its MeshMaterial)
    struct mesh
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "Mesh" };

        geom_static_ref                     m_Geom;
        std::vector<material_instance_ref>  m_Materials;    // DEPRECATED: slot swaps live in MeshMaterial::m_Slots (which wins per slot); still read from old files and drawn
        gi_role                             m_GIRole = gi_role::ASSET;

        // ponytail: old files keep their swaps here, shown and saved only while there are any. Moving them into a MeshMaterial at load
        // needs an archetype change in the scene loader (xECSV2 xecs_scene_inline.h ReadEntityFile); do it there if the old field must go

        XPROPERTY_DEF
        ( "Mesh", mesh
        , obj_member<"Geom",      &mesh::m_Geom,      member_help<"The GeomStatic resource to draw">>
        , obj_member<"Materials", &mesh::m_Materials, member_dynamic_flags<+[](const mesh& M) { xproperty::flags::type F{}; F.m_bDontShow = F.m_bDontSave = M.m_Materials.empty(); return F; }>, member_help<"Deprecated (old levels): use the MeshMaterial component's Slots, which win per slot. Material instances per material slot of the geometry, replacing the geometry's own">>
        , obj_member<"GIRole",    &mesh::m_GIRole,    member_enum_span<gi_role_v>, member_help<"What it does in the GI: ASSET as its geometry says; RECEIVER lit by it but neither blocking nor bouncing light (a prop, a character: moving it costs the GI nothing elsewhere); FULL blocks, bounces and emits">>
        )
    };

    // One entity value of a MeshMaterial (linear; a color's alpha in W): with Override on, it replaces the material's value for this
    // entity wherever the material lets that field override a parameter (plugins/xmaterial.plugin/documentation/material_contract.md section 4)
    struct entity_value
    {
        bool    m_bOverride = false;
        float   m_X = 0, m_Y = 0, m_Z = 0, m_W = 0;

        XPROPERTY_DEF
        ( "EntityValue", entity_value
        , obj_member<"Override", &entity_value::m_bOverride, member_help<"On: this value replaces the material instance's for this entity; off: the material instance's value">>
        , obj_member<"X",        &entity_value::m_X>
        , obj_member<"Y",        &entity_value::m_Y>
        , obj_member<"Z",        &entity_value::m_Z>
        , obj_member<"W",        &entity_value::m_W>
        )
    };
    XPROPERTY_REG(entity_value)

    // What a Mesh entity draws with: per material slot a material instance replacing the geometry's own, and the entity's own values for
    // the parameters its materials open to entities (v1 schema: a Tint, two generic params). Without one: the geometry's materials with their
    // own values. The values reach the shaders through the draw's instance slot (draw::m_EntityParams)
    struct mesh_material
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "MeshMaterial" };

        std::vector<material_instance_ref>  m_Slots;        // per material slot of the geometry; an empty entry (or a missing slot): the geometry's own
        entity_value                        m_Tint      { .m_X = 1, .m_Y = 1, .m_Z = 1, .m_W = 1 };
        entity_value                        m_Param0, m_Param1;

        XPROPERTY_DEF
        ( "MeshMaterial", mesh_material
        , obj_member<"Slots",  &mesh_material::m_Slots,  member_help<"Material instances per material slot of the Mesh's geometry, replacing the geometry's own (leave an entry empty to keep it)">>
        , obj_member<"Tint",   &mesh_material::m_Tint,   member_help<"The entity's tint (linear rgba): replaces the parameter its materials expose as the entity Tint">>
        , obj_member<"Param0", &mesh_material::m_Param0, member_help<"The entity's first generic value: replaces the parameter its materials expose as entity Param0">>
        , obj_member<"Param1", &mesh_material::m_Param1, member_help<"The entity's second generic value: replaces the parameter its materials expose as entity Param1">>
        )

        entity_params Params() const noexcept
        {
            entity_params P;
            const auto Field = [&](const entity_value& V, std::uint32_t Bit, xmath::fvec4& Out) { if (V.m_bOverride) { P.m_Mask |= Bit; Out = { V.m_X, V.m_Y, V.m_Z, V.m_W }; } };
            Field(m_Tint,   entity_params::TINT,   P.m_Tint);
            Field(m_Param0, entity_params::PARAM0, P.m_Param0);
            Field(m_Param1, entity_params::PARAM1, P.m_Param1);
            return P;
        }
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
    , xproperty::settings::enum_item("COVERAGE", gi_debug::COVERAGE)
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
        , obj_member<"GIDebug",      &environment::m_GIDebug, member_enum_span<gi_debug_v>, member_help<"SURFELS: every pixel in the color of the surfel it reads most (black: no surfel reaches it); LIGHT: the GI light alone; SURFACE_COLOR: the surface color of the surfel each pixel reads most (the material capture: what the GI bounces); COVERAGE: which read lights each pixel - green its own surfels (per pixel), blue the vertices' (per pixel unsupported there), magenta none (no GI there: the ambient as without GI)">>
        , obj_member<"GIPerPixel",   &environment::m_bGIPerPixel, member_help<"Near meshes read the GI per pixel (a mesh needs no extra vertices for it); off: per vertex only (cheaper, needs dense meshes)">>
        )
    };

    // After the Primitive (50) and the Text
    XSCRIPT_REGISTER_COMPONENT(mesh,        "Rendering", 52)
    XSCRIPT_REGISTER_COMPONENT(light_sun,   "Rendering", 53)
    XSCRIPT_REGISTER_COMPONENT(light_point, "Rendering", 54)
    XSCRIPT_REGISTER_COMPONENT(light_spot,  "Rendering", 55)
    XSCRIPT_REGISTER_COMPONENT(environment, "Rendering", 56)
    XSCRIPT_REGISTER_COMPONENT(mesh_material, "Rendering", 57)
}

#endif
