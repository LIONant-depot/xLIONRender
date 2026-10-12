#ifndef XLIONRENDER_SCENE_H
#define XLIONRENDER_SCENE_H
#pragma once

// The snapshot a tool or the game hands to the renderer: plain data, no GPU objects, no ECS.
// Meshes are GPU resources owned by an xlionrender::instance and referenced here by id.

#include "dependencies/xmath/source/xmath.h"
#include <cstdint>
#include <vector>
#include <optional>

namespace xlionrender
{
    // Vertex layout of meshes created through instance::CreateMesh. Matches the mb_* varying interface
    // (position, UV, color, normal, tangent with the binormal sign in w).
    struct vertex
    {
        float           m_Pos[3];
        float           m_UV[2];
        std::uint32_t   m_Color;
        float           m_Normal[3];
        float           m_Tangent[4];
    };

    // Generation-checked ids: a destroyed slot can be reused without old ids reaching the new object
    struct mesh_id
    {
        std::uint32_t   m_Index         = ~0u;
        std::uint32_t   m_Generation    = 0;
        bool isValid() const noexcept { return m_Index != ~0u; }
    };

    // A compiled material (its HDR shader + textures), see instance::CreateMaterial
    struct material_id
    {
        std::uint32_t   m_Index         = ~0u;
        std::uint32_t   m_Generation    = 0;
        bool isValid() const noexcept { return m_Index != ~0u; }
    };

    // Standard metal/roughness material (glTF-style), scene-linear values
    struct material
    {
        xmath::fvec3    m_BaseColor     { 0.8f, 0.8f, 0.8f };
        float           m_Metallic      = 0.0f;
        float           m_Roughness     = 0.5f;
        float           m_AO            = 1.0f;
        float           m_F0            = 0.04f;        // dielectric reflectance
        xmath::fvec3    m_Emissive      { 0, 0, 0 };
    };

    struct draw
    {
        mesh_id                     m_Mesh;
        xmath::fmat4                m_L2W           = xmath::fmat4::fromIdentity();
        material                    m_Material;         // used where m_Materials has no valid entry for a material slot
        std::vector<material_id>    m_Materials;        // per material slot of the mesh (xgeom_static submesh m_iMaterial)
        std::uint64_t               m_UserID        = 0;    // e.g. the entity; what editor pick/ID passes write
    };

    // The directional light (the sun): shades everything, shadowed by the shadow clipmap
    struct sun_light
    {
        xmath::fvec3    m_Direction     { -0.5f, -1.0f, -0.3f };    // the direction the light travels
        float           m_Temperature   = 0;                        // Kelvin tint, 0 = none
        xmath::fvec3    m_Color         { 1, 1, 1 };
        float           m_Intensity     = 4.0f;                     // ponytail: unitless until the physical-unit contract (research doc §6) is approved
        bool            m_bCastShadows  = true;
        float           m_ShadowTileSize = 1.0f;                    // world size of a clipmap level-0 tile (512 texels); the levels double it
    };

    // A point or spot light (up to max_local_lights_v per scene, the nearest first is up to the caller); shadows from atlas tiles:
    // 6 for a point light (a cube), 1 for a spot
    struct local_light
    {
        enum class type : std::uint8_t { POINT, SPOT };

        type            m_Type          = type::POINT;
        xmath::fvec3    m_Position      { 0, 0, 0 };
        xmath::fvec3    m_Direction     { 0, -1, 0 };               // SPOT: where it points
        float           m_Radius        = 5.0f;                     // falloff radius (zero light at and beyond it)
        float           m_InnerAngle    = 0.4f;                     // SPOT: radians from the axis: full light inside, fading to the outer
        float           m_OuterAngle    = 0.6f;                     // SPOT: radians (at most ~1 rad: a 115 degree shadow frustum)
        xmath::fvec3    m_Color         { 1, 1, 1 };
        float           m_Intensity     = 4.0f;
        bool            m_bCastShadows  = true;
    };
    constexpr int max_local_lights_v = 32;

    struct scene
    {
        std::vector<draw>           m_Draws;
        sun_light                   m_Sun;
        std::vector<local_light>    m_LocalLights;
        xmath::fvec3                m_Ambient   { 0, 0, 0 };
    };
}

#endif
