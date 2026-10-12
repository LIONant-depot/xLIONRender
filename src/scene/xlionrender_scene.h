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

    // The one light of the first slice (mb_standard_pbr.frag shades one): directional when m_Radius == 0, else a point light
    struct light
    {
        xmath::fvec3    m_Direction     { -0.5f, -1.0f, -0.3f };    // directional: the direction the light travels
        xmath::fvec3    m_Position      { 0, 0, 0 };                // point: world position
        float           m_Radius        = 0;                        // point: falloff radius (0 = directional)
        float           m_AreaRadius    = 0;                        // point: softens the falloff and highlight
        float           m_Temperature   = 0;                        // Kelvin tint, 0 = none
        xmath::fvec3    m_Color         { 1, 1, 1 };
        float           m_Intensity     = 4.0f;                     // ponytail: unitless until the physical-unit contract (research doc §6) is approved
        bool            m_bCastShadows  = true;                     // directional: the shadow clipmap (point lights: phase 2)
        float           m_ShadowTileSize = 1.0f;                    // directional: world size of a clipmap level-0 tile (512 texels); the levels double it
    };

    struct scene
    {
        std::vector<draw>           m_Draws;
        light                       m_Light;
        xmath::fvec3                m_Ambient   { 0, 0, 0 };
    };
}

#endif
