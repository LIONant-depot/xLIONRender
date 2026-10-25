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
#include "dependencies/xLIONCore/src/resources/xlioncore_resources_api.h"     // the Mesh's slots and their parameters, as loaded (the inspector's rows)
#include "plugins/xmaterial.plugin/source/xmaterial_xgpu_rsc_loader.h"
#include "plugins/xmaterial_instance.plugin/source/xmaterial_instance_xgpu_rsc_loader.h"
#include "plugins/xmaterial.plugin/source/xmaterial_runtime.h"
#include "plugins/xmaterial.plugin/source/xmaterial_params.h"
#include "plugins/xmaterial_instance.plugin/source/xmaterial_instance_runtime.h"
#include "plugins/xgeom_static.plugin/source/xgeom_static_xgpu_rsc_loader.h"
#include "plugins/xgeom_static.plugin/source/xgeom_static_xgpu_runtime.h"

#include <vector>
#include <string>
#include <algorithm>
#include <format>

namespace xlionrender
{
    // The resources, by the type guids of their plugins (derived the same way every plugin derives its own: no plugin header here)
    inline constexpr auto geom_static_type_guid_v       = xresource::type_guid(xresource::guid_generator::Instance64FromString("GeomStatic"));
    inline constexpr auto material_instance_type_guid_v = xresource::type_guid(xresource::guid_generator::Instance64FromString("MaterialInstance"));
    inline constexpr auto material_type_guid_v          = xresource::type_guid(xresource::guid_generator::Instance64FromString("Material"));
    inline constexpr auto texture_type_guid_v           = xresource::type_guid(xresource::guid_generator::Instance64FromString("texture"));
    using geom_static_ref       = xresource::def_guid<geom_static_type_guid_v>;
    using material_instance_ref = xresource::def_guid<material_instance_type_guid_v>;
    using texture_ref           = xresource::def_guid<texture_type_guid_v>;

    // One parameter a Mesh slot overrides (sparse): a parameter of the slot's material instance, by its id - or only by its name when game code
    // set it that way (mesh::SetColor ...): the renderer finds it in the material's parameters. Values are linear; a texture by its resource
    struct mesh_param
    {
        std::uint64_t   m_ID        = 0;
        std::string     m_Name;
        float           m_X = 0, m_Y = 0, m_Z = 0, m_W = 0;
        texture_ref     m_Texture;

        XPROPERTY_DEF
        ( "MeshParam", mesh_param
        , obj_member<"ID",      &mesh_param::m_ID>
        , obj_member<"Name",    &mesh_param::m_Name>
        , obj_member<"X",       &mesh_param::m_X>
        , obj_member<"Y",       &mesh_param::m_Y>
        , obj_member<"Z",       &mesh_param::m_Z>
        , obj_member<"W",       &mesh_param::m_W>
        , obj_member<"Texture", &mesh_param::m_Texture>
        )
    };
    XPROPERTY_REG(mesh_param)

    struct mesh_slot;

    // The inspector's row of one parameter of a slot's material instance (never saved): its name, whether the slot overrides it, and its value -
    // the override's, else the instance's - through a view of its type (the Material Instance editor's widgets: drag bars, color groups, a texture).
    // An edit sets the slot's override; Override off removes it
    struct mesh_param_row
    {
        mesh_slot*      m_pSlot     = nullptr;
        std::uint64_t   m_ID        = 0;
        std::string     m_Name;
        std::uint8_t    m_Type      = 0;            // xmaterial::param_type
        float           m_Default[4]= { 0, 0, 0, 0 };
        texture_ref     m_DefaultTexture;

        inline bool         isOverride  ( void ) const noexcept;
        inline void         setOverride ( bool b ) noexcept;
        inline float        Get         ( int i ) const noexcept;
        inline void         Set         ( int i, float V ) noexcept;
        inline texture_ref  getTexture  ( void ) const noexcept;
        inline void         setTexture  ( const texture_ref& T ) noexcept;
        static std::pair<const xproperty::type::object*, void*> ValueView( mesh_param_row& O ) noexcept;

        XPROPERTY_DEF
        ( "MeshParamRow", mesh_param_row, xproperty::settings::sprop_union_group
        , obj_member<"Name",     &mesh_param_row::m_Name, member_flags<flags::SHOW_READONLY>>
        , obj_member<"Override", +[](mesh_param_row& O, bool bRead, bool& V) { if (bRead) V = O.isOverride(); else O.setOverride(V); }, member_help<"On: this entity's value; off: the material instance's">>
        , obj_member<"Value",    +[](mesh_param_row& O) -> std::pair<const xproperty::type::object*, void*> { return ValueView(O); }>
        )
    };
    XPROPERTY_REG(mesh_param_row)

    struct mesh_param_float_view  : mesh_param_row { XPROPERTY_DEF("mesh param: float",  mesh_param_float_view,  obj_member<"Value", +[](mesh_param_float_view& O, bool bRead, float& V) { if (bRead) V = O.Get(0); else O.Set(0, V); }, member_ui<float>::drag_bar<0.005f>>) };
    struct mesh_param_float2_view : mesh_param_row { XPROPERTY_DEF("mesh param: float2", mesh_param_float2_view, xproperty::settings::vector2_group
        , obj_member<"X", +[](mesh_param_float2_view& O, bool bRead, float& V) { if (bRead) V = O.Get(0); else O.Set(0, V); }, member_ui<float>::drag_bar<0.005f>>
        , obj_member<"Y", +[](mesh_param_float2_view& O, bool bRead, float& V) { if (bRead) V = O.Get(1); else O.Set(1, V); }, member_ui<float>::drag_bar<0.005f>>) };
    struct mesh_param_float3_view : mesh_param_row { XPROPERTY_DEF("mesh param: float3", mesh_param_float3_view, xproperty::settings::vector3_group
        , obj_member<"X", +[](mesh_param_float3_view& O, bool bRead, float& V) { if (bRead) V = O.Get(0); else O.Set(0, V); }, member_ui<float>::drag_bar<0.005f>>
        , obj_member<"Y", +[](mesh_param_float3_view& O, bool bRead, float& V) { if (bRead) V = O.Get(1); else O.Set(1, V); }, member_ui<float>::drag_bar<0.005f>>
        , obj_member<"Z", +[](mesh_param_float3_view& O, bool bRead, float& V) { if (bRead) V = O.Get(2); else O.Set(2, V); }, member_ui<float>::drag_bar<0.005f>>) };
    struct mesh_param_float4_view : mesh_param_row { XPROPERTY_DEF("mesh param: float4", mesh_param_float4_view, xproperty::settings::vector4_group
        , obj_member<"X", +[](mesh_param_float4_view& O, bool bRead, float& V) { if (bRead) V = O.Get(0); else O.Set(0, V); }, member_ui<float>::drag_bar<0.005f>>
        , obj_member<"Y", +[](mesh_param_float4_view& O, bool bRead, float& V) { if (bRead) V = O.Get(1); else O.Set(1, V); }, member_ui<float>::drag_bar<0.005f>>
        , obj_member<"Z", +[](mesh_param_float4_view& O, bool bRead, float& V) { if (bRead) V = O.Get(2); else O.Set(2, V); }, member_ui<float>::drag_bar<0.005f>>
        , obj_member<"W", +[](mesh_param_float4_view& O, bool bRead, float& V) { if (bRead) V = O.Get(3); else O.Set(3, V); }, member_ui<float>::drag_bar<0.005f>>) };
    struct mesh_param_color_view  : mesh_param_row { XPROPERTY_DEF("mesh param: color",  mesh_param_color_view,  xproperty::settings::color4_group
        , obj_member<"R", +[](mesh_param_color_view& O, bool bRead, float& V) { if (bRead) V = O.Get(0); else O.Set(0, V); }, member_ui<float>::drag_bar<0.005f>>
        , obj_member<"G", +[](mesh_param_color_view& O, bool bRead, float& V) { if (bRead) V = O.Get(1); else O.Set(1, V); }, member_ui<float>::drag_bar<0.005f>>
        , obj_member<"B", +[](mesh_param_color_view& O, bool bRead, float& V) { if (bRead) V = O.Get(2); else O.Set(2, V); }, member_ui<float>::drag_bar<0.005f>>
        , obj_member<"A", +[](mesh_param_color_view& O, bool bRead, float& V) { if (bRead) V = O.Get(3); else O.Set(3, V); }, member_ui<float>::drag_bar<0.005f>>) };
    struct mesh_param_texture_view : mesh_param_row { XPROPERTY_DEF("mesh param: texture", mesh_param_texture_view
        , obj_member<"Value", +[](mesh_param_texture_view& O, bool bRead, texture_ref& V) { if (bRead) V = O.getTexture(); else O.setTexture(V); }>) };
    XPROPERTY_REG(mesh_param_float_view)
    XPROPERTY_REG(mesh_param_float2_view)
    XPROPERTY_REG(mesh_param_float3_view)
    XPROPERTY_REG(mesh_param_float4_view)
    XPROPERTY_REG(mesh_param_color_view)
    XPROPERTY_REG(mesh_param_texture_view)

    inline std::pair<const xproperty::type::object*, void*> mesh_param_row::ValueView( mesh_param_row& O ) noexcept
    {
        using T = xmaterial::param_type;
        switch (T(O.m_Type))
        {
        case T::FLOAT:   return { xproperty::getObjectByType<mesh_param_float_view>(),   &O };
        case T::FLOAT2:  return { xproperty::getObjectByType<mesh_param_float2_view>(),  &O };
        case T::FLOAT3:  return { xproperty::getObjectByType<mesh_param_float3_view>(),  &O };
        case T::FLOAT4:  return { xproperty::getObjectByType<mesh_param_float4_view>(),  &O };
        case T::COLOR:   return { xproperty::getObjectByType<mesh_param_color_view>(),   &O };
        case T::TEXTURE: return { xproperty::getObjectByType<mesh_param_texture_view>(), &O };
        }
        return { nullptr, nullptr };
    }

    // One material slot of a Mesh with "Override materials" on: a material instance replacing the geometry's (empty: the geometry's own) and the
    // entity's own values for its parameters (sparse: only what it overrides)
    struct mesh_slot
    {
        material_instance_ref       m_Material;
        std::vector<mesh_param>     m_Params;

        // The inspector's (never saved): the slot's name, the geometry's own instance (mesh::Slots), the rows of the resolved instance's parameters
        std::string                 m_Name;
        material_instance_ref       m_Default;
        std::vector<mesh_param_row> m_Rows;
        const void*                 m_pRowsOf   = nullptr;          // the loaded instance the rows were made from

        // The instance it draws with: its own, else the geometry's
        std::uint64_t Instance( void ) const noexcept { return m_Material.m_Instance.m_Value ? m_Material.m_Instance.m_Value : m_Default.m_Instance.m_Value; }

        mesh_param*       Find( std::uint64_t ID, std::string_view Name = {} ) noexcept
        {
            for (auto& P : m_Params) if ((ID && P.m_ID == ID) || (!Name.empty() && P.m_Name == Name)) return &P;
            return nullptr;
        }
        mesh_param& Set( std::uint64_t ID, std::string_view Name ) noexcept
        {
            if (auto* P = Find(ID, Name)) { if (ID) P->m_ID = ID; return *P; }
            auto& P = m_Params.emplace_back(mesh_param{ .m_ID = ID, .m_Name = std::string(Name) });
            if (ID) std::ranges::sort(m_Params, {}, &mesh_param::m_ID);
            return *Find(ID, Name);
        }
        bool Clear( std::uint64_t ID, std::string_view Name = {} ) noexcept
        {
            return std::erase_if(m_Params, [&](const mesh_param& P) { return (ID && P.m_ID == ID) || (!Name.empty() && P.m_Name == Name); }) > 0;
        }

        // The rows of the instance it draws with (its material's parameters, in the material's order), made again only when that instance (as
        // loaded) changed: the inspector asks for them many times a frame, and holds on to them while it walks them
        std::vector<mesh_param_row>& Rows( void ) noexcept
        {
            using namespace xlioncore::resources;
            const std::uint64_t MI = Instance();
            auto* pMI  = MI ? static_cast<xmaterial_instance::rt*>(GetResource(material_instance_type_guid_v.m_Value, MI)) : nullptr;
            auto* pMat = pMI ? static_cast<xmaterial::rt*>(GetResource(material_type_guid_v.m_Value, pMI->m_MaterialRef.m_Instance.m_Value)) : nullptr;
            if (pMat == nullptr || pMI == m_pRowsOf)
            {
                for (auto& R : m_Rows) R.m_pSlot = this;            // (the component may have moved)
                if (pMat == nullptr && m_pRowsOf) { m_Rows.clear(); m_pRowsOf = nullptr; }
                return m_Rows;
            }
            m_pRowsOf = pMI;
            m_Rows.clear();
            for (const auto& Info : pMat->getParams())
            {
                auto& Row  = m_Rows.emplace_back();
                Row.m_pSlot = this;
                Row.m_ID    = Info.m_ID;
                Row.m_Name  = Info.m_Name;
                Row.m_Type  = Info.m_Type;
                if (xmaterial::param_type(Info.m_Type) == xmaterial::param_type::TEXTURE)
                {
                    // the instance's texture of that slot (its list has the textures from slot 1 when the material keeps slot 0 for the shadow atlas)
                    // (a loaded instance holds its textures by pointer, and nothing here turns one back into its guid: then the row shows none, which
                    // means the instance's own until the slot overrides it)
                    const std::size_t i = Info.m_Slot + (pMat->m_Flags.m_bSamplersFrom1 ? 0u : 1u);
                    if (i < pMI->getTextures().size() && !pMI->getTextures()[i].m_TexureRef.m_Instance.isPointer()) Row.m_DefaultTexture.m_Instance = pMI->getTextures()[i].m_TexureRef.m_Instance;
                }
                else for (int c = 0; c < 4 && Info.m_ByteOffset / 4u + c < pMI->m_BlockBytes / 4u; ++c) Row.m_Default[c] = pMI->m_pParams[Info.m_ByteOffset / 4u + c];
            }
            return m_Rows;
        }

        XPROPERTY_DEF
        ( "MeshSlot", mesh_slot
        , obj_member<"Name",        &mesh_slot::m_Name,     member_flags<flags::SHOW_READONLY, flags::DONT_SAVE>>
        , obj_member<"Material",    &mesh_slot::m_Material, member_help<"A material instance replacing the geometry's for this slot (empty: the geometry's own)">>
        , obj_member<"Parameters",  +[](mesh_slot& O) -> std::vector<mesh_param_row>& { return O.Rows(); }, member_flags<flags::DONT_SAVE>, member_array_size_readonly<>, member_ui_open<true>
            , member_help<"The parameters of the instance this slot draws with: Override on, this entity's value; off, the instance's">>
        , obj_member<"Overrides",   &mesh_slot::m_Params,   member_flags<flags::DONT_SHOW>>
        )
    };
    XPROPERTY_REG(mesh_slot)

    bool mesh_param_row::isOverride( void ) const noexcept { return m_pSlot && m_pSlot->Find(m_ID, m_Name) != nullptr; }
    void mesh_param_row::setOverride( bool b ) noexcept
    {
        if (!m_pSlot) return;
        if (!b) { m_pSlot->Clear(m_ID, m_Name); return; }
        if (isOverride()) return;
        auto& P = m_pSlot->Set(m_ID, m_Name);            // starts at the instance's value
        P.m_X = m_Default[0]; P.m_Y = m_Default[1]; P.m_Z = m_Default[2]; P.m_W = m_Default[3];
        P.m_Texture = m_DefaultTexture;
    }
    float mesh_param_row::Get( int i ) const noexcept
    {
        if (const auto* P = m_pSlot ? m_pSlot->Find(m_ID, m_Name) : nullptr) return i == 0 ? P->m_X : i == 1 ? P->m_Y : i == 2 ? P->m_Z : P->m_W;
        return m_Default[i];
    }
    void mesh_param_row::Set( int i, float V ) noexcept
    {
        if (!m_pSlot) return;
        setOverride(true);
        auto& P = *m_pSlot->Find(m_ID, m_Name);
        (i == 0 ? P.m_X : i == 1 ? P.m_Y : i == 2 ? P.m_Z : P.m_W) = V;
    }
    texture_ref mesh_param_row::getTexture( void ) const noexcept
    {
        if (const auto* P = m_pSlot ? m_pSlot->Find(m_ID, m_Name) : nullptr) return P->m_Texture;
        return m_DefaultTexture;
    }
    void mesh_param_row::setTexture( const texture_ref& T ) noexcept
    {
        if (!m_pSlot) return;
        setOverride(true);
        m_pSlot->Find(m_ID, m_Name)->m_Texture = T;
    }

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
        bool                                m_bOverrideMaterials = false;
        std::vector<mesh_slot>              m_Slots;        // with m_bOverrideMaterials: per material slot of the geometry

        // The slots, one per material slot of the geometry (as loaded: until then as they are), each knowing the geometry's own instance
        std::vector<mesh_slot>& Slots( void ) noexcept
        {
            auto* pGeom = m_Geom.empty() ? nullptr : static_cast<xgeom_static::xgpu::geom*>(xlioncore::resources::GetResource(geom_static_type_guid_v.m_Value, m_Geom.m_Instance.m_Value));
            if (pGeom == nullptr) return m_Slots;
            const auto Defaults = pGeom->getDefaultMaterialInstances();
            if (m_Slots.size() != Defaults.size()) m_Slots.resize(Defaults.size());
            for (std::size_t i = 0; i < Defaults.size(); ++i)
            {
                m_Slots[i].m_Default.m_Instance = Defaults[i].m_Instance;
                if (m_Slots[i].m_Name.empty()) m_Slots[i].m_Name = std::format("Slot {}", i);
            }
            return m_Slots;
        }

        // Game code: the entity's own value of a parameter of slot iSlot's material instance, by the parameter's name (as the material names it)
        // or its id; Override Materials turns on. Values are linear; a color's alpha in A. Clear: back to the instance's value
        mesh_param& Param( int iSlot, std::uint64_t ID, std::string_view Name ) noexcept
        {
            m_bOverrideMaterials = true;
            if (m_Slots.size() <= std::size_t(iSlot)) m_Slots.resize(std::size_t(iSlot) + 1);
            return m_Slots[iSlot].Set(ID, Name);
        }
        void SetFloat  ( int iSlot, std::string_view Name, float V ) noexcept                       { Param(iSlot, 0, Name).m_X = V; }
        void SetColor  ( int iSlot, std::string_view Name, const xmath::fvec4& C ) noexcept         { auto& P = Param(iSlot, 0, Name); P.m_X = C.m_X; P.m_Y = C.m_Y; P.m_Z = C.m_Z; P.m_W = C.m_W; }
        void SetVector ( int iSlot, std::string_view Name, const xmath::fvec4& V ) noexcept         { SetColor(iSlot, Name, V); }
        void SetTexture( int iSlot, std::string_view Name, const texture_ref& T ) noexcept          { Param(iSlot, 0, Name).m_Texture = T; }
        void SetFloat  ( int iSlot, std::uint64_t ID, float V ) noexcept                            { Param(iSlot, ID, {}).m_X = V; }
        void SetColor  ( int iSlot, std::uint64_t ID, const xmath::fvec4& C ) noexcept              { auto& P = Param(iSlot, ID, {}); P.m_X = C.m_X; P.m_Y = C.m_Y; P.m_Z = C.m_Z; P.m_W = C.m_W; }
        void SetTexture( int iSlot, std::uint64_t ID, const texture_ref& T ) noexcept               { Param(iSlot, ID, {}).m_Texture = T; }
        void Clear     ( int iSlot, std::string_view Name ) noexcept                                { if (std::size_t(iSlot) < m_Slots.size()) m_Slots[iSlot].Clear(0, Name); }
        void SetMaterial( int iSlot, const material_instance_ref& MI ) noexcept                     { m_bOverrideMaterials = true; if (m_Slots.size() <= std::size_t(iSlot)) m_Slots.resize(std::size_t(iSlot) + 1); m_Slots[iSlot].m_Material = MI; }

        // ponytail: old files keep their swaps here, shown and saved only while there are any. Moving them into a MeshMaterial at load
        // needs an archetype change in the scene loader (xECSV2 xecs_scene_inline.h ReadEntityFile); do it there if the old field must go

        XPROPERTY_DEF
        ( "Mesh", mesh
        , obj_member<"Geom",      &mesh::m_Geom,      member_help<"The GeomStatic resource to draw">>
        , obj_member<"Materials", &mesh::m_Materials, member_dynamic_flags<+[](const mesh& M) { xproperty::flags::type F{}; F.m_bDontShow = F.m_bDontSave = M.m_Materials.empty(); return F; }>, member_help<"Deprecated (old levels): use the MeshMaterial component's Slots, which win per slot. Material instances per material slot of the geometry, replacing the geometry's own">>
        , obj_member<"GIRole",    &mesh::m_GIRole,    member_enum_span<gi_role_v>, member_help<"What it does in the GI: ASSET as its geometry says; RECEIVER lit by it but neither blocking nor bouncing light (a prop, a character: moving it costs the GI nothing elsewhere); FULL blocks, bounces and emits">>
        , obj_member<"Override Materials", &mesh::m_bOverrideMaterials, member_help<"This entity's own materials: per material slot of the geometry, another material instance and its own values for the parameters it chooses">>
        , obj_member<"Slots", +[](mesh& O) -> std::vector<mesh_slot>& { return O.Slots(); }, member_array_size_readonly<>, member_ui_open<true>
            , member_dynamic_flags<+[](const mesh& M) { xproperty::flags::type F{}; F.m_bDontShow = !M.m_bOverrideMaterials; F.m_bDontSave = !M.m_bOverrideMaterials; return F; }>
            , member_help<"The geometry's material slots: each a material instance (empty: the geometry's own) and the parameters this entity overrides">>
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

    // Old files' material data into the Mesh's own slots (Override Materials on): the swaps of its deprecated Materials list and of the
    // MeshMaterial (which won per slot), and the MeshMaterial's entity values (Tint, Param0, Param1) as overrides of the parameter each slot's
    // material lets that field replace (its manifest's entity field; none: the value never reached that material, nothing to keep). The old data
    // is emptied, so the MeshMaterial draws nothing twice until the editor removes it. Whether it has old data at all: HasOldMaterials.
    // False (nothing changed) while the geometry or a material the values need is not loaded.
    inline bool HasOldMaterials( const mesh& Mesh, const mesh_material* pOld ) noexcept
    {
        return pOld || !Mesh.m_Materials.empty();
    }
    inline bool UpgradeOldMaterials( mesh& Mesh, mesh_material* pOld ) noexcept
    {
        using namespace xlioncore::resources;
        if (!HasOldMaterials(Mesh, pOld)) return false;
        auto* pGeom = Mesh.m_Geom.empty() ? nullptr : static_cast<xgeom_static::xgpu::geom*>(GetResource(geom_static_type_guid_v.m_Value, Mesh.m_Geom.m_Instance.m_Value));
        if (pGeom == nullptr) return false;
        const auto Defaults = pGeom->getDefaultMaterialInstances();
        const auto SwapOf = [&](std::size_t i) -> const material_instance_ref*
        {
            if (pOld && i < pOld->m_Slots.size() && pOld->m_Slots[i].m_Instance.m_Value) return &pOld->m_Slots[i];
            if (i < Mesh.m_Materials.size() && Mesh.m_Materials[i].m_Instance.m_Value) return &Mesh.m_Materials[i];
            return nullptr;
        };
        const entity_value* Values[] = { pOld ? &pOld->m_Tint : nullptr, pOld ? &pOld->m_Param0 : nullptr, pOld ? &pOld->m_Param1 : nullptr };     // entity_field TINT, PARAM0, PARAM1
        const bool bValues = std::ranges::any_of(Values, [](const entity_value* p) { return p && p->m_bOverride; });

        // The materials the values go to, all loaded first (a half-done conversion would be saved half done)
        std::vector<const xmaterial::rt*> Mats(Defaults.size(), nullptr);
        if (bValues)
            for (std::size_t i = 0; i < Defaults.size(); ++i)
            {
                const auto* pSwap = SwapOf(i);
                const std::uint64_t MI = pSwap ? pSwap->m_Instance.m_Value : Defaults[i].m_Instance.m_Value;
                auto* pMI = MI ? static_cast<xmaterial_instance::rt*>(GetResource(material_instance_type_guid_v.m_Value, MI)) : nullptr;
                Mats[i]   = pMI ? static_cast<const xmaterial::rt*>(GetResource(material_type_guid_v.m_Value, pMI->m_MaterialRef.m_Instance.m_Value)) : nullptr;
                if (Mats[i] == nullptr) return false;
            }

        auto& Slots = Mesh.Slots();
        for (std::size_t i = 0; i < Slots.size() && i < Defaults.size(); ++i)
        {
            if (const auto* pSwap = SwapOf(i)) Slots[i].m_Material = *pSwap;
            if (!bValues) continue;
            for (int f = 0; f < 3; ++f)
            {
                if (Values[f] == nullptr || !Values[f]->m_bOverride) continue;
                for (const auto& Info : Mats[i]->getParams())
                {
                    if (Info.m_EntityField != f + 1 || xmaterial::param_type(Info.m_Type) == xmaterial::param_type::TEXTURE) continue;
                    auto& P = Slots[i].Set(Info.m_ID, Info.m_Name);
                    P.m_X = Values[f]->m_X; P.m_Y = Values[f]->m_Y; P.m_Z = Values[f]->m_Z; P.m_W = Values[f]->m_W;
                }
            }
        }
        Mesh.m_bOverrideMaterials = true;
        Mesh.m_Materials.clear();
        if (pOld) *pOld = mesh_material{ .m_Tint = {} };
        return true;
    }

    // The directional light: it shines along the entity's -Y; one per level (the first one found)
    struct light_sun
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "LightSun" };

        xmath::fvec3    m_Color         = xmath::fvec3::fromOne();
        float           m_Intensity     = 3.0f;
        bool            m_bCastShadows  = true;
        float           m_ShadowTileSize = 4.0f;           // world size of a level-0 shadow tile; the clipmap levels double it
        bool            m_bGI           = true;             // in the GI's lights (its bounce); off: direct light only

        XPROPERTY_DEF
        ( "LightSun", light_sun
        , obj_member<"Color",          &light_sun::m_Color>
        , obj_member<"Intensity",      &light_sun::m_Intensity>
        , obj_member<"CastShadows",    &light_sun::m_bCastShadows>
        , obj_member<"ShadowTileSize", &light_sun::m_ShadowTileSize, member_help<"World size of the finest shadow tile: smaller is sharper near the camera">>
        , obj_member<"GI",             &light_sun::m_bGI, member_help<"Bounce light (global illumination). Off: direct light only, no bounce and no GI cost - for short-lived or fast lights such as bullets">>
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
        bool            m_bGI           = true;             // in the GI's lights (its bounce); off: direct light only

        XPROPERTY_DEF
        ( "LightPoint", light_point
        , obj_member<"Color",       &light_point::m_Color>
        , obj_member<"Intensity",   &light_point::m_Intensity>
        , obj_member<"Radius",      &light_point::m_Radius, member_help<"No light at and beyond this distance">>
        , obj_member<"CastShadows", &light_point::m_bCastShadows>
        , obj_member<"GI",          &light_point::m_bGI, member_help<"Bounce light (global illumination). Off: direct light only, no bounce and no GI cost - for short-lived or fast lights such as bullets">>
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
        bool            m_bGI           = true;             // in the GI's lights (its bounce); off: direct light only

        XPROPERTY_DEF
        ( "LightSpot", light_spot
        , obj_member<"Color",       &light_spot::m_Color>
        , obj_member<"Intensity",   &light_spot::m_Intensity>
        , obj_member<"Radius",      &light_spot::m_Radius>
        , obj_member<"InnerAngle",  &light_spot::m_InnerAngle, member_help<"Radians from the axis inside which the light is full">>
        , obj_member<"OuterAngle",  &light_spot::m_OuterAngle, member_help<"Radians from the axis beyond which there is no light (at most about 1)">>
        , obj_member<"CastShadows", &light_spot::m_bCastShadows>
        , obj_member<"GI",          &light_spot::m_bGI, member_help<"Bounce light (global illumination). Off: direct light only, no bounce and no GI cost - for short-lived or fast lights such as bullets">>
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
        bool            m_bGIHalfRes    = true;            // that read at half resolution, upsampled (a quarter of the reads; its pass draws in indirect runs)
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
        , obj_member<"GIHalfRes",    &environment::m_bGIHalfRes, member_help<"The per-pixel GI read at half resolution, upsampled by depth and normal: a quarter of the reads (opaque surfaces; not in the GI debug views)">>
        )
    };

    // After the Primitive (50) and the Text
    XSCRIPT_REGISTER_COMPONENT(mesh,        "Rendering", 52)
    XSCRIPT_REGISTER_COMPONENT(light_sun,   "Rendering", 53)
    XSCRIPT_REGISTER_COMPONENT(light_point, "Rendering", 54)
    XSCRIPT_REGISTER_COMPONENT(light_spot,  "Rendering", 55)
    XSCRIPT_REGISTER_COMPONENT(environment, "Rendering", 56)
    XSCRIPT_REGISTER_COMPONENT(mesh_material, "Deprecated", 57)     // (old files: the editor moves it into the Mesh, UpgradeOldMaterials; not offered to add)
}

#endif
