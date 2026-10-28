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
#include "plugins/xmaterial.plugin/source/xmaterial_render_settings.h"
#include "plugins/xmaterial_instance.plugin/source/xmaterial_instance_runtime.h"
#include "plugins/xgeom_static.plugin/source/xgeom_static_xgpu_rsc_loader.h"
#include "plugins/xgeom_static.plugin/source/xgeom_static_xgpu_runtime.h"
#include "plugins/xgeom_skin.plugin/source/xgeom_skin_xgpu_rsc_loader.h"
#include "plugins/xgeom_skin.plugin/source/xgeom_skin_xgpu_runtime.h"

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
    inline constexpr auto geom_skin_type_guid_v         = xresource::type_guid(xresource::guid_generator::Instance64FromString("GeomSkin"));
    inline constexpr auto skeleton_type_guid_v          = xresource::type_guid(xresource::guid_generator::Instance64FromString("xSkeleton"));
    inline constexpr auto anim_package_type_guid_v      = xresource::type_guid(xresource::guid_generator::Instance64FromString("xAnimPackage"));
    using geom_static_ref       = xresource::def_guid<geom_static_type_guid_v>;
    using geom_skin_ref         = xresource::def_guid<geom_skin_type_guid_v>;
    using anim_package_ref      = xresource::def_guid<anim_package_type_guid_v>;
    using material_instance_ref = xresource::def_guid<material_instance_type_guid_v>;
    using texture_ref           = xresource::def_guid<texture_type_guid_v>;

    // One parameter a Mesh slot overrides (sparse): a parameter of the slot's material instance, by its id - or only by its name when game code
    // set it that way (mesh::SetColor ...): the renderer finds it in the material's parameters. Values are linear; a texture by its resource
    // The key a parameter's override and row are addressed by (xproperty::keyed_vector: "Overrides[G:key]", "Parameters[G:key]"): its id - stable
    // across renames and reorders -, or a hash of its name when only the name is known (game code that set it by name)
    inline std::uint64_t ParamKey( std::uint64_t ID, std::string_view Name ) noexcept
    {
        if (ID) return ID;
        std::uint64_t H = 14695981039346656037ull;
        for (const char c : Name) { H ^= std::uint8_t(c); H *= 1099511628211ull; }
        return H;
    }

    struct mesh_param
    {
        std::uint64_t   m_Key       = 0;            // ParamKey: its address in the slot's Overrides (the path's, not saved as a property)
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
    struct mesh;

    // The marker and the revert of a slot's rows (member_override_state): a parameter row is overridden when the slot sets it, the Material row when
    // the slot swaps the instance; the revert sets the row's Override text to empty / the Material to the geometry's own (defined after mesh)
    inline void MeshSlotOverrideState( const xproperty::type::object& Type, void* pMesh, void* pBase, std::string_view Path, bool& bOverridden
                                     , std::vector<xproperty::settings::member_override_state_t::revert>* pChoices ) noexcept;

    // The inspector's row of one parameter of a slot's material instance (never saved): its name and the value the entity draws with - its
    // override's, else the instance's - through a view of its type (the Material Instance editor's widgets: drag bars, color groups, a texture).
    // An edit sets the slot's override; the marker (>) reverts it. Override: the whole override as text ("" inherits; numbers "x, y, z, w"; a
    // texture "#guid") - one property for undo, prefabs and the AI; Source: where the value comes from. An orphan row: an override of a parameter
    // the material does not have (a swapped instance) - kept, not applied, read only until reverted
    struct mesh_param_row
    {
        mesh_slot*      m_pSlot     = nullptr;
        std::uint64_t   m_Key       = 0;            // ParamKey: its address ("Parameters[G:key]")
        std::uint64_t   m_ID        = 0;
        std::string     m_Name;
        std::uint8_t    m_Type      = 0;            // xmaterial::param_type
        bool            m_bOrphan   = false;
        float           m_Default[4]= { 0, 0, 0, 0 };
        texture_ref     m_DefaultTexture;

        inline const mesh_param* Entry       ( void ) const noexcept;
        inline float        Get         ( int i ) const noexcept;
        inline void         Set         ( int i, float V ) noexcept;
        inline texture_ref  getTexture  ( void ) const noexcept;
        inline void         setTexture  ( const texture_ref& T ) noexcept;
        inline std::string  getOverride ( void ) const noexcept;
        inline static std::string FormatOverride( std::uint8_t Type, const mesh_param* P ) noexcept;        // an override as the Override text ("" for none)
        inline void         setOverride ( const std::string& Text ) noexcept;
        inline std::string  Source      ( void ) const noexcept;
        static std::pair<const xproperty::type::object*, void*> ValueView( mesh_param_row& O ) noexcept;

        XPROPERTY_DEF
        ( "MeshParamRow", mesh_param_row, xproperty::settings::sprop_union_group
        , obj_member<"Name",     &mesh_param_row::m_Name, member_flags<flags::SHOW_READONLY>>
        , obj_member<"Value",    +[](mesh_param_row& O) -> std::pair<const xproperty::type::object*, void*> { return ValueView(O); }
            , member_dynamic_flags<+[](const mesh_param_row& O) { xproperty::flags::type F{}; F.m_bShowReadOnly = O.m_bOrphan; return F; }>>
        , obj_member<"Override", +[](mesh_param_row& O, bool bRead, std::string& V) { if (bRead) V = O.getOverride(); else O.setOverride(V); }, member_flags<flags::DONT_SHOW>
            , member_help<"The entity's own value as text: empty inherits the material instance's; \"x, y, z, w\" (as many numbers as the type has); \"#guid\" a texture">>
        , obj_member<"Source",   +[](mesh_param_row& O, bool bRead, std::string& V) { if (bRead) V = O.Source(); }, member_flags<flags::DONT_SHOW, flags::SHOW_READONLY>
            , member_help<"Where the value comes from: material instance, entity, or not applied (the material has no such parameter)">>
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
        , obj_member<"Value", +[](mesh_param_texture_view& O, bool bRead, xresource::full_guid& V)                      // (a full guid: a lambda's value is one, with its type)
            { if (bRead) V = { .m_Instance = O.getTexture().m_Instance, .m_Type = texture_ref::m_Type }; else O.setTexture(texture_ref{ .m_Instance = V.m_Instance }); }>) };
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

    // One material slot of a Mesh: the instance it draws with - the geometry's own unless the entity swaps it (m_Material) - and the entity's own
    // values for its parameters (sparse: only what it overrides; kept across a swap, applied where the new material has the parameter)
    struct mesh_slot
    {
        material_instance_ref       m_Material;                     // the swap (empty: the geometry's own)
        xproperty::keyed_vector<mesh_param, &mesh_param::m_Key> m_Params;      // by ParamKey: a path names a parameter, not a place

        // The inspector's (never saved): the slot's name, the geometry's own instance (mesh::Slots), the rows of the resolved instance's parameters
        std::string                 m_Name;
        material_instance_ref       m_Default;
        xproperty::keyed_vector<mesh_param_row, &mesh_param_row::m_Key> m_Rows;
        const void*                 m_pRowsOf   = nullptr;          // the loaded instance the rows were made from
        std::size_t                 m_nRowsOf   = 0;                // ... and how many overrides there were (orphans come and go with them)

        // The instance it draws with: its own, else the geometry's
        std::uint64_t Instance( void ) const noexcept { return m_Material.m_Instance.m_Value ? m_Material.m_Instance.m_Value : m_Default.m_Instance.m_Value; }

        // An override by the parameter's id, else by its name (an override set by name, or a parameter the material renamed keeps its value by id)
        mesh_param*       Find( std::uint64_t ID, std::string_view Name = {} ) noexcept
        {
            if (ID) for (auto& P : m_Params) if (P.m_ID == ID) return &P;
            if (!Name.empty()) for (auto& P : m_Params) if (P.m_Name == Name) return &P;
            return nullptr;
        }
        const mesh_param* Find( std::uint64_t ID, std::string_view Name = {} ) const noexcept { return const_cast<mesh_slot*>(this)->Find(ID, Name); }
        mesh_param& Set( std::uint64_t ID, std::string_view Name ) noexcept
        {
            if (auto* P = Find(ID, Name)) { if (ID) P->m_ID = ID; if (!Name.empty()) P->m_Name = std::string(Name); return *P; }
            m_Params.emplace_back(mesh_param{ .m_Key = ParamKey(ID, Name), .m_ID = ID, .m_Name = std::string(Name) });
            return m_Params.back();
        }
        bool Revert( std::uint64_t ID, std::string_view Name = {} ) noexcept
        {
            if (auto* P = Find(ID, Name)) { m_Params.erase(m_Params.begin() + (P - m_Params.data())); return true; }
            return false;
        }

        // The rows of the instance it draws with (its material's parameters in the order the Material Instance editor lists them - the compiled
        // list is in it, xmaterial::InspectorOrder - but the render settings, then the orphans), made again only when
        // that instance (as loaded) or the number of overrides changed: the inspector asks for them many times a frame, and holds on to them
        xproperty::keyed_vector<mesh_param_row, &mesh_param_row::m_Key>& Rows( void ) noexcept
        {
            using namespace xlioncore::resources;
            // A row a path adds (an override applied before the instance is loaded: a prefab instance's, at a load) is this slot's, the
            // parameter its key names (ParamKey: the id); the rows made when the instance is loaded replace it
            m_Rows.m_pOnAdd     = +[](mesh_param_row& R, void* pSlot) { R.m_pSlot = static_cast<mesh_slot*>(pSlot); R.m_ID = R.m_Key; R.m_Type = std::uint8_t(xmaterial::param_type::FLOAT4); };
            m_Rows.m_pOnAddUser = this;
            const std::uint64_t MI = Instance();
            auto* pMI  = MI ? static_cast<xmaterial_instance::rt*>(GetResource(material_instance_type_guid_v.m_Value, MI)) : nullptr;
            auto* pMat = pMI ? static_cast<xmaterial::rt*>(GetResource(material_type_guid_v.m_Value, pMI->m_MaterialRef.m_Instance.m_Value)) : nullptr;
            if (pMat == nullptr || (pMI == m_pRowsOf && m_Params.size() == m_nRowsOf))
            {
                for (auto& R : m_Rows) R.m_pSlot = this;            // (the component may have moved)
                if (pMat == nullptr && m_pRowsOf) { m_Rows.clear(); m_pRowsOf = nullptr; }
                return m_Rows;
            }
            m_pRowsOf = pMI;
            m_nRowsOf = m_Params.size();
            m_Rows.clear();
            for (const auto& Info : pMat->getParams())
            {
                if (xmaterial::isRenderSettingParam(Info.m_ID)) continue;          // (the instance's Render Settings, not a parameter row)
                auto& Row  = m_Rows.emplace_back();
                Row.m_pSlot = this;
                Row.m_Key   = ParamKey(Info.m_ID, Info.m_Name);
                Row.m_ID    = Info.m_ID;
                if (auto* P = m_Params.find(Row.m_Key); P && P->m_Name.empty()) P->m_Name = Info.m_Name;      // (an override applied by its key alone: its name too)
                Row.m_Name  = Info.m_Name;
                Row.m_Type  = Info.m_Type;
                if (xmaterial::param_type(Info.m_Type) == xmaterial::param_type::TEXTURE)
                {
                    // the instance's texture of that slot (its list has the textures from slot 1 when the material keeps slot 0 for the shadow atlas),
                    // by its guid (a loaded instance holds it as a pointer)
                    const std::size_t i = Info.m_Slot + (pMat->m_Flags.m_bSamplersFrom1 ? 0u : 1u);
                    if (i < pMI->getTextures().size()) Row.m_DefaultTexture.m_Instance.m_Value = GuidOf(texture_type_guid_v.m_Value, pMI->getTextures()[i].m_TexureRef.m_Instance.m_Value);
                }
                else for (int c = 0; c < 4 && Info.m_ByteOffset / 4u + c < pMI->m_BlockBytes / 4u; ++c) Row.m_Default[c] = pMI->m_pParams[Info.m_ByteOffset / 4u + c];
            }
            const std::size_t nParams = m_Rows.size();
            for (const auto& P : m_Params)
            {
                const auto Rows = std::span(m_Rows.data(), nParams);
                if (std::ranges::any_of(Rows, [&](const mesh_param_row& R) { return (P.m_ID && R.m_ID == P.m_ID) || (!P.m_Name.empty() && R.m_Name == P.m_Name); })) continue;
                auto& Row    = m_Rows.emplace_back();
                Row.m_pSlot  = this;
                Row.m_Key    = P.m_Key;
                Row.m_ID     = P.m_ID;
                Row.m_Name   = P.m_Name.empty() ? std::format("{:016X}", P.m_ID) : P.m_Name;
                Row.m_Type   = std::uint8_t(P.m_Texture.empty() ? xmaterial::param_type::FLOAT4 : xmaterial::param_type::TEXTURE);
                Row.m_bOrphan = true;
            }
            return m_Rows;
        }

        XPROPERTY_DEF
        ( "MeshSlot", mesh_slot
        , obj_member<"Name",        &mesh_slot::m_Name,     member_flags<flags::SHOW_READONLY, flags::DONT_SAVE>>
        , obj_member<"Material",    +[](mesh_slot& O, bool bRead, xresource::full_guid& V)
                                    {
                                        if (bRead) V = { .m_Instance = (O.m_Material.m_Instance.m_Value ? O.m_Material : O.m_Default).m_Instance, .m_Type = material_instance_ref::m_Type };
                                        else       O.m_Material.m_Instance = V.m_Instance.m_Value == O.m_Default.m_Instance.m_Value ? xresource::instance_guid{} : V.m_Instance;
                                    }
            , member_flags<flags::DONT_SAVE>, member_override_state<&MeshSlotOverrideState>
            , member_help<"The material instance this slot draws with: the geometry's own unless this entity picks another (then > goes back to the geometry's)">>
        , obj_member<"Parameters",  +[](mesh_slot& O) -> xproperty::keyed_vector<mesh_param_row, &mesh_param_row::m_Key>& { return O.Rows(); }, member_flags<flags::DONT_SAVE>, member_array_size_readonly<>, member_ui_open<true>
            , member_override_state<&MeshSlotOverrideState>
            , member_help<"The parameters of the instance this slot draws with, at the values this entity draws with: an edit overrides the instance's value, > reverts it">>
        , obj_member<"MaterialOverride", &mesh_slot::m_Material, member_flags<flags::DONT_SHOW>, member_help<"Saved: the instance this entity swaps in (empty: the geometry's own)">>
        , obj_member<"Overrides",   &mesh_slot::m_Params,   member_flags<flags::DONT_SHOW>, member_help<"Saved: the parameters this entity overrides, by id and name">>
        )
    };
    XPROPERTY_REG(mesh_slot)

    const mesh_param* mesh_param_row::Entry( void ) const noexcept { return m_pSlot ? std::as_const(*m_pSlot).Find(m_ID, m_Name) : nullptr; }
    float mesh_param_row::Get( int i ) const noexcept
    {
        if (const auto* P = Entry()) return i == 0 ? P->m_X : i == 1 ? P->m_Y : i == 2 ? P->m_Z : P->m_W;
        return m_Default[i];
    }
    void mesh_param_row::Set( int i, float V ) noexcept
    {
        if (!m_pSlot || m_bOrphan) return;
        const bool bNew = Entry() == nullptr;
        auto& P = m_pSlot->Set(m_ID, m_Name);
        if (bNew) { P.m_X = m_Default[0]; P.m_Y = m_Default[1]; P.m_Z = m_Default[2]; P.m_W = m_Default[3]; }        // starts at the instance's value
        (i == 0 ? P.m_X : i == 1 ? P.m_Y : i == 2 ? P.m_Z : P.m_W) = V;
    }
    texture_ref mesh_param_row::getTexture( void ) const noexcept
    {
        if (const auto* P = Entry()) return P->m_Texture;
        return m_DefaultTexture;
    }
    void mesh_param_row::setTexture( const texture_ref& T ) noexcept
    {
        if (!m_pSlot || m_bOrphan) return;
        if (T.m_Instance.m_Value == m_DefaultTexture.m_Instance.m_Value || T.empty()) { m_pSlot->Revert(m_ID, m_Name); return; }        // the instance's own (or none picked): inherits
        m_pSlot->Set(m_ID, m_Name).m_Texture = T;
    }
    std::string mesh_param_row::getOverride( void ) const noexcept { return FormatOverride(m_Type, Entry()); }
    std::string mesh_param_row::FormatOverride( std::uint8_t Type, const mesh_param* P ) noexcept
    {
        if (P == nullptr) return {};
        if (!P->m_Texture.empty()) return std::format("#{:016X}", P->m_Texture.m_Instance.m_Value);
        using T = xmaterial::param_type;
        const int n = T(Type) == T::FLOAT ? 1 : T(Type) == T::FLOAT2 ? 2 : T(Type) == T::FLOAT3 ? 3 : 4;
        const float V[] = { P->m_X, P->m_Y, P->m_Z, P->m_W };
        std::string S;
        for (int c = 0; c < n; ++c) S += std::format("{}{}", c ? ", " : "", V[c]);
        return S;
    }
    void mesh_param_row::setOverride( const std::string& Text ) noexcept
    {
        if (!m_pSlot) return;
        if (Text.empty()) { m_pSlot->Revert(m_ID, m_Name); return; }
        auto& P = m_pSlot->Set(m_ID, m_Name);
        if (Text[0] == '#') { P.m_Texture.m_Instance.m_Value = std::strtoull(Text.c_str() + 1, nullptr, 16); return; }
        P.m_Texture = {};
        float V[4] = { m_Default[0], m_Default[1], m_Default[2], m_Default[3] };
        const char* p = Text.c_str();
        for (int c = 0; c < 4 && *p; ++c)
        {
            char* pEnd = nullptr;
            V[c] = std::strtof(p, &pEnd);
            if (pEnd == p) break;
            p = pEnd;
            while (*p == ',' || *p == ' ') ++p;
        }
        P.m_X = V[0]; P.m_Y = V[1]; P.m_Z = V[2]; P.m_W = V[3];
    }
    std::string mesh_param_row::Source( void ) const noexcept
    {
        if (m_bOrphan) return "not applied (the material has no such parameter)";
        return Entry() ? "entity" : "material instance";
    }

    static constexpr auto gi_role_v = std::array
    { xproperty::settings::enum_item("ASSET",    gi_role::ASSET)
    , xproperty::settings::enum_item("RECEIVER", gi_role::RECEIVER)
    , xproperty::settings::enum_item("FULL",     gi_role::FULL)
    };

    // A static geometry drawn with its materials: per material slot the geometry's own instance, or one the entity swaps in, with the entity's own
    // values for the parameters it overrides (mesh_slot). Old files: the MeshMaterial / Materials list, moved in by UpgradeOldMaterials
    struct mesh
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "Mesh" };

        geom_static_ref                     m_Geom;
        std::vector<material_instance_ref>  m_Materials;    // DEPRECATED (old files): slot swaps, moved into m_Slots by UpgradeOldMaterials
        gi_role                             m_GIRole = gi_role::ASSET;
        bool                                m_bOverrideMaterials = false;   // off (most Meshes): the geometry's materials, no slots shown, saved or drawn
        std::vector<mesh_slot>              m_Slots;        // with m_bOverrideMaterials: per material slot of the geometry (fewer: the rest are the geometry's own)

        // The slots, one per material slot of the geometry (as loaded: until then as they are), each knowing the geometry's own instance
        std::vector<mesh_slot>& Slots( void ) noexcept
        {
            auto* pGeom = m_Geom.empty() ? nullptr : static_cast<xgeom_static::xgpu::geom*>(xlioncore::resources::GetResource(geom_static_type_guid_v.m_Value, m_Geom.m_Instance.m_Value));
            if (pGeom == nullptr) return m_Slots;
            const auto Defaults = pGeom->getDefaultMaterialInstances();
            if (m_Slots.size() != Defaults.size()) m_Slots.resize(Defaults.size());
            for (std::size_t i = 0; i < Defaults.size(); ++i)
            {
                m_Slots[i].m_Default.m_Instance.m_Value = xlioncore::resources::GuidOf(material_instance_type_guid_v.m_Value, Defaults[i].m_Instance.m_Value);   // (a loaded geometry holds it as a pointer: the editor shows guids)
                if (m_Slots[i].m_Name.empty()) m_Slots[i].m_Name = std::format("Slot {}", i);
            }
            return m_Slots;
        }

        // Game code: the entity's own value of a parameter of slot iSlot's material instance, by the parameter's name (as the material names it)
        // or its id; Override Materials turns on. Values are linear; a color's alpha in A. Revert: back to the instance's value. SetMaterial: another
        // instance for the slot (an empty reference: the geometry's own); the overrides stay, applied where the new material has the parameter
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
        bool Revert    ( int iSlot, std::string_view Name ) noexcept                                { return std::size_t(iSlot) < m_Slots.size() && m_Slots[iSlot].Revert(0, Name); }
        bool Revert    ( int iSlot, std::uint64_t ID ) noexcept                                     { return std::size_t(iSlot) < m_Slots.size() && m_Slots[iSlot].Revert(ID, {}); }
        void SetMaterial( int iSlot, const material_instance_ref& MI ) noexcept                     { m_bOverrideMaterials = true; if (m_Slots.size() <= std::size_t(iSlot)) m_Slots.resize(std::size_t(iSlot) + 1); m_Slots[iSlot].m_Material = MI; }

        XPROPERTY_DEF
        ( "Mesh", mesh
        , obj_member<"Geom",      &mesh::m_Geom,      member_help<"The GeomStatic resource to draw">>
        , obj_member<"Materials", &mesh::m_Materials, member_dynamic_flags<+[](const mesh& M) { xproperty::flags::type F{}; F.m_bDontShow = F.m_bDontSave = M.m_Materials.empty(); return F; }>, member_help<"Deprecated (old levels): the editor moves these into the Slots">>
        , obj_member<"GIRole",    &mesh::m_GIRole,    member_enum_span<gi_role_v>, member_help<"What it does in the GI: ASSET as its geometry says; RECEIVER lit by it but neither blocking nor bouncing light (a prop, a character: moving it costs the GI nothing elsewhere); FULL blocks, bounces and emits">>
        , obj_member<"Override Materials", &mesh::m_bOverrideMaterials, member_help<"This entity's own materials: per material slot of the geometry, another material instance and its own values for the parameters it chooses. Off: the geometry's materials (the slots are not saved)">>
        , obj_member<"Slots", +[](mesh& O) -> std::vector<mesh_slot>& { return O.Slots(); }, member_array_size_readonly<>, member_ui_open<true>
            , member_dynamic_flags<+[](const mesh& M) { xproperty::flags::type F{}; F.m_bDontShow = !M.m_bOverrideMaterials; F.m_bDontSave = !M.m_bOverrideMaterials; return F; }>
            , member_help<"The geometry's material slots: the material instance each draws with and the values this entity draws its parameters with. An edit overrides; > reverts">>
        )
    };

    // A skinned geometry drawn with its materials, deformed by its skeleton's pose (an Animation on the same entity plays one of its clips; without
    // one: the rest pose). The materials as the Mesh's: per material slot of the geometry its own instance, or one the entity swaps in, with the
    // entity's own parameter values (mesh_slot). Not in the GI's geometry (it moves every frame): lit by it like a RECEIVER (the probes)
    struct skinned_mesh
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "SkinnedMesh" };

        geom_skin_ref                       m_Geom;
        bool                                m_bOverrideMaterials = false;
        std::vector<mesh_slot>              m_Slots;

        // The slots, one per material slot of the geometry (as loaded), each knowing the geometry's own instance (as mesh::Slots)
        std::vector<mesh_slot>& Slots( void ) noexcept
        {
            auto* pGeom = m_Geom.empty() ? nullptr : static_cast<xgeom_skin::xgpu::geom*>(xlioncore::resources::GetResource(geom_skin_type_guid_v.m_Value, m_Geom.m_Instance.m_Value));
            if (pGeom == nullptr) return m_Slots;
            const auto Defaults = pGeom->getDefaultMaterialInstances();
            if (m_Slots.size() != Defaults.size()) m_Slots.resize(Defaults.size());
            for (std::size_t i = 0; i < Defaults.size(); ++i)
            {
                m_Slots[i].m_Default.m_Instance.m_Value = xlioncore::resources::GuidOf(material_instance_type_guid_v.m_Value, Defaults[i].m_Instance.m_Value);
                if (m_Slots[i].m_Name.empty()) m_Slots[i].m_Name = std::format("Slot {}", i);
            }
            return m_Slots;
        }

        XPROPERTY_DEF
        ( "SkinnedMesh", skinned_mesh
        , obj_member<"Geom", &skinned_mesh::m_Geom, member_help<"The GeomSkin resource to draw: its skeleton is the one its pose comes from (an Animation on this entity plays it; without one, the rest pose)">>
        , obj_member<"Override Materials", &skinned_mesh::m_bOverrideMaterials, member_help<"This entity's own materials: per material slot of the geometry, another material instance and its own values for the parameters it chooses. Off: the geometry's materials">>
        , obj_member<"Slots", +[](skinned_mesh& O) -> std::vector<mesh_slot>& { return O.Slots(); }, member_array_size_readonly<>, member_ui_open<true>
            , member_dynamic_flags<+[](const skinned_mesh& M) { xproperty::flags::type F{}; F.m_bDontShow = !M.m_bOverrideMaterials; F.m_bDontSave = !M.m_bOverrideMaterials; return F; }>
            , member_help<"The geometry's material slots: the material instance each draws with and the values this entity draws its parameters with. An edit overrides; > reverts">>
        )
    };

    // Plays one clip of an animation package on the entity's Skinned Mesh (its skeleton's bones): the Animation system moves Time on while the game
    // runs (Play); the renderer poses the skeleton at Time. The shape of a single-clip player (Unity's legacy Animation, Godot's AnimationPlayer,
    // Unreal's single-node mode): which clip, where in it, how fast, looping or not, playing or not - a state machine or blending would sit above
    // it later. Outside Play the pose stays at Time (edit Time to look through the clip)
    struct animation
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "Animation" };

        anim_package_ref    m_Package;              // empty: the Skinned Mesh's geometry's own animations (GeomSkin's AnimationRef)
        int                 m_Clip      = 0;        // ponytail: by index (the compiled clips keep only their names' hashes); by name when a tool lists them
        float               m_Time      = 0;        // seconds into the clip
        float               m_Speed     = 1;
        bool                m_bLoop     = true;
        bool                m_bPlaying  = true;

        XPROPERTY_DEF
        ( "Animation", animation
        , obj_member<"Package", &animation::m_Package, member_help<"The animation package whose clip plays (empty: the Skinned Mesh's geometry's own animations). Its skeleton must be the geometry's">>
        , obj_member<"Clip",    &animation::m_Clip,    member_help<"Which clip of the package plays (its index)">>
        , obj_member<"Time",    &animation::m_Time,    member_help<"Seconds into the clip: the pose drawn. The Animation system moves it on while the game runs">>
        , obj_member<"Speed",   &animation::m_Speed,   member_help<"How fast time moves in the clip: 1 as authored, 0 frozen, negative backwards">>
        , obj_member<"Loop",    &animation::m_bLoop,   member_help<"At the end the clip starts again; off: it stops at its last frame (Playing turns off)">>
        , obj_member<"Playing", &animation::m_bPlaying, member_help<"Whether time moves while the game runs">>
        )
    };

    // The rows' marker and revert (member_override_state): Path names the slot and the row ("...Slots[G:s]/Parameters[G:r]..." or ".../Material").
    // The levels a row inherits through: the material (and its instances), the geometry's instance or the slot's swap, a prefab (pBase: the
    // prefab's Mesh, when this one is a prefab instance's), the entity. Overridden: the entity differs from what it inherits (from the prefab
    // when there is one). The revert goes one level up; when the prefab overrides the row too, a second choice goes past it
    inline void MeshSlotOverrideState( const xproperty::type::object& Type, void* pMesh, void* pBase, std::string_view Path, bool& bOverridden
                                     , std::vector<xproperty::settings::member_override_state_t::revert>* pChoices ) noexcept
    {
        bOverridden = false;
        const auto IndexAfter = [&](std::string_view Key, std::size_t& End) -> int
        {
            const auto At = Path.find(Key);
            if (At == std::string_view::npos) return -1;
            End = Path.find(']', At);
            return End == std::string_view::npos ? -1 : std::atoi(std::string(Path.substr(At + Key.size(), End - At - Key.size())).c_str());
        };
        std::size_t SlotEnd = 0, RowEnd = 0;
        const int iSlot = IndexAfter("Slots[G:", SlotEnd);
        // (the slots' owner: a Mesh or a Skinned Mesh)
        const bool bSkinned = &Type == xproperty::getObjectByType<skinned_mesh>();
        auto& Slots = bSkinned ? static_cast<skinned_mesh*>(pMesh)->m_Slots : static_cast<mesh*>(pMesh)->m_Slots;
        if (iSlot < 0 || std::size_t(iSlot) >= Slots.size()) return;
        auto& Slot = Slots[iSlot];
        static const mesh_slot s_None{};
        const auto* pBSlots   = pBase == nullptr ? nullptr : bSkinned ? &static_cast<const skinned_mesh*>(pBase)->m_Slots : &static_cast<const mesh*>(pBase)->m_Slots;
        const auto* pB        = pBSlots;
        const auto& BaseSlot  = pB && std::size_t(iSlot) < pB->size() ? (*pB)[iSlot] : s_None;     // (a prefab without the slot: nothing set there)
        using revert          = xproperty::settings::member_override_state_t::revert;

        if (Path.substr(SlotEnd + 1).starts_with("/Material"))
        {
            const std::string Row = std::string(Path.substr(0, SlotEnd + 1)) + "/Material";
            const auto Choice = [&](const char* pLabel, const material_instance_ref& To)
            {
                revert R{ .m_Label = pLabel, .m_Path = Row };
                R.m_Before.set<xresource::full_guid>({ .m_Instance = (Slot.m_Material.m_Instance.m_Value ? Slot.m_Material : Slot.m_Default).m_Instance, .m_Type = material_instance_ref::m_Type });
                R.m_After.set<xresource::full_guid>({ .m_Instance = To.m_Instance, .m_Type = material_instance_ref::m_Type });
                pChoices->push_back(std::move(R));
            };
            if (pB)
            {
                bOverridden = Slot.m_Material.m_Instance.m_Value != BaseSlot.m_Material.m_Instance.m_Value;
                if (!pChoices || !bOverridden) return;
                Choice("Revert to the prefab", BaseSlot.m_Material.m_Instance.m_Value ? BaseSlot.m_Material : Slot.m_Default);
                if (BaseSlot.m_Material.m_Instance.m_Value && Slot.m_Material.m_Instance.m_Value) Choice("Revert to the geometry's material instance", Slot.m_Default);
                return;
            }
            bOverridden = Slot.m_Material.m_Instance.m_Value != 0;
            if (pChoices && bOverridden) Choice("Revert to the geometry's material instance", Slot.m_Default);
            return;
        }

        const auto RowAt = Path.find("Parameters[G:");
        RowEnd = RowAt == std::string_view::npos ? std::string_view::npos : Path.find(']', RowAt);
        if (RowEnd == std::string_view::npos) return;
        const std::uint64_t RowKey = std::strtoull(std::string(Path.substr(RowAt + 13, RowEnd - RowAt - 13)).c_str(), nullptr, 10);     // (by key: "Parameters[G:key]")
        const auto* pRow = Slot.Rows().find(RowKey);
        if (pRow == nullptr) return;
        const auto& Row  = *pRow;
        const auto* pOwn = Row.Entry();
        const std::string Text = Row.getOverride();
        const auto Choice = [&](const char* pLabel, std::string To)
        {
            revert R{ .m_Label = pLabel, .m_Path = std::string(Path.substr(0, RowEnd + 1)) + "/Override" };
            R.m_Before.set<std::string>(Text);
            R.m_After.set<std::string>(std::move(To));
            pChoices->push_back(std::move(R));
        };
        const char* pToInstance = Row.m_bOrphan ? "Clear (not applied: the material has no such parameter)" : "Revert to the material instance";
        if (pB)
        {
            const std::string BaseText = mesh_param_row::FormatOverride(Row.m_Type, BaseSlot.Find(Row.m_ID, Row.m_Name));
            bOverridden = Text != BaseText;
            if (!pChoices || !bOverridden) return;
            Choice("Revert to the prefab", BaseText);
            if (!BaseText.empty() && !Text.empty()) Choice(pToInstance, {});           // (past the prefab: unless it is there already)
            return;
        }
        bOverridden = pOwn != nullptr;
        if (pChoices && bOverridden) Choice(pToInstance, {});
    }

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
        float           m_SourceAngle   = 0.5357f;          // the sun's disc, degrees across (scene's sun_light::m_SourceAngle): contact-hardening

        XPROPERTY_DEF
        ( "LightSun", light_sun
        , obj_member<"Color",          &light_sun::m_Color>
        , obj_member<"Intensity",      &light_sun::m_Intensity>
        , obj_member<"CastShadows",    &light_sun::m_bCastShadows>
        , obj_member<"ShadowTileSize", &light_sun::m_ShadowTileSize, member_help<"World size of the finest shadow tile: smaller is sharper near the camera">>
        , obj_member<"GI",             &light_sun::m_bGI, member_help<"Bounce light (global illumination). Off: direct light only, no bounce and no GI cost - for short-lived or fast lights such as bullets">>
        , obj_member<"SourceAngle",    &light_sun::m_SourceAngle, member_ui<float>::scroll_bar<0.0f, 10.0f>, member_help<"The sun's disc, degrees across (the real sun: about 0.53). Shadows are sharp where the caster touches the ground and soften with the distance to it; 0: hard shadows">>
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
        float           m_SourceRadius  = 0.0f;             // the emitter's size (world units): contact-hardening shadows; 0: hard

        XPROPERTY_DEF
        ( "LightPoint", light_point
        , obj_member<"Color",       &light_point::m_Color>
        , obj_member<"Intensity",   &light_point::m_Intensity>
        , obj_member<"Radius",      &light_point::m_Radius, member_help<"No light at and beyond this distance">>
        , obj_member<"CastShadows", &light_point::m_bCastShadows>
        , obj_member<"GI",          &light_point::m_bGI, member_help<"Bounce light (global illumination). Off: direct light only, no bounce and no GI cost - for short-lived or fast lights such as bullets">>
        , obj_member<"SourceRadius", &light_point::m_SourceRadius, member_ui<float>::scroll_bar<0.0f, 1.0f>, member_help<"The size of the emitter (a bulb, a flame), in world units. Shadows are sharp where the caster touches the surface and soften with the distance to it; 0: hard shadows">>
        )
    };

    // A reflection probe at the entity's position (scene::reflection_probe): the renderer captures the scene around it - static and
    // moving objects alike - and surfaces in its box reflect it, box-projected, by roughness (the rough reflections; the traced ones
    // serve the sharp: Environment's ReflectionMaxRoughness). Captured nearest to the camera first, a few faces a frame, again when
    // a light reaching it, an object in its box or the sky changes (Unreal's runtime captures: research/reflection_probes_competitors.md)
    struct reflection_probe_component
    {
        constexpr static auto typedef_v = xecs::component::type::data{ .m_pName = "ReflectionProbe" };

        xmath::fvec3    m_Extents       = { 5, 5, 5 };      // half the box (world units, around the entity): its influence and its parallax
        float           m_Blend         = 0.1f;             // the box's faces fade it over this distance (1 m left a room's spheres' lower metre to the flat ambient: black; Unreal's box transition is ~1 cm) (1 m left the lower metre of a room's spheres to the flat ambient: black; Unreal's box transition is ~1 cm)

        XPROPERTY_DEF
        ( "ReflectionProbe", reflection_probe_component
        , obj_member<"Extents", &reflection_probe_component::m_Extents, member_help<"Half the size of its box, around the entity: the surfaces inside reflect this probe, projected onto the box (a room's walls: the box fitted to them)">>
        , obj_member<"Blend",   &reflection_probe_component::m_Blend, member_help<"The box's faces fade the probe out over this distance (world units): no seam where two probes meet">>
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
        float           m_OuterAngle    = 0.6f;            // radians: no light outside (shadowed up to about 1.45)
        bool            m_bCastShadows  = true;
        bool            m_bGI           = true;             // in the GI's lights (its bounce); off: direct light only
        float           m_SourceRadius  = 0.0f;             // the emitter's size (world units): contact-hardening shadows; 0: hard

        XPROPERTY_DEF
        ( "LightSpot", light_spot
        , obj_member<"Color",       &light_spot::m_Color>
        , obj_member<"Intensity",   &light_spot::m_Intensity>
        , obj_member<"Radius",      &light_spot::m_Radius>
        , obj_member<"InnerAngle",  &light_spot::m_InnerAngle, member_help<"Radians from the axis inside which the light is full">>
        , obj_member<"OuterAngle",  &light_spot::m_OuterAngle, member_help<"Radians from the axis beyond which there is no light (its shadow reaches up to about 1.45)">>
        , obj_member<"CastShadows", &light_spot::m_bCastShadows>
        , obj_member<"GI",          &light_spot::m_bGI, member_help<"Bounce light (global illumination). Off: direct light only, no bounce and no GI cost - for short-lived or fast lights such as bullets">>
        , obj_member<"SourceRadius", &light_spot::m_SourceRadius, member_ui<float>::scroll_bar<0.0f, 1.0f>, member_help<"The size of the emitter (a bulb, a lens), in world units. Shadows are sharp where the caster touches the surface and soften with the distance to it; 0: hard shadows">>
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
        bool            m_bGIEmitterLights = false;         // the emissive surfaces as lights: their direct light by lighting (contact, highlights), the GI keeps their bounces
        int             m_GIRaysPerSurfel = 8;              // rays per surfel update
        int             m_GIFilterPasses  = 4;              // the GI light's spatial filter passes (smoother with more)
        bool            m_bGIMoments      = true;           // the reads' depth-moment visibility test
        float           m_ReflectionMaxRoughness = 0.0f;    // traced reflections (they need the GI): the receivers up to this roughness; 0: off (the probes)
        bool            m_bReflectionFloor   = false;       // a floor is a planar receiver: the scene mirrored in it (exact reflections)
        bool            m_bReflectionFloorAuto = true;      // that floor: the one under the camera (else at m_ReflectionFloorY)
        float           m_ReflectionFloorY   = 0.0f;
        bool            m_bReflectionDebug   = false;       // every lit pixel shows only its environment / reflection specular
        float           m_ReflectionTextureMaxRoughness = 0.3f;     // the receivers up to this roughness see the hits' own material (textures)
        float           m_ReflectionTextureMinReflectance = 0.6f;   // ... when their reflection shows at least this much (visible reflectance x sharpness)

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
        , obj_member<"GIEmitterLights", &environment::m_bGIEmitterLights, member_help<"Self-illuminated surfaces as lights: their direct light shaded per pixel by the lighting (sharp contact, highlights, shadowed by the screen trace and the GI's geometry), the GI keeps only their bounces. Off: the GI alone lights with them (soft, its surfels' resolution)">>
        , obj_member<"GIDebug",      &environment::m_GIDebug, member_enum_span<gi_debug_v>, member_help<"SURFELS: every pixel in the color of the surfel it reads most (black: no surfel reaches it); LIGHT: the GI light alone; SURFACE_COLOR: the surface color of the surfel each pixel reads most (the material capture: what the GI bounces); COVERAGE: which read lights each pixel - green its own surfels (per pixel), blue the vertices' (per pixel unsupported there), magenta none (no GI there: the ambient as without GI)">>
        , obj_member<"GIPerPixel",   &environment::m_bGIPerPixel, member_help<"Near meshes read the GI per pixel (a mesh needs no extra vertices for it); off: per vertex only (cheaper, needs dense meshes)">>
        , obj_member<"GIHalfRes",    &environment::m_bGIHalfRes, member_help<"The per-pixel GI read at half resolution, upsampled by depth and normal: a quarter of the reads (opaque surfaces; not in the GI debug views)">>
        , obj_member<"ReflectionMaxRoughness", &environment::m_ReflectionMaxRoughness, member_help<"Traced reflections (they need the GI): surfaces up to this roughness reflect the scene - a ray a pixel through the GI's geometry, its hit lit by the GI's model - instead of the reflection probes (from half of it they fade back to the probes). 0: off">>
        , obj_member<"ReflectionFloor",  &environment::m_bReflectionFloor, member_help<"The floor at ReflectionFloorY is a mirror plane: the scene is drawn mirrored in it (everything on screen, characters included) and the floor reflects that image (with ReflectionMaxRoughness above 0)">>
        , obj_member<"ReflectionFloorAuto", &environment::m_bReflectionFloorAuto, member_help<"The reflecting floor is the one under the camera (found every frame); off: the floor at ReflectionFloorY">>
        , obj_member<"ReflectionFloorY", &environment::m_ReflectionFloorY, member_help<"The height of the reflecting floor (world Y), when ReflectionFloorAuto is off">>
        , obj_member<"ReflectionTextureMaxRoughness", &environment::m_ReflectionTextureMaxRoughness, member_help<"Traced reflections on surfaces up to this roughness show what they hit with its material's own textures (each hit shaded by its material: costs more with more materials hit); rougher ones show its captured color (blended over the last quarter). 0: none">>
        , obj_member<"ReflectionTextureMinReflectance", &environment::m_ReflectionTextureMinReflectance, member_help<"Hits are shaded by their material only where the reflection shows: the surface's visible reflectance (its F0 through the view angle) times its sharpness at least this (1: mirrors and polished metals only; lower: more surfaces, more cost). A white dielectric reflects about 4%: its reflection keeps the captured color">>
        , obj_member<"ReflectionDebug",  &environment::m_bReflectionDebug, member_help<"Every lit pixel shows only its environment / reflection specular (what the probes or the traced reflections give it)">>
        )
    };

    // After the Primitive (50) and the Text
    XSCRIPT_REGISTER_COMPONENT(mesh,        "Rendering", 52)
    XSCRIPT_REGISTER_COMPONENT(light_sun,   "Rendering", 53)
    XSCRIPT_REGISTER_COMPONENT(light_point, "Rendering", 54)
    XSCRIPT_REGISTER_COMPONENT(light_spot,  "Rendering", 55)
    XSCRIPT_REGISTER_COMPONENT(environment, "Rendering", 56)
    XSCRIPT_REGISTER_COMPONENT(skinned_mesh, "Rendering", 58)
    XSCRIPT_REGISTER_COMPONENT(animation,   "Rendering", 59)
    XSCRIPT_REGISTER_COMPONENT(reflection_probe_component, "Rendering", 60)
    XSCRIPT_REGISTER_COMPONENT(mesh_material, "Deprecated", 57)     // (old files: the editor moves it into the Mesh, UpgradeOldMaterials; not offered to add)
}

#endif
