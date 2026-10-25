#ifndef XLIONRENDER_MATERIAL_OVERRIDES_H
#define XLIONRENDER_MATERIAL_OVERRIDES_H
#pragma once

// An entity's own values for a material slot (the Mesh's "Override materials"): its material instance with some parameters replaced. No resource
// here - the scene renderer hands over what the instance is made of (material_source) - so a test can drive it.
//
//  - A slot without overrides draws with the instance's own material (not here).
//  - Slots with the same instance and the same overrides share one renderer material (deduplicated by the overrides' hash): one parameter block,
//    one texture table; and every material of one shader and state shares one pipeline (instance::m_SharedPipelines) - overrides never make one.
//  - A slot whose overrides change: when it is the only one on its material, the material is changed in place (its block's vec4s, a texture's
//    id: SetMaterialParams / SetMaterialTexture); when others share it, the slot gets a private copy first (copy on write). A slot ending on
//    overrides another slot already has shares that one instead.
//  - A texture override of a material read through samplers 1-7 (a shader-file material) cannot change in place: a new material.

#include "../xlionrender_instance.h"
#include <bit>
#include <map>

namespace xlionrender
{
    // What overrides can address in a material: its manifest (xmaterial::manifest_entry), as the material resource carries it
    struct material_param
    {
        enum class type : std::uint8_t { FLOAT, FLOAT2, FLOAT3, FLOAT4, COLOR, TEXTURE };      // xmaterial::param_type's order
        std::uint64_t   m_ID            = 0;
        type            m_Type          = type::FLOAT;
        std::uint16_t   m_ByteOffset    = 0;        // into the block (not TEXTURE)
        std::uint8_t    m_Slot          = 0;        // TEXTURE: the texture slot (Textures[m_Slot + 1] of CreateMaterial)
    };

    // One overridden parameter: its value (linear), or its texture
    struct param_override
    {
        std::uint64_t   m_ID        = 0;
        float           m_V[4]      = { 0, 0, 0, 0 };
        xgpu::texture*  m_pTexture  = nullptr;      // TEXTURE
        std::uint64_t   m_TextureKey= 0;            // TEXTURE: what identifies it (its resource guid)
    };

    // A material instance as made into a renderer material (instance::CreateMaterial's arguments) and what its overrides can address
    struct material_source
    {
        std::uint64_t                       m_Key           = 0;        // identifies the instance and what it was loaded from
        const xgpu::shader*                 m_pFragment     = nullptr;
        std::span<xgpu::texture* const>     m_Textures;                 // slot 0: the shadow atlas (CreateMaterial's convention)
        material_state                      m_State;
        std::span<const float>              m_Block;
        bool                                m_bTextureTable = false;
        std::span<const material_param>     m_Params;
    };

    struct material_overrides
    {
        // One slot of one entity: the material it draws with now
        struct slot
        {
            std::uint64_t   m_Key       = 0;        // the cache entry (0: none)
            material_id     m_Id;
        };

        // Slot's material for Source with Overrides (sorted by id, at most one per id): shared, changed in place or made (an invalid id: it could
        // not be made). Call it every frame the slot is drawn; Release when it is not drawn any more
        material_id Acquire( instance& Renderer, const material_source& Source, std::span<const param_override> Overrides, slot& Slot ) noexcept
        {
            const std::uint64_t Key = KeyOf(Source.m_Key, Overrides);
            if( Slot.m_Key == Key ) return Slot.m_Id;

            // Shared with another slot already
            if( auto It = m_Entries.find(Key); It != m_Entries.end() )
            {
                Release(Renderer, Slot);
                ++It->second.m_nUsers;
                Slot = { Key, It->second.m_Id };
                return Slot.m_Id;
            }

            // Its own material (the only user) of the same instance: changed in place, then filed under the new key
            if( auto Old = m_Entries.find(Slot.m_Key); Old != m_Entries.end() && Old->second.m_nUsers == 1 && Old->second.m_SourceKey == Source.m_Key && Update(Renderer, Source, Overrides, Old->second) )
            {
                auto E = Old->second;
                m_Entries.erase(Old);
                m_Entries.emplace(Key, E);
                Slot.m_Key = Key;
                ++m_nInPlace;
                return Slot.m_Id;
            }

            // A new one (copy on write when the old one is shared: it stays with its other users)
            entry E{ .m_SourceKey = Source.m_Key, .m_nUsers = 1 };
            if( !Make(Renderer, Source, Overrides, E) ) return {};
            Release(Renderer, Slot);
            m_Entries.emplace(Key, E);
            Slot = { Key, E.m_Id };
            ++m_nMade;
            return Slot.m_Id;
        }

        void Release( instance& Renderer, slot& Slot ) noexcept
        {
            if( auto It = m_Entries.find(Slot.m_Key); It != m_Entries.end() && --It->second.m_nUsers == 0 )
            {
                Renderer.DestroyMaterial(It->second.m_Id);
                m_Entries.erase(It);
            }
            Slot = {};
        }

        void Clear( instance& Renderer ) noexcept
        {
            for( auto& [Key, E] : m_Entries ) Renderer.DestroyMaterial(E.m_Id);
            m_Entries.clear();
        }

        int Count( void ) const noexcept { return int(m_Entries.size()); }
        int m_nMade = 0, m_nInPlace = 0;        // (tests, stats)

    private:

        struct entry
        {
            material_id     m_Id;
            std::uint64_t   m_SourceKey = 0;
            int             m_nUsers    = 0;
            std::vector<std::uint64_t> m_Textures;      // per texture slot: the texture override's key in place now (0: the instance's)
        };
        std::map<std::uint64_t, entry> m_Entries;

        static std::uint64_t KeyOf( std::uint64_t SourceKey, std::span<const param_override> Overrides ) noexcept
        {
            std::uint64_t H = 14695981039346656037ull ^ SourceKey;
            const auto Mix = [&](std::uint64_t V) { for( int i = 0; i < 8; ++i ) { H ^= (V >> (8 * i)) & 0xff; H *= 1099511628211ull; } };
            for( const auto& O : Overrides )
            {
                Mix(O.m_ID);
                for( float f : O.m_V ) Mix(std::bit_cast<std::uint32_t>(f));
                Mix(O.m_TextureKey);
            }
            return H ? H : 1;
        }

        static const material_param* Find( const material_source& S, std::uint64_t ID ) noexcept
        {
            for( const auto& P : S.m_Params ) if( P.m_ID == ID ) return &P;
            return nullptr;
        }

        static int Components( material_param::type T ) noexcept
        {
            return T == material_param::type::FLOAT ? 1 : T == material_param::type::FLOAT2 ? 2 : T == material_param::type::FLOAT3 ? 3 : 4;
        }

        // The instance's block with the overrides written in, and its textures with theirs
        static void Resolve( const material_source& S, std::span<const param_override> Overrides, std::vector<float>& Block, std::vector<xgpu::texture*>& Textures ) noexcept
        {
            Block.assign(S.m_Block.begin(), S.m_Block.end());
            Textures.assign(S.m_Textures.begin(), S.m_Textures.end());
            for( const auto& O : Overrides )
            {
                const auto* P = Find(S, O.m_ID);
                if( P == nullptr ) continue;                                    // (a parameter the material no longer has)
                if( P->m_Type == material_param::type::TEXTURE )
                {
                    if( O.m_pTexture && std::size_t(P->m_Slot) + 1 < Textures.size() ) Textures[P->m_Slot + 1] = O.m_pTexture;
                    continue;
                }
                const std::size_t At = P->m_ByteOffset / 4;
                for( int c = 0; c < Components(P->m_Type) && At + c < Block.size(); ++c ) Block[At + c] = O.m_V[c];
            }
        }

        bool Make( instance& Renderer, const material_source& S, std::span<const param_override> Overrides, entry& E ) noexcept
        {
            std::vector<float>          Block;
            std::vector<xgpu::texture*> Textures;
            Resolve(S, Overrides, Block, Textures);
            if( Renderer.CreateMaterial(E.m_Id, *S.m_pFragment, Textures, S.m_State, Block, S.m_bTextureTable) ) return false;
            E.m_Textures.assign(Textures.size(), 0);
            for( const auto& O : Overrides ) if( const auto* P = Find(S, O.m_ID); P && P->m_Type == material_param::type::TEXTURE && P->m_Slot + 1u < E.m_Textures.size() ) E.m_Textures[P->m_Slot + 1] = O.m_TextureKey;
            return true;
        }

        // The material changed in place: its whole block, and the textures whose override changed (only a texture-table material can)
        bool Update( instance& Renderer, const material_source& S, std::span<const param_override> Overrides, entry& E ) noexcept
        {
            std::vector<float>          Block;
            std::vector<xgpu::texture*> Textures;
            Resolve(S, Overrides, Block, Textures);
            std::vector<std::uint64_t> Keys(Textures.size(), 0);
            for( const auto& O : Overrides ) if( const auto* P = Find(S, O.m_ID); P && P->m_Type == material_param::type::TEXTURE && P->m_Slot + 1u < Keys.size() ) Keys[P->m_Slot + 1] = O.m_TextureKey;
            if( Keys != E.m_Textures && !S.m_bTextureTable ) return false;
            if( Renderer.SetMaterialParams(E.m_Id, Block) ) return false;
            for( std::size_t i = 1; i < Keys.size(); ++i )
                if( Keys[i] != E.m_Textures[i] && Textures[i] && Renderer.SetMaterialTexture(E.m_Id, int(i) - 1, *Textures[i]) ) return false;
            E.m_Textures = std::move(Keys);
            return true;
        }
    };
}

#endif
