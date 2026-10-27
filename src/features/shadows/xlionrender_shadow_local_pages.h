#ifndef XLIONRENDER_SHADOW_LOCAL_PAGES_H
#define XLIONRENDER_SHADOW_LOCAL_PAGES_H
#pragma once

// Point and spot light shadows as virtual pages (documentation/lighting_strategy.md P1d; the sun's are xlionrender_shadow_vsm.h):
//  - A light owns a light slot (max_local_lights_v of them, matched from frame to frame by the light's key: position, radius, cone);
//    slot k owns faces 6k .. 6k+5: a point light's 6 cube faces (+X -X +Y -Y +Z -Z, shadows::CubeFaceAxis), a spot's first one (its
//    frustum). A face is a virtual map of level0_pages_v x level0_pages_v pages of page_texels_v (2048^2 texels), levels_v levels down
//    to one page; a page holds the face projection's hyperbolic depth (shadows::FaceW2C, [Near, Radius] -> [0, 1]), rendered with the
//    page's off-centre part of it (PageW2C).
//  - The pool: a square of slots (the local part of the atlas, 1024 slots of 128^2). A light's coarsest page per face is pinned: rendered
//    first every frame it is missing (mandatory, outside the budget), never evicted while the light is there - the fallback the shading
//    walks to. The other pages by request (the request pass over the main view's depth, a ring late, xlionrender_local_request_comp.glsl):
//    resident ones kept, missing ones admitted (a free slot, else the least recently requested unpinned page not asked for this frame),
//    rendered coarse first within the budget.
//  - The table (BuildTable): an entry per face and page, 16 bits (slot + 1, 0 none), a face's entries contiguous (face_entries_v): the
//    shading reads it from the cell buffer. A page a changed caster touches renders again when asked for; until then it keeps its last
//    render (stale, counted - as the maps before it).
//  - A light that changes (type, radius, cone) or goes frees its slot's pages: a new light (its key) starts again. A light that only
//    moves or turns (the same scene light, the same type: a torch, a car's lights) keeps its slot and its requests (a page's key is
//    relative to the light): its requested pages render again every frame it moves, mandatory as the pinned ones (Unreal re-renders a
//    moving light's pages uncached alike), the ones not asked for go - none keeps a render from where the light was.
// ponytail: a page's casters by a frustum test per caster and page on the CPU (the sun's pairs: P1c); dirty pages by a frustum test per
// changed box and resident page of the lights it reaches

#include "features/shadows/xlionrender_shadow_local.h"
#include "features/culling/xlionrender_frustum.h"
#include <algorithm>
#include <span>
#include <unordered_map>

namespace xlionrender::shadows
{
    struct local_pages
    {
        static constexpr int page_texels_v   = 128;
        static constexpr int level0_pages_v  = 16;              // a face's finest level: 16 x 16 pages (2048^2 texels)
        static constexpr int levels_v        = 5;               // 16, 8, 4, 2, 1 pages across
        static constexpr int face_entries_v  = 16 * 16 + 8 * 8 + 4 * 4 + 2 * 2 + 1;
        static constexpr int faces_per_light_v = 6;
        static constexpr int max_faces_v     = max_local_lights_v * faces_per_light_v;

        struct setup
        {
            int     m_PoolTexels = 4096;        // the pool's square (texels); its origin in the atlas is the caller's
            int     m_Budget     = 64;          // unpinned pages rendered a frame (pinned ones: always)
            float   m_ResolutionBias = 0;       // levels added to the request's (negative: finer)
            int Slots(void) const noexcept { return (m_PoolTexels / page_texels_v) * (m_PoolTexels / page_texels_v); }
            int SlotsPerRow(void) const noexcept { return m_PoolTexels / page_texels_v; }
        };

        struct key
        {
            int m_Face = 0, m_Level = 0, m_X = 0, m_Y = 0;
            bool operator==(const key&) const noexcept = default;
        };

        struct page_to_render
        {
            int             m_Light;                // the scene's local light
            key             m_Key;
            int             m_Slot;
            int             m_AtlasX, m_AtlasY;     // in the pool (the caller adds the pool's origin)
            xmath::fmat4    m_W2C;                  // world -> the page's clip space
            bool            m_bPinned;
        };

        struct stats
        {
            int m_Requested = 0, m_Rendered = 0, m_RenderedPinned = 0, m_Rejected = 0, m_Dirtied = 0, m_Resident = 0, m_ResidentPinned = 0, m_Stale = 0, m_Lights = 0;
        };

        static constexpr int LevelBase(int L) noexcept { int b = 0; for (int l = 0; l < L; ++l) b += (level0_pages_v >> l) * (level0_pages_v >> l); return b; }
        static constexpr int Entry(const key& K) noexcept { return K.m_Face * face_entries_v + LevelBase(K.m_Level) + K.m_Y * (level0_pages_v >> K.m_Level) + K.m_X; }
        static key KeyOf(int i) noexcept
        {
            key K{ i / face_entries_v, 0, 0, 0 };
            int e = i % face_entries_v;
            while (e >= (level0_pages_v >> K.m_Level) * (level0_pages_v >> K.m_Level)) { e -= (level0_pages_v >> K.m_Level) * (level0_pages_v >> K.m_Level); ++K.m_Level; }
            K.m_X = e % (level0_pages_v >> K.m_Level); K.m_Y = e / (level0_pages_v >> K.m_Level);
            return K;
        }

        void Init(const setup& Setup) noexcept
        {
            m_Setup = Setup;
            m_Slots.assign(std::size_t(Setup.Slots()), slot{});
            m_Map.clear();
            m_Waiting.clear();
            for (auto& L : m_Lights) L = {};
        }
        const setup& getSetup(void) const noexcept { return m_Setup; }
        void setBudget(int Budget) noexcept { m_Setup.m_Budget = Budget; }

        // The resolution bias of a light slot's requests: a light moving now or within moving_frames_v frames asks for pages a level coarser
        // (4 times fewer, rendered again each frame it moves; Unreal: r.Shadow.Virtual.ResolutionLodBiasLocalMoving, 1)
        static constexpr std::uint64_t moving_frames_v = 8;
        float MovingBias(int LightSlot) const noexcept
        {
            const auto& L = m_Lights[LightSlot];
            return L.m_bUsed && L.m_MovedFrame != 0 && m_Frame - L.m_MovedFrame < moving_frames_v ? 1.0f : 0.0f;
        }

        // A light slot's generation (a request names the generation it was made for; the light changed since: dropped)
        std::uint32_t Generation(int LightSlot) const noexcept { return m_Lights[LightSlot].m_Generation; }
        // The scene light a light slot holds (-1: none) and a scene light's slot (-1: none)
        int LightOfSlot(int LightSlot) const noexcept { return m_Lights[LightSlot].m_bUsed ? m_Lights[LightSlot].m_iLight : -1; }
        int SlotOfLight(int iLight) const noexcept { return iLight < int(m_SlotOf.size()) ? m_SlotOf[iLight] : -1; }

        // The requests a request pass made (a bit per table entry), with the generations of that frame's light slots
        static void DecodeRequests(std::span<const std::uint32_t> Bits, std::span<const std::uint32_t> Generations, const local_pages& P, std::vector<key>& Out) noexcept
        {
            Out.clear();
            const int n = std::min(int(Bits.size()) * 32, max_faces_v * face_entries_v);
            for (int w = 0; w < (n + 31) / 32; ++w)
            {
                if (Bits[std::size_t(w)] == 0) continue;
                for (int b = 0; b < 32; ++b)
                {
                    if ((Bits[std::size_t(w)] & (1u << b)) == 0) continue;
                    const key K = KeyOf(w * 32 + b);
                    const int LS = K.m_Face / faces_per_light_v;
                    if (LS < int(Generations.size()) && Generations[std::size_t(LS)] == P.Generation(LS)) Out.push_back(K);
                }
            }
        }

        // The frame: the lights matched to their slots (a new or changed light: a new slot, its old pages freed), the pages a changed caster
        // touches dirtied, the pinned pages, then the requested ones within the budget (bCache false: every resident page again, no budget)
        void Update(const std::vector<local_light>& Lights, std::span<const xmath::fbbox> Changed, bool bCache, std::span<const key> Requests, std::uint64_t Frame) noexcept
        {
            m_ToRender.clear();
            m_Stats = {};
            m_Frame = Frame;
            const int nLights = std::min(int(Lights.size()), max_local_lights_v);
            m_SlotOf.assign(Lights.size(), -1);

            // Match: a slot keeps its light while the key is the same; then a slot whose scene light only moved or turned keeps it (moved);
            // the rest go (their pages freed), new lights take free slots
            for (auto& L : m_Lights) L.m_bSeen = L.m_bMoved = false;
            for (int i = 0; i < nLights; ++i)
            {
                const auto& L = Lights[i];
                if (!L.m_bCastShadows || L.m_Radius <= 0) continue;
                const std::uint64_t K = LightKey(L);
                for (int s = 0; s < max_local_lights_v; ++s)
                    if (m_Lights[s].m_bUsed && !m_Lights[s].m_bSeen && m_Lights[s].m_Key == K) { m_Lights[s].m_bSeen = true; m_Lights[s].m_iLight = i; m_SlotOf[i] = s; break; }
            }
            for (int s = 0; s < max_local_lights_v; ++s)
            {
                auto& LS = m_Lights[s];
                if (!LS.m_bUsed || LS.m_bSeen) continue;
                const int i = LS.m_iLight;
                if (i >= 0 && i < nLights && m_SlotOf[i] < 0 && Lights[i].m_bCastShadows && Lights[i].m_Radius > 0 && ShapeKey(Lights[i]) == LS.m_Shape)
                {
                    LS.m_Key = LightKey(Lights[i]);
                    LS.m_bSeen = LS.m_bMoved = true;
                    LS.m_MovedFrame = Frame;
                    m_SlotOf[i] = s;
                }
            }
            for (int s = 0; s < max_local_lights_v; ++s) if (m_Lights[s].m_bUsed && !m_Lights[s].m_bSeen) FreeLight(s);
            for (int i = 0; i < nLights; ++i)
            {
                const auto& L = Lights[i];
                if (!L.m_bCastShadows || L.m_Radius <= 0 || m_SlotOf[i] >= 0) continue;
                for (int s = 0; s < max_local_lights_v; ++s)
                    if (!m_Lights[s].m_bUsed)
                    {
                        auto& LS = m_Lights[s];
                        LS = { LightKey(L), LS.m_Generation + 1, i, true, true, false, ShapeKey(L) };
                        m_SlotOf[i] = s;
                        break;
                    }
            }
            for (const auto& L : m_Lights) m_Stats.m_Lights += L.m_bUsed;

            // A moved light's pages: the pinned and the requested ones render again (mandatory), the others go
            {
                bool bAnyMoved = false;
                for (const auto& L : m_Lights) bAnyMoved |= L.m_bMoved;
                if (bAnyMoved)
                {
                    std::unordered_map<int, int> Asked;
                    for (const auto& K : Requests) Asked[Entry(K)] = 1;
                    for (int s = 0; s < int(m_Slots.size()); ++s)
                    {
                        auto& S = m_Slots[s];
                        if (!S.m_bUsed || !m_Lights[S.m_Key.m_Face / faces_per_light_v].m_bMoved) continue;
                        if (S.m_bPinned || Asked.contains(Entry(S.m_Key))) S.m_bDirty = true;
                        else Free(s);
                    }
                }
            }

            // Dirty: every resident page of a light a changed caster reaches whose frustum holds it (bCache false: all of them)
            for (int s = 0; s < int(m_Slots.size()) && (!bCache || !Changed.empty()); ++s)
            {
                auto& S = m_Slots[s];
                if (!S.m_bUsed || !S.m_bValid) continue;
                bool bDirty = !bCache;
                if (!bDirty)
                {
                    const auto& L = Lights[std::size_t(m_Lights[S.m_Key.m_Face / faces_per_light_v].m_iLight)];
                    bool bNear = false;
                    for (const auto& B : Changed) if (Overlaps(B, L.m_Position, L.m_Radius)) { bNear = true; break; }
                    if (bNear)
                    {
                        const frustum F = frustum::FromW2C(PageW2C(L, S.m_Key));
                        for (std::size_t c = 0; !bDirty && c < Changed.size(); ++c) bDirty = F.Overlaps(Changed[c], false);
                    }
                }
                if (bDirty && !S.m_bDirty) { S.m_bDirty = true; ++m_Stats.m_Dirtied; }
            }

            // The pinned pages: each light face's coarsest page, mandatory
            for (int ls = 0; ls < max_local_lights_v; ++ls)
            {
                if (!m_Lights[ls].m_bUsed) continue;
                const auto& L = Lights[std::size_t(m_Lights[ls].m_iLight)];
                for (int f = 0; f < FacesOf(L); ++f)
                {
                    const key K{ ls * faces_per_light_v + f, levels_v - 1, 0, 0 };
                    int s = Find(K);
                    if (s < 0) { s = Admit(K, true); if (s < 0) { ++m_Stats.m_Rejected; continue; } }
                    m_Slots[s].m_LastRequested = Frame;
                    if (!m_Slots[s].m_bValid || m_Slots[s].m_bDirty) Render(L, m_Lights[ls].m_iLight, s, true);
                }
            }

            // The requested pages: resident ones kept (rendered again when dirty), missing ones admitted; coarse first within the budget
            std::vector<key> Want;
            for (const auto& K : Requests)
            {
                const int ls = K.m_Face / faces_per_light_v;
                if (ls >= max_local_lights_v || !m_Lights[ls].m_bUsed || K.m_Level >= levels_v - 1) continue;      // (the coarsest: pinned)
                if (K.m_Face % faces_per_light_v >= FacesOf(Lights[std::size_t(m_Lights[ls].m_iLight)])) continue;
                ++m_Stats.m_Requested;
                if (const int s = Find(K); s >= 0) m_Slots[s].m_LastRequested = Frame;
                Want.push_back(K);
            }
            // (a page waiting urgent_age_v frames or more goes first, the longest first: pages dirtied every frame by a mover never starve the others)
            std::unordered_map<int, std::uint64_t> Waiting;
            for (const auto& K : Want)
            {
                const auto It = m_Waiting.find(Entry(K));
                Waiting[Entry(K)] = It == m_Waiting.end() ? Frame : It->second;
            }
            m_Waiting = std::move(Waiting);
            const auto Age = [&](const key& K) { return Frame - m_Waiting[Entry(K)]; };
            std::sort(Want.begin(), Want.end(), [&](const key& a, const key& b)
            {
                const bool aUrgent = Age(a) >= std::uint64_t(urgent_age_v), bUrgent = Age(b) >= std::uint64_t(urgent_age_v);
                if (aUrgent != bUrgent) return aUrgent;
                if (aUrgent && Age(a) != Age(b)) return Age(a) > Age(b);
                return a.m_Level > b.m_Level;
            });
            int Budget = bCache ? m_Setup.m_Budget : 1 << 30;
            for (const auto& K : Want)
            {
                int s = Find(K);
                if (s >= 0 && m_Slots[s].m_bValid && !m_Slots[s].m_bDirty) { m_Waiting.erase(Entry(K)); continue; }
                const bool bMoved = m_Lights[K.m_Face / faces_per_light_v].m_bMoved;
                if (Budget <= 0 && !bMoved) continue;
                if (s < 0) { s = Admit(K, false); if (s < 0) { ++m_Stats.m_Rejected; continue; } }
                m_Slots[s].m_LastRequested = Frame;
                const int iLight = m_Lights[K.m_Face / faces_per_light_v].m_iLight;
                Render(Lights[std::size_t(iLight)], iLight, s, false);
                m_Waiting.erase(Entry(K));
                if (!bMoved) --Budget;
            }
            if (!bCache)        // a full rebuild: every resident page again
                for (int s = 0; s < int(m_Slots.size()); ++s)
                    if (m_Slots[s].m_bUsed && m_Slots[s].m_bDirty)
                    {
                        const int iLight = m_Lights[m_Slots[s].m_Key.m_Face / faces_per_light_v].m_iLight;
                        Render(Lights[std::size_t(iLight)], iLight, s, m_Slots[s].m_bPinned);
                    }

            for (const auto& S : m_Slots)
            {
                if (!S.m_bUsed) continue;
                ++(S.m_bPinned ? m_Stats.m_ResidentPinned : m_Stats.m_Resident);
                m_Stats.m_Stale += S.m_bValid && S.m_bDirty;
            }
        }

        // The table: every face of the light slots up to the last one used (TableEntries()), 16 bits an entry: slot + 1 of a rendered page
        int TableEntries(void) const noexcept
        {
            int Last = -1;
            for (int s = 0; s < max_local_lights_v; ++s) if (m_Lights[s].m_bUsed) Last = s;
            return (Last + 1) * faces_per_light_v * face_entries_v;
        }
        void BuildTable(std::vector<std::uint32_t>& Out) const noexcept
        {
            Out.assign(std::size_t((TableEntries() + 1) / 2), 0u);
            for (int s = 0; s < int(m_Slots.size()); ++s)
            {
                const auto& S = m_Slots[s];
                if (!S.m_bUsed || !S.m_bValid) continue;
                const int i = Entry(S.m_Key);
                if (std::size_t(i >> 1) < Out.size()) Out[std::size_t(i) >> 1] |= std::uint32_t(s + 1) << ((i & 1) * 16);
            }
        }

        // Invariant 1 on a table: every entry names a used, rendered slot holding exactly that page
        bool Validate(std::span<const std::uint32_t> Table) const noexcept
        {
            for (int i = 0; i < int(Table.size()) * 2; ++i)
            {
                const std::uint32_t E = (Table[std::size_t(i) >> 1] >> ((i & 1) * 16)) & 0xFFFFu;
                if (E == 0) continue;
                const int s = int(E) - 1;
                if (s >= int(m_Slots.size()) || !m_Slots[s].m_bUsed || !m_Slots[s].m_bValid || !(m_Slots[s].m_Key == KeyOf(i))) return false;
            }
            return true;
        }

        // A face's world -> clip: the light's position, the face's axis and half field of view, [Near, Radius] -> [0, 1]
        static xmath::fmat4 FaceProjection(const local_light& L, int Face) noexcept
        {
            const bool bPoint = L.m_Type == local_light::type::POINT;
            return FaceW2C(L.m_Position, bPoint ? CubeFaceAxis(Face) : L.m_Direction.NormalizeCopy(), bPoint ? 1.0f : SpotTanHalf(L), LocalNear(L.m_Radius), L.m_Radius);
        }

        // A page's world -> clip: its part of the face's projection stretched over its clip square (x' = x n - w (2 i + 1 - n))
        static xmath::fmat4 PageW2C(const local_light& L, const key& K) noexcept
        {
            xmath::fmat4 M = FaceProjection(L, K.m_Face % faces_per_light_v);
            const float n = float(level0_pages_v >> K.m_Level);
            const float Ox = 2.0f * float(K.m_X) + 1.0f - n, Oy = 2.0f * float(K.m_Y) + 1.0f - n;
            for (int c = 0; c < 4; ++c)
            {
                M(0, c) = M(0, c) * n - M(3, c) * Ox;
                M(1, c) = M(1, c) * n - M(3, c) * Oy;
            }
            return M;
        }

        static int FacesOf(const local_light& L) noexcept { return L.m_Type == local_light::type::POINT ? 6 : 1; }

        std::vector<page_to_render> m_ToRender;
        stats                       m_Stats;

    private:

        struct slot
        {
            key             m_Key;
            std::uint64_t   m_LastRequested = 0;
            bool            m_bUsed = false, m_bValid = false, m_bDirty = false, m_bPinned = false;
        };
        struct light_slot
        {
            std::uint64_t   m_Key = 0;
            std::uint32_t   m_Generation = 0;
            int             m_iLight = -1;
            bool            m_bUsed = false, m_bSeen = false;
            bool            m_bMoved = false;           // this frame: the same light, moved or turned
            std::uint64_t   m_Shape = 0;                // its type, radius and cone (ShapeKey): what a move keeps
            std::uint64_t   m_MovedFrame = 0;           // the last frame it moved (0: never)
        };

        int Find(const key& K) const noexcept { const auto It = m_Map.find(Entry(K)); return It == m_Map.end() ? -1 : It->second; }

        // A slot for K: a free one, else the least recently requested unpinned page not asked for this frame (-1: none)
        int Admit(const key& K, bool bPinned) noexcept
        {
            int Best = -1;
            for (int s = 0; s < int(m_Slots.size()); ++s)
            {
                const auto& S = m_Slots[s];
                if (!S.m_bUsed) { Best = s; break; }
                if (S.m_bPinned || S.m_LastRequested == m_Frame) continue;
                if (Best < 0 || S.m_LastRequested < m_Slots[Best].m_LastRequested) Best = s;
            }
            if (Best < 0) return -1;
            if (m_Slots[Best].m_bUsed) Free(Best);
            m_Slots[Best] = { K, m_Frame, true, false, false, bPinned };
            m_Map[Entry(K)] = Best;
            return Best;
        }

        void Free(int s) noexcept
        {
            m_Map.erase(Entry(m_Slots[s].m_Key));
            m_Slots[s] = {};
        }

        void FreeLight(int LightSlot) noexcept
        {
            for (int s = 0; s < int(m_Slots.size()); ++s) if (m_Slots[s].m_bUsed && m_Slots[s].m_Key.m_Face / faces_per_light_v == LightSlot) Free(s);
            auto& L = m_Lights[LightSlot];
            L.m_bUsed = false;
            L.m_iLight = -1;
            ++L.m_Generation;               // (requests made for it are dropped)
        }

        void Render(const local_light& L, int iLight, int s, bool bPinned) noexcept
        {
            auto& S = m_Slots[s];
            S.m_bValid = true;
            S.m_bDirty = false;
            const int R = m_Setup.SlotsPerRow();
            m_ToRender.push_back({ iLight, S.m_Key, s, (s % R) * page_texels_v, (s / R) * page_texels_v, PageW2C(L, S.m_Key), bPinned });
            ++(bPinned ? m_Stats.m_RenderedPinned : m_Stats.m_Rendered);
        }

        static std::uint64_t ShapeKey(const local_light& L) noexcept
        {
            const bool  bSpot = L.m_Type == local_light::type::SPOT;
            const float V[] = { float(L.m_Type), L.m_Radius, bSpot ? L.m_OuterAngle : 0 };
            std::uint64_t H = 1469598103934665603ull;
            for (auto* p = reinterpret_cast<const std::uint8_t*>(V); p != reinterpret_cast<const std::uint8_t*>(V + std::size(V)); ++p) H = (H ^ *p) * 1099511628211ull;
            return H;
        }

        static std::uint64_t LightKey(const local_light& L) noexcept
        {
            const bool  bSpot = L.m_Type == local_light::type::SPOT;
            const float V[] = { float(L.m_Type), L.m_Position.m_X, L.m_Position.m_Y, L.m_Position.m_Z, L.m_Radius
                              , bSpot ? L.m_Direction.m_X : 0, bSpot ? L.m_Direction.m_Y : 0, bSpot ? L.m_Direction.m_Z : 0, bSpot ? L.m_OuterAngle : 0 };
            std::uint64_t H = 1469598103934665603ull;
            for (auto* p = reinterpret_cast<const std::uint8_t*>(V); p != reinterpret_cast<const std::uint8_t*>(V + std::size(V)); ++p) H = (H ^ *p) * 1099511628211ull;
            return H;
        }

        setup                                       m_Setup;
        std::vector<slot>                           m_Slots;
        std::unordered_map<int, int>                m_Map;          // table entry -> slot
        std::array<light_slot, max_local_lights_v>  m_Lights{};
        std::vector<int>                            m_SlotOf;       // scene light -> light slot (-1: none)
        std::unordered_map<int, std::uint64_t>      m_Waiting;      // a wanted page not rendered yet: the frame it was first wanted
        std::uint64_t                               m_Frame = 0;
    };
}

#endif
