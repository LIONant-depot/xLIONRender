#ifndef XLIONRENDER_INSTANCE_SLOTS_H
#define XLIONRENDER_INSTANCE_SLOTS_H
#pragma once

// Persistent instance data (research doc: per-draw data). A draw that carries an instance id (scene::draw::m_Instance, low 32
// bits; the caller owns the ids - an ECS entity, an editor's preview, ...) keeps a slot in the instance buffer across frames:
// its data is written when it changes, not when it becomes visible. In tight spaces visibility flips every few steps; the
// slots do not care.
//  - The instance buffer is device-local; the renderer writes the pending slots (Flush) into a CPU ring and the GPU copies
//    them over (xgpu::window::CopyBuffer), as Unreal's GPUScene and Bevy upload their deltas.
//  - A slot is not freed when its id is missing from a frame (invisible is not dead): slots are evicted least recently
//    seen first, only when the buffer is full.

#include "scene/xlionrender_scene.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace xlionrender
{
    struct instance_slots
    {
        static constexpr std::uint32_t none_v = ~0u;

        void Init(int Capacity) noexcept
        {
            m_Slots.assign(Capacity, {});
            m_Free.clear();
            for (int i = Capacity - 1; i >= 0; --i) m_Free.push_back(std::uint32_t(i));
            m_ByID.clear();
            m_Pending.clear();
        }

        // The slot of ID this frame (none_v: no room - every slot was seen this frame). New or changed (transform, constant
        // material or entity parameters): pending.
        // bChanged: the slot is new, or its transform or constant material changed (what the shadow casters key on; entity
        // parameters only change how a material shades it).
        std::uint32_t Acquire(std::uint32_t ID, const xmath::fmat4& L2W, std::uint64_t Frame, const material& Mat = {}, bool* pChanged = nullptr, const entity_params& Ent = {}) noexcept
        {
            if (pChanged) *pChanged = true;
            if (auto It = m_ByID.find(ID); It != m_ByID.end())
            {
                auto& S = m_Slots[It->second];
                S.m_LastSeen = Frame;
                const bool bMoved   = std::memcmp(&S.m_L2W, &L2W, sizeof(L2W)) != 0;
                const bool bMatEdit = std::memcmp(&S.m_Material, &Mat, sizeof(Mat)) != 0;
                const bool bEntEdit = std::memcmp(&S.m_Entity, &Ent, sizeof(Ent)) != 0;
                // last frame's transform (the screen tail's motion): the one it had before it moved this frame; once it stopped, written
                // again as this frame's (else it would keep moving in the shaders)
                if (bMoved && S.m_MovedFrame != Frame) { S.m_PrevL2W = S.m_L2W; S.m_MovedFrame = Frame; }
                else if (!bMoved && S.m_MovedFrame != Frame && std::memcmp(&S.m_PrevL2W, &S.m_L2W, sizeof(L2W)) != 0) { S.m_PrevL2W = S.m_L2W; MarkPending(It->second); }
                if (bMoved || bMatEdit || bEntEdit) { S.m_L2W = L2W; S.m_Material = Mat; S.m_Entity = Ent; MarkPending(It->second); }
                if (pChanged) *pChanged = bMoved || bMatEdit;
                return It->second;
            }

            if (m_Free.empty() && !Evict(Frame)) return none_v;
            const std::uint32_t Index = m_Free.back();
            m_Free.pop_back();
            m_Slots[Index] = { L2W, Mat, Ent, Frame, ID, false, true, L2W, Frame };
            m_ByID.emplace(ID, Index);
            MarkPending(Index);
            return Index;
        }

        // Every live slot pending (the world origin moved: the stored rows are relative to it)
        void MarkAllPending() noexcept { for (std::uint32_t i = 0; i < m_Slots.size(); ++i) if (m_Slots[i].m_bLive) MarkPending(i); }

        // Writes up to Max pending slots: Write(slot index, const fmat4& L2W, const fmat4& PrevL2W (last frame's), const material&,
        // const entity_params&). The rest stay pending. Returns how many.
        template<typename T_WRITE>
        int Flush(int Max, T_WRITE&& Write) noexcept
        {
            int n = 0;
            std::size_t i = 0;
            for (; i < m_Pending.size() && n < Max; ++i)
            {
                auto& S = m_Slots[m_Pending[i]];
                S.m_bPending = false;
                if (S.m_bLive) { Write(m_Pending[i], S.m_L2W, S.m_PrevL2W, S.m_Material, S.m_Entity); ++n; }
            }
            m_Pending.erase(m_Pending.begin(), m_Pending.begin() + i);
            return n;
        }

        int getLiveCount()    const noexcept { return int(m_ByID.size()); }
        int getPendingCount() const noexcept { return int(m_Pending.size()); }

    private:
        struct slot
        {
            xmath::fmat4    m_L2W;
            material        m_Material;
            entity_params   m_Entity;
            std::uint64_t   m_LastSeen  = 0;
            std::uint32_t   m_ID        = 0;
            bool            m_bPending  = false;
            bool            m_bLive     = false;
            xmath::fmat4    m_PrevL2W;                      // last frame's (Acquire)
            std::uint64_t   m_MovedFrame = 0;               // the last frame it moved in
        };

        void MarkPending(std::uint32_t i) noexcept
        {
            if (m_Slots[i].m_bPending) return;
            m_Slots[i].m_bPending = true;
            m_Pending.push_back(i);
        }

        // Frees the least recently seen slot not seen this frame. O(slots), only when full.
        bool Evict(std::uint64_t Frame) noexcept
        {
            std::uint32_t Best = none_v;
            for (std::uint32_t i = 0; i < m_Slots.size(); ++i)
                if (m_Slots[i].m_bLive && m_Slots[i].m_LastSeen < Frame && (Best == none_v || m_Slots[i].m_LastSeen < m_Slots[Best].m_LastSeen)) Best = i;
            if (Best == none_v) return false;
            m_ByID.erase(m_Slots[Best].m_ID);
            m_Slots[Best].m_bLive = false;
            m_Free.push_back(Best);
            return true;
        }

        std::vector<slot>                                m_Slots;
        std::vector<std::uint32_t>                       m_Free;
        std::vector<std::uint32_t>                       m_Pending;
        std::unordered_map<std::uint32_t, std::uint32_t> m_ByID;
    };

    // Instance ids for a caller whose draws are keyed by a 64-bit handle (an ECS entity): one id per handle, never shared (a 32-bit
    // hash of the handle gave two entities one slot), given back when the handle was not drawn for a whole frame. Never 0.
    struct instance_ids
    {
        // A new frame: the handles not drawn last frame give their ids back
        void Begin() noexcept
        {
            ++m_Frame;
            std::erase_if(m_ByHandle, [&](const auto& E) { if (E.second.m_Frame + 1 >= m_Frame) return false; m_Free.push_back(E.second.m_ID); return true; });
        }

        std::uint32_t Of(std::uint64_t Handle) noexcept
        {
            auto [It, bNew] = m_ByHandle.try_emplace(Handle);
            if (bNew)
            {
                if (m_Free.empty()) It->second.m_ID = m_Next++;
                else { It->second.m_ID = m_Free.back(); m_Free.pop_back(); }
            }
            It->second.m_Frame = m_Frame;
            return It->second.m_ID;
        }

    private:
        struct entry { std::uint32_t m_ID = 0; std::uint64_t m_Frame = 0; };
        std::unordered_map<std::uint64_t, entry>    m_ByHandle;
        std::vector<std::uint32_t>                  m_Free;
        std::uint32_t                               m_Next  = 1;
        std::uint64_t                               m_Frame = 0;
    };
}

#endif
