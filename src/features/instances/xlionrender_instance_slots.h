#ifndef XLIONRENDER_INSTANCE_SLOTS_H
#define XLIONRENDER_INSTANCE_SLOTS_H
#pragma once

// Persistent instance data (research doc: per-draw data). A draw that carries an instance id (scene::draw::m_Instance, low 32
// bits; the caller owns the ids - an ECS entity, an editor's preview, ...) keeps a slot in the instance buffer across frames:
// its data is written when it changes, not when it becomes visible. In tight spaces visibility flips every few steps; the
// slots do not care.
//  - The GPU buffer has one copy per frame in flight (copies_v): a slot changed now is written into each copy when that
//    copy's frame comes around (per slot pending bits), so a frame still on the GPU never sees a write.
//  - A slot is not freed when its id is missing from a frame (invisible is not dead): slots are evicted least recently
//    seen first, only when the buffer is full.
// ponytail: a slot changed between two renders of the same frame shows its new data in both (an editor drawing the same id
// twice per frame with different transforms); per-render copies if that matters.

#include "dependencies/xmath/source/xmath.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace xlionrender
{
    struct instance_slots
    {
        static constexpr int copies_v = 3;                  // frames in flight (xGPU keeps resources 2 flips) + the one being recorded
        static constexpr std::uint32_t none_v = ~0u;

        void Init(int Capacity) noexcept
        {
            m_Slots.assign(Capacity, {});
            m_Free.clear();
            for (int i = Capacity - 1; i >= 0; --i) m_Free.push_back(std::uint32_t(i));
            m_ByID.clear();
            m_Pending.clear();
        }

        // The slot of ID this frame (none_v: no room - every slot was seen this frame). New or changed: pending in every copy.
        std::uint32_t Acquire(std::uint32_t ID, const xmath::fmat4& L2W, std::uint64_t Frame) noexcept
        {
            if (auto It = m_ByID.find(ID); It != m_ByID.end())
            {
                auto& S = m_Slots[It->second];
                S.m_LastSeen = Frame;
                if (std::memcmp(&S.m_L2W, &L2W, sizeof(L2W)) != 0) { S.m_L2W = L2W; MarkPending(It->second); }
                return It->second;
            }

            if (m_Free.empty() && !Evict(Frame)) return none_v;
            const std::uint32_t Index = m_Free.back();
            m_Free.pop_back();
            m_Slots[Index] = { L2W, Frame, ID, 0, true };
            m_ByID.emplace(ID, Index);
            MarkPending(Index);
            return Index;
        }

        // Every live slot pending (the world origin moved: the stored rows are relative to it)
        void MarkAllPending() noexcept { for (std::uint32_t i = 0; i < m_Slots.size(); ++i) if (m_Slots[i].m_bLive) MarkPending(i); }

        // Writes the slots pending for Copy: Write(slot index, const fmat4& L2W)
        template<typename T_WRITE>
        void Flush(int Copy, T_WRITE&& Write) noexcept
        {
            const std::uint8_t Bit = std::uint8_t(1u << Copy);
            for (auto i : m_Pending)
            {
                auto& S = m_Slots[i];
                if ((S.m_Pending & Bit) == 0) continue;
                S.m_Pending &= ~Bit;
                if (S.m_bLive) Write(i, S.m_L2W);
            }
            std::erase_if(m_Pending, [&](std::uint32_t i) { return m_Slots[i].m_Pending == 0; });
        }

        int getLiveCount() const noexcept { return int(m_ByID.size()); }

    private:
        struct slot
        {
            xmath::fmat4    m_L2W;
            std::uint64_t   m_LastSeen  = 0;
            std::uint32_t   m_ID        = 0;
            std::uint8_t    m_Pending   = 0;        // a bit per copy still to write
            bool            m_bLive     = false;
        };

        void MarkPending(std::uint32_t i) noexcept
        {
            if (m_Slots[i].m_Pending == 0) m_Pending.push_back(i);
            m_Slots[i].m_Pending = std::uint8_t((1u << copies_v) - 1);
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

        std::vector<slot>                               m_Slots;
        std::vector<std::uint32_t>                      m_Free;
        std::vector<std::uint32_t>                      m_Pending;
        std::unordered_map<std::uint32_t, std::uint32_t> m_ByID;
    };
}

#endif
