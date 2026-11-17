#ifndef XLIONRENDER_ANIM_POSE_H
#define XLIONRENDER_ANIM_POSE_H
#pragma once
// K0c: the animation pose stack, out of the scene renderer: the library's pose of a skeleton (a clip's sample, the transition when its source
// changes, stride/orientation warping, the ground fit, foot locking, look-at) and the per-entity state it keeps. The Animation system owns
// one per world and poses every animated character once per update (its game time); the scene renderer is one too, for what the system
// did not pose (no update ran, a preview). Moved verbatim from xlionrender_scene_renderer.h; only the clock is new (m_GameDt).
#include "dependencies/xAnimation/source/xanimation.h"
#include "dependencies/xAnimation/source/adapters/xanimation_resources.h"
#include "xlionrender_scene_components.h"      // (the Animation component the pose reads)
#include <array>
#include <chrono>
#include <span>
#include <unordered_map>
#include <vector>

namespace xlionrender
{
    struct anim_poser
    {
        xanimation::pose                                m_AnimPose;                     // (scratch: the library's local pose of the character being added)
        xanimation::pose                                m_AnimScratch;                  // (scratch: the unmirrored sample of a mirrored clip)
        const xanimation::pose*                         m_pAnimUsed = nullptr;          // (the pose PoseWithLibrary posed: m_AnimPose or the rest)
        std::unordered_map<const void*, xanimation::skeleton_data> m_AnimSkeleton;      // per skeleton resource: the library's copy of its hierarchy and rest pose
        float                                           m_GameDt = -1;                  // >= 0: the per-entity state advances by this (the update's game time), not the wall clock

        // What an update posed: per entity, its bone worlds (model space) and local pose, stamped with the update that made them
        // ponytail: a map keyed by entity, swept of entities not posed for 256 updates; a slot in the component if levels churn entities
        struct posed { std::uint32_t m_Update = 0; std::vector<xmath::fmat4> m_Worlds; xanimation::pose m_Local; const xskeleton::skeleton* m_pSkel = nullptr; };
        std::unordered_map<std::uint64_t, posed>        m_Posed;
        std::uint32_t                                   m_Update = 0;
        const posed* Posed(std::uint64_t Entity) const noexcept
        {
            const auto It = m_Posed.find(Entity);
            return It != m_Posed.end() && It->second.m_Update == m_Update ? &It->second : nullptr;
        }
        void SweepPosed(void) noexcept
        {
            if ((m_Update & 255) != 0) return;
            for (auto It = m_Posed.begin(); It != m_Posed.end(); ) if (m_Update - It->second.m_Update > 256) It = m_Posed.erase(It); else ++It;
        }

        // A skinned character's pose from its Animation component as the renderer always made it: the clip at its time, the transition, the
        // Animation system's warp and ground casts, foot locking and look-at in the frame L2W. Entity 0: no per-entity state (no transitions).
        void PoseEntity(const xskeleton::skeleton& Skel, const xanim_package::anim_package* pPkg, bool bClip, const animation* pAnim, const xmath::fmat4& L2W
                       , std::uint64_t Entity, std::vector<xmath::fmat4>& Out) noexcept
        {
            xanimation::warp_params Wp;          // K2: the Animation system's warping this frame
            if (bClip) { Wp.m_Stride = pAnim->m_WarpStride; Wp.m_Angle = pAnim->m_WarpAngle; Wp.m_Dir = pAnim->m_WarpDir; }
            std::array<xanimation::ground_contact, 4> Ground; std::size_t nGround = 0;        // K3: the Animation system's casts this frame
            if (bClip && pAnim->m_bFootIK) for (; nGround < pAnim->m_nGround && nGround < Ground.size(); ++nGround)
                Ground[nGround] = { pAnim->m_Ground[nGround].m_bHit, pAnim->m_Ground[nGround].m_Height, pAnim->m_Ground[nGround].m_Normal };
            PoseWithLibrary(Skel, bClip ? pPkg : nullptr, bClip ? pAnim->m_Clip : 0, bClip ? pAnim->m_Time : 0.0f, bClip && pAnim->m_bMirror, Out
                           , pAnim ? Entity : 0, pAnim ? pAnim->m_BlendTime : 0.0f, &Wp, std::span<const xanimation::ground_contact>(Ground.data(), nGround)
                           , pAnim ? pAnim->m_MatchJumps : 0u, bClip && pAnim->m_bFootLock ? &L2W : nullptr
                           , &L2W, pAnim ? pAnim->m_LookAt : xmath::fvec3(0, 0, 0), pAnim ? pAnim->m_LookAtWeight : 0.0f);
        }
        float Elapsed(std::chrono::steady_clock::time_point Last, std::chrono::steady_clock::time_point Now) const noexcept
            { return m_GameDt >= 0 ? std::min(m_GameDt, 0.1f) : std::clamp(std::chrono::duration<float>(Now - Last).count(), 0.0f, 0.1f); }

        // The library's pose of a skeleton at a clip's time (no clip: its rest pose) as bone worlds. The skeleton's parent array is copied once
        // per skeleton resource (re-checked against the resource each time: a reload with other parents rebuilds it).
        // Entity / BlendTime (K1): when what poses the entity changes (another clip or package, mirrored or not, the rest), the new pose starts
        // from the one shown and eases into the new source over BlendTime (inertialization: the old clip is not sampled any more; 0: a cut).
        void PoseWithLibrary(const xskeleton::skeleton& Skel, const xanim_package::anim_package* pPkg, int iClip, float Time, bool bMirror, std::vector<xmath::fmat4>& Out, std::uint64_t Entity = 0, float BlendTime = 0, const xanimation::warp_params* pWarp = nullptr
                            , std::span<const xanimation::ground_contact> Ground = {}, std::uint32_t Jumps = 0      // (Jumps: motion matching's jump count - a jump within a clip is a new source too)
                            , const xmath::fmat4* pLockL2W = nullptr                                                  // (set: foot locking, in this world frame)
                            , const xmath::fmat4* pL2W = nullptr, const xmath::fvec3& LookAt = {}, float LookAtWeight = 0)    // (look-at: a world point, in the frame pL2W)
        {
            auto& S = m_AnimSkeleton[&Skel];
            const auto Bones = Skel.getBones();
            bool bSame = S.m_Parent.size() == Bones.size();
            for (std::size_t i = 0; bSame && i < Bones.size(); ++i) bSame = S.m_Parent[i] == Bones[i].m_iParent;
            if (!bSame) S = xanimation::MakeSkeleton(Skel);
            Out.resize(Bones.size());
            const bool bClip = pPkg && xanimation::SampleClip(S, *pPkg, iClip, Time, bMirror, m_AnimPose, m_AnimScratch);
            if (Entity == 0) { xanimation::ComputeBoneWorlds(S.view(), bClip ? m_AnimPose : S.m_Rest, Out); m_pAnimUsed = bClip ? &m_AnimPose : &S.m_Rest; return; }
            if (!bClip) m_AnimPose = S.m_Rest;
            Transition(Entity, bClip ? (std::uint64_t(reinterpret_cast<std::uintptr_t>(pPkg)) * 1000003u + std::uint64_t(iClip) * 2 + (bMirror ? 1 : 0) + std::uint64_t(Jumps) * 0x9E3779B97F4A7C15ull) : 1, &Skel, BlendTime, m_AnimPose);
            if (bClip && pWarp && pWarp->active()) xanimation::WarpPose(S.view(), S.m_Warp, m_AnimPose, *pWarp);     // (after the blend: the blend's poses stay the clip's own)
            if (bClip && Ground.size() == S.m_Warp.m_Legs.size() && !Ground.empty()) xanimation::FitGround(S.view(), S.m_Warp, m_AnimPose, Ground, S.m_FootHeights);
            if (bClip && pLockL2W)                     // foot locking: last, on the pose as drawn (per entity: its pins and when it was last drawn)
            {
                auto& Tr = m_Tracks[Entity];
                const auto Now = std::chrono::steady_clock::now();
                const float Dt = Tr.m_LockT.time_since_epoch().count() ? Elapsed(Tr.m_LockT, Now) : 0.0f;
                Tr.m_LockT = Now;
                // (through a transition's blend a planted foot is held farther: the blend moves the pose's foot between two clips while the body
                // goes on, and at the usual 25 cm the lock let go mid-blend - SoccerAnim's feet slid ~125 cm/s around matching jumps)
                xanimation::foot_lock_params LockPm; if (Tr.m_In.m_bActive) LockPm.m_MaxPull = 0.6f;
                xanimation::LockFeet(S.view(), S.m_Warp, m_AnimPose, pLockL2W->ExtractPosition(), pLockL2W->ExtractRotation().NormalizeSafeCopy(), pLockL2W->ExtractScale(), S.m_FootHeights, Dt, Tr.m_Lock, LockPm);
            }
            if (bClip && pL2W && S.m_LookAt.valid())  // look-at: after the feet (the chain is above the hips), toward a world point in the entity's frame
            {
                auto& Tr = m_Tracks[Entity];
                if (LookAtWeight > 0 || Tr.m_Look.m_W > 0)
                {
                    const auto Now = std::chrono::steady_clock::now();
                    const float Dt = Tr.m_LookT.time_since_epoch().count() ? Elapsed(Tr.m_LookT, Now) : 0.0f;
                    Tr.m_LookT = Now;
                    const auto Inv = pL2W->InverseSRT();
                    xanimation::LookAt(S.view(), S.m_LookAt, m_AnimPose, Inv * LookAt, LookAtWeight, Dt, Tr.m_Look);
                }
            }
            xanimation::ComputeBoneWorlds(S.view(), m_AnimPose, Out); m_pAnimUsed = &m_AnimPose;
        }

        // the poses an entity showed on its last two frames, what posed it, and its transition (K1)
        struct anim_track
        {
            std::uint64_t                           m_Source = 0;
            const void*                             m_pSkel  = nullptr;
            xanimation::pose                        m_Cur, m_Prev;
            std::chrono::steady_clock::time_point   m_T;
            xanimation::inertializer                m_In;
            xanimation::foot_lock_state             m_Lock;         // (FootLock: the feet's pins)
            std::chrono::steady_clock::time_point   m_LockT{};
            xanimation::look_at_state               m_Look;         // (LookAt: the aim's eased angles and weight)
            std::chrono::steady_clock::time_point   m_LookT{};
        };
        std::unordered_map<std::uint64_t, anim_track>   m_Tracks;

        void Transition(std::uint64_t Entity, std::uint64_t Source, const void* pSkel, float BlendTime, xanimation::pose& P)
        {
            if (m_Tracks.size() > 4096) m_Tracks.clear();       // ponytail: entities that went away are forgotten in bulk; a per-frame sweep if levels spawn that many
            auto& Tr = m_Tracks[Entity];
            const auto Now = std::chrono::steady_clock::now();
            const bool bHad = Tr.m_pSkel == pSkel && Tr.m_Cur.count() == P.count();
            const float Dt = bHad ? Elapsed(Tr.m_T, Now) : 0.0f;
            if (bHad && Tr.m_Source != Source && BlendTime > 0.0f)
            {
                Tr.m_In.Start(Tr.m_Cur, Tr.m_Prev, P, P, Dt, BlendTime * 0.25f);     // (a half-life of a quarter of the blend time: ~94 % gone by its end)
                static const bool bLog = [] { const char* e = std::getenv("XLION_ANIM_EVENTS"); return e && *e == '1'; }();
                if (bLog) { float W = 0; for (const auto& R : Tr.m_In.m_RotX) W = std::max(W, std::sqrt(R.Dot(R))); std::printf("[AnimBlend] entity %llu: a new source eases in over %.2f s (largest turn to absorb %.1f deg)\n", (unsigned long long)Entity, BlendTime, W * 57.29578f); }
            }
            if (bHad) Tr.m_In.Apply(P, Dt); else Tr.m_In.m_bActive = false;
            if (bHad) Tr.m_Prev = Tr.m_Cur; else Tr.m_Prev = P;
            Tr.m_Cur = P; Tr.m_Source = Source; Tr.m_pSkel = pSkel; Tr.m_T = Now;
        }
    };
}
#endif
