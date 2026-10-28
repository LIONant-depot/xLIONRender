#ifndef XLIONRENDER_SHADOW_VSM_H
#define XLIONRENDER_SHADOW_VSM_H
#pragma once

// The directional light's virtual shadow pages (documentation/lighting_strategy.md sec. 7, P1a): the clipmap's levels made virtual.
//
//  - Levels: level L's texel is Level0Texel x 2^L; a page is page_texels_v texels square, world-anchored in light space (its integer
//    x, y index), so a page stays valid while the camera moves. Each level keeps a window of setup::m_Window pages around the camera;
//    the page table is that window, toroidal (the page's index mod the window), as the clipmap's slots are.
//  - Demand: a pass over the main view's depth (shaders/xlionrender_vsm_request_comp.glsl) marks, per pixel, the page of the level
//    whose texel fits the pixel's footprint, dilated by the filter's reach; read back a ring late, the requests are decoded with the
//    windows of the frame that made them (Snapshot).
//  - The pool: the clipmap's part of the shadow atlas cut into page slots. The first setup::PinnedSlots() hold the coarse net (the
//    coarsest level's pages within m_PinnedRadius of the camera): never evicted, rendered first, and rendered whatever the budget (the
//    overrun is counted) - so a fine page that is not there degrades to the net, never to a hole. The rest hold requested fine pages:
//    a page requested by the latest requests or rendered this frame is protected; the others are evicted least recently requested
//    first; a page with no slot left is rejected (counted apart from a miss).
//  - States: a resident page is VALID or DIRTY. A caster change makes the pages under it DIRTY at once (they leave the table: the
//    shader falls back to a coarser level), the budget decides when they are rendered again. A new light direction or level-0 size
//    opens a new epoch; a level's depth range moving makes that level's pages go.
//  - The moving sun (P1e): epochs are quantized (setup::m_EpochDegrees). At a new epoch the VALID fine pages become the previous
//    epoch's (their own basis, windows and table: APPROXIMATE for the current direction, as an epoch is), the net is rendered again
//    at once, and the requests of older epochs are not dropped: each carries its page's depth range, so it is moved into the new
//    basis (Remap). The shading reads the current epoch where it has the page, the previous one where not, then coarser levels; once
//    the current epoch has every page asked for, it cross-fades from the previous one over setup::m_FadeFrames, then the previous
//    pages go. They are the first evicted when a slot is needed.
//  - Movers (P1b): a page holds the static casters only; a mover (a caster that changed lately, or skinned) never dirties it. Each
//    frame every page asked for (and of the net) that a mover overlaps gets a mover page - its own slot, its own table - with only the
//    movers drawn; the shading takes the nearer of the two depths per filter tap: the depth both drawn into one page would have, without
//    re-rendering the static casters under a moving character every frame. (The strategy's restore-then-draw needs a copy within the
//    atlas; this is its R-C alternative, exact per tap.)
//  - The table holds a page only when it is VALID this frame (its slot + 1, 16 bits; 0: none), so the shader never reads a slot that
//    holds another page or stale depth (invariant 1); Validate checks it. A slot reused this frame is rendered before the scene pass
//    that samples it reads the table naming it; earlier frames still on the GPU read it before (one queue, the passes in order).
// ponytail: CPU admission from a ring-late readback (GPU-driven page lists: P1c); one unordered_map lookup per requested page

#include "features/shadows/xlionrender_shadow_clipmap.h"
#include <algorithm>
#include <bit>
#include <span>
#include <unordered_map>

namespace xlionrender::shadows
{
    struct vsm
    {
        static constexpr int page_texels_v  = 128;
        static constexpr int max_levels_v   = 12;
        static constexpr float max_level0_texel_v = 0.002f;    // level 0 is never coarser than 2 mm (Unreal's finest directional level is under 0.1 mm)
        // A level's depth range: +-depth_range_v x its window's extent around the camera, re-centered (snapped to depth_recenter_v of it)
        // once the camera is depth_recenter_v of it away. A pixel shaded by a level is up to 4 x H / (2 tan(fov/2)) of its texels from the
        // camera (resolution bias -1; 10430 at 2160 lines and 45 degrees) - along the light too, so the range less the drift must reach that:
        // with the window's extent (4096) and half of it as the drift, a low sun left the ground and the casters near it out (Tomas: the
        // Velociraptor's leg shadow cut off). 4 and a quarter: 12288 texels; 16 bits still give half a texel a step
        static constexpr float depth_range_v      = 4.0f;
        static constexpr float depth_recenter_v   = 0.25f;
        static constexpr int filter_reach_v = 2;        // texels a lookup's filter reaches past its centre texel (the request pass dilates by it)

        struct setup
        {
            int     m_PoolTexels    = 4096;             // the pool's width in the atlas (the renderer's: 8192 x 4096 x 16-bit = 64 MiB, 2048 slots -
            int     m_PoolHeight    = 0;                // Unreal's r.Shadow.Virtual.MaxPhysicalPages, at half its 32-bit memory); its height, 0: square
            int     m_Levels        = 10;               // level L's texel: level 0's x 2^L; the coarsest is the net
            int     m_Window        = 32;               // pages per level side kept in the table around the camera
            int     m_PinnedRadius  = 8;                // the net: the coarsest level's pages within this many of the camera's (a 16 x 16 square)
            int     m_Budget        = 48;               // fine pages rendered a frame (the net's are not counted against it) - with m_NetBudget the shadows' 2 ms slice (methodology 3: ~0.03 ms a page in Bistro)
            int     m_ProtectFrames = 4;                // a page requested this many frames ago is still protected (the requests are a ring late)
            float   m_ResolutionBias = -1;              // levels added to the footprint's (-1: texels half the pixel; Unreal's ResolutionLodBiasDirectional);
                                                        // -1 measured against the clipmap and the rays (tests/xlionrender_test_vsm.cpp)
            float   m_EpochDegrees  = 0.25f;            // the light turning less than this keeps its epoch: the pages are of its direction then, the
                                                        // shading's is the current one (an approximation; the mismatch in stats::m_EpochMismatchDegrees)
            // Dynamic resolution (Unreal's r.Shadow.Virtual.DynamicRes): while the pages asked for pass m_PoolLoadFactor of the fine slots,
            // the whole view's resolution bias rises (toward the bias that would fit them: each +1 a quarter of the pages), and comes back
            // slowly once they fit again - every pixel a little coarser instead of some patches a level coarser (rejected pages)
            float   m_PoolLoadFactor = 0.85f;
            float   m_MaxDynamicBias = 2.0f;            // 0: off
            int     m_FadeFrames     = 8;               // the previous epoch to the current one, once the current has every page asked for
            int     m_MaxPrevFrames  = 120;             // the previous epoch goes after this many frames whatever (a rebuild that never ends)
            // The net's pages rendered a frame (nearest the camera first) - the frame budget (methodology 3: the whole frame 16 ms; a new
            // epoch rendered all 256 at once, 148k draws in Bistro). The previous epoch's net shows meanwhile (its own slots: NetSlots),
            // and a new epoch waits until the previous one is gone (never a hole). 0: every page of the net at once, its old net freed
            int     m_NetBudget      = 16;

            int SlotsPerRow ( void ) const noexcept { return m_PoolTexels / page_texels_v; }
            int Slots       ( void ) const noexcept { return SlotsPerRow() * ((m_PoolHeight > 0 ? m_PoolHeight : m_PoolTexels) / page_texels_v); }
            int PinnedSlots ( void ) const noexcept { return (2 * m_PinnedRadius) * (2 * m_PinnedRadius); }
            int NetSlots    ( void ) const noexcept { return PinnedSlots() * (m_NetBudget > 0 ? 2 : 1); }      // (the net's, the previous epoch's net's)
            int TableEntries( void ) const noexcept { return m_Levels * m_Window * m_Window; }
            int TableUints  ( void ) const noexcept { return (TableEntries() + 1) / 2; }
        };

        struct key
        {
            int m_Level = 0, m_X = 0, m_Y = 0;
            bool operator == (const key&) const noexcept = default;
        };

        struct level
        {
            int     m_OriginX = 0, m_OriginY = 0;       // the window's first page index
            float   m_PageSize = 1;                     // world units per page side
            float   m_DepthCenter = 0;                  // light-space z the level's depth range is centered on (snapped)
        };

        struct page_to_render
        {
            key             m_Key;
            int             m_Slot;
            int             m_AtlasX, m_AtlasY;         // its slot's texel origin in the atlas
            xmath::fmat4    m_W2C;                      // world -> the page's clip space
            float           m_MinX, m_MinY, m_Size;     // its light-space rect (culling the casters)
            bool            m_bPinned;
            bool            m_bMover = false;           // a mover page: the movers only (else the static casters only)
        };

        struct stats
        {
            int m_Requested = 0;            // distinct pages the latest requests asked for
            int m_Reused    = 0;            //   of them VALID already
            int m_Admitted  = 0;            //   given a slot this frame
            int m_Rejected  = 0;            //   no slot left (pressure)
            int m_Deferred  = 0;            //   waiting for the budget (absent or dirty)
            int m_NetDeferred = 0;          //   of them the net's (setup::m_NetBudget)
            int m_Rendered  = 0;            // fine pages rendered
            int m_RenderedMover = 0;        // mover pages rendered (each frame, the movers only)
            int m_RenderedPinned = 0;       // net pages rendered (mandatory: beyond the budget)
            int m_Dirtied   = 0;            // pages a caster change invalidated
            int m_Evicted   = 0;
            int m_ResidentFine = 0, m_ResidentPinned = 0, m_ResidentMover = 0;
            int m_PinnedNotValid = 0;       // net pages not VALID when the table is published (a readiness failure: 0 by construction)
            int m_Epochs    = 0;            // 1: this frame opened a new epoch (a new direction or scale)
            float m_EpochMismatchDegrees = 0;   // the angle between the light's direction and its epoch's (under setup::m_EpochDegrees)
            float m_PoolLoad     = 0;       // the fine pages asked for / the fine slots (above 1: more than fit)
            float m_DynamicBias  = 0;       // the resolution bias dynamic resolution adds now (ResolutionBias())
            int   m_Remapped     = 0;       // requests of an older epoch moved into this one (pages, after the move)
            int   m_ResidentPrev = 0;       // the previous epoch's pages still resident
            float m_Fade         = 1;       // the current epoch's weight against the previous one (1: the current only)
        };

        // A request's depth range (the request pass: per table entry, the largest of z and of 1 - z, z in 0..1 over its level's range)
        struct request_depth { std::uint32_t m_MaxZ = 0, m_MaxInvZ = 0; };

        // Windows of one frame, to decode its requests later (with its basis and depth ranges: a later epoch moves them into its own)
        struct snapshot
        {
            std::array<int, max_levels_v * 2> m_Origin{};
            std::array<float, max_levels_v>   m_DepthCenter{};
            xmath::fvec3  m_X{ 1, 0, 0 }, m_Y{ 0, 1, 0 }, m_Z{ 0, 0, 1 };
            float         m_Level0Texel = 0;
            std::uint64_t m_Epoch = 0;
            bool          m_bValid = false;
        };

        void Init(const setup& Setup) noexcept
        {
            m_Setup = Setup;
            m_Setup.m_Levels = std::clamp(m_Setup.m_Levels, 1, max_levels_v);
            m_Slots.assign(std::size_t(m_Setup.Slots()), slot{});
            m_RenderedFrame.assign(m_Slots.size(), ~0ull);
            m_Map.clear();
            m_PrevMap.clear();
            m_DynMap.clear();
            m_bHasBasis = m_bHasPrev = false;
            m_DynamicBias = 0; m_LastOverload = 0;
        }

        const setup& getSetup( void ) const noexcept { return m_Setup; }

        static std::uint64_t Pack(const key& K) noexcept
        {
            return (std::uint64_t(std::uint32_t(K.m_Level)) << 56) | ((std::uint64_t(std::uint32_t(K.m_X)) & 0xFFFFFFF) << 28) | (std::uint64_t(std::uint32_t(K.m_Y)) & 0xFFFFFFF);
        }

        // Toroidal: a page's entry in its level's window, and back (Origin: the window's first page index on that axis)
        int  Torus  (int Index) const noexcept { return PositiveMod(Index, m_Setup.m_Window); }
        int  Unwrap (int T, int Origin) const noexcept { return Origin + PositiveMod(T - Origin, m_Setup.m_Window); }
        bool InWindow(const key& K) const noexcept
        {
            const auto& Lv = m_Levels[K.m_Level];
            return K.m_X >= Lv.m_OriginX && K.m_X < Lv.m_OriginX + m_Setup.m_Window && K.m_Y >= Lv.m_OriginY && K.m_Y < Lv.m_OriginY + m_Setup.m_Window;
        }
        int TableIndex(const key& K) const noexcept { return (K.m_Level * m_Setup.m_Window + Torus(K.m_Y)) * m_Setup.m_Window + Torus(K.m_X); }

        bool IsPinned(const key& K) const noexcept
        {
            if (K.m_Level != m_Setup.m_Levels - 1) return false;
            return K.m_X >= m_PinX - m_Setup.m_PinnedRadius && K.m_X < m_PinX + m_Setup.m_PinnedRadius
                && K.m_Y >= m_PinY - m_Setup.m_PinnedRadius && K.m_Y < m_PinY + m_Setup.m_PinnedRadius;
        }

        snapshot Snapshot(void) const noexcept
        {
            snapshot S;
            for (int L = 0; L < m_Setup.m_Levels; ++L) { S.m_Origin[2 * L] = m_Levels[L].m_OriginX; S.m_Origin[2 * L + 1] = m_Levels[L].m_OriginY; S.m_DepthCenter[L] = m_Levels[L].m_DepthCenter; }
            S.m_X = m_X; S.m_Y = m_Y; S.m_Z = m_Z; S.m_Level0Texel = m_Level0Texel;
            S.m_Epoch = m_Epoch;
            S.m_bValid = m_bHasBasis;
            return S;
        }

        // The request pass's bits (a bit per table entry, level-major) as pages, with the windows of the frame that made them. Requests of
        // an older epoch are moved into this one (Depths: the request pass's per entry; empty: they are dropped) - call it after Update
        // opened this frame's epoch
        void DecodeRequests(std::span<const std::uint32_t> Bits, const snapshot& S, std::vector<key>& Out, std::span<const request_depth> Depths = {}) noexcept
        {
            Out.clear();
            if (!S.m_bValid || !m_bHasBasis) return;
            const bool bOld = S.m_Epoch != m_Epoch;
            if (bOld && Depths.empty()) return;
            const int W = m_Setup.m_Window, n = std::min(int(Bits.size()) * 32, m_Setup.TableEntries());
            for (int i = 0; i < n; ++i)
            {
                if ((Bits[std::size_t(i) >> 5] & (1u << (i & 31))) == 0) continue;
                const int L = i / (W * W), ty = (i / W) % W, tx = i % W;
                const key K{ L, Unwrap(tx, S.m_Origin[2 * L]), Unwrap(ty, S.m_Origin[2 * L + 1]) };
                if (!bOld) { Out.push_back(K); continue; }
                if (std::size_t(i) < Depths.size()) Remap(K, S, Depths[std::size_t(i)], Out);
            }
            if (bOld)
            {
                std::sort(Out.begin(), Out.end(), [](const key& a, const key& b) { return Pack(a) < Pack(b); });
                Out.erase(std::unique(Out.begin(), Out.end()), Out.end());
                m_FrameRemapped = int(Out.size());
            }
        }

        // A page asked for in an older epoch (S), into this one: its light-space box (its rect, the depth range its pixels had) through
        // the world into this basis, the pages of the same level it covers. The previous epoch's texel size if the scale changed
        void Remap(const key& K, const snapshot& S, const request_depth& D, std::vector<key>& Out) const noexcept
        {
            const float P0 = S.m_Level0Texel * float(page_texels_v) * float(1 << K.m_Level);
            const float H0 = P0 * float(m_Setup.m_Window) * depth_range_v;
            const float Z0 = std::bit_cast<float>(D.m_MaxInvZ), Z1 = std::bit_cast<float>(D.m_MaxZ);       // (1 - min z, max z)
            const float ZLo = S.m_DepthCenter[K.m_Level] - H0 + (D.m_MaxInvZ ? 1.0f - Z0 : 0.5f) * 2.0f * H0;
            const float ZHi = S.m_DepthCenter[K.m_Level] - H0 + (D.m_MaxZ ? Z1 : 0.5f) * 2.0f * H0;
            float MinX = 1e30f, MinY = 1e30f, MaxX = -1e30f, MaxY = -1e30f;
            for (int c = 0; c < 8; ++c)
            {
                const float x = float(K.m_X + (c & 1)) * P0, y = float(K.m_Y + ((c >> 1) & 1)) * P0, z = (c & 4) ? ZHi : ZLo;
                const xmath::fvec3 World = S.m_X * x + S.m_Y * y + S.m_Z * z;
                const float nx = World.Dot(m_X), ny = World.Dot(m_Y);
                MinX = std::min(MinX, nx); MaxX = std::max(MaxX, nx); MinY = std::min(MinY, ny); MaxY = std::max(MaxY, ny);
            }
            const float P = PageSize(K.m_Level);
            const int x0 = int(std::floor(MinX / P)), x1 = int(std::floor(MaxX / P)), y0 = int(std::floor(MinY / P)), y1 = int(std::floor(MaxY / P));
            if ((x1 - x0 + 1) * (y1 - y0 + 1) > 16) return;      // (a degenerate move: the next requests will say)
            for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x) Out.push_back({ K.m_Level, x, y });
        }

        // The basis: a new direction (past setup::m_EpochDegrees) or scale opens a new epoch (KeepAsPrevious). Update calls it; call it
        // before DecodeRequests so a request of an older epoch is moved into this frame's (a second call the same frame: nothing)
        bool OpenEpoch(const xmath::fvec3& LightDir, float Level0Texel, std::uint64_t Frame) noexcept
        {
            const xmath::fvec3 Z = LightDir.NormalizeCopy();
            const float EpochCos = std::cos(m_Setup.m_EpochDegrees * 3.14159265f / 180.0f);
            if (m_bHasBasis && Z.Dot(m_Z) >= EpochCos && Level0Texel == m_Level0Texel) return false;
            if (m_Setup.m_NetBudget > 0 && m_bHasPrev && Frame - m_EpochFrame <= std::uint64_t(m_Setup.m_MaxPrevFrames)) return false;     // (the last one not complete yet)
            m_PX = m_X; m_PY = m_Y; m_PZ = m_Z; m_PrevLevels = m_Levels; m_PrevLevel0Texel = m_Level0Texel;   // (the epoch now ending)
            m_bPrevUsable = m_bHasBasis;
            const xmath::fvec3 Up = std::abs(Z.m_Y) < 0.99f ? xmath::fvec3(0, 1, 0) : xmath::fvec3(1, 0, 0);
            m_X = xmath::fvec3::Cross(Up, Z).NormalizeCopy();
            m_Y = xmath::fvec3::Cross(Z, m_X);
            m_Z = Z;
            m_Level0Texel = Level0Texel;
            m_bHasBasis = true;
            ++m_Epoch; ++m_FrameEpochs;
            KeepAsPrevious(Frame);
            for (auto& b : m_bDepthSet) b = false;
            return true;
        }

        // Once a frame, before the shadow pass: the windows, the invalidations, the net, the requested pages admitted and the pages to
        // render (m_ToRender), then the table (BuildTable). Level0Texel: world units per texel of level 0. bCache false: every resident
        // page renders again (the reference)
        void Update(const xmath::fvec3& LightDir, float Level0Texel, const xmath::fvec3& Camera, const std::vector<xmath::fbbox>& ChangedWorld
                   , bool bCache, std::span<const key> Requests, std::uint64_t Frame, std::span<const xmath::fbbox> MoverWorld = {}) noexcept
        {
            OpenEpoch(LightDir, Level0Texel, Frame);
            m_ToRender.clear();
            m_Stats = {};
            m_Stats.m_Epochs = m_FrameEpochs; m_Stats.m_Remapped = m_FrameRemapped; m_FrameEpochs = m_FrameRemapped = 0;
            m_Frame = Frame;
            const xmath::fvec3 Z = LightDir.NormalizeCopy();

            m_Stats.m_EpochMismatchDegrees = std::acos(std::clamp(Z.Dot(m_Z), -1.0f, 1.0f)) * 180.0f / 3.14159265f;

            // The windows; a level whose depth range moves loses its pages
            const xmath::fvec3 C = ToLight(Camera);
            for (int L = 0; L < m_Setup.m_Levels; ++L)
            {
                auto& Lv = m_Levels[L];
                Lv.m_PageSize = PageSize(L);
                Lv.m_OriginX  = int(std::floor(C.m_X / Lv.m_PageSize)) - m_Setup.m_Window / 2;
                Lv.m_OriginY  = int(std::floor(C.m_Y / Lv.m_PageSize)) - m_Setup.m_Window / 2;
                const float Half = DepthHalfRange(L);
                if (!m_bDepthSet[L] || std::abs(C.m_Z - Lv.m_DepthCenter) > Half * depth_recenter_v)
                {
                    Lv.m_DepthCenter = std::round(C.m_Z / (Half * depth_recenter_v)) * (Half * depth_recenter_v);
                    m_bDepthSet[L] = true;
                    for (int s = 0; s < int(m_Slots.size()); ++s) if (m_Slots[s].m_bUsed && !m_Slots[s].m_bPrev && !m_Slots[s].m_bMover && m_Slots[s].m_Key.m_Level == L) Free(s);
                }
            }
            m_PinX = int(std::floor(C.m_X / m_Levels[m_Setup.m_Levels - 1].m_PageSize));
            m_PinY = int(std::floor(C.m_Y / m_Levels[m_Setup.m_Levels - 1].m_PageSize));

            // Pages out of their window, or a net page out of the net, go
            for (int s = 0; s < int(m_Slots.size()); ++s)
            {
                auto& S = m_Slots[s];
                if (!S.m_bUsed || S.m_bPrev || S.m_bMover) continue;
                if (!InWindow(S.m_Key) || (s < m_Setup.NetSlots()) != IsPinned(S.m_Key)) Free(s);
            }

            // Caster changes: the pages under them are DIRTY now (out of the table until rendered again)
            if (!ChangedWorld.empty() || !bCache)
            {
                std::vector<lbox> Changed;
                Changed.reserve(ChangedWorld.size());
                for (auto& B : ChangedWorld) Changed.push_back(Bounds(B));
                for (auto& S : m_Slots)
                {
                    if (!S.m_bUsed || !S.m_bValid || S.m_bPrev || S.m_bMover) continue;
                    const float P = m_Levels[S.m_Key.m_Level].m_PageSize;
                    bool bDirty = !bCache;
                    for (std::size_t i = 0; !bDirty && i < Changed.size(); ++i) bDirty = RectOverlaps(Changed[i], S.m_Key.m_X * P, S.m_Key.m_Y * P, P);
                    if (bDirty) { S.m_bValid = false; ++m_Stats.m_Dirtied; }
                }
                if (m_bHasPrev)                                 // (the previous epoch's pages under them go: never rendered again)
                {
                    std::vector<lbox> PrevChanged;
                    for (auto& B : ChangedWorld) PrevChanged.push_back(BoundsIn(B, m_PX, m_PY, m_PZ));
                    for (int s = 0; s < int(m_Slots.size()); ++s)
                    {
                        auto& S = m_Slots[s];
                        if (!S.m_bUsed || !S.m_bPrev) continue;
                        const float P = m_PrevLevels[S.m_Key.m_Level].m_PageSize;
                        bool bDirty = !bCache;
                        for (std::size_t i = 0; !bDirty && i < PrevChanged.size(); ++i) bDirty = RectOverlaps(PrevChanged[i], S.m_Key.m_X * P, S.m_Key.m_Y * P, P);
                        if (bDirty) Free(s);
                    }
                }
            }

            // The net: every page present and VALID - setup::m_NetBudget of them a frame, nearest the camera first (the rest deferred: the
            // previous epoch's net shows there meanwhile)
            {
                const int Lc = m_Setup.m_Levels - 1, R = m_Setup.m_PinnedRadius;
                std::vector<key> Net;
                for (int y = m_PinY - R; y < m_PinY + R; ++y)
                for (int x = m_PinX - R; x < m_PinX + R; ++x)
                {
                    const key K{ Lc, x, y };
                    const int s = Find(K);
                    if (s < 0 || !m_Slots[s].m_bValid) Net.push_back(K);
                }
                std::sort(Net.begin(), Net.end(), [&](const key& a, const key& b) { return PageDistance2(a, C) < PageDistance2(b, C); });
                int NetBudget = m_Setup.m_NetBudget > 0 && m_bHasPrev ? m_Setup.m_NetBudget : int(Net.size());     // (no previous net to show: all of it - the first)
                for (const auto& K : Net)
                {
                    if (NetBudget <= 0) { ++m_Stats.m_Deferred; ++m_Stats.m_NetDeferred; continue; }
                    int s = Find(K);
                    if (s < 0)
                    {
                        s = FreePinnedSlot();
                        if (s < 0) continue;                    // (a net larger than its slots: setup error, counted below)
                        Assign(s, K, Frame);
                    }
                    Render(s, true); ++m_Stats.m_RenderedPinned; --NetBudget;
                }
            }

            // The requests: touched (protected), admitted by priority (coarser levels first, then nearer pages), rendered within the budget
            std::vector<key> Want;
            Want.reserve(Requests.size());
            for (const auto& K : Requests)
            {
                if (K.m_Level < 0 || K.m_Level >= m_Setup.m_Levels || !InWindow(K) || IsPinned(K)) continue;
                ++m_Stats.m_Requested;
                if (const int s = Find(K); s >= 0)
                {
                    m_Slots[s].m_LastRequested = Frame;
                    if (m_Slots[s].m_bValid) { ++m_Stats.m_Reused; continue; }
                }
                Want.push_back(K);
            }
            std::sort(Want.begin(), Want.end(), [&](const key& a, const key& b)
            {
                if (a.m_Level != b.m_Level) return a.m_Level > b.m_Level;
                return PageDistance2(a, C) < PageDistance2(b, C);
            });
            int Budget = m_Setup.m_Budget;
            for (const auto& K : Want)
            {
                if (Budget <= 0) { ++m_Stats.m_Deferred; continue; }
                int s = Find(K);
                if (s < 0)
                {
                    s = AdmitSlot(Frame);
                    if (s < 0) { ++m_Stats.m_Rejected; continue; }
                    Assign(s, K, Frame);
                    ++m_Stats.m_Admitted;
                }
                Render(s, false);
                --Budget;
                ++m_Stats.m_Rendered;
            }

            // The movers' pages: every page asked for (and of the net) a mover overlaps, the movers drawn into it this frame; the others go
            {
                std::vector<lbox> Movers;
                for (const auto& B : MoverWorld) Movers.push_back(Bounds(B));
                std::vector<key> Need;
                const auto Consider = [&](const key& K)
                {
                    const float P = m_Levels[K.m_Level].m_PageSize;
                    for (const auto& M : Movers) if (RectOverlaps(M, K.m_X * P, K.m_Y * P, P)) { Need.push_back(K); return; }
                };
                if (!Movers.empty())
                {
                    for (const auto& K : Requests) if (K.m_Level >= 0 && K.m_Level < m_Setup.m_Levels && InWindow(K)) Consider(K);
                    const int Lc = m_Setup.m_Levels - 1, R = m_Setup.m_PinnedRadius;
                    for (int y = m_PinY - R; y < m_PinY + R; ++y) for (int x = m_PinX - R; x < m_PinX + R; ++x) Consider({ Lc, x, y });
                    std::sort(Need.begin(), Need.end(), [](const key& a, const key& b) { return Pack(a) < Pack(b); });
                    Need.erase(std::unique(Need.begin(), Need.end()), Need.end());
                }
                std::unordered_map<std::uint64_t, int> Keep;
                for (const auto& K : Need)
                {
                    int s = -1;
                    if (const auto It = m_DynMap.find(Pack(K)); It != m_DynMap.end()) s = It->second;
                    else
                    {
                        s = AdmitSlot(Frame);
                        if (s < 0) { ++m_Stats.m_Rejected; continue; }
                        auto& S = m_Slots[s];
                        S = { K, Frame, S.m_Generation + 1, true, false, false, true };
                    }
                    m_Slots[s].m_LastRequested = Frame;
                    Keep[Pack(K)] = s;
                    Render(s, false, true);
                    ++m_Stats.m_RenderedMover;
                }
                for (const auto& [k, s] : m_DynMap) if (!Keep.contains(k)) { auto& S = m_Slots[s]; S.m_bUsed = S.m_bValid = S.m_bMover = false; }
                m_DynMap = std::move(Keep);
            }

            // The cross-fade: once this epoch has every page asked for (none deferred or rejected), toward it over m_FadeFrames; then the
            // previous epoch's pages go
            if (m_bHasPrev)
            {
                if ((m_Stats.m_Deferred == 0 && m_Stats.m_Rejected == 0 && Frame > m_EpochFrame) || Frame - m_EpochFrame > std::uint64_t(m_Setup.m_MaxPrevFrames))
                    m_Fade = std::min(1.0f, m_Fade + 1.0f / float(std::max(1, m_Setup.m_FadeFrames)));
                if (m_Fade >= 1.0f) DropPrevious();
            }

            for (int s = 0; s < int(m_Slots.size()); ++s)
            {
                if (!m_Slots[s].m_bUsed) continue;
                if (m_Slots[s].m_bMover)       ++m_Stats.m_ResidentMover;
                else if (m_Slots[s].m_bPrev)   ++m_Stats.m_ResidentPrev;
                else if (s < m_Setup.NetSlots()) { ++m_Stats.m_ResidentPinned; m_Stats.m_PinnedNotValid += !m_Slots[s].m_bValid; }
                else                           ++m_Stats.m_ResidentFine;
            }
            m_Stats.m_Fade = m_bHasPrev ? m_Fade : 1.0f;

            // Dynamic resolution (as Unreal's FVirtualShadowMapArrayCacheManager: down fast, up after 10 frames that fit, slowly); a frame
            // without requests (an epoch's first frames, nothing in view) leaves it as it is
            if (!Requests.empty() && m_Setup.m_MaxDynamicBias > 0)
            {
                const float Load   = float(m_Stats.m_Requested) / float(std::max(1, m_Setup.Slots() - m_Setup.NetSlots()));
                const float Target = std::max(0.0f, m_DynamicBias + std::log2(std::max(Load, 1e-3f) / m_Setup.m_PoolLoadFactor));
                if (Load > m_Setup.m_PoolLoadFactor) { m_LastOverload = Frame; m_DynamicBias += (Target - m_DynamicBias) * 0.5f; }
                else if (Frame - m_LastOverload > 10) m_DynamicBias += (Target - m_DynamicBias) * 0.1f;
                m_DynamicBias = std::clamp(m_DynamicBias, 0.0f, m_Setup.m_MaxDynamicBias);
                m_Stats.m_PoolLoad = Load;
            }
            m_Stats.m_DynamicBias = m_DynamicBias;
        }

        // The resolution bias of this frame's request pass and shading: the setup's and dynamic resolution's
        float ResolutionBias(void) const noexcept { return m_Setup.m_ResolutionBias + m_DynamicBias; }

        // The page table of this frame: per entry (level-major, then the toroidal row, column) a 16-bit slot + 1 of a VALID page, 0 none;
        // two entries a uint (even index: low half)
        void BuildTable(std::vector<std::uint32_t>& Out) const noexcept
        {
            Out.assign(std::size_t(m_Setup.TableUints()), 0u);
            for (int s = 0; s < int(m_Slots.size()); ++s)
            {
                const auto& S = m_Slots[s];
                if (!S.m_bUsed || !S.m_bValid || S.m_bPrev || S.m_bMover || !InWindow(S.m_Key)) continue;
                const int i = TableIndex(S.m_Key);
                Out[std::size_t(i) >> 1] |= std::uint32_t(s + 1) << ((i & 1) * 16);
            }
        }

        // The movers' table (this frame's mover pages): the same layout
        void BuildMoverTable(std::vector<std::uint32_t>& Out) const noexcept
        {
            Out.assign(std::size_t(m_Setup.TableUints()), 0u);
            for (const auto& [k, s] : m_DynMap)
            {
                const auto& S = m_Slots[s];
                if (!S.m_bUsed || !S.m_bValid || !InWindow(S.m_Key)) continue;
                const int i = TableIndex(S.m_Key);
                Out[std::size_t(i) >> 1] |= std::uint32_t(s + 1) << ((i & 1) * 16);
            }
        }
        bool HasMovers(void) const noexcept { return !m_DynMap.empty(); }

        // The previous epoch's table (its windows; empty when there is none): the same layout
        void BuildPrevTable(std::vector<std::uint32_t>& Out) const noexcept
        {
            Out.assign(std::size_t(m_Setup.TableUints()), 0u);
            if (!m_bHasPrev) return;
            for (int s = 0; s < int(m_Slots.size()); ++s)
            {
                const auto& S = m_Slots[s];
                if (!S.m_bUsed || !S.m_bPrev || !S.m_bValid) continue;
                const auto& Lv = m_PrevLevels[S.m_Key.m_Level];
                if (S.m_Key.m_X < Lv.m_OriginX || S.m_Key.m_X >= Lv.m_OriginX + m_Setup.m_Window || S.m_Key.m_Y < Lv.m_OriginY || S.m_Key.m_Y >= Lv.m_OriginY + m_Setup.m_Window) continue;
                const int i = TableIndex(S.m_Key);
                Out[std::size_t(i) >> 1] |= std::uint32_t(s + 1) << ((i & 1) * 16);
            }
        }
        bool HasPrevious(void) const noexcept { return m_bHasPrev; }
        float Fade(void) const noexcept { return m_bHasPrev ? m_Fade : 1.0f; }

        // Invariant 1 (the tests): every table entry names a used, VALID slot holding exactly the page of that entry in this frame's window
        bool Validate(std::span<const std::uint32_t> Table) const noexcept
        {
            const int W = m_Setup.m_Window;
            for (int i = 0; i < m_Setup.TableEntries(); ++i)
            {
                const std::uint32_t E = (Table[std::size_t(i) >> 1] >> ((i & 1) * 16)) & 0xFFFFu;
                if (E == 0) continue;
                const int s = int(E) - 1;
                if (s >= int(m_Slots.size())) return false;
                const auto& S = m_Slots[s];
                const int L = i / (W * W);
                const key K{ L, Unwrap(i % W, m_Levels[L].m_OriginX), Unwrap((i / W) % W, m_Levels[L].m_OriginY) };
                if (!S.m_bUsed || !S.m_bValid || S.m_bPrev || S.m_bMover || !(S.m_Key == K)) return false;
            }
            return true;
        }

        // A slot's generation (it changes whenever the slot takes another page): the tests check reuse with it
        std::uint32_t Generation(int Slot) const noexcept { return m_Slots[Slot].m_Generation; }
        int  Find(const key& K) const noexcept { const auto It = m_Map.find(Pack(K)); return It == m_Map.end() ? -1 : It->second; }
        bool IsValid(const key& K) const noexcept { const int s = Find(K); return s >= 0 && m_Slots[s].m_bValid; }

        float PageSize      (int Level) const noexcept { return m_Level0Texel * float(page_texels_v) * float(1 << Level); }
        float DepthHalfRange(int Level) const noexcept { return PageSize(Level) * float(m_Setup.m_Window) * depth_range_v; }
        float Level0Texel   (void) const noexcept { return m_Level0Texel; }
        xmath::fvec3 ToLight(const xmath::fvec3& P) const noexcept { return { P.Dot(m_X), P.Dot(m_Y), P.Dot(m_Z) }; }
        lbox Bounds(const xmath::fbbox& World) const noexcept { return BoundsIn(World, m_X, m_Y, m_Z); }
        static lbox BoundsIn(const xmath::fbbox& World, const xmath::fvec3& X, const xmath::fvec3& Y, const xmath::fvec3& Z) noexcept
        {
            lbox B;
            for (int i = 0; i < 8; ++i)
            {
                const xmath::fvec3 P{ (i & 1) ? World.m_Max.m_X : World.m_Min.m_X, (i & 2) ? World.m_Max.m_Y : World.m_Min.m_Y, (i & 4) ? World.m_Max.m_Z : World.m_Min.m_Z };
                B.Add({ P.Dot(X), P.Dot(Y), P.Dot(Z) });
            }
            return B;
        }

        // World -> clip of one page: x, y over its light-space rect to [-1,1], z over its level's depth range to [0,1]
        xmath::fmat4 PageW2C(const key& K) const noexcept
        {
            const auto& Lv   = m_Levels[K.m_Level];
            const float Half = DepthHalfRange(K.m_Level), P = Lv.m_PageSize;
            const float SX = 2.0f / P, SZ = 1.0f / (2.0f * Half);
            const float MinX = K.m_X * P, MinY = K.m_Y * P, ZMin = Lv.m_DepthCenter - Half;
            xmath::fmat4 M = xmath::fmat4::fromZero();
            M(0, 0) = m_X.m_X * SX; M(0, 1) = m_X.m_Y * SX; M(0, 2) = m_X.m_Z * SX; M(0, 3) = -MinX * SX - 1.0f;
            M(1, 0) = m_Y.m_X * SX; M(1, 1) = m_Y.m_Y * SX; M(1, 2) = m_Y.m_Z * SX; M(1, 3) = -MinY * SX - 1.0f;
            M(2, 0) = m_Z.m_X * SZ; M(2, 1) = m_Z.m_Y * SZ; M(2, 2) = m_Z.m_Z * SZ; M(2, 3) = -ZMin * SZ;
            M(3, 3) = 1.0f;
            return M;
        }

        // Every page goes on the next Update (the sun stopped casting: what changed meanwhile never reached the pages)
        void Invalidate(void) noexcept { m_bHasBasis = false; }

        xmath::fvec3                            m_X{ 1, 0, 0 }, m_Y{ 0, 1, 0 }, m_Z{ 0, 0, 1 };      // world -> light rotation rows
        std::array<level, max_levels_v>         m_Levels{};
        xmath::fvec3                            m_PX{ 1, 0, 0 }, m_PY{ 0, 1, 0 }, m_PZ{ 0, 0, 1 };   // the previous epoch's (HasPrevious)
        std::array<level, max_levels_v>         m_PrevLevels{};
        float                                   m_PrevLevel0Texel = 0;
        std::vector<page_to_render>             m_ToRender;
        stats                                   m_Stats;

    private:

        struct slot
        {
            key             m_Key;
            std::uint64_t   m_LastRequested = 0;
            std::uint32_t   m_Generation = 0;
            bool            m_bUsed  = false;
            bool            m_bValid = false;
            bool            m_bPrev  = false;           // the previous epoch's page (in m_PrevMap)
            bool            m_bMover = false;           // a mover page (in m_DynMap): the movers only, this frame
        };

        float PageDistance2(const key& K, const xmath::fvec3& C) const noexcept
        {
            const float P = m_Levels[K.m_Level].m_PageSize, dx = (K.m_X + 0.5f) * P - C.m_X, dy = (K.m_Y + 0.5f) * P - C.m_Y;
            return dx * dx + dy * dy;
        }

        void Assign(int s, const key& K, std::uint64_t Frame) noexcept
        {
            auto& S = m_Slots[s];
            S = { K, Frame, S.m_Generation + 1, true, false, false, false };
            m_Map[Pack(K)] = s;
        }

        void Free(int s) noexcept
        {
            auto& S = m_Slots[s];
            if (!S.m_bUsed) return;
            if (S.m_bMover)     m_DynMap.erase(Pack(S.m_Key));
            else if (S.m_bPrev) m_PrevMap.erase(Pack(S.m_Key));
            else                m_Map.erase(Pack(S.m_Key));
            S.m_bUsed = S.m_bValid = S.m_bPrev = S.m_bMover = false;
            if (s >= m_Setup.NetSlots()) ++m_Stats.m_Evicted;
        }

        // A new epoch: the previous one's pages go; this one's VALID fine pages become the previous (the net and the rest go)
        void KeepAsPrevious(std::uint64_t Frame) noexcept
        {
            DropPrevious();
            for (int s = 0; s < int(m_Slots.size()); ++s)
            {
                auto& S = m_Slots[s];
                if (!S.m_bUsed) continue;
                if (S.m_bMover) { Free(s); continue; }
                if ((s < m_Setup.NetSlots() && m_Setup.m_NetBudget <= 0) || !S.m_bValid || !m_bPrevUsable) { Free(s); continue; }
                m_Map.erase(Pack(S.m_Key));
                S.m_bPrev = true;
                m_PrevMap[Pack(S.m_Key)] = s;
            }
            m_Map.clear();
            m_bHasPrev = m_bPrevUsable && !m_PrevMap.empty();
            m_EpochFrame = Frame;
            m_Fade = 0;
        }

        void DropPrevious(void) noexcept
        {
            for (int s = 0; s < int(m_Slots.size()); ++s) if (m_Slots[s].m_bUsed && m_Slots[s].m_bPrev) Free(s);
            m_PrevMap.clear();
            m_bHasPrev = false;
            m_Fade = 1;
        }

        int FreePinnedSlot(void) const noexcept
        {
            for (int s = 0; s < std::min(m_Setup.NetSlots(), int(m_Slots.size())); ++s) if (!m_Slots[s].m_bUsed) return s;
            return -1;
        }

        // A fine slot: a free one, else the least recently requested page that is neither requested lately nor rendered this frame
        int AdmitSlot(std::uint64_t Frame) noexcept
        {
            int Best = -1, Prev = -1;
            for (int s = m_Setup.NetSlots(); s < int(m_Slots.size()); ++s)
            {
                const auto& S = m_Slots[s];
                if (!S.m_bUsed) return s;
                if (S.m_bPrev) { if (Prev < 0) Prev = s; continue; }
                if (S.m_LastRequested + std::uint64_t(m_Setup.m_ProtectFrames) >= Frame) continue;     // protected (requested lately)
                if (m_RenderedFrame[s] == Frame) continue;                                              // in flight (rendered this frame)
                if (Best < 0 || S.m_LastRequested < m_Slots[Best].m_LastRequested) Best = s;
            }
            if (Prev >= 0) Best = Prev;                     // (the previous epoch's pages first: they only bridge the rebuild)
            if (Best >= 0) Free(Best);
            return Best;
        }

        void Render(int s, bool bPinned, bool bMover = false) noexcept
        {
            auto& S = m_Slots[s];
            S.m_bValid = true;
            m_RenderedFrame[s] = m_Frame;
            const float P = m_Levels[S.m_Key.m_Level].m_PageSize;
            const int   R = m_Setup.SlotsPerRow();
            m_ToRender.push_back({ S.m_Key, s, (s % R) * page_texels_v, (s / R) * page_texels_v, PageW2C(S.m_Key), S.m_Key.m_X * P, S.m_Key.m_Y * P, P, bPinned, bMover });
        }

        setup                                   m_Setup;
        std::uint64_t                           m_Frame = 0;
        float                                   m_DynamicBias  = 0;
        std::uint64_t                           m_LastOverload = 0;
        std::vector<slot>                       m_Slots;
        std::vector<std::uint64_t>              m_RenderedFrame;
        std::unordered_map<std::uint64_t, int>  m_Map;
        std::unordered_map<std::uint64_t, int>  m_PrevMap;
        std::unordered_map<std::uint64_t, int>  m_DynMap;               // this frame's mover pages
        std::uint64_t                           m_EpochFrame = 0;
        int                                     m_FrameEpochs = 0, m_FrameRemapped = 0;     // (this frame's, before Update resets the stats)
        float                                   m_Fade = 1;
        bool                                    m_bHasPrev = false, m_bPrevUsable = false;
        std::array<bool, max_levels_v>          m_bDepthSet{};
        std::uint64_t                           m_Epoch = 0;
        float                                   m_Level0Texel = 0;
        int                                     m_PinX = 0, m_PinY = 0;
        bool                                    m_bHasBasis = false;
    };
}

#endif
