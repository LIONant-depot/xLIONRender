#pragma once
// GI chunks (documentation/gi_chunks_plan.md, v3): a geometry over the 16-bit surfel cap baked once as a whole (wide indices), then
// partitioned into chunks of whole clusters. Each chunk is a surfel_mesh of its own: its triangles, the surfels it owns, then ghost
// slots (copies of other chunks' surfels within reach of its surface: the reads near a border blend across it), all with chunk-local
// 16-bit indices; its read grid, neighbours and origins are built on that chunk with the whole geometry's visibility. Every surfel has
// exactly one owner; ghosts are never traced, never emit, never count as coverage. Equal levels only (mixed levels are gated: plan v3).
#include "xlionrender_surfel_gi.h"
#include <algorithm>
#include <chrono>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <atomic>

namespace xlionrender::gi
{
    constexpr std::size_t chunk_local_cap_v = 0xFFFE;      // a chunk's local indices (owned + ghosts): 0xFFFF is "none" everywhere
    constexpr int         chunk_halo_passes_v = 3;          // the filter passes a chunk's ghosts carry (the shown light's: world's filter passes)

    struct chunk_ghost { std::uint32_t m_Chunk = 0, m_Local = 0; };     // a ghost slot's owner: its chunk and its index there

    struct chunked_mesh
    {
        std::vector<surfel_mesh>                m_Chunks;
        std::vector<std::uint32_t>              m_nOwned;           // per chunk: its owned surfels (local 0 .. n - 1); the ghosts after them
        std::vector<std::vector<chunk_ghost>>   m_Ghosts;           // per chunk: per ghost slot (local m_nOwned + k), its owner
        std::vector<std::uint16_t>              m_ClusterChunk;     // per cluster: its chunk
        std::vector<std::array<std::uint32_t, 2>> m_Home;           // per whole surfel: { its chunk, its local index there }
        std::vector<std::vector<std::uint32_t>> m_VertWhole;        // per chunk: per vertex of it, the whole's (SectionGeometry)
    };

    struct chunk_report
    {
        int         m_nChunks = 0, m_nSplitsAgain = 0;
        std::size_t m_OwnedMin = 0, m_OwnedMax = 0, m_GhostsTotal = 0, m_GhostsMax = 0, m_LocalMax = 0;
        int         m_nClamped = 0;                                 // chunks whose ghost candidates passed the cap: the nearest kept
        double      m_PartitionMs = 0, m_BuildMs = 0;
        bool        m_bFits = true;                                 // every chunk's owned + ghosts within chunk_local_cap_v
        std::function<void(float)> m_OnProgress;                    // (in) after each chunk built, 0..1 (the compiler's progress bar)
    };

    // A section's geometry: Whole's triangles whose cluster is in it (InG), in Whole's order, and their vertices in Whole's order (VMap: whole
    // -> section vertex, -1 none; VertWhole: section -> whole). In order: a cluster's vertices (a range of Whole's) stay a range, so a draw of it
    // finds its section's vertex as its own plus a constant (the renderer's per-cluster delta). The compiler and the renderer both make it this
    // way: a section's packed structures (its origins, its vertex surfels) name these vertices
    inline void SectionGeometry(const surfel_mesh& Whole, std::span<const int> TriCluster, std::span<const char> InG, surfel_mesh& C,
                                std::vector<int>& VMap, std::vector<std::uint32_t>& VertWhole) noexcept
    {
        VMap.assign(Whole.m_Pos.size(), -1);
        VertWhole.clear();
        for (std::size_t t = 0; t < Whole.m_Tris.size(); ++t)
            if (InG[std::size_t(TriCluster[t])]) for (const int v : Whole.m_Tris[t]) VMap[std::size_t(v)] = 0;
        for (std::size_t v = 0; v < Whole.m_Pos.size(); ++v)
        {
            if (VMap[v] < 0) continue;
            VMap[v] = int(C.m_Pos.size());
            VertWhole.push_back(std::uint32_t(v));
            C.m_Pos.push_back(Whole.m_Pos[v]); C.m_Nrm.push_back(Whole.m_Nrm[v]);
            if (!Whole.m_UV.empty()) C.m_UV.push_back(Whole.m_UV[v]);
        }
        for (std::size_t t = 0; t < Whole.m_Tris.size(); ++t)
        {
            if (!InG[std::size_t(TriCluster[t])]) continue;
            const auto& W = Whole.m_Tris[t];
            C.m_Tris.push_back({ VMap[std::size_t(W[0])], VMap[std::size_t(W[1])], VMap[std::size_t(W[2])] });
            if (!Whole.m_TriMaterial.empty()) C.m_TriMaterial.push_back(Whole.m_TriMaterial[t]);
        }
    }

    // A section's neighbour lists from the whole mesh's: the same pairs (the section's tests see the whole geometry: its occluder), kept where
    // both surfels are in it, in the section's indices. FineWhole / ParentWhole: per section surfel / parent, the whole's. Building them again
    // on a section was its longest step (54 s for one of the Bistro's, its rays against 2.8M triangles), and the same answer
    inline void NeighborsFromWhole(const surfel_mesh& Whole, surfel_mesh& C, std::span<const std::uint32_t> FineWhole, std::span<const std::uint32_t> ParentWhole) noexcept
    {
        for (int Lv = 0; Lv < 2; ++Lv)
        {
            const auto& WN  = Whole.m_Neighbors[std::size_t(Lv)];
            const auto  Map = Lv ? ParentWhole : FineWhole;
            auto&       Nb  = C.m_Neighbors[std::size_t(Lv)];
            Nb = {};
            if (WN.m_First.size() != Whole.m_Levels[std::size_t(Lv)].size() + 1) continue;
            std::vector<std::uint32_t> ToLocal(Whole.m_Levels[std::size_t(Lv)].size(), 0xFFFFFFFFu);
            for (std::size_t k = 0; k < Map.size(); ++k) ToLocal[Map[k]] = std::uint32_t(k);
            Nb.m_First.assign(Map.size() + 1, 0);
            for (std::size_t k = 0; k < Map.size(); ++k)
            {
                const std::uint32_t w = Map[k];
                for (std::uint32_t j = WN.m_First[w]; j < WN.m_First[w + 1]; ++j)
                    if (const std::uint32_t n = ToLocal[WN.m_Index[j]]; n != 0xFFFFFFFFu) { Nb.m_Index.push_back(n); Nb.m_Weight.push_back(WN.m_Weight[j]); }
                Nb.m_First[k + 1] = std::uint32_t(Nb.m_Index.size());
            }
        }
    }

    // Whole: baked (Bake: its levels, parents, vertex surfels), its BVH built (BuildRays). TriCluster: per triangle of Whole its cluster.
    // TargetOwned: the owned surfels a chunk aims at (a start: every chunk is checked against chunk_local_cap_v after its ghosts, and
    // split again when it does not fit)
    inline bool PartitionChunks(const surfel_mesh& Whole, std::span<const int> TriCluster, int nClusters, std::size_t TargetOwned,
                                chunked_mesh& Out, chunk_report* pReport = nullptr, std::span<const int> FixedGroup = {}) noexcept
    {
        using clock = std::chrono::steady_clock;
        const auto T0 = clock::now();
        const auto& L0 = Whole.m_Levels[0];
        const std::size_t nS = L0.size();
        if (nS == 0 || TriCluster.size() != Whole.m_Tris.size() || nClusters <= 0) return false;

        // 1. each surfel's cluster: the majority of the vertices that name it (Bake's vertex surfels); none: the nearest cluster centre
        std::vector<int> VertCluster(Whole.m_Pos.size(), -1);
        for (std::size_t t = 0; t < Whole.m_Tris.size(); ++t) for (const int v : Whole.m_Tris[t]) if (VertCluster[std::size_t(v)] < 0) VertCluster[std::size_t(v)] = TriCluster[t];
        std::vector<v3> CC(std::size_t(nClusters), v3{ 0, 0, 0 }); std::vector<int> CN(std::size_t(nClusters), 0);
        for (std::size_t t = 0; t < Whole.m_Tris.size(); ++t)
        {
            const auto& T = Whole.m_Tris[t];
            CC[std::size_t(TriCluster[t])] = CC[std::size_t(TriCluster[t])] + (Whole.m_Pos[T[0]] + Whole.m_Pos[T[1]] + Whole.m_Pos[T[2]]) * (1.0f / 3.0f); ++CN[std::size_t(TriCluster[t])];
        }
        for (int c = 0; c < nClusters; ++c) if (CN[std::size_t(c)]) CC[std::size_t(c)] = CC[std::size_t(c)] * (1.0f / float(CN[std::size_t(c)]));
        std::vector<std::unordered_map<int, int>> Votes(nS);
        for (std::size_t v = 0; v < Whole.m_VertSurfels.size(); ++v)
            for (const int s : Whole.m_VertSurfels[v]) if (s >= 0 && VertCluster[v] >= 0) ++Votes[std::size_t(s)][VertCluster[v]];
        std::vector<int> SurfCluster(nS, 0);
        for (std::size_t s = 0; s < nS; ++s)
        {
            int Best = -1, BestN = 0;
            for (const auto& [c, n] : Votes[s]) if (n > BestN || (n == BestN && c < Best)) { Best = c; BestN = n; }
            if (Best < 0)      // ponytail: brute force over the clusters (rare: a surfel no vertex names); a grid of the centres if it shows
            {
                float D = 1e30f;
                for (int c = 0; c < nClusters; ++c) if (CN[std::size_t(c)]) { const float d = (CC[std::size_t(c)] - L0[s].m_Center).Dot(CC[std::size_t(c)] - L0[s].m_Center); if (d < D) { D = d; Best = c; } }
            }
            SurfCluster[s] = std::max(Best, 0);
        }
        Votes.clear(); Votes.shrink_to_fit();

        // 2. the partition: recursive median split of the clusters by their centres along the longest axis, weighted by the surfels
        //    they own, until a group owns at most Target
        std::vector<std::size_t> W(std::size_t(nClusters), 0);
        for (const int c : SurfCluster) ++W[std::size_t(c)];
        Out = {};
        Out.m_ClusterChunk.assign(std::size_t(nClusters), 0);
        chunk_report Rep;
        std::vector<std::vector<int>> Groups;
        const auto Split = [&](auto&& Self, std::vector<int> G, std::size_t Target) -> void
        {
            std::size_t Sum = 0; for (const int c : G) Sum += W[std::size_t(c)];
            if (Sum <= Target || G.size() <= 1) { Groups.push_back(std::move(G)); return; }
            v3 Lo{ 1e30f, 1e30f, 1e30f }, Hi{ -1e30f, -1e30f, -1e30f };
            for (const int c : G) { Lo = v3::Min(Lo, CC[std::size_t(c)]); Hi = v3::Max(Hi, CC[std::size_t(c)]); }
            const v3 E = Hi - Lo;
            const int Ax = E.m_X >= E.m_Y && E.m_X >= E.m_Z ? 0 : E.m_Y >= E.m_Z ? 1 : 2;
            const auto At = [&](int c) { const v3& p = CC[std::size_t(c)]; return Ax == 0 ? p.m_X : Ax == 1 ? p.m_Y : p.m_Z; };
            std::sort(G.begin(), G.end(), [&](int a, int b) { return At(a) != At(b) ? At(a) < At(b) : a < b; });
            std::size_t Acc = 0, k = 0;
            while (k + 1 < G.size() && Acc + W[std::size_t(G[k])] <= Sum / 2) Acc += W[std::size_t(G[k++])];
            k = std::max<std::size_t>(k, 1);
            Self(Self, std::vector<int>(G.begin(), G.begin() + std::ptrdiff_t(k)), Target);
            Self(Self, std::vector<int>(G.begin() + std::ptrdiff_t(k), G.end()), Target);
        };
        {
            if (FixedGroup.size() == std::size_t(nClusters))      // the groups given (the compiler's regions: its clusters cut at them); one that does not fit is split below
            {
                std::map<int, std::vector<int>> ByGroup;
                for (int c = 0; c < nClusters; ++c) if (CN[std::size_t(c)]) ByGroup[std::max(FixedGroup[std::size_t(c)], 0)].push_back(c);
                for (auto& [g, G] : ByGroup) Groups.push_back(std::move(G));
            }
            else
            {
                std::vector<int> All; for (int c = 0; c < nClusters; ++c) if (CN[std::size_t(c)]) All.push_back(c);
                Split(Split, std::move(All), TargetOwned);
            }
        }
        Rep.m_PartitionMs = std::chrono::duration<double, std::milli>(clock::now() - T0).count();

        // 3. the chunks; one that does not fit after its ghosts is split again (its clusters, half the target)
        const auto T1 = clock::now();
        const float Rmax = [&] { float r = 0; for (const auto& S : L0) r = std::max(r, S.m_Radius); return r; }();
        surfel_grid WG{ 5 * Rmax, {} };                         // the whole's surfels (the ghosts' candidates)
        for (std::size_t s = 0; s < nS; ++s) WG.Add(L0[s].m_Center, int(s));
        for (std::size_t g = 0; g < Groups.size(); ++g)
        {
            const auto G = Groups[g];       // (a copy: a group that does not fit appends its parts to Groups)
            auto TS = clock::now();                                 // (XLION_GI_CHUNK_TRACE: each chunk's steps, ms, to stderr)
            std::string Trace;
            const auto Step = [&](const char* pN) { if (!std::getenv("XLION_GI_CHUNK_TRACE")) return; const auto T = clock::now(); Trace += std::string(" ") + pN + " " + std::to_string(int(std::chrono::duration<double, std::milli>(T - TS).count())); TS = T; };
            const std::size_t Chunk = Out.m_Chunks.size();
            std::vector<char> InG(std::size_t(nClusters), 0);
            for (const int c : G) InG[std::size_t(c)] = 1;
            surfel_mesh C;
            std::vector<int> VMap;
            std::vector<std::uint32_t> VertWhole;
            SectionGeometry(Whole, TriCluster, InG, C, VMap, VertWhole);
            Step("tris");
            // its surfels: the owned ones, then ghosts - the others' whose reach (2 radii) touches its bounds
            std::vector<std::uint32_t> Local;      // local -> whole surfel
            std::vector<chunk_ghost> Ghosts;
            std::unordered_map<std::uint32_t, std::uint32_t> ToLocal;
            for (std::size_t s = 0; s < nS; ++s) if (InG[std::size_t(SurfCluster[s])]) { ToLocal[std::uint32_t(s)] = std::uint32_t(Local.size()); Local.push_back(std::uint32_t(s)); }
            const std::size_t nOwned = Local.size();
            // ghost candidates: the others' surfels near its surface - within 5 radii of points on its triangles a radius apart (a read cell, at
            // most a radius wide, is crossed by the surface: its far corner within 1.8 radii of a surface point, its list reaches 2 radii past
            // it; a point is within 0.6 radii of a sample). Its bounds' box was a wide net: 129% of the owned on Police; the compaction below
            // keeps the ones named
            // The samples by cell of a radius (each cell asked once, from its center, its half diagonal added: the Bistro's ~600k samples a
            // chunk, each looking at a 15 m box of surfels, took more than 25 minutes)
            // Each candidate's distance to the surface (its nearest cell centre, cells of half a radius): over the local cap the nearest are kept.
            // The Bistro at 3.1 m: a section's 18 m net held more than a section can index, and splitting it again went down to 1,048 sections
            // of 2 surfels. The far ones are the reads' least (a list keeps the 32 nearest seen) and the compaction drops most of them anyway
            const float Hc = 0.5f * Rmax;
            std::vector<float> Dist(nS, 1e30f);
            std::unordered_set<std::uint64_t> Cells;
            for (const auto& T : C.m_Tris)
            {
                const v3 A = C.m_Pos[std::size_t(T[0])], B = C.m_Pos[std::size_t(T[1])], Cc = C.m_Pos[std::size_t(T[2])];
                const int k = std::min(1024, std::max(1, int(std::ceil(std::max({ (B - A).Length(), (Cc - B).Length(), (A - Cc).Length() }) / Hc))));
                for (int i = 0; i <= k; ++i) for (int j = 0; i + j <= k; ++j)
                {
                    const v3 P = A + (B - A) * (float(i) / float(k)) + (Cc - A) * (float(j) / float(k));
                    Cells.insert(surfel_grid::Key(int(std::floor(P.m_X / Hc)), int(std::floor(P.m_Y / Hc)), int(std::floor(P.m_Z / Hc))));
                }
            }
            const auto Sx = [](std::uint64_t v) { const int i = int(v & 0x1FFFFF); return i >= 0x100000 ? i - 0x200000 : i; };     // (surfel_grid::Key's fields)
            const std::vector<std::uint64_t> CellList(Cells.begin(), Cells.end());
            ParallelFor(int(CellList.size()), [&](int c)
            {
                const std::uint64_t K = CellList[std::size_t(c)];
                const v3 P{ (float(Sx(K >> 42)) + 0.5f) * Hc, (float(Sx(K >> 21)) + 0.5f) * Hc, (float(Sx(K)) + 0.5f) * Hc };
                WG.Near(P, 1, [&](int s)
                {
                    if (InG[std::size_t(SurfCluster[std::size_t(s)])]) return;
                    const float d = (L0[std::size_t(s)].m_Center - P).Length();
                    if (d >= 4.5f * Rmax) return;
                    std::atomic_ref<float> A(Dist[std::size_t(s)]);
                    for (float o = A.load(std::memory_order_relaxed); d < o && !A.compare_exchange_weak(o, d, std::memory_order_relaxed);) {}
                });
            });
            std::vector<std::pair<float, std::uint32_t>> Cand;
            for (std::size_t s = 0; s < nS; ++s) if (Dist[s] < 1e29f) Cand.push_back({ Dist[s], std::uint32_t(s) });
            const std::size_t Room = chunk_local_cap_v > nOwned ? chunk_local_cap_v - nOwned : 0;
            if (Cand.size() > Room) { std::nth_element(Cand.begin(), Cand.begin() + std::ptrdiff_t(Room), Cand.end()); Cand.resize(Room); ++Rep.m_nClamped; }
            std::sort(Cand.begin(), Cand.end(), [](const auto& a, const auto& b) { return a.second < b.second; });       // (the whole's order, as before)
            for (const auto& [d, s] : Cand)
            {
                ToLocal[s] = std::uint32_t(Local.size()); Local.push_back(s);
                Ghosts.push_back({ 0xFFFFFFFFu, s });        // (its owner chunk and local index: resolved below, once all chunks exist)
            }
            if (nOwned + 1024 > chunk_local_cap_v && G.size() > 1)      // (its owned alone too many: split again - not for its ghosts, kept the nearest above)
            {
                // does not fit: this group again, half its owned surfels as the target (its parts appended to the groups to make)
                std::size_t Sum = 0; for (const int c : G) Sum += W[std::size_t(c)];
                Split(Split, G, std::max<std::size_t>(Sum / 2, 1));
                Groups[g].clear();
                ++Rep.m_nSplitsAgain;
                continue;
            }
            for (const int c : G) Out.m_ClusterChunk[std::size_t(c)] = std::uint16_t(Chunk);
            C.m_Levels[0].reserve(Local.size());
            for (const std::uint32_t s : Local) C.m_Levels[0].push_back(L0[s]);
            if (!Whole.m_SurfelMaterial.empty()) for (const std::uint32_t s : Local) C.m_SurfelMaterial.push_back(Whole.m_SurfelMaterial[s]);
            // parents: those of its owned surfels, in the order first met (a parent with children in two chunks is in both). A ghost's parent
            // is its own when the chunk has it, else the chunk's nearest facing parent (level 1 reads through it; ghosts are equal-level
            // copies only). Every ghost's own parent made the Bistro's 211k parents 524k in its sections
            std::unordered_map<int, int> PMap;
            C.m_ParentOf.resize(Local.size(), 0);
            const auto AddParent = [&](std::size_t k)
            {
                const int p = Whole.m_ParentOf.empty() ? 0 : Whole.m_ParentOf[Local[k]];
                auto [It, bNew] = PMap.try_emplace(p, int(C.m_Levels[1].size()));
                if (bNew && std::size_t(p) < Whole.m_Levels[1].size()) C.m_Levels[1].push_back(Whole.m_Levels[1][std::size_t(p)]);
                C.m_ParentOf[k] = It->second;
            };
            for (std::size_t k = 0; k < nOwned; ++k) AddParent(k);
            if (Local.size() > nOwned && !C.m_Levels[1].empty())
            {
                const auto& P1 = C.m_Levels[1];
                surfel_grid PG{ 2 * P1[0].m_Radius, {} };
                for (int p = 0; p < int(P1.size()); ++p) PG.Add(P1[std::size_t(p)].m_Center, p);
                for (std::size_t k = nOwned; k < Local.size(); ++k)
                {
                    const int wp = Whole.m_ParentOf.empty() ? 0 : Whole.m_ParentOf[Local[k]];
                    if (const auto It = PMap.find(wp); It != PMap.end()) { C.m_ParentOf[k] = It->second; continue; }
                    const auto& S = L0[Local[k]];
                    int Best = -1; float BestD = 1e30f;
                    PG.Near(S.m_Center, 1, [&](int p) { const auto& Pp = P1[std::size_t(p)]; const float d = (Pp.m_Center - S.m_Center).Length(); if (Pp.m_Normal.Dot(S.m_Normal) >= 0.5f && d < BestD) { BestD = d; Best = p; } });
                    if (Best >= 0) C.m_ParentOf[k] = Best; else AddParent(k);
                }
            }
            else for (std::size_t k = nOwned; k < Local.size(); ++k) AddParent(k);
            Step("ghosts+parents");
            // vertex surfels, chunk-local (a surfel no slot holds: none)
            C.m_VertSurfels.assign(C.m_Pos.size(), { -1, -1, -1, -1 });
            for (std::size_t v = 0; v < Whole.m_Pos.size(); ++v)
            {
                if (VMap[v] < 0 || v >= Whole.m_VertSurfels.size()) continue;
                for (int k = 0; k < 4; ++k)
                {
                    const int s = Whole.m_VertSurfels[v][std::size_t(k)];
                    if (s < 0) continue;
                    const auto It = ToLocal.find(std::uint32_t(s));
                    C.m_VertSurfels[std::size_t(VMap[v])][std::size_t(k)] = It == ToLocal.end() ? -1 : int(It->second);
                }
            }
            C.m_bLeakProtection = Whole.m_bLeakProtection; C.m_bBounce = Whole.m_bBounce;
            // its structures, with the whole geometry's visibility
            Step("vertsurfels");
            C.BuildRays();
            Step("rays");
            C.m_pOccluder = &Whole;
            std::vector<std::uint32_t> ParentWhole(C.m_Levels[1].size(), 0);
            for (const auto& [p, l] : PMap) if (std::size_t(l) < ParentWhole.size()) ParentWhole[std::size_t(l)] = std::uint32_t(p);
            C.BuildOrigins();
            NeighborsFromWhole(Whole, C, Local, ParentWhole);
            Step("neighbours");
            C.BuildReadGrid();
            Step("readgrid");
            if (std::getenv("XLION_GI_CHUNK_TRACE")) Trace += " (probes " + std::to_string(int(C.m_ReadGridMs[0])) + ", samples " + std::to_string(int(C.m_ReadGridMs[1])) + ", parts " + std::to_string(int(C.m_ReadGridMs[2])) + ", lists " + std::to_string(int(C.m_ReadGridMs[3])) + ")";
            // Only the ghosts its consumers name stay (the bounds' ghosts were 129% of the owned on Police, 9% named): the read lists, the
            // vertex surfels, and the neighbour lists from the owned surfels out chunk_halo_passes_v filter passes; the rest are dropped and
            // the indices remapped (the owned keep theirs), its neighbours and origins made again on what is left
            {
                const std::uint32_t nAll = std::uint32_t(C.m_Levels[0].size());
                std::vector<char> Used(nAll, 0);
                for (std::uint32_t o = 0; o < std::uint32_t(nOwned); ++o) Used[o] = 1;
                for (const auto& E : C.m_ReadTable)
                {
                    if (E[1] == 0xFFFFFFFFu) continue;
                    for (std::uint32_t p = 0; p < E[3]; ++p)
                    {
                        const std::uint32_t F = C.m_ReadList[E[2] + 4 * p + 2], n = C.m_ReadList[E[2] + 4 * p + 3] & 0xFFFFu;
                        for (std::uint32_t k = 0; k < n; ++k) Used[C.m_ReadList[F + k] & 0xFFFFu] = 1;
                    }
                }
                for (const auto& V : C.m_VertSurfels) for (const int sv : V) if (sv >= 0) Used[std::size_t(sv)] = 1;
                const auto& Nb = C.m_Neighbors[0];
                if (Nb.m_First.size() == std::size_t(nAll) + 1)
                    for (int Pass = 0; Pass < chunk_halo_passes_v; ++Pass)
                    {
                        std::vector<char> Next = Used;
                        for (std::uint32_t s = 0; s < nAll; ++s) if (Used[s]) for (std::uint32_t k = Nb.m_First[s]; k < Nb.m_First[s + 1]; ++k) Next[Nb.m_Index[k]] = 1;
                        Used = std::move(Next);
                    }
                std::vector<std::uint32_t> Map(nAll, 0xFFFFFFFFu);
                std::uint32_t nKeep = 0;
                for (std::uint32_t s = 0; s < nAll; ++s) if (Used[s]) Map[s] = nKeep++;
                if (nKeep < nAll)
                {
                    const auto Keep = [&](auto& V) { std::size_t j = 0; for (std::uint32_t s = 0; s < nAll && s < V.size(); ++s) if (Used[s]) V[j++] = V[s]; V.resize(j); };
                    Keep(C.m_Levels[0]); Keep(C.m_SurfelMaterial); Keep(C.m_ParentOf);
                    std::vector<std::uint32_t> KeptLocal; std::vector<chunk_ghost> KeptGhosts;
                    for (std::uint32_t s = 0; s < nAll; ++s) if (Used[s]) { KeptLocal.push_back(Local[s]); if (s >= nOwned) KeptGhosts.push_back(Ghosts[s - nOwned]); }
                    Local = std::move(KeptLocal); Ghosts = std::move(KeptGhosts);
                    for (auto& V : C.m_VertSurfels) for (auto& sv : V) if (sv >= 0) sv = int(Map[std::size_t(sv)]);
                    for (const auto& E : C.m_ReadTable)
                    {
                        if (E[1] == 0xFFFFFFFFu) continue;
                        for (std::uint32_t p = 0; p < E[3]; ++p)
                        {
                            const std::uint32_t F = C.m_ReadList[E[2] + 4 * p + 2], n = C.m_ReadList[E[2] + 4 * p + 3] & 0xFFFFu;
                            for (std::uint32_t k = 0; k < n; ++k) { auto& w = C.m_ReadList[F + k]; w = (w & 0xFFFF0000u) | Map[w & 0xFFFFu]; }
                        }
                    }
                    C.RefreshPackedReadGrid();      // (the lists changed after their sort: the packed form again)
                    C.BuildOrigins();
                    NeighborsFromWhole(Whole, C, Local, ParentWhole);   // (the owned surfels' neighbours are all kept)
                }
            }
            Step("compact");
            if (!Trace.empty()) { std::fprintf(stderr, "CHUNKTRACE %zu: owned %zu, local %zu:%s\n", Out.m_Chunks.size(), nOwned, Local.size(), Trace.c_str()); std::fflush(stderr); }
            C.m_pOccluder = nullptr;
            Out.m_Chunks.push_back(std::move(C));
            if (pReport && pReport->m_OnProgress) pReport->m_OnProgress(float(g + 1) / float(Groups.size()));
            Out.m_VertWhole.push_back(std::move(VertWhole));
            Out.m_nOwned.push_back(std::uint32_t(nOwned));
            Out.m_Ghosts.push_back(std::move(Ghosts));
            Rep.m_LocalMax = std::max(Rep.m_LocalMax, Local.size());
            if (Local.size() > chunk_local_cap_v) Rep.m_bFits = false;      // (a single cluster over the cap: the plan's repair, not done yet)
        }
        // every surfel's home, then the ghosts' owners
        Out.m_Home.assign(nS, { 0xFFFFFFFFu, 0xFFFFFFFFu });
        {
            std::vector<std::uint32_t> Next(Out.m_Chunks.size(), 0);
            for (std::size_t s = 0; s < nS; ++s)
            {
                const std::uint32_t c = Out.m_ClusterChunk[std::size_t(SurfCluster[s])];
                Out.m_Home[s] = { c, Next[c]++ };
            }
        }
        for (auto& Gs : Out.m_Ghosts) for (auto& Gh : Gs) Gh = { Out.m_Home[Gh.m_Local][0], Out.m_Home[Gh.m_Local][1] };
        Rep.m_BuildMs = std::chrono::duration<double, std::milli>(clock::now() - T1).count();
        Rep.m_nChunks = int(Out.m_Chunks.size());
        Rep.m_OwnedMin = Out.m_nOwned.empty() ? 0 : *std::min_element(Out.m_nOwned.begin(), Out.m_nOwned.end());
        Rep.m_OwnedMax = Out.m_nOwned.empty() ? 0 : *std::max_element(Out.m_nOwned.begin(), Out.m_nOwned.end());
        for (const auto& Gs : Out.m_Ghosts) { Rep.m_GhostsTotal += Gs.size(); Rep.m_GhostsMax = std::max(Rep.m_GhostsMax, Gs.size()); }
        if (pReport) *pReport = Rep;
        return Rep.m_bFits;
    }

    //---------------------------------------------------------------------------------------------------------------------------------
    // Sections in a resource (xgeom_static version 7): a geometry's chunks flattened into plain arrays (the compiler), and back into one
    // surfel_mesh a section (the renderer, from the geometry's whole mesh: SectionGeometry the same way)

    struct section_record
    {
        std::uint32_t m_iFine = 0, m_nFine = 0, m_nOwned = 0;     // its fine surfels (owned, then ghosts) in the fine array
        std::uint32_t m_iParent = 0, m_nParents = 0;               // its parents in the parent array
        std::uint32_t m_iDerived = 0, m_nDerived = 0;              // its packed structures (surfel_mesh::PackDerived) in the derived words
        std::uint32_t m_iGhost = 0, m_nGhosts = 0;                 // its ghosts' owners (m_nFine - m_nOwned of them) in the ghost array
    };

    struct sections_flat
    {
        std::vector<section_record>                     m_Sections;
        std::vector<surfel>                             m_Fine, m_Parents;
        std::vector<std::uint32_t>                      m_ParentOf;         // per fine surfel: its section-local parent
        std::vector<surfel_mesh::surfel_material>       m_Materials;        // per fine surfel
        std::vector<std::uint16_t>                      m_VertSurfels;      // per whole vertex 4: its section's local surfels (0xFFFF none)
        std::vector<std::uint32_t>                      m_Derived;
        std::vector<chunk_ghost>                        m_Ghosts;
        std::vector<std::uint16_t>                      m_ClusterSection;   // per cluster
        std::size_t                                     m_nSharedVerts = 0; // vertices in two sections' triangles (their vertex surfels: the last section's).
                                                                            // xgeom_static has none: a cluster's vertices are its own
    };

    // What a resource keeps of them (spans into its arrays: the renderer's load, the tests)
    struct sections_view
    {
        std::span<const section_record>                 m_Sections;
        std::span<const surfel>                         m_Fine, m_Parents;
        std::span<const std::uint32_t>                  m_ParentOf;
        std::span<const surfel_mesh::surfel_material>   m_Materials;
        std::span<const std::uint16_t>                  m_VertSurfels;
        std::span<const std::uint32_t>                  m_Derived;
        std::span<const chunk_ghost>                    m_Ghosts;
        std::span<const std::uint16_t>                  m_ClusterSection;
    };

    inline sections_flat FlattenSections(const chunked_mesh& Ch, std::size_t nWholeVerts) noexcept
    {
        sections_flat F;
        F.m_ClusterSection = Ch.m_ClusterChunk;
        F.m_VertSurfels.assign(4 * nWholeVerts, 0xFFFF);
        std::vector<std::uint8_t> Seen(nWholeVerts, 0);
        for (std::size_t c = 0; c < Ch.m_Chunks.size(); ++c)
        {
            const auto& C = Ch.m_Chunks[c];
            section_record R;
            R.m_iFine = std::uint32_t(F.m_Fine.size()); R.m_nFine = std::uint32_t(C.m_Levels[0].size()); R.m_nOwned = Ch.m_nOwned[c];
            R.m_iParent = std::uint32_t(F.m_Parents.size()); R.m_nParents = std::uint32_t(C.m_Levels[1].size());
            F.m_Fine.insert(F.m_Fine.end(), C.m_Levels[0].begin(), C.m_Levels[0].end());
            F.m_Parents.insert(F.m_Parents.end(), C.m_Levels[1].begin(), C.m_Levels[1].end());
            for (std::size_t s = 0; s < C.m_Levels[0].size(); ++s) F.m_ParentOf.push_back(s < C.m_ParentOf.size() ? std::uint32_t(C.m_ParentOf[s]) : 0u);
            for (std::size_t s = 0; s < C.m_Levels[0].size(); ++s) F.m_Materials.push_back(s < C.m_SurfelMaterial.size() ? C.m_SurfelMaterial[s] : surfel_mesh::surfel_material{});
            const auto D = C.PackDerived();
            R.m_iDerived = std::uint32_t(F.m_Derived.size()); R.m_nDerived = std::uint32_t(D.size());
            F.m_Derived.insert(F.m_Derived.end(), D.begin(), D.end());
            R.m_iGhost = std::uint32_t(F.m_Ghosts.size()); R.m_nGhosts = std::uint32_t(Ch.m_Ghosts[c].size());
            F.m_Ghosts.insert(F.m_Ghosts.end(), Ch.m_Ghosts[c].begin(), Ch.m_Ghosts[c].end());
            const auto& VW = Ch.m_VertWhole[c];
            for (const std::uint32_t w : VW) F.m_nSharedVerts += Seen[w]++ == 1;
            for (std::size_t v = 0; v < VW.size() && v < C.m_VertSurfels.size(); ++v)
                for (int k = 0; k < 4; ++k)
                {
                    const int s = C.m_VertSurfels[v][std::size_t(k)];
                    F.m_VertSurfels[4 * std::size_t(VW[v]) + std::size_t(k)] = s < 0 ? std::uint16_t(0xFFFF) : std::uint16_t(s);
                }
            F.m_Sections.push_back(R);
        }
        return F;
    }

    inline sections_view ViewOf(const sections_flat& F) noexcept
    {
        return { F.m_Sections, F.m_Fine, F.m_Parents, F.m_ParentOf, F.m_Materials, F.m_VertSurfels, F.m_Derived, F.m_Ghosts, F.m_ClusterSection };
    }

    // Section k of Whole (its geometry: FromXGeomStatic; TriCluster: per triangle its cluster) as its own surfel_mesh: its geometry, its surfels
    // (owned, then ghosts), parents, vertex surfels, materials, rays and packed structures. False: the arrays do not fit this geometry
    // pVMap: the whole mesh's vertex -> the section's (-1: not in it): a cluster's draw finds its section's vertex at its own + VMap[first] - first
    inline bool LoadSection(const surfel_mesh& Whole, std::span<const int> TriCluster, const sections_view& V, std::size_t k, surfel_mesh& C, std::vector<int>* pVMap = nullptr) noexcept
    {
        if (k >= V.m_Sections.size() || V.m_VertSurfels.size() != 4 * Whole.m_Pos.size()) return false;
        const auto& R = V.m_Sections[k];
        if (std::size_t(R.m_iFine) + R.m_nFine > V.m_Fine.size() || std::size_t(R.m_iParent) + R.m_nParents > V.m_Parents.size()
            || std::size_t(R.m_iDerived) + R.m_nDerived > V.m_Derived.size() || std::size_t(R.m_iFine) + R.m_nFine > V.m_ParentOf.size()) return false;
        std::vector<char> InG(V.m_ClusterSection.size(), 0);
        for (std::size_t c = 0; c < V.m_ClusterSection.size(); ++c) InG[c] = V.m_ClusterSection[c] == k;
        for (const int c : TriCluster) if (c < 0 || std::size_t(c) >= InG.size()) return false;
        C = {};
        std::vector<int> VMap; std::vector<std::uint32_t> VertWhole;
        SectionGeometry(Whole, TriCluster, InG, C, VMap, VertWhole);
        C.m_bLeakProtection = Whole.m_bLeakProtection; C.m_bBounce = Whole.m_bBounce;
        C.m_Levels[0].assign(V.m_Fine.begin() + R.m_iFine, V.m_Fine.begin() + R.m_iFine + R.m_nFine);
        C.m_Levels[1].assign(V.m_Parents.begin() + R.m_iParent, V.m_Parents.begin() + R.m_iParent + R.m_nParents);
        C.m_ParentOf.assign(V.m_ParentOf.begin() + R.m_iFine, V.m_ParentOf.begin() + R.m_iFine + R.m_nFine);
        if (std::size_t(R.m_iFine) + R.m_nFine <= V.m_Materials.size()) C.m_SurfelMaterial.assign(V.m_Materials.begin() + R.m_iFine, V.m_Materials.begin() + R.m_iFine + R.m_nFine);
        C.m_VertSurfels.resize(C.m_Pos.size());
        for (std::size_t v = 0; v < VertWhole.size(); ++v)
            for (int j = 0; j < 4; ++j)
            {
                const std::uint16_t s = V.m_VertSurfels[4 * std::size_t(VertWhole[v]) + std::size_t(j)];
                C.m_VertSurfels[v][std::size_t(j)] = s == 0xFFFF ? -1 : int(s);
            }
        C.BuildRays();
        if (!C.UnpackDerived(V.m_Derived.subspan(R.m_iDerived, R.m_nDerived)))
        {
            std::printf("GI section %zu: its packed structures (%u words) do not fit it (%zu surfels, %zu vertices): made again\n", k, R.m_nDerived, C.m_Levels[0].size(), C.m_Pos.size()); std::fflush(stdout);
            C.BuildNeighbors();
        }
        if (std::getenv("XLION_GI_REGRID")) C.BuildReadGrid();     // measurement: the grid rebuilt at load (as SurfelsFromXGeomStatic), no recompile
        if (pVMap) *pVMap = std::move(VMap);
        return true;
    }

    // Per triangle (FromXGeomStatic's order: cluster by cluster, its indices in threes) its cluster
    template <class T_GEOM> std::vector<int> TriClustersOf(const T_GEOM& Geom) noexcept
    {
        std::vector<int> Out;
        const auto Clusters = Geom.getClusters();
        const std::size_t nIdx = Geom.getIndices().size();
        for (std::size_t c = 0; c < Clusters.size(); ++c)
            for (std::uint32_t i = Clusters[c].m_iIndex; i + 2 < Clusters[c].m_iIndex + Clusters[c].m_nIndices && i + 2 < nIdx; i += 3) Out.push_back(int(c));
        return Out;
    }

    // A compiled xgeom_static's GI sections (version 7) as section meshes: Whole is its whole mesh (FromXGeomStatic, no surfels needed). Per
    // section its surfel_mesh, owned count and ghosts' owners; per cluster its section and the delta from its vertices to its section's. False:
    // the geometry has no sections, or they do not fit it
    struct loaded_section { surfel_mesh m_Surfels; std::uint32_t m_nOwned = 0; std::vector<chunk_ghost> m_Ghosts; };
    template <class T_GEOM> bool SectionsFromXGeomStatic(const surfel_mesh& Whole, const T_GEOM& Geom, std::vector<loaded_section>& Out,
                                                         std::vector<std::uint16_t>& ClusterSection, std::vector<int>& ClusterVertexDelta) noexcept
    {
        const auto Secs = Geom.getGISections();
        if (Secs.empty() || Geom.getGIClusterSections().size() != Geom.getClusters().size()) return false;
        const auto ToSurfel = [](const auto& S) { return surfel{ v3{ S.m_Center.m_X, S.m_Center.m_Y, S.m_Center.m_Z }, v3{ S.m_Normal.m_X, S.m_Normal.m_Y, S.m_Normal.m_Z }, S.m_Radius }; };
        sections_flat F;
        for (const auto& S : Secs) F.m_Sections.push_back({ S.m_iFine, S.m_nFine, S.m_nOwned, S.m_iParent, S.m_nParents, S.m_iDerived, S.m_nDerived, S.m_iGhost, S.m_nGhosts });
        for (const auto& S : Geom.getGISurfels()) { F.m_Fine.push_back(ToSurfel(S)); F.m_ParentOf.push_back(S.m_iParent); }
        for (const auto& S : Geom.getGIParents()) F.m_Parents.push_back(ToSurfel(S));
        for (const auto& M : Geom.getGISurfelMaterials()) F.m_Materials.push_back({ M.m_U, M.m_V, M.m_UVRadius, M.m_iMaterial });
        F.m_VertSurfels.assign(Geom.getGIVertexSurfels().begin(), Geom.getGIVertexSurfels().end());
        F.m_Derived.assign(Geom.getGIDerived().begin(), Geom.getGIDerived().end());
        for (const auto& G : Geom.getGIGhosts()) F.m_Ghosts.push_back({ G.m_Section, G.m_Local });
        F.m_ClusterSection.assign(Geom.getGIClusterSections().begin(), Geom.getGIClusterSections().end());
        const auto TriCluster = TriClustersOf(Geom);
        if (TriCluster.size() != Whole.m_Tris.size()) return false;
        const auto V = ViewOf(F);
        Out.assign(F.m_Sections.size(), {});
        ClusterSection = F.m_ClusterSection;
        ClusterVertexDelta.assign(Geom.getClusters().size(), 0);
        for (std::size_t k = 0; k < F.m_Sections.size(); ++k)
        {
            std::vector<int> VMap;
            if (!LoadSection(Whole, TriCluster, V, k, Out[k].m_Surfels, &VMap)) return false;
            Out[k].m_Surfels.m_bBounce = Geom.m_GIMode != decltype(Geom.m_GIMode)::RECEIVE;
            Out[k].m_nOwned = F.m_Sections[k].m_nOwned;
            Out[k].m_Ghosts.assign(F.m_Ghosts.begin() + F.m_Sections[k].m_iGhost, F.m_Ghosts.begin() + F.m_Sections[k].m_iGhost + F.m_Sections[k].m_nGhosts);
            for (std::size_t c = 0; c < Geom.getClusters().size(); ++c)
            {
                if (ClusterSection[c] != k) continue;
                const std::uint32_t First = Geom.getClusters()[c].m_iVertex;
                if (First >= VMap.size() || VMap[First] < 0) continue;      // (a cluster with no triangles)
                ClusterVertexDelta[c] = VMap[First] - int(First);
            }
        }
        return true;
    }
}
