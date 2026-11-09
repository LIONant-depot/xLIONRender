// The render graph's barrier decisions (R1a, rg::graph::Sync), on the CPU: the post chain's pattern - TAA writes its target, the bloom
// levels each read the one before, the exposure reads the TAA and is read by every bloom level - gets a barrier exactly where a scope
// reads what an earlier one wrote (read after write), or writes what an earlier one read or wrote (write after read / write), since the
// last barrier; reads of the same thing need none. (The Vulkan sync validation did not flag the post chain run with no barriers at all,
// 2026-11-09: these decisions are what keeps it correct.)
#include "xlionrender_render_graph.h"
#include <cstdio>
#include <cstdlib>

namespace rg = xlionrender::rg;

#define CHECK(c, what) do { if( !(c) ) { std::printf("FAIL: %s\n", what); std::exit(1); } } while(0)

int main()
{
    rg::graph G;
    const int HDR = G.Resource("HDR", rg::lifetime::FRAME), TAA = G.Resource("TAA", rg::lifetime::FRAME), Expo = G.Resource("exposure", rg::lifetime::FRAME);
    const int B0 = G.Resource("bloom0", rg::lifetime::FRAME), B1 = G.Resource("bloom1", rg::lifetime::FRAME);
    const auto Scope = [&](const char* pName, std::initializer_list<int> Reads, std::initializer_list<int> Writes)
    {
        const int S = G.Begin(pName, "");
        for( int r : Reads )  G.Read(S, r, rg::usage::SAMPLED);
        for( int w : Writes ) G.Write(S, w, rg::usage::STORAGE, rg::policy::FULL_OVERWRITE);
        return G.Sync(S);
    };
    G.BeginFrame();
    { const int S = G.Begin("scene", ""); G.Write(S, HDR, rg::usage::COLOR_ATTACHMENT, rg::policy::FULL_OVERWRITE); G.Write(S, Expo, rg::usage::STORAGE, rg::policy::FULL_OVERWRITE); }
    G.SyncStart();      // (the section's own barrier: what came before is visible)
    CHECK(!Scope("TAA", { HDR }, { TAA }),                   "TAA: reads only what the section's start made visible");
    CHECK( Scope("bloom down 0", { TAA, Expo }, { B0 }),     "bloom down 0 reads the TAA's write: read after write");
    CHECK( Scope("bloom down 1", { B0, Expo }, { B1 }),      "bloom down 1 reads bloom 0: read after write");
    CHECK( Scope("exposure", { TAA }, { Expo }),             "the exposure overwrites what the bloom levels read: write after read");
    CHECK( Scope("bloom up 0", { B1, B0, Expo }, { B0 }),    "bloom up reads the exposure just written: read after write");
    CHECK(!Scope("reader A", { HDR }, {}),                   "a read of something no scope wrote since the barrier: none");
    CHECK(!Scope("reader B", { HDR }, {}),                   "two reads of the same thing: none");
    CHECK( Scope("writer", {}, { HDR }),                     "a write of what was read: write after read");
    CHECK( Scope("writer again", {}, { HDR }),               "a write of what was written: write after write");
    std::printf("render graph barriers: ok\n");
    return 0;
}
