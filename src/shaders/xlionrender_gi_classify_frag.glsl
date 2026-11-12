#version 450
// The GI gather's classification (xlionrender_gi_gather_common.glsl): each half-size pixel's GI buffer as its depth - (index + 1) / 65536;
// no GI surface there: the cleared far depth, no pass runs on it
#include "xlionrender_surfel_gi.glsl"
#include "xlionrender_gi_gather_common.glsl"

void main()
{
    const gather_sample S = GatherSample(ivec2(gl_FragCoord.xy));
    if (!S.bHit) discard;
    gl_FragDepth = float((GITable[push.TableBase + 39u] != 0u ? 0u : S.Entry.z) + 1u) / 65536.0;      // (word 39, XLION_GI_ONE_PASS: every pixel the first pass's - the voxels' read takes no GI buffer)
}
