#version 450
// The voxel rays alone (xlionrender_gi_ref.glsl VoxMain): with the ground truth in it, it compiled to 471 KB against 205 and took about a third fewer rays in 2 ms
#define GI_VOX_ONLY
#include "xlionrender_gi_ref.glsl"
