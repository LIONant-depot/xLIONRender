#version 450
#extension GL_EXT_nonuniform_qualifier : require
// A texture-table material's bindless HDR variant: the heaps by the ids in its block (xlionrender_test_table_body.glsl)
#define XGPU_BINDLESS
#include "xlionrender_test_table_body.glsl"
