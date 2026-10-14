#version 450
// The GI's material capture (features/gi: surfel colors): each surfel a 2 x 2 pixel quad of the capture target, drawn with its slot's
// real material fragment shader in capture mode (mb_standard_pbr.frag: GIDebug 3 writes the albedo x (1 - metal), or the emission).
// The quad's UVs span the surfel's footprint (its center +- its UV radius): the texture's mips average the material over it.
//  in_Pos: the corner, clip space (the CPU places each surfel's block)
//  Capture[surfel]: { u, v, UV radius, - }, { normal xyz, - } (mesh space)

#include "mb_varying_definition_full.glsl"

layout(location = 0) in vec2 in_Pos;

layout(std430, set = 1, binding = 0) readonly buffer CaptureBuffer { vec4 Capture[]; };

layout(push_constant) uniform PushConstants
{
    uint    firstSurfel;   // the batch's first surfel in Capture
    uint    instanceIndex; // the draw's slot in the instance buffer (the constant material reads it)
    uint    giOffset;      // unused
    uint    giLevel;       // the capture's flags: (3 << 9) | 0x800 for the emission (gi_capture_emission_v)
} push;

layout(location = 0) out VaryingFull Out;
layout(location = 8) flat out uint OutInstance;
layout(location = 9) out vec4 OutGI;
layout(location = 10) out vec3 OutGIPos;
layout(location = 11) out vec3 OutGINormal;
layout(location = 12) flat out uvec2 OutGIRef;
layout(location = 13) out vec4 OutGITangent;

void main()
{
    const uint  s      = push.firstSurfel + uint(gl_VertexIndex) / 4u;
    const uint  Corner = uint(gl_VertexIndex) % 4u;
    const vec4  M      = Capture[2u * s];
    const vec3  N      = normalize(Capture[2u * s + 1u].xyz);
    const vec2  Sign   = vec2((Corner == 1u || Corner == 2u) ? 1.0 : -1.0, Corner >= 2u ? 1.0 : -1.0);
    const vec3  T      = normalize(cross(abs(N.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0), N));

    gl_Position         = vec4(in_Pos, 0.5, 1.0);
    Out.wSpacePosition  = vec4(0.0, 0.0, 0.0, 1.0);
    Out.ShadowPosition  = vec4(0.5, 0.5, 2.0, 1.0);
    Out.VertColor       = vec4(1.0);
    Out.Tangent         = vec4(T, 1.0);
    Out.Normal          = N;
    Out.UV              = M.xy + Sign * M.z;
    OutInstance         = push.instanceIndex;
    OutGI               = vec4(0.0);
    OutGIPos            = vec3(0.0);
    OutGINormal         = N;
    OutGIRef            = uvec2(0u, push.giLevel);
    OutGITangent        = vec4(T, 1.0);
}
