#version 450
#extension GL_ARB_separate_shader_objects : enable

// The shared cube/sphere/cylinder pipeline (xlionrender_primitive_vert.glsl) applies Transform.Scale
// as one flat matrix baked into L2C, which stretches EVERY vertex - including the hemisphere caps -
// by the same per-axis factor. For a capsule that turns the caps into pointed cones the moment
// Scale.y != Scale.x/z (direct user report: caps look "too sharp"/the whole thing "looks more like a
// sphere... specially when you scale it in Y"). A true capsule is a line SEGMENT swept by a sphere of
// RADIUS r: only the segment's length should follow Scale.y - the sphere sweep radius stays uniform
// regardless of Y. Scale.x and Scale.z still apply independently to the sweep, same as every other
// shape (direct user report on a first version of this shader that averaged them together: "when I
// scale in X the entire capsule scales in the X/Z plane"). This shader reconstructs both in local
// space before L2C.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(std140, push_constant) uniform PushConstants
{
    mat4 L2C;   // World rotation+translation only (Scale stripped - see renderer::DrawItem)
    vec4 Color;
    vec4 RadiusHeightScale; // x local radius (xprim_geom::capsule::Generate's own Radius, see Init); y Transform.Scale.x; z Transform.Scale.z; w Transform.Scale.y
} uniforms;

layout(location = 0) out vec3 outNormal;

void main()
{
    // inNormal is already the unit outward radial direction for every vertex the generator emits
    // (both the hemisphere caps and the cylinder band - see xprim_geom_capsule.h), so this holds
    // exactly for the whole mesh: inPosition = spinePoint + LocalRadius * inNormal, where spinePoint
    // is (0, +-halfCylinderLength, 0) for a cap vertex or (0, y, 0) for a cylinder-band vertex.
    float localRadius = uniforms.RadiusHeightScale.x;
    float scaleX       = uniforms.RadiusHeightScale.y;
    float scaleZ       = uniforms.RadiusHeightScale.z;
    float scaleY       = uniforms.RadiusHeightScale.w;

    vec3 spinePoint  = inPosition - localRadius * inNormal;
    vec3 scaledSpine = vec3(0.0, spinePoint.y * scaleY, 0.0);

    // Sweep radius per axis (X and Z independent, matching every other primitive's Scale behavior) -
    // except the cap's OWN "vertical" protrusion (inNormal.y, nonzero only near/at the poles), which
    // must keep following the radius concept, not Scale.y, or the caps go pointy again the moment Y is
    // stretched. average(scaleX,scaleZ) is the natural stand-in for "radius" there once X and Z differ.
    vec3 radialOffset = localRadius * vec3(inNormal.x * scaleX, inNormal.y * (0.5 * (scaleX + scaleZ)), inNormal.z * scaleZ);

    gl_Position = uniforms.L2C * vec4(scaledSpine + radialOffset, 1.0);
    outNormal   = inNormal;
}
