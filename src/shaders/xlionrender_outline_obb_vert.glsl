#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal; // Existing vertex layout; not used for expansion.

layout(std140, push_constant) uniform PushConstants
{
    mat4 L2C;
    vec4 ViewportAndRadius; // xy viewport px; z radius px; w max fractional expansion
    vec4 Color;
    vec4 Bounds;            // xyz local box center; w cage growth in local units
    vec4 Axis;              // xyz local box half-extents; w near-parallel rejection angle (degrees)
} uniforms;

const float kMinPixelMove = 0.5;
const float kMinW = 1e-5;

void useRadialFallback(vec4 clip, vec2 viewportPx, float radiusPx, float maxScale)
{
    vec4 pivot = uniforms.L2C * vec4(uniforms.Bounds.xyz, 1.0);
    vec2 pivotNdc = pivot.w > kMinW ? pivot.xy / pivot.w : vec2(0.0);
    vec2 delta = clip.xy - pivotNdc * clip.w;
    float distanceTimesW = length(delta * (0.5 * viewportPx));
    float factor = clip.w <= 0.0 ? maxScale
                 : min(maxScale, radiusPx * clip.w / max(distanceTimesW, 1e-5));
    gl_Position.xy += delta * factor;
}

void main()
{
    vec4 clip = uniforms.L2C * vec4(inPosition, 1.0);
    gl_Position = clip;

    vec2 viewportPx = uniforms.ViewportAndRadius.xy;
    float radiusPx = max(uniforms.ViewportAndRadius.z, 0.0);
    float maxScale = max(uniforms.ViewportAndRadius.w, 0.0);
    if (viewportPx.x <= 0.0 || viewportPx.y <= 0.0 ||
        radiusPx <= 0.0 || maxScale <= 0.0)
        return;

    vec3 localCenter = uniforms.Bounds.xyz;
    vec3 localHalf = max(uniforms.Axis.xyz, vec3(0.0));
    float growth = max(uniforms.Bounds.w, 0.0);
    vec3 cageHalf = localHalf + vec3(growth);
    if (clip.w <= 0.0)
    {
        useRadialFallback(clip, viewportPx, radiusPx, maxScale);
        return;
    }

    // Camera center in local coordinates for a perspective L2C matrix.
    // At the eye: clip.x = clip.y = clip.w = 0, so clip (0,0,1,0)
    // unprojects to the local homogeneous camera position.
    vec4 localEyeH = inverse(uniforms.L2C) * vec4(0.0, 0.0, 1.0, 0.0);
    vec3 towardEye = vec3(0.0);
    bool haveEye = abs(localEyeH.w) > 1e-8;
    if (haveEye)
    {
        vec3 eyeDelta = localEyeH.xyz / localEyeH.w - inPosition;
        float eyeLengthSq = dot(eyeDelta, eyeDelta);
        if (eyeLengthSq > 1e-12)
            towardEye = eyeDelta * inversesqrt(eyeLengthSq);
        else
            haveEye = false;
    }

    vec3 cageMin = localCenter - cageHalf;
    vec3 cageMax = localCenter + cageHalf;
    vec2 vertexPx = (clip.xy / clip.w * 0.5 + 0.5) * viewportPx;
    float facingLimit = cos(radians(clamp(uniforms.Axis.w, 0.0, 89.0)));

    bool accepted[6];
    vec2 directions[6];
    float distances[6];
    float closestDistance = 1e30;
    int closestFace = -1;

    for (int i = 0; i < 6; ++i)
    {
        int faceAxis = i / 2;
        float side = (i % 2) == 0 ? 1.0 : -1.0;
        vec3 nearest = clamp(inPosition, cageMin, cageMax);
        vec3 faceNormal = vec3(0.0);
        if (faceAxis == 0)
        {
            nearest.x = localCenter.x + side * cageHalf.x;
            faceNormal.x = side;
        }
        else if (faceAxis == 1)
        {
            nearest.y = localCenter.y + side * cageHalf.y;
            faceNormal.y = side;
        }
        else
        {
            nearest.z = localCenter.z + side * cageHalf.z;
            faceNormal.z = side;
        }

        vec4 target = uniforms.L2C * vec4(nearest, 1.0);
        bool valid = target.w > kMinW && target.z >= 0.0 && target.z <= target.w;
        if (haveEye)
            valid = valid && abs(dot(faceNormal, towardEye)) <= facingLimit;

        vec2 pixelVector = vec2(0.0);
        if (valid)
        {
            vec2 targetPx = (target.xy / target.w * 0.5 + 0.5) * viewportPx;
            pixelVector = targetPx - vertexPx;
            valid = dot(pixelVector, pixelVector) >= kMinPixelMove * kMinPixelMove;
        }
        accepted[i] = valid;
        directions[i] = valid ? normalize(pixelVector) : vec2(0.0);
        distances[i] = length(nearest - inPosition);
        if (valid && distances[i] < closestDistance)
        {
            closestDistance = distances[i];
            closestFace = i;
        }
    }

    if (closestFace < 0)
    {
        useRadialFallback(clip, viewportPx, radiusPx, maxScale);
        return;
    }

    vec2 direction = directions[closestFace];
    vec2 blended = vec2(0.0);
    float blendWidth = max(0.75 * growth, 0.025);
    for (int i = 0; i < 6; ++i)
    {
        if (!accepted[i]) continue;
        float t = (distances[i] - closestDistance) / blendWidth;
        blended += directions[i] * exp(-t * t);
    }
    float blendedLengthSq = dot(blended, blended);
    if (blendedLengthSq > 1e-10)
        direction = blended * inversesqrt(blendedLengthSq);

    // Find the projected size of the ORIGINAL local box along the chosen
    // direction. Clip the box edges against Vulkan's z=0 near plane first.
    vec4 corners[8];
    for (int i = 0; i < 8; ++i)
    {
        vec3 p = localCenter + localHalf * vec3(
            (i & 1) != 0 ? 1.0 : -1.0,
            (i & 2) != 0 ? 1.0 : -1.0,
            (i & 4) != 0 ? 1.0 : -1.0);
        corners[i] = uniforms.L2C * vec4(p, 1.0);
    }
    const ivec2 edges[12] = ivec2[12](
        ivec2(0,1), ivec2(0,2), ivec2(0,4), ivec2(1,3),
        ivec2(1,5), ivec2(2,3), ivec2(2,6), ivec2(3,7),
        ivec2(4,5), ivec2(4,6), ivec2(5,7), ivec2(6,7));

    float low = 1e30, high = -1e30;
    int pointCount = 0;
    for (int i = 0; i < 8; ++i)
    {
        vec4 p = corners[i];
        if (p.w <= kMinW || p.z < 0.0) continue;
        vec2 px = (p.xy / p.w * 0.5 + 0.5) * viewportPx;
        float projected = dot(px, direction);
        low = min(low, projected);
        high = max(high, projected);
        ++pointCount;
    }
    for (int i = 0; i < 12; ++i)
    {
        vec4 a = corners[edges[i].x], b = corners[edges[i].y];
        if ((a.z >= 0.0) == (b.z >= 0.0)) continue;
        float t = a.z / (a.z - b.z);
        vec4 p = mix(a, b, t);
        if (p.w <= kMinW) continue;
        vec2 px = (p.xy / p.w * 0.5 + 0.5) * viewportPx;
        float projected = dot(px, direction);
        low = min(low, projected);
        high = max(high, projected);
        ++pointCount;
    }
    if (pointCount == 0)
    {
        useRadialFallback(clip, viewportPx, radiusPx, maxScale);
        return;
    }

    float projectedHalfSize = max(0.5 * (high - low), 1e-5);
    float movementPx = min(radiusPx, maxScale * projectedHalfSize);
    gl_Position.xy += direction * movementPx * (2.0 / viewportPx) * clip.w;
}
