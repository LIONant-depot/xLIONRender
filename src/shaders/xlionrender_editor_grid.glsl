// The editors' ground grid as a surface's albedo (scene::material::m_bEditorGrid, xlionrender_forward_standard_frag.glsl): the pattern of
// xeditor_tools' grid (xeditor_tools_grid_frag.glsl: the lines of two scales blended by the distance, the X and Z axes with their arrows)
// in world XZ (metres) - lit by PBRLighting, so the ground receives the same sun shadow (virtual pages or clipmap), lights and GI as the
// geometry on it, instead of a shadow map of its own.
#ifndef XLIONRENDER_EDITOR_GRID_GLSL
#define XLIONRENDER_EDITOR_GRID_GLSL
const float kGridBias        = 1.0;
const float kGridMajorDiv    = 10.0;
const vec4  kGridBaseColor       = vec4(0.61, 0.63, 0.7, 1.0);
const vec4  kGridLineColor       = vec4(0.31, 0.33, 0.4, 1.0);
const float kGridLineWidth       = 1.00;
const float kGridMajorLineWidth  = 2.5;
const float kGridFadePower       = 0.7;
const float kGridArrowSpacing    = 10.0;
const float kGridArrowStemLength = 4.0;
const float kGridArrowHeadLength = 1.0;
const float kGridArrowStemWidth  = 0.5;
const float kGridArrowHeadWidth  = 1.5;
const vec4  kGridXAxisColor      = vec4(1.0, 0.0, 0.0, 1.0);
const vec4  kGridZAxisColor      = vec4(0.0, 0.0, 1.0, 1.0);
const float kGridTanHalfFov     = 0.4142;       // the grid's line scale follows the distance as xeditor_tools' does at its 45 degree field of view

float GridSdBox(in vec2 p, in vec2 a, in vec2 b, float th)
{
    const float l = length(b - a);
    const vec2  d = (b - a) / l;
    vec2 q = p - (a + b) * 0.5;
    q = mat2(d.x, -d.y, d.y, d.x) * q;
    q = abs(q) - vec2(l, th) * 0.5;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}

float GridSdTriangle(in vec2 p, in vec2 q)
{
    p.x = abs(p.x);
    const vec2  a = p - q * clamp(dot(p, q) / dot(q, q), 0.0, 1.0);
    const vec2  b = p - q * vec2(clamp(p.x / q.x, 0.0, 1.0), 1.0);
    const float s = -sign(q.y);
    const vec2  d = min(vec2(dot(a, a), s * (p.x * q.y - p.y * q.x)), vec2(dot(b, b), s * (p.y - q.y)));
    return -sqrt(d.x) * sign(d.y);
}

float GridSdArrow(in vec2 p)
{
    const float d1 = GridSdBox(p, vec2(kGridArrowStemLength, 0.0), vec2(0.0, 0.0), kGridArrowStemWidth);
    const vec2  pr = mat2(0.0, 1.0, -1.0, 0.0) * p;
    const float d2 = GridSdTriangle(pr + vec2(0, kGridArrowHeadLength), vec2(kGridArrowHeadWidth / 2.0, kGridArrowHeadLength));
    return min(d1, d2);
}

// UV: world XZ; ToFrag: the camera to the fragment (world)
vec3 GridAlbedo(in const vec2 UV, in const vec3 ToFrag)
{
    const float Dist      = max(length(ToFrag), 1e-4);
    const float CosTheta  = pow(abs(ToFrag.y / Dist), kGridFadePower);
    const float LogLength = log(Dist * kGridTanHalfFov) / log(kGridMajorDiv) - kGridBias;
    const float LogA      = floor(LogLength);
    const float Blend     = fract(LogLength);
    const float ScaleA = pow(kGridMajorDiv, LogA), ScaleB = ScaleA * kGridMajorDiv, ScaleC = ScaleB * kGridMajorDiv;
    const vec2  GA = 1.0 - abs(fract(UV / ScaleA) * 2.0 - 1.0);
    const vec2  GB = 1.0 - abs(fract(UV / ScaleB) * 2.0 - 1.0);
    const vec2  GC = 1.0 - abs(fract(UV / ScaleC) * 2.0 - 1.0);
    const float WA = kGridLineWidth * (1.0 - Blend), WB = mix(kGridMajorLineWidth, kGridLineWidth, Blend), WC = kGridMajorLineWidth * Blend;
    const vec2  dX = dFdx(UV), dY = dFdy(UV);
    const vec2  Len = vec2(length(vec2(dX.x, dY.x)), length(vec2(dX.y, dY.y)));
    const vec2  LA = 1.0 - smoothstep((WA - 1.5) * Len / ScaleA, (WA + 1.5) * Len / ScaleA, GA);
    const vec2  LB = 1.0 - smoothstep((WB - 1.5) * Len / ScaleB, (WB + 1.5) * Len / ScaleB, GB);
    const vec2  LC = 1.0 - smoothstep((WC - 1.5) * Len / ScaleC, (WC + 1.5) * Len / ScaleC, GC);
    const float Grid = clamp(clamp(LA.x + LA.y, 0.0, 1.0) * clamp(WA, 0.0, 1.0)
                           + max(clamp(LB.x + LB.y, 0.0, 1.0) * clamp(WB, 0.0, 1.0), clamp(LC.x + LC.y, 0.0, 1.0) * clamp(WC, 0.0, 1.0)), 0.0, 1.0);
    const float AxisZ = 1.0 - smoothstep((WB - 1.5) * Len.x, (WB + 1.5) * Len.x, abs(UV.x));        // the Z axis at x = 0
    const float AxisX = 1.0 - smoothstep((WB - 1.5) * Len.y, (WB + 1.5) * Len.y, abs(UV.y));        // the X axis at z = 0

    float ArrowX = 0.0, ArrowZ = 0.0;
    const float aa = 0.5 * (Len.x + Len.y);
    const float CellX = round(UV.x / kGridArrowSpacing), CellZ = round(UV.y / kGridArrowSpacing);
    if (CellX >= 0.0) ArrowX = max(ArrowX, smoothstep(-aa, aa, -GridSdArrow(mat2(-1.0, 0.0, 0.0, -1.0) * (UV - vec2(CellX * kGridArrowSpacing + kGridArrowStemLength, 0.0)))));
    if (CellX <= 0.0) ArrowX = max(ArrowX, smoothstep(-aa, aa, -GridSdBox(UV - vec2(CellX * kGridArrowSpacing, 0.0), vec2(0.0), vec2(-kGridArrowStemLength, 0.0), kGridArrowStemWidth)));
    if (CellZ >= 0.0) ArrowZ = max(ArrowZ, smoothstep(-aa, aa, -GridSdArrow(mat2(0.0, 1.0, -1.0, 0.0) * (UV - vec2(0.0, CellZ * kGridArrowSpacing + kGridArrowStemLength)))));
    if (CellZ <= 0.0) ArrowZ = max(ArrowZ, smoothstep(-aa, aa, -GridSdBox(mat2(0.0, 1.0, -1.0, 0.0) * (UV - vec2(0.0, CellZ * kGridArrowSpacing - kGridArrowStemLength)), vec2(0.0), vec2(-kGridArrowStemLength, 0.0), kGridArrowStemWidth)));

    vec4 C = mix(kGridBaseColor, kGridLineColor, Grid * CosTheta);
    C = mix(C, kGridXAxisColor, AxisX * CosTheta);
    C = mix(C, kGridZAxisColor, AxisZ * CosTheta);
    C = mix(C, kGridXAxisColor, ArrowX * CosTheta);
    C = mix(C, kGridZAxisColor, ArrowZ * CosTheta);
    return C.rgb * C.rgb;           // (the grid's own approximation of sRGB to linear)
}

#endif
