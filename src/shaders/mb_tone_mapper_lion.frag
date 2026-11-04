// LionMapper, is a real - time HDR - to - LDR tone mapping operator designed for photorealistic 
// rendering in engines like LIONAnt Engine.

vec3 ToneMapper_lion(vec3 v)
{
    const mat3 m = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772, 0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104);

    const mat3 m_inv = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116);

    v = clamp(log2(m * v) * 0.060606 + 0.755696, 0., 1.);
    v = (((2.3756 * v - 3.5265) * v + 0.4727) * v - 0.218) * v + 0.5815;
    v = (((1.7907 * v + 0.6957) * v - 1.0095) * v - 1.23219) * v + 0.716327;

    // The saturation in the curve's own space, before the power (as Filament's AgX looks): done after it, the luma added a grey veil to the darks
    float l = dot(v, vec3(0.2126, 0.7152, 0.0722));

    v = m_inv * pow(max(l + 1.2 * (v - l), 0.0), vec3(2.5));
    v *= 1.3 / (1.0 + 0.34 * max(v.r, max(v.g, v.b)));    // (lifts the darks and mids, its top as AgX; on the brightest channel: the color keeps its ratios)
    return v;
}

