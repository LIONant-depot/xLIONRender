// What the G-buffer prepass keeps for the material tile pass (material_tiles_plan.md): 64 bits, Tomas's polar codec: U, V 16-bit fixed point (wrapped: the UV's tile is lost, a repeating sampler doesn't mind);
// each gradient 8 bits of log length, 8 of direction. The tangent 32 bits: octahedral, snorm 16 + 16, the low bit of x its binormal's sign. xlionrender_tile_standard_frag.glsl and xlionrender_gprepass_tile_frag.glsl
const float tile_log_min_v = -24.0, tile_log_range_v = 32.0, tile_turn_v = 6.28318530718;
uint TileGradEncode(in const vec2 g)
{
    const float L = dot(g, g);
    if (L < 1e-30) return 0u;
    const uint Len = uint(clamp((0.5 * log2(L) - tile_log_min_v) / tile_log_range_v, 0.0, 1.0) * 254.0 + 1.5);       // (0: no gradient)
    const uint Dir = uint(floor(fract(atan(g.y, g.x) / tile_turn_v) * 256.0 + 0.5)) & 255u;
    return Len | (Dir << 8u);
}
vec2 TileGradDecode(in const uint w)
{
    const uint Len = w & 255u;
    if (Len == 0u) return vec2(0.0);
    const float a = float((w >> 8u) & 255u) / 256.0 * tile_turn_v;
    return vec2(cos(a), sin(a)) * pow(2.0, float(Len - 1u) / 254.0 * tile_log_range_v + tile_log_min_v);
}
uvec2 TileEncode(in const vec2 uv, in const vec2 dx, in const vec2 dy) { return uvec2(packUnorm2x16(fract(uv)), TileGradEncode(dx) | (TileGradEncode(dy) << 16u)); }
void TileDecode(in const uvec2 w, out vec2 uv, out vec2 dx, out vec2 dy) { uv = unpackUnorm2x16(w.x); dx = TileGradDecode(w.y & 0xFFFFu); dy = TileGradDecode(w.y >> 16u); }
uint TileTangentEncode(in const vec2 o, in const float Sign) { return (packSnorm2x16(o) & 0xFFFFFFFEu) | (Sign < 0.0 ? 1u : 0u); }
