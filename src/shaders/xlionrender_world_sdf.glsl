// The world's distance field (reflections_sdf_plan, prototype): cascades of 128^3 around a center, each cell twice the last, as Flax's and Godot's
// (their reflection rays march this, no BVH). The includer declares the buffer WorldSDF (floats): [0..2] the center, [3] cascade 0's cell, then the cascades.
const int sdf_dim_v = 128, sdf_cascades_v = 4;

float SDFLoad(int c, ivec3 i)
{
    i = clamp(i, ivec3(0), ivec3(sdf_dim_v - 1));
    return WorldSDF[4 + c * sdf_dim_v * sdf_dim_v * sdf_dim_v + i.z * sdf_dim_v * sdf_dim_v + i.y * sdf_dim_v + i.x];
}

float SDFAt(int c, vec3 g)      // (g: in the cascade's cells)
{
    const vec3  q = g - 0.5, b = floor(q), f = q - b;
    const ivec3 i = ivec3(b);
    const float d00 = mix(SDFLoad(c, i), SDFLoad(c, i + ivec3(1, 0, 0)), f.x), d10 = mix(SDFLoad(c, i + ivec3(0, 1, 0)), SDFLoad(c, i + ivec3(1, 1, 0)), f.x);
    const float d01 = mix(SDFLoad(c, i + ivec3(0, 0, 1)), SDFLoad(c, i + ivec3(1, 0, 1)), f.x), d11 = mix(SDFLoad(c, i + ivec3(0, 1, 1)), SDFLoad(c, i + ivec3(1, 1, 1)), f.x);
    return mix(mix(d00, d10, f.y), mix(d01, d11, f.y), f.z);
}

// The surface a hit found (XLION_SDF 3): the 8 cells around it, filtered, those whose rays met something; the normal facing the ray
bool SDFSurface(vec3 P, vec3 D, out vec3 Albedo, out vec3 N)
{
    const vec3  Center = vec3(WorldSDF[0], WorldSDF[1], WorldSDF[2]);
    const float Cell0  = WorldSDF[3];
    Albedo = vec3(0.25); N = -D;
    for (int c = 0; c < sdf_cascades_v; ++c)
    {
        const float Cell = Cell0 * float(1 << c);
        const vec3  g = (P - Center) / Cell + float(sdf_dim_v) * 0.5;
        if (any(lessThan(g, vec3(1.0))) || any(greaterThan(g, vec3(float(sdf_dim_v) - 1.0)))) continue;
        const vec3  q = g - 0.5, b = floor(q), f = q - b;
        vec3  Sa = vec3(0.0), Sn = vec3(0.0);
        float Sw = 0.0;
        for (int k = 0; k < 8; ++k)
        {
            const ivec3 o = ivec3(k & 1, (k >> 1) & 1, k >> 2), i = clamp(ivec3(b) + o, ivec3(0), ivec3(sdf_dim_v - 1));
            const uvec2 s = WorldSDFSurf[c * sdf_dim_v * sdf_dim_v * sdf_dim_v + i.z * sdf_dim_v * sdf_dim_v + i.y * sdf_dim_v + i.x];
            if ((s.x >> 24) == 0u) continue;
            const vec3 w3 = mix(vec3(1.0) - f, f, vec3(o));
            const float w = w3.x * w3.y * w3.z + 1e-4;
            const vec3 a = vec3(float(s.x & 255u), float((s.x >> 8) & 255u), float((s.x >> 16) & 255u)) / 255.0;
            vec2 e = vec2(float(s.y & 65535u), float(s.y >> 16)) / 65535.0 * 2.0 - 1.0;
            vec3 n = vec3(e, 1.0 - abs(e.x) - abs(e.y));
            if (n.z < 0.0) n.xy = (1.0 - abs(e.yx)) * vec2(e.x >= 0.0 ? 1.0 : -1.0, e.y >= 0.0 ? 1.0 : -1.0);
            n = normalize(n);
            if (dot(n, D) > 0.0) n = -n;     // (both sides of a wall: the side the ray meets)
            Sa += a * a * w; Sn += n * w; Sw += w;
        }
        if (Sw <= 0.0) return false;
        Albedo = Sa / Sw;
        if (dot(Sn, Sn) > 1e-8) N = normalize(Sn);
        return true;
    }
    return false;
}

// The march: a cascade at a time, never back to a finer one. The thickness of a hit grows with the distance up to half a cell (Flax: RayTraceGlobalSDF)
bool SDFCast(vec3 O, vec3 D, out float T)
{
    const vec3  Center = vec3(WorldSDF[0], WorldSDF[1], WorldSDF[2]);
    const float Cell0  = WorldSDF[3];
    float t = 2.0 * Cell0;
    for (int c = 0; c < sdf_cascades_v; ++c)
    {
        const float Cell = Cell0 * float(1 << c);
        const vec3  Org  = Center - vec3(float(sdf_dim_v) * 0.5 * Cell);
        for (int s = 0; s < 48; ++s)
        {
            const vec3 g = (O + D * t - Org) / Cell;
            if (any(lessThan(g, vec3(1.0))) || any(greaterThan(g, vec3(float(sdf_dim_v) - 1.0)))) break;
            const float d = SDFAt(c, g);
            if (d < 0.75 * Cell * clamp(t / (2.0 * Cell) - 0.5, 0.0, 1.0)) { T = t; return true; }      // (unsigned: at a surface the filtered distance only falls to about half a cell)
            t += max(d, 0.5 * Cell) * 0.9;
        }
    }
    T = 1e30;
    return false;
}
