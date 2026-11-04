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
