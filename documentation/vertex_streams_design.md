# Flexible vertex streams (COLOR0 and UV1 built 2026-10-22)

Tomas, 2026-10-22: some streams are required (the core); extra UVs, colors, etc. come as additional streams that the
pipeline allows and handles. The vertex shader always gives the fragment stage every input: from the stream when the
geometry has it, otherwise a constant (or a uniform).

## Today

- xraw3d already imports n UVs and n colors per vertex (`m_nUVs`, `m_nColors`).
- xgeom_static / xgeom_skin compilers drop them: colors are never written, `SubMesh.m_nUVs = 1; //HACK`.
- Both formats have fixed streams: index, stream 0 (position + packed bits), stream 1 (`vertex_extras`: UV0, oct normal/tangent).
- The vertex shaders set `VertColor = vec4(1)`.
- The renderer has one vertex descriptor per kind (`m_StaticGeomVD`, `m_SkinGeomVD`), material pipelines are cached per
  `(material, geom_kind)` in `getMaterialInstance`, and draws bind `setStreamingBuffers({ G.m_pStreams, 3 })`.
- xgeom (the old generic format) already had the vocabulary: `stream_info::element_def` masks (`color_mask_v`, `uv_mask_v`...).

## Proposal

**Core streams (required, unchanged):** index, stream 0, stream 1. Every pass that needs only position (shadows, depth,
masks, GI capture) keeps binding just these, so none of those pipelines change.

**Optional streams:** one GPU vertex buffer each, named by a bit in a geometry-wide mask:

| bit | stream  | format                                   | missing: the vertex shader gives |
|-----|---------|------------------------------------------|----------------------------------|
| 0   | COLOR0  | RGBA8 sRGB (UINT8_4D_NORMALIZED), decoded to linear in the VS | `vec4(1)`            |
| 1   | UV1     | FLOAT_2D (ponytail: half floats once xGPU has the format)     | UV0                  |
| ... | COLOR1, UV2, ... later, same pattern |                              |                                  |

**Format (both geometry types):** `m_OptionalStreams` (the mask) plus one offset into `m_pData` per present stream;
serialized; `xserializer_version_v` bumped. The loader creates one buffer per present stream; `runtime_allocation`
grows to hold them. The compilers stop dropping colors and extra UVs.

**Missing stream = a stride-0 default buffer (revised after the competitor check below).** One vertex shader per kind
declares every optional input; there are no shader variants. A stream the geometry lacks is bound to a shared
one-element default buffer (white for COLOR0) with stride 0, so every vertex reads that constant. This is what Godot 4
does (`mesh_default_rd_buffers`, `vd.stride = 0`) and why Unreal has `FNullColorVertexBuffer` (a one-color buffer "to
keep from needing a separate vertex factory").

**One pipeline per kind (Tomas: Vulkan 1.3 is fine):** the vertex descriptors `m_StaticGeomVD` / `m_SkinGeomVD` declare
every stream and `m_bDynamicStride`, so the stride (4, or 0 for the default) is given at bind time
(`vkCmdBindVertexBuffers2`) and the geometry's mask only changes what gets bound. No pipeline or shader variants.

**xGPU:** `vertex_descriptor::setup::m_bDynamicStride`; `cmd_buffer::setStreamingBuffers(span<buffer* const>, ConstantStreams)`
(stride = the buffer's entry size, 0 for a ConstantStreams bit). Binding a pipeline with static strides overwrites the
dynamic ones, so the command buffer keeps the last strided streams and binding a dynamic-stride pipeline applies them
again. The instance asks for API 1.3; a 1.2 device still runs, but a dynamic-stride pipeline then fails to create.

**Draw:** bind the core streams, then each optional slot: the geometry's buffer, or the default buffer.

## Competitors

- **Godot 4:** surfaces carry a format bitmask; missing attributes bind shared default buffers with stride 0 (color
  white, UVs 0). Pipelines are cached per vertex format, shaders are not permuted.
- **Unreal:** `FNullColorVertexBuffer` for meshes without colors, so the local vertex factory has no no-color permutation.
- **Unity:** up to 4 vertex streams; Color and TexCoord0..7 are optional and live in a second stream; skinned meshes
  split deformed / static (colors, UVs) / skinning streams. Same split as our core + optional.
- Vertex pulling from a storage buffer (my first idea) measures about even with vertex buffers for indexed draws, but
  nobody needs it for this; the stride-0 buffer is the common answer.

**Material side:** the Vertex Input node has Color, UV and UV1 pins. `VaryingFull` is unchanged: UV1 is its own varying at
location 6 (`OutUV1` in every vertex shader that feeds material fragments, `InUV1` where a fragment reads it), so the
fragment shaders that never read it do not change.

**Importer:** store colors as 8-bit sRGB. FBX colors are already sRGB-like; glTF `COLOR_0` is linear and is converted.

## Phases

1. DONE: COLOR0 in both formats (version 5: `m_OptionalStreams`, `m_VertexColorOffset`), compilers, loaders
   (`ColorBuffer`), xGPU dynamic strides, the renderer's white default (`BindClusterStreams`), the shaders decode sRGB.
   Tests: xlionrender_test_static_geom (a colored and an uncolored sphere in one frame), xgeom_skin_compiler -SELFTEST
   (the colors follow their vertices through the cluster split). The geom editors shade through xLIONRender: they get it.
2. DONE: UV1 in both formats (version 6: `stream_uv1_v`, `m_UV1Offset`), compilers (UV0 for a submesh without a second set
   when another has one), loaders (`UV1Buffer`), the renderer's default (`m_NoUV1Stream`: one float2 marker, stride 0;
   `mb_UV1` reads UV0 below -1e38), the Vertex Input node's UV1 pin. Tests: xlionrender_test_static_geom (a sphere with UV1
   and one without in one frame), xgeom_skin_compiler -SELFTEST (UV1 follows its vertex).
3. DONE: glTF linear colors converted to sRGB at import (xraw3d assimp_v3 `StoredColor`, .gltf/.glb only).

## Open points

- Missing color: constant white (the default buffer) for now; a per-material uniform would replace the default buffer's
  binding with a material parameter.
