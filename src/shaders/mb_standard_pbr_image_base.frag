// The image-based PBR (the "Default PBR Image Based" material): the standard PBR, mb_standard_pbr.frag, whose image-based light is the
// renderer's environment probe (EnvLighting: the sky and the reflection probes). This file once had its own copy of that shader with an
// IBL of its own for the editors (XLION_LDR_OUTPUT): an envMap and a brdfLUT at set 2 bindings 0 and 1 - no editor bound them, and
// binding 1 is the lighting buffer (a sampler there made the validation layer crash drawing the material) - and the multiscatter term
// mb_standard_pbr.frag removed as wrong. PBRLightingImageBase is the name the material calls.
#include "mb_standard_pbr.frag"

#define PBRLightingImageBase PBRLighting
