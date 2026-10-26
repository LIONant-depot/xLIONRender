# Lighting fixtures (P0 definitions)

The break tests of `xLION/documentation/lighting_strategy.md` (sec. 11) as fixture definitions: the scene, the action, what is
checked, and the phase that turns each one on. P0 defines them; the phase that needs a fixture builds it. Tests live in
`src/tests/`; images are compared with the saved outputs (`src/tests/run_baseline.py`, `XLIONRENDER_BASELINE`).

## What exists (P0)

| Fixture | Where | Checks |
|---|---|---|
| Saved outputs | `run_baseline.py save/compare`, `tests/xlionrender_test_common.h` (`baseline`) | every HDR readback of the lighting tests, bitwise (or `XLIONRENDER_BASELINE_TOL`) |
| Sampling reference | `tests/xlionrender_test_light_lab.cpp` | EXACT vs CELLS (same image), SAMPLED converging to EXACT on the same visibility (shadow maps); 64-frame mean = the many-sample tail reference |
| Visibility reference | `tests/xlionrender_test_light_reference.cpp` | per pixel, one light's shadow-map visibility vs the GI's rays (`gi::world::Occluded`) on the same triangles; opaque rigid only (label printed) |
| LightEval analytic values | `tests/xlionrender_test_light_reference.cpp` | falloff ends, half radius, monotonic, scale; cone inside, outside, halfway |
| Radiometry vs visibility | `tests/xlionrender_test_light_reference.cpp` | a colour/intensity change keeps the shadow map and scales the light exactly; a move renders the map again |
| Ledger | `XLION_LEDGER` (renderer), `xLION/source/Editors/LevelEditor/smoke/lighting_ledger.py` | median/p95/p99 per row, clocks, compile idle; stress city and Police |
| Counters | `DescribeScene`: `LightsLocal`, `LightsDropped`, `LightsShadowed`, `LightsShadowFallback`, `LightsShadowStale`, `LightCellsOverflow` | what the last render could not shade as asked |

## Break tests (defined; built by the phase named)

| # | Fixture | Scene and action | Check | Phase |
|---|---|---|---|---|
| B1 | Door opening (rediscovery) | A room lit only through a door by a light outside; the door opens over 10 frames | The light reaches the room's pixels within the discovery bound; no light without support in the tail (p > 0) | P2 |
| B2 | Small blocker, long shadow | A 0.2 m box near a low light, its shadow 8 m long; the box moves 0.5 m | Every tile under the old and the new shadow re-validated; the far end of the shadow correct the same frame (non-local invalidation) | P1a |
| B3 | Mover stops, becomes cache-eligible, moves again | One box moving for 30 frames, still for N + 10, moving again | Working depth = a full redraw at every frame (the six transitions of sec. 7); the static cache invalidated on the second move | P1a |
| B4 | Light colour change | A shadowed light changes colour and intensity | No shadow map rendered; the image scales exactly (exists in P0, see above) | P0 (done) |
| B5 | Moving sun | The sun turns 0.1 degree a frame for 300 frames | Pages rendered per frame and cost stay bounded; no stale tile sampled as evidence | P1a |
| B6 | Glossy floor across tiles | A low-roughness floor under a tail light whose highlight crosses a tile boundary at half resolution | No seam or pop at the boundary above the reconstruction tolerance | P2-P3 |
| B7 | Camera teleport into a cold region | Teleport 200 m to an unvisited street | Cold-start p95/p99 for 60 frames; fallbacks counted, none sampled as evidence | P1a (sun), P2 (lists) |
| B8 | Alpha-tested foliage | A masked leaf card between a light and a floor | Raster shadow vs rays: rays report UNKNOWN (unsupported semantics) until P4's alpha any-hit; never CLEAR from a masked hit | P4 |
| B9 | GI state eviction | A GI chunk's light evicted while its geometry shadows a lit floor | Direct shadows unchanged; rays report UNKNOWN where geometry left the resident coverage, never CLEAR | P4 |
| B10 | Many overlapping lights | 300 lights within one cell (exists as the light lab's "overflow") | The cell shades exactly (all lights); counted in `LightCellsOverflow` | P0 (counter), P2 (per-tile) |
| B11 | Forced overflow | Shadow update budget 0 with 50 movers; cell buffer forced full | Declared fallback everywhere it applies, counted (`LightsShadowFallback`, stale 0 after P1a); no crash, no stale evidence | P1a |
