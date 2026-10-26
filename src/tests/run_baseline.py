"""Saved outputs (documentation/lighting_strategy.md P0): runs the lighting tests and saves or compares every HDR readback
(tests/xlionrender_test_common.h, baseline).

    python run_baseline.py save    <folder> [Debug|Release]    the reference images, before a change
    python run_baseline.py compare <folder> [Debug|Release]    after it: bitwise, or XLIONRENDER_BASELINE_TOL
"""
import os, subprocess, sys
from pathlib import Path

TESTS = ["xlionrender_test_light_lab", "xlionrender_test_light_reference", "xlionrender_test_local_shadows", "xlionrender_test_probes",
         "xlionrender_test_forward", "xlionrender_test_shadows"]

def main():
    mode, root = sys.argv[1], Path(sys.argv[2]).resolve()
    config = sys.argv[3] if len(sys.argv) > 3 else "Release"
    exe_dir = Path(__file__).resolve().parents[2] / "Build" / "test" / "vs2022" / config
    failed = []
    for t in TESTS:
        folder = root / t
        folder.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, **{("XLIONRENDER_BASELINE_SAVE" if mode == "save" else "XLIONRENDER_BASELINE"): str(folder)})
        r = subprocess.run([str(exe_dir / f"{t}.exe")], cwd=exe_dir, env=env, capture_output=True, text=True, timeout=900)
        lines = [l for l in r.stdout.splitlines() if l.startswith(("BASELINE", "FAIL", "PASS", "[error]"))]
        print(f"{t}: exit {r.returncode}")
        for l in lines: print("   ", l)
        if r.returncode: failed.append(t)
    print("FAILED: " + ", ".join(failed) if failed else "ALL PASS")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
