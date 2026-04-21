"""Blender GLB → relay-swarm sim map converter.

Usage:
    python tools/export_sim_map.py --in path/to/scene.blend --out environments/map_name
    python tools/export_sim_map.py --in path/to/scene.glb   --out environments/map_name

Produces two outputs:
    environments/map_name.sim.json   — canonical runtime format (ingested by sim)
    environments/map_name.glb        — passthrough mesh for Unreal / visualization

Object naming convention (enforced):
    BLD_<name>         Building  — custom prop: sim_height (float, required)
    WALL_<name>        Wall      — custom prop: sim_height (float, optional)
    TREE_<name>        Tree      — custom prop: sim_radius (float, required)
    ZONE_TARGET_<name> Target zone
    ZONE_NOFLY_<name>  No-fly zone
    BASE_MAIN          Friendly base (exactly one required)
    BASE_ENEMY         Enemy base
    TURRET_<id>        Turret    — custom props: sim_range, sim_fire_rate
    WP_<index>         Waypoint  — index is a 1-based integer, must be contiguous

Scene convention: units in meters, X-east Y-north Z-up.
Scale factor: 1 Blender meter = SCALE_FACTOR world units (default 10).

Validation failures exit with code 1 and print actionable error messages.
"""
import argparse
import json
import os
import struct
import sys
import math

# Allow running from repo root or from tools/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── constants ─────────────────────────────────────────────────────────────────
SCALE_FACTOR = 10.0          # Blender meters → world units
WORLD_W      = 1200.0
WORLD_H      = 800.0
WORLD_DEPTH  = 400.0


# ── GLB parsing ───────────────────────────────────────────────────────────────

def _parse_glb(path: str) -> dict:
    """Parse a GLB file and return the embedded glTF JSON dict.

    GLB format: 12-byte header + one or more chunks.
    Chunk 0 is always JSON (gltf). Chunk 1 (if present) is binary buffer.
    """
    with open(path, "rb") as f:
        # 12-byte GLB header
        magic, version, total_length = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:  # 'glTF'
            raise ValueError(f"Not a GLB file: {path!r}")

        # Chunk 0 — JSON
        chunk_len, chunk_type = struct.unpack("<II", f.read(8))
        if chunk_type != 0x4E4F534A:  # 'JSON'
            raise ValueError("GLB chunk 0 is not JSON")
        json_bytes = f.read(chunk_len)

    return json.loads(json_bytes.decode("utf-8"))


def _get_custom_prop(node: dict, key: str, required=False):
    """Read a custom property from a glTF node extras dict."""
    extras = node.get("extras", {}) or {}
    val    = extras.get(key) or extras.get(f"sim_{key.lstrip('sim_')}")
    if val is None and required:
        raise ValueError(
            f"Node {node.get('name')!r} is missing required custom property {key!r}"
        )
    return val


def _node_world_translation(node: dict, gltf: dict) -> list:
    """Return [x, y, z] world position of a glTF node, applying SCALE_FACTOR.

    Converts from Blender (X-east, Y-forward/north, Z-up) to sim convention.
    glTF stores Y-up with Z-back, so axis remapping is applied.

    Blender → glTF export (default): X=X, Y=Z, Z=-Y  (Y-up convention)
    glTF → sim: X=X (east), Y=Y_gltf_z (north), Z=Z_gltf_y (up)
    Net: sim_x = gltf_x, sim_y = -gltf_z, sim_z = gltf_y
    Then translate to sim world origin:
        sim_x += WORLD_W / 2,  sim_y += WORLD_H / 2
    """
    t = node.get("translation", [0.0, 0.0, 0.0])
    s = SCALE_FACTOR
    sim_x = t[0] * s + WORLD_W / 2.0
    sim_y = (-t[2]) * s + WORLD_H / 2.0
    sim_z = t[1] * s
    return [round(sim_x, 2), round(sim_y, 2), round(sim_z, 2)]


def _node_bbox_rect(node: dict, gltf: dict) -> list:
    """Approximate a building/zone rect from the node's mesh bbox or scale.

    Returns [x, y, w, h] in sim world units.
    Falls back to a 60x60 box centered at the node's position.
    """
    pos   = _node_world_translation(node, gltf)
    scale = node.get("scale", [1.0, 1.0, 1.0])

    # Use X and -Z Blender scale for footprint width/height
    s     = SCALE_FACTOR
    w_raw = abs(scale[0]) * s * 2   # half-extent → full width
    h_raw = abs(scale[2]) * s * 2   # Blender Z-scale → sim Y-depth

    if w_raw < 1.0:
        w_raw = 60.0
    if h_raw < 1.0:
        h_raw = 60.0

    x = round(pos[0] - w_raw / 2.0, 2)
    y = round(pos[1] - h_raw / 2.0, 2)
    return [x, y, round(w_raw, 2), round(h_raw, 2)]


# ── conversion ────────────────────────────────────────────────────────────────

def gltf_to_sim(gltf: dict) -> dict:
    """Convert glTF scene nodes (by naming convention) to sim.json dict."""

    nodes = gltf.get("nodes", [])

    buildings  = []
    walls      = []
    trees      = []
    zones      = []
    turrets    = []
    waypoints  = {}     # index → position
    base_main  = None
    base_enemy = None

    errors = []

    for node in nodes:
        name = node.get("name", "")

        if name.startswith("BLD_"):
            try:
                height_m = _get_custom_prop(node, "sim_height", required=True)
                height   = float(height_m) * SCALE_FACTOR
                if height <= 0:
                    errors.append(f"{name}: sim_height must be positive (got {height_m})")
                    continue
                rect = _node_bbox_rect(node, gltf)
                if rect[2] <= 0 or rect[3] <= 0:
                    errors.append(f"{name}: zero-area footprint")
                    continue
                buildings.append({"rect": rect, "height": round(height, 2)})
            except ValueError as e:
                errors.append(str(e))

        elif name.startswith("WALL_"):
            pos     = _node_world_translation(node, gltf)
            scale   = node.get("scale", [1.0, 1.0, 1.0])
            half_l  = abs(scale[0]) * SCALE_FACTOR
            height_m = _get_custom_prop(node, "sim_height")
            height   = float(height_m) * SCALE_FACTOR if height_m else 50.0
            # Approximate: wall runs along X axis at this position
            walls.append({
                "start":  [round(pos[0] - half_l, 2), round(pos[1], 2)],
                "end":    [round(pos[0] + half_l, 2), round(pos[1], 2)],
                "height": round(height, 2),
            })

        elif name.startswith("TREE_"):
            try:
                radius_m = _get_custom_prop(node, "sim_radius", required=True)
                radius   = float(radius_m) * SCALE_FACTOR
                if radius <= 0:
                    errors.append(f"{name}: sim_radius must be positive")
                    continue
                pos = _node_world_translation(node, gltf)
                trees.append({"position": pos[:2], "radius": round(radius, 2)})
            except ValueError as e:
                errors.append(str(e))

        elif name.startswith("ZONE_TARGET_"):
            rect = _node_bbox_rect(node, gltf)
            if rect[2] <= 0 or rect[3] <= 0:
                errors.append(f"{name}: zero-area zone footprint")
                continue
            zones.append({"rect": rect, "type": "TARGET"})

        elif name.startswith("ZONE_NOFLY_"):
            rect = _node_bbox_rect(node, gltf)
            if rect[2] <= 0 or rect[3] <= 0:
                errors.append(f"{name}: zero-area zone footprint")
                continue
            zones.append({"rect": rect, "type": "NOFLYZONE"})

        elif name == "BASE_MAIN":
            pos       = _node_world_translation(node, gltf)
            base_main = {"position": pos[:2]}

        elif name == "BASE_ENEMY":
            pos        = _node_world_translation(node, gltf)
            base_enemy = {"position": pos[:2]}

        elif name.startswith("TURRET_"):
            pos = _node_world_translation(node, gltf)
            t   = {"position": pos[:2]}
            r   = _get_custom_prop(node, "sim_range")
            fr  = _get_custom_prop(node, "sim_fire_rate")
            if r:  t["range"]     = float(r) * SCALE_FACTOR
            if fr: t["fire_rate"] = float(fr)
            turrets.append(t)

        elif name.startswith("WP_"):
            suffix = name[3:]
            if not suffix.isdigit():
                errors.append(f"{name}: waypoint suffix must be an integer (got {suffix!r})")
                continue
            idx = int(suffix)
            if idx <= 0:
                errors.append(f"{name}: waypoint index must be ≥ 1")
                continue
            pos = _node_world_translation(node, gltf)
            waypoints[idx] = pos[:2]

    # ── validation ────────────────────────────────────────────────────────────
    if base_main is None:
        errors.append("Scene is missing exactly one BASE_MAIN object")

    if waypoints:
        indices = sorted(waypoints.keys())
        expected = list(range(1, len(indices) + 1))
        if indices != expected:
            errors.append(
                f"Waypoint indices are not contiguous: found {indices}, "
                f"expected {expected}"
            )

    for bld in buildings:
        x, y, w, h = bld["rect"]
        if x < 0 or y < 0 or x + w > WORLD_W or y + h > WORLD_H:
            errors.append(
                f"Building rect {bld['rect']} extends outside world bounds "
                f"({WORLD_W}×{WORLD_H})"
            )

    if errors:
        print("\nValidation FAILED — fix the following errors in the Blender scene:\n")
        for e in errors:
            print(f"  ✗  {e}")
        print()
        sys.exit(1)

    # ── assemble output ───────────────────────────────────────────────────────
    ordered_wps = [waypoints[i] for i in sorted(waypoints.keys())]

    return {
        "meta":        {"version": 2, "units": "world_units"},
        "world":       {"width": WORLD_W, "height": WORLD_H, "depth": WORLD_DEPTH},
        "buildings":   sorted(buildings,  key=lambda b: b["rect"]),
        "walls":       walls,
        "trees":       sorted(trees,      key=lambda t: t["position"]),
        "zones":       zones,
        "base":        base_main,
        "enemy_base":  base_enemy,
        "turrets":     turrets,
        "waypoints":   ordered_wps,
        "enemy_swarm": None,
    }


# ── blend file handling (requires bpy — optional) ─────────────────────────────

def _blend_to_glb(blend_path: str, glb_path: str):
    """Export a .blend to .glb using Blender's Python API (bpy).

    This runs in-process only when Blender's Python is available.
    If bpy is not available, the caller should use Blender's CLI:
        blender --background scene.blend --python-expr \
            "import bpy; bpy.ops.export_scene.gltf(filepath='out.glb')"
    """
    try:
        import bpy  # type: ignore
    except ImportError:
        print("bpy not available — run this script inside Blender or pre-export to .glb:")
        print(f"  blender --background {blend_path} --python-expr \\")
        print(f'    "import bpy; bpy.ops.export_scene.gltf(filepath=\\"{glb_path}\\")"')
        sys.exit(1)

    bpy.ops.wm.open_mainfile(filepath=blend_path)
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format="GLB",
        export_extras=True,          # include custom properties
        export_apply=True,           # apply modifiers
        export_yup=True,             # Y-up (glTF convention)
    )
    print(f"Exported GLB → {glb_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Convert a Blender scene to relay-swarm sim map files."
    )
    ap.add_argument("--in",  dest="input",  required=True,
                    help="Input file: .blend (requires bpy) or .glb")
    ap.add_argument("--out", dest="output", required=True,
                    help="Output base path (e.g. environments/my_map). "
                         "Produces <out>.sim.json and <out>.glb")
    ap.add_argument("--scale", type=float, default=SCALE_FACTOR,
                    help=f"Blender-meters → world-units scale factor (default {SCALE_FACTOR})")
    ap.add_argument("--validate-only", action="store_true",
                    help="Run validation checks but do not write output files")
    args = ap.parse_args()

    global SCALE_FACTOR
    SCALE_FACTOR = args.scale

    inp = args.input
    glb_out  = args.output + ".glb"
    json_out = args.output + ".sim.json"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Convert .blend → .glb if needed
    if inp.endswith(".blend"):
        _blend_to_glb(inp, glb_out)
        glb_path = glb_out
    elif inp.endswith(".glb"):
        glb_path = inp
        if not args.validate_only:
            import shutil
            shutil.copy2(inp, glb_out)
            print(f"Copied GLB → {glb_out}")
    else:
        print(f"Unsupported input format: {inp}")
        sys.exit(1)

    # Parse GLB
    gltf = _parse_glb(glb_path)

    # Convert to sim format (validation is embedded; exits on failure)
    sim_data = gltf_to_sim(gltf)

    if args.validate_only:
        print("Validation passed.")
        return

    with open(json_out, "w") as f:
        json.dump(sim_data, f, indent=2)
    print(f"Sim map written  → {json_out}")

    bld_count = len(sim_data["buildings"])
    wpt_count = len(sim_data["waypoints"])
    tur_count = len(sim_data["turrets"])
    print(f"  buildings={bld_count}  waypoints={wpt_count}  turrets={tur_count}"
          f"  base={'✓' if sim_data['base'] else '✗'}"
          f"  enemy_base={'✓' if sim_data['enemy_base'] else '—'}")


if __name__ == "__main__":
    main()
