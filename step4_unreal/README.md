# Step 4 (bonus) — Unreal Engine visualization

Displays the triangulated 3D player skeletons (step 2 output) in Unreal Engine.

## Files

- `export_for_unreal.py` — converts `step2_triangulation/output/results_3d.json`
  to `skeletons_unreal.json` in Unreal conventions (cm, left-handed, Z-up).
- `unreal_import_skeletons.py` — run inside the UE5 editor (Python plugin):
  spawns one colored sphere per joint per player and keyframes a
  LevelSequence over the annotated frames.
- `preview_animation.py` — local MP4 preview of the same data (no Unreal
  required), with orbiting camera. Output: `skeletons_preview.mp4`.

## Usage

```bash
# 1. export (after running step 2)
python3 step4_unreal/export_for_unreal.py

# 2. local preview (optional, no Unreal needed)
python3 step4_unreal/preview_animation.py

# 3. in Unreal Editor (UE 5.x, Python Editor Script Plugin enabled):
#    set JSON_PATH in unreal_import_skeletons.py to the absolute path of
#    skeletons_unreal.json, then in the editor Python console:
#    exec(open(r"/abs/path/unreal_import_skeletons.py").read())
```

## Coordinate conversion

| ours (calibration) | Unreal |
|---|---|
| right-handed, Z-up | left-handed, Z-up |
| millimeters | centimeters |
| origin = court center | origin = world origin |

`UE = (X/10, -Y/10, Z/10)`

## Note

Only 5 annotated frames exist (one action, ~1s clip), so the sequence holds
each pose 0.5 s. With a pose detector on the full clips (see report,
future work) the same pipeline would produce continuous motion.
