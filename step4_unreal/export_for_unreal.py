"""Step 4 (bonus) — Export triangulated skeletons to an Unreal-friendly format.

Reads step2_triangulation/output/results_3d.json and writes
skeletons_unreal.json with coordinates converted to Unreal Engine
conventions:

  ours:   right-handed, Z-up, millimeters, origin at court center
  Unreal: left-handed,  Z-up, centimeters, X forward / Y right

  UE_X = X_mm / 10
  UE_Y = -Y_mm / 10     (handedness flip)
  UE_Z = Z_mm / 10

Output schema:
{
  "fps": 25,
  "keypoint_names": [...18...],
  "bones": [[a, b], ...],           # keypoint index pairs
  "frames": [
    {"index": 1,
     "players": {"Red_11": [[x,y,z] or null, ...18...], ...}},
    ...
  ]
}
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.coco_utils import KEYPOINT_NAMES, SKELETON_EDGES

STEP2_RESULTS = Path(__file__).parent.parent / 'step2_triangulation' / 'output' / 'results_3d.json'
OUT_PATH = Path(__file__).parent / 'skeletons_unreal.json'


def to_unreal(p_mm):
    return [p_mm[0] / 10.0, -p_mm[1] / 10.0, p_mm[2] / 10.0]


def main():
    data = json.load(open(STEP2_RESULTS))
    results = data['results']

    frames = []
    for f_idx in sorted(results, key=int):
        players = {}
        for player, pdata in results[f_idx].items():
            kpts = []
            for p in pdata['X_3d_mm']:
                if p[0] is None or p[0] != p[0]:   # null / NaN
                    kpts.append(None)
                else:
                    kpts.append(to_unreal(p))
            players[player] = kpts
        frames.append({'index': int(f_idx), 'players': players})

    out = {
        'fps': 25,
        'units': 'cm (Unreal), left-handed Z-up',
        'keypoint_names': KEYPOINT_NAMES,
        'bones': [list(e) for e in SKELETON_EDGES],
        'frames': frames,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f)
    n_pl = len(frames[0]['players']) if frames else 0
    print(f'Exported {len(frames)} frames x {n_pl} players -> {OUT_PATH}')


if __name__ == '__main__':
    main()
