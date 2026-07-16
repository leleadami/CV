"""Step 4 (bonus) — Local 3D preview animation of the triangulated skeletons.

Does not require Unreal: renders the same exported data (skeletons_unreal.json,
converted back to meters/right-handed) to an MP4 with matplotlib + cv2.
The camera orbits the court while the 5 annotated frames play in a loop.

Output: skeletons_preview.mp4 (also usable as the "sample video results"
supplementary deliverable).
"""
import sys
import json
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.court import fiba_court_points, COURT_COLORS

HERE = Path(__file__).parent
DATA = HERE / 'skeletons_unreal.json'
OUT = HERE / 'skeletons_preview.mp4'

FPS = 12
SECONDS_PER_FRAME = 1.0     # each annotated frame is held this long
LOOPS = 2                   # play the 5-frame sequence twice while orbiting


def from_unreal(p_cm):
    """UE cm left-handed -> our meters right-handed."""
    return np.array([p_cm[0] / 100.0, -p_cm[1] / 100.0, p_cm[2] / 100.0])


def main():
    data = json.load(open(DATA))
    frames = data['frames']
    bones = data['bones']
    players = sorted(frames[0]['players'].keys())
    colors = {pl: ('#d62728' if pl.startswith('Red') else '#7f7f7f')
              for pl in players}

    steps_per_frame = int(FPS * SECONDS_PER_FRAME)
    total_steps = steps_per_frame * len(frames) * LOOPS

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    writer = None

    court = fiba_court_points()
    for step in range(total_steps):
        f_i = (step // steps_per_frame) % len(frames)
        azim = -80 + 50 * step / total_steps      # slow orbit
        ax.cla()

        for name, pts in court.items():
            if name in ('baskets', 'uwb_ceiling', 'hall_perimeter', 'corners'):
                continue
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3,
                       c=COURT_COLORS.get(name, 'gray'), alpha=0.4)

        for pl in players:
            kpts = frames[f_i]['players'].get(pl)
            if kpts is None:
                continue
            X = np.array([from_unreal(p) if p is not None else [np.nan]*3
                          for p in kpts])
            valid = ~np.isnan(X[:, 0])
            ax.scatter(X[valid, 0], X[valid, 1], X[valid, 2], s=18,
                       c=colors[pl])
            for a, b in bones:
                if valid[a] and valid[b]:
                    ax.plot([X[a, 0], X[b, 0]], [X[a, 1], X[b, 1]],
                            [X[a, 2], X[b, 2]], c=colors[pl], linewidth=1.4)

        ax.set_xlim(-15, 15); ax.set_ylim(-9, 9); ax.set_zlim(0, 4)
        ax.set_box_aspect([2.0, 1.2, 0.5])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title(f'Triangulated 3D skeletons — annotated frame {frames[f_i]["index"]}')
        ax.view_init(elev=24, azim=azim)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        if writer is None:
            h, w = buf.shape[:2]
            writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*'mp4v'),
                                     FPS, (w, h))
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f'Preview saved: {OUT} ({total_steps} frames @ {FPS} fps)')


if __name__ == '__main__':
    main()
