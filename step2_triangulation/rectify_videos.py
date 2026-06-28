"""Task 2.1 — Rettificazione video (wrapper attorno a utils/rectified_videos.py).

Lo script `utils/rectified_videos.py` è fornito dalla prof. Questo wrapper:
1. Si limita a chiamare la sua funzione process_video()
2. Risolve i path corretti del nostro layout
3. Produce video rettificati in data/rectified_videos/hpe_01/

Output: out{N}_rect.mp4 per ogni camera, dove la distorsione lente è stata rimossa
applicando cv2.undistortPoints + cv2.remap al video originale.

NOTA: la pipeline principale (run_triangulation.py) NON usa questi video.
Le annotazioni Roboflow sono già state fatte sui frame DISTORTI; per coerenza
slide 11 ("if you rectify the video you also have to apply the same
transformation to the GT"), la triangolazione applica `cv2.undistortPoints`
direttamente ai keypoint 2D — matematicamente equivalente.
Questo script serve come deliverable visivo (video con campo non più curvato).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.rectified_videos import process_video


PROJECT = Path(__file__).resolve().parent.parent
VIDEO_IN_DIR = Path('/home/lele/Desktop/CV/HPE/material4project/video/hpe_01')
OUT_DIR = PROJECT / 'data' / 'rectified_videos' / 'hpe_01'
CALIB_VERSION = 'camera_config_v2'
CAM_IDS = ('cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5', 'cam_7')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Output dir: {OUT_DIR}')

    for cid in CAM_IDS:
        n = int(cid.split('_')[1])
        video_in = VIDEO_IN_DIR / f'out{n}.mp4'
        calib_path = PROJECT / 'data' / 'cameras' / CALIB_VERSION / cid / 'camera_calib.json'
        video_out = OUT_DIR / f'out{n}_rect.mp4'

        if not video_in.exists():
            print(f'[SKIP] {cid}: input video mancante {video_in}')
            continue
        if not calib_path.exists():
            print(f'[SKIP] {cid}: calib mancante {calib_path}')
            continue

        print(f'\n=== {cid} ===')
        print(f'  input : {video_in.name}')
        print(f'  calib : {calib_path.relative_to(PROJECT)}')
        print(f'  output: {video_out.name}')
        process_video(str(video_in), str(calib_path), str(video_out))

    print('\n' + '='*60)
    print('Rettificazione completata.')
    print(f'Video rettificati salvati in: {OUT_DIR}')


if __name__ == '__main__':
    main()
