"""Parsing delle annotazioni Roboflow COCO → struttura usabile per triangolazione.

Struttura output:
    obs[frame_idx][cam_id][player_id] = (kpts (18,3), bbox (4,))

dove kpts[:, 2] è la visibility (0 o 2 con Roboflow).
"""
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np



KEYPOINT_NAMES = [
    'Head', 'LShoulder', 'Neck', 'RShoulder',
    'LElbow', 'LHand', 'RElbow', 'RHand',
    'Spine', 'Hips', 'LHip', 'RHip',
    'LKnee', 'RKnee', 'LAnkle', 'RAnkle',
    'LFoot', 'RFoot'
]
N_KPTS = len(KEYPOINT_NAMES)  # 18

# Edges per disegnare lo scheletro
SKELETON_EDGES = [
    (0, 2), (2, 1), (2, 3),         # testa-collo-spalle
    (1, 4), (4, 5), (3, 6), (6, 7), # braccia
    (2, 8), (8, 9),                 # tronco
    (9, 10), (9, 11),               # bacino
    (10, 12), (12, 14), (14, 16),   # gamba sx
    (11, 13), (13, 15), (15, 17),   # gamba dx
]

# Pattern per estrarre cam_id e frame_idx dal nome file Roboflow:
#   out{C}_frame_{F:04d}_png.rf.{hash}.png
FNAME_RE = re.compile(r'out(\d+)_frame_(\d+)_png\.rf\.')


def parse_filename(fname: str):
    """Estrae (cam_id, frame_idx) dal nome file Roboflow."""
    m = FNAME_RE.search(fname)
    if not m:
        return None
    cam_idx = int(m.group(1))
    frame_idx = int(m.group(2))
    return f'cam_{cam_idx}', frame_idx


def load_annotations(coco_path):
    """Carica COCO e restituisce:
        obs[frame_idx][cam_id][player_name] = {'kpts': (18,3), 'bbox': (4,)}
        players: set di nomi giocatori (es. {'Red_11', 'White_22', ...})
    """
    d = json.load(open(coco_path))

    # mappa image_id → (cam_id, frame_idx)
    img_map = {}
    for im in d['images']:
        parsed = parse_filename(im['file_name'])
        if parsed is None:
            continue
        img_map[im['id']] = parsed

    # mappa category_id → nome giocatore (es. 'Red_11')
    # Roboflow mette tutti i giocatori come sotto-categorie del nome dataset
    cat_map = {}
    for c in d['categories']:
        name = c['name']
        # Saltiamo la "supercategoria" che ha il nome del dataset
        if c['supercategory'] == 'none':
            continue
        cat_map[c['id']] = name

    obs = defaultdict(lambda: defaultdict(dict))
    players = set()
    for a in d['annotations']:
        if a['image_id'] not in img_map:
            continue
        if a['category_id'] not in cat_map:
            continue
        if 'keypoints' not in a:
            continue  # solo bbox, no scheletro
        cam_id, frame_idx = img_map[a['image_id']]
        player = cat_map[a['category_id']]
        # Roboflow tronca i keypoint trailing con visibility=0.
        # Padding fino a 18 keypoint con (0,0,0).
        raw = a['keypoints']
        n = len(raw) // 3
        kpts = np.zeros((N_KPTS, 3), dtype=np.float64)
        kpts[:n] = np.array(raw, dtype=np.float64).reshape(n, 3)
        bbox = np.array(a['bbox'], dtype=np.float64)
        obs[frame_idx][cam_id][player] = {'kpts': kpts, 'bbox': bbox}
        players.add(player)

    # Converti defaultdict in dict normale
    obs = {f: dict(cams) for f, cams in obs.items()}
    return obs, sorted(players)


if __name__ == '__main__':
    from pathlib import Path
    coco = str(Path(__file__).resolve().parent.parent / 'data' / 'annotations' / '_annotations.coco.json')
    obs, players = load_annotations(coco)
    print(f'Frame disponibili: {sorted(obs.keys())}')
    print(f'Giocatori: {players}')
    print()
    print('Annotazioni per frame e camera:')
    for f in sorted(obs):
        for cam in sorted(obs[f]):
            n = len(obs[f][cam])
            print(f'  frame {f}, {cam}: {n} giocatori')
