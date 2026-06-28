"""Esperimento "best achievable" — chiude due domande aperte:

1. LOO post-BA: il BA migliora anche la metrica ONESTA (leave-one-view-out),
   o solo il fit? Il fit MPJPE riusa le viste della triangolazione; il LOO no.
   Se il LOO scende, il miglioramento di calibrazione è reale generalizzazione.

2. BA iterativo (round 2): dopo il primo BA la calib è migliore → riproiettando
   i court points il prior è più vicino alle intersezioni vere → il matching
   (radius 30px) può agganciare più punti → secondo BA meglio vincolato.
   Misura se 48/96 match e court RMSE migliorano.

Output: eval_best_summary.json
"""
import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.camera import Camera, load_cameras
from utils.coco_utils import load_annotations
from utils.triangulation import triangulate_skeleton, loo_reprojection_skeleton
from utils.bundle_adjustment import run_bundle_adjustment
from court_detector import build_court_observations_multiframe
from run_ba_real import (build_court_3d_extended, find_video_paths,
                         triangulate_all, average_mpjpe,
                         N_FRAMES_AGGR, MAX_MATCH_DIST_PX, BA_LOSS, BA_F_SCALE)

OUT_DIR = Path(__file__).parent / 'output_real'
PROJECT = Path(__file__).resolve().parent.parent


def average_loo(cams_dict, obs_players, players):
    """MPJPE leave-one-view-out medio su tutti gli scheletri (>=3 viste)."""
    vals = []
    for f in sorted(obs_players.keys()):
        for pl in players:
            kpts_by_cam = {}
            for cid in cams_dict:
                if cid in obs_players[f] and pl in obs_players[f][cid]:
                    kpts_by_cam[cid] = obs_players[f][cid][pl]['kpts']
            if len(kpts_by_cam) < 3:
                continue
            loo = loo_reprojection_skeleton(kpts_by_cam, cams_dict)
            m = np.nanmean(loo)
            if not np.isnan(m):
                vals.append(m)
    return float(np.mean(vals)) if vals else float('nan')


def detect_and_ba(cams_prior_dict, cam_ids, X_world, video_paths, tag):
    """Detection con prior dato + BA. Ritorna (cams_refined_list, stats)."""
    print(f'\n--- [{tag}] detection multi-frame (prior = {tag}) ---')
    obs_court, vis_court = build_court_observations_multiframe(
        cams_prior_dict, X_world, video_paths,
        n_frames=N_FRAMES_AGGR, max_dist_px=MAX_MATCH_DIST_PX,
        hough_kwargs=dict(canny_low=60, canny_high=180,
                          hough_threshold=150, min_line_len=200, max_line_gap=30,
                          use_floor_mask=True),
        save_debug_dir=None,
    )
    n_total = int(vis_court.sum())
    keep = vis_court.sum(axis=1) >= 2
    print(f'[{tag}] match: {n_total}/{vis_court.size}, punti utili: {int(keep.sum())}/{len(X_world)}')
    X_f = X_world[keep]
    obs_safe = np.nan_to_num(obs_court[keep], nan=0.0)
    vis_f = vis_court[keep]
    cams_prior_list = [cams_prior_dict[c] for c in cam_ids]
    new_extr, _, rmse_i, rmse_f = run_bundle_adjustment(
        cams_prior_list, X_f, obs_safe, vis_f,
        verbose=False, loss=BA_LOSS, f_scale=BA_F_SCALE)
    cams_ref = [Camera(c.cam_id, c.K, c.dist, rv, tv)
                for c, (rv, tv) in zip(cams_prior_list, new_extr)]
    print(f'[{tag}] court RMSE {rmse_i:.3f} -> {rmse_f:.3f} px')
    return cams_ref, dict(n_matched=n_total, n_points=int(keep.sum()),
                          rmse_init=float(rmse_i), rmse_final=float(rmse_f))


def main():
    cam_ids = ('cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5', 'cam_7')
    cams_v2 = load_cameras(PROJECT / 'data' / 'cameras',
                           cam_ids=cam_ids, version='camera_config_v2')
    X_world = build_court_3d_extended()
    video_paths = find_video_paths(cam_ids)
    obs_players, players = load_annotations(
        PROJECT / 'data' / 'annotations' / '_annotations.coco.json')

    # Baseline v2
    mpjpe_v2 = average_mpjpe(triangulate_all(cams_v2, obs_players, players))
    loo_v2 = average_loo(cams_v2, obs_players, players)
    print(f'v2 baseline: MPJPE fit={mpjpe_v2:.2f} px, LOO={loo_v2:.2f} px')

    # BA iterativo: round di (re-detection con prior corrente → BA) finché
    # il numero di match aumenta (schema EM/ICP-like). Max 4 round.
    summary = {'baseline_v2': {'mpjpe_fit_px': mpjpe_v2, 'mpjpe_loo_px': loo_v2}}
    cams_cur = cams_v2
    prev_matched = -1
    for r in range(1, 5):
        cams_ref, s = detect_and_ba(cams_cur, cam_ids, X_world, video_paths, f'round{r}')
        cams_ref_d = {c.cam_id: c for c in cams_ref}
        mpjpe_r = average_mpjpe(triangulate_all(cams_ref_d, obs_players, players))
        loo_r = average_loo(cams_ref_d, obs_players, players)
        print(f'post-BA r{r}:  MPJPE fit={mpjpe_r:.2f} px, LOO={loo_r:.2f} px')
        summary[f'round{r}'] = {**s, 'mpjpe_fit_px': mpjpe_r, 'mpjpe_loo_px': loo_r}
        if s['n_matched'] <= prev_matched:
            print(f'Plateau: match {s["n_matched"]} <= {prev_matched}, stop.')
            break
        prev_matched = s['n_matched']
        cams_cur = cams_ref_d
    with open(OUT_DIR / 'eval_best_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n=== SUMMARY ===')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
