"""Step 3 — Bundle Adjustment.

Approccio:
1. Carica le 3D coordinate dei punti notevoli del campo (FIBA + corners + canestri)
2. Proietta questi punti con la calibrazione v2 attuale per OGNI camera → punti 2D "GT"
3. Aggiunge rumore + perturba leggermente le estrinseche (simula calibrazione imperfetta)
4. BA ottimizza rvec/tvec per minimizzare reprojection error sui court points
5. Rifa la triangolazione step 2 con le camere raffinate
6. Confronta MPJPE prima vs dopo BA

NOTA: per usare i punti REALI di dump (file con 2D detection di court lines),
basta sostituire il blocco "genera observations" con il loading dei dump
+ matching ai 3D corretti.
"""
import sys
from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.camera import Camera, load_cameras
from utils.coco_utils import load_annotations, SKELETON_EDGES
from utils.court import fiba_court_points
from utils.triangulation import triangulate_skeleton
from utils.bundle_adjustment import run_bundle_adjustment


PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).parent / 'output'
OUT_DIR.mkdir(exist_ok=True)

NOISE_PX = 2.0           # rumore osservazioni 2D (pixel)
PERTURB_RVEC = 0.02      # perturbazione iniziale rvec (rad)
PERTURB_TVEC = 100.0     # perturbazione iniziale tvec (mm)
RNG = np.random.default_rng(42)


def build_court_3d_points():
    """Tutti i punti notevoli del campo in mm (la calibrazione è in mm)."""
    court = fiba_court_points()
    # Raccogliamo i punti più "vincolanti" per BA
    parts = []
    parts.append(court['court_corners'])     # 4
    parts.append(court['paint_corners'])     # 8
    parts.append(court['t_midcourt'])        # 2
    parts.append(court['midcourt_circle'])   # 2
    parts.append(court['center_circle'])     # 24
    parts.append(court['ft_circle_left'])    # 24
    parts.append(court['ft_circle_right'])   # 24
    parts.append(court['three_pt_left'])     # 24
    parts.append(court['three_pt_right'])    # 24
    parts.append(court['perimeter'])         # 60
    X_m = np.vstack(parts)                   # in metri
    return X_m * 1000.0                      # → mm


def synthetic_observations(cameras_gt, X_world_mm, image_size=(3840, 2160),
                            noise_px=2.0):
    """Genera osservazioni 2D 'misurate' proiettando con le GT camere + rumore.

    Returns:
        obs: (N, M, 2)
        vis: (N, M) bool — True se proiezione cade dentro l'immagine
    """
    N = len(X_world_mm)
    M = len(cameras_gt)
    obs = np.zeros((N, M, 2))
    vis = np.zeros((N, M), dtype=bool)
    W, H = image_size
    for j, cam in enumerate(cameras_gt):
        proj, _ = cv2.projectPoints(X_world_mm, cam.rvec, cam.tvec, cam.K, cam.dist)
        proj = proj.reshape(-1, 2)
        # Rumore gaussiano
        noise = RNG.normal(0, noise_px, proj.shape)
        obs[:, j] = proj + noise
        # Visibility: dentro l'immagine
        vis[:, j] = (proj[:,0] >= 0) & (proj[:,0] < W) & (proj[:,1] >= 0) & (proj[:,1] < H)
    return obs, vis


def perturb_cameras(cameras_gt):
    """Ritorna nuove Camera con rvec/tvec perturbati (input per BA)."""
    perturbed = []
    for cam in cameras_gt:
        d_rvec = RNG.normal(0, PERTURB_RVEC, 3)
        d_tvec = RNG.normal(0, PERTURB_TVEC, 3)
        new_rvec = cam.rvec + d_rvec
        new_tvec = cam.tvec + d_tvec
        perturbed.append(Camera(cam.cam_id, cam.K, cam.dist, new_rvec, new_tvec))
    return perturbed


def average_mpjpe(triang_results):
    """Media degli errori di riproiezione per giocatore/frame."""
    errs = []
    for frame_data in triang_results.values():
        for player_data in frame_data.values():
            re = np.array(player_data['repro_errs_px'])
            if (~np.isnan(re)).any():
                errs.append(np.nanmean(re))
    return np.mean(errs) if errs else np.nan


def triangulate_all(cameras_dict, obs, players):
    """Triangola tutti gli scheletri usando un dict di Camera."""
    results = {}
    for frame_idx in sorted(obs.keys()):
        results[frame_idx] = {}
        for player in players:
            kpts_by_cam = {}
            for cam_id in cameras_dict:
                if cam_id in obs[frame_idx] and player in obs[frame_idx][cam_id]:
                    kpts_by_cam[cam_id] = obs[frame_idx][cam_id][player]['kpts']
            if len(kpts_by_cam) < 2:
                continue
            X_3d, repro_errs, n_views = triangulate_skeleton(kpts_by_cam, cameras_dict)
            results[frame_idx][player] = {
                'X_3d_mm': X_3d.tolist(),
                'repro_errs_px': repro_errs.tolist(),
                'n_views': n_views.tolist(),
            }
    return results


def main():
    # 1. Carica calibrazione GT (v2) e annotazioni player
    cam_ids = ('cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5', 'cam_7')
    cams_gt = load_cameras(PROJECT / 'data' / 'cameras',
                            cam_ids=cam_ids, version='camera_config_v2')
    cams_gt_list = [cams_gt[c] for c in cam_ids]
    print(f'GT cameras: {[c.cam_id for c in cams_gt_list]}')

    # 2. Punti 3D del campo
    X_world = build_court_3d_points()
    print(f'Punti campo 3D: {len(X_world)}')

    # 3. Genera osservazioni 2D sintetiche (con rumore)
    obs_2d, vis = synthetic_observations(cams_gt_list, X_world, noise_px=NOISE_PX)
    print(f'Osservazioni 2D: {obs_2d.shape}, visibili: {int(vis.sum())}/{vis.size}')

    # 4. Perturba i parametri delle camere (simula calibrazione imperfetta)
    cams_perturbed = perturb_cameras(cams_gt_list)
    print('\nPerturbazione iniziale applicata:')
    for cam_gt, cam_p in zip(cams_gt_list, cams_perturbed):
        drv = np.linalg.norm(cam_p.rvec - cam_gt.rvec)
        dtv = np.linalg.norm(cam_p.tvec - cam_gt.tvec)
        print(f'  {cam_gt.cam_id}: |Δrvec|={drv:.4f} rad, |Δtvec|={dtv:.1f} mm')

    # 5. Triangolazione PRE-BA (con camere perturbate)
    print('\n--- Step 2 PRE-BA (camere perturbate) ---')
    coco_path = PROJECT / 'data' / 'annotations' / '_annotations.coco.json'
    obs_players, players = load_annotations(coco_path)
    cams_perturbed_dict = {c.cam_id: c for c in cams_perturbed}
    triang_pre = triangulate_all(cams_perturbed_dict, obs_players, players)
    mpjpe_pre = average_mpjpe(triang_pre)
    print(f'Reprojection error medio (PRE-BA): {mpjpe_pre:.2f} px')

    # 6. Bundle Adjustment
    print('\n--- Bundle Adjustment ---')
    new_extr, result, initial_rmse, final_rmse = run_bundle_adjustment(
        cams_perturbed, X_world, obs_2d, vis, verbose=True)

    # 7. Costruisci nuove camere raffinate
    cams_refined = []
    for cam_orig, (rv, tv) in zip(cams_perturbed, new_extr):
        cams_refined.append(Camera(cam_orig.cam_id, cam_orig.K, cam_orig.dist, rv, tv))

    print('\nConfronto con GT dopo BA:')
    for cam_gt, cam_r in zip(cams_gt_list, cams_refined):
        drv = np.linalg.norm(cam_r.rvec - cam_gt.rvec)
        dtv = np.linalg.norm(cam_r.tvec - cam_gt.tvec)
        print(f'  {cam_gt.cam_id}: |Δrvec|={drv:.5f} rad, |Δtvec|={dtv:.2f} mm')

    # 8. Triangolazione POST-BA (con camere raffinate)
    print('\n--- Step 2 POST-BA ---')
    cams_refined_dict = {c.cam_id: c for c in cams_refined}
    triang_post = triangulate_all(cams_refined_dict, obs_players, players)
    mpjpe_post = average_mpjpe(triang_post)
    print(f'Reprojection error medio (POST-BA): {mpjpe_post:.2f} px')

    # 9. Plot confronto
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, mp, title in [(axes[0], mpjpe_pre, 'PRE-BA'), (axes[1], mpjpe_post, 'POST-BA')]:
        errs = []
        for frame_data in (triang_pre if 'PRE' in title else triang_post).values():
            for player_data in frame_data.values():
                re = np.array(player_data['repro_errs_px'])
                if (~np.isnan(re)).any():
                    errs.append(np.nanmean(re))
        ax.hist(errs, bins=20, color='steelblue' if 'PRE' in title else 'green',
                edgecolor='black', alpha=0.75)
        ax.axvline(mp, color='red', linestyle='--', linewidth=2,
                    label=f'media = {mp:.2f} px')
        ax.set_xlabel('Reprojection error medio (px)')
        ax.set_ylabel('# scheletri (player × frame)')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f'Bundle Adjustment — riduzione MPJPE: {mpjpe_pre:.2f} → {mpjpe_post:.2f} px '
                 f'(-{(1-mpjpe_post/mpjpe_pre)*100:.1f}%)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = OUT_DIR / 'ba_comparison.png'
    plt.savefig(out_path, dpi=120)
    print(f'\nPlot salvato: {out_path}')

    # 10. Sommario finale
    summary = {
        'noise_px': NOISE_PX,
        'perturb_rvec': PERTURB_RVEC,
        'perturb_tvec_mm': PERTURB_TVEC,
        'n_court_points': int(len(X_world)),
        'n_observations': int(vis.sum()),
        'ba_initial_rmse_px': float(initial_rmse),
        'ba_final_rmse_px': float(final_rmse),
        'ba_reduction_pct': float((1 - final_rmse/initial_rmse)*100),
        'mpjpe_pre_ba_px': float(mpjpe_pre),
        'mpjpe_post_ba_px': float(mpjpe_post),
        'mpjpe_reduction_pct': float((1 - mpjpe_post/mpjpe_pre)*100),
    }
    with open(OUT_DIR / 'ba_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Sommario salvato: {OUT_DIR / "ba_summary.json"}')
    print('\n' + '='*60)
    print('RISULTATI BA:')
    print('='*60)
    for k, v in summary.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
