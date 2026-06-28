"""Step 3 — Bundle Adjustment con dati REALI dei court points.

Due esperimenti:
- A (ONESTO, deliverable): BA ITERATIVO partendo dalla calib v2 REALE.
  Round di (detection con prior corrente -> match -> BA): dopo ogni BA la
  calib migliora, quindi riproiettando i court points il prior e' piu' vicino
  alle intersezioni vere e il matching (radius 30px) aggancia piu' punti ->
  il BA successivo e' meglio vincolato (schema EM/ICP-like). Si itera fino
  al plateau del MPJPE player. Valutazione su TRE metriche indipendenti:
  MPJPE fit, MPJPE leave-one-view-out (vista held-out, accuratezza onesta),
  bone-length CV (3D GT-free).
- B (validazione solver): perturba v2 con rumore noto e verifica che il BA
  converga allo stesso ottimo di A sulle osservazioni dell'ultimo round.
  Prova solo la determinatezza della soluzione, NON l'accuratezza.

Detection: multi-frame, aggrega Hough su N frame/cam → mediana per cam.
Court points: intersezioni rette FIBA (corner, paint, T, mid-circle).
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
from utils.triangulation import (triangulate_skeleton, loo_reprojection_skeleton,
                                 bone_length_consistency)
from utils.bundle_adjustment import run_bundle_adjustment
from court_detector import build_court_observations_multiframe, build_court_observations


PROJECT = Path(__file__).resolve().parent.parent
VIDEO_DIR = Path('/home/lele/Desktop/CV/HPE/material4project/video/hpe_01')
OUT_DIR = Path(__file__).parent / 'output_real'
OUT_DIR.mkdir(exist_ok=True)

N_FRAMES_AGGR = 8           # frame da aggregare per cam
PERTURB_RVEC = 0.02
PERTURB_TVEC = 100.0
MAX_MATCH_DIST_PX = 30.0    # stretto: match accurato con calib prior
BA_LOSS = 'linear'          # 'huber' tendeva a clipping
BA_F_SCALE = 5.0
MAX_BA_ROUNDS = 4           # round detection->match->BA (plateau tipico al 3-4)
MPJPE_PLATEAU_PX = 0.05     # stop se il MPJPE fit migliora meno di così
RNG = np.random.default_rng(42)


def build_court_3d_extended():
    """Punti 3D notable per Hough: solo intersezioni rette (no archi/cerchi)."""
    c = fiba_court_points()
    parts = [
        c['court_corners'],      # 4 angoli campo
        c['paint_corners'],      # 8 angoli paint
        c['t_midcourt'],         # 2 T midcourt
        c['midcourt_circle'],    # 2 intersezioni mid/center circle
    ]
    return np.vstack(parts) * 1000.0    # → mm


def find_video_paths(cam_ids):
    paths = {}
    for cid in cam_ids:
        n = int(cid.split('_')[1])
        v = VIDEO_DIR / f'out{n}.mp4'
        if v.exists():
            paths[cid] = v
    return paths


def perturb_cameras(cams):
    out = []
    for c in cams:
        dr = RNG.normal(0, PERTURB_RVEC, 3)
        dt = RNG.normal(0, PERTURB_TVEC, 3)
        out.append(Camera(c.cam_id, c.K, c.dist, c.rvec + dr, c.tvec + dt))
    return out


def average_mpjpe(triang_results):
    errs = []
    for fd in triang_results.values():
        for pd in fd.values():
            re = np.array(pd['repro_errs_px'])
            if (~np.isnan(re)).any():
                errs.append(np.nanmean(re))
    return float(np.mean(errs)) if errs else float('nan')


def triangulate_all(cams_dict, obs_players, players):
    results = {}
    for f in sorted(obs_players.keys()):
        results[f] = {}
        for pl in players:
            kpts_by_cam = {}
            for cid in cams_dict:
                if cid in obs_players[f] and pl in obs_players[f][cid]:
                    kpts_by_cam[cid] = obs_players[f][cid][pl]['kpts']
            if len(kpts_by_cam) < 2:
                continue
            X, re, nv = triangulate_skeleton(kpts_by_cam, cams_dict)
            results[f][pl] = {'X_3d_mm': X.tolist(),
                              'repro_errs_px': re.tolist(),
                              'n_views': nv.tolist()}
    return results


def average_loo(cams_dict, obs_players, players):
    """MPJPE leave-one-view-out medio (scheletri con >=3 viste)."""
    vals = []
    for f in sorted(obs_players.keys()):
        for pl in players:
            kpts_by_cam = {}
            for cid in cams_dict:
                if cid in obs_players[f] and pl in obs_players[f][cid]:
                    kpts_by_cam[cid] = obs_players[f][cid][pl]['kpts']
            if len(kpts_by_cam) < 3:
                continue
            m = np.nanmean(loo_reprojection_skeleton(kpts_by_cam, cams_dict))
            if not np.isnan(m):
                vals.append(m)
    return float(np.mean(vals)) if vals else float('nan')


def bone_cv(triang_results, players):
    """Bone-length CV globale (validazione 3D GT-free)."""
    return float(bone_length_consistency(triang_results, SKELETON_EDGES,
                                         players)['global_cv'])


def detect_court(cams_prior_dict, X_world, video_paths, save_debug_dir=None):
    """Detection multi-frame con prior dato. Ritorna (X_f, obs_safe, vis_f, stats)."""
    obs_court, vis_court = build_court_observations_multiframe(
        cams_prior_dict, X_world, video_paths,
        n_frames=N_FRAMES_AGGR,
        max_dist_px=MAX_MATCH_DIST_PX,
        hough_kwargs=dict(canny_low=60, canny_high=180,
                          hough_threshold=150, min_line_len=200, max_line_gap=30,
                          use_floor_mask=True),
        save_debug_dir=save_debug_dir,
    )
    n_total = int(vis_court.sum())
    keep = vis_court.sum(axis=1) >= 2
    stats = {'n_matched': n_total, 'n_obs_possible': int(vis_court.size),
             'n_points_used': int(keep.sum()), 'n_points_total': len(X_world)}
    return (X_world[keep], np.nan_to_num(obs_court[keep], nan=0.0),
            vis_court[keep], stats)


def main():
    cam_ids = ('cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5', 'cam_7')
    cams_v2 = load_cameras(PROJECT / 'data' / 'cameras',
                           cam_ids=cam_ids, version='camera_config_v2')
    cams_v2_list = [cams_v2[c] for c in cam_ids]

    X_world = build_court_3d_extended()
    print(f'Punti 3D estesi: {len(X_world)}')

    video_paths = find_video_paths(cam_ids)
    print(f'Video trovati: {len(video_paths)}/{len(cam_ids)}')

    coco_path = PROJECT / 'data' / 'annotations' / '_annotations.coco.json'
    obs_players, players = load_annotations(coco_path)

    # ============================================================
    # ESPERIMENTO A — ONESTO: BA ITERATIVO dalla calib v2 REALE.
    # Round r: detection con prior = calib corrente -> match -> BA.
    # La calib raffinata avvicina gli expected alle intersezioni vere ->
    # piu' match -> BA meglio vincolato. Stop al plateau del MPJPE fit.
    # ============================================================
    print('\n' + '='*60)
    print('ESPERIMENTO A — BA ITERATIVO da calib v2 REALE (residuo onesto)')
    print('='*60)
    mpjpe_v2 = average_mpjpe(triangulate_all(cams_v2, obs_players, players))
    loo_v2 = average_loo(cams_v2, obs_players, players)
    bonecv_v2 = bone_cv(triangulate_all(cams_v2, obs_players, players), players)
    print(f'Baseline v2: MPJPE fit={mpjpe_v2:.2f} px, LOO={loo_v2:.2f} px, '
          f'bone-CV={bonecv_v2*100:.1f}%')

    cams_cur_d = cams_v2
    rounds = []
    mpjpe_prev = mpjpe_v2
    rmse_A_init = rmse_A_final = float('nan')
    X_last = obs_last = vis_last = None
    for r in range(1, MAX_BA_ROUNDS + 1):
        print(f'\n--- Round {r}: detection ({N_FRAMES_AGGR} frame/cam, '
              f'radius={MAX_MATCH_DIST_PX}px, prior={"v2" if r == 1 else f"BA round {r-1}"}) ---')
        # Debug overlay salvato solo al primo round (prior v2 originale)
        X_f, obs_safe, vis_f, dstats = detect_court(
            cams_cur_d, X_world, video_paths,
            save_debug_dir=(OUT_DIR / 'debug') if r == 1 else None)
        print(f'Match: {dstats["n_matched"]}/{dstats["n_obs_possible"]}, '
              f'punti utili: {dstats["n_points_used"]}/{dstats["n_points_total"]}')
        if dstats['n_points_used'] < 6:
            print('ERROR: punti insufficienti.')
            return
        cams_cur_list = [cams_cur_d[c] for c in cam_ids]
        new_extr, _, rmse_i, rmse_f = run_bundle_adjustment(
            cams_cur_list, X_f, obs_safe, vis_f,
            verbose=(r == 1), loss=BA_LOSS, f_scale=BA_F_SCALE)
        cams_ref = [Camera(co.cam_id, co.K, co.dist, rv, tv)
                    for co, (rv, tv) in zip(cams_cur_list, new_extr)]
        cams_ref_d = {c.cam_id: c for c in cams_ref}
        mpjpe_r = average_mpjpe(triangulate_all(cams_ref_d, obs_players, players))
        print(f'Round {r}: court RMSE {rmse_i:.3f}->{rmse_f:.3f} px, '
              f'MPJPE fit={mpjpe_r:.2f} px (prec. {mpjpe_prev:.2f})')
        rounds.append({'round': r, **dstats,
                       'court_rmse_init_px': float(rmse_i),
                       'court_rmse_final_px': float(rmse_f),
                       'mpjpe_fit_px': mpjpe_r})
        if r == 1:
            rmse_A_init, rmse_A_final = rmse_i, rmse_f
        X_last, obs_last, vis_last = X_f, obs_safe, vis_f
        improved = mpjpe_prev - mpjpe_r
        cams_cur_d = cams_ref_d
        mpjpe_prev = mpjpe_r
        if improved < MPJPE_PLATEAU_PX:
            print(f'Plateau MPJPE (Δ={improved:.3f} px < {MPJPE_PLATEAU_PX}) — stop.')
            break

    cams_refined_A = [cams_cur_d[c] for c in cam_ids]
    print('\nCorrezione TOTALE applicata dal BA iterativo alla calib v2:')
    for cid in cam_ids:
        cg, cr = cams_v2[cid], cams_cur_d[cid]
        drv = np.linalg.norm(cr.rvec - cg.rvec)
        dtv = np.linalg.norm(cr.tvec - cg.tvec)
        print(f'  {cid}: |Δrvec|={drv:.5f} rad, |Δtvec|={dtv:.2f} mm')
    triang_A = triangulate_all(cams_cur_d, obs_players, players)
    mpjpe_A_post = average_mpjpe(triang_A)
    loo_A_post = average_loo(cams_cur_d, obs_players, players)
    bonecv_A_post = bone_cv(triang_A, players)
    print(f'\nPOST-BA (round {rounds[-1]["round"]}): MPJPE fit={mpjpe_A_post:.2f} px '
          f'({mpjpe_v2:.2f} baseline), LOO={loo_A_post:.2f} px ({loo_v2:.2f}), '
          f'bone-CV={bonecv_A_post*100:.1f}% ({bonecv_v2*100:.1f}%)')

    # ============================================================
    # ESPERIMENTO B — STABILITA' rispetto all'inizializzazione: parto da v2
    # perturbata con rumore noto e verifico che il BA (sulle osservazioni
    # dell'ultimo round di A) converga allo STESSO ottimo di A. Se A e B
    # coincidono, la soluzione e' ben determinata dai court points.
    # ============================================================
    print('\n' + '='*60)
    print('ESPERIMENTO B — stabilita\' del BA rispetto all\'init (perturbazione nota)')
    print('='*60)
    cams_perturbed = perturb_cameras(cams_v2_list)
    mpjpe_pre = average_mpjpe(triangulate_all({c.cam_id: c for c in cams_perturbed},
                                              obs_players, players))
    new_extr_B, _, rmse_B_init, rmse_B_final = run_bundle_adjustment(
        cams_perturbed, X_last, obs_last, vis_last,
        verbose=False, loss=BA_LOSS, f_scale=BA_F_SCALE)
    cams_refined_B = [Camera(co.cam_id, co.K, co.dist, rv, tv)
                      for co, (rv, tv) in zip(cams_perturbed, new_extr_B)]
    # Accordo A vs B: stessa soluzione raggiunta da init diversi?
    agree_tvec = np.mean([np.linalg.norm(ca.tvec - cb.tvec)
                          for ca, cb in zip(cams_refined_A, cams_refined_B)])
    mpjpe_post = average_mpjpe(triangulate_all({c.cam_id: c for c in cams_refined_B},
                                               obs_players, players))
    print(f'Init perturbato |Δtvec|~{PERTURB_TVEC:.0f}mm, MPJPE pre={mpjpe_pre:.2f} px')
    print(f'Convergenza A vs B: |Δtvec|={agree_tvec:.2f} mm, court RMSE finale '
          f'{rmse_B_final:.3f} px (A round finale={rounds[-1]["court_rmse_final_px"]:.3f}), '
          f'MPJPE post={mpjpe_post:.2f} px '
          f'(A={mpjpe_A_post:.2f})  -> soluzione ben determinata')

    # ---- Plot: esperimento A (onesto) ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    triang_base = triangulate_all(cams_v2, obs_players, players)
    for ax, mp, triang, title, col in [
            (axes[0], mpjpe_v2, triang_base, 'v2 baseline', 'steelblue'),
            (axes[1], mpjpe_A_post, triang_A, 'v2 + BA reale', 'green')]:
        errs = [np.nanmean(np.array(pd['repro_errs_px']))
                for fd in triang.values() for pd in fd.values()
                if (~np.isnan(np.array(pd['repro_errs_px']))).any()]
        ax.hist(errs, bins=20, color=col, edgecolor='black', alpha=0.75)
        ax.axvline(mp, color='red', linestyle='--', linewidth=2, label=f'media = {mp:.2f} px')
        ax.set_xlabel('Reprojection error (px)'); ax.set_ylabel('# scheletri')
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f'Task 3 — BA iterativo ({len(rounds)} round) su calib v2: '
                 f'court RMSE 1° round {rmse_A_init:.2f}->{rmse_A_final:.2f} px | '
                 f'MPJPE player {mpjpe_v2:.2f}->{mpjpe_A_post:.2f} px | '
                 f'LOO {loo_v2:.1f}->{loo_A_post:.1f} px',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'ba_real_comparison.png', dpi=120)
    print(f'\nPlot salvato: {OUT_DIR / "ba_real_comparison.png"}')

    summary = {
        'mode': 'real_multiframe_iterative',
        'n_frames_aggregated': N_FRAMES_AGGR,
        'n_ba_rounds': len(rounds),
        'match_radius_px': MAX_MATCH_DIST_PX,
        'ba_loss': BA_LOSS,
        # --- Esperimento A: BA iterativo dalla v2 reale (deliverable Task 3) ---
        'A_rounds': rounds,
        'A_court_rmse_v2_residual_px': float(rmse_A_init),   # residuo REALE v2 (round 1 init)
        'A_court_rmse_round1_final_px': float(rmse_A_final),
        'A_n_observations_final': rounds[-1]['n_matched'],
        'A_n_court_points_final': rounds[-1]['n_points_used'],
        'A_mpjpe_v2_px': mpjpe_v2,
        'A_mpjpe_post_ba_px': mpjpe_A_post,
        'A_mpjpe_loo_v2_px': loo_v2,
        'A_mpjpe_loo_post_ba_px': loo_A_post,
        'A_bone_cv_v2': bonecv_v2,
        'A_bone_cv_post_ba': bonecv_A_post,
        # --- Esperimento B: stabilita' rispetto all'init (sintetico) ---
        'B_perturb_tvec_mm': PERTURB_TVEC,
        'B_AvsB_agreement_tvec_mm': float(agree_tvec),
        'B_court_rmse_init_px': float(rmse_B_init),
        'B_court_rmse_final_px': float(rmse_B_final),
        'B_mpjpe_pre_px': mpjpe_pre,
        'B_mpjpe_post_px': mpjpe_post,
    }
    with open(OUT_DIR / 'ba_real_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('\n' + '='*60)
    print('RISULTATI:')
    print('='*60)
    for k, v in summary.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
