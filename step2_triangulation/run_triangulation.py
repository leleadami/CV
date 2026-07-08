"""Step 2 — Triangolazione DLT multi-view per tutti i frame e giocatori.

Pipeline:
  1. Carica annotazioni COCO (5 frame × 6 camere × 10 giocatori)
  2. Carica calibrazioni v2 per cam_1,2,3,4,5,7
  3. Per ogni (frame, player), triangola i 18 keypoint
  4. Calcola reprojection error per ogni vista
  5. Salva risultati 3D e plot

Output:
  - results_3d.json: scheletri 3D per ogni (frame, player)
  - per_frame_player_plots/*.png: visualizzazioni
"""
import sys
from pathlib import Path
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.camera import load_cameras
from utils.coco_utils import load_annotations, SKELETON_EDGES, KEYPOINT_NAMES, N_KPTS
from utils.triangulation import (triangulate_skeleton, loo_reprojection_skeleton,
                                  bone_length_consistency)
from utils.court import fiba_court_points, COURT_COLORS


PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).parent / 'output'
OUT_DIR.mkdir(exist_ok=True)
IMG_DIR = PROJECT / 'data' / 'images'


def find_frame_image(cam_id, frame_idx):
    """Trova l'immagine Roboflow di (cam, frame): out{C}_frame_{F:04d}_png.rf.*.png"""
    n = cam_id.split('_')[1]
    hits = sorted(IMG_DIR.glob(f'out{n}_frame_{frame_idx:04d}_png.rf.*.png'))
    return hits[0] if hits else None


def main():
    # 1. Carica camere v2 per le 6 cam usate da hpe_01
    cams = load_cameras(PROJECT / 'data' / 'cameras',
                        cam_ids=('cam_1','cam_2','cam_3','cam_4','cam_5','cam_7'),
                        version='camera_config_v2')
    print(f'Camere caricate: {list(cams.keys())}')

    # 2. Carica annotazioni
    coco = PROJECT / 'data' / 'annotations' / '_annotations.coco.json'
    obs, players = load_annotations(coco)
    print(f'Players: {players}')
    print(f'Frame: {sorted(obs.keys())}')

    # 3. Triangola tutto
    results = {}
    summary = []
    for frame_idx in sorted(obs.keys()):
        results[frame_idx] = {}
        for player in players:
            kpts_by_cam = {}
            for cam_id in cams:
                if cam_id in obs[frame_idx] and player in obs[frame_idx][cam_id]:
                    kpts_by_cam[cam_id] = obs[frame_idx][cam_id][player]['kpts']
            n_cams_with_player = len(kpts_by_cam)
            if n_cams_with_player < 2:
                continue
            X_3d, repro_errs, n_views = triangulate_skeleton(kpts_by_cam, cams)
            loo_errs = loo_reprojection_skeleton(kpts_by_cam, cams)
            valid = ~np.isnan(X_3d[:, 0])
            results[frame_idx][player] = {
                'X_3d_mm': X_3d.tolist(),
                'n_views': n_views.tolist(),
                'repro_errs_px': repro_errs.tolist(),
                'loo_errs_px': loo_errs.tolist(),
            }
            mean_repro = np.nanmean(repro_errs)
            mean_loo = np.nanmean(loo_errs)
            summary.append({
                'frame': frame_idx, 'player': player,
                'n_cams': n_cams_with_player,
                'n_kpts_triangulated': int(valid.sum()),
                'mean_repro_err_px': float(mean_repro) if not np.isnan(mean_repro) else None,
                'mean_loo_err_px': float(mean_loo) if not np.isnan(mean_loo) else None,
            })

    # 4. Stampa sommario
    print()
    print('='*92)
    print(f'{"frame":>5} {"player":>10} {"#cams":>6} {"#kpts":>6} '
          f'{"repro err (px)":>16} {"LOO err (px)":>16}')
    print('-'*92)
    for s in summary:
        e = f'{s["mean_repro_err_px"]:.2f}' if s["mean_repro_err_px"] else 'n/a'
        lo = f'{s["mean_loo_err_px"]:.2f}' if s["mean_loo_err_px"] else 'n/a'
        print(f'{s["frame"]:>5} {s["player"]:>10} {s["n_cams"]:>6} '
              f'{s["n_kpts_triangulated"]:>6} {e:>16} {lo:>16}')

    # Metriche aggregate
    rep_all = [s['mean_repro_err_px'] for s in summary if s['mean_repro_err_px']]
    loo_all = [s['mean_loo_err_px'] for s in summary if s['mean_loo_err_px']]
    bone = bone_length_consistency(results, SKELETON_EDGES, players)
    print('-'*92)
    print(f'MPJPE reprojection (fit)        : {np.mean(rep_all):.2f} px')
    print(f'MPJPE leave-one-view-out (gen.) : {np.mean(loo_all):.2f} px   '
          f'<- accuratezza onesta (vista held-out)')
    print(f'Bone-length CV medio (3D GT-free): {bone["global_cv"]*100:.1f}%   '
          f'<- coerenza fisica scheletri 3D')

    # 5. Salva risultati
    metrics = {
        'mpjpe_reproj_fit_px': float(np.mean(rep_all)),
        'mpjpe_loo_px': float(np.mean(loo_all)),
        'bone_length_global_cv': bone['global_cv'],
        'bone_length_per_player_cv': bone['per_player_cv'],
    }
    with open(OUT_DIR / 'results_3d.json', 'w') as f:
        json.dump({'results': results, 'summary': summary, 'metrics': metrics}, f, indent=2)
    print(f'\nRisultati salvati in {OUT_DIR / "results_3d.json"}')

    # 5b. Istogramma reprojection error PER CAMERA.
    # Ricalcolato esplicitamente (le colonne di repro_errs nei results dipendono
    # dalle cam che vedono ogni player → non allineabili). Per ogni (frame,player)
    # proietto X_3d in ciascuna camera che ha annotato il joint e misuro la L2.
    per_cam = {cid: [] for cid in cams}
    for f_idx in results:
        for pl, pdata in results[f_idx].items():
            X = np.array(pdata['X_3d_mm'])  # (18,3) mm
            for cid, cam in cams.items():
                if cid not in obs.get(f_idx, {}) or pl not in obs[f_idx][cid]:
                    continue
                kpts = obs[f_idx][cid][pl]['kpts']  # (18,3)
                for k in range(N_KPTS):
                    if np.isnan(X[k, 0]) or kpts[k, 2] < 1:
                        continue
                    proj = cam.project(X[k:k+1])[0]
                    per_cam[cid].append(float(np.linalg.norm(proj - kpts[k, :2])))

    cam_order = sorted(per_cam.keys())
    fig_e, axes_e = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax_e, cid in zip(axes_e.ravel(), cam_order):
        errs = np.array(per_cam[cid])
        if errs.size == 0:
            ax_e.set_title(f'{cid} — nessun dato'); continue
        med = np.median(errs)
        ax_e.hist(errs, bins=30, range=(0, 80), color='steelblue',
                  edgecolor='black', alpha=0.8)
        ax_e.axvline(med, color='red', linestyle='--', linewidth=2,
                     label=f'mediana={med:.1f}px\nN={errs.size}')
        ax_e.set_title(cid); ax_e.set_xlabel('Reprojection error (px)')
        ax_e.set_ylabel('# joint'); ax_e.legend(fontsize=8); ax_e.grid(alpha=0.3)
    glob_med = np.median([e for v in per_cam.values() for e in v])
    fig_e.suptitle(f'Reprojection error per camera (Task 2a) — mediana globale {glob_med:.1f} px',
                   fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'repro_errors.png', dpi=120)
    plt.close(fig_e)
    print(f'Plot salvato: {OUT_DIR / "repro_errors.png"}')

    # 6. Visualizzazione: tutti gli scheletri del frame 1
    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection='3d')

    # Disegna i punti del campo
    court = fiba_court_points()
    for name, pts in court.items():
        if name == 'corners':
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=40, marker='s', c='black',
                       edgecolor='yellow', linewidth=1, zorder=4)
        elif name in ('baskets', 'uwb_ceiling', 'hall_perimeter'):
            continue
        else:
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=4,
                       c=COURT_COLORS.get(name, 'gray'), alpha=0.5)

    # Disegna le camere con linea verticale a terra per disambiguare altezza
    for cam_id, cam in cams.items():
        C = cam.center / 1000.0  # m
        ax.scatter(C[0], C[1], C[2], s=200, c='red', marker='^', edgecolor='black')
        ax.plot([C[0], C[0]], [C[1], C[1]], [0, C[2]],
                color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax.scatter(C[0], C[1], 0, s=40, c='red', marker='x', alpha=0.4)
        ax.text(C[0], C[1], C[2]+0.5, f'{cam_id}\nZ={C[2]:.1f}m', fontsize=8, ha='center')

    # Disegna gli scheletri triangolati del frame 1
    frame_to_plot = 1
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, player in enumerate(players):
        if player not in results[frame_to_plot]:
            continue
        X = np.array(results[frame_to_plot][player]['X_3d_mm']) / 1000.0  # m
        valid = ~np.isnan(X[:, 0])
        ax.scatter(X[valid, 0], X[valid, 1], X[valid, 2], s=30, c=[colors[i]], label=player)
        for a, b in SKELETON_EDGES:
            if valid[a] and valid[b]:
                ax.plot([X[a,0], X[b,0]], [X[a,1], X[b,1]], [X[a,2], X[b,2]],
                        c=colors[i], linewidth=1.5)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(f'Scheletri 3D triangolati — frame {frame_to_plot}')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    # box_aspect proporzionale alle dimensioni del palasport (Z non schiacciato)
    ax.set_box_aspect([2.0, 1.2, 1.0])
    ax.set_zlim(0, 8)
    ax.view_init(elev=22, azim=-55)   # vista più alta riduce confusione XY-Z
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'skeletons_3d_frame{frame_to_plot}.png', dpi=140)
    plt.close(fig)
    print(f'Plot salvato: {OUT_DIR / f"skeletons_3d_frame{frame_to_plot}.png"}')

    # 7. Overlay riproiezione: scheletri 3D riproiettati sui frame reali.
    # Verifica qualitativa che il 3D sia nel posto giusto in immagine:
    # x bianche = keypoint 2D annotati, colori = scheletro 3D riproiettato.
    # project() include la distorsione, coerente con le immagini raw Roboflow.
    fig_o, axes_o = plt.subplots(2, 3, figsize=(18, 9))
    for ax_o, cid in zip(axes_o.ravel(), cam_order):
        img_path = find_frame_image(cid, frame_to_plot)
        if img_path is None:
            ax_o.set_title(f'{cid} — immagine mancante'); ax_o.axis('off')
            continue
        # Roboflow salva JPEG con estensione .png: cv2 sniffa il contenuto
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        ax_o.imshow(img)
        cam = cams[cid]
        errs_cam = []
        for i, player in enumerate(players):
            if player not in results[frame_to_plot]:
                continue
            X = np.array(results[frame_to_plot][player]['X_3d_mm'])
            valid = ~np.isnan(X[:, 0])
            if not valid.any():
                continue
            proj = np.full((N_KPTS, 2), np.nan)
            proj[valid] = cam.project(X[valid])
            for a, b in SKELETON_EDGES:
                if valid[a] and valid[b]:
                    ax_o.plot([proj[a, 0], proj[b, 0]], [proj[a, 1], proj[b, 1]],
                              c=colors[i], linewidth=1.2)
            # Non tutti i player hanno 18 keypoint annotati in questa vista:
            # distinguo i joint 3D CON riscontro 2D (pieni + x bianca) da quelli
            # SENZA annotazione qui (cerchietti vuoti — nessun confronto possibile).
            vis = np.zeros(N_KPTS, dtype=bool)
            if cid in obs[frame_to_plot] and player in obs[frame_to_plot][cid]:
                kpts = obs[frame_to_plot][cid][player]['kpts']
                vis = kpts[:, 2] > 0
            both = vis & valid
            only3d = valid & ~vis
            ax_o.scatter(proj[both, 0], proj[both, 1], s=8, c=[colors[i]], zorder=3)
            ax_o.scatter(proj[only3d, 0], proj[only3d, 1], s=14, facecolors='none',
                         edgecolors=[colors[i]], linewidths=0.8, zorder=3)
            if vis.any():
                # x bianca = annotato con 3D valido; x grigia = annotato ma
                # joint non triangolato (niente proiezione da confrontare)
                orphan = vis & ~valid
                ax_o.scatter(kpts[both, 0], kpts[both, 1], s=16, marker='x',
                             c='white', linewidths=0.9, zorder=4)
                ax_o.scatter(kpts[orphan, 0], kpts[orphan, 1], s=16, marker='x',
                             c='gray', linewidths=0.9, zorder=4)
                if both.any():
                    errs_cam.extend(
                        np.linalg.norm(proj[both] - kpts[both, :2], axis=1))
        h, w = img.shape[:2]
        ax_o.set_xlim(0, w); ax_o.set_ylim(h, 0); ax_o.axis('off')
        med = np.median(errs_cam) if errs_cam else float('nan')
        ax_o.set_title(f'{cid} — err mediano {med:.1f} px', fontsize=10)
    fig_o.suptitle(f'Riproiezione scheletri 3D sui frame reali — frame {frame_to_plot} '
                   '(x bianca = annotato+3D, cerchio vuoto = 3D senza annotazione '
                   'in questa vista, x grigia = annotato senza 3D)',
                   fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'reproj_overlay_frame{frame_to_plot}.png', dpi=130)
    plt.close(fig_o)
    print(f'Plot salvato: {OUT_DIR / f"reproj_overlay_frame{frame_to_plot}.png"}')


if __name__ == '__main__':
    main()
