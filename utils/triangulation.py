"""DLT multi-view triangulation (Hartley-Zisserman, ed. slide 05 pp.34-36).

Per ogni vista j con matrice di proiezione P_j (3x4) e osservazione 2D (u_j, v_j):
    u_j = (P_j[0] · X) / (P_j[2] · X)
    v_j = (P_j[1] · X) / (P_j[2] · X)
Riarrangiando:
    (u_j · P_j[2] - P_j[0]) · X = 0
    (v_j · P_j[2] - P_j[1]) · X = 0

Per N viste si ottiene una matrice A di forma (2N, 4) e si risolve min ||A·X||
con SVD imponendo X[3]=1 (coordinata omogenea).
"""
from typing import Sequence
import numpy as np
from scipy.optimize import least_squares


def dlt_triangulate(pts2d: np.ndarray, P_list: Sequence[np.ndarray]) -> np.ndarray:
    """Triangola un singolo punto 3D da N viste.

    Args:
        pts2d: array (N, 2) di osservazioni 2D in pixel (già undistorte).
        P_list: lista di N matrici di proiezione 3x4.

    Returns:
        X: array (3,) con coordinate 3D mondo.
    """
    pts2d = np.asarray(pts2d, dtype=np.float64)
    N = len(P_list)
    assert pts2d.shape == (N, 2), f'Expected ({N},2), got {pts2d.shape}'

    A = np.zeros((2 * N, 4))
    for j, P in enumerate(P_list):
        u, v = pts2d[j]
        A[2*j]     = u * P[2] - P[0]
        A[2*j + 1] = v * P[2] - P[1]

    # SVD: X è l'autovettore destro associato al minimo valore singolare
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) < 1e-12:
        return np.array([np.nan, np.nan, np.nan])
    return X_h[:3] / X_h[3]


def refine_point_nonlinear(X0: np.ndarray, pts2d_orig: np.ndarray,
                           cameras: Sequence) -> np.ndarray:
    """Raffina un punto 3D minimizzando l'errore di riproiezione GEOMETRICO.

    Il DLT minimizza solo l'errore algebrico ||A·X||. Qui partiamo da quella
    stima e minimizziamo il vero errore di riproiezione in pixel (con distorsione)
    via Gauss-Newton (Levenberg-Marquardt), Hartley-Zisserman §12.3.

    Args:
        X0: stima iniziale 3D (3,) dal DLT.
        pts2d_orig: osservazioni 2D ORIGINALI (N,2), con distorsione.
        cameras: lista di N Camera (project() include distorsione).

    Returns:
        X: punto 3D raffinato (3,). Ritorna X0 se non valido.
    """
    if X0 is None or np.any(np.isnan(X0)):
        return X0
    pts2d_orig = np.asarray(pts2d_orig, dtype=np.float64)

    def residuals(X):
        r = []
        for cam, pt in zip(cameras, pts2d_orig):
            proj = cam.project(X.reshape(1, 3))[0]
            r.extend(proj - pt)
        return np.asarray(r)

    sol = least_squares(residuals, X0.astype(np.float64), method='lm',
                        xtol=1e-10, ftol=1e-10, max_nfev=100)
    return sol.x


def triangulate_robust(pts_und: np.ndarray, pts_orig: np.ndarray,
                       P_list: Sequence[np.ndarray], cams: Sequence,
                       reproj_thresh: float = 25.0, min_views: int = 2):
    """Triangolazione robusta: scarta le viste outlier (annotazione 2D errata).

    Una singola vista con keypoint sbagliato corrompe la SVD del DLT. Qui si
    rimuove iterativamente la vista col reprojection error peggiore finché tutte
    le viste residue sono sotto `reproj_thresh` (o si raggiunge `min_views`),
    ri-triangolando ogni volta. Equivale a un RANSAC greedy sulle viste.

    Returns:
        X: punto 3D robusto (3,)
        inliers: lista di indici (in pts_und/cams) delle viste tenute
    """
    idx = list(range(len(P_list)))

    def fit(ii):
        X = dlt_triangulate(pts_und[ii], [P_list[i] for i in ii])
        return refine_point_nonlinear(X, pts_orig[ii], [cams[i] for i in ii])

    X = fit(idx)
    while len(idx) > min_views:
        if X is None or np.any(np.isnan(X)):
            break
        errs = np.array([np.linalg.norm(cams[i].project(X.reshape(1, 3))[0] - pts_orig[i])
                         for i in idx])
        w = int(np.argmax(errs))
        if errs[w] <= reproj_thresh:
            break
        idx.pop(w)
        X = fit(idx)
    return X, idx


def reprojection_error(X: np.ndarray, pts2d: np.ndarray,
                       cameras: Sequence) -> np.ndarray:
    """Errore di riproiezione per ogni vista.

    Args:
        X: punto 3D (3,)
        pts2d: osservazioni 2D originali (N, 2), con distorsione
        cameras: lista di N oggetti Camera (con project() che include distorsione)

    Returns:
        errors: array (N,) con errore euclideo in pixel per vista.
    """
    errs = np.zeros(len(cameras))
    for j, cam in enumerate(cameras):
        proj = cam.project(X.reshape(1, 3))[0]
        errs[j] = np.linalg.norm(proj - pts2d[j])
    return errs


def triangulate_skeleton(kpts_by_cam: dict, cameras: dict,
                          min_views: int = 2,
                          vis_threshold: int = 1,
                          refine: bool = True,
                          robust: bool = False,
                          reproj_thresh: float = 25.0):
    """Triangola tutti i 18 keypoint di un giocatore.

    Args:
        kpts_by_cam: {cam_id: (18,3) array} keypoint 2D + visibility per camera
        cameras: {cam_id: Camera}
        min_views: minimo numero di viste valide per triangolare
        vis_threshold: visibility >= threshold per considerare valido (Roboflow usa 2)
        refine: se True, raffina ogni punto col Gauss-Newton non-lineare dopo il DLT.

    Returns:
        X_3d: (18, 3) array — np.nan per joint non triangolabili
        repro_errs: (18, N_cams_max) array — np.nan per viste non usate
        n_views: (18,) numero di viste usate per joint
    """
    n_kpts = 18
    cam_ids = sorted(kpts_by_cam.keys())
    X_3d = np.full((n_kpts, 3), np.nan)
    n_views = np.zeros(n_kpts, dtype=int)
    repro_errs = np.full((n_kpts, len(cam_ids)), np.nan)

    for k in range(n_kpts):
        pts_valid = []
        P_valid = []
        cams_valid = []
        cam_indices_valid = []
        for ci, cam_id in enumerate(cam_ids):
            kpts = kpts_by_cam[cam_id]
            x, y, v = kpts[k]
            if v < vis_threshold:
                continue
            cam = cameras[cam_id]
            # Undistort prima di triangolare
            pt_und = cam.undistort_points(np.array([[x, y]]))[0]
            pts_valid.append(pt_und)
            P_valid.append(cam.P)
            cams_valid.append(cam)
            cam_indices_valid.append(ci)

        if len(pts_valid) < min_views:
            continue

        pts_valid = np.array(pts_valid)
        # Punti ORIGINALI (con distorsione) per refinement e reprojection error
        pts_orig = []
        for ci, cam_id in enumerate(cam_ids):
            kpts = kpts_by_cam[cam_id]
            x, y, v = kpts[k]
            if v >= vis_threshold:
                pts_orig.append([x, y])
        pts_orig = np.array(pts_orig)

        if robust:
            # RANSAC greedy: scarta viste outlier, poi reproj solo sugli inlier
            X, inliers = triangulate_robust(pts_valid, pts_orig, P_valid, cams_valid,
                                            reproj_thresh=reproj_thresh, min_views=min_views)
            X_3d[k] = X
            n_views[k] = len(inliers)
            if X is not None and not np.any(np.isnan(X)):
                errs = reprojection_error(X, pts_orig[inliers],
                                          [cams_valid[i] for i in inliers])
                for jj, pos in enumerate(inliers):
                    repro_errs[k, cam_indices_valid[pos]] = errs[jj]
        else:
            X = dlt_triangulate(pts_valid, P_valid)
            # Refinement non-lineare: minimizza reprojection geometrico (post-DLT)
            if refine:
                X = refine_point_nonlinear(X, pts_orig, cams_valid)
            X_3d[k] = X
            n_views[k] = len(pts_valid)
            errs = reprojection_error(X, pts_orig, cams_valid)
            for idx, ci in enumerate(cam_indices_valid):
                repro_errs[k, ci] = errs[idx]

    return X_3d, repro_errs, n_views


def loo_reprojection_skeleton(kpts_by_cam: dict, cameras: dict,
                              vis_threshold: int = 1,
                              refine: bool = True) -> np.ndarray:
    """Errore di riproiezione LEAVE-ONE-VIEW-OUT per ogni joint.

    Per ogni joint visto in >=3 viste: si triangola usando N-1 viste e si
    riproietta nella vista esclusa, misurando l'errore sulla vista MAI usata
    per stimare il 3D. E' una accuratezza di GENERALIZZAZIONE, piu' onesta
    del reprojection error standard (che riusa le stesse viste = errore di fit).

    Returns:
        loo_errs: (18,) errore medio held-out in px per joint (np.nan se <3 viste).
    """
    n_kpts = 18
    cam_ids = sorted(kpts_by_cam.keys())
    loo_errs = np.full(n_kpts, np.nan)

    for k in range(n_kpts):
        views = []  # (cam, pt_undist, pt_orig, P)
        for cam_id in cam_ids:
            x, y, v = kpts_by_cam[cam_id][k]
            if v < vis_threshold:
                continue
            cam = cameras[cam_id]
            pt_und = cam.undistort_points(np.array([[x, y]]))[0]
            views.append((cam, pt_und, np.array([x, y]), cam.P))
        if len(views) < 3:
            continue

        errs = []
        for h in range(len(views)):
            train = [views[i] for i in range(len(views)) if i != h]
            pts_und = np.array([t[1] for t in train])
            P_list = [t[3] for t in train]
            X = dlt_triangulate(pts_und, P_list)
            if refine:
                X = refine_point_nonlinear(
                    X, np.array([t[2] for t in train]), [t[0] for t in train])
            if X is None or np.any(np.isnan(X)):
                continue
            held_cam, _, held_pt, _ = views[h]
            proj = held_cam.project(X.reshape(1, 3))[0]
            errs.append(np.linalg.norm(proj - held_pt))
        if errs:
            loo_errs[k] = float(np.mean(errs))

    return loo_errs


def bone_length_consistency(results: dict, edges, players) -> dict:
    """Consistenza 3D GT-free: una stessa "osso" deve avere lunghezza ~costante
    tra i frame. Misura il coefficiente di variazione (std/media) per osso.

    Una bassa CV indica triangolazione 3D fisicamente coerente, indipendente
    dal reprojection error 2D.

    Args:
        results: {frame: {player: {'X_3d_mm': (18,3) list}}}
        edges: lista di coppie (a, b) indici joint (SKELETON_EDGES)
        players: lista nomi giocatori

    Returns:
        {'per_player_cv': {player: mean_cv}, 'global_cv': float,
         'lengths_mm': {player: {edge: [len_per_frame]}}}
    """
    lengths = {pl: {e: [] for e in edges} for pl in players}
    for fd in results.values():
        for pl, pdata in fd.items():
            if pl not in lengths:
                continue
            X = np.array(pdata['X_3d_mm'])
            for (a, b) in edges:
                if np.isnan(X[a, 0]) or np.isnan(X[b, 0]):
                    continue
                lengths[pl][(a, b)].append(float(np.linalg.norm(X[a] - X[b])))

    per_player_cv = {}
    all_cv = []
    for pl in players:
        cvs = []
        for e, vals in lengths[pl].items():
            if len(vals) >= 2:
                m = np.mean(vals)
                if m > 1e-6:
                    cv = np.std(vals) / m
                    cvs.append(cv)
                    all_cv.append(cv)
        if cvs:
            per_player_cv[pl] = float(np.mean(cvs))
    return {
        'per_player_cv': per_player_cv,
        'global_cv': float(np.mean(all_cv)) if all_cv else float('nan'),
        'lengths_mm': {pl: {f'{a}-{b}': v for (a, b), v in d.items()}
                       for pl, d in lengths.items()},
    }
