"""Bundle Adjustment per raffinare i parametri di calibrazione delle camere.

Formula (slide 08 SfM, p.13):
    arg min Σ_i Σ_j v_ij · ||q_ij - π(K_j, R_j, t_j, X_i)||²
       K,R,t

dove:
  X_i = 3D punti noti del campo (fissi, known)
  q_ij = 2D osservazioni del punto i nella camera j
  v_ij = visibility binaria
  π = funzione di proiezione (cv2.projectPoints, include distorsione)

Per il nostro setup, ottimizziamo SOLO rvec, tvec di ogni camera
(intrinseche K e dist le lasciamo fisse — sono ricavate dalla calibrazione
con scacchiera che è più affidabile).
"""
import numpy as np
import cv2
from scipy.optimize import least_squares


def _pack_params(cameras_to_optim, optimize_intrinsics=False):
    """Concatena rvec+tvec (+ fx,fy,cx,cy se optimize_intrinsics) per camera in 1D."""
    params = []
    for cam in cameras_to_optim:
        params.extend(cam.rvec.tolist())
        params.extend(cam.tvec.tolist())
        if optimize_intrinsics:
            params.extend([cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]])
    return np.array(params)


def _unpack_params(params, n_cams, optimize_intrinsics=False):
    """Estrai (rvec, tvec, K_or_None) per ogni camera dal vettore 1D."""
    stride = 10 if optimize_intrinsics else 6
    out = []
    for j in range(n_cams):
        b = j * stride
        rvec = np.array(params[b:b+3])
        tvec = np.array(params[b+3:b+6])
        K = None
        if optimize_intrinsics:
            fx, fy, cx, cy = params[b+6:b+10]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        out.append((rvec, tvec, K))
    return out


def _residuals(params, cameras, X_world, observations, visibility,
               optimize_intrinsics=False):
    """Vettore residui per scipy.optimize.least_squares.

    Args:
        params: vettore 1D di tutti i (rvec, tvec) delle camere
        cameras: lista di Camera (per K, dist, fissi)
        X_world: (N, 3) punti 3D noti del campo
        observations: (N, M, 2) osservazioni 2D — N punti × M cam
        visibility: (N, M) binaria

    Returns:
        residuals: array 1D di tutti i residui (dx, dy per ogni osservazione valida)
    """
    n_cams = len(cameras)
    extrinsics = _unpack_params(params, n_cams, optimize_intrinsics)
    residuals = []
    for j, cam in enumerate(cameras):
        rvec, tvec, K = extrinsics[j]
        K_use = K if K is not None else cam.K
        # Proietta TUTTI i punti X_world con questi nuovi rvec, tvec (+K se ottimizzata)
        proj, _ = cv2.projectPoints(X_world.astype(np.float64),
                                     rvec.astype(np.float64),
                                     tvec.astype(np.float64),
                                     K_use, cam.dist)
        proj = proj.reshape(-1, 2)
        for i in range(len(X_world)):
            if visibility[i, j]:
                residuals.extend((proj[i] - observations[i, j]).tolist())
    return np.array(residuals)


def run_bundle_adjustment(cameras, X_world, observations, visibility,
                           verbose=True, loss='linear', f_scale=5.0,
                           optimize_intrinsics=False):
    """Esegue Bundle Adjustment su rvec/tvec (+ K se optimize_intrinsics) di tutte le camere.

    Args:
        optimize_intrinsics: se True ottimizza anche fx,fy,cx,cy per camera (dist resta fissa).
            ATTENZIONE: con poche osservazioni rischia overfit — validare su MPJPE player.

    Returns:
        new_extrinsics: lista di (rvec, tvec) — oppure (rvec, tvec, K) se optimize_intrinsics
        result: ScipyOptimizeResult
        initial_repro_err: RMSE px iniziale
        final_repro_err: RMSE px finale
    """
    p0 = _pack_params(cameras, optimize_intrinsics)
    n_cams = len(cameras)
    args = (cameras, X_world, observations, visibility, optimize_intrinsics)

    r0 = _residuals(p0, *args)
    initial_rmse = np.sqrt(np.mean(r0**2)) if len(r0) > 0 else np.nan

    if verbose:
        print(f'BA: {n_cams} camere, {len(X_world)} punti 3D, '
              f'{int(visibility.sum())} osservazioni'
              f'{" (+K intrinsics)" if optimize_intrinsics else ""}')
        print(f'Residuo iniziale: RMSE={initial_rmse:.3f} px')

    # Solver: 'lm' (Gauss-Newton) per loss='linear' (default), 'trf' (trust-region) per Huber robust.
    method = 'lm' if loss == 'linear' else 'trf'
    extra = {}
    if loss != 'linear':
        extra['loss'] = loss
        extra['f_scale'] = f_scale
    result = least_squares(
        _residuals, p0, args=args,
        method=method,
        xtol=1e-8, ftol=1e-8, max_nfev=500,
        verbose=2 if verbose else 0,
        **extra,
    )

    r1 = _residuals(result.x, *args)
    final_rmse = np.sqrt(np.mean(r1**2)) if len(r1) > 0 else np.nan

    if verbose:
        print(f'\nResiduo finale:   RMSE={final_rmse:.3f} px')
        print(f'Riduzione: {(1 - final_rmse/initial_rmse)*100:.1f}%')

    unpacked = _unpack_params(result.x, n_cams, optimize_intrinsics)
    if optimize_intrinsics:
        new_extrinsics = unpacked   # list of (rvec, tvec, K)
    else:
        new_extrinsics = [(rv, tv) for rv, tv, _ in unpacked]
    return new_extrinsics, result, initial_rmse, final_rmse
