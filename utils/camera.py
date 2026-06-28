"""Classe Camera per gestire calibrazione, proiezione e undistortion."""
import json
from pathlib import Path
import numpy as np
import cv2


class Camera:
    """Modello pinhole con distorsione: s·x = K·[R|t]·X (slide 25, Geometry).

    Tutte le coordinate mondo sono in mm (come la calibrazione fornita).
    """
    def __init__(self, cam_id: str, K, dist, rvec, tvec):
        self.cam_id = cam_id
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).flatten()
        self.rvec = np.asarray(rvec, dtype=np.float64).flatten()
        self.tvec = np.asarray(tvec, dtype=np.float64).flatten()
        self.R, _ = cv2.Rodrigues(self.rvec)
        self.t = self.tvec.reshape(3, 1)
        self.Rt = np.hstack([self.R, self.t])     # 3x4
        self.P = self.K @ self.Rt                  # 3x4 matrice di proiezione

    @classmethod
    def from_json(cls, cam_id: str, calib_path: Path):
        d = json.load(open(calib_path))
        return cls(cam_id, d['mtx'], d['dist'], d['rvecs'], d['tvecs'])

    @property
    def center(self) -> np.ndarray:
        """Centro camera in coordinate mondo: C = -R^T · t."""
        return (-self.R.T @ self.t).flatten()

    def project(self, X_world: np.ndarray) -> np.ndarray:
        """Proietta punti 3D mondo (N,3) in pixel 2D (N,2) — include distorsione."""
        X = np.atleast_2d(X_world).astype(np.float64)
        pts2d, _ = cv2.projectPoints(X, self.rvec, self.tvec, self.K, self.dist)
        return pts2d.reshape(-1, 2)

    def undistort_points(self, pts2d: np.ndarray) -> np.ndarray:
        """Rimuove la distorsione da punti pixel (N,2) → (N,2) ancora in pixel."""
        pts = np.atleast_2d(pts2d).astype(np.float64).reshape(-1, 1, 2)
        undist = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        return undist.reshape(-1, 2)

    def __repr__(self):
        return f'Camera({self.cam_id}, center={self.center.round(1)})'


def load_cameras(cams_dir: Path,
                 cam_ids=('cam_1', 'cam_2', 'cam_3', 'cam_4', 'cam_5', 'cam_7'),
                 version='camera_config_v2') -> dict:
    """Carica un dict {cam_id: Camera} per le camere richieste."""
    cams = {}
    for cid in cam_ids:
        p = Path(cams_dir) / version / cid / 'camera_calib.json'
        cams[cid] = Camera.from_json(cid, p)
    return cams
