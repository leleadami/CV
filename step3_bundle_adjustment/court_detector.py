"""Court points detector: estrae intersezioni linee campo via Hough + matching
con punti 3D noti FIBA proiettati tramite calibrazione iniziale.

Pipeline:
1. Canny edge detection sulla luminanza
2. HoughLinesP per estrarre segmenti di linea
3. Cluster in 2 orientazioni principali (linee del campo sono ortogonali in vista bird-eye,
   ma in vista obliqua restano due famiglie dominanti)
4. Pairwise intersezioni tra linee di famiglie diverse
5. Per ogni 3D notable point del campo FIBA, proietta con calib iniziale → expected 2D
6. Match: nearest neighbor entro `max_dist_px`

Output: dict {cam_id: list of (X_3d_mm, x_2d_observed)}
"""
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2


def court_floor_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Maschera grossolana del pavimento (legno chiaro, basso saturazione).
    Esclude soffitto/gradinate/persone scure.

    Returns: mask uint8 (255 = pavimento candidato).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    # Legno: hue ~10-30, saturation 30-180, luminosità medio-alta
    m = ((H >= 5) & (H <= 35) & (S >= 20) & (S <= 200) & (V >= 80)).astype(np.uint8) * 255
    # Chiudi buchi piccoli + dilata per includere linee bianche del campo
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.dilate(m, np.ones((25, 25), np.uint8))
    return m


def detect_line_segments(img_bgr: np.ndarray,
                         canny_low: int = 60,
                         canny_high: int = 180,
                         hough_threshold: int = 150,
                         min_line_len: int = 200,
                         max_line_gap: int = 30,
                         use_floor_mask: bool = True) -> np.ndarray:
    """Estrae segmenti di linea via Canny + HoughLinesP. Return (N,4) [x1,y1,x2,y2]."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    if use_floor_mask:
        mask = court_floor_mask(img_bgr)
        edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                            threshold=hough_threshold,
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    if lines is None:
        return np.zeros((0, 4), dtype=np.float32)
    return lines.reshape(-1, 4).astype(np.float32)


def line_angle(seg: np.ndarray) -> float:
    """Angolo del segmento in [0, π)."""
    x1, y1, x2, y2 = seg
    a = np.arctan2(y2 - y1, x2 - x1)
    return a % np.pi


def segment_intersect(s1: np.ndarray, s2: np.ndarray,
                       extend: bool = True) -> np.ndarray:
    """Intersezione tra due segmenti (eventualmente estesi a rette).
    Returns (x,y) o None se paralleli/non si intersecano nel range immagine.
    """
    x1, y1, x2, y2 = s1
    x3, y3, x4, y4 = s2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den
    if not extend and (t < 0 or t > 1 or u < 0 or u > 1):
        return None
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return np.array([px, py], dtype=np.float32)


def all_intersections(segments: np.ndarray,
                       angle_diff_min: float = np.deg2rad(20),
                       img_shape: Tuple[int, int] = (2160, 3840),
                       extend: bool = True) -> np.ndarray:
    """Tutte le intersezioni tra coppie di segmenti con orientazioni diverse.
    Filtra a quelle dentro la cornice immagine."""
    angles = np.array([line_angle(s) for s in segments])
    H, W = img_shape
    ints = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            da = abs(angles[i] - angles[j])
            da = min(da, np.pi - da)
            if da < angle_diff_min:
                continue
            p = segment_intersect(segments[i], segments[j], extend=extend)
            if p is None:
                continue
            x, y = p
            if x < -50 or x > W + 50 or y < -50 or y > H + 50:
                continue
            ints.append(p)
    if not ints:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(ints, dtype=np.float32)


def match_expected(expected_2d: np.ndarray,
                    detected_2d: np.ndarray,
                    max_dist_px: float = 60.0) -> List[int]:
    """Per ogni expected, indice del detected più vicino (o -1 se nessuno entro soglia)."""
    if len(detected_2d) == 0:
        return [-1] * len(expected_2d)
    out = []
    for e in expected_2d:
        d = np.linalg.norm(detected_2d - e, axis=1)
        k = int(np.argmin(d))
        out.append(k if d[k] < max_dist_px else -1)
    return out


def _gate_to_court_region(ints: np.ndarray, expected: np.ndarray,
                          img_shape, margin_frac: float = 0.12,
                          margin_min: float = 60.0) -> np.ndarray:
    """Tiene solo le intersezioni dentro la bbox dei court points proiettati + margine.

    Usa il prior di calibrazione (expected = court FIBA proiettato con v2) per sapere
    DOVE sta il campo nell'immagine ed eliminare il rumore Hough fuori campo
    (gradinate, soffitto, canestri). bbox robusta via percentili 5-95 (ignora punti
    proiettati anomali). Clampata ai bordi immagine.
    """
    if len(ints) == 0 or len(expected) == 0:
        return ints
    H, W = img_shape[:2]
    ex = np.asarray(expected, dtype=np.float64)
    # bbox robusta (percentili) per ignorare eventuali proiezioni degeneri
    x0, x1 = np.percentile(ex[:, 0], [5, 95])
    y0, y1 = np.percentile(ex[:, 1], [5, 95])
    mg = max(margin_min, margin_frac * max(x1 - x0, y1 - y0))
    x0 = max(0, x0 - mg); x1 = min(W, x1 + mg)
    y0 = max(0, y0 - mg); y1 = min(H, y1 + mg)
    m = ((ints[:, 0] >= x0) & (ints[:, 0] <= x1) &
         (ints[:, 1] >= y0) & (ints[:, 1] <= y1))
    return ints[m]


def _detect_cam_multiframe(args):
    """Worker: processa UNA camera su n_frames. Indipendente per cam → parallelizzabile.

    Ritorna (j, cid, acc_cam, n_matched_per_frame) dove acc_cam[k] = lista di obs 2D
    del court point k per questa camera.
    """
    (j, cid, cam, vid, court_points_3d_mm, n_frames,
     max_dist_px, hough_kwargs, cv2_threads, save_debug_dir) = args
    N = len(court_points_3d_mm)
    acc_cam = [[] for _ in range(N)]
    if cv2_threads is not None:
        cv2.setNumThreads(int(cv2_threads))
    if vid is None or not Path(vid).exists():
        return j, cid, acc_cam, None
    cap = cv2.VideoCapture(str(vid))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return j, cid, acc_cam, None
    sample = np.linspace(0, total - 1, n_frames, dtype=int)
    expected = cam.project(court_points_3d_mm)
    n_matched_per_frame = []
    for fi, fr in enumerate(sample):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr))
        ret, img = cap.read()
        if not ret:
            continue
        segs = detect_line_segments(img, **hough_kwargs)
        ints = all_intersections(segs, img_shape=img.shape[:2])
        # Spatial gating: tieni solo intersezioni nella regione del campo proiettato
        # (prior v2). Elimina il rumore Hough su gradinate/soffitto/canestri.
        ints = _gate_to_court_region(ints, expected, img.shape[:2])
        mi = match_expected(expected, ints, max_dist_px=max_dist_px)
        n_m = 0
        for k, idx in enumerate(mi):
            if idx >= 0:
                acc_cam[k].append(ints[idx])
                n_m += 1
        n_matched_per_frame.append(n_m)
        # Debug overlay sul PRIMO frame campionato: rosso=expected, verde=intersezioni, magenta=match
        if save_debug_dir is not None and fi == 0:
            sd = Path(save_debug_dir); sd.mkdir(parents=True, exist_ok=True)
            dbg = img.copy()
            for s in segs:
                cv2.line(dbg, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (0, 200, 200), 2)
            for p in ints:
                cv2.circle(dbg, (int(p[0]), int(p[1])), 6, (0, 255, 0), 2)
            for k, e in enumerate(expected):
                cv2.circle(dbg, (int(e[0]), int(e[1])), 12, (0, 0, 255), 3)
                if mi[k] >= 0:
                    m = ints[mi[k]]
                    cv2.line(dbg, (int(e[0]), int(e[1])), (int(m[0]), int(m[1])), (255, 0, 255), 2)
            small = cv2.resize(dbg, None, fx=0.4, fy=0.4)
            cv2.imwrite(str(sd / f'{cid}_court_detect.png'), small)
    cap.release()
    return j, cid, acc_cam, n_matched_per_frame


def build_court_observations_multiframe(cameras: dict,
                                         court_points_3d_mm: np.ndarray,
                                         video_paths: dict,
                                         n_frames: int = 5,
                                         max_dist_px: float = 50.0,
                                         hough_kwargs: dict = None,
                                         save_debug_dir=None) -> Tuple[np.ndarray, np.ndarray]:
    """Aggrega court points su multipli frame video: usa MEDIANA delle detections matched.
    Riduce rumore puntuale di Hough.

    Args:
        video_paths: {cam_id: Path video out{N}.mp4}
        n_frames: numero di frame equispaziati da estrarre.

    Returns:
        obs: (N, M, 2) — mediana delle detections
        vis: (N, M) bool — True se >= 50% dei frame ha matched

    Nota perf: testato un pool multiprocessing sulle 6 cam → SCARTATO. (1) fork+OpenCV
    va in deadlock (i thread interni di cv2 lasciano lock held nei figli). (2) Profilando,
    il carico è I/O-bound (seek+decode video 4K su un disco), load avg ~0: i core NON sono
    il collo di bottiglia. cv2 in serial usa già tutti i thread per Canny/Hough → ottimo.
    """
    if hough_kwargs is None:
        hough_kwargs = {}
    cam_ids = list(cameras.keys())
    N = len(court_points_3d_mm)
    M = len(cam_ids)

    tasks = [(j, cid, cameras[cid], video_paths.get(cid), court_points_3d_mm,
              n_frames, max_dist_px, hough_kwargs, None, save_debug_dir)
             for j, cid in enumerate(cam_ids)]

    acc = [[[] for _ in range(M)] for _ in range(N)]   # [court_pt][cam] -> lista obs
    results = [_detect_cam_multiframe(t) for t in tasks]

    for j, cid, acc_cam, n_mpf in results:
        if n_mpf is None:
            print(f'  [WARN] {cid}: video mancante o vuoto')
            continue
        for k in range(N):
            acc[k][j] = acc_cam[k]
        print(f'  {cid}: matches/frame = {n_mpf}')

    obs = np.full((N, M, 2), np.nan)
    vis = np.zeros((N, M), dtype=bool)
    threshold = max(1, n_frames // 3)   # almeno 1/3 dei frame
    for k in range(N):
        for j in range(M):
            pts = acc[k][j]
            if len(pts) >= threshold:
                arr = np.array(pts)
                # Mediana resistente a outlier
                obs[k, j] = np.median(arr, axis=0)
                vis[k, j] = True
    return obs, vis


def build_court_observations(cameras: dict,
                              court_points_3d_mm: np.ndarray,
                              image_paths: dict,
                              max_dist_px: float = 60.0,
                              hough_kwargs: dict = None,
                              save_debug_dir: Path = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Costruisce le osservazioni 2D reali dei court points.

    Args:
        cameras: {cam_id: Camera} con calibrazione iniziale (possibilmente buona).
        court_points_3d_mm: (N, 3) punti 3D noti del campo in mm.
        image_paths: {cam_id: Path} a immagine 3840×2160 per camera.
        max_dist_px: raggio massimo di matching tra proiezione attesa e intersezione detected.

    Returns:
        obs: (N, M, 2) — osservazioni 2D (NaN se non matched)
        vis: (N, M) bool
        debug: dict per visualizzazione
    """
    if hough_kwargs is None:
        hough_kwargs = {}
    cam_ids = list(cameras.keys())
    N = len(court_points_3d_mm)
    M = len(cam_ids)
    obs = np.full((N, M, 2), np.nan)
    vis = np.zeros((N, M), dtype=bool)
    debug = {}

    for j, cid in enumerate(cam_ids):
        cam = cameras[cid]
        img_path = image_paths.get(cid)
        if img_path is None or not Path(img_path).exists():
            print(f'  [WARN] {cid}: immagine mancante')
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'  [WARN] {cid}: imread fallito')
            continue

        # 1. Proietta punti 3D con calibrazione iniziale
        expected_2d = cam.project(court_points_3d_mm)

        # 2. Detect linee + intersezioni
        segs = detect_line_segments(img, **hough_kwargs)
        ints = all_intersections(segs, img_shape=img.shape[:2])

        # 3. Matching nearest neighbor
        match_idx = match_expected(expected_2d, ints, max_dist_px=max_dist_px)
        n_matched = 0
        for k, mi in enumerate(match_idx):
            if mi >= 0:
                obs[k, j] = ints[mi]
                vis[k, j] = True
                n_matched += 1
        print(f'  {cid}: {len(segs)} segs, {len(ints)} intersezioni, {n_matched}/{N} match')

        debug[cid] = {
            'expected_2d': expected_2d,
            'segments': segs,
            'intersections': ints,
            'match_idx': match_idx,
            'img_shape': img.shape[:2],
            'img_path': str(img_path),
        }

        if save_debug_dir is not None:
            save_debug_dir = Path(save_debug_dir)
            save_debug_dir.mkdir(parents=True, exist_ok=True)
            dbg = img.copy()
            for s in segs:
                cv2.line(dbg, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])),
                         (0, 200, 200), 2)
            for p in ints:
                cv2.circle(dbg, (int(p[0]), int(p[1])), 6, (0, 255, 0), 2)
            for k, e in enumerate(expected_2d):
                cv2.circle(dbg, (int(e[0]), int(e[1])), 12, (0, 0, 255), 3)
                if match_idx[k] >= 0:
                    m = ints[match_idx[k]]
                    cv2.line(dbg, (int(e[0]), int(e[1])), (int(m[0]), int(m[1])),
                             (255, 0, 255), 2)
            scale = 0.4
            small = cv2.resize(dbg, None, fx=scale, fy=scale)
            cv2.imwrite(str(save_debug_dir / f'{cid}_court_detect.png'), small)

    return obs, vis, debug
