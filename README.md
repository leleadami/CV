# HPE Project — Human Pose Estimation in Sports

Progetto **CV2 UniTN A.Y. 2025-2026**, Sanbapolis multi-view setup.
Implementazione completa dei task 1, 2, 2a, 3, 3a (team da 1 persona).

---

## 1. Obiettivo

Stimare le pose 3D dei giocatori da viste multi-camera del palasport
**Sanbapolis** (azione `hpe_01`) e valutare l'accuratezza della
ricostruzione prima e dopo il **Bundle Adjustment** sui punti del campo.

Setup:
- 6 camere fisse (`cam_1, cam_2, cam_3, cam_4, cam_5, cam_7`) montate sui muri
- Risoluzione: 3840 × 2160 px @ 25 fps
- 10 giocatori per frame, 2 squadre (Red + White) identificati per numero
- 5 frame annotati manualmente via Roboflow (Task 1)

---

## 2. Mappatura tasks ↔ implementazione

| Assignment | Implementazione | File principale |
|---|---|---|
| **1.** Annotate player's poses (Roboflow) | 30 PNG + COCO export, 18 keypoint custom, 10 player × 5 frame × 6 cam | `data/annotations/_annotations.coco.json` |
| **2.1** Rectify video at input (slide 11) | Wrapper su `utils/rectified_videos.py` (script prof). Opzionale: triangulation applica già `undistortPoints` ai keypoint GT (slide: "se rettifichi video, applica stessa transform a GT") | `step2_triangulation/rectify_videos.py` |
| **2.2** Get 3D position via triangulation | DLT multi-view su SVD `2N × 4`, undistorzione preliminare con `cv2.undistortPoints` | `utils/triangulation.py`, `step2_triangulation/run_triangulation.py` |
| **2a.** Reprojection accuracy vs GT 2D | Proiezione del 3D su ogni vista + distanza L2 da kpt Roboflow → **MPJPE** | stesso file, `reprojection_error()` |
| **3.** Bundle Adjustment sui court points | Court detector automatico (Canny + HoughLinesP + matching), median multi-frame, Levenberg-Marquardt scipy | `step3_bundle_adjustment/court_detector.py`, `run_ba_real.py` |
| **3a.** Accuracy triangolazione post-BA | Re-triangola player con calib raffinata, ricalcola MPJPE | `run_ba_real.py` (sezione `Step 2 POST-BA`) |

---

## 3. Struttura repo

```
project/
├── data/
│   ├── annotations/        # Roboflow COCO (5 frame × 6 cam, hpe_01)
│   ├── images/             # 30 PNG originali 3840×2160
│   └── cameras/
│       ├── camera_config_v1/   # NON usata (valori incoerenti, cam_1 a Z=45m)
│       └── camera_config_v2/   # 10 camere con intrinsiche + estrinseche affidabili
│
├── utils/
│   ├── camera.py           # Classe Camera (K, dist, rvec, tvec, project, undistort)
│   ├── coco_utils.py       # Parser Roboflow COCO
│   ├── court.py            # Punti 3D campo FIBA (28×15m) + dimensioni palasport
│   ├── triangulation.py    # DLT multi-view (slide 05 Geometry)
│   ├── bundle_adjustment.py # BA con scipy.least_squares (LM o Huber)
│   └── rectified_videos.py # (fornito dal tutor)
│
├── step2_triangulation/
│   ├── rectify_videos.py       # Task 2.1 — wrapper su utils/rectified_videos.py (script prof)
│   └── run_triangulation.py    # Task 2.2 + 2a — DLT → results_3d.json + plot 3D
│
├── step3_bundle_adjustment/
│   ├── court_detector.py       # Canny + HoughLinesP + match court FIBA
│   ├── run_ba.py               # Task 3 — versione SINTETICA (baseline)
│   └── run_ba_real.py          # Task 3 + 3a — versione REALE (deliverable)
│
├── sanity_check_cameras.py     # Validazione calibrazione + planimetria 2D/3D
├── run_all.py                  # Entry point completo
├── setup_new_pc.sh             # Setup automatico (conda env + deps + CUDA torch)
├── requirements.txt
└── PROJECT_CONTEXT.md          # Note dettagliate su decisioni e dati
```

---

## 4. Sistema di riferimento

- **Origine**: centro del campo basket (centro circle)
- **Pavimento**: a **Z = 0** (i court points sono a Z=0 e si proiettano sulle linee dipinte).
  Verificato: i piedi/caviglie dei giocatori triangolano a **Z ≈ 0.15 m** (sul pavimento),
  hips ≈ 1.04 m, testa ≈ 1.71 m → altezza ~1.56 m, coerente con persone in piedi.
- **Camere**: a **Z ≈ 6–7 m** (montate in alto sui muri, guardano in basso — vedi debug Hough).
  ⚠️ La planimetria GT riporta altezze ~2–3 m che **non** coincidono col calib v2; ci si fida
  del calib perché la triangolazione è fisicamente corretta. (Niente offset "+4 m": era una
  riconciliazione errata con la GT, ora rimossa.)
- **Palasport**: 47.31 × 29.61 m, centro a (-1.06, +3.80) m rispetto al campo
- **Unità**: tutte le coordinate sono in **millimetri** (coerente con calibrazione fornita)

---

## 5. Setup

### Primo avvio (PC nuovo, GPU NVIDIA)

```bash
cd project
bash setup_new_pc.sh
```

Lo script:
1. Crea env conda `hpe` (Python 3.10)
2. Installa PyTorch CUDA 12.1 (auto-detect NVIDIA)
3. Installa `requirements.txt` (numpy, scipy, opencv, matplotlib, ultralytics, pycocotools, …)
4. Verifica torch CUDA disponibile
5. Lancia `run_all.py`

### Avvio normale

```bash
conda activate hpe
cd project
python run_all.py
```

### Lancio singoli step

```bash
python sanity_check_cameras.py                  # planimetria + verifica calib
python step2_triangulation/run_triangulation.py # Task 2 + 2a
python step3_bundle_adjustment/run_ba.py        # Task 3 (baseline sintetica)
python step3_bundle_adjustment/run_ba_real.py   # Task 3 + 3a (deliverable)
```

---

## 6. Output prodotti

```
project/
├── sanity_check_cameras.png            # planimetria 2D + tabella camere v2 (X,Y,Z,h_pav) + tabella court FIBA
│
├── step2_triangulation/output/
│   ├── results_3d.json                 # X_3d_mm + repro_errs_px + loo_errs_px + metrics (LOO, bone-CV)
│   ├── repro_errors.png                # istogramma reprojection error PER CAMERA (6 subplot + mediana)
│   ├── skeletons_3d_frame1.png         # scheletri 3D triangolati + posizioni cam (vista 3D)
│   └── reproj_overlay_frame1.png       # scheletri 3D riproiettati sui frame reali (check qualitativo)
│
└── step3_bundle_adjustment/
    ├── output/                          # versione SINTETICA (baseline)
    │   ├── ba_summary.json
    │   └── ba_comparison.png
    └── output_real/                     # versione REALE (deliverable)
        ├── ba_real_summary.json         # esperimenti A (onesto) + B (stabilità init)
        ├── ba_real_comparison.png       # ECDF sovrapposte: v2 baseline vs v2+BA reale (media, P50, P90)
        └── debug/cam_*_court_detect.png # debug Hough: rosso=expected, verde=intersezioni, magenta=match
```

---

## 7. Risultati (deliverable principale)

**Task 2 + 2a — Triangolazione** (calib v2 nominale, DLT + refinement non-lineare):

```
MPJPE reprojection (fit):   25.0 px   # errore sulle viste usate per triangolare
MPJPE leave-one-view-out:   41.6 px   # accuratezza ONESTA: vista held-out, mai usata
Bone-length CV (3D GT-free):15.9%     # coerenza fisica scheletri 3D tra frame
Joint triangolati:          18/18 per ~95% dei player
Player triangolati:         50/50 (5 frame × 10 player)
```

> Il reprojection error standard (25 px) riusa le stesse viste della triangolazione → è
> errore di *fit*, ottimistico. La metrica onesta è il **leave-one-view-out** (41.6 px):
> triangola con N−1 viste e misura sulla vista esclusa. La **bone-length CV** è una
> validazione 3D indipendente dal 2D (un osso deve avere lunghezza ~costante tra frame).

**Task 3 + 3a — Bundle Adjustment REALE ITERATIVO** (`ba_real_summary.json`):

Due esperimenti separati per evitare circolarità (vedi §9). L'esperimento A è
**iterativo**: round di (detection con prior corrente → match → BA); dopo ogni
BA il prior migliora → più match entro il radius 30 px → BA successivo meglio
vincolato (schema EM/ICP-like). Stop al plateau del MPJPE player.

```
Esperimento A — ONESTO (BA iterativo dalla calib v2 reale):
  round   match   punti   court RMSE fin   MPJPE fit   MPJPE LOO
  v2      —       —       (6.86 residuo)   25.05 px    41.57 px
  1       48/96   13/16   3.66 px          20.81 px    34.48 px
  2       52/96   14/16   4.93 px          18.95 px    31.39 px
  3       54/96   14/16   4.59 px          18.28 px    30.21 px
  4       55/96   15/16   5.29 px          18.26 px    30.18 px   <- plateau, stop
  bone-CV: 15.9% → 15.8% (≈ invariato)
  Miglioramento totale: MPJPE fit -27.1%, LOO -27.4%

Esperimento B — stabilità rispetto all'init (perturbazione 100 mm nota):
  B_AvsB_agreement_tvec_mm: ~0.00  # A e B convergono allo STESSO ottimo
  -> soluzione ben determinata dai court points, indipendente dall'init
```

**Cosa significa**:
- La calib v2 fornita ha un residuo REALE di **6.86 px** sui court points; il primo BA lo dimezza a 3.66 px.
- MPJPE player **25.05 → 18.26 px (-27%)** e — più importante — **LOO 41.57 → 30.18 px (-27%)**: il BA, pur ottimizzando solo sui court points, trasferisce il miglioramento alla metrica onesta (vista held-out mai usata). Non circolare.
- Il court RMSE dei round successivi NON è confrontabile tra round (insiemi di osservazioni diversi: round 4 include punti più difficili). Le metriche confrontabili (MPJPE fit/LOO, sempre sugli stessi dati player) migliorano monotone.
- **bone-CV invariato (15.9→15.8%)**: coerente — il bias di calibrazione è sistematico (sposta lo scheletro intero), le lunghezze relative quasi non cambiano; il CV è dominato dal rumore di annotazione per-frame.
- Δtvec totale del BA iterativo: cam_3 = 288 mm, cam_5 = 435 mm (le camere anomale negli istogrammi per-camera) → la v2 conteneva bias reali che il BA rileva e corregge.
- A e B convergono a |Δtvec| ≈ 0 mm → l'ottimo è ben determinato, non un artefatto dell'inizializzazione.

---

## 8. Pipeline tecnica

### Task 2 — DLT multi-view

Per ogni joint visibile in ≥ 2 camere (`visibility ≥ 1`):

```
1. Undistorta (u, v) → (u', v') con cv2.undistortPoints + intrinsiche K
2. Per ogni vista j, costruisci riga:
     u'·P[2] − P[0]       (P = matrice proiezione 3×4)
     v'·P[2] − P[1]
3. Matrice A di forma (2N, 4)
4. SVD: X_3d = ultimo singular vector / X_3d[3]   (omogenee → euclidee)
5. Reprojection error: proietta X_3d back, calcola L2 vs kpt originale
```

### Task 3 — Court detector + BA

```
1. Per ogni cam, estrai 8 frame equispaziati dal video out{N}.mp4
2. Per ogni frame:
   - Canny edge detection sulla luminanza
   - Maschera HSV pavimento (legno chiaro, esclude gradinate/soffitto)
   - HoughLinesP (min_line_len=200, threshold=150)
   - Compute intersezioni tra coppie di linee con orientazioni diverse
3. Match con punti 3D FIBA noti, usando calib v2 come prior:
   - Proietta i 16 court points → expected_2d
   - Per ogni expected, prendi intersezione più vicina entro 30 px
4. Aggrega: per ogni (court_point, cam), mediana delle detection su 8 frame
5. Levenberg-Marquardt minimizza:
     Σ_ij visibility_ij · ‖projection_ij(rvec_j, tvec_j) − obs_ij‖²
   ottimizzando solo (rvec, tvec) per camera; intrinsiche K e dist fisse.
6. ITERA (1-5) con prior = calib raffinata: più match per round
   (48→55/96), stop al plateau del MPJPE player (max 4 round).
7. Re-triangola tutti i player con calib finale → MPJPE fit + LOO + bone-CV post-BA
```

---

## 9. Note implementative

- **Calibrazione v2 vs v1**: scartata v1 (cam_1 risultava a Z=45 m). v2 ha tutte le camere a Z=6-7 m (≈ 2-3 m sopra il pavimento, coerente con altezze GT della planimetria).
- **Detector Hough**: scelto come compromesso tra automazione (no annotazione manuale) e robustezza. Per ottenere il match servono 4 mitigazioni: maschera HSV del pavimento, **spatial gating** (scarta intersezioni fuori dalla bbox del campo proiettato con v2 → elimina rumore Hough su gradinate/soffitto), multi-frame median, match radius stretto (30 px) usando calib v2 come prior. Il gating non cambia il risultato BA (48 vs 49 match, RMSE 3.66 vs 3.61) ma pulisce drasticamente il debug e riduce il rischio di falsi match.
- **Huber loss**: testato `loss='huber'` con `f_scale=8`, ma il clipping aggressivo dei residui degrada MPJPE post-BA (35 vs 20 px). Linear loss + multi-frame median offre robustezza sufficiente.
- **Center circle / archi 3pt**: esclusi dai court points perché Hough non rileva curve in modo affidabile.
- **Rectification video**:
  - Script `utils/rectified_videos.py` (**fornito dalla prof**) lasciato intatto.
  - Wrapper `step0_rectification/rectify_videos.py` lo invoca con i path corretti del nostro layout; produce `data/rectified_videos/hpe_01/out{N}_rect.mp4`.
  - **Default**: step 0 è disabilitato in `run_all.py` (commentato) perché lungo (~5 min, ~1.5 GB output) e **matematicamente equivalente** all'undistortion on-the-fly che `utils/triangulation.py` esegue su ogni keypoint via `cv2.undistortPoints`.
  - Slide 11 dice: "if you rectify the video you also have to apply the same transformation to the GT". Le due opzioni:
    - **Opzione A** (slide): `rectify(video) → annota su frame rectified → DLT con dist=0`
    - **Opzione B** (questa pipeline): `annota su frame distorti → undistort(kpt) → DLT con dist=0`
  - Producono lo stesso 3D output. Opzione B evita 1.5 GB di video rigenerati e mantiene le annotazioni Roboflow originali (fatte sui frame distorti).
  - Per abilitare opzione A: decommenta riga step 0 in `run_all.py`, poi lancia.

---

## 10. Limitazioni e possibili miglioramenti

| Limitazione | Possibile miglioramento | Margine atteso |
|---|---|---|
| 15/16 court points, 55/96 osservazioni | Annotazione manuale 1 frame/cam | satura la copertura |
| Hough lossy su archi | Detector CNN (CourtPointNet) | +10-15% MPJPE |
| 1 sola azione (hpe_01) | YOLOv8-pose su hpe_02..09 (richiede CUDA + post-process) | scala detection |
| Frame triangolati indipendenti | Smoothing temporale / vincolo bone-length | riduce jitter (ma "impone" la coerenza oggi usata come metrica) |

**Miglioramento ADOTTATO** (2026-06-09): BA **iterativo** (re-match con prior
raffinato) — match 48→55/96, MPJPE fit 20.8→18.3 px, LOO 34.5→30.2 px rispetto
al BA singolo round. Vedi `eval_best.py` per l'esperimento di convergenza
(plateau round 3-4, Δ<0.05 px).

**Ablation testate e SCARTATE** (con dati, vedi `bundle_adjustment.optimize_intrinsics` e `triangulate_skeleton(robust=...)`):

| Idea | Risultato misurato | Esito |
|---|---|---|
| Refinare K (fx,fy,cx,cy) nel BA | court RMSE 3.61→2.88 px **ma MPJPE player 20.8→39.5 px**, focali shift >1000px | ❌ overfit (49 oss insufficienti) → K resta fissa da scacchiera |
| RANSAC/outlier rejection sulle viste | fit MPJPE 25→13 px ma LOO invariato (41.6) e **bone-CV peggiora** 15.9→18.0% | ❌ errori sono residuo calib sistematico, non outlier di vista |
| Parallelizzare detection (multiprocessing) | fork+OpenCV deadlock; carico I/O-bound (load avg ~0) | ❌ serial con cv2 16-thread già ottimo |

---

## 11. Reference

- **Slide 05 Geometry** pp. 25-46 → camera pinhole + DLT triangolazione
- **Slide 08 SfM – SLAM** pp. 12-15 → formula Bundle Adjustment
- **Reference paper** (assignment slide 12): https://eth-ait.github.io/WorldPoseDataset/
- **Consegna**: GitHub Classroom https://classroom.github.com/a/sDy4Ysr0 — formato LNCS, max 5 pagine, deliverable 3 giorni prima dell'esame
