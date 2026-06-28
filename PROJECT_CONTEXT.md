# Contesto progetto — leggere PRIMA di lavorare

Questo file riassume **tutto il contesto** della chat che ha generato il progetto.
La prossima sessione Claude deve leggerlo per partire allineata.

---

## Obiettivo

Progetto **CV2 UniTN — Human Pose Estimation in Sports** (Sanbapolis).
Implementare i task **1, 2, 3** del PDF `HPE.pptx.pdf`.

- **Task 1**: Annotazione 2D scheletri (FATTO via Roboflow, 30 frame)
- **Task 2**: Triangolazione 3D multi-view + reprojection error (IMPLEMENTATO)
- **Task 3**: Bundle Adjustment sui punti del campo + re-triangolazione (IMPLEMENTATO con dati sintetici)

Il progetto deve poter girare su **GPU NVIDIA** spostandolo, ma la GPU NON è
necessaria per task 1-2-3 (sono numpy/scipy). Servirebbe solo se si sostituiscono
le annotazioni manuali con un detector (YOLOv8-pose).

---

## Dati di partenza (in `/Users/lele/Desktop/CV/`)

- `HPE.pptx.pdf` — specifica del progetto (16 pagine)
- `HPE/material4project/video/hpe_01..09/` — 6 video MP4 per azione (out1..7, no out6)
- `HPE/material4project/3D Pose Estimation Material/` — calibrazione camere (estratti dai zip)
  - `cam_X/camera_calib.json` — SOLO intrinseche (rvec=tvec=0)
  - `cam_X/camera_calib_real.json` — intrinseche + tvec approssimato (rvec=0)
  - `camera_data_with_Rvecs.zip` → estratto in `camera_config_v1/` — **VALORI SBAGLIATI**
  - `camera_data_with_Rvecs_2ndversion.zip` → estratto in `camera_config_v2/` — **USARE QUESTA**
  - `dump/*.json` per ogni camera = corner di SCACCHIERA per calibrazione
    (gruppi da 35 pt = griglia 7×5 regolare, N pose board per file). **NON sono
    punti del campo** — verificato ispezionando i dati. Non usabili come court points
    (manca geometria 3D board / dimensione quadrati). Servono solo a calibrare le cam.
- `HPE/material4project/GT_camera_positions.png` — planimetria con altezze rosse e distanze blu
- `dataset/_annotations.coco.json` + 30 PNG — annotazioni Roboflow per `hpe_01`
- `file/` — slide del corso (cartella teoria)
  - **`05 Geometry.pdf`** pp.25-46 → camera model + triangolazione DLT
  - **`08 SfM - SLAM.pdf`** pp.12-15 → formula Bundle Adjustment

---

## Decisioni prese (NON discutere di nuovo)

### Calibrazione
- **Usata `camera_config_v2`** — v1 ha valori incoerenti (cam_1 a Z=45m).
- v2 ha tutte le camere a ~Z=6-7m. Sono **6-7m sopra il pavimento** (mount alto sui muri).
- **Intrinseche** (`mtx`, `dist`) e **estrinseche** (`rvec`, `tvec`) sono tutte in v2.
- Unità: tutte in **mm**.

### Sistema di riferimento mondo
- Origine al **centro del campo basket** (centro circle).
- **Pavimento a Z = 0** (i court points a Z=0 si proiettano sulle linee dipinte).
  CORRETTO il vecchio "offset +4m": era una riconciliazione SBAGLIATA con altezze GT
  planimetria (~2.4m) che NON coincidono col calib. Verifica oggettiva: i piedi dei
  giocatori triangolano a **Z≈0.15m** (sul pavimento), hips≈1.04m, testa≈1.71m. Se il
  pavimento fosse a +4m i piedi cadrebbero a ~4m → falso. Camere davvero a ~6-7m.
- Palasport reale: **47.31 × 29.61 m**, centro a **(-1.06, +3.80)** rispetto al campo
  (campo spostato a sud perché le gradinate principali sono sul lato top).

### Camere usate per `hpe_01`
- 6 camere: `cam_1, cam_2, cam_3, cam_4, cam_5, cam_7` (no cam_6).
- File video corrispondenti: `out1.mp4, out2.mp4, ..., out5.mp4, out7.mp4`.

### Annotazioni
- **18 keypoint** (non 17), ordine custom (vedi `utils/coco_utils.py:KEYPOINT_NAMES`).
- Roboflow tronca i keypoint trailing con visibility=0 → padding necessario nel parser.
- 10 player annotati per frame (5 Red + 5 White), identificati per nome
  (es. `Red_11`, `White_22`) → **cross-camera matching gratis**, non serve re-id.
- Visibility: 0 = non annotato, 2 = annotato/visibile. (1 non usato.)
- `out{cam}_frame_{idx}_png.rf.{hash}.png` → pattern nomi file.

### Frame sync
- L'utente ha CONFERMATO: `out1_frame_0001` e `out2_frame_0001` sono lo stesso istante.

### Pipeline
- **Task 1**: già fatto via Roboflow, non serve codice.
- **Task 2**: DLT multi-view classico (slide 05 pp.34-36) via SVD.
  Undistortion 2D → costruzione matrice A 2N×4 → SVD → punto 3D.
- **Task 3**: BA via `scipy.optimize.least_squares` (Levenberg-Marquardt).
  Ottimizza SOLO `rvec, tvec` per camera (K e dist fissi, derivano da
  calibrazione con scacchiera affidabile).

---

## Risultati attuali

```
Step 2 + 2a — Triangolazione su 50 scheletri (DLT + refinement non-lineare):
  • MPJPE reprojection (fit):    25.0 px  (riusa viste = ottimistico)
  • MPJPE leave-one-view-out:    41.6 px  (vista held-out = accuratezza ONESTA)
  • Bone-length CV (3D GT-free): 15.9%    (coerenza fisica scheletri)
  • Tutti i 18 joint triangolati con successo

Step 3 + 3a — BA ITERATIVO su court points REALI (Hough), 2 esperimenti:
  A (ONESTO, da v2 reale, 4 round detection→match→BA, stop a plateau MPJPE):
     match 48→55/96, punti 13→15/16; residuo v2 = 6.86 px (round1 → 3.66)
     MPJPE fit 25.05→18.26 px (-27%); LOO 41.57→30.18 px (-27%)
     bone-CV 15.9→15.8% (invariato: bias calib = sistematico, CV dominato
     da rumore annotazione per-frame)
     Δtvec totale: cam_3=288mm, cam_5=435mm (le cam anomale)
     NB: court RMSE tra round NON confrontabile (insiemi obs diversi);
     confrontare solo MPJPE fit/LOO player.
  B (stabilità init): A e B convergono a |Δtvec|≈0mm → ottimo ben determinato.
```

**Relazione completa** (teoria+progetto+interpretazione, livello neofita):
`/home/lele/Desktop/CV/relazione/main.tex` → `main.pdf` (13 pp, compila con
pdflatex; babel italian non installato → nomi italiani settati a mano).
Il report LNCS max-5-pagine per la consegna è ANCORA DA FARE (è un documento
diverso dalla relazione divulgativa).

---

## Cose da fare (rimaste / migliorie)

1. ~~BA con dati REALI~~ **FATTO**: `run_ba_real.py` usa court points Hough reali,
   esperimento A parte dalla calib v2 reale (non più sintetico). I dump NON servono
   (sono scacchiere di calibrazione, non court points). Migliorìa residua: annotare
   manualmente i court corner in 1 frame/cam per superare il 49/96 match di Hough.
2. **Task 4 (bonus)**: visualizzazione su Unreal Engine — non richiesto per gruppo da 1.
3. **Test su altre azioni**: il codice è per `hpe_01`; per `hpe_02..09` servirebbe
   un detector automatico (es. YOLOv8-pose con `ultralytics`).
4. **Report finale**: max 5 pagine in formato LNCS, da consegnare 3 giorni prima dell'esame.
5. **Consegna**: GitHub Classroom (`https://classroom.github.com/a/sDy4Ysr0`).

---

## File del progetto (`project/`)

```
project/
├── data/
│   ├── annotations/_annotations.coco.json
│   ├── images/*.png (30 frame)
│   └── cameras/{camera_config_v1, camera_config_v2}/
├── utils/
│   ├── camera.py              # Modello pinhole
│   ├── coco_utils.py          # COCO parser
│   ├── court.py               # 3D points FIBA + dimensioni palasport
│   ├── triangulation.py       # DLT
│   ├── bundle_adjustment.py   # BA scipy
│   └── rectified_videos.py    # (fornito dal tutor)
├── step2_triangulation/run_triangulation.py
├── step3_bundle_adjustment/run_ba.py
├── sanity_check_cameras.py    # Validazione + planimetria
├── run_all.py                 # Entry point completo
├── requirements.txt
├── README.md
└── PROJECT_CONTEXT.md         ← questo file
```

---

## Comandi rapidi per la prossima sessione

```bash
cd /path/to/project
pip install -r requirements.txt
python3 run_all.py                          # tutto in sequenza
python3 sanity_check_cameras.py             # solo validazione
python3 step2_triangulation/run_triangulation.py
python3 step3_bundle_adjustment/run_ba.py
```

## Stile preferito dall'utente (dalle correzioni in chat)

- **Risposte in italiano**, dirette e concrete
- **Non sovrapporre** mai etichette/elementi nei plot — usare tabelle esterne se necessario
- Prima di scrivere codice **discutere l'approccio** e verificare la calibrazione (sanity check)
- Usare slide ufficiali del corso (`05 Geometry.pdf`, `08 SfM - SLAM.pdf`) come riferimento
