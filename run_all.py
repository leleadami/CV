"""Entry point: esegue tutto il progetto in sequenza.

Pipeline:
  1. Sanity check delle camere v2 + planimetria → PNG
  2. Triangolazione DLT multi-view (step 2)
  3. Bundle Adjustment (step 3)

Output finali:
  - sanity_check_cameras.png
  - step2_triangulation/output/results_3d.json + plot
  - step3_bundle_adjustment/output/ba_summary.json + plot
"""
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).parent
STEPS = [
    ('Sanity check camere + planimetria', PROJECT / 'sanity_check_cameras.py'),
    # Step 2.1 — Rettificazione video (script prof). Opzionale, lungo (~5min).
    # Mantieni commentato: triangulation fa undistortion on-the-fly via cv2.undistortPoints.
    # ('Step 2.1 — Rettificazione video (script prof)', PROJECT / 'step2_triangulation' / 'rectify_videos.py'),
    ('Step 2 — Triangolazione DLT', PROJECT / 'step2_triangulation' / 'run_triangulation.py'),
    ('Step 3 — Bundle Adjustment (sintetico, baseline)', PROJECT / 'step3_bundle_adjustment' / 'run_ba.py'),
    ('Step 3 — Bundle Adjustment (court detector reale)', PROJECT / 'step3_bundle_adjustment' / 'run_ba_real.py'),
]


def main():
    for title, script in STEPS:
        print('\n' + '='*70)
        print(f'>>> {title}')
        print('='*70)
        r = subprocess.run([sys.executable, str(script)], cwd=str(PROJECT))
        if r.returncode != 0:
            print(f'!! FALLITO: {script}')
            sys.exit(1)
    print('\n' + '='*70)
    print('TUTTI GLI STEP COMPLETATI ✓')
    print('='*70)


if __name__ == '__main__':
    main()
