"""
Sanity check: plot pulito senza sovrapposizioni.
Strategia: marker leggeri nel plot, TUTTE le coordinate in tabella esterna.
"""
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.court import (fiba_court_points, COURT_COLORS,
                          COURT_L, COURT_W,
                          HALL_L, HALL_W, HALL_CENTER_X, HALL_CENTER_Y, UWB_Z)

CAMS_DIR = Path(__file__).resolve().parent / 'data' / 'cameras'

EXPECTED_HEIGHTS = {
    'cam_1': 2.40, 'cam_2': 2.35, 'cam_3': 2.60, 'cam_4': 1.80, 'cam_5': 3.05,
    'cam_6': 2.20, 'cam_7': 2.35, 'cam_8': 2.20, 'cam_12': 2.80, 'cam_13': 2.20,
}


def load_calib(cam, version='camera_config_v2'):
    p = CAMS_DIR / version / cam / 'calib' / 'camera_calib.json'
    d = json.load(open(p))
    return (np.array(d['mtx'], dtype=np.float64),
            np.array(d['dist'], dtype=np.float64).flatten(),
            np.array(d['rvecs'], dtype=np.float64).flatten(),
            np.array(d['tvecs'], dtype=np.float64).flatten())


def camera_center(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    return (-R.T @ tvec).flatten()


def main():
    centers = {}
    for cam in [f'cam_{i}' for i in [1,2,3,4,5,6,7,8,12,13]]:
        K, dist, rvec, tvec = load_calib(cam, 'camera_config_v2')
        centers[cam] = camera_center(rvec, tvec) / 1000.0

    court = fiba_court_points()

    # =========================================================================
    # Layout: plot a sinistra (large), tabelle a destra
    # =========================================================================
    fig = plt.figure(figsize=(24, 14))
    ax = fig.add_axes([0.03, 0.05, 0.62, 0.92])   # plot area
    tab_cam = fig.add_axes([0.68, 0.55, 0.30, 0.42]); tab_cam.axis('off')
    tab_corn = fig.add_axes([0.68, 0.05, 0.30, 0.48]); tab_corn.axis('off')

    # -------- Palasport con dimensioni reali
    hall_x0 = HALL_CENTER_X - HALL_L/2
    hall_y0 = HALL_CENTER_Y - HALL_W/2
    ax.add_patch(Rectangle((hall_x0, hall_y0), HALL_L, HALL_W,
                           fill=False, edgecolor='gray', linewidth=2.5, linestyle='--', zorder=1))

    # Gradinate top
    top_y0 = COURT_W/2 + 1.0
    top_y1 = HALL_CENTER_Y + HALL_W/2 - 0.3
    top_h = top_y1 - top_y0
    bleach_xs = [-20, -7, +7, +20]
    posti = [100, 160, 100]
    for i, n in enumerate(posti):
        x0, x1 = bleach_xs[i], bleach_xs[i+1]
        ax.add_patch(Rectangle((x0, top_y0), x1-x0, top_h, fc='#dde9f7',
                                ec='#8ba6c8', lw=0.7, alpha=0.5, zorder=1))
        ax.text((x0+x1)/2, top_y0 + top_h/2, f'{n} posti',
                ha='center', va='center', fontsize=8, color='#3a5478', alpha=0.7)

    # Gradinate laterali
    for side, x0, x1 in [('L', hall_x0+0.3, -COURT_L/2-0.5), ('R', +COURT_L/2+0.5, hall_x0+HALL_L-0.3)]:
        ax.add_patch(Rectangle((x0, -COURT_W/2-1.5), x1-x0, COURT_W+2.0,
                                fc='#dde9f7', ec='#8ba6c8', lw=0.7, alpha=0.5, zorder=1))
        ax.text((x0+x1)/2, 0, '51 posti', ha='center', va='center',
                fontsize=8, color='#3a5478', rotation=90, alpha=0.7)

    # Panchine + Tavolo arbitri
    bench_y = top_y0 - 1.2
    bench_h = 1.0
    ax.add_patch(Rectangle((-12, bench_y), 8, bench_h, fc='#fff3d6', ec='#a87b00', lw=0.6, alpha=0.7))
    ax.text(-8, bench_y + bench_h/2, 'PANCHINA', ha='center', va='center', fontsize=7, alpha=0.8)
    ax.add_patch(Rectangle((4, bench_y), 8, bench_h, fc='#fff3d6', ec='#a87b00', lw=0.6, alpha=0.7))
    ax.text(8, bench_y + bench_h/2, 'PANCHINA', ha='center', va='center', fontsize=7, alpha=0.8)
    ax.add_patch(Rectangle((-3, bench_y), 6, bench_h, fc='#ffe0d6', ec='#a83300', lw=0.6, alpha=0.7))
    ax.text(0, bench_y + bench_h/2, 'TAVOLO', ha='center', va='center', fontsize=7, alpha=0.8)

    # Linee campo
    drawn = set()
    for name, pts in court.items():
        if name in ('corners','court_corners','paint_corners','t_midcourt',
                    'midcourt_circle','baskets','uwb_ceiling','hall_perimeter'):
            continue
        base = name.replace('_left','').replace('_right','')
        lbl = base if base not in drawn else None
        drawn.add(base)
        ax.scatter(pts[:,0], pts[:,1], s=8, c=COURT_COLORS.get(name,'gray'),
                   label=lbl, alpha=0.7, zorder=2)

    # ANGOLI con SOLO sigla (coords in tabella)
    cc_list = court['court_corners']
    cc_names = ['SW','SE','NE','NW']
    for nm, p in zip(cc_names, cc_list):
        ax.scatter(p[0], p[1], s=130, marker='s', c='black',
                   edgecolor='yellow', linewidth=2, zorder=8)
        # etichetta breve, posizionata in direzione OPPOSTA al campo
        dx = -15 if p[0] < 0 else 15
        dy = -12 if p[1] < 0 else 12
        ha = 'right' if p[0] < 0 else 'left'
        ax.annotate(nm, (p[0], p[1]), xytext=(dx, dy), textcoords='offset points',
                    fontsize=11, fontweight='bold', ha=ha, color='black',
                    bbox=dict(boxstyle='circle,pad=0.25', fc='yellow', ec='black', lw=1))

    # Paint corners: solo punti (no label)
    for pc in court['paint_corners']:
        ax.scatter(pc[0], pc[1], s=45, marker='s', c='black',
                   edgecolor='cyan', linewidth=1.0, zorder=7)

    # T-intersection
    for tp in court['t_midcourt']:
        ax.scatter(tp[0], tp[1], s=70, marker='D', c='magenta', zorder=7)

    # Canestri - solo simbolo
    for b in court['baskets']:
        ax.scatter(b[0], b[1], s=200, marker='*', c='gold',
                   edgecolor='black', linewidth=1, zorder=7)

    # UWB
    for u in court['uwb_ceiling']:
        ax.scatter(u[0], u[1], s=400, marker='o', facecolor='none',
                   edgecolor='gold', linewidth=2.5, zorder=7)
        ax.text(u[0], u[1], '10,0', ha='center', va='center', fontsize=8,
                color='darkgoldenrod', fontweight='bold', zorder=8)

    # CAMERE - solo cerchio rosso + cam_id come label corto
    # Etichette posizionate FAR FROM the box, lontano da bleachers
    cam_label_offsets = {
        'cam_1':  (-3,   0,  'right'),    # left mid, label a sinistra fuori palasport
        'cam_2':  (-3,  -2,  'right'),    # bottom-left, label fuori
        'cam_3':  ( 3,   2,  'left'),     # top-right corner, label fuori
        'cam_4':  ( 3,  -2,  'left'),     # bottom-right, label fuori
        'cam_5':  ( 3,   0,  'left'),     # right mid, label fuori
        'cam_6':  ( 3,   3,  'left'),     # top edge right, label sopra
        'cam_7':  ( 0,  -3,  'center'),   # bottom mid, label sotto
        'cam_8':  (-3,   3,  'right'),    # top edge left, label sopra-sx
        'cam_12': (-3,   2,  'right'),    # top corner left, label fuori
        'cam_13': ( 0,   3,  'center'),   # top edge mid, label sopra
    }
    for cam, C in centers.items():
        ax.scatter(C[0], C[1], s=300, c='red', edgecolor='black', linewidth=1.5, zorder=9)
        dx_units, dy_units, ha = cam_label_offsets.get(cam, (3, 3, 'left'))
        ax.annotate(cam, (C[0], C[1]),
                    xytext=(C[0] + dx_units, C[1] + dy_units),
                    fontsize=11, fontweight='bold', ha=ha, va='center', color='darkred',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#fff8dc', ec='red', lw=1.2),
                    zorder=10,
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8, alpha=0.7))

    # Origine
    ax.axhline(0, color='gray', lw=0.3, ls=':')
    ax.axvline(0, color='gray', lw=0.3, ls=':')
    ax.scatter(0, 0, s=110, marker='+', c='black', zorder=8)
    ax.text(0.5, -0.5, '(0,0,0)', fontsize=8, fontweight='bold')

    # Quote campo
    ax.annotate('', xy=(COURT_L/2, -COURT_W/2-2.6), xytext=(-COURT_L/2, -COURT_W/2-2.6),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.0))
    ax.text(0, -COURT_W/2-3.2, f'{COURT_L}m', color='blue', ha='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(-COURT_L/2-0.6, COURT_W/2), xytext=(-COURT_L/2-0.6, -COURT_W/2),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=1.0))
    ax.text(-COURT_L/2-1.5, 0, f'{COURT_W}m', color='blue', ha='center', va='center',
            rotation=90, fontsize=10, fontweight='bold')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'Sanbapolis — palasport {HALL_L:.1f}×{HALL_W:.1f}m (centro {HALL_CENTER_X:+.2f}, {HALL_CENTER_Y:+.2f})', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(alpha=0.15)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.85)
    ax.set_xlim(-32, 32)
    ax.set_ylim(-22, 26)

    # =========================================================================
    # TABELLA CAMERE (alto a destra)
    # =========================================================================
    tab_cam.text(0.5, 1.0, 'CAMERE (v2)', ha='center', fontsize=13, fontweight='bold',
                 transform=tab_cam.transAxes)
    header = f'{"cam":<7}{"X":>8}{"Y":>8}{"Z":>8} {"h_flr":>7}{"  h_gt":>7}'
    tab_cam.text(0.0, 0.93, header, family='monospace', fontsize=9,
                 fontweight='bold', transform=tab_cam.transAxes)
    tab_cam.text(0.0, 0.91, '-'*55, family='monospace', fontsize=9, transform=tab_cam.transAxes)
    for i, (cam, C) in enumerate(centers.items()):
        h_flr = C[2]                      # pavimento a Z=0 → altezza = Z
        h_exp = EXPECTED_HEIGHTS[cam]      # GT planimetria (riferimento, vedi NB)
        ok = '✓' if 4.0 <= C[2] <= 9.0 else '⚠'   # mount alto plausibile (sopra l'azione)
        row = f'{cam:<7}{C[0]:>+8.2f}{C[1]:>+8.2f}{C[2]:>+8.2f} {h_flr:>+7.2f}{h_exp:>+7.2f} {ok}'
        tab_cam.text(0.0, 0.85 - i*0.07, row, family='monospace', fontsize=9.5,
                     transform=tab_cam.transAxes)
    tab_cam.text(0.0, 0.85 - 10*0.07,
                 'NB: pavimento a Z=0 (confermato: piedi giocatori triangolano a\n'
                 'Z~0.15m). h_flr=Z = altezza camera sul pavimento (~6-7m, mount\n'
                 'alto sui muri). Le altezze GT planimetria (h_exp) NON sono coerenti\n'
                 'col calib v2; si fida del calib (triangolazione fisicamente corretta).',
                 fontsize=7.5, style='italic', color='gray', transform=tab_cam.transAxes)

    # =========================================================================
    # TABELLA ANGOLI CAMPO (basso a destra)
    # =========================================================================
    tab_corn.text(0.5, 1.0, 'ANGOLI CAMPO (FIBA, Z=0)', ha='center', fontsize=13,
                  fontweight='bold', transform=tab_corn.transAxes)
    tab_corn.text(0.0, 0.94, f'{"nome":<10}{"X":>9}{"Y":>9}{"Z":>9}',
                  family='monospace', fontsize=9, fontweight='bold',
                  transform=tab_corn.transAxes)
    tab_corn.text(0.0, 0.92, '-'*40, family='monospace', fontsize=9, transform=tab_corn.transAxes)

    rows = []
    for nm, p in zip(['SW','SE','NE','NW'], court['court_corners']):
        rows.append((f'court_{nm}', p))
    for i, pc in enumerate(court['paint_corners']):
        side = 'L' if pc[0]<0 else 'R'
        ya = 'B' if pc[1]<0 else 'T'
        is_baseline = abs(abs(pc[0]) - COURT_L/2) < 0.1
        kind = 'base' if is_baseline else 'ft'
        rows.append((f'paint_{side}{ya}_{kind}', pc))
    for i, tp in enumerate(court['t_midcourt']):
        side = 'B' if tp[1]<0 else 'T'
        rows.append((f'midcourt_{side}', tp))
    for i, b in enumerate(court['baskets']):
        side = 'L' if b[0]<0 else 'R'
        rows.append((f'basket_{side}', b))

    for i, (nm, p) in enumerate(rows):
        row = f'{nm:<10}{p[0]:>+9.2f}{p[1]:>+9.2f}{p[2]:>+9.2f}'
        tab_corn.text(0.0, 0.88 - i*0.045, row, family='monospace', fontsize=9,
                      transform=tab_corn.transAxes)

    # Legenda simboli
    tab_corn.text(0.0, 0.05,
                  '■ giallo = court corner\n'
                  '■ cyan   = paint corner\n'
                  '◆ magenta= midcourt T-intersect\n'
                  '★ oro    = canestro (Z=3.05)\n'
                  '○ oro    = UWB soffitto (Z=10)\n'
                  '● rosso  = camera v2',
                  family='monospace', fontsize=8, transform=tab_corn.transAxes,
                  bbox=dict(boxstyle='round,pad=0.4', fc='#fafafa', ec='gray'))

    out = str(Path(__file__).resolve().parent / 'sanity_check_cameras.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f'Plot 2D salvato: {out}')


if __name__ == '__main__':
    main()
