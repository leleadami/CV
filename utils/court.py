"""Punti 3D del campo basket FIBA, origine al centro campo, Z=0 (pavimento).

Dimensioni FIBA standard:
  - Campo: 28m × 15m
  - 3-point line: raggio 6.75m dal canestro
  - Canestro a 1.575m dalla baseline
  - Free throw line: 5.80m dalla baseline, FT circle r=1.80m
  - Restricted area (key): 4.90m × 5.80m
  - Center circle: raggio 1.80m
"""
import numpy as np


COURT_L = 28.0
COURT_W = 15.0
R_3PT = 6.75
BASKET_OFFSET = 1.575
KEY_W = 4.90
KEY_L = 5.80
R_CIRC = 1.80

# Palasport (Sanbapolis) — dimensioni ricavate dai centri camera v2
# (le camere sono montate sulle pareti del palasport)
HALL_L = 47.31     # ampiezza X effettiva (cam_1=-24.72 a cam_3=+22.59)
HALL_W = 29.61     # ampiezza Y effettiva (cam_2=-11.01 a cam_8=+18.60)
HALL_CENTER_X = -1.06   # centro palasport NON all'origine
HALL_CENTER_Y = 3.80    # campo basket spostato verso il bottom (gradinate top)
HALL_Y_OFFSET = HALL_CENTER_Y  # alias per retro-compatibilità
UWB_Z = 10.0       # altezza UWB a soffitto


def fiba_court_points():
    """Restituisce dict {nome: (N,3)} con punti 3D del campo basket FIBA.

    Include una chiave speciale 'corners' con i punti notevoli (angoli, T-line)
    da evidenziare separatamente per BA.
    """
    pts = {}

    # 4 angoli campo (notevoli) + dizionario nomi
    court_corners = np.array([
        [-COURT_L/2, -COURT_W/2, 0],  # corner SW
        [+COURT_L/2, -COURT_W/2, 0],  # corner SE
        [+COURT_L/2, +COURT_W/2, 0],  # corner NE
        [-COURT_L/2, +COURT_W/2, 0],  # corner NW
    ])
    pts['court_corners'] = court_corners  # 4 punti campo

    # Angoli area (paint corners) — 4 per ogni lato, totale 8
    paint_corners = []
    for sign in [-1, +1]:
        x_base = sign * COURT_L/2
        x_ft = sign * (COURT_L/2 - KEY_L)
        for y in [-KEY_W/2, +KEY_W/2]:
            paint_corners.append([x_base, y, 0])  # baseline-side
            paint_corners.append([x_ft, y, 0])    # free-throw-line-side
    paint_corners = np.array(paint_corners)
    pts['paint_corners'] = paint_corners

    # T-intersection: midcourt × sideline
    t_inter = np.array([[0, -COURT_W/2, 0], [0, +COURT_W/2, 0]])
    pts['t_midcourt'] = t_inter

    # Intersezione linea metà campo × cerchio centro
    midcircle_t = np.array([[0, -R_CIRC, 0], [0, +R_CIRC, 0]])
    pts['midcourt_circle'] = midcircle_t

    # Tutti i corner insieme (per BA)
    pts['corners'] = np.vstack([court_corners, paint_corners, t_inter, midcircle_t])

    # Perimetro (linea continua)
    corners = np.array([[-COURT_L/2, -COURT_W/2], [+COURT_L/2, -COURT_W/2],
                        [+COURT_L/2, +COURT_W/2], [-COURT_L/2, +COURT_W/2]])
    perim = []
    for i in range(4):
        a, b = corners[i], corners[(i+1) % 4]
        for t in np.linspace(0, 1, 15, endpoint=False):
            perim.append(a + t * (b - a))
    pts['perimeter'] = np.column_stack([perim, np.zeros(len(perim))])

    # Linea metà campo
    midcourt = np.column_stack([np.zeros(11), np.linspace(-COURT_W/2, +COURT_W/2, 11), np.zeros(11)])
    pts['midcourt_line'] = midcourt

    # Center circle
    th = np.linspace(0, 2*np.pi, 24, endpoint=False)
    cc = np.column_stack([R_CIRC*np.cos(th), R_CIRC*np.sin(th), np.zeros_like(th)])
    pts['center_circle'] = cc

    # Aree, free throw circles, archi 3 punti (left + right)
    for side, sign in [('left', -1), ('right', +1)]:
        x_base = sign * COURT_L/2
        x_ft = sign * (COURT_L/2 - KEY_L)
        rect = []
        for t in np.linspace(0, 1, 8):
            rect.append([x_base, -KEY_W/2 + t*KEY_W])
        for t in np.linspace(0, 1, 10):
            rect.append([x_base + (x_ft - x_base)*t, -KEY_W/2])
        for t in np.linspace(0, 1, 10):
            rect.append([x_base + (x_ft - x_base)*t, +KEY_W/2])
        rect = np.array(rect)
        pts[f'paint_{side}'] = np.column_stack([rect, np.zeros(len(rect))])

        ft = np.column_stack([x_ft + R_CIRC*np.cos(th), R_CIRC*np.sin(th), np.zeros_like(th)])
        pts[f'ft_circle_{side}'] = ft

        x_basket = sign * (COURT_L/2 - BASKET_OFFSET)
        ang = np.linspace(-np.pi/2, np.pi/2, 24) if sign == -1 else np.linspace(np.pi/2, 3*np.pi/2, 24)
        arc3 = np.column_stack([x_basket + R_3PT*np.cos(ang),
                                R_3PT*np.sin(ang), np.zeros_like(ang)])
        pts[f'three_pt_{side}'] = arc3

    # Canestri (basket positions, Z=3.05m altezza canestro FIBA)
    pts['baskets'] = np.array([
        [-(COURT_L/2 - BASKET_OFFSET), 0, 3.05],
        [+(COURT_L/2 - BASKET_OFFSET), 0, 3.05],
    ])

    # UWB ceiling anchors (3 cerchi gialli nell'immagine, a Z=10m)
    pts['uwb_ceiling'] = np.array([
        [-COURT_L/4, 0, UWB_Z],   # left
        [0, 0, UWB_Z],            # center
        [+COURT_L/4, 0, UWB_Z],   # right
    ])

    # Perimetro palasport (dimensioni esterne)
    # Centro Y spostato perché campo non centrato (gradinate top)
    hall_corners = np.array([
        [-HALL_L/2, -HALL_W/2 + HALL_Y_OFFSET],
        [+HALL_L/2, -HALL_W/2 + HALL_Y_OFFSET],
        [+HALL_L/2, +HALL_W/2 + HALL_Y_OFFSET],
        [-HALL_L/2, +HALL_W/2 + HALL_Y_OFFSET],
    ])
    hall = []
    for i in range(4):
        a, b = hall_corners[i], hall_corners[(i+1) % 4]
        for t in np.linspace(0, 1, 25, endpoint=False):
            hall.append(a + t * (b - a))
    pts['hall_perimeter'] = np.column_stack([hall, np.zeros(len(hall))])

    return pts


COURT_COLORS = {
    'perimeter': '#1f77b4', 'midcourt_line': '#1f77b4', 'center_circle': '#1f77b4',
    'paint_left': '#ff7f0e', 'paint_right': '#ff7f0e',
    'ft_circle_left': '#2ca02c', 'ft_circle_right': '#2ca02c',
    'three_pt_left': '#d62728', 'three_pt_right': '#d62728',
    'corners': 'black',
}


def draw_court_3d(ax, z=0.0, alpha=0.5, show_corners=True):
    """Disegna i punti del campo su un Axes3D (mette tutti i punti a Z=z metri)."""
    pts = fiba_court_points()
    for name, P in pts.items():
        if name == 'corners':
            if show_corners:
                ax.scatter(P[:,0], P[:,1], P[:,2] + z, s=60, c='black',
                           marker='s', edgecolor='yellow', linewidth=1.2, zorder=10)
        else:
            ax.scatter(P[:,0], P[:,1], P[:,2] + z, s=4,
                       c=COURT_COLORS.get(name, 'gray'), alpha=alpha)
