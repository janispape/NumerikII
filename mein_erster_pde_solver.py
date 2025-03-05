# I am absolutely clueless about the numerical handling of pdes.
# I am nonetheless interested in a naive approach similar to Euler.
# After some research I begin with a 1-dim Finite-Differences-Method
import numpy as np
import os

import EulerMaruyama
import mehrschritt
import AlphaBeta
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections import namedtuple

np.set_printoptions(precision=15, linewidth=100, suppress=True)

def finite_differenzen_waermeleitung(f, T, Omega, u0, hx, ht, args=()):
    """
    erg = finite_differenzen_wärmeleitung(f, T, u0, h, args=()):

    Euler-Verfahren mit fester Schrittweite

    IN:
        f(t, u) : muss array auf array abbilden
        T       : [t0,te]
        Omega   : [x0,xe]
        u0      : muss Zahl auf Zahl abbilden
        hx      : Raumschrittweite
        ht      : Raumschrittweite

    OUT:
        erg.y    : Lösungskomponenten (Zeilen des arrays)
        erg.t    : Zeitschritte (array)
        erg.hx   : Raumschrittweite
        erg.ht   : Zeitschrittweite
        erg.nfev : Anzahl f-Auswertungen

    t0 und te sind sicher im Output
    """

    # Start- und Endzeit/raum ermitteln
    t0, te = T
    x0, xe = Omega

    # Diskretisierung des Raums gemäß deiner Vorgabe
    # N+2 Punkte insgesamt, wobei x[0]=x0 und x[N+1]=xe
    N = int(np.floor((xe - x0) / hx))
    # Raumgitter erstellen: N+2 Punkte mit x[0]=x0 und x[N+1]=xe
    x = np.linspace(x0, xe, N + 2)

    # Diskretisierungsmatrix erstellen gemäß gewünschter Struktur
    # Matrix hat die Größe N x N für die inneren Punkte (ohne Randpunkte)
    # Die Randpunkte von u sind 0 und das auch konstant
    A_h = np.zeros(shape = (N,N))
    for i in range(N):
        A_h[i,i] = 2
    for i in range(N-1):
        A_h[i,i+1] = -1
        A_h[i+1,i] = -1

    # Modify first and last rows to enforce Neumann condition
    A_h[0, 0] = 1  # Diagonal for first row
    A_h[-1, -1] = 1  # Diagonal for last row

    A_h = 1/(hx**2) * A_h

    def A_safe(t, u_vector):
        # Zu float64 konvertieren
        u_vector = np.asarray(u_vector, dtype=np.float64)

        # Inhomogenitätsterm berechnen
        f_h = np.zeros(N, dtype=np.float64)
        for i in range(N):
            f_h[i] = f(t, x[i + 1])
        return -A_h @ u_vector + f_h

    # Anfangsbedingung für die inneren Raumpunkte (ohne Ränder)
    u0_vector = np.zeros(N)
    for i in range(N):
        # i+1, da die inneren Punkte bei Index 1 bis N im Gesamtgitter liegen
        u0_vector[i] = u0(x[i + 1])

    # eigentliches Verfahren im deterministischen Fall:
    # Adams-Moulton-Verfahren anwenden
    #erg_am = mehrschritt.LMMexp_PEC(A_safe, T, u0_vector,
    #                                3,  # k=3 für 3-Schritt-Verfahren
    #                                AlphaBeta.alphaA, AlphaBeta.betaAB,  # Prädiktor
    #                                AlphaBeta.alphaA, AlphaBeta.betaAM,  # Korrektor
    #                                ht,
    #                                3)  # 3 Korrekturen

    # eigentliches Verfahren im stochastischen Fall:
    # Euler Maruyama nutzen
    def b(t,x):
        return 5*np.ones(x.shape, dtype=x.dtype)

    erg_am = EulerMaruyama.eulermaruyama(A_safe, b, T, u0_vector, 0.0001)

    # Vollständige Lösung inklusive Randpunkte konstruieren
    # Wir haben nur die inneren Punkte berechnet, müssen also die Randwerte ergänzen
    full_y = np.zeros((N + 2, len(erg_am.t)))

    # Innere Punkte übernehmen
    full_y[1:-1, :] = erg_am.y

    # Randwerte gemäß Neumann-Randbedingungen setzen
    full_y[0, :] = full_y[1, :]  # Linker Rand
    full_y[-1, :] = full_y[-2, :]  # Rechter Rand

    # Ergebnis zurückgeben
    Result = namedtuple('Result', ['y', 't', 'x', 'hx', 'ht', 'nfev'])
    return Result(y=full_y, t=erg_am.t, x=x, hx=hx, ht=ht, nfev=erg_am.nfev)


def create_heat_equation_animation(erg, save_path=None, max_frames = 200):
    """
    Create an animation of the heat equation solution

    Parameters:
    erg: Result from finite_differenzen_waermeleitung
    save_path: Optional path to save the animation
    """
    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    y_min = np.min(erg.y)
    y_max = np.max(erg.y)
    padding = 0.1 * (y_max - y_min)

    total_frames = len(erg.t)
    if total_frames > max_frames:
        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(total_frames)

    # Set up the plot
    line, = ax.plot(erg.x, erg.y[:, 0])
    ax.set_xlim(erg.x[0], erg.x[-1])
    ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_xlabel('x')
    ax.set_ylabel('u(t,x)')
    ax.set_title('Heat Equation Solution')

    # Add time text
    time_text = ax.text(0.02, 0.95, f'Time: 0.0000', transform=ax.transAxes)

    # Initialization function
    def init():
        line.set_data(erg.x, erg.y[:, 0])
        time_text.set_text(f'Time: 0.0000')
        return line, time_text

    # Animation update function
    def update(frame):
        line.set_data(erg.x, erg.y[:, frame])
        time_text.set_text(f'Time: {erg.t[frame]:.4f}')
        return line, time_text

    # Create the animation
    # Create the animation
    anim = animation.FuncAnimation(fig, update,
                                   frames=len(frame_indices),
                                   init_func=init,
                                   blit=True,
                                   interval=100)  # 100 ms between frames

    anim.save(save_path, writer='pillow')
    plt.close(fig)

    return anim


if __name__ == "__main__":
    def u(x):
        if x < np.pi * 0.5:
            return np.cos(25 * x)
        if x < np.pi * 0.75:
            return np.cos(12.5 * np.pi) - x + np.pi * 0.5
        else:
            return np.cos(12.5 * np.pi) - np.pi * 0.25

    def f(t,x):
        return 0

    T = [0, 10]
    Omega = [0, np.pi]

    hx = 0.03
    ht = 0.4 * hx**2
    erg = finite_differenzen_waermeleitung(f, T, Omega, u, hx, ht)
    print(erg.y.shape)
    print("Berechne PDE-Lösung...")

    print(f"Berechnung abgeschlossen: {len(erg.t)} Zeitschritte")

    # Nach der Berechnung und vor der Animation
    print(f"Min-Wert in Lösung: {np.min(erg.y)}")
    print(f"Max-Wert in Lösung: {np.max(erg.y)}")
    print(f"Enthält NaN-Werte: {np.any(np.isnan(erg.y))}")
    print(f"Enthält Inf-Werte: {np.any(np.isinf(erg.y))}")

    print("Erstelle Animation...")

    create_heat_equation_animation(erg, save_path='my_animation.gif')