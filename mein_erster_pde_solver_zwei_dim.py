# I am absolutely clueless about the numerical handling of pdes.
# I am nonetheless interested in a naive approach similar to Euler.
# After some research I begin with a 2-dim Finite-Differences-Method
import numpy as np
import os
import mehrschritt
import AlphaBeta
import scipy.linalg
import rungekutta
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from collections import namedtuple

np.set_printoptions(precision=15, linewidth=100, suppress=True)

def finite_differenzen_waermeleitung(f, T, Omega, u0, h_omega, ht, args=()):
    """
    erg = finite_differenzen_wärmeleitung(f, T, u0, h, args=()):

    Euler-Verfahren mit fester Schrittweite

    IN:
        f(t, u) : muss array auf array abbilden
        T       : [t0,te]
        Omega   : [x0,xe,y0,ye]
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

    # Start- und Endpunkt des Raumes ermitteln
    x0, xe, y0, ye = Omega

    # Diskretisierung des Raums gemäß deiner Vorgabe
    # N+2 Punkte insgesamt, wobei x[0]=x0 und x[N+1]=xe
    Nx = int(np.floor((xe - x0) / h_omega))
    Ny = int(np.floor((ye - y0) / h_omega))

    # Raumgitter erstellen: N+2 Punkte mit x[0]=x0 und x[N+1]=xe
    x = np.linspace(x0, xe, Nx + 2)
    y = np.linspace(y0, ye, Ny + 2)

    # Diskretisierungsmatrix erstellen gemäß gewünschter Struktur
    # Matrix hat die Größe N x N für die inneren Punkte (ohne Randpunkte)
    # Die Randpunkte von u sind 0 und das auch konstant
    N = Nx * Ny
    A_h = np.zeros(shape = (N,N))

    for i in range(Nx):
        for j in range(Ny):
            # Aktuelle Gitterpunkt-Indizes in linearen Index umwandeln
            idx = i * Ny + j

            # Standardwert für Hauptdiagonale: -4
            diag_value = -4

            # Nachbarn in x-Richtung
            if i > 0:  # Linker Nachbar
                A_h[idx, (i - 1) * Ny + j] = 1
            else:  # Am linken Rand: Neumann-Bedingung
                diag_value += 1  # Reflektierter Punkt addiert zum Diagonalelement

            if i < Nx - 1:  # Rechter Nachbar
                A_h[idx, (i + 1) * Ny + j] = 1
            else:  # Am rechten Rand: Neumann-Bedingung
                diag_value += 1

            # Nachbarn in y-Richtung
            if j > 0:  # Unterer Nachbar
                A_h[idx, i * Ny + (j - 1)] = 1
            else:  # Am unteren Rand: Neumann-Bedingung
                diag_value += 1

            if j < Ny - 1:  # Oberer Nachbar
                A_h[idx, i * Ny + (j + 1)] = 1
            else:  # Am oberen Rand: Neumann-Bedingung
                diag_value += 1

            # Hauptdiagonale setzen (angepasst an Ränder)
            A_h[idx, idx] = diag_value

    A_h = 1/(h_omega**2) * A_h

    print(f"Matrix norm: {np.linalg.norm(A_h)}")
    print(f"Matrix min: {np.min(A_h)}, max: {np.max(A_h)}")

    def A(t, u_vector):
        # Inhomogenitätsterm berechnen
        f_h = np.zeros(Nx * Ny)
        idx = 0
        for i in range(1, Nx + 1):  # Innere Punkte
            for j in range(1, Ny + 1):
                f_h[idx] = f(t, x[i], y[j])
                idx += 1

        # Wärmeleitungsgleichung: u_t = Δu + f
        return -A_h @ u_vector + f_h



    test_vector = np.ones(N)
    test_erg = -A_h @ test_vector
    print(np.min(test_erg))
    print(np.max(test_erg))

    # Anfangsbedingung für die inneren Raumpunkte (ohne Ränder)
    u0_vector = np.zeros(N)
    for i in range(Nx):
        for j in range(Ny):
            u0_vector[i*Ny + j] = u0(x[i + 1], y[j+1])


    # Butcher tableau for classical RK4
    RK4 = np.array([
        [0, 0, 0, 0, 0],
        [0.5, 0.5, 0, 0, 0],
        [0.5, 0, 0.5, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 1 / 6, 1 / 3, 1 / 3, 1 / 6]
    ])

    # eigentliches Verfahren
    # Adams-Moulton-Verfahren anwenden
    erg_rk = rungekutta.rungekutta(A, T, u0_vector, ht, RK4)

    # Vollständige Lösung inklusive Randpunkte konstruieren
    # Wir haben nur die inneren Punkte berechnet, müssen also die Randwerte ergänzen
    full_y = np.zeros(((Nx+2)*(Ny+2), len(erg_rk.t)))

    # Innere Punkte übernehmen
    for t_idx in range(len(erg_rk.t)):
        for i in range(Nx):
            for j in range(Ny):
                inner_idx = i * Ny + j
                full_idx = (i + 1) * (Ny + 2) + (j + 1)
                full_y[full_idx, t_idx] = erg_rk.y[inner_idx, t_idx]

    # Randwerte gemäß Neumann-Randbedingungen setzen
    # Linker und rechter Rand
    for j in range(1, Ny + 1):
        for t_idx in range(len(erg_rk.t)):
            full_y[j, t_idx] = full_y[Ny + 2 + j, t_idx]  # Linker Rand
            full_y[(Nx + 1) * (Ny + 2) + j, t_idx] = full_y[Nx * (Ny + 2) + j, t_idx]  # Rechter Rand

    # Unterer und oberer Rand
    for i in range(Ny + 2):
        for t_idx in range(len(erg_rk.t)):
            full_y[i * (Ny + 2), t_idx] = full_y[i * (Ny + 2) + 1, t_idx]  # Unterer Rand
            full_y[i * (Ny + 2) + (Ny + 1), t_idx] = full_y[i * (Ny + 2) + Ny, t_idx]  # Oberer Rand

    # Ergebnis zurückgeben
    Result = namedtuple('Result', ['y', 't', 'x', 'h_omega', 'ht', 'nfev'])
    return Result(y=full_y, t=erg_rk.t, x=x, h_omega=h_omega, ht=ht, nfev=erg_rk.nfev)


def create_heat_equation_animation(erg, save_path=None):
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
    anim = animation.FuncAnimation(fig, update,
                                   frames=len(erg.t),
                                   init_func=init,
                                   blit=True,
                                   interval=100)  # 50 ms between frames

    anim.save(save_path, writer='pillow')
    plt.close(fig)

    return anim


if __name__ == "__main__":
    def u(x,y):
        return np.sin(x)*(y+1)

    def f(t,x,y):
        return 0

    T = [0, 0.01]
    Omega = [0, 1, 0, 1]

    h_omega = 0.05
    h_t = 0.1 * h_omega**2
    erg = finite_differenzen_waermeleitung(f, T, Omega, u, h_omega, h_t)
    print(erg.y.shape)
    print("Berechne PDE-Lösung...")

    print(f"Berechnung abgeschlossen: {len(erg.t)} Zeitschritte")

    # Nach der Berechnung und vor der Animation
    print(f"Min-Wert in Lösung: {np.min(erg.y)}")
    print(f"Max-Wert in Lösung: {np.max(erg.y)}")
    print(f"Enthält NaN-Werte: {np.any(np.isnan(erg.y))}")
    print(f"Enthält Inf-Werte: {np.any(np.isinf(erg.y))}")

    print("Erstelle Animation...")

    x0, xe, y0, ye = Omega

    # Diskretisierung des Raums gemäß deiner Vorgabe
    # N+2 Punkte insgesamt, wobei x[0]=x0 und x[N+1]=xe
    Nx = int(np.floor((xe - x0) / h_omega))
    Ny = int(np.floor((ye - y0) / h_omega))

    # Raumgitter erstellen: N+2 Punkte mit x[0]=x0 und x[N+1]=xe
    x = np.linspace(x0, xe, Nx + 2)
    y = np.linspace(y0, ye, Ny + 2)
    # Create a meshgrid for plotting
    X, Y = np.meshgrid(x[1:-1], y[1:-1])  # Use only interior points

    # Reshape the initial condition from 1D to 2D
    Z = erg.y[:Nx * Ny, 0].reshape(Nx, Ny).T  # Using only the first timestep (u0)

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')

    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('u0(x,y)')
    ax.set_title('Initial Condition u0')

    plt.tight_layout()
    plt.show()