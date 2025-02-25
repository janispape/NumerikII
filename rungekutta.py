# -*- coding: utf-8 -*-
"""
Numerik 2, Collatz-Verfahren

"""

import numpy as np

from collections import namedtuple

def schema(butcher):
    if not (len(butcher.shape) == 2 and butcher.shape[0] == butcher.shape[1] and butcher.shape[0] >= 2):
        raise Exception(f"Array must be quadratic and at least 2x2. Got shape: {butcher.shape}")
    else:
        c = butcher[:-1, 0]  # First column without last element
        b = butcher[-1, 1:]  # Last row without first element
        A = butcher[:-1, 1:]  # Everything else
        A_row_sums = A.sum(axis=1)
    if not np.allclose(c, A_row_sums, rtol=1e-14, atol=1e-14):
        raise Exception("Each c_i must equal the sum of the i-th row of A")
    return A,b,c

def rungekutta(f, T, u0, h, butcher, args=()):
    """
    erg = rungekutta(f, T, u0, h, butcher, args=()):

    Runge-Kutta-Verfahren mit fester Schrittweite

    IN:
        f(t, u) : muss array auf array abbilden
        T      : [t0,te]
        u0     : Startwert (array)
        h      : Zeitschrittweite
        butcher : Butcher-Schema, welches das Verfahren präzisiert

    OUT:
        erg.y    : Lösungskomponenten (Zeilen des arrays)
        erg.t    : Zeitschritte (array)
        erg.h    : Zeitschrittweite
        erg.nfev : Anzahl f-Auswertungen

    t0 und te sind sicher im Output
    """

    # Start- und Endzeit ermitteln
    t0 = T[0]
    te = T[1]

    # Anfangswert ggf. auf Zeilenvektor transformieren
    u0 = np.array([u0]).flatten()

    # Anfangszeit und -wert in Lösungsliste eintragen
    tv = [t0]
    uv = [u0]

    # Butcher-Schema auflösen
    A,b,c = schema(butcher)

    # eigentliches Verfahren
    t = t0
    u = u0

    ttol = 1e-8 * h  # Toleranz für Schlussschritt

    nfev = 0
    n = len(u0)
    k = np.zeros((butcher.shape[0]-1, n))

    while t < te:
        # Letzten Schritt evtl. verkürzen, damit man genau auf te landet
        tn = t + h
        if tn > te - ttol:
            tn = te
            h = tn - t

        # Nun der Schritt: erst die k_i in einer Liste speichern
        k = np.zeros((butcher.shape[0]-1,n))
        for i in range(butcher.shape[0]-1):
            sum = np.zeros(n)
            for j in range(i):
                sum += A[i,j] * k[j,:]
            k[i,:]=f(t+ c[i] * h, u + h * sum)
            nfev += 1

        un = np.copy(u)
        for i in range(butcher.shape[0]-1):
            un += h * b[i]*k[i]

        # neuer Zeitpunkt und neue Näherung übernehmen
        t = tn
        u = np.copy(un)

        # aktuelles Ergebnis an den Ergebnisvektor anhängen
        tv.append(t)
        uv.append(u)

    erg = namedtuple("erg", "t y h nfev")
    return erg(np.array(tv), np.array(uv).T, h, nfev)


def rungekutta_SWS(f, T, u0, butcher_erw, p, p_hut, lok_tol, args=()):
    """
    erg = rungekutta(f, T, u0, h, butcher_erw, args=()):

    Runge-Kutta-Verfahren mit fester Schrittweite

    IN:
        f(t, u) : muss array auf array abbilden
        T      : [t0,te]
        u0     : Startwert (array)
        h      : Zeitschrittweite
        butcher : Butcher-Schema, welches das Verfahren präzisiert

    OUT:
        erg.y    : Lösungskomponenten (Zeilen des arrays)
        erg.t    : Zeitschritte (array)
        erg.h    : Zeitschrittweite
        erg.nfev : Anzahl f-Auswertungen

    t0 und te sind sicher im Output
    """

    # Start- und Endzeit ermitteln
    t0 = T[0]
    te = T[1]

    # Speicher für Fehlbergtrick, gleichzeitig erstes k1
    speicher = f(t0, u0)

    # Anfangswert ggf. auf Zeilenvektor transformieren
    u0 = np.array([u0]).flatten()

    # Anfangszeit und -wert in Lösungsliste eintragen
    tv = [t0]
    uv = [u0]
    hv = []

    # Butcher-Schema auflösen
    A,b,c = schema(butcher_erw[:-1])
    b_hut = butcher_erw[-1, 1:]

    # eigentliches Verfahren
    t = t0
    u = u0

    ttol = 1e-8 * lok_tol  # Toleranz für Schlussschritt

    nfev = 0
    n = len(u0)

    while t < te:
        # Letzten Schritt evtl. verkürzen, damit man genau auf te landet
        h = lok_tol
        tn = t + h
        if tn > te - ttol:
            tn = te
            h = tn - t

        iter_count = 0
        while True:
            # Nun der Schritt: erst die k_i in einer Liste speichern
            k = np.zeros((butcher_erw.shape[0] - 2, n))

            k[0, :] = speicher

            for i in range(1, butcher_erw.shape[0] - 2):
                sum = np.zeros(n)
                for j in range(i):
                    sum += A[i, j] * k[j, :]
                k[i, :] = f(t + c[i] * h, u + h * sum)
                nfev += 1

            un = np.copy(u)
            un_hut = np.copy(u)

            # erst un ermitteln
            for i in range(butcher_erw.shape[0] - 2):
                un += h * b[i] * k[i]

            # dann un_hut ermitteln
            for i in range(butcher_erw.shape[0] - 2):
                un_hut += h * b_hut[i] * k[i]

            l_quer_fehler = np.linalg.norm(un_hut - un)
            print(f"l_quer_fehler: {l_quer_fehler}, lok_tol: {lok_tol}, h: {h}")

            if l_quer_fehler <= lok_tol:
                print("Schritt akzeptiert")
                hv.append(h)
                speicher = k[-1, :]
                break

            h_opt = h_opt = h * (lok_tol/l_quer_fehler)**(1/(p_hut + 1))
            print(f"Neue Schrittweite h_opt: {h_opt}")
            h = h_opt

        # neuer Zeitpunkt und neue Näherung übernehmen
        t = tn
        u = np.copy(un)

        # aktuelles Ergebnis an den Ergebnisvektor anhängen
        tv.append(t)
        uv.append(u)

    erg = namedtuple("erg", "t y h nfev steps")
    return erg(np.array(tv), np.array(uv).T, h, nfev, np.array(hv))




if __name__ == "__main__":
    import matplotlib.pyplot as plt


    def f(t, u):
        A = np.array([[-2, 5 * t], [0, -1]])
        return A.dot(u)


    u0 = [1, 2]


    def uex(t):
        return np.array([10 * t * np.exp(-t) - 10 * np.exp(-t) + 11 * np.exp(-2 * t), 2 * np.exp(-t)])


    T = [0, 1]

    h = 0.01

    erg = rungekutta(f, T, u0, 1,  np.array([[0,0],[0,1]]), h)

    plt.plot(erg.t, erg.y.T, "b", label="collatz")
    tt = np.linspace(*T, 11)
    plt.plot(tt, uex(tt).T, "r.", label="exakt")
    plt.legend()