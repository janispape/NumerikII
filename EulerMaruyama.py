# -*- coding: utf-8 -*-
"""
Numerik 2, explizites Euler-Verfahren

"""

import numpy as np

from collections import namedtuple

def eulermaruyama(a, b, T, u0, h, args=()):
    """
    erg = euler(f, T, u0, h, args=()):

    Euler-Verfahren mit fester Schrittweite

    IN:
        a(t, u) : muss array auf array abbilden
        b(t, u) : muss array auf array abbilden
        T      : [t0,te]
        u0     : Startwert (array)
        h      : Zeitschrittweite

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

    # eigentliches Verfahren
    t = t0
    u = u0

    ttol = 1e-8 * h  # Toleranz für Schlussschritt

    nfev = 0

    while t < te:
        # Letzten Schritt evtl. verkürzen, damit man genau auf te landet
        tn = t + h
        if tn > te - ttol:
            tn = te
            h = tn - t

        # Euler-Schritt
        un = u + h * a(t, u, *args) + b(t, u, *args) * np.random.normal(0, np.sqrt(h), size=len(u))
        nfev += 1

        # neuer Zeitpunkt und neue Näherung übernehmen
        t = tn
        u = un

        # aktuelles Ergebnis an den Ergebnisvektor anhängen
        tv.append(t)
        uv.append(u)

    erg = namedtuple("erg", "t y h nfev")
    return erg(np.array(tv), np.array(uv).T, h, nfev)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    def a(t, u):
        return 5*(5-u)
    def b(t, u, sigma):
        return sigma * np.sqrt(np.abs(u + 0.01))

    def a_wrapped(t, u):
        return a(t, u, 100, 100)

    def b_wrapped(t, u):
        return b(t, u, 10)

    T = [0, 10]

    h = 0.01

    erg = eulermaruyama(a, b_wrapped, T, np.array([0]), h)

    plt.plot(erg.t, erg.y.T)
    plt.show()