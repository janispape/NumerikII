# -*- coding: utf-8 -*-
"""
Numerik 2, explizites Euler-Verfahren 

"""

import numpy as np

from collections import namedtuple


def euler(f, T, u0, h, args=()):
    """ 
    erg = euler(f, T, u0, h, args=()):
    
    Euler-Verfahren mit fester Schrittweite
       
    IN:
        f(t, u) : muss array auf array abbilden    
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
    
    ttol = 1e-8 * h # Toleranz für Schlussschritt 
    
    nfev = 0
    
    while t<te:
    # Letzten Schritt evtl. verkürzen, damit man genau auf te landet
        tn = t + h
        if tn > te-ttol:
            tn = te
            h  = tn - t
    
    # Euler-Schritt
        un = u + h * f(t, u, *args)
        nfev += 1
    
    # neuer Zeitpunkt und neue Näherung übernehmen    
        t  = tn
        u  = un
        
    # aktuelles Ergebnis an den Ergebnisvektor anhängen
        tv.append(t)
        uv.append(u)
    
    erg = namedtuple("erg", "t y h nfev") 
    return erg(np.array(tv), np.array(uv).T, h, nfev)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    def f(t, u):
        A = np.array([[-2, 5*t], [0, -1]])
        return A.dot(u)
    
    u0 = [1, 2]
    
    def uex(t):
        return np.array([10*t*np.exp(-t) - 10*np.exp(-t) + 11*np.exp(-2*t), 2*np.exp(-t)])
    
    T  = [0, 1]
    
    h  = 0.01
    
    erg = euler(f, T, u0, h)
    
    plt.plot(erg.t, erg.y.T, "b", label="euler")
    tt = np.linspace(*T, 11)
    plt.plot(tt, uex(tt).T, "r.", label = "exakt")
    plt.legend()