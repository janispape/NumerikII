# -*- coding: utf-8 -*-
"""
Numerik 2, explizites Euler-Verfahren 

"""

import numpy as np

from collections import namedtuple

def rk(f, T, u0, abc, h, args=()):
    """ 
    erg = rk(f, T, u0, h, args=()):
    
    Euler-Verfahren mit fester Schrittweite
       
    IN:
        f(t, x) : muss array auf array abbilden    
        T      : [t0,te]       
        u0     : Startwert (array)
        h      : Zeitschrittweite
        
    OUT:
        erg.y    : Lösungskomponenten (Zeilen des arrays)
        erg.t    : Zeitschritte (array)
        erg.abc  : Butcher-Schema
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
    
    # Stufenzahl und Koeffizienten bestimmen
    abc = np.array(abc)
    s = abc.shape[0] - 1
    a = abc[:-1, 1:]
    c = abc[:-1, 0 ]
    b = abc[ -1, 1:]

    # Systemdimension bestimmen und k vorbesetzen
    d = len(u0)
    k = np.zeros((d,s))    
    
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
    
    # eigentlicher Verfahrensschritt
        for i in range(s):
            tt = t + c[i]*h
            uu = u + h * k[:,:i].dot(a[i,:i])
                        
            k[:,i] = f(tt, uu, *args)
            nfev += 1
    
        un = u + h * k.dot(b)
    
    # neuer Zeitpunkt und neue Näherung übernehmen    
        t  = tn
        u  = un
        
    # aktuelles Ergebnis an den Ergebnisvektor anhängen
        tv.append(t)
        uv.append(u)
    
    erg = namedtuple("erg", "t y abc h nfev") 
    return erg(np.array(tv), np.array(uv).T, abc, h, nfev)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from butcher import *
    
    def f(t, u):
        A = np.array([[-2, 5*t], [0, -1]])
        return A.dot(u)
    
    u0 = [1, 2]
    
    def uex(t):
        return np.array([10*t*np.exp(-t) - 10*np.exp(-t) + 11*np.exp(-2*t), 2*np.exp(-t)])
    
    T  = [0, 1]
    
    h  = 0.01
    
    m = "rk4"
    erg = rk(f, T, u0, butcher(m), h)
    
    plt.plot(erg.t, erg.y.T, "b", label=m)
    tt = np.linspace(*T, 11)
    plt.plot(tt, uex(tt).T, "r.", label = "exakt")
    plt.legend()