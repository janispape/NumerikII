# -*- coding: utf-8 -*-
"""
Implementierung einer expliziten LMM ohne Schrittweitensteuerung,
selbststartend
"""

import numpy as np

from collections import namedtuple


def LMMexp(f, T, u0, k, alpha, beta, h, args=()):
    
    """ 
    erg = LMMexp(f, T, u0, k, alpha, beta, h, args=()):
    
    LMM mit fester Schrittweite
    IN:
        f(t, u) : muss array auf array abbilden
        
        T      : [t0,te]
        u0     : Startwert (Numpy-array)
        h      : Zeitschrittweite
        alpha, beta : Koeffizientefunktionen
        
    OUT:
        erg.y    : Lösungskomponenten (Zeilen des arrays)
        erg.t    : Zeitschritte (array)
        erg.k    : Ordnung
        erg.h    : Zeitschrittweite
        erg.nfev : Anzahl f-Auswertungen
        
    t0 und te sind sicher im Output
    """
    
    # Start- und Endzeit ermitteln
    t0 = T[0]
    te = T[1]
            
    # Anfangswert ggf. auf Zeilenvektor transformieren
    u0 = np.array(u0).flatten()
    
    # Anfangszeit und -wert in Lösungsliste eintragen
    tv = [t0]
  
    # Umwandlung von 1d- auf 2d-array nötig, um .dot einsetzen zu können
    uv = u0.reshape(1,u0.shape[0])
    
    nfev = 0
    
    # Gedächtnis-Vektor
    fv = f(t0, u0, *args)
    nfev += 1
    fv = np.array(fv).flatten()
    fv = fv.reshape(1, fv.shape[0])
    
    
    i = 1

    t = t0
    kloc = 1
    al = alpha(kloc)
    be = beta(kloc)
    
    # Schleife über Zeitschritte
    while(t<te):

        # ermittle dt und verkürze es ggf. im letzten Schritt
        t_neu = t + h
        #ttol = 1e-8 * h # Toleranz für Schlussschritt 

        # if t_neu > te-ttol:
        #     t_neu = te
        #     h  = t_neu - t

        # eigentlicher Verfahrensschritt
        al_u = al[0:kloc].dot(uv[i-kloc:i])

        # Berechnung Linearkombination f-Werte ohne f_i+1, da explizit
        be_f = be[0:kloc,].dot(fv[i-kloc:i])
    
        # Auflösen    
        u_neu = (h*be_f - al_u)/al[kloc]
    
        # f-Auswertung in u_{i+1}
        f_neu = f(t_neu, u_neu, *args)
        nfev += 1

        # Ordnung hochfahren, wenn noch in der Startphase
        if(kloc < k):
            kloc = kloc + 1
            al = alpha(kloc)
            be = beta(kloc)
          
        # Schritt vorwärts
        i  = i+1
        
        # neuer Zeitpunkt und neue Näherung übernehmen    
        t  = t_neu
        u  = u_neu
            
        # Ausgabe analog zu solve_ivp
        tv = np.append(tv, t)
        uv = np.vstack((uv,  u))
        fv = np.vstack((fv, f_neu))
        
    erg   = namedtuple("erg", "t y k h nfev") 
    return erg(np.array(tv), np.array(uv).T, k, h, nfev)

    
if __name__ == "__main__":
    from AlphaBeta import *
    
    import matplotlib.pyplot as plt
    
    
    def f(t, u):
        A = np.array([[-2, 5*t], [0, -1]])
        return A.dot(u)
    
    u0 = [1, 2]
    
    def uex(t):
        return np.array([10*t*np.exp(-t) - 10*np.exp(-t) + 11*np.exp(-2*t), 2*np.exp(-t)])
    
    T  = [0, 1]
    
    k = 5
    h  = 0.01
    
    erg = LMMexp(f, T, u0, k, alphaA, betaAB, h)
    
    plt.plot(erg.t, erg.y.T, "b", label = f"LMMexp: nfev={erg.nfev}")
    tt = np.linspace(*T, 11)
    plt.plot(tt, uex(tt).T, "r.", label = "exakt")
    plt.legend()