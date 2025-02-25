# -*- coding: utf-8 -*-
"""
@author: mre
"""

import matplotlib.pyplot as plt
import numpy as np

def RichtungsFeld(f, T, X, args):
    # Eingaben
    #f: f(t,x) ist die rechte Seite der Differentialgleichung
    # T = [t0, te] : Zeitintervall
    # X = [x0, x1] : Wertebereich
    
    m = 20
    n = 20
    
    t0 = T[0]
    te = T[1]
    
    x0 = X[0]
    x1 = X[1]
        
    tvec = np.linspace(t0, te, m)
    xvec = np.linspace(x0, x1, n)
    
    tt, xx = np.meshgrid(tvec, xvec)

    T_component = np.ones([m,n])

    vecf = np.vectorize(f)

    X_component  = vecf(tt, xx, *args)
    
    plt.quiver(tt, xx, T_component, X_component, color='0.5', angles='xy')
