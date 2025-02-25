#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:49:00 2022

@author: mre
"""

import numpy as np
import sympy as sy

tsy = sy.symbols("t", real=True)
def L(i,n):
    return sy.Lambda(tsy, sy.prod([(tsy-j)/(i-j) for j in range(n) if j!=i]))

# Adams
def alphaA(k, sym=False):
    w = (k-1) * [0] + [-1, 1]
    return w if sym else np.array(w, dtype=float)
    
def betaAB(k, sym=False):
    w = [sy.integrate(L(i,k)(tsy),(tsy, k-1, k)) for i in range(k)] + [0]
    return w if sym else np.array(w, dtype=float)

def betaAM(k, sym=False):
    w = [sy.integrate(L(i,k+1)(tsy),(tsy, k-1, k)) for i in range(k+1)]
    return w if sym else np.array(w, dtype=float)

# BDF mit Interpolationspraediktor
def alphaBDF(k, sym=False):
    w = [L(i,k+1)(tsy).diff(tsy).subs(tsy, k) for i in range(k+1)]
    return w if sym else np.array(w, dtype=float)

def betaBDF(k, sym=False):
    w = k * [0] + [1]
    return w if sym else np.array(w, dtype=float)

def alphaP(k, sym=False):
    w = [-L(i,k)(k) for i in range(0,k)] + [1]
    return w if sym else np.array(w, dtype=float)

def betaP(k, sym=False):
    w = (k+1) * [0]
    return w if sym else np.array(w, dtype=float)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    kmax = 3
    
    for sym in [False, True]:
        for al, be in zip([alphaA, alphaA, alphaBDF, alphaP], [betaAB, betaAM, betaBDF, betaP]):
            for k in range(1,kmax+1):
                print("k = {}  {:10s}".format(k, al.__name__), al(k, sym))
                print("k = {}  {:10s}".format(k, be.__name__), be(k, sym))
                print()
        print("**********************")
        
        
    def ze(al, be, n=200):
        rho   = np.poly1d(al[::-1])
        sigma = np.poly1d(be[::-1])
        
        phi = np.linspace(0, 2*np.pi, n)
        e   = np.exp(1j * phi)
        
        z = rho(e) / sigma(e)
        
        return z

# aus NDGL Skript Bilder
    fsize = (6,6)
    yminmax = (-5,5)
    xminmax = (-7,3)
    
    for al, be in zip([alphaA, alphaA, alphaBDF], [betaAB, betaAM, betaBDF]):
        for k in range(1, kmax+1):
            plt.figure(f"{al.__name__} - {be.__name__} - {k}", figsize = fsize)
            
            ax = plt.subplot(111, aspect='equal')
            
            # ax.set_facecolor((1.0, 0.9, 0.9))
            
            z = ze(al(k), be(k)) 
            # poly = plt.Polygon(np.c_[z.real, z.imag], color = (0.9,1.0,0.9))
            
            # ax.add_patch(poly)
            
            plt.plot(z.real, z.imag)
            plt.xlim(*xminmax)
            plt.ylim(*yminmax)
            
            ax.set_aspect(1)
        
            plt.axvline(0, color = 'k')
            plt.axhline(0, color = 'k')
            
            