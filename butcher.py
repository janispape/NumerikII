#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Butcher-Schemata

- bei Aufruf als main wird Konsistenzordnung überprüft
  und der Rand des A-Stabilitätsgebiets gezeichnet
  
- für main braucht man sympy und matplotlib, sonst
  reicht numpy

@author: mre
"""

import numpy as np
from numpy import sqrt

## ERK
euler = "np.array([\
    [0, 0], \
    [1, 1]  \
    ])"
                
collatz = "np.array([\
    [ 0 ,   0 , 0],\
    [1/2,  1/2, 0],\
    [ 2 ,   0 , 1] \
    ])" 
    
heun = "np.array([\
    [ 0 ,   0 ,  0 ],\
    [ 1,    1 ,  0 ],\
    [ 2 ,  1/2, 1/2] \
    ])" 

rk4= "np.array([\
    [ 0 ,  0 ,  0 ,  0 ,   0 ],\
    [1/2, 1/2,  0 ,  0 ,   0 ],\
    [1/2,  0 , 1/2,  0 ,   0 ],\
    [ 1 ,  0 ,  0 ,  1 ,   0 ],\
    [ 4 , 1/6, 1/3, 1/3,  1/6] \
    ])"
    
k38= "np.array([\
    [ 0 ,   0 ,  0 ,  0 ,  0 ],\
    [1/3,  1/3,  0 ,  0 ,  0 ],\
    [2/3, -1/3,  1 ,  0 ,  0 ],\
    [ 1 ,   1 , -1 ,  1 ,  0 ],\
    [ 4 ,  1/8, 3/8, 3/8, 1/8] \
    ])"
    
## ERK embedded
rkf23 = "np.array([\
    [ 0 ,  0 ,  0 ,  0 ],\
    [ 1 ,  1 ,  0 ,  0 ],\
    [1/2, 1/4, 1/4,  0 ],\
    [ 2 , 1/2, 1/2,  0 ],\
    [ 3 , 1/6, 1/6, 2/3] \
    ])"


rkf2b = "np.array([\
    [  0  ,     0    ,    0   ,    0    ,    0 ],\
    [ 1/4 ,    1/4   ,    0   ,    0    ,    0 ],\
    [27/40, -189/800 , 729/800,    0    ,    0 ],\
    [  1  ,  214/891 ,   1/33 ,  50/891 ,    0 ],\
    [  2  ,  214/891 ,   1/33 , 650/891 ,    0 ],\
    [  3  ,  533/2106,    0   , 800/1053, -1/78] \
    ])"
    
rk43 = "np.array([\
    [ 0 ,  0 ,  0 ,  0 ,  0 ,  0 ],\
    [1/2, 1/2,  0 ,  0 ,  0 ,  0 ],\
    [1/2,  0 , 1/2,  0 ,  0 ,  0 ],\
    [ 1 ,  0 ,  0 ,  1 ,  0 ,  0 ],\
    [ 1 , 1/6, 1/3, 1/3, 1/6,  0 ],\
    [ 4 , 1/6, 1/3, 1/3, 1/6,  0 ],\
    [ 3 , 1/6, 1/3, 1/3,  0 , 1/6] \
    ])"

dp54 = "np.array([\
    [  0 ,      0     ,       0    ,      0     ,     0   ,       0      ,    0    ,  0  ],\
    [ 1/5,     1/5    ,       0    ,      0     ,     0   ,       0      ,    0    ,  0  ],\
    [3/10,     3/40   ,      9/40  ,      0     ,     0   ,       0      ,    0    ,  0  ],\
    [ 4/5,    44/45   ,    -56/15  ,    32/9    ,     0   ,       0      ,    0    ,  0  ],\
    [ 8/9, 19372/6561 , -25360/2187, 64448/6561 , -212/729,       0      ,    0    ,  0  ],\
    [  1 ,  9017/3168 ,   -355/33  , 46732/5247 ,   49/176,  -5103/18656 ,    0    ,  0  ],\
    [  1 ,    35/384  ,       0    ,   500/1113 ,  125/192,  -2187/6784  ,  11/84  ,  0  ],\
    [  5 ,    35/384  ,       0    ,   500/1113 ,  125/192,  -2187/6784  ,  11/84  ,  0  ],\
    [  4 ,  5179/57600,       0    ,  7571/16695,  393/640, -92097/339200, 187/2100, 1/40] \
    ])"

## IRK
ieuler = "np.array([\
    [1, 1],\
    [1, 1] \
    ])"

imipu = "np.array([\
    [1/2, 1/2],\
    [ 2 ,  1 ] \
    ])"

itrapez = "np.array([\
    [0,  0 ,  0 ],\
    [1, 1/2, 1/2],\
    [2, 1/2, 1/2] \
    ])"

gauss4 = "np.array([\
    [1/2 - sqrt(3)/6, 1/4            , 1/4 - sqrt(3)/6],\
    [1/2 + sqrt(3)/6, 1/4 + sqrt(3)/6, 1/4            ],\
    [ 4             , 1/2            , 1/2            ] \
    ])"

sdirk3L = "np.array([\
    [1/2,  1/2,   0 ,  0 ,  0 ],\
    [2/3,  1/6,  1/2,  0 ,  0 ],\
    [1/2, -1/2,  1/2, 1/2,  0 ],\
    [ 1 ,  3/2, -3/2, 1/2, 1/2],\
    [ 3 ,  3/2, -3/2, 1/2, 1/2] \
    ])"

radau2A3 = "np.array([\
    [1/3, 5/12, -1/12],\
    [ 1 , 3/4 ,  1/4 ],\
    [ 3 , 3/4 ,  1/4 ]\
    ])"
    
## IRK embedded
gauss4e = "np.array([\
    [1/2 - sqrt(3)/6, 1/4            , 1/4 - sqrt(3)/6],\
    [1/2 + sqrt(3)/6, 1/4 + sqrt(3)/6, 1/4            ],\
    [ 4             , 1/2            , 1/2            ],\
    [ 1             , 1/2 + sqrt(3)/2, 1/2 - sqrt(3)/2] \
    ])"

gauss6e = "np.array([\
    [1/2 - sqrt(15)/10,  5/36            , 2/9-sqrt(15)/15,  5/36-sqrt(15)/30],\
    [1/2              ,  5/36+sqrt(15)/24, 2/9            ,  5/36-sqrt(15)/24],\
    [1/2 + sqrt(15)/10,  5/36+sqrt(15)/30, 2/9+sqrt(15)/15,  5/36            ],\
    [ 6               ,  5/18            , 4/9            ,  5/18            ],\
    [ 2               , -5/6             , 8/3            , -5/6             ] \
    ])"

sdirk21 = "np.array([\
    [1,  1 ,  0 ],\
    [0, -1 ,  1 ],\
    [2, 1/2, 1/2],\
    [1,  1 ,  0 ] \
    ])"

# H-W Band 2, S.100
sdirk43 = "np.array([\
    [ 1/4 ,   1/4   ,     0    ,   0   ,    0,    0 ],\
    [ 3/4 ,   1/2   ,    1/4   ,   0   ,    0,    0 ],\
    [11/20,  17/50  ,   -1/25  ,  1/4  ,    0 ,   0 ],\
    [ 1/2 , 371/1360, -137/2720, 15/544,   1/4 ,  0 ],\
    [  1  ,  25/24  ,  -49/48  , 125/16, -85/12, 1/4],\
    [  4  ,  25/24  ,  -49/48  , 125/16, -85/12, 1/4],\
    [  3  ,  59/48  ,  -17/96  , 225/32, -85/12,  0 ] \
    ])"

lobatto3c = "np.array([\
    [ 0 , 1/6, -1/3 ,  1/6 ],\
    [1/2, 1/6,  5/12, -1/12],\
    [ 1 , 1/6,  2/3 ,  1/6 ],\
    [ 4 , 1/6,  2/3 ,  1/6 ],\
    [ 2 ,-1/2,   2  , -1/2 ] \
    \])"


erks  = ["euler", "collatz", "rk4", "k38"]
eerks = ["rkf23", "rkf2b", "rk43", "dp54"]
irks  = ["ieuler", "itrapez", "imipu", "gauss4", "sdirk3L", "radau2A3"]
eirks = ["gauss4e", "gauss6e", "sdirk21", "sdirk43", "lobatto3c"]

rks = erks + eerks + irks + eirks
#rks = ["dp54"]

def butcher(m = None):
    if m == None:
        print("**** The following methods are available ****")
        for a,b in zip([erks, eerks, irks, eirks], ["Expl", "EmbExpl", "Impl", "EmbImpl"]):
            print(b)
            print(a)
            print() 
        return 
    return eval(eval(m))
            

if __name__ == "__main__":
    "compute leading order term of local error"
    import sympy as sy
    import matplotlib.pyplot as plt
    
    def str2rat(st):
        s2 = "".join(st.split("    ")[1:-1])
        ra = list(map(sy.S, s2.replace("[", "").replace("]", "").split(",")))
        l = len(ra)
        s = int(np.sqrt(l))
        return sy.Matrix(ra).reshape(l//s,s)
    
    
    def plotA(r, mstr):
        # Rand des A-Stabilitätsgebiets
        x,y = sy.symbols("x,y", real=True)
        rz = r.subs("h", x + sy.I * y).expand()
        #print(rz)
        p = sy.plot_implicit( ((rz*rz.conjugate()) - 1).simplify(), (x, -6, 6), (y, -5, 5), adaptive = False, points = 2000 , title = mstr, aspect_ratio=[1,1])

        
        
    def plotAmpl(r, mstr):
        # Rand des A-Stabilitätsgebiets
        plt.figure(mstr)
        
        x,y = sy.symbols("x,y", real=True)
        rz = r.subs("h", x + sy.I * y).expand()
        us = ((rz*rz.conjugate()) - 1).simplify()
        
        u = sy.lambdify((x,y), us, "numpy")
        
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-3, 3, 100)
        
        yy, xx = np.meshgrid(x, y)
        
        zz = u(xx, yy)
        
        plt.contour(xx, yy, zz, levels = [0.0])
        plt.axis("equal")

        
    
    def ko(sabc):
        sabc = eval(mstr)
        abc = str2rat((sabc))
        m,n = abc.shape
        s = n - 1
        
        A = abc[:s, 1:]
        c = abc[:s, 0]
        b = abc[ s, 1:]
        
        I = sy.eye(s)
        e = sy.ones(s,1)
        
        h = sy.symbols("h")
        
        k = (I - h * A).inv() * e
        
        r = (1 + (h * k.T * b.T)[0]).simplify()
        
        p = 2*s + 2
        t = [(r.series(h, 0, p) - sy.exp(h).series(h, 0, p)).leadterm("h")[0]]
        #print(k)
                
        plotA(r, mstr)

        # eingebettete Verfahren
        if m > n:
            bd = abc[s+1, 1:]
            rd = (1 + (h * k.T * bd.T)[0]).simplify()
            td = (rd.series(h, 0, p) - sy.exp(h).series(h, 0, p)).leadterm("h")[0]
            t.append(td)
            plotA(rd, mstr + "-hat")
            
        return t
    
    
    #for mstr in rks: #  ["euler", "collatz"]: #  
    for mstr in erks+eerks: #  ["euler", "collatz"]: #  
        print()
        print("********** {} **********".format(mstr))
        taylor = ko(mstr)
        for t in taylor:
            print(t)
