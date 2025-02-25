import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from Richtungsfeld import RichtungsFeld
import collatz
import rungekutta

lower_time_bound = 0
upper_time_bound = 100
T = [lower_time_bound, upper_time_bound]

def f_2(t,u):
    phi = u[0]
    phi_punkt = u[1]

    b = 0.5
    k = 1
    return np.array([phi_punkt, -b*phi_punkt - k * np.sin(phi)])

def f_4(t,u):
    grenzpopulation = 100
    g = grenzpopulation

    wachstumsrate = 0.001
    a = wachstumsrate

    return a * u * (g-u)

#erg2_solve = solve_ivp(f_2, T, np.array([np.pi/8,0]), rtol = 1e-6)
#erg2_collatz = collatz.collatzheun(f_2, T, np.array([np.pi/8,0]), 0.1, verfahren = "collatz")
#erg2_heun = collatz.collatzheun(f_2, T, np.array([np.pi/8,0]), 0.1, verfahren = "heun")

#plt.plot(erg2_solve.t, erg2_solve.y[0])
#plt.plot(erg2_collatz.t, erg2_collatz.y[0])
#plt.plot(erg2_heun.t, erg2_heun.y[0])
#plt.plot(erg2_heun.t, erg2_collatz.y[0]-erg2_heun.y[0])

#erg4_rk_collatz = rungekutta.rungekutta(f_4, T, [40.0], 5, np.array([[0,0,0],[1/2,1/2,0],[0,0,1]]))
#erg4_rk_classic = rungekutta.rungekutta(f_4, T, [40.0], 5, np.array([[0,0,0,0,0],[1/2,1/2,0,0,0],[1/2,0,1/2,0,0],[1,0,0,1,0],[0,1/6,1/3,1/3,1/6]]))

#plt.plot(erg4_rk_collatz.t, erg4_rk_collatz.y[0])
#plt.plot(erg4_rk_classic.t, erg4_rk_classic.y[0])

erg5_rk_classic = rungekutta.rungekutta(f_2, T, [np.pi/4, 0], 0.1, np.array([[0,0,0],[1/2,1/2,0],[0,0,1]], dtype=np.float64))
plt.plot(erg5_rk_classic.y[0], erg5_rk_classic.y[1])

plt.show()