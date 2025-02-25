import matplotlib.pyplot as plt
import numpy as np
from llvmlite.binding import dump_refprune_stats
from scipy.integrate import solve_ivp
from scipy.ndimage import label

import mehrschritt
import rungekutta
import AlphaBeta

lower_time_bound = 0
upper_time_bound = 30
T = [lower_time_bound, upper_time_bound]

def grenzpop(t,u):
    grenzpopulation = 100
    g = grenzpopulation

    wachstumsrate = 0.001
    a = wachstumsrate

    return a * u * (g-u)

def doppelpendel(t,u, m1, m2, l1, l2, g, b):
    phi_1 = u[0]
    phi_1_dot= u[1]
    phi_2 = u[2]
    phi_2_dot= u[3]

    A = np.array([[(m1+m2) * l1**2,     m2 * l1 * l2 * np.cos(phi_1-phi_2)],
                  [m2 * l1 * l2 * np.cos(phi_1 - phi_2),    m2 * l2**2]])
    b = np.array([-m2 * l1 * l2 * np.sin(phi_1 - phi_2) * phi_2_dot**2 - (m1 + m2) * g * l1 * np.sin(phi_1) - b * phi_1_dot,
                  m2 * l1 * l2 * np.sin(phi_1 - phi_2) * phi_1_dot**2 - m2 * g * l2 * np.sin(phi_2) - b * phi_1_dot])

    lsg = np.linalg.solve(A,b)

    return np.array([
        phi_1_dot, lsg[0], phi_2_dot, lsg[1]
    ])

def pendel_wrapped(t, u):
    return doppelpendel(t, u, m1 = 10, m2 = 1, l1 = 1, l2 = 0.75, g = 9.81, b = 0)

#erg_dp_rk = rungekutta.rungekutta(pendel_wrapped, T,
#                                np.array([1,0,1.5,0], dtype = np.float64),
#                                0.005,
#                                np.array(
#                                    [
#                                        [0,0,0,0,0],
#                                        [1/2,1/2,0,0,0],
#                                        [1/2,0,1/2,0,0],
#                                        [1,0,0,1,0],
#                                        [0,1/6,2/6,2/6,1/6]
#                                    ],
#                                    dtype = np.float64
#                                ))
#plt.plot(erg_dp_rk.t, erg_dp_rk.y[0])
#plt.plot(erg_dp_rk.t, erg_dp_rk.y[2])

#erg_grenzpop_mehrschritt = mehrschritt.LMMexp(grenzpop, T, np.array([50]), 3, AlphaBeta.alphaA, AlphaBeta.betaAB, 0.1)

#plt.plot(erg_grenzpop_mehrschritt.t, erg_grenzpop_mehrschritt.y.T)
#plt.grid(True)
#plt.xlabel('Zeit')
#plt.ylabel('Population')

erg_dp_mehrschritt = mehrschritt.LMMexp_PEC(pendel_wrapped, T, np.array([1,0,0,0]), 3, AlphaBeta.alphaA, AlphaBeta.betaAB, AlphaBeta.alphaA, AlphaBeta.betaAM, 0.01, 3)

erg_dp_solve = solve_ivp(pendel_wrapped, T, np.array([1,0,0,0]))

plt.plot(erg_dp_mehrschritt.t, erg_dp_mehrschritt.y[0], label = 'Pendel 1 mit Mehrschritt')
plt.plot(erg_dp_mehrschritt.t, erg_dp_mehrschritt.y[2], label = 'Pendel 2 mit Mehrschritt')
plt.plot(erg_dp_solve.t, erg_dp_solve.y[0], label = 'Pendel 1 mit Solve')
plt.plot(erg_dp_solve.t, erg_dp_solve.y[2], label = 'Pendel 2 mit Solve')

plt.show()