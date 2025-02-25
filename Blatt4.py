import matplotlib.pyplot as plt
import numpy as np
from llvmlite.binding import dump_refprune_stats
from scipy.integrate import solve_ivp
from Richtungsfeld import RichtungsFeld
import collatz
import rungekutta

lower_time_bound = 0
upper_time_bound = 100
T = [lower_time_bound, upper_time_bound]

def f_2(t,u, gamma, m_r, m_p, m_s):
    r_1, r_2, r_dot_1, r_dot_2 = u[0:4]
    p_1, p_2, p_dot_1, p_dot_2 = u[4:8]
    s_1, s_2, s_dot_1, s_dot_2 = u[8:12]

    eps = 1e-10

    r_s_vec = np.array([r_1 - s_1, r_2 - s_2])
    r_p_vec = np.array([r_1 - p_1, r_2 - p_2])
    p_s_vec = np.array([p_1 - s_1, p_2 - s_2])

    dist_r_s = np.sqrt(np.sum(r_s_vec ** 2) + eps)
    dist_r_p = np.sqrt(np.sum(r_p_vec ** 2) + eps)
    dist_p_s = np.sqrt(np.sum(p_s_vec ** 2) + eps)

    # Pre-calculate inverse cube distances for better stability
    inv_dist_r_s_cube = 1.0 / (dist_r_s ** 3)
    inv_dist_r_p_cube = 1.0 / (dist_r_p ** 3)
    inv_dist_p_s_cube = 1.0 / (dist_p_s ** 3)

    return np.array([
        r_dot_1,
        r_dot_2,
        -gamma * (m_s * (r_1 - s_1) * inv_dist_r_s_cube + m_p * (r_1 - p_1) * inv_dist_r_p_cube),
        -gamma * (m_s * (r_2 - s_2) * inv_dist_r_s_cube + m_p * (r_2 - p_2) * inv_dist_r_p_cube),
        p_dot_1,
        p_dot_2,
        -gamma * (m_s * (p_1 - s_1) * inv_dist_p_s_cube + m_r * (p_1 - r_1) * inv_dist_r_p_cube),
        -gamma * (m_s * (p_2 - s_2) * inv_dist_p_s_cube + m_r * (p_2 - r_2) * inv_dist_r_p_cube),
        s_dot_1,
        s_dot_2,
        -gamma * (m_r * (s_1 - r_1) * inv_dist_r_s_cube + m_p * (s_1 - p_1) * inv_dist_p_s_cube),
        -gamma * (m_r * (s_2 - r_2) * inv_dist_r_s_cube + m_p * (s_2 - p_2) * inv_dist_p_s_cube)
    ])

def f_2_wrapped(t, u):
    return f_2(t, u, gamma=1, m_r=1, m_s=1e9, m_p=1e8)

#erg1_solve = solve_ivp(f_2_wrapped, T, np.array([1100,0,-100,0,1000,0,0,1000,-100,0,0,-100], dtype = np.float64), rtol = 1e-6)
#plt.plot(erg1_solve.y[0], erg1_solve.y[1], 'b-', label='Rocket (r)', alpha=0.7)
#plt.plot(erg1_solve.y[4], erg1_solve.y[5], 'r-', label='Planet (p)', alpha=0.7)
#plt.plot(erg1_solve.y[8], erg1_solve.y[9], 'r-', label='Planet (p)', alpha=0.7)

#erg1_rk = rungekutta.rungekutta(f_2_wrapped, T,
#                                np.array([1500,0,0,100,1000,0,0,1000,-100,0,0,-100], dtype = np.float64),
#                                0.001,
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
#plt.plot(erg1_rk.y[0], erg1_rk.y[1], 'b-', label='Rocket (r)', alpha=0.7)
#plt.plot(erg1_rk.y[4], erg1_rk.y[5], 'r-', label='Planet (p)', alpha=0.7)
#plt.plot(erg1_rk.y[8], erg1_rk.y[9], 'r-', label='Planet (p)', alpha=0.7)

erg3_rk_34 = rungekutta.rungekutta_SWS(
    f_2_wrapped,
    T,
    np.array([1100,0,-100,0,1000,0,0,1000,-100,0,0,-100], dtype = np.float64),
    np.array([
        [0, 0, 0, 0, 0, 0],
        [1 / 2, 1 / 2, 0, 0, 0, 0],
        [1 / 2, 0, 1 / 2, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 1 / 6, 1 / 3, 1 / 3, 1 / 6, 0],
        [0, 1 / 6, 1 / 3, 1 / 3, 1 / 6, 0],
        [0, 1 / 6, 1 / 3, 1 / 3, 0, 1 / 6]
    ]),
    p = 4,
    p_hut = 3,
    lok_tol= 0.1
)

plt.plot(erg3_rk_34.y[0], erg3_rk_34.y[1])
plt.plot(erg3_rk_34.y[4], erg3_rk_34.y[5])
plt.plot(erg3_rk_34.y[8], erg3_rk_34.y[9])

print(erg3_rk_34.nfev)

plt.show()