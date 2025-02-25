import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from Richtungsfeld import RichtungsFeld
import euler

lower_time_bound = 0
upper_time_bound = 100
T = [lower_time_bound, upper_time_bound]

def f_1(t,u):
    phi = u[0]
    phi_punkt = u[1]
    psi = u[2]
    psi_punkt = u[3]
    return np.array([phi_punkt, -2*np.cos(psi)*phi_punkt*psi_punkt/np.sin(psi), psi_punkt, np.sin(psi)*np.cos(psi)*phi_punkt**2-np.sin(psi)])

erg1 = solve_ivp(f_1, T, [np.pi/2,0.1,0.1,0.1], rtol=1e-6)

def f_2(t,u):
    sigma = 10
    b = 8/3
    r = 28
    return np.array([sigma*(-u[0]+u[1]), -u[0]*u[2] + r*u[0] - u[1], u[0]*u[1] - b * u[2]])

erg2 = solve_ivp(f_2, T, [-8,8,27], rtol = 1e-6)

def f_3(t,u):
    grenzpopulation = 100
    g = grenzpopulation

    wachstumsrate = 0.001
    a = wachstumsrate

    return a * u * (g-u)

erg3_solve = solve_ivp(f_3, T, np.array([200]), rtol = 1e-6)
erg3_euler = euler.euler(f_3, T, np.array([200]), 1)

## Plot Aufgabe 1:
## plt.plot(np.cos(erg1.y[0])*np.sin(erg1.y[2]), np.sin(erg1.y[0])*np.sin(erg1.y[2]))

## Plot Aufgabe 2:
##figure = plt.figure()
##axes = figure.add_subplot(111, projection='3d')
##axes.plot(erg2.y[0], erg2.y[1], erg2.y[2])
print(len(erg3_solve.y[0]))
print(len(erg3_euler.y[0]))
plt.plot(erg3_solve.t, erg3_solve.y[0])
plt.plot(erg3_euler.t, erg3_euler.y[0])

plt.show()