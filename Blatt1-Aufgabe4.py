import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from Richtungsfeld import RichtungsFeld

lower_time_bound = 0
upper_time_bound = 20
T = [lower_time_bound,upper_time_bound]

def alpha_1(b):
    return 2-b

def alpha_2(b):
    return 2

def alpha_3(b):
    return (2-b)*(1.5*b*(b-1)+1)

beta = 1
gamma = 1
delta = 0.2


def f_aufgabe5(t,u):
    b = u[0]
    r = u[1]
    return np.array([alpha_3(b)*b-beta*b*r, -gamma*r + delta*b*r])

erg5 = solve_ivp(f_aufgabe5, T, np.array([8,5]))

def f_aufgabe6(t,u):
    phi = u[0]
    phi_punkt = u[1]
    return np.array([phi_punkt, -9.81/5*np.sin(phi)])

erg6 = solve_ivp(f_aufgabe6, T, np.array([np.pi/5,0]), rtol=1e-6)

def f_aufgabe6_b(t,u):
    phi = u[0]
    phi_punkt = u[1]
    return np.array([phi_punkt, -9.81/5*phi])

erg6_b = solve_ivp(f_aufgabe6_b, T, np.array([np.pi/10, 0]), rtol=1e-6)

def f_aufgabe7(t,u):
    phi = u[0]
    phi_punkt = u[1]
    return np.array([phi_punkt, -9.81/5*np.sin(phi) - 1*phi_punkt])

erg7 = solve_ivp(f_aufgabe7, T, np.array([np.pi/5, 0]), rtol=1e-6)

plt.plot(erg7.y[0], erg7.y[1])

plt.show()