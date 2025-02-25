import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from Richtungsfeld import RichtungsFeld

global a, g
a = 0.005
g = 150

lower_time_bound = 0
upper_time_bound = 5
lower_space_bound = 0
startwert = 300

def f(t,u):
    return a * (g-u) * u

erg = solve_ivp(f, [lower_time_bound, upper_time_bound], np.array([startwert]))

upper_space_bound = max(erg.y.T) * 1.5

RichtungsFeld(f, [lower_time_bound, upper_time_bound], [lower_space_bound, upper_space_bound], [])
plt.plot(erg.t, erg.y.T, label = "$u$")
plt.legend(loc = 0)
plt.show()