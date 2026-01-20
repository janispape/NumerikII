import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.sparse import diags, identity, csc_matrix
from scipy.sparse.linalg import splu
from joblib import Parallel, delayed

# -----------------------------
# Problem setup: 1D heat equation
# u_t = nu * u_xx on (0,1), Dirichlet BC
# Method of lines with N interior points -> u' = nu A u
# -----------------------------
nu = 0.1
N = 600
T = 0.2                    # final time
slices = 64                # number of time slices (parareal parallelism)
K = 2                      # parareal iterations (often small for diffusion)

# Grid (interior points only)
h = 1.0 / (N + 1)
x = np.linspace(h, 1.0 - h, N)

# Sparse Laplacian (Dirichlet BC implicitly handled by using interior points only)
main = (-2.0) * np.ones(N)
off = (1.0) * np.ones(N - 1)
A = diags([off, main, off], offsets=[-1, 0, 1], shape=(N, N), format="csc") / (h * h)

# RHS: u' = nu A u
def f(t, u):
    return nu * (A @ u)

# Jacobian for Radau: J = nu * A
J = nu * A

# Initial condition (smooth)
u0 = np.cos(2 * np.pi * x)  # satisfies Dirichlet BC

# -----------------------------
# Time grid
# -----------------------------
t0, t1 = 0.0, T
Tgrid = np.linspace(t0, t1, slices + 1)
dt_slice = T / slices

# -----------------------------
# Fine and coarse propagators
# -----------------------------
# Fine: stiff accurate solve on [t0,t1] using Radau.
fine_atol = 1e-9
fine_rtol = 1e-7

def fine(t_start, t_end, u_start):
    sol = solve_ivp(
        f, (t_start, t_end), u_start,
        method="Radau",
        atol=fine_atol, rtol=fine_rtol,
        jac=J,
        vectorized=False
    )
    return sol.y[:, -1]

# Coarse: one big implicit Euler step per slice:
# (I - dt*nu*A) u_{n+1} = u_n
# Pre-factorize once since dt is constant and A is constant.
I = identity(N, format="csc")
M = (I - dt_slice * nu * A).tocsc()
M_lu = splu(M)  # sparse LU factorization

def coarse(t_start, t_end, u_start):
    # Here t_end - t_start should equal dt_slice; we assume uniform slices for speed.
    return M_lu.solve(u_start)

# -----------------------------
# Baseline: sequential fine solve over slices
# -----------------------------
U_fine_seq = np.zeros((slices + 1, N))
U_fine_seq[0] = u0

start_seq = time.time()
for n in range(slices):
    U_fine_seq[n + 1] = fine(Tgrid[n], Tgrid[n + 1], U_fine_seq[n])
end_seq = time.time()

print(f"Sequential fine (Radau) over {slices} slices: {end_seq - start_seq:.3f} s")

# -----------------------------
# Parareal
# Initialization: coarse sweep
# Then K iterations:
# U_{n+1}^{k+1} = G(U_n^{k+1}) + ( F(U_n^k) - G(U_n^k) )
# -----------------------------
def parareal(
    u0,
    K=4,
    n_jobs=-1,
    verbose=True
):
    U = np.zeros((slices + 1, N))
    U[0] = u0

    # Initial coarse sweep
    for n in range(slices):
        U[n + 1] = coarse(Tgrid[n], Tgrid[n + 1], U[n])

    # Parareal iterations
    for k in range(K):
        U_old = U.copy()

        # Parallel evaluation of F and G on the OLD iterate
        def FG(n):
            tA, tB = Tgrid[n], Tgrid[n + 1]
            uA = U_old[n]
            Fn = fine(tA, tB, uA)
            Gn = coarse(tA, tB, uA)
            return Fn, Gn

        FG_vals = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(FG)(n) for n in range(slices)
        )
        F_vals = [v[0] for v in FG_vals]
        G_old  = [v[1] for v in FG_vals]

        # Sequential correction sweep (depends on U_new[n])
        U_new = np.zeros_like(U_old)
        U_new[0] = u0
        for n in range(slices):
            G_new = coarse(Tgrid[n], Tgrid[n + 1], U_new[n])
            U_new[n + 1] = G_new + (F_vals[n] - G_old[n])

        U = U_new

        # Convergence metric (cheap, informative for contractive problems)
        rel = np.linalg.norm(U[-1] - U_old[-1]) / (np.linalg.norm(U[-1]) + 1e-16)
        if verbose:
            print(f"Parareal iter {k+1}/{K}: rel change at final time = {rel:.3e}")

    return U


start_para = time.time()
U_para = parareal(u0, K=K, n_jobs=-1, verbose=True)
end_para = time.time()

print(f"Parareal total wall time (init + {K} iters): {end_para - start_para:.3f} s")

# -----------------------------
# Visualization: compare endpoint and show a space profile
# -----------------------------
err_final = np.linalg.norm(U_para[-1] - U_fine_seq[-1]) / (np.linalg.norm(U_fine_seq[-1]) + 1e-16)
print(f"Relative difference at final time (parareal vs seq fine): {err_final:.3e}")

plt.figure()
plt.plot(x, u0, label="initial condition")
plt.plot(x, U_fine_seq[-1], label="Sequential fine (Radau)")
plt.plot(x, U_para[-1], "--", label=f"Parareal (K={K})")
plt.xlabel("x")
plt.ylabel("u(x,T)")
plt.legend()
plt.title("Heat equation solution profile at final time")
plt.show()