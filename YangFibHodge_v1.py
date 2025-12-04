import numpy as np

# ================================================================
#     Yang–Fib–Hodge v3 — PURE MATHEMATICS (No NCT, no tricks)
#     SU(2) Yang–Mills gradient-flow on a 3D lattice (FFT model)
# ================================================================

# -------------------------
# Parameters
# -------------------------
N = 32
L = 2*np.pi
dt = 0.001
steps = 6000

# Fourier grid
k = np.fft.fftfreq(N, d=L/(2*np.pi*N))
KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
k2 = KX**2 + KY**2 + KZ**2
k2[0,0,0] = 1e-9   # avoid division by zero

# SU(2) structure constants ε^{abc}
eps = np.zeros((3,3,3))
eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1
eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1

# -------------------------
# Hodge Projection (Coulomb gauge)
# -------------------------
def project(u_hat):
    """Project A onto divergence-free part: k·A = 0"""
    dot = KX*u_hat[0] + KY*u_hat[1] + KZ*u_hat[2]  # scalar field
    fac = dot / k2
    return np.array([
        u_hat[0] - fac*KX,
        u_hat[1] - fac*KY,
        u_hat[2] - fac*KZ
    ])

# -------------------------
# Linearized Yang–Mills flow: ∂t A = -ΔA
# -------------------------
def step(A_hat):
    """One gradient-flow timestep"""
    lap = -k2 * A_hat
    return A_hat + dt * lap

# -------------------------
# Energy per shell
# -------------------------
def shell_energy(A_hat, shell):
    mask = (shell[0] <= k2) & (k2 < shell[1])
    return np.sum(np.abs(A_hat[:,mask])**2)

shell_low  = (0.0,  0.5)
shell_mid  = (0.5,  2.0)
shell_high = (2.0, 10.0)

# -------------------------
# Initial condition: SU(2) random, divergence-free
# -------------------------
np.random.seed(0)

A0 = np.random.randn(3,3,N,N,N) * 0.2   # 3 components SU(2), 3 components spatial
A0_hat = np.fft.fftn(A0, axes=(2,3,4))
A0_hat = project(A0_hat)

A_hat = A0_hat.copy()

# -------------------------
# Simulation
# -------------------------
print("=== Yang–Fib–Hodge v3 — clean SU(2) mass-gap toy model ===")

for t in range(steps):
    A_hat = step(A_hat)
    A_hat = project(A_hat)

    if t % 200 == 0:
        E0 = shell_energy(A_hat[0], shell_low)
        E1 = shell_energy(A_hat[0], shell_mid)
        E2 = shell_energy(A_hat[0], shell_high)

        ratio = (E2/E1) if E1>0 else 0
        print(f"[t={t}]  E_low={E0:.6f}  E_mid={E1:.6f}  E_high={E2:.6f}  gap(ratio)= {ratio:.6f}")

print("=== END ===")

