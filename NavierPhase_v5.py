import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ================================
# NAVIER–PHASE v5 (caótico real)
# ================================
# 3D Navier–Stokes reducido (sin presión explícita)
# Caos inducido por alta energía inicial + baja viscosidad
# Includes:
# - CFL dt
# - BKM criterion
# - sigma(t) activation
# - energy tracking
# - 3D vorticity map
# ================================

# Parámetros
N = 32                 # resolución 3D
A = 2.5                # amplitud inicial (alto → caos)
nu = 0.005             # viscosidad (baja → régimen no lineal)
steps = 8000           # iteraciones
omega_crit = 8.0       # umbral de transición de fase
CFL = 0.2              # condición CFL

# Dominio 3D
L = 2*np.pi
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
z = np.linspace(0, L, N, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
dx = L / N

# ========================
# Campos iniciales
# ========================
u = np.zeros((N, N, N, 3))

# Flujo tipo Taylor–Green turbulento-inductor
u[...,0] = A * np.sin(X) * np.cos(Y) * np.cos(Z)
u[...,1] = A *-np.cos(X) * np.sin(Y) * np.cos(Z)
u[...,2] = 0.3 * A * np.cos(X+Y+Z)

# Historial
hist_omega = []
hist_energy = []
hist_BKM = []
hist_sigma = []

BKM_int = 0.0

# ========================
# Helpers numéricos
# ========================
def grad(f):
    fx = (np.roll(f,-1,0) - np.roll(f,1,0)) / (2*dx)
    fy = (np.roll(f,-1,1) - np.roll(f,1,1)) / (2*dx)
    fz = (np.roll(f,-1,2) - np.roll(f,1,2)) / (2*dx)
    return fx, fy, fz

def lap(f):
    return (np.roll(f,1,0) + np.roll(f,-1,0) +
            np.roll(f,1,1) + np.roll(f,-1,1) +
            np.roll(f,1,2) + np.roll(f,-1,2) -
            6*f) / dx**2

def vorticity(u):
    ux, uy, uz = u[...,0], u[...,1], u[...,2]
    dux = grad(ux)
    duy = grad(uy)
    duz = grad(uz)
    wx = duz[1] - duy[2]
    wy = dux[2] - duz[0]
    wz = duy[0] - dux[1]
    return wx, wy, wz


# ============================
# Bucle principal
# ============================
for t in range(steps):

    # === calcular vorticidad ===
    wx, wy, wz = vorticity(u)
    omega_mag = np.sqrt(wx**2 + wy**2 + wz**2)
    max_omega = np.max(omega_mag)

    # === energía ===
    E = np.mean(np.sum(u**2, axis=-1))

    # === BKM ===
    BKM_int += max_omega * 1e-3

    # === sigma(t) ===
    sigma = 1.0 if max_omega > omega_crit else 0.0

    # registrar
    hist_omega.append(max_omega)
    hist_energy.append(E)
    hist_BKM.append(BKM_int)
    hist_sigma.append(sigma)

    # log cada 500 steps
    if t % 500 == 0:
        print(f"[t={t}] max|ω|={max_omega:.3f}  BKM={BKM_int:.3f}  σ={sigma}  E={E:.4f}")

    # ==========================================
    # dt adaptativo via CFL
    # ==========================================
    umax = np.max(np.abs(u))
    dt = CFL * dx / (umax + 1e-8)

    # ==========================================
    # Paso de Navier-Stokes reducido
    # ==========================================
    ux, uy, uz = u[...,0], u[...,1], u[...,2]

    dux = grad(ux)
    duy = grad(uy)
    duz = grad(uz)

    conv = np.stack([
        ux*dux[0] + uy*dux[1] + uz*dux[2],
        ux*duy[0] + uy*duy[1] + uz*duy[2],
        ux*duz[0] + uy*duz[1] + uz*duz[2]
    ], axis=-1)

    visc = lap(u)

    # avance temporal
    u = u + dt * (-conv + nu * visc)

    # Estabilizador leve para evitar overflow numérico
    u = 0.995*u + 0.005*lap(u)

# ====================
# GRÁFICOS
# ====================
fig, ax = plt.subplots(1,3,figsize=(15,5))

ax[0].plot(hist_omega)
ax[0].axhline(omega_crit, ls='--', c='k')
ax[0].set_title("Vorticidad máxima")
ax[0].set_xlabel("iteración")

ax[1].plot(hist_energy)
ax[1].set_title("Energía E(t)")

ax[2].plot(hist_BKM)
ax[2].set_title("Integral BKM acumulada")

plt.tight_layout()
plt.show()

# ====================
# MAPA 3D DE VORTICIDAD
# ====================
wplot = omega_mag / np.max(omega_mag)

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection="3d")
idx = np.where(wplot > 0.6)

ax.scatter(X[idx], Y[idx], Z[idx], s=4, c=wplot[idx], cmap="inferno")
ax.set_title("Isosuperficie caótica de vorticidad")
plt.show()

print("\n=== FIN NAVIER–PHASE v5 ===")
print(f"Integral BKM final = {BKM_int}")
print(f"¿Hubo fase σ(t)? {'Sí' if np.max(hist_sigma)>0 else 'No'}")
