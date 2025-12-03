import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================
#   PARÁMETROS DEL MODELO
# ============================
N = 32                 # malla 32x32x32
dx = 1.0 / N
nu = 1e-4              # viscosidad muy baja
dt = 0.0005            # paso de tiempo
steps = 6000           # iteraciones

gamma = 0.10           # fase disipativa (σ)
forcing_rate = 50      # cada 50 pasos aplicamos causal injection
forcing_strength = 1.5 # amplitud del pulso en zonas críticas
percent = 0.03         # top 3% de vorticidad recibe forzamiento

# ============================
#   CAMPOS INICIALES
# ============================
rng = np.random.default_rng(1)
u = 0.5 * rng.normal(size=(N, N, N, 3))  # campo inicial turbulento

# ============================
#   DERIVADAS FINITAS
# ============================
def grad(f):
    fx = (np.roll(f,-1,axis=0) - np.roll(f,1,axis=0)) / (2*dx)
    fy = (np.roll(f,-1,axis=1) - np.roll(f,1,axis=1)) / (2*dx)
    fz = (np.roll(f,-1,axis=2) - np.roll(f,1,axis=2)) / (2*dx)
    return fx, fy, fz

def laplacian(f):
    return (
        np.roll(f,-1,axis=0) + np.roll(f,1,axis=0)
        + np.roll(f,-1,axis=1) + np.roll(f,1,axis=1)
        + np.roll(f,-1,axis=2) + np.roll(f,1,axis=2)
        - 6*f
    ) / dx**2

# ============================
#   VORTICIDAD  ω = ∇ × u
# ============================
def compute_vorticity(u):
    ux, uy, uz = u[...,0], u[...,1], u[...,2]
    dux = grad(ux)
    duy = grad(uy)
    duz = grad(uz)
    wx = duy[2] - duz[1]
    wy = duz[0] - dux[2]
    wz = dux[1] - duy[0]
    return np.stack([wx, wy, wz], axis=-1)

# ============================
#   PLOT CONFIG
# ============================
plt.ion()
fig, ax = plt.subplots(1,3,figsize=(15,4))

ax_w, ax_E, ax_BKM = ax

ax_w.set_title("Vorticidad máxima")
ax_E.set_title("Energía E(t)")
ax_BKM.set_title("Integral BKM acumulada")

ax_w.set_ylim(0,8)
ax_E.set_ylim(0,3)
ax_BKM.set_ylim(0,1.5)

line_w, = ax_w.plot([],[])
line_E, = ax_E.plot([],[])
line_BKM, = ax_BKM.plot([] ,[])

ts, wmaxs, Es, BKMs = [], [], [], []

# ============================
#   INYECCIÓN ADAPTATIVA
# ============================
def forcing_adaptativo(u, omega, strength):
    ω_norm = np.linalg.norm(omega, axis=-1).ravel()
    k = int(len(ω_norm) * percent)

    idx = np.argpartition(ω_norm, -k)[-k:]    # top 3%
    x, y, z = np.unravel_index(idx, (N, N, N))

    # inyectamos energía en la dirección de la vorticidad local
    for i in range(k):
        u[x[i], y[i], z[i]] += strength * omega[x[i], y[i], z[i]]

    return u

# ============================
#   LOOP PRINCIPAL
# ============================
BKM = 0.0

for t in range(steps):
    omega = compute_vorticity(u)
    maxw = np.max(np.linalg.norm(omega, axis=-1))
    E = np.mean(np.sum(u**2, axis=-1))

    BKM += maxw * dt

    # ------------------
    # Forzamiento adaptativo
    # ------------------
    if t % forcing_rate == 0 and t > 0:
        u = forcing_adaptativo(u, omega, forcing_strength)

    # ------------------
    # Evolución NS simplificada
    # ------------------
    du = np.zeros_like(u)
    for j in range(3):
        dgrad = grad(u[...,j])
        du[...,j] -= (u[...,0]*dgrad[0] + u[...,1]*dgrad[1] + u[...,2]*dgrad[2])
        du[...,j] += nu * laplacian(u[...,j])

    u += dt*du

    # ------------------
    # Guardar datos
    # ------------------
    ts.append(t)
    wmaxs.append(maxw)
    Es.append(E)
    BKMs.append(BKM)

    # ------------------
    # Logs y gráficos
    # ------------------
    if t % 200 == 0:
        print(f"[t={t}] max|ω|={maxw:.3f}   E={E:.6f}   BKM={BKM:.3f}")

    line_w.set_data(ts, wmaxs)
    line_E.set_data(ts, Es)
    line_BKM.set_data(ts, BKMs)

    ax_w.set_xlim(0, t+10)
    ax_E.set_xlim(0, t+10)
    ax_BKM.set_xlim(0, t+10)

    plt.pause(0.0001)

print("\n=== FIN NAVIER–PHASE v6 ===")
print(f"Integral BKM final = {BKM}")
print("====================================")
