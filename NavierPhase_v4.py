import numpy as np
import matplotlib.pyplot as plt

# ==========================================
#  NAVIER–PHASE v4b (3D Real + Test BKM)
# ==========================================

# Dominio y discretización
N   = 24        # malla 24^3 (más liviano que 32)
L   = 1.0
dx  = L / N
dt  = 5e-4      # paso de tiempo más chico
steps = 8000    # ~4 unidades de tiempo

# Parámetros físicos / de fase
nu    = 0.05    # viscosidad (más grande para estabilizar)
gamma = 0.3     # fuerza de la disipación de fase
omega_c = 5.0   # umbral de vorticidad para activar la fase

# ==========================================
#   Inicialización del campo de velocidad
# ==========================================

x = np.linspace(0, L, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

# Campo inicial: mezcla de modos (con vorticidad no trivial)
amp = 0.2
u = np.zeros((N, N, N, 3), dtype=np.float64)
u[..., 0] =  amp * np.sin(2*np.pi*Y) * np.cos(2*np.pi*Z)        # u_x
u[..., 1] = -amp * np.sin(2*np.pi*Z) * np.cos(2*np.pi*X)        # u_y
u[..., 2] =  amp * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)        # u_z

# ==========================================
#   Operadores diferenciales (periódicos)
# ==========================================

def grad(f):
    """Gradiente de un escalar f en 3D (condiciones periódicas)."""
    fx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dx)
    fy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2*dx)
    fz = (np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)) / (2*dx)
    return fx, fy, fz

def laplacian(f):
    """Laplaciano componente a componente."""
    return (
        np.roll(f, -1, axis=0) + np.roll(f, 1, axis=0) +
        np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) +
        np.roll(f, -1, axis=2) + np.roll(f, 1, axis=2) -
        6.0 * f
    ) / dx**2

def curl(u):
    """Vorticidad: ω = ∇ × u."""
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    dux = grad(ux)
    duy = grad(uy)
    duz = grad(uz)

    wx = duy[2] - duz[1]
    wy = duz[0] - dux[2]
    wz = dux[1] - duy[0]

    return np.stack([wx, wy, wz], axis=-1)

# ==========================================
#   Métricas y listas para gráficos
# ==========================================

max_omega_list = []
energy_list    = []
sigma_list     = []
BKM_list       = []

BKM_int = 0.0

# Umbral de “blow-up numérico” (seguro)
U_MAX_NUM = 1e3

# ==========================================
#   Loop temporal principal
# ==========================================

for t in range(steps):

    # vorticidad y normas
    omega = curl(u)
    omega_norm = np.linalg.norm(omega, axis=-1)
    max_omega  = float(np.max(omega_norm))

    # energía (cuadrado medio de la velocidad)
    u2 = np.sum(u*u, axis=-1)
    E  = float(np.mean(u2))

    # Test BKM: acumulamos ∫ ||ω||_∞ dt
    if np.isfinite(max_omega):
        BKM_int += max_omega * dt
    else:
        print(f"[t={t}] max|ω| no finito → paro simulación.")
        break

    # fase σ(t): se activa cuando la vorticidad supera omega_c
    sigma = 1.0 if max_omega > omega_c else 0.0

    # Guardamos para graficar
    max_omega_list.append(max_omega)
    energy_list.append(E)
    sigma_list.append(sigma)
    BKM_list.append(BKM_int)

    # Chequeo de estabilidad numérica (blow-up)
    max_u = float(np.max(np.abs(u)))
    if (not np.isfinite(max_u)) or (max_u > U_MAX_NUM):
        print(f"[t={t}] ¡Blow-up numérico! ||u||_∞ ≈ {max_u:.3e}")
        break

    # ------------------------------
    #    Término convectivo u·∇u
    # ------------------------------
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    dux = grad(ux)
    duy = grad(uy)
    duz = grad(uz)

    conv = np.stack([
        ux*dux[0] + uy*dux[1] + uz*dux[2],
        ux*duy[0] + uy*duy[1] + uz*duy[2],
        ux*duz[0] + uy*duz[1] + uz*duz[2]
    ], axis=-1)

    # difusión viscosa
    diff = nu * laplacian(u)

    # disipación de fase: sólo actúa cuando sigma = 1
    phase = -gamma * sigma * max_omega * u

    # actualización explícita
    u = u + dt * (-conv + diff + phase)

    if t % 500 == 0:
        print(f"[t={t}] max|ω|={max_omega:.3f}  BKM={BKM_int:.3f}  σ={sigma}  E={E:.3f}")

# ==========================================
#           Gráficos
# ==========================================

iters = np.arange(len(max_omega_list))

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(iters, max_omega_list)
plt.axhline(omega_c, color="k", ls="--", label="ω_c")
plt.title("Máxima vorticidad")
plt.xlabel("iteración")
plt.legend()

plt.subplot(1,3,2)
plt.plot(iters, energy_list)
plt.title("Energía E(t)")
plt.xlabel("iteración")

plt.subplot(1,3,3)
plt.plot(iters, BKM_list)
plt.title("Integral BKM acumulada")
plt.xlabel("iteración")

plt.tight_layout()
plt.show()

print("\n=== DIAGNÓSTICO FINAL v4b ===")
print(f"Iteraciones efectivas: {len(max_omega_list)}")
print(f"Integral BKM = {BKM_int:.4f}")
print(f"¿σ(t) se activó?: {'Sí' if max(sigma_list) > 0 else 'No'}")
print("=================================")

