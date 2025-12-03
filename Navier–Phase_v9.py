# ============================================================
# Navier–Phase v9 – Forcing Explosivo (ω × (∇×ω))
# Simulación 3D con renormalización dinámica tipo Hou–Luo
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Parámetros del dominio
# ------------------------------
N = 32
L = 2*np.pi
dx = L/N
dt = 0.0005
nu = 0.002        # viscosidad muy baja
A_force = 5.0     # amplitud del forcing explosivo

# ------------------------------
# Derivadas y Laplaciano
# ------------------------------
def grad(f):
    return np.stack([
        (np.roll(f,-1,0) - np.roll(f,1,0))/(2*dx),
        (np.roll(f,-1,1) - np.roll(f,1,1))/(2*dx),
        (np.roll(f,-1,2) - np.roll(f,1,2))/(2*dx)
    ], axis=0)

def lap(f):
    return (
        np.roll(f,-1,0)+np.roll(f,1,0)+
        np.roll(f,-1,1)+np.roll(f,1,1)+
        np.roll(f,-1,2)+np.roll(f,1,2)-6*f
    ) / dx**2

# ------------------------------
# Campo inicial caótico
# ------------------------------
x = np.linspace(0,L,N,endpoint=False)
X,Y,Z = np.meshgrid(x,x,x,indexing='ij')

u = np.zeros((N,N,N,3))
u[...,0] = np.sin(Y)+0.3*np.cos(Z)
u[...,1] = np.sin(Z)+0.3*np.cos(X)
u[...,2] = np.sin(X)+0.3*np.cos(Y)
u *= 1.5   # más energía inicial

# ------------------------------
# Vorticidad
# ------------------------------
def compute_vorticity(u):
    ux,uy,uz = u[...,0],u[...,1],u[...,2]
    dux, duy, duz = grad(ux), grad(uy), grad(uz)
    wx = duy[2]-duz[1]
    wy = duz[0]-dux[2]
    wz = dux[1]-duy[0]
    return np.stack([wx,wy,wz],axis=-1)

# ------------------------------
# Forcing explosivo: ω × (∇×ω)
# ------------------------------
def explosive_force(omega):
    curl_w = np.stack(grad(omega[...,0]),axis=-1)
    curl_w = np.stack([curl_w[...,0],
                       grad(omega[...,1])[1],
                       grad(omega[...,2])[2]],axis=-1)

    cross = np.cross(omega, curl_w)
    return A_force * cross

# ------------------------------
# Renormalización dinámica
# ------------------------------
W0 = 2.0
time_rescaled = [0]
omega_hist = []

# ------------------------------
# Loop principal
# ------------------------------
steps = 8000
t = 0.0

for n in range(steps):

    omega = compute_vorticity(u)
    wmax = np.max(np.abs(omega))
    omega_hist.append(wmax)

    # --- RENORMALIZAR ---
    if wmax > 0:
        u *= W0 / wmax
        omega = compute_vorticity(u)
        wmax = np.max(np.abs(omega))

    # forcing explosivo
    F = explosive_force(omega)

    # Euler explícito
    du = np.zeros_like(u)
    for j in range(3):
        du[...,j] -= (u[...,0]*grad(u[...,j])[0] +
                      u[...,1]*grad(u[...,j])[1] +
                      u[...,2]*grad(u[...,j])[2])
        du[...,j] += nu * lap(u[...,j])
        du[...,j] += F[...,j]

    # avanzar
    u += dt * du
    t += dt * (wmax/W0)
    time_rescaled.append(t)

    if n % 500 == 0:
        print(f"[t={n}] wmax={wmax:.4f}")

# ------------------------------
# Graficar
# ------------------------------
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(omega_hist)
plt.title("Renormalized max|ω|")
plt.xlabel("iteración")

plt.subplot(1,2,2)
plt.plot(time_rescaled[:-1])
plt.title("Tiempo rescalado Hou–Luo")
plt.xlabel("iteración")

plt.tight_layout()
plt.show()


