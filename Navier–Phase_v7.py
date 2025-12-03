import numpy as np
import matplotlib.pyplot as plt

# ===============================
#  Navier–Phase v7 (Dynamic Rescaling)
# ===============================

N = 32
L = 2*np.pi
dx = L/N
nu = 1e-3
dt = 1e-3

# Campos
u = np.zeros((N,N,N,3))
omega_max_hist = []
energy_hist = []
time_rescaled = [0.0]

# Inicialización (perturbación suave)
x = np.linspace(0,L,N,endpoint=False)
X,Y,Z = np.meshgrid(x,x,x, indexing='ij')
u[...,0] = np.sin(Y) + 0.1*np.sin(3*Z)
u[...,1] = np.sin(Z) + 0.1*np.sin(3*X)
u[...,2] = np.sin(X) + 0.1*np.sin(3*Y)

def grad(f):
    return np.stack([
        (np.roll(f,-1,0)-np.roll(f,1,0))/(2*dx),
        (np.roll(f,-1,1)-np.roll(f,1,1))/(2*dx),
        (np.roll(f,-1,2)-np.roll(f,1,2))/(2*dx)
    ],axis=0)

def lap(f):
    return (
        np.roll(f,-1,0)+np.roll(f,1,0)+
        np.roll(f,-1,1)+np.roll(f,1,1)+
        np.roll(f,-1,2)+np.roll(f,1,2) - 6*f
    ) / dx**2

def compute_vorticity(u):
    ux, uy, uz = u[...,0], u[...,1], u[...,2]
    dux, duy, duz = grad(ux), grad(uy), grad(uz)
    wx = duy[2]-duz[1]
    wy = duz[0]-dux[2]
    wz = dux[1]-duy[0]
    return np.stack([wx,wy,wz],axis=-1)

# ================================
# MAIN LOOP
# ================================
for step in range(8000):
    omega = compute_vorticity(u)
    omega_mag = np.sqrt(np.sum(omega**2,axis=-1))
    om = np.max(omega_mag)

    # Dinamic rescaling
    if om > 10:
        u /= om          # renormaliza la velocidad
        dt /= om         # reescala el tiempo efectivo
        om = 1.0

    omega_max_hist.append(om)
    energy_hist.append(np.mean(np.sum(u**2,axis=-1)))

    # Tiempo renormalizado
    time_rescaled.append(time_rescaled[-1] + dt)

    # Calcular du/dt
    du = np.zeros_like(u)
    for j in range(3):
        du[...,j] -= (u[...,0]*grad(u[...,j])[0] +
                      u[...,1]*grad(u[...,j])[1] +
                      u[...,2]*grad(u[...,j])[2])
        du[...,j] += nu * lap(u[...,j])

    u += dt * du

    if step % 500 == 0:
        print(f"[t={step}]  max|ω|={om:.3f}   E={energy_hist[-1]:.4f}")

# ===============================
# PLOTS
# ===============================
plt.figure(figsize=(14,4))

plt.subplot(1,3,1)
plt.plot(time_rescaled,omega_max_hist)
plt.xlabel("t_rescaled"); plt.title("max|ω| renormalizado")

plt.subplot(1,3,2)
plt.plot(time_rescaled,energy_hist)
plt.xlabel("t_rescaled"); plt.title("Energía")

plt.subplot(1,3,3)
plt.plot(time_rescaled,np.cumsum(omega_max_hist)*dt)
plt.xlabel("t_rescaled"); plt.title("Integral BKM acumulada")

plt.tight_layout()
plt.show()
