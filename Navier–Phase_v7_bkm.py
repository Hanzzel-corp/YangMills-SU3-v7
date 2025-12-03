import numpy as np
import matplotlib.pyplot as plt

N = 32
L = 2*np.pi
dx = L/N
nu = 1e-3
dt = 1e-3

u = np.zeros((N,N,N,3))
bkm_integral = []
omega_max_hist = []
energy_hist = []

x = np.linspace(0,L,N,endpoint=False)
X,Y,Z = np.meshgrid(x,x,x, indexing='ij')
u[...,0] = np.sin(Y) + 0.3*np.sin(3*Z)
u[...,1] = np.sin(Z) + 0.3*np.sin(3*X)
u[...,2] = np.sin(X) + 0.3*np.sin(3*Y)

def grad(f):
    return np.stack([
        (np.roll(f,-1,0)-np.roll(f,1,0))/(2*dx),
        (np.roll(f,-1,1)-np.roll(f,1,1))/(2*dx),
        (np.roll(f,-1,2)-np.roll(f,1,2))/(2*dx)
    ],axis=0)

def lap(f):
    return (np.roll(f,-1,0)+np.roll(f,1,0)+np.roll(f,-1,1)+np.roll(f,1,1)+
            np.roll(f,-1,2)+np.roll(f,1,2)-6*f)/dx**2

def compute_vorticity(u):
    ux, uy, uz = u[...,0], u[...,1], u[...,2]
    dux, duy, duz = grad(ux), grad(uy), grad(uz)
    wx = duy[2]-duz[1]
    wy = duz[0]-dux[2]
    wz = dux[1]-duy[0]
    return np.sqrt(wx**2 + wy**2 + wz**2)

# =========================
# MAIN LOOP
# =========================
bkm = 0.0
for t in range(6000):
    w = compute_vorticity(u)
    om = np.max(w)

    omega_max_hist.append(om)
    energy_hist.append(np.mean(np.sum(u**2,axis=-1)))

    bkm += om * dt
    bkm_integral.append(bkm)

    du = np.zeros_like(u)
    for j in range(3):
        du[...,j] -= (u[...,0]*grad(u[...,j])[0] +
                      u[...,1]*grad(u[...,j])[1] +
                      u[...,2]*grad(u[...,j])[2])
        du[...,j] += nu * lap(u[...,j])

    u += dt*du

    if t % 500 == 0:
        print(f"[t={t}] max|ω|={om:.3f},  BKM={bkm:.3f}")

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(omega_max_hist); plt.title("max|ω|")

plt.subplot(1,3,2)
plt.plot(energy_hist); plt.title("E(t)")

plt.subplot(1,3,3)
plt.plot(bkm_integral); plt.title("Integral BKM(t)")

plt.tight_layout()
plt.show()

