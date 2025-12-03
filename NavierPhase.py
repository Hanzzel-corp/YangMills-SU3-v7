import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ======================================
#         NAVIER–PHASE v3 (estable)
# ======================================

nu = 0.1
kappa = 0.8
gamma = 0.35
Psi_c = 1.5       # mucho más sensible
E_max = 200       # más realista
dt0 = 0.001
steps = 8000

def step(u, dt, sigma):
    x,y,z = u

    omega = np.array([y - z, z - x, x - y])
    vort = np.linalg.norm(omega)

    dudt = np.array([
        y*z - nu*x,
        z*x - nu*y,
        x*y - nu*z
    ])

    # Regularización Leray-α
    dudt = dudt / (1 + 0.1*np.linalg.norm(u))

    if sigma == 1:
        dudt -= kappa * u
        dudt -= gamma * u * np.linalg.norm(u)

    return u + dt * dudt, vort

def simulate(u0):
    u = np.array(u0,float)
    traj = []
    sigma = 0
    dt = dt0

    for _ in tqdm(range(steps)):
        E = np.linalg.norm(u)**2

        # transición de fase precoz
        if np.isnan(E) or E > E_max or sigma == 1:
            sigma = 1
            dt = dt0 * 0.2  # dt adaptativo (más chico)

        # integración segura
        u_next, vort = step(u, dt, sigma)

        # saturación física
        if np.linalg.norm(u_next) > 50:
            u_next = u_next / np.linalg.norm(u_next) * 50

        traj.append([u_next[0],u_next[1],u_next[2],vort,E,sigma])
        u = u_next

    return np.array(traj)

traj = simulate([0.8,0.4,-0.5])
x,y,z = traj[:,0], traj[:,1], traj[:,2]
vort, E, sigma = traj[:,3], traj[:,4], traj[:,5]

fig = plt.figure(figsize=(12,5))

ax = fig.add_subplot(121, projection="3d")
ax.plot(x,y,z,lw=0.7)
ax.set_title("Trayectoria 3D Navier–Phase v3")

ax2 = fig.add_subplot(122)
ax2.plot(vort,label="‖ω‖",color="red")
ax2.plot(E,label="E(t)",color="blue")
ax2.plot(sigma*max(E),label="σ(t)",color="green")
ax2.axhline(Psi_c,ls="--",color="black")
ax2.legend()
ax2.set_title("Vorticidad, Energía y Fase")

plt.tight_layout()
plt.show()

