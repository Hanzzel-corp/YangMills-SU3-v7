# ----------------------------------------------------------
# NAVIER–PHASE v8 — Renormalized Blow-Up Detector (Hou–Luo style)
# ----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ---------- PARAMETERS ----------
N = 32
L = 2*np.pi
dx = L/N
nu = 0.002         # small viscosity (near–singular)
dt = 0.001
steps = 8000

# ---------- INITIAL CONDITION ----------
def initial_u():
    u = np.zeros((N,N,N,3))
    x = np.linspace(0,L,N,endpoint=False)
    X,Y,Z = np.meshgrid(x,x,x,indexing='ij')

    u[...,0] =  np.sin(X)*np.cos(Y)*np.cos(Z)
    u[...,1] = -np.cos(X)*np.sin(Y)*np.cos(Z)
    u[...,2] =  np.zeros_like(X)

    return u

# ---------- NUMERICAL OPERATORS ----------
def grad(f):
    return np.array([
        (np.roll(f,-1,0)-np.roll(f,1,0))/(2*dx),
        (np.roll(f,-1,1)-np.roll(f,1,1))/(2*dx),
        (np.roll(f,-1,2)-np.roll(f,1,2))/(2*dx)
    ])

def lap(f):
    return (
        np.roll(f,-1,0)+np.roll(f,1,0)+
        np.roll(f,-1,1)+np.roll(f,1,1)+
        np.roll(f,-1,2)+np.roll(f,1,2)-6*f
    )/(dx*dx)

def vorticity(u):
    ux,uy,uz = u[...,0], u[...,1], u[...,2]
    dux, duy, duz = grad(ux), grad(uy), grad(uz)
    wx = duy[2] - duz[1]
    wy = duz[0] - dux[2]
    wz = dux[1] - duy[0]
    return wx,wy,wz

# ---------- RENORMALIZATION ----------
def renormalize(u):
    wx,wy,wz = vorticity(u)
    omega = np.sqrt(wx**2 + wy**2 + wz**2)
    wmax = np.max(omega)

    if wmax == 0 or np.isnan(wmax):
        return u, omega, 1.0

    # Normalize velocity by wmax
    u = u / wmax
    omega = omega / wmax
    return u, omega, wmax

# ---------- SIMULATION ----------
u = initial_u()

omega_hist = []
wmax_hist = []
similarity_hist = []
profile_prev = None
s_time = [0.0]

for t in range(steps):

    # ---- Renormalize before stepping ----
    u, omega, wmax = renormalize(u)

    # hyperbolic time increment
    s_time.append(s_time[-1] + dt * wmax)

    wmax_hist.append(wmax)

    # -------- store profile for shape comparison --------
    profile = np.sort(omega.flatten())
    if profile_prev is not None:
        sim = np.linalg.norm(profile - profile_prev) / np.linalg.norm(profile_prev)
    else:
        sim = 1.0
    similarity_hist.append(sim)
    profile_prev = profile

    # ------------ TIME STEP ------------
    du = np.zeros_like(u)
    for j in range(3):
        dgj = grad(u[...,j])
        du[...,j] -= (u[...,0]*dgj[0] + u[...,1]*dgj[1] + u[...,2]*dgj[2])
        du[...,j] += nu * lap(u[...,j])

    u += dt * du

    # ------------ PRINT ------------
    if t % 500 == 0:
        print(f"[t={t}]  wmax={wmax:.4f},  sim={sim:.6f}")

# ------------ PLOTS ------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(wmax_hist)
plt.title("Renormalized max|ω|")

plt.subplot(1,2,2)
plt.plot(similarity_hist)
plt.title("Profile similarity evolution")

plt.show()

