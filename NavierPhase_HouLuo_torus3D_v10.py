import numpy as np

# =====================
# Parámetros
# =====================
N = 32
L = 2*np.pi
nu = 0.0001
dt = 0.0005
steps = 20000

# =====================
# Fourier grid
# =====================
k = np.fft.fftfreq(N, d=L/(2*np.pi*N))
KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
K2 = KX**2 + KY**2 + KZ**2
K2[0,0,0] = 1e-14

# =====================
# Dealiasing
# =====================
mask = (np.abs(KX) < N/3) & (np.abs(KY) < N/3) & (np.abs(KZ) < N/3)
def dealias(a):
    return a * mask

# =====================
# Gradiente físico
# =====================
def grad(f):
    fx = np.fft.ifftn(1j*KX*np.fft.fftn(f)).real
    fy = np.fft.ifftn(1j*KY*np.fft.fftn(f)).real
    fz = np.fft.ifftn(1j*KZ*np.fft.fftn(f)).real
    return fx, fy, fz

# =====================
# Proyector de Leray
# =====================
def project(u_hat):
    dot = KX*u_hat[0] + KY*u_hat[1] + KZ*u_hat[2]
    fac = dot / K2
    return np.stack([
        u_hat[0] - fac*KX,
        u_hat[1] - fac*KY,
        u_hat[2] - fac*KZ
    ])

# =====================
# RHS
# =====================
def rhs(u_hat):
    u_hat = dealias(u_hat)
    u = np.fft.ifftn(u_hat, axes=(1,2,3)).real
    ux, uy, uz = u

    ux_x, ux_y, ux_z = grad(ux)
    uy_x, uy_y, uy_z = grad(uy)
    uz_x, uz_y, uz_z = grad(uz)

    conv0 = ux*ux_x + uy*ux_y + uz*ux_z
    conv1 = ux*uy_x + uy*uy_y + uz*uy_z
    conv2 = ux*uz_x + uy*uz_y + uz*uz_z

    conv_hat = np.fft.fftn([conv0, conv1, conv2], axes=(1,2,3))
    dissip = -nu*K2*u_hat
    return project(-conv_hat + dissip)

# =====================
# Inicialización solenoidal Hou–Luo
# =====================
A = np.zeros((3,N,N,N))
A[0] = np.sin(KY) * np.sin(KZ)
A[1] = np.sin(KZ) * np.sin(KX)
A[2] = np.sin(KX) * np.sin(KY)

u0_hat = np.fft.fftn(A, axes=(1,2,3))
u_hat = project(u0_hat)

# =====================
# Simulación
# =====================
for t in range(steps):
    k1 = rhs(u_hat)
    k2 = rhs(u_hat + 0.5*dt*k1)
    k3 = rhs(u_hat + 0.5*dt*k2)
    k4 = rhs(u_hat + dt*k3)

    u_hat += dt/6*(k1 + 2*k2 + 2*k3 + k4)

    if t % 500 == 0:
        u = np.fft.ifftn(u_hat, axes=(1,2,3)).real
        ux, uy, uz = u

        ux_x, ux_y, ux_z = grad(ux)
        uy_x, uy_y, uy_z = grad(uy)
        uz_x, uz_y, uz_z = grad(uz)

        w = np.sqrt((uy_z-uz_y)**2 + (uz_x-ux_z)**2 + (ux_y-uy_x)**2)
        print(f"[t={t}]  wmax={np.max(w):.4f}")






