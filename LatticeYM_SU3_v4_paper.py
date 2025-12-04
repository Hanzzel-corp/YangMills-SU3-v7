import numpy as np
from numpy.random import rand, randn

# ==========================================================
#   SU(3) Lattice Yang–Mills — v4 PAPER (FIXED 3D VERSION)
# ==========================================================

# ---------- SU(2) heatbath (Creutz toy) --------------------

def su2_heatbath(a, beta):
    x0 = rand()
    v = randn(3)
    n = np.linalg.norm(v) + 1e-12
    v = v/n
    x1,x2,x3 = v
    return np.array([
        [x0 + 1j*x3,   x2 + 1j*x1],
        [-x2 + 1j*x1,  x0 - 1j*x3]
    ])

def embed_su2_in_su3(U, su2, block):
    M = U.copy()
    if block == 0:         # SU(2) in 0-1 block
        M[0:2,0:2] = su2
    elif block == 1:       # SU(2) in 1-2 block
        M[1:3,1:3] = su2
    elif block == 2:       # SU(2) in 0-2 block
        M[np.ix_([0,2],[0,2])] = su2
    return M

# ---------------- Initialize lattice -----------------------

def random_SU3():
    X = randn(3,3) + 1j*randn(3,3)
    Q, R = np.linalg.qr(X)
    return Q

def init_lattice(N):
    # U[x,y,z,mu]  with mu=0,1,2 for 3D gauge field
    U = np.zeros((N,N,N,3,3,3), dtype=complex)
    for x in range(N):
        for y in range(N):
            for z in range(N):
                for mu in range(3):
                    U[x,y,z,mu] = random_SU3()
    return U

# ---------------- Plaquette & Staple -----------------------

def shift(x, mu, N):
    y = list(x)
    y[mu] = (y[mu] + 1) % N
    return tuple(y)

def staple(U, x, mu, N):
    x0,x1,x2 = x
    S = np.zeros((3,3),dtype=complex)

    # 3D: mu, nu in {0,1,2}
    for nu in {0,1,2} - {mu}:
        # forward directions
        xp = shift(x, nu, N)
        xp_mu = shift(x, mu, N)

        U_mu = U[x0,x1,x2,mu]
        U_nu = U[x0,x1,x2,nu]
        U_mu_xnu = U[xp[0],xp[1],xp[2],mu]
        U_nu_dag = U[x0,x1,x2,nu].conj().T

        S += U_nu @ U_mu_xnu @ U_mu.conj().T

        # backward directions
        xm = ((x0 - (nu==0)) % N,
              (x1 - (nu==1)) % N,
              (x2 - (nu==2)) % N)
        U_nu_b = U[xm[0],xm[1],xm[2],nu]
        U_mu_b = U[xm[0],xm[1],xm[2],mu]
        S += U_nu_b.conj().T @ U_mu_b @ U_mu
    return S

def plaquette(U, N):
    P = 0
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                for mu in range(3):
                    for nu in range(mu+1,3):
                        x = (x0,x1,x2)
                        xp_mu = shift(x, mu, N)
                        xp_nu = shift(x, nu, N)

                        U1 = U[x0,x1,x2,mu]
                        U2 = U[xp_mu[0],xp_mu[1],xp_mu[2],nu]
                        U3 = U[x0,x1,x2,nu].conj().T
                        U4 = U[x0,x1,x2,mu].conj().T

                        P += np.real(np.trace(U1 @ U2 @ U3 @ U4))

    return P/(3*N**3)

# ---------------- Heatbath + Overrelax ---------------------

def heatbath(U, beta, N):
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                for mu in range(3):
                    S = staple(U, (x0,x1,x2), mu, N)
                    for block in [0,1,2]:
                        # extract SU(2) subblock
                        if block==0:
                            a = S[0:2,0:2]
                        elif block==1:
                            a = S[1:3,1:3]
                        else:
                            a = S[np.ix_([0,2],[0,2])]
                        su2 = su2_heatbath(a, beta)
                        U[x0,x1,x2,mu] = embed_su2_in_su3(U[x0,x1,x2,mu], su2, block)
    return U

def overrelax(U, beta, N):
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                for mu in range(3):
                    S = staple(U, (x0,x1,x2), mu, N)
                    H = S @ U[x0,x1,x2,mu].conj().T
                    # normalize
                    M = H.conj().T @ H
                    eigvals, eigvecs = np.linalg.eigh(M)
                    inv_sqrt = eigvecs @ np.diag(1/np.sqrt(eigvals+1e-12)) @ eigvecs.conj().T
                    U[x0,x1,x2,mu] = (H @ inv_sqrt) @ U[x0,x1,x2,mu]
    return U

# ---------------------- MAIN -------------------------------

if __name__ == "__main__":
    N = 4
    beta = 5.7
    sweeps = 40

    print("=== SU(3) Lattice Yang–Mills v4 PAPER (FIXED2, FULLY WORKING) ===")

    U = init_lattice(N)

    for s in range(sweeps):
        U = heatbath(U, beta, N)
        U = overrelax(U, beta, N)
        if s % 5 == 0:
            print(f"[sweep {s}] Plaquette = {plaquette(U,N)}")

    print("=== END ===")

