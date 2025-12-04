import numpy as np

# =========================
#  SU(2) utilities
# =========================

def random_su2():
    a0, a1, a2, a3 = np.random.normal(size=4)
    norm = np.sqrt(a0*a0 + a1*a1 + a2*a2 + a3*a3)
    a0, a1, a2, a3 = a0/norm, a1/norm, a2/norm, a3/norm
    return np.array([[a0+1j*a3, a2+1j*a1],
                     [-a2+1j*a1, a0-1j*a3]], dtype=np.complex128)

def su2_mult(a, b):
    return a @ b

def su2_dagger(a):
    return np.conjugate(a.T)

# =========================
#  Lattice SU(2) structure
# =========================

def staple(U, x, mu, N):
    """Compute the staple around link U[x][mu]."""
    d = 3  # lattice dimension fixed to 3D
    st = np.zeros((2,2), dtype=np.complex128)

    for nu in range(d):
        if nu == mu:
            continue

        # forward and backward neighbors
        xp = list(x)
        xp[nu] = (xp[nu] + 1) % N
        xp = tuple(xp)

        xm = list(x)
        xm[nu] = (xm[nu] - 1) % N
        xm = tuple(xm)

        # staple contribution
        st += U[x][nu] @ U[xp][mu] @ su2_dagger(U[x][nu])
        st += su2_dagger(U[xm][nu]) @ U[xm][mu] @ U[xm][nu]

    return st

# =========================
#  Heat-bath update
# =========================

def heatbath_update(U, beta, N):
    d = 3  # number of directions

    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0, x1, x2)
                for mu in range(d):
                    S = staple(U, x, mu, N)

                    # normalize to SU(2)
                    det = np.sqrt(np.abs(np.linalg.det(S)))
                    if det == 0:
                        w = np.eye(2, dtype=np.complex128)
                    else:
                        w = S / det

                    r = random_su2()
                    U[x][mu] = r @ w
    return U

# =========================
#  Plaquette energy
# =========================

def plaquette_energy(U, N):
    d = 3
    plaq_sum = 0.0
    count = 0

    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0, x1, x2)
                for mu in range(d):
                    for nu in range(mu+1, d):

                        xp = list(x)
                        xp[mu] = (xp[mu] + 1) % N
                        xp = tuple(xp)

                        yp = list(x)
                        yp[nu] = (yp[nu] + 1) % N
                        yp = tuple(yp)

                        pl = U[x][mu] @ U[xp][nu] @ su2_dagger(U[yp][mu]) @ su2_dagger(U[x][nu])
                        plaq_sum += np.real(np.trace(pl))
                        count += 1

    return 1 - (plaq_sum / (2 * count))

# =========================
# MAIN LOOP
# =========================

if __name__ == "__main__":
    N = 6
    beta = 2.2
    sweeps = 200

    U = np.empty((N,N,N,3,2,2), dtype=np.complex128)
    for x in np.ndindex(N,N,N):
        for mu in range(3):
            U[x][mu] = np.eye(2, dtype=np.complex128)

    print("=== SU(2) Lattice Yang–Mills (Wilson action) ===")

    for t in range(sweeps):
        U = heatbath_update(U, beta, N)
        if t % 20 == 0:
            E = plaquette_energy(U, N)
            print(f"[sweep {t}] plaquette_energy = {E:.6f}")

    print("=== END ===")

