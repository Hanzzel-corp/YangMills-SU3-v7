import numpy as np

# ============================================================
#   SU(2) operations (safe & normalized)
# ============================================================

def su2_project(M):
    """Project any 2x2 complex matrix onto SU(2)."""
    alpha = np.sqrt(0.5 * np.trace(M.conj().T @ M).real)
    if alpha < 1e-12:
        return np.eye(2, dtype=np.complex128)
    return M / alpha

def random_su2_heatbath(beta, staple_norm):
    """
    Proper heatbath according to Creutz (1980).
    Generates a SU(2) matrix with correct Boltzmann weight.
    """
    k = staple_norm

    # generate a0 from P(a0) ~ a0 * sqrt(1-a0^2) * exp(beta*k*a0)
    # rejection method
    while True:
        a0 = np.random.rand()
        u = np.random.rand()

        p = a0 * np.sqrt(1 - a0*a0) * np.exp(beta * k * a0)
        q = np.exp(beta * k)   # upper bound

        if u < p/q:
            break

    # random direction in R^3
    v = np.random.normal(size=3)
    v = v / np.linalg.norm(v)

    a = np.array([
        [a0 + 1j*v[2]*np.sqrt(1-a0*a0),   (v[1]+1j*v[0])*np.sqrt(1-a0*a0)],
        [-(v[1]-1j*v[0])*np.sqrt(1-a0*a0),  a0 - 1j*v[2]*np.sqrt(1-a0*a0)]
    ], dtype=np.complex128)

    return a


# ============================================================
#   Lattice Yang–Mills
# ============================================================

def staple(U, x, mu, N):
    d = 3
    S = np.zeros((2,2), dtype=np.complex128)

    for nu in range(d):
        if nu == mu:
            continue

        xp = list(x)
        xp[nu] = (xp[nu] + 1) % N
        xp = tuple(xp)

        xm = list(x)
        xm[nu] = (xm[nu] - 1) % N
        xm = tuple(xm)

        S += U[x][nu] @ U[xp][mu] @ U[x][nu].conj().T
        S += U[xm][nu].conj().T @ U[xm][mu] @ U[xm][nu]

    return S


def heatbath_update(U, beta, N):
    d = 3

    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0,x1,x2)

                for mu in range(d):
                    S = staple(U, x, mu, N)

                    # normalize staple
                    S_proj = su2_project(S)
                    k = np.sqrt(np.linalg.det(S_proj)).real

                    R = random_su2_heatbath(beta, k)

                    U[x][mu] = R @ S_proj

    return U


def plaquette_energy(U, N):
    d = 3
    P = 0
    count = 0

    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0,x1,x2)

                for mu in range(d):
                    for nu in range(mu+1, d):

                        xp = list(x); xp[mu]=(xp[mu]+1)%N; xp=tuple(xp)
                        yp = list(x); yp[nu]=(yp[nu]+1)%N; yp=tuple(yp)

                        pl = U[x][mu] @ U[xp][nu] @ \
                             U[yp][mu].conj().T @ U[x][nu].conj().T

                        P += np.real(np.trace(pl))
                        count += 1

    return 1 - P/(2*count)


# ============================================================
#   MAIN
# ============================================================

if __name__ == "__main__":
    N = 6
    beta = 2.0
    sweeps = 200

    U = np.empty((N,N,N,3,2,2), dtype=np.complex128)
    for x in np.ndindex(N,N,N):
        for mu in range(3):
            U[x][mu] = np.eye(2)

    print("=== SU(2) Lattice Yang–Mills v2 (stable) ===")

    for t in range(sweeps):
        U = heatbath_update(U, beta, N)

        if t % 20 == 0:
            E = plaquette_energy(U, N)
            print(f"[sweep {t}] Plaquette = {E:.6f}")

    print("=== END ===")
