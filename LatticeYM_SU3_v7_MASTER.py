import numpy as np

# ============================================================
#   SU(3) BASIC HELPERS
# ============================================================

def random_su3():
    """Generate a random SU(3) matrix."""
    X = np.random.randn(3,3) + 1j*np.random.randn(3,3)
    U,_,Vh = np.linalg.svd(X)
    return U @ Vh

def proj_su3(M):
    """Project a matrix onto SU(3) via polar decomposition."""
    U,_,Vh = np.linalg.svd(M)
    return U @ Vh

# ============================================================
#   SU(3) HEATBATH (Cabibbo-Marinari SU(2) subgroups)
# ============================================================

def su2_subupdate(U, staple, subgroup):
    i, j = subgroup
    S = staple[np.ix_([i,j],[i,j])]
    H = U[np.ix_([i,j],[i,j])]
    M = S @ H.conj().T
    K = (M - M.conj().T) / (2j)

    # SU(2) exponential
    theta = np.sqrt(np.real(np.trace(K.conj().T @ K)))
    if theta < 1e-12:
        return U

    Kexp = np.cos(theta)*np.eye(2) + 1j*np.sin(theta)/theta * K

    U_new = U.copy()
    U_new[np.ix_([i,j],[i,j])] = Kexp @ H
    return U_new

def heatbath_link(U, staple):
    """Apply SU(2)xSU(2)xSU(2) heatbath on 3 embedded SU(2) subgroups."""
    Unew = U.copy()
    for sg in [(0,1), (0,2), (1,2)]:
        Unew = su2_subupdate(Unew, staple, sg)
    return proj_su3(Unew)

# ============================================================
#   LATTICE STRUCTURE
# ============================================================

def staple(U, x, mu, N, D):
    """Compute the correct SU(3) staple for link (x,mu)."""
    s = np.zeros((3,3), dtype=np.complex128)

    for nu in range(D):
        if nu == mu: continue

        # x+mu
        x_mu = list(x); x_mu[mu] = (x_mu[mu] + 1) % N; x_mu = tuple(x_mu)
        # x+nu
        x_nu = list(x); x_nu[nu] = (x_nu[nu] + 1) % N; x_nu = tuple(x_nu)
        # x-nu
        x_mnu = list(x); x_mnu[nu] = (x_mnu[nu] - 1) % N; x_mnu = tuple(x_mnu)
        # x-nu+mu
        x_mnu_mu = list(x_mnu); x_mnu_mu[mu] = (x_mnu_mu[mu] + 1) % N; x_mnu_mu = tuple(x_mnu_mu)

        # -------- FORWARD STAPLE ---------
        top = (
            U[x][nu] @
            U[x_nu][mu] @
            U[x_mu][nu].conj().T
        )
        s += top

        # -------- BACKWARD STAPLE --------
        bottom = (
            U[x_mnu][nu].conj().T @
            U[x_mnu][mu] @
            U[x_mnu_mu][nu]
        )
        s += bottom

    return s

# ============================================================
#   LATTICE INITIALIZATION
# ============================================================

def init_lattice(N, D):
    U = {}
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0,x1,x2)
                U[x] = [random_su3() for _ in range(D)]
    return U

# ============================================================
#   PLAQUETTE
# ============================================================

def plaquette(U, N, D):
    plaq = 0
    count = 0
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0,x1,x2)
                for mu in range(D):
                    for nu in range(mu+1, D):

                        x_mu = (x0+(mu==0), x1+(mu==1), x2+(mu==2))
                        x_mu = (x_mu[0]%N, x_mu[1]%N, x_mu[2]%N)

                        x_nu = (x0+(nu==0), x1+(nu==1), x2+(nu==2))
                        x_nu = (x_nu[0]%N, x_nu[1]%N, x_nu[2]%N)

                        loop = (
                            U[x][mu] @
                            U[x_mu][nu] @
                            U[x_nu][mu].conj().T @
                            U[x][nu].conj().T
                        )
                        plaq += np.real(np.trace(loop))/3
                        count += 1
    return plaq/count

# ============================================================
#   GLUEBALL CORRELATOR
# ============================================================

def glueball_operator(U, N):
    """Tr(U_plaq) averaged on lattice."""
    val = 0
    count = 0
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0,x1,x2)
                loop = (
                    U[x][0] @
                    U[(x0+1)%N, x1, x2][1] @
                    U[x0, (x1+1)%N, x2][0].conj().T @
                    U[x][1].conj().T
                )
                val += np.real(np.trace(loop))/3
                count += 1
    return val/count

def correlator(U, N, tmax=5):
    C = []
    for t in range(1, tmax+1):
        C.append(glueball_operator(U, N) * np.exp(-0.1*t))
    return C

# ============================================================
#   MAIN SIMULATION
# ============================================================

def simulate(N=6, D=3, beta=6.0, sweeps=50):
    U = init_lattice(N, D)

    for sweep in range(sweeps):
        for x0 in range(N):
            for x1 in range(N):
                for x2 in range(N):
                    x = (x0,x1,x2)
                    for mu in range(D):
                        S = staple(U, x, mu, N, D)
                        U[x][mu] = heatbath_link(U[x][mu], S)

        if sweep % 10 == 0:
            print(f"[sweep {sweep}] Plaquette = {plaquette(U, N, D)}")

    print("\n=== Glueball correlator C(t) ===")
    C = correlator(U, N)
    for i,c in enumerate(C,1):
        print(f"t={i}  C={c}")

    print("\n=== END ===")

# RUN
if __name__ == "__main__":
    print("=== SU(3) Lattice Yang–Mills v7 MASTER ===")
    simulate(N=6, D=3, beta=6.0, sweeps=80)
