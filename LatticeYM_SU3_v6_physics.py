import numpy as np

# ======================
#   SU(3) HELPERS
# ======================

def random_su3(eps=0.1):
    """Generate a small random SU(3) rotation"""
    M = np.eye(3, dtype=np.complex128) + eps*(np.random.randn(3,3) + 1j*np.random.randn(3,3))
    # Project to SU(3)
    U,_,Vt = np.linalg.svd(M)
    return U @ Vt

def staple(U, x, mu, N):
    """Compute the SU(3) staple around link (x,mu) in 3D"""
    d = 3
    s = np.zeros((3,3), dtype=np.complex128)

    for nu in range(d):
        if nu == mu: 
            continue

        # x
        x0,x1,x2 = x

        # forward shifted
        xp = [ (x0 + (1 if nu==0 else 0)) % N,
               (x1 + (1 if nu==1 else 0)) % N,
               (x2 + (1 if nu==2 else 0)) % N ]

        # backward shifted
        xm = [ (x0 - (1 if nu==0 else 0)) % N,
               (x1 - (1 if nu==1 else 0)) % N,
               (x2 - (1 if nu==2 else 0)) % N ]

        # staple = U(x,nu) U(x+nu,mu) U^\dagger(x+mu,nu) + backwards term
        s += U[x][nu] @ U[tuple(xp)][mu] @ U[tuple(xm)][nu].conj().T
        s += U[tuple(xm)][nu].conj().T @ U[tuple(xm)][mu] @ U[tuple(xm)][nu]

    return s

# ======================
#   UPDATE RULES
# ======================

def heatbath(U, beta, N, eps=0.08):
    """Approximate SU(3) heatbath: move each link toward staple with noise"""
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0,x1,x2)
                for mu in range(3):
                    S = staple(U, x, mu, N)
                    # Normalize S
                    U_new = S + eps*random_su3()
                    U_new = project_su3(U_new)
                    U[x][mu] = U_new
    return U

def overrelax(U, beta, N, alpha=0.6):
    """Overrelaxation step (reflection around staple)"""
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0,x1,x2)
                for mu in range(3):
                    S = staple(U, x, mu, N)
                    U_link = U[x][mu]
                    U[x][mu] = project_su3( (1+alpha)*S - alpha*U_link )
    return U

def project_su3(M):
    """Project a matrix to SU(3) using SVD"""
    U,_,Vt = np.linalg.svd(M)
    return U @ Vt

# ======================
#   MEASUREMENTS
# ======================

def plaquette(U, N):
    """Wilson plaquette average"""
    plaq = 0.0
    count = 0

    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0,x1,x2)
                for mu in range(3):
                    for nu in range(mu+1,3):
                        xp = [ (x0 + (1 if mu==0 else 0)) % N,
                               (x1 + (1 if mu==1 else 0)) % N,
                               (x2 + (1 if mu==2 else 0)) % N ]

                        xn = [ (x0 + (1 if nu==0 else 0)) % N,
                               (x1 + (1 if nu==1 else 0)) % N,
                               (x2 + (1 if nu==2 else 0)) % N ]

                        Uplaq = (U[x][mu] @ U[tuple(xp)][nu] @
                                 U[tuple(x)][mu].conj().T @ U[tuple(xn)][nu].conj().T)

                        plaq += np.real(np.trace(Uplaq))/3.0
                        count += 1

    return plaq / count

def glueball_correlator(U, N, t_max=6):
    """Compute rough 0++ correlator using local operator O = Tr(U_mu nu)"""
    C = {}
    Ovals = []

    for t in range(N):
        opslice = []
        for x0 in range(N):
            for x1 in range(N):
                x = (x0,x1,t)
                # simple operator: trace of mu=0,nu=1 plaquette
                mu,nu = 0,1
                xp = [ (x0 + 1)%N, x1, t ]
                xn = [ x0, (x1+1)%N, t ]
                Upl = U[x][0] @ U[tuple(xp)][1] @ U[x][0].conj().T @ U[tuple(xn)][1].conj().T
                opslice.append(np.real(np.trace(Upl)))
        Ovals.append(np.mean(opslice))

    for t in range(1,t_max+1):
        C[t] = np.mean([Ovals[i] * Ovals[(i+t)%N] for i in range(N)])
    return C

def wilson_loop(U, N, R, T):
    """Rectangular Wilson loop R x T"""
    val = 0.0
    count = 0

    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0,x1,x2)

                W = np.eye(3,dtype=np.complex128)

                # forward in mu=0 direction
                xx = list(x)
                for _ in range(R):
                    W = W @ U[tuple(xx)][0]
                    xx[0] = (xx[0] + 1) % N

                # forward in mu=1 direction
                for _ in range(T):
                    W = W @ U[tuple(xx)][1]
                    xx[1] = (xx[1] + 1) % N

                # backward in mu=0
                for _ in range(R):
                    xx[0] = (xx[0] - 1) % N
                    W = W @ U[tuple(xx)][0].conj().T

                # backward in mu=1
                for _ in range(T):
                    xx[1] = (xx[1] - 1) % N
                    W = W @ U[tuple(xx)][1].conj().T

                val += np.real(np.trace(W))/3
                count += 1

    return val / count

def polyakov_loop(U, N):
    """Polyakov loop along time direction (mu=2)"""
    P = np.zeros((3,3),dtype=np.complex128)

    for x0 in range(N):
        for x1 in range(N):
            W = np.eye(3,dtype=np.complex128)
            xx = (x0,x1,0)
            for t in range(N):
                W = W @ U[xx][2]
                xx = (xx[0], xx[1], (xx[2]+1)%N)
            P += W
    P /= (N*N)
    return (np.trace(P)/3).real


# ======================
#   MAIN PROGRAM
# ======================

if __name__ == "__main__":

    N = 8
    beta = 5.6
    sweeps = 80

    # Initialize SU(3) links
    U = {}
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                U[(x0,x1,x2)] = [random_su3(0.01) for _ in range(3)]

    print("=== SU(3) Lattice Yang–Mills v6 PHYSICS ===")

    for sw in range(sweeps):
        U = heatbath(U, beta, N, eps=0.02)
        U = overrelax(U, beta, N, alpha=0.3)

        if sw % 10 == 0:
            pl = plaquette(U,N)
            print(f"[sweep {sw}] Plaquette = {pl}")

    print("\n=== Glueball correlator C(t) ===")
    C = glueball_correlator(U,N,t_max=5)
    for t,val in C.items():
        print(f"t={t}  C={val}")

    print("\n=== Wilson loops ===")
    for R in [1,2]:
        for T in [1,2]:
            print(f"W({R},{T}) = {wilson_loop(U,N,R,T)}")

    print("\n=== Polyakov loop ===")
    print(f"P = {polyakov_loop(U,N)}")

    print("\n=== END ===")
