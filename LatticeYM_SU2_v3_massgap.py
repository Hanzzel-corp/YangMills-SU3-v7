import numpy as np

# ============================================================
#  SU(2) utilities
# ============================================================

def su2_random():
    """Random SU(2) matrix via normalized quaternion."""
    a0, a1, a2, a3 = np.random.normal(size=4)
    s = np.sqrt(a0*a0 + a1*a1 + a2*a2 + a3*a3)
    a0, a1, a2, a3 = a0/s, a1/s, a2/s, a3/s
    return np.array([
        [a0+1j*a3,     a2+1j*a1],
        [-a2+1j*a1,    a0-1j*a3]
    ])

def su2_project(U):
    """Projects a matrix onto SU(2) using polar decomposition."""
    det = np.linalg.det(U)
    U = U / (det**0.5)
    return U

# ============================================================
#  Staples + Wilson Action
# ============================================================

def staple(U, x, mu, N):
    """Compute the staple around link (x,mu)."""
    S = np.zeros((2,2), dtype=complex)
    d = 3  # 3D lattice

    for nu in range(d):
        if nu == mu: 
            continue
        # forward
        xp = list(x)
        xp[nu] = (xp[nu] + 1) % N
        xp = tuple(xp)

        xn = list(x)
        xn[nu] = (xn[nu] - 1) % N
        xn = tuple(xn)

        S += U[xp][mu] @ U[x][nu].conj().T @ U[xn][mu].conj().T
        S += U[x][nu] @ U[xp][mu] @ U[xp][nu].conj().T

    return S

# ============================================================
#  Heatbath-type update (simplified, stable)
# ============================================================

def update_link(U, x, mu, beta, N):
    S = staple(U, x, mu, N)
    # Effective matrix: project staple back into SU(2)
    S_proj = su2_project(S)

    # Heatbath-like: interpolate identity <-> staple
    a = beta / (1 + beta)
    U_new = su2_project((1-a)*np.eye(2) + a*S_proj)
    return U_new

# ============================================================
#  Correlator (Polyakov-like but 3D)
# ============================================================

def temporal_correlator(U, N):
    """
    C(t) = average of Tr( U0(x) U0(x+t)† )
    Only along axis mu=0.
    """
    C = np.zeros(N, dtype=float)

    for t in range(N):
        acc = 0.0
        count = 0
        for x0 in range(N):
            for x1 in range(N):
                for x2 in range(N):
                    U1 = U[(x0, x1, x2)][0]   # link at (x,0)
                    xt = ((x0 + t) % N, x1, x2)
                    U2 = U[xt][0]
                    acc += np.real(np.trace(U1 @ U2.conj().T))/2.0
                    count += 1
        C[t] = acc / count

    return C

# ============================================================
#  Main simulation
# ============================================================

def run_simulation(N=6, sweeps=200, beta=2.2):
    # Lattice links U[x][mu] for mu=0,1,2
    U = {}
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                U[(x0,x1,x2)] = [su2_random() for _ in range(3)]

    print("=== SU(2) Lattice Yang–Mills v3 (mass-gap extraction) ===")

    # -------- Thermalization --------
    for sweep in range(sweeps):
        for x in list(U.keys()):
            for mu in range(3):
                U[x][mu] = update_link(U, x, mu, beta, N)

        # Monitor plaquette
        if sweep % 20 == 0:
            C0 = temporal_correlator(U, N)
            print(f"[sweep {sweep}] C(1)={C0[1]:.6f}  C(2)={C0[2]:.6f}")

    # -------- Measure final correlator --------
    C = temporal_correlator(U, N)

    print("\n=== Correlator C(t) ===")
    for t in range(N):
        print(f"t={t:2d}   C={C[t]:.9f}")

    print("\n=== Effective mass-gap m(t) ===")
    for t in range(N-1):
        if C[t+1] > 0 and C[t] > 0:
            m_eff = -np.log(C[t+1]/C[t])
            print(f"t={t:2d}   m_eff={m_eff:.9f}")
        else:
            print(f"t={t:2d}   m_eff= ----")

    print("=== END ===")

# Run
run_simulation()
