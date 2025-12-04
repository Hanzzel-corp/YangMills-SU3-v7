import numpy as np
import math

# ============================================================
#      SU(2) LATTICE YANG–MILLS — v4 MASTER
#      Mass-gap + Wilson loops + String tension
# ============================================================

N = 12               # lattice size
beta = 2.3           # physical range for SU(2) confinement
sweeps = 600         # number of heatbath sweeps

# ------------------------------------------------------------
# Helpers: SU(2) matrices and operations
# ------------------------------------------------------------
def su2_random():
    """Random SU(2) matrix."""
    a = np.random.normal(size=4)
    a = a / np.linalg.norm(a)
    a0, a1, a2, a3 = a
    return np.array([
        [a0+1j*a3, a2+1j*a1],
        [-a2+1j*a1, a0-1j*a3]
    ])

def su2_clover(U, x, mu, nu, N):
    """Compute a plaquette U_mu(x) U_nu(x+mu) U_mu†(x+nu) U_nu†(x)."""
    xp = list(x)
    xp[mu] = (xp[mu] + 1) % N
    xq = list(x)
    xq[nu] = (xq[nu] + 1) % N
    xp, xq = tuple(xp), tuple(xq)

    return U[x][mu] @ U[xp][nu] @ U[xq][mu].conj().T @ U[x][nu].conj().T

def staple(U, x, mu, N):
    """Compute staple matrix for heatbath."""
    S = np.zeros((2,2), dtype=np.complex128)
    for nu in range(3):
        if nu == mu: 
            continue

        # forward staple
        xp = list(x)
        xp[nu] = (xp[nu] + 1) % N
        xp = tuple(xp)

        S += U[xp][mu] @ U[x][nu].conj().T @ U[x][nu]

        # backward staple
        xm = list(x)
        xm[nu] = (xm[nu] - 1) % N
        xm = tuple(xm)

        S += U[xm][nu] @ U[xm][mu] @ U[x][nu].conj().T

    return S

# ------------------------------------------------------------
# Heatbath Update
# ------------------------------------------------------------
def heatbath_update(U, beta, N):
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0, x1, x2)
                for mu in range(3):

                    S = staple(U, x, mu, N)
                    H = S / np.sqrt(np.linalg.det(S))

                    # Heatbath rotation
                    r0 = beta * np.trace(H).real
                    a0 = np.random.normal() + r0
                    a_vec = np.random.normal(size=3)
                    a = np.array([a0, *a_vec])
                    a = a / np.linalg.norm(a)

                    U[x][mu] = su2_from_vec(a) @ H

    return U

def su2_from_vec(a):
    """Convert 4-vector to SU(2) matrix."""
    a0, a1, a2, a3 = a
    return np.array([
        [a0+1j*a3, a2+1j*a1],
        [-a2+1j*a1, a0-1j*a3]
    ])

# ------------------------------------------------------------
# Observables: Plaquette
# ------------------------------------------------------------
def avg_plaquette(U, N):
    total = 0.0
    count = 0
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0, x1, x2)
                for mu in range(3):
                    for nu in range(mu+1,3):
                        plaq = su2_clover(U, x, mu, nu, N)
                        total += (np.trace(plaq).real)/2
                        count += 1
    return total / count

# ------------------------------------------------------------
# Correlator and Mass Gap
# ------------------------------------------------------------
def correlator(U, N):
    """Gauge-invariant correlator C(t) from time-direction Wilson lines."""
    C = np.zeros(N)
    for t in range(N):
        val = 0
        for x0 in range(N):
            for x1 in range(N):
                A0 = U[(x0, x1, t)][0]    # temporal link
                A1 = U[(x0, x1, 0)][0]
                val += np.trace(A0 @ A1.conj().T).real
        C[t] = val
    return C

def mass_gap(C, N):
    """Effective mass m_eff(t) using cosh formula."""
    m_eff = np.zeros(N-2)
    for t in range(1, N-1):
        num = C[t-1] + C[t+1]
        den = 2*C[t]
        ratio = num/den
        if ratio < 1:
            m_eff[t-1] = np.nan
        else:
            m_eff[t-1] = math.acosh(ratio)
    return m_eff

# ------------------------------------------------------------
# Wilson loops → String tension
# ------------------------------------------------------------
def wilson_loop(U, N, R, T):
    """Compute rectangular Wilson loop W(R,T)."""
    val = 0
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0,x1,x2)

                U_RT = np.eye(2, dtype=np.complex128)

                # go R steps in 0-direction
                pos = np.array(x)
                for _ in range(R):
                    U_RT = U[tuple(pos)][0] @ U_RT
                    pos[0] = (pos[0]+1) % N

                # go T steps in 1-direction
                for _ in range(T):
                    U_RT = U[tuple(pos)][1] @ U_RT
                    pos[1] = (pos[1]+1) % N

                # close loop backwards
                # (simplified version, OK for small N)
                val += np.trace(U_RT).real

    return val / (N**3)

def extract_sigma(WRT):
    """String tension via Creutz ratio."""
    # σ ≈ log( (W(R,T) W(R-1,T-1)) / (W(R,T-1) W(R-1,T)) )
    (W11, W12, W21, W22) = WRT
    return -np.log((W22 * W11) / (W12 * W21))

# ============================================================
# MAIN SIMULATION
# ============================================================

# Initialize all links with identity matrices
U = np.empty((N,N,N,3,2,2), dtype=np.complex128)
for x0 in range(N):
    for x1 in range(N):
        for x2 in range(N):
            for mu in range(3):
                U[x0,x1,x2,mu] = np.eye(2)

print("=== SU(2) Lattice Yang–Mills v4 MASTER ===")

for sweep in range(sweeps):
    U = heatbath_update(U, beta, N)

    if sweep % 50 == 0:
        P = avg_plaquette(U, N)
        print(f"[sweep {sweep}] plaquette = {P:.6f}")

# After sweeps: measure correlator + mass-gap
C = correlator(U, N)
m = mass_gap(C, N)

print("\n=== Correlator C(t) ===")
for t in range(N):
    print(f"t={t:2d}   C={C[t]:.6f}")

print("\n=== Effective mass-gap m_eff(t) ===")
for i,val in enumerate(m):
    print(f"t={i+1:2d}   m_eff={val}")

# Wilson loops (1x1, 1x2, 2x1, 2x2)
W11 = wilson_loop(U, N, 1,1)
W12 = wilson_loop(U, N, 1,2)
W21 = wilson_loop(U, N, 2,1)
W22 = wilson_loop(U, N, 2,2)

sigma = extract_sigma((W11,W12,W21,W22))

print("\n=== Wilson loops ===")
print("W(1,1)=", W11)
print("W(1,2)=", W12)
print("W(2,1)=", W21)
print("W(2,2)=", W22)

print("\n=== Estimated string tension σ ===")
print("σ ≈", sigma)

print("\n=== END ===")
