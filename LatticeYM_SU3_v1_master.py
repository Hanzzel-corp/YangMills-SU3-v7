import numpy as np

# ============================
#  SU(3) LATTICE — MASTER v1
#  Stable + modular + scalable
# ============================

np.random.seed(7)

# ----------------------------
# 1) Rutinas básicas SU(3)
# ----------------------------

def random_su3():
    """
    Genera una matriz SU(3) usando el método de Ginibre + QR normalizado.
    Esto es más liviano que un heatbath real.
    """
    Z = np.random.normal(size=(3,3)) + 1j*np.random.normal(size=(3,3))
    Q, R = np.linalg.qr(Z)
    diag = np.diag(R)
    Q *= diag/np.abs(diag)
    return Q

def su2_embed(a, i, j):
    """
    Inserta una submatriz SU(2) en SU(3).
    Esto permite Cabibbo–Marinari.
    """
    U = np.eye(3, dtype=np.complex128)
    U[i,i] = a[0] + 1j*a[3]
    U[j,j] = a[0] - 1j*a[3]
    U[i,j] = a[1] + 1j*a[2]
    U[j,i] = -a[1] + 1j*a[2]
    return U

def random_su2():
    """
    Genera una SU(2) como 4 números normalizados.
    """
    a = np.random.normal(size=4)
    return a / np.linalg.norm(a)

# ----------------------------
# 2) Acción de Wilson SU(3)
# ----------------------------

def staple(U, x, mu, N):
    """
    Calcula la 'staple' alrededor de un link.
    """
    d = 4
    S = np.zeros((3,3), dtype=np.complex128)
    for nu in range(d):
        if nu == mu:
            continue
        x_fwd = tuple((x[k] + (1 if k == nu else 0)) % N for k in range(d))
        x_bwd = tuple((x[k] - (1 if k == nu else 0)) % N for k in range(d))

        # Plaqueta hacia adelante
        S += U[x][nu] @ U[x_fwd][mu] @ U[x_bwd][nu].conj().T

        # Plaqueta hacia atrás
        S += U[x_bwd][nu].conj().T @ U[x_bwd][mu] @ U[x_bwd][nu]
    return S

# ----------------------------
# 3) Update Metropolis + CM
# ----------------------------

def update_link(U, x, mu, beta, N):
    S = staple(U, x, mu, N)
    U_old = U[x][mu]

    # Cabibbo–Marinari: 3 subgrupos SU(2)
    for (i, j) in [(0,1), (0,2), (1,2)]:
        a = random_su2()
        R = su2_embed(a, i, j)
        U_new = R @ U_old

        dS = -beta * (np.real(np.trace((U_new - U_old) @ S.conj().T)))
        if dS < 0 or np.random.rand() < np.exp(-dS):
            U_old = U_new

    U[x][mu] = U_old

# ----------------------------
# 4) Observables
# ----------------------------

def plaquette(U, N):
    d = 4
    P = 0
    count = 0
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                for x3 in range(N):
                    x = (x0, x1, x2, x3)
                    for mu in range(d):
                        for nu in range(mu+1, d):
                            x_fwd_mu = tuple((x[k] + (1 if k == mu else 0)) % N for k in range(d))
                            x_fwd_nu = tuple((x[k] + (1 if k == nu else 0)) % N for k in range(d))
                            U_plaq = (
                                U[x][mu] @ U[x_fwd_mu][nu] @
                                U[x_fwd_nu][mu].conj().T @ U[x][nu].conj().T
                            )
                            P += np.real(np.trace(U_plaq))/3
                            count += 1
    return P / count

def correlator(U, N, t):
    """
    Operador de glueball muy simple: tr(U_mu(x)).
    """
    d = 4
    vals = []
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0, x1, x2, 0)
                y = (x0, x1, x2, t % N)
                O1 = np.real(np.trace(U[x][0]))
                O2 = np.real(np.trace(U[y][0]))
                vals.append(O1 * O2)
    return np.mean(vals)

def wilson_loop(U, R, T, N):
    """
    Loop rectangular R×T para medir confinamiento.
    """
    vals = []
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                for x3 in range(N):
                    x = (x0, x1, x2, x3)

                    U_R = np.eye(3, dtype=np.complex128)
                    pos = x
                    for _ in range(R):
                        U_R = U_R @ U[pos][0]
                        pos = ((pos[0]+1)%N, pos[1], pos[2], pos[3])

                    U_T = np.eye(3, dtype=np.complex128)
                    for _ in range(T):
                        U_T = U_T @ U[pos][3]
                        pos = (pos[0], pos[1], pos[2], (pos[3]+1)%N)

                    # cierre (simplificado)
                    W = np.trace(U_R @ U_T) / 3.0
                    vals.append(np.real(W))
    return np.mean(vals)


# ----------------------------
# MAIN RUN
# ----------------------------

def run():
    N = 6          # Lattice liviano para hardware normal
    sweeps = 200   # Podés subirlo si querés
    beta = 5.5     # Típico SU(3) fuerte–confinado

    # Inicializar U[x][mu] en SU(3)
    U = {}
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                for x3 in range(N):
                    x = (x0,x1,x2,x3)
                    U[x] = [random_su3() for _ in range(4)]

    print("=== SU(3) Lattice Yang–Mills v1 MASTER ===")

    # SWEPS
    for s in range(sweeps):
        for x0 in range(N):
            for x1 in range(N):
                for x2 in range(N):
                    for x3 in range(N):
                        x = (x0,x1,x2,x3)
                        for mu in range(4):
                            update_link(U, x, mu, beta, N)

        if s % 20 == 0:
            P = plaquette(U, N)
            print(f"[sweep {s}] plaquette = {P}")

    print("\n=== Correlator C(t) ===")
    for t in range(1,6):
        C = correlator(U, N, t)
        print(f"t={t}   C={C}")

    print("\n=== Wilson loops ===")
    for R in [1,2]:
        for T in [1,2]:
            W = wilson_loop(U, R, T, N)
            print(f"W({R},{T})={W}")

    print("\n=== END ===")

if __name__ == "__main__":
    run()
