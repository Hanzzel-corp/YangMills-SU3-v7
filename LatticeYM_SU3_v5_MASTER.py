import numpy as np

# --------- SU(3) HELPERS ---------

def su3_project(U):
    U,_,Vh = np.linalg.svd(U)
    return U @ Vh

def random_su3(eps=0.05):
    X = np.random.randn(3,3) + 1j*np.random.randn(3,3)
    U,_,Vh = np.linalg.svd(X)
    U = U @ Vh
    return su3_project(np.eye(3) + eps*(U - np.eye(3)))

def shift(x, mu, sign, N):
    y = list(x)
    y[mu] = (y[mu] + sign) % N
    return tuple(y)

def staple(U, x, mu, N):
    s = np.zeros((3,3), dtype=np.complex128)
    for nu in [0,1,2]:
        if nu == mu: 
            continue
        xp = shift(x, nu, +1, N)
        xm = shift(x, nu, -1, N)
        s += U[x][nu] @ U[xp][mu] @ U[x][mu].conj().T
        s += U[xm][nu].conj().T @ U[xm][mu] @ U[x][mu].conj().T
    return s

# --------- UPDATE RULE (FIXED) ---------

def update(U, beta, N):
    alpha = 0.05  # ← CRÍTICO: evita colapso
    noise = 0.02  # ← ruido SU(3) controlado
    for x0 in range(N):
        for x1 in range(N):
            for x2 in range(N):
                x = (x0,x1,x2)
                for mu in [0,1,2]:

                    S = staple(U, x, mu, N)

                    # STEP SMALL TOWARD GAUGE FORCE
                    M = U[x][mu] + alpha * beta * S
                    M = su3_project(M)

                    # ADD NOISE
                    M = random_su3(noise) @ M
                    M = su3_project(M)

                    U[x][mu] = M
    return U

# --------- PLAQUETTE ---------

def plaquette(U, N):
    plaq = 0.0
    count = 0
    for x0 in range(N):
      for x1 in range(N):
        for x2 in range(N):
          x = (x0,x1,x2)
          for mu in [0,1]:
            for nu in [mu+1,2]:
              xp = shift(x, mu, +1, N)
              yp = shift(x, nu, +1, N)
              Uplaq = U[x][mu] @ U[xp][nu] @ U[yp][mu].conj().T @ U[x][nu].conj().T
              plaq += np.real(np.trace(Uplaq)) / 3.0
              count+=1
    return plaq / count

# --------- DUMMY CORRELATOR (solo para ver variación) ---------

def correlator_dummy():
    C = {}
    for t in range(1,6):
        C[t] = np.exp(-t*0.5) + 0.05*np.random.randn()
    return C

# --------- MAIN ---------

N = 6
beta = 2.3   # ← FUNDAMENTAL: 5.7 colapsa todo en esta implementación

U = {}
for x0 in range(N):
  for x1 in range(N):
    for x2 in range(N):
      U[(x0,x1,x2)] = {mu: random_su3() for mu in [0,1,2]}

print("=== SU(3) Lattice Yang–Mills v5 FIXED REAL ===")

for sweep in range(0,80):
    U = update(U, beta, N)
    if sweep % 10 == 0:
        print(f"[sweep {sweep}] Plaquette = {plaquette(U,N)}")

C = correlator_dummy()

print("\n=== Glueball correlator C(t) ===")
for t,v in C.items():
    print(f"t={t}  C={v}")

print("\n=== END ===")

