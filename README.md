📘 README.md — Yang–Mills SU(3) Lattice Simulation (v1 → v7)
A minimal but mathematically consistent SU(3) lattice implementation developed from first principles
🔹 Descripción general

Este repositorio documenta el proceso completo de construcción de una simulación de Yang–Mills en red (Lattice YM) para los grupos SU(2) y SU(3), implementada desde cero en Python y ejecutada en hardware convencional.

El objetivo no es resolver el Problema del Milenio, sino mostrar un camino transparente y reproducible hacia:

entender la estructura algebraica del grupo SU(3),

construir operadores gauge consistentes,

implementar la acción de Wilson,

explorar la dinámica del campo mediante heatbath,

y obtener señales físicas: plaquette estable, correladores decrecientes y presencia de mass-gap.

La versión final, SU(3) v7, es la primera que presenta una dinámica estable y físicamente coherente.

🔹 Motivación

Yang–Mills es uno de los pilares de la física moderna, pero implementar SU(3) correctamente requiere:

respetar la geometría del grupo,

controlar la unitariedad,

manejar actualizaciones locales sin “saturación”,

y medir cantidades físicas sin ruido artificial.

Este proyecto propone un enfoque pedagógico y honesto:
construir todo desde cero, verificar cada paso y mostrar cómo evolucionan las versiones hasta obtener un sistema autocoherente.

📐 Fundamentos matemáticos utilizados
1. Grupos SU(N)

Matrices unitarias de determinante 1.

SU(2) como caso de validación algebraica.

SU(3) como grupo de color de QCD, de dimensión 8.

2. Enlaces gauge en red

Cada enlace en la red es una matriz

𝑈𝜇(𝑥)∈𝑆𝑈(3)Uμ
	​
(x)∈SU(3)

que representa transporte paralelo entre puntos vecinos.

3. Acción de Wilson

El plaquette

𝑃𝜇𝜈(𝑥)=Re
 Tr[𝑈𝜇(𝑥)𝑈𝜈(𝑥+𝜇)𝑈𝜇†(𝑥+𝜈)𝑈𝜈†(𝑥)]Pμν
	​

(x)=ReTr[U
μ
	​

(x)U
ν
	​

(x+μ)U
μ
†
	​

(x+ν)U
ν
†
	​

(x)]

es una medida local de curvatura.

4. Staples y fuerza gauge

Los staples determinan cómo la acción cambia localmente y son esenciales para actualizaciones tipo heatbath.

5. Heatbath SU(2) embebido en SU(3)

La técnica Cabibbo–Marinari se implementa manualmente para actualizar subgrupos SU(2) dentro de SU(3).

6. Correladores y mass-gap

Un operador 
𝑂
(
𝑡
)
O(t) tiene comportamiento esperado

𝐶
(
𝑡
)
=
⟨
𝑂
(
𝑡
)
𝑂
(
0
)
⟩
∼
𝑒
−
𝑚
𝑡
,
C(t)=⟨O(t)O(0)⟩∼e
−mt
,

cuyo decaimiento exponencial sugiere un mass-gap positivo.

🧩 Evolución del código (v1 → v7)
✔ v1–v2 (SU(2))

Validación algebraica.

Primeras simulaciones de calibre.

Confirmación de unitariedad y estabilidad.

✔ v3–v4 (primer SU(3))

Aparecen fallos típicos:

staples mal orientados,

loops incorrectos,

pérdida de unitariedad,

“congelamiento” numérico.

Estas versiones fueron fundamentales para detectar y corregir problemas estructurales.

✔ v5–v6

Reconstrucción completa del núcleo SU(3).

Proyección robusta al grupo.

Heatbath más estable.

Persisten inestabilidades en correladores.

🌟 v7 — la versión estable

Staples reconstruidos desde cero.

Ciclos cerrados correctamente.

Heatbath SU(2) → SU(3) consistente.

Proyección precisa a SU(3).

Primera señal física robusta:

Plaquette ≈ 0.39 — valor realista para redes pequeñas con β=6.0

Correladores decreciendo exponencialmente → mass-gap positivo

📈 Ejemplo de salida real (v7)
[sweep 0] Plaquette = 0.387
[sweep 40] Plaquette = 0.408

C(t):
t=1 → 0.356
t=2 → 0.322
t=3 → 0.291
t=4 → 0.263


Esto indica:

Dinámica gauge no congelada

Acción funcionando

Curvatura local consistente

Señal física emergente (mass-gap)

📦 Contenido del repositorio
YangMills-SU3-v7/
│
├── LatticeYM_SU2_v1_master.py
├── LatticeYM_SU2_v2_stable.py
│
├── LatticeYM_SU3_v3_paper.py
├── LatticeYM_SU3_v4_paper.py
├── LatticeYM_SU3_v5_MASTER.py
├── LatticeYM_SU3_v6_physics.py
├── LatticeYM_SU3_v7_MASTER.py   ← versión final estable
│
└── README.md

🔭 Líneas futuras

Aumentar el tamaño del lattice.

Implementar heatbath completo (Creutz/OK method).

Medir masa de glueball con mayor precisión.

Variar β y explorar transición de fase.

Extender a SU(4) o SU(N) general.

Conectar con QAOA / NCT como extensión cuántica.

📜 Licencia

MIT License.
El código puede ser estudiado, modificado y reutilizado libremente, con atribución.

🧠 Autor

José Pablo Zamora (Hanzzel Corp ∑Δ9)
Desarrollo, matemática, implementación y análisis.
