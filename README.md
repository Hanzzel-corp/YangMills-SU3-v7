📘 README.md — versión final para subir

Copiá y pegá esto como README.md dentro de tu carpeta:

🌀 Exploración de Regímenes Críticos en Navier–Stokes 3D
Simulaciones V2 → V10 desde una notebook, buscando señales de singularidad

Este repositorio documenta un recorrido experimental construyendo, desde cero, una serie de motores numéricos destinados a estudiar el crecimiento de la vorticidad en la ecuación de Navier–Stokes 3D, utilizando únicamente una notebook y Python.

El objetivo no es “resolver el problema del milenio”, sino construir un marco reproducible que permita observar:

crecimiento superlineal de vorticidad,

regiones de estiramiento intenso,

patrones críticos tipo Hou–Luo,

y comportamientos precursores estudiados en el criterio Beale–Kato–Majda.

📂 Estructura del repositorio

El repositorio contiene una colección evolutiva de motores numéricos:

Archivo	Descripción
Navier-Phase_v2.py	Primer flujo base, sin forcing complejo
Navier-Phase_v3.py	Ajustes y control de estabilidad
Navier-Phase_v4.py	Forcing simple + mejoras de gradientes
Navier-Phase_v5.py	Regímenes caóticos controlados
Navier-Phase_v6.py	Motor con forcing fuerte y respuesta explosiva
Navier-Phase_v7.py	Rescaling dinámico + detección de filamentos
Navier-Phase_v7_bkm.py	Versión orientada al criterio BKM
Navier-Phase_v8.py	Motor simétrico con estiramiento controlado
Navier-Phase_v9.py	Forcing matemático explosivo (pre-blow up)
NavierPhase_HouLuo_torus3D_v10.py	Primer motor estilo Hou–Luo con toro 3D

Todos los modelos funcionan con:

FFT espectrales en las 3 dimensiones

Proyector de Leray

Dealiasing 2/3

Gradientes espectrales

Integración RK4

🧪 Resultados principales

La última versión (V10–V11 experimental) mostró:

vorticidades máximas superiores a 350

crecimiento sostenido durante miles de iteraciones

oscilaciones críticas tipo plateau dinámico

comportamiento precursor de singularidad según interpretación Hou–Luo / BKM

sin blow-up real, pero con intensificación extrema del flujo

Esto representa el límite razonable para una notebook sin GPU y con malla 32³.

El código queda abierto para investigadores que quieran:

aumentar la resolución a 64³, 128³ o 256³

aplicar GPUs

utilizar doble precisión extendida

comparar con simulaciones de referencia

▶️ Cómo ejecutar

Instalar dependencias:

pip install numpy matplotlib


Ejemplo:

python NavierPhase_HouLuo_torus3D_v10.py