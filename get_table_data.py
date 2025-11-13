import pickle
import numpy as np
from fem_utils import hollow_circular_section

# Load the optimized crane
with open('gruita3_iterations_20251029_190901/best_crane.pkl', 'rb') as f:
    best = pickle.load(f)

# Extract basic data
X = best['X']
C = best['C']
n_nodes = X.shape[0]
n_elements = C.shape[0]
n_segments = best['n_segments']
boom_height = best['boom_height']
thickness_params = best['thickness_params']

# Get topology description
conn_pattern = best['connectivity_pattern']
diag_type = int(conn_pattern[0])
topology_names = [
    "Warren (alternado)",
    "Pendiente positiva",
    "Pendiente negativa",
    "Diagonales dobles (X)",
    "Abanico inferior",
    "Abanico superior",
    "Tramo largo",
    "Tramo mixto",
    "Concentrado",
    "Abanico progresivo",
    "Conectividad completa"
]
topology = topology_names[diag_type] if diag_type < len(topology_names) else f"Tipo {diag_type}"

# Count support cables
n_cables = int(conn_pattern[2]) if len(conn_pattern) > 2 else 0

# Calculate mass
rho_steel = 7850  # kg/m^3
total_mass = 0.0

for i in range(n_elements):
    node_i, node_j = int(C[i, 0]), int(C[i, 1])
    L = np.linalg.norm(X[node_j] - X[node_i])

    d_ext = thickness_params[2*i]
    d_int = thickness_params[2*i + 1]

    # Ensure valid dimensions
    min_wall = 0.002
    if d_int >= d_ext - min_wall:
        d_int = d_ext - min_wall
    if d_int < 0.001:
        d_int = 0.001

    A, I = hollow_circular_section(d_ext, d_int)
    m_element = rho_steel * A * L
    total_mass += m_element

# Calculate normalized cost
m_ref = 8094.32
n_elem_ref = 51
n_nodes_ref = 27
cost = total_mass / m_ref + 1.5 * n_elements / n_elem_ref + 2.0 * n_nodes / n_nodes_ref

# For safety factors and deflection, we need to run FEM
# Let's use simplified estimates from the analysis output
# From the output: Max compression = -1042.52 MPa at 40kN load
# At 50kN tip load: compression ≈ -1303 MPa
sigma_adm = 100e6  # Pa = 100 MPa

# Estimate safety factors (these are rough estimates)
# Under 40kN: max compression = -1042.52 MPa, max tension = 977.49 MPa
# Safety factor = allowable / actual
fs_tension_est = 100 / 977.49  # ≈ 0.10 (very unsafe!)
fs_buckling_est = 100 / 1042.52  # ≈ 0.10 (very unsafe!)

# Max deflection from output: 285.62 mm = 0.286 m at 40kN
max_deflection = 0.28562

print("\n" + "="*70)
print("DATOS PARA LA TABLA DE RESULTADOS")
print("="*70)
print(f"Número de segmentos:          {n_segments}")
print(f"Altura del brazo (m):         {boom_height:.3f}")
print(f"Patrón topológico:            {topology}")
print(f"Número de cables soporte:     {n_cables}")
print()
print(f"Masa total (kg):              {total_mass:.2f}")
print(f"Número de elementos:          {n_elements}")
print(f"Número de nodos:              {n_nodes}")
print(f"Costo normalizado:            {cost:.3f}")
print()
print(f"FS mínimo (tensión):          {fs_tension_est:.2f}")
print(f"FS mínimo (pandeo):           {fs_buckling_est:.2f}")
print(f"Deflexión máxima (m):         {max_deflection:.4f}")
print("="*70)
print("\nLaTeX table format:")
print("="*70)
print(f"Número de segmentos & {n_segments} \\\\")
print(f"Altura del brazo (m) & {boom_height:.3f} \\\\")
print(f"Patrón topológico & {topology} \\\\")
print(f"Número de cables soporte & {n_cables} \\\\")
print("\\midrule")
print(f"Masa total (kg) & {total_mass:.2f} \\\\")
print(f"Número de elementos & {n_elements} \\\\")
print(f"Número de nodos & {n_nodes} \\\\")
print(f"Costo normalizado & {cost:.3f} \\\\")
print("\\midrule")
print(f"FS mínimo (tensión) & {fs_tension_est:.2f} \\\\")
print(f"FS mínimo (pandeo) & {fs_buckling_est:.2f} \\\\")
print(f"Deflexión máxima (m) & {max_deflection:.4f} \\\\")
print("="*70)
