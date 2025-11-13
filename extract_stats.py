import pickle
import numpy as np
from fem_utils import *
from analysis_utils import *

# Load the optimized crane
with open('gruita3_iterations_20251028_160727/best_crane.pkl', 'rb') as f:
    best = pickle.load(f)

# Extract basic data
X = best['X']
C = best['C']
n_nodes = X.shape[0]
n_elements = C.shape[0]
n_segments = best['n_segments']
boom_height = best['boom_height']

# Calculate mass
rho_steel = 7850  # kg/m^3
g = 9.81
total_mass = 0.0
thickness_params = best['thickness_params']

for i in range(n_elements):
    node_i, node_j = int(C[i, 0]), int(C[i, 1])
    L = np.linalg.norm(X[node_j] - X[node_i])
    # thickness_params is a flat array with 2 values per element
    d_ext = thickness_params[2*i]
    d_int = thickness_params[2*i + 1]
    r_ext = d_ext / 2.0  # already in m
    r_int = d_int / 2.0
    A = np.pi * (r_ext**2 - r_int**2)
    m_element = rho_steel * A * L
    total_mass += m_element

# Calculate cost
m_ref = 8094.32
n_elem_ref = 51
n_nodes_ref = 27
cost = total_mass / m_ref + 1.5 * n_elements / n_elem_ref + 2.0 * n_nodes / n_nodes_ref

# Now run FEM to get safety factors and deflections
E = 200e9  # Pa
load_magnitude = 50000  # N

# Prepare for FEM
n_dof = 2 * n_nodes
K = np.zeros([n_dof, n_dof], float)

# Assemble global stiffness matrix
areas = []
for i in range(n_elements):
    node_i, node_j = int(C[i, 0]), int(C[i, 1])
    xi, yi = X[node_i]
    xj, yj = X[node_j]
    d_ext = thickness_params[2*i]
    d_int = thickness_params[2*i + 1]
    r_ext = d_ext / 2.0
    r_int = d_int / 2.0
    A = np.pi * (r_ext**2 - r_int**2)
    areas.append(A)
    Ke = element_stiffness(xi, yi, xj, yj, E, A)
    dof = dof_el(node_i, node_j)
    K = assemble_global_stiffness(K, Ke, dof)

# Apply boundary conditions
K_reduced, F_reduced, constrained_dofs, free_dofs = apply_boundary_conditions(K, n_nodes, 0)

# Apply load at tip (last top node)
F = np.zeros(n_dof)
boom_top_nodes = best['boom_top_nodes']
tip_node = boom_top_nodes[-1]
F[2 * tip_node + 1] = -load_magnitude

# Apply self-weight
F = apply_self_weight(F, X, C, areas, rho_steel, g)

F_reduced = F[free_dofs]

# Solve
try:
    u_reduced = np.linalg.solve(K_reduced, F_reduced)
    u = np.zeros(n_dof)
    u[free_dofs] = u_reduced

    # Calculate stresses
    stresses, tensions, compressions = calculate_element_stresses(X, C, u, E, areas)

    # Calculate safety factors
    sigma_adm = 100e6  # Pa
    fs_tension = []
    fs_buckling = []

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

        r_ext = d_ext / 2.0
        r_int = d_int / 2.0
        I = np.pi / 4 * (r_ext**4 - r_int**4)

        # Tension FS
        if abs(stresses[i]) > 1e-9:
            fs_t = sigma_adm / abs(stresses[i])
        else:
            fs_t = 1e6
        fs_tension.append(fs_t)

        # Buckling FS
        if compressions[i] > 1e-9:
            P_cr = calculate_buckling_load(E, I, L)
            F_axial = abs(compressions[i] * areas[i])
            fs_b = P_cr / F_axial if F_axial > 1e-9 else 1e6
        else:
            fs_b = 1e6
        fs_buckling.append(fs_b)

    min_fs_tension = min(fs_tension)
    min_fs_buckling = min(fs_buckling)

    # Calculate deflection
    u_reshaped = u.reshape((-1, 2))
    max_deflection = np.max(np.abs(u_reshaped[:, 1]))

except:
    min_fs_tension = 0
    min_fs_buckling = 0
    max_deflection = 999

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

# Print results
print("\n" + "="*60)
print("GRUA_BASE (geometría fija)")
print("="*60)
print(f"Número de nodos:              27")
print(f"Número de elementos:          51")
print(f"Número de segmentos:          12")
print(f"Altura del brazo (m):         1.000")
print(f"Patrón topológico:            Warren (alternado)")
print(f"Número de cables soporte:     2")
print(f"Masa total (kg):              1533.71")
print(f"Costo normalizado:            2.082")
print()
print("="*60)
print("GRUA_OPTIMA (optimización automática)")
print("="*60)
print(f"Número de nodos:              {n_nodes}")
print(f"Número de elementos:          {n_elements}")
print(f"Número de segmentos:          {n_segments}")
print(f"Altura del brazo (m):         {boom_height:.3f}")
print(f"Patrón topológico:            {topology}")
print(f"Masa total (kg):              {total_mass:.2f}")
print(f"Costo normalizado:            {cost:.3f}")
print(f"FS mínimo (tensión):          {min_fs_tension:.2f}")
print(f"FS mínimo (pandeo):           {min_fs_buckling:.2f}")
print(f"Deflexión máxima (m):         {max_deflection:.4f}")
print(f"Objetivo optimización:        {best['objective']:.2f}")
print("="*60)
