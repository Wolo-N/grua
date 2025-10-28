import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import os
from pathlib import Path

'''
GRUITA 3 FAST - SELF-OPTIMIZING CRANE (Simplified)
===================================================
The crane designs itself by optimizing:
1. Number of segments (8-20)
2. Boom height (0.5-2.0m)
3. Connectivity pattern (diagonal type, vertical spacing, support cables)
4. Member thicknesses BY TYPE (5 groups: top_struts, bot_struts, verticals, diagonals, supports)

Much faster than full individual optimization!
'''

# Global variables for iteration tracking
iteration_counter = 0
best_design_history = []
output_dir = Path("gruita3_iterations")

def element_stiffness(n1, n2, A, E):
    '''Returns Rod Elemental Stiffness Matrix in global coordinates'''
    d = n2 - n1
    L = np.linalg.norm(d)

    if L < 1e-10:
        return np.zeros((4, 4), dtype=float)

    c = d[0] / L
    s = d[1] / L

    k_local = (A * E / L) * np.array([[ 1, -1],
                                      [-1,  1]], dtype=float)

    T = np.array([[ c, s, 0, 0],
                  [ 0, 0, c, s]], dtype=float)

    k_structural = np.matmul(T.T, np.matmul(k_local, T))

    return k_structural

def dof_el(nnod1, nnod2):
    '''Returns Elemental DOF for Assembly'''
    return [2*(nnod1+1)-2,2*(nnod1+1)-1,2*(nnod2+1)-2,2*(nnod2+1)-1]

def hollow_circular_section(d_outer, d_inner):
    '''Calculate cross-sectional area and moment of inertia for hollow circular section'''
    r_outer = d_outer / 2
    r_inner = d_inner / 2

    A = np.pi * (r_outer**2 - r_inner**2)
    I = np.pi / 4 * (r_outer**4 - r_inner**4)

    return A, I

def design_parametric_crane(n_segments, boom_height, diag_type, vertical_spacing, num_support_cables):
    '''
    Design crane geometry with variable parameters

    Returns:
    --------
    X, C, element_types, tower_base, boom_top_nodes, boom_bot_nodes
    '''
    crane_length = 30.0
    tower_height = 0.0

    boom_x = np.linspace(0, crane_length, n_segments + 1)
    boom_y_top = np.linspace(tower_height + boom_height, tower_height, n_segments + 1)
    boom_y_bot = np.full(n_segments + 1, tower_height)

    total_nodes = 1 + 2 * (n_segments + 1)
    X = np.zeros([total_nodes, 2], float)

    node_idx = 0
    X[0] = [0, tower_height + 5]
    tower_base = 0
    node_idx += 1

    boom_top_nodes = []
    for i in range(n_segments + 1):
        X[node_idx] = [boom_x[i], boom_y_top[i]]
        boom_top_nodes.append(node_idx)
        node_idx += 1

    boom_bot_nodes = []
    for i in range(n_segments + 1):
        X[node_idx] = [boom_x[i], boom_y_bot[i]]
        boom_bot_nodes.append(node_idx)
        node_idx += 1

    elements = []
    element_types = []

    # Top struts
    for i in range(len(boom_top_nodes) - 1):
        elements.append([boom_top_nodes[i], boom_top_nodes[i+1]])
        element_types.append('top_strut')

    # Bottom struts
    for i in range(len(boom_bot_nodes) - 1):
        elements.append([boom_bot_nodes[i], boom_bot_nodes[i+1]])
        element_types.append('bot_strut')

    # Verticals
    v_spacing = max(1, min(int(vertical_spacing), n_segments))
    for i in range(0, len(boom_top_nodes), v_spacing):
        elements.append([boom_top_nodes[i], boom_bot_nodes[i]])
        element_types.append('vertical')

    # Diagonals
    diag_t = int(diag_type)
    if diag_t == 0:  # Alternating
        for i in range(len(boom_top_nodes) - 1):
            if i % 2 == 0:
                elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
            else:
                elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
            element_types.append('diagonal')
    elif diag_t == 1:  # All positive
        for i in range(len(boom_top_nodes) - 1):
            elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
            element_types.append('diagonal')
    elif diag_t == 2:  # All negative
        for i in range(len(boom_top_nodes) - 1):
            elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
            element_types.append('diagonal')
    elif diag_t == 3:  # Double (X-pattern)
        for i in range(len(boom_top_nodes) - 1):
            elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
            elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
            element_types.append('diagonal')
            element_types.append('diagonal')

    # Support cables
    n_supports = max(1, min(int(num_support_cables), len(boom_top_nodes) - 1))
    support_indices = np.linspace(len(boom_top_nodes)//2, len(boom_top_nodes)-2, n_supports, dtype=int)

    for idx in support_indices:
        elements.append([tower_base, boom_top_nodes[idx]])
        element_types.append('support')

    return X, np.array(elements, dtype=int), element_types, tower_base, boom_top_nodes, boom_bot_nodes

def save_iteration_graph(design_vector, objective, iteration_num):
    '''Save a graph of the current design without displaying it'''
    global output_dir

    try:
        n_segments = int(round(design_vector[0]))
        boom_height = design_vector[1]
        diag_type = design_vector[2]
        vertical_spacing = design_vector[3]
        num_support_cables = design_vector[4]

        X, C, element_types, _, _, _ = design_parametric_crane(
            n_segments, boom_height, diag_type, vertical_spacing, num_support_cables)

        thickness_dict = {
            'top_strut': (design_vector[5], design_vector[6]),
            'bot_strut': (design_vector[7], design_vector[8]),
            'vertical': (design_vector[9], design_vector[10]),
            'diagonal': (design_vector[11], design_vector[12]),
            'support': (design_vector[13], design_vector[14])
        }

        colors = {'top_strut': 'blue', 'bot_strut': 'green', 'vertical': 'purple',
                  'diagonal': 'orange', 'support': 'red'}

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        for iEl in range(C.shape[0]):
            etype = element_types[iEl]
            d_outer = thickness_dict[etype][0] * 1000
            lw = max(1, min(5, d_outer / 10))  # Cap linewidth

            ax.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                   [X[C[iEl,0],1], X[C[iEl,1],1]],
                   color=colors.get(etype, 'gray'), linewidth=lw, alpha=0.7)

        ax.scatter(X[:,0], X[:,1], c='black', s=30, zorder=5)
        ax.set_xlabel('x (m)', fontsize=10)
        ax.set_ylabel('y (m)', fontsize=10)
        ax.set_title(f'Iteration {iteration_num} - Objective: {objective:.2f}\n' +
                    f'{n_segments} segments, h={boom_height:.2f}m, {C.shape[0]} elements',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend(handles=[plt.Line2D([0], [0], color=c, lw=2, label=t)
                          for t, c in colors.items()], loc='upper right', fontsize=8)

        plt.tight_layout()
        filename = output_dir / f"iteration_{iteration_num:04d}_obj_{objective:.2f}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        # If graph fails, don't crash optimization
        pass

def evaluate_crane_design(design_vector, max_load=40000, verbose=False, save_graph=False):
    '''
    Design vector: [n_segments, boom_height, diag_type, vertical_spacing, num_support_cables,
                    d_outer_top, d_inner_top, d_outer_bot, d_inner_bot,
                    d_outer_vert, d_inner_vert, d_outer_diag, d_inner_diag,
                    d_outer_supp, d_inner_supp]
    Total: 15 variables (much faster!)
    '''
    global iteration_counter, best_design_history

    n_segments = int(round(design_vector[0]))
    boom_height = design_vector[1]
    diag_type = design_vector[2]
    vertical_spacing = design_vector[3]
    num_support_cables = design_vector[4]

    # Member thicknesses by type
    thickness_dict = {
        'top_strut': (design_vector[5], design_vector[6]),
        'bot_strut': (design_vector[7], design_vector[8]),
        'vertical': (design_vector[9], design_vector[10]),
        'diagonal': (design_vector[11], design_vector[12]),
        'support': (design_vector[13], design_vector[14])
    }

    try:
        X, C, element_types, tower_base, boom_top_nodes, boom_bot_nodes = \
            design_parametric_crane(n_segments, boom_height, diag_type, vertical_spacing, num_support_cables)
    except:
        return 1e9

    n_elements = C.shape[0]
    n_nodes = X.shape[0]

    E = 200e9
    rho_steel = 7850
    g = 9.81

    element_areas = []
    element_inertias = []
    total_mass = 0

    for iEl in range(n_elements):
        etype = element_types[iEl]
        d_outer, d_inner = thickness_dict[etype]

        # Ensure valid geometry
        min_wall = 0.002
        if d_inner >= d_outer - min_wall:
            d_inner = d_outer - min_wall
        if d_inner < 0.001:
            d_inner = 0.001

        A, I = hollow_circular_section(d_outer, d_inner)
        element_areas.append(A)
        element_inertias.append(I)

        n1, n2 = C[iEl]
        L = np.linalg.norm(X[n2] - X[n1])
        volume = A * L
        mass = rho_steel * volume
        total_mass += mass

    # Boundary conditions
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True
    bc[tower_base, :] = True
    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Assemble stiffness
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)

    for iEl in range(n_elements):
        A = element_areas[iEl]
        dof = dof_el(C[iEl,0], C[iEl,1])
        k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    try:
        cond = np.linalg.cond(k_reduced)
        if cond > 1e15:
            return 1e8
    except:
        return 1e8

    # Loads
    loads = np.zeros([n_nodes, 2], float)
    loads[boom_top_nodes[-1], 1] = -max_load

    # Self-weight
    for iEl in range(n_elements):
        n1, n2 = C[iEl]
        L = np.linalg.norm(X[n2] - X[n1])
        A = element_areas[iEl]
        weight = rho_steel * A * L * g
        loads[n1, 1] -= weight / 2
        loads[n2, 1] -= weight / 2

    load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

    try:
        displacements = np.zeros([2*n_nodes], float)
        displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
        D = displacements.reshape(n_nodes, 2)
    except:
        return 1e8

    # Calculate forces and stresses
    element_forces = []
    element_stresses = []

    for iEl in range(n_elements):
        n1, n2 = C[iEl]
        d_el = np.concatenate([D[n1], D[n2]])
        A = element_areas[iEl]

        k_el = element_stiffness(X[n1], X[n2], A, E)
        f_el = k_el @ d_el

        d_vec = X[n2] - X[n1]
        L = np.linalg.norm(d_vec)

        if L < 1e-10:
            element_forces.append(0)
            element_stresses.append(0)
            continue

        u_vec = d_vec / L
        du = np.array([d_el[2] - d_el[0], d_el[3] - d_el[1]])
        epsilon = np.dot(du, u_vec) / L
        stress = E * epsilon
        force = stress * A

        element_forces.append(force)
        element_stresses.append(stress)

    element_forces = np.array(element_forces)
    element_stresses = np.array(element_stresses)

    # Safety factors
    sigma_max = np.max(np.abs(element_stresses))
    sigma_adm = 100e6

    FS_tension = sigma_adm / (sigma_max + 1e-6)

    # Buckling
    P_max_compression = abs(np.min(element_forces))
    P_critica = 1e12

    for iEl in range(n_elements):
        if element_forces[iEl] < -1:
            n1, n2 = C[iEl]
            L = np.linalg.norm(X[n2] - X[n1])
            I = element_inertias[iEl]
            P_cr = (np.pi**2 * E * I) / (L**2)
            if P_cr < P_critica:
                P_critica = P_cr

    FS_pandeo = P_critica / (P_max_compression + 1e-6)

    # Cost
    m0 = 1000
    n_elementos_0 = 50
    n_uniones_0 = 25

    cost = (total_mass / m0) + 1.5 * (n_elements / n_elementos_0) + 2 * (n_nodes / n_uniones_0)

    # Penalties
    penalty = 0
    min_FS = 2.0

    if FS_tension < min_FS:
        penalty += 1000 * (min_FS - FS_tension)**2

    if FS_pandeo < min_FS:
        penalty += 1000 * (min_FS - FS_pandeo)**2

    tip_deflection = abs(D[boom_top_nodes[-1], 1])
    max_deflection = 0.200
    if tip_deflection > max_deflection:
        penalty += 500 * (tip_deflection - max_deflection)**2

    objective = cost + penalty

    if verbose:
        print(f"n_seg={n_segments}, h={boom_height:.2f}m, diag={int(diag_type)}, n_el={n_elements}")
        print(f"  Mass: {total_mass:.1f}kg, Cost: {cost:.2f}, Penalty: {penalty:.2f}")
        print(f"  FS_tension: {FS_tension:.2f}, FS_pandeo: {FS_pandeo:.2f}")
        print(f"  Deflection: {tip_deflection*1000:.1f}mm")
        print(f"  Objective: {objective:.2f}")

    return objective

def optimize_crane():
    print("="*80)
    print("SELF-OPTIMIZING CRANE (GRUITA 3 FAST)")
    print("="*80)
    print("\nOptimizes: segments, boom height, connectivity, 5 thickness groups")
    print("Variables: 15 (much faster than individual member optimization!)")
    print("="*80)

    # Bounds: [n_seg, height, diag, v_space, n_supp, + 5 pairs of (d_outer, d_inner)]
    bounds = [
        (8, 20),      # n_segments
        (0.5, 2.0),   # boom_height
        (0, 3),       # diag_type
        (1, 3),       # vertical_spacing
        (1, 3),       # num_support_cables
        (0.020, 0.050), (0.010, 0.045),  # top_strut (outer, inner)
        (0.020, 0.050), (0.010, 0.045),  # bot_strut
        (0.015, 0.045), (0.010, 0.040),  # vertical
        (0.010, 0.040), (0.005, 0.035),  # diagonal
        (0.010, 0.040), (0.005, 0.035),  # support
    ]

    # Initial guess - make sure wall thickness is reasonable
    x0 = [12, 1.0, 0, 1, 2,
          0.045, 0.035,  # top (10mm wall)
          0.045, 0.035,  # bot (10mm wall)
          0.035, 0.025,  # vert (10mm wall)
          0.025, 0.018,  # diag (7mm wall)
          0.030, 0.022]  # support (8mm wall)

    print("\nEvaluating initial design...")
    initial_obj = evaluate_crane_design(np.array(x0), verbose=True)

    if initial_obj > 1e7:
        print("\nWARNING: Initial design has issues. Adjusting...")
        x0 = [12, 1.2, 0, 1, 2,
              0.048, 0.038,
              0.048, 0.038,
              0.038, 0.028,
              0.028, 0.020,
              0.032, 0.024]
        initial_obj = evaluate_crane_design(np.array(x0), verbose=True)

    print("\nStarting optimization...")
    result = differential_evolution(
        evaluate_crane_design,
        bounds,
        init='latinhypercube',  # Better initial population
        strategy='best1bin',
        maxiter=80,
        popsize=12,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        workers=1,
        disp=True,
        polish=False,
        atol=0.01,
        tol=0.01
    )

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    print("\nOptimal design:")
    best_obj = evaluate_crane_design(result.x, verbose=True)

    # Visualize
    n_segments = int(round(result.x[0]))
    boom_height = result.x[1]
    X, C, types, _, _, _ = design_parametric_crane(
        n_segments, boom_height, result.x[2], result.x[3], result.x[4])

    plot_crane(X, C, types, result.x[5:], n_segments, boom_height)

    return result

def plot_crane(X, C, element_types, thickness_params, n_segments, boom_height):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    thickness_dict = {
        'top_strut': (thickness_params[0], thickness_params[1]),
        'bot_strut': (thickness_params[2], thickness_params[3]),
        'vertical': (thickness_params[4], thickness_params[5]),
        'diagonal': (thickness_params[6], thickness_params[7]),
        'support': (thickness_params[8], thickness_params[9])
    }

    colors = {'top_strut': 'blue', 'bot_strut': 'green', 'vertical': 'purple',
              'diagonal': 'orange', 'support': 'red'}

    for iEl in range(C.shape[0]):
        etype = element_types[iEl]
        d_outer = thickness_dict[etype][0] * 1000
        lw = max(1, d_outer / 10)

        ax1.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                [X[C[iEl,0],1], X[C[iEl,1],1]],
                color=colors[etype], linewidth=lw, alpha=0.7)

    ax1.scatter(X[:,0], X[:,1], c='black', s=30, zorder=5)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title(f'Optimized Crane\n{n_segments} segments, h={boom_height:.2f}m')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend(handles=[plt.Line2D([0], [0], color=c, lw=2, label=t)
                       for t, c in colors.items()])

    # Thickness comparison
    types_list = ['top_strut', 'bot_strut', 'vertical', 'diagonal', 'support']
    d_outer = [thickness_dict[t][0]*1000 for t in types_list]
    d_inner = [thickness_dict[t][1]*1000 for t in types_list]
    wall = [d_outer[i] - d_inner[i] for i in range(5)]

    x = np.arange(5)
    width = 0.35

    ax2.bar(x - width/2, d_outer, width, label='Outer diameter', alpha=0.7)
    ax2.bar(x + width/2, wall, width, label='Wall thickness', alpha=0.7)
    ax2.axhline(y=50, color='r', linestyle='--', label='Max outer (50mm)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(types_list, rotation=45, ha='right')
    ax2.set_ylabel('Dimension (mm)')
    ax2.set_title('Member Thickness by Type')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print("\n" + "="*80)
    print("THICKNESS SUMMARY")
    print("="*80)
    for t in types_list:
        d_o, d_i = thickness_dict[t]
        wall_t = (d_o - d_i) * 1000
        print(f"{t:12s}: D_outer={d_o*1000:5.1f}mm, D_inner={d_i*1000:5.1f}mm, wall={wall_t:5.1f}mm")
    print("="*80)

if __name__ == '__main__':
    result = optimize_crane()
