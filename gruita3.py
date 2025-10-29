import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import os
from pathlib import Path
from datetime import datetime
import pickle
import time

'''
GRUITA 3 - SELF-OPTIMIZING CRANE
================================
The crane designs itself by optimizing:
1. Number of nodes along the 30m length
2. Connectivity pattern (which diagonals to include, vertical spacing)
3. Individual member thicknesses (each element optimized separately)

Objective: Minimize cost while maintaining safety factors >= 2.0

Saves a graph for each iteration without displaying it.
'''

# Global variables for iteration tracking
iteration_counter = 0
best_objective = float('inf')
best_design_vector = None  # Store the best design vector
last_save_time = None  # Track when we last saved
output_dir = None  # Will be set in optimize_crane() with timestamp

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

def design_parametric_crane(n_segments, boom_height, connectivity_pattern, taper_ratio=1.0):
    '''
    Design crane geometry with variable parameters

    Parameters:
    -----------
    n_segments : int
        Number of segments along the 30m boom (affects node count)
    boom_height : float
        Depth of the truss boom at the base (m)
    taper_ratio : float
        Ratio of tip height to base height (0.0 to 1.0)
        1.0 = rectangular (no taper), 0.0 = full taper to zero height at tip
    connectivity_pattern : array
        Pattern encoding: [diag_type, vertical_spacing, support_cables]
        - diag_type:
            0 = Alternating (Warren truss)
            1 = All positive slope
            2 = All negative slope
            3 = Double diagonals (X-pattern)
            4 = Fan from bottom (multiple diagonals per bottom node)
            5 = Fan from top (multiple diagonals per top node)
            6 = Long-span (skip 2 segments)
            7 = Mixed span (asymmetric, adjacent + long)
            8 = Concentrated (denser at ends, sparse in center)
            9 = Progressive fan (more connections toward tip)
            10 = Full connectivity (EVERY bottom node to EVERY top node)

    Returns:
    --------
    X : ndarray
        Node coordinates
    C : ndarray
        Connectivity matrix
    element_types : list
        Type of each element ('top_strut', 'bot_strut', 'vertical', 'diagonal', 'support')
    '''
    crane_length = 30.0  # Fixed 30m requirement
    tower_height = 0.0

    # Create boom nodes
    boom_x = np.linspace(0, crane_length, n_segments + 1)

    # Taper control: taper_ratio determines tip height relative to base height
    # taper_ratio = 1.0 → rectangular (constant height)
    # taper_ratio = 0.0 → fully tapered (zero height at tip)
    tip_height = boom_height * taper_ratio
    boom_y_top = np.linspace(tower_height + boom_height, tower_height + tip_height, n_segments + 1)
    boom_y_bot = np.full(n_segments + 1, tower_height)

    # Initialize nodes
    total_nodes = 1 + 2 * (n_segments + 1)  # Tower + top + bottom
    X = np.zeros([total_nodes, 2], float)

    node_idx = 0

    # Tower
    X[0] = [0, tower_height + 5]
    tower_base = 0
    node_idx += 1

    # Top nodes
    boom_top_nodes = []
    for i in range(n_segments + 1):
        X[node_idx] = [boom_x[i], boom_y_top[i]]
        boom_top_nodes.append(node_idx)
        node_idx += 1

    # Bottom nodes
    boom_bot_nodes = []
    for i in range(n_segments + 1):
        X[node_idx] = [boom_x[i], boom_y_bot[i]]
        boom_bot_nodes.append(node_idx)
        node_idx += 1

    # Create connectivity based on pattern
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

    # Vertical members (controlled by vertical_spacing parameter)
    vertical_spacing = int(connectivity_pattern[1])  # Every Nth node gets a vertical
    vertical_spacing = max(1, min(vertical_spacing, n_segments))

    for i in range(0, len(boom_top_nodes), vertical_spacing):
        elements.append([boom_top_nodes[i], boom_bot_nodes[i]])
        element_types.append('vertical')

    # Diagonal pattern
    diag_type = int(connectivity_pattern[0])

    if diag_type == 0:  # Alternating (classic Warren truss)
        for i in range(len(boom_top_nodes) - 1):
            if i % 2 == 0:
                elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
                element_types.append('diagonal')
            else:
                elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
                element_types.append('diagonal')

    elif diag_type == 1:  # All positive slope
        for i in range(len(boom_top_nodes) - 1):
            elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
            element_types.append('diagonal')

    elif diag_type == 2:  # All negative slope
        for i in range(len(boom_top_nodes) - 1):
            elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
            element_types.append('diagonal')

    elif diag_type == 3:  # Double diagonals (X-pattern, symmetric)
        for i in range(len(boom_top_nodes) - 1):
            elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
            elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
            element_types.append('diagonal')
            element_types.append('diagonal')

    elif diag_type == 4:  # Fan pattern from bottom nodes (multiple diagonals from single node)
        for i in range(len(boom_bot_nodes)):
            # Each bottom node connects to multiple top nodes (fan out)
            for j in range(max(0, i-1), min(len(boom_top_nodes), i+3)):
                if j != i:  # Skip direct vertical (already handled)
                    elements.append([boom_bot_nodes[i], boom_top_nodes[j]])
                    element_types.append('diagonal')

    elif diag_type == 5:  # Fan pattern from top nodes (multiple diagonals from single node)
        for i in range(len(boom_top_nodes)):
            # Each top node connects to multiple bottom nodes (fan out)
            for j in range(max(0, i-1), min(len(boom_bot_nodes), i+3)):
                if j != i:  # Skip direct vertical
                    elements.append([boom_top_nodes[i], boom_bot_nodes[j]])
                    element_types.append('diagonal')

    elif diag_type == 6:  # Long-span diagonals (skip 2 segments)
        for i in range(len(boom_top_nodes) - 2):
            elements.append([boom_bot_nodes[i], boom_top_nodes[i+2]])
            element_types.append('diagonal')
        for i in range(len(boom_bot_nodes) - 2):
            elements.append([boom_top_nodes[i], boom_bot_nodes[i+2]])
            element_types.append('diagonal')

    elif diag_type == 7:  # Mixed span (both adjacent and long-span, asymmetric)
        for i in range(len(boom_top_nodes) - 1):
            # Adjacent span
            if i % 2 == 0:
                elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
                element_types.append('diagonal')
            # Long span (every 3rd segment)
            if i % 3 == 0 and i + 2 < len(boom_top_nodes):
                elements.append([boom_top_nodes[i], boom_bot_nodes[i+2]])
                element_types.append('diagonal')

    elif diag_type == 8:  # Concentrated diagonals (denser at ends, sparser in middle)
        for i in range(len(boom_top_nodes) - 1):
            # More diagonals near ends (higher shear)
            n_mid = len(boom_top_nodes) // 2
            dist_from_center = abs(i - n_mid)

            if dist_from_center > n_mid * 0.6:  # Outer 40% - double diagonals
                elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
                elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
                element_types.append('diagonal')
                element_types.append('diagonal')
            elif dist_from_center > n_mid * 0.3:  # Middle 30% - single diagonal
                elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
                element_types.append('diagonal')
            # Center 30% - no diagonals (lower shear region)

    elif diag_type == 9:  # Progressive fan (more connections toward tip)
        for i in range(len(boom_bot_nodes)):
            # Number of connections increases toward the tip
            progress = i / max(1, len(boom_bot_nodes) - 1)
            max_span = int(1 + progress * 3)  # 1 to 4 connections

            for j in range(max(0, i - max_span), min(len(boom_top_nodes), i + max_span + 1)):
                if abs(j - i) > 0 and abs(j - i) <= max_span:
                    elements.append([boom_bot_nodes[i], boom_top_nodes[j]])
                    element_types.append('diagonal')

    elif diag_type == 10:  # Full connectivity (any bottom node to any top node)
        # Connect EVERY bottom node to EVERY top node (excluding direct verticals)
        # This creates a fully triangulated structure
        for i in range(len(boom_bot_nodes)):
            for j in range(len(boom_top_nodes)):
                if i != j:  # Skip direct vertical connections (already handled)
                    elements.append([boom_bot_nodes[i], boom_top_nodes[j]])
                    element_types.append('diagonal')

    # Support cables from tower (controlled by connectivity_pattern[2])
    num_support_cables = int(connectivity_pattern[2])
    num_support_cables = max(1, min(num_support_cables, len(boom_top_nodes) - 1))

    # Distribute support cables evenly
    support_indices = np.linspace(len(boom_top_nodes)//2, len(boom_top_nodes)-2,
                                  num_support_cables, dtype=int)

    for idx in support_indices:
        elements.append([tower_base, boom_top_nodes[idx]])
        element_types.append('support')

    return X, np.array(elements, dtype=int), element_types, tower_base, boom_top_nodes, boom_bot_nodes

def save_iteration_graph(X, C, element_types, thickness_params, objective, iteration_num, n_segments, boom_height):
    '''Save a graph of the current design without displaying it'''
    global output_dir

    try:
        n_elements = C.shape[0]

        colors = {
            'top_strut': 'blue',
            'bot_strut': 'green',
            'vertical': 'purple',
            'diagonal': 'orange',
            'support': 'red'
        }

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        # Plot elements with thickness-based line width
        for iEl in range(n_elements):
            if 2*iEl < len(thickness_params):
                d_outer = thickness_params[2*iEl] * 1000  # mm
                lw = max(0.5, min(5, d_outer / 10))
            else:
                lw = 1.5

            etype = element_types[iEl] if iEl < len(element_types) else 'diagonal'
            color = colors.get(etype, 'gray')

            ax.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                   [X[C[iEl,0],1], X[C[iEl,1],1]],
                   color=color, linewidth=lw, alpha=0.7)

        ax.scatter(X[:,0], X[:,1], c='black', s=30, zorder=5)
        ax.set_xlabel('x (m)', fontsize=10)
        ax.set_ylabel('y (m)', fontsize=10)
        ax.set_title(f'Iteration {iteration_num} - Objective: {objective:.2f}\n' +
                    f'{n_segments} segments, h={boom_height:.2f}m, {n_elements} elements, {X.shape[0]} nodes',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Add legend
        legend_elements = [plt.Line2D([0], [0], color=c, lw=2, label=t)
                          for t, c in colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()

        # Save with objective in filename for easy sorting
        filename = output_dir / f"iter_{iteration_num:04d}_obj_{objective:.2f}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        # If graph fails, don't crash optimization
        plt.close('all')
        pass

def save_intermediate_design(design_vector, objective, iteration_num):
    '''
    Save an intermediate crane design during optimization
    This is called periodically (every N minutes) to avoid losing progress

    Parameters:
    -----------
    design_vector : array
        Complete design vector
    objective : float
        Objective value
    iteration_num : int
        Current iteration number
    '''
    global output_dir

    try:
        # Extract design parameters
        n_segments = int(round(design_vector[0]))
        boom_height = design_vector[1]
        connectivity_pattern = design_vector[2:5]
        taper_ratio = design_vector[5]

        # Regenerate geometry
        X, C, types, tower_base, boom_top_nodes, boom_bot_nodes = \
            design_parametric_crane(n_segments, boom_height, connectivity_pattern, taper_ratio)

        # Extract thickness parameters
        n_elements = C.shape[0]
        thickness_params = design_vector[6:6+n_elements*2]

        # Package everything
        crane_data = {
            'design_vector': design_vector,
            'n_segments': n_segments,
            'boom_height': boom_height,
            'connectivity_pattern': connectivity_pattern,
            'taper_ratio': taper_ratio,
            'X': X,
            'C': C,
            'element_types': types,
            'tower_base': tower_base,
            'boom_top_nodes': boom_top_nodes,
            'boom_bot_nodes': boom_bot_nodes,
            'thickness_params': thickness_params,
            'objective': objective,
            'iteration': iteration_num
        }

        # Save with timestamp in filename
        timestamp = datetime.now().strftime("%H%M%S")
        filepath = output_dir / f'checkpoint_iter{iteration_num:04d}_obj{objective:.2f}_{timestamp}.pkl'

        with open(filepath, 'wb') as f:
            pickle.dump(crane_data, f)

        print(f"  [CHECKPOINT] Saved intermediate design to: {filepath.name}")

    except Exception as e:
        # Don't crash optimization if save fails
        print(f"  [WARNING] Failed to save intermediate design: {e}")

def evaluate_crane_design(design_vector, max_load=40000, verbose=False):
    '''
    Evaluate a crane design and return cost with penalties for safety violations

    Design vector encoding:
    [0]: n_segments (8-20)
    [1]: boom_height (0.5-2.0 m)
    [2]: diag_type (0-9)
    [3]: vertical_spacing (1-3)
    [4]: num_support_cables (1-3)
    [5]: taper_ratio (0.0-1.0)
    [6:]: member thicknesses - pairs of (d_outer, d_inner) for each element

    Returns:
    --------
    objective : float
        Cost + penalties (minimize this)
    '''
    # Decode design
    n_segments = int(round(design_vector[0]))
    boom_height = design_vector[1]
    connectivity_pattern = design_vector[2:5]
    taper_ratio = design_vector[5]

    # Generate geometry
    try:
        X, C, element_types, tower_base, boom_top_nodes, boom_bot_nodes = \
            design_parametric_crane(n_segments, boom_height, connectivity_pattern, taper_ratio)
    except Exception as e:
        if verbose:
            print(f"Geometry generation failed: {e}")
        return 1e9  # Huge penalty for invalid geometry

    n_elements = C.shape[0]
    n_nodes = X.shape[0]

    # Check if we have enough thickness parameters
    expected_thickness_params = n_elements * 2  # 2 per element (d_outer, d_inner)
    if len(design_vector) < 6 + expected_thickness_params:
        if verbose:
            print(f"Not enough thickness parameters: need {expected_thickness_params}, got {len(design_vector)-6}")
        return 1e9

    # Extract member thicknesses
    thickness_params = design_vector[6:6+expected_thickness_params]

    # Material properties
    E = 200e9
    rho_steel = 7850
    g = 9.81

    # Build element areas and inertias
    element_areas = []
    element_inertias = []
    total_mass = 0

    for iEl in range(n_elements):
        d_outer = thickness_params[2*iEl]
        d_inner = thickness_params[2*iEl + 1]

        # Ensure inner < outer with minimum wall thickness of 2mm
        min_wall = 0.002  # 2mm minimum wall thickness
        if d_inner >= d_outer - min_wall:
            d_inner = d_outer - min_wall

        # Ensure positive inner diameter
        if d_inner < 0.001:
            d_inner = 0.001

        A, I = hollow_circular_section(d_outer, d_inner)
        element_areas.append(A)
        element_inertias.append(I)

        # Calculate mass
        n1, n2 = C[iEl]
        L = np.linalg.norm(X[n2] - X[n1])
        volume = A * L
        mass = rho_steel * volume
        total_mass += mass

    # Boundary conditions
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True  # Pin support
    bc[tower_base, :] = True  # Tower fixed

    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Assemble global stiffness matrix
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)

    for iEl in range(n_elements):
        A = element_areas[iEl]
        dof = dof_el(C[iEl,0], C[iEl,1])
        k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

    # Check for singular matrix
    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    try:
        # Test matrix conditioning
        cond = np.linalg.cond(k_reduced)
        if cond > 1e12:
            if verbose:
                print(f"Ill-conditioned matrix: cond={cond:.2e}")
            return 1e8
    except:
        return 1e8

    # Apply loads (tip load + self-weight)
    loads = np.zeros([n_nodes, 2], float)
    loads[boom_top_nodes[-1], 1] = -max_load

    # Add self-weight
    for iEl in range(n_elements):
        n1, n2 = C[iEl]
        L = np.linalg.norm(X[n2] - X[n1])
        A = element_areas[iEl]
        weight = rho_steel * A * L * g
        loads[n1, 1] -= weight / 2
        loads[n2, 1] -= weight / 2

    load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

    # Solve
    try:
        displacements = np.zeros([2*n_nodes], float)
        displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
        D = displacements.reshape(n_nodes, 2)
    except np.linalg.LinAlgError:
        if verbose:
            print("Singular matrix during solve")
        return 1e8

    # Calculate stresses and forces
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

    # Calculate safety factors
    sigma_max = np.max(np.abs(element_stresses))
    sigma_adm = 100e6  # 100 MPa

    if sigma_max < 1e-6:
        FS_tension = 1000
    else:
        FS_tension = sigma_adm / sigma_max

    # Buckling check
    P_max_compression = abs(np.min(element_forces))

    # Find critical buckling load
    P_critica = 1e12
    for iEl in range(n_elements):
        if element_forces[iEl] < -1:  # Compression
            n1, n2 = C[iEl]
            L = np.linalg.norm(X[n2] - X[n1])
            I = element_inertias[iEl]
            P_cr = (np.pi**2 * E * I) / (L**2)
            if P_cr < P_critica:
                P_critica = P_cr

    if P_max_compression < 1e-6:
        FS_pandeo = 1000
    else:
        FS_pandeo = P_critica / P_max_compression

    # Calculate cost
    m0 = 1000  # Reference mass (kg)
    n_elementos_0 = 50
    n_uniones_0 = 25

    n_uniones = n_nodes
    cost = (total_mass / m0) + 1.5 * (n_elements / n_elementos_0) + 2 * (n_uniones / n_uniones_0)

    # Penalties for safety violations
    penalty = 0
    min_FS = 2.0

    if FS_tension < min_FS:
        penalty += 1000 * (min_FS - FS_tension)**2

    if FS_pandeo < min_FS:
        penalty += 1000 * (min_FS - FS_pandeo)**2

    # Deflection penalty (max 200mm at tip)
    tip_deflection = abs(D[boom_top_nodes[-1], 1])
    max_deflection = 0.200  # 200mm
    if tip_deflection > max_deflection:
        penalty += 500 * (tip_deflection - max_deflection)**2

    objective = cost + penalty

    # Track iterations and save graphs ONLY for improving designs
    global iteration_counter, best_objective, best_design_vector, last_save_time

    iteration_counter += 1

    # Save graph ONLY if this is a better design
    if objective < best_objective and objective < 1e7:  # Only save valid designs that improve
        # Extract thickness parameters
        expected_thickness_params = n_elements * 2
        if len(design_vector) >= 6 + expected_thickness_params:
            thickness_params = design_vector[6:6+expected_thickness_params]
        else:
            thickness_params = []

        save_iteration_graph(X, C, element_types, thickness_params,
                           objective, iteration_counter, n_segments, boom_height)

        best_objective = objective
        best_design_vector = design_vector.copy()  # Store the best design
        print(f"[Iter {iteration_counter}] NEW BEST! Objective: {objective:.2f} (Cost: {cost:.2f}, Penalty: {penalty:.2f})")

        # Check if enough time has passed to save an intermediate checkpoint
        current_time = time.time()
        if last_save_time is None:
            last_save_time = current_time

        time_since_last_save = current_time - last_save_time
        save_interval = 300  # Save every 5 minutes (300 seconds)

        if time_since_last_save >= save_interval:
            save_intermediate_design(design_vector, objective, iteration_counter)
            last_save_time = current_time

    if verbose:
        print(f"Design: n_seg={n_segments}, h={boom_height:.2f}m, n_el={n_elements}, n_nodes={n_nodes}")
        print(f"  Mass: {total_mass:.1f} kg, Cost: {cost:.2f}, Penalty: {penalty:.2f}")
        print(f"  FS_tension: {FS_tension:.2f}, FS_pandeo: {FS_pandeo:.2f}")
        print(f"  Tip deflection: {tip_deflection*1000:.1f} mm")
        print(f"  Objective: {objective:.2f}")

    return objective

def optimize_crane(maxiter=100, popsize=15):
    '''
    Main optimization routine

    Parameters:
    -----------
    maxiter : int
        Maximum number of iterations for differential evolution (default: 100)
    popsize : int
        Population size for differential evolution (default: 15)
    '''
    global output_dir, iteration_counter, best_objective

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"gruita3_iterations_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"Saving iteration graphs to: {output_dir.absolute()}")

    # Reset counters
    iteration_counter = 0
    best_objective = float('inf')

    print("="*80)
    print("SELF-OPTIMIZING CRANE (GRUITA 3)")
    print("="*80)
    print("\nThe crane will design itself by optimizing:")
    print("  1. Number of segments (node distribution)")
    print("  2. Boom height (truss depth)")
    print("  3. Connectivity pattern (diagonals, verticals, support cables)")
    print("  4. Individual member thicknesses (each element optimized)")
    print("\nObjective: Minimize cost while maintaining FS >= 2.0")
    print("\nGraphs will be saved ONLY when a better design is found!")
    print("="*80)

    # Start with a reasonable baseline to estimate element count
    baseline_segments = 12
    baseline_height = 1.0
    baseline_pattern = [0, 1, 2]  # Alternating diags, vertical every node, 2 support cables

    X_base, C_base, types_base, _, _, _ = design_parametric_crane(
        baseline_segments, baseline_height, baseline_pattern)

    n_elements_base = C_base.shape[0]
    print(f"\nBaseline design has {n_elements_base} elements")

    # Create bounds for optimization
    # [n_segments, boom_height, diag_type, vertical_spacing, num_support_cables, taper_ratio,
    # d_outer_0, d_inner_0, d_outer_1, d_inner_1, ...]

    bounds = [
        (8, 20),      # n_segments
        (0.5, 2.0),   # boom_height
        (0, 10),      # diag_type (now 0-10 including full connectivity)
        (1, 3),       # vertical_spacing
        (1, 3),       # num_support_cables
        (0.0, 1.0),   # taper_ratio (0=full taper, 1=rectangular)
    ]

    # Add thickness bounds for maximum possible elements
    # Type 10 (full connectivity) with 20 segments creates ~500 elements
    max_possible_elements = 600  # Conservative overestimate for type 10

    for i in range(max_possible_elements):
        bounds.append((0.010, 0.050))  # d_outer: 10mm to 50mm (max constraint)
        bounds.append((0.005, 0.045))  # d_inner: 5mm to 45mm

    print(f"\nOptimization variables: {len(bounds)}")
    print("\nStarting global optimization (differential evolution)...")
    print("This may take several minutes...\n")

    # Create initial guess with reasonable design
    initial_guess = []
    initial_guess.append(12)   # n_segments
    initial_guess.append(1.0)  # boom_height
    initial_guess.append(0)    # diag_type (alternating)
    initial_guess.append(1)    # vertical_spacing (every node)
    initial_guess.append(2)    # num_support_cables
    initial_guess.append(0.0)  # taper_ratio (0=full taper like original)

    # Add initial thicknesses: struts thicker than cables
    for i in range(max_possible_elements):
        if i < 40:  # First elements are likely struts
            initial_guess.append(0.040)  # 40mm outer
            initial_guess.append(0.030)  # 30mm inner (10mm wall)
        else:  # Later elements are likely cables
            initial_guess.append(0.025)  # 25mm outer
            initial_guess.append(0.020)  # 20mm inner (2.5mm wall)

    print("\nEvaluating initial design...")
    initial_obj = evaluate_crane_design(np.array(initial_guess), max_load=40000, verbose=True)

    # Use differential evolution for global optimization
    print(f"\nStarting optimization with differential evolution...")
    print(f"  Max iterations: {maxiter}")
    print(f"  Population size: {popsize}")
    result = differential_evolution(
        evaluate_crane_design,
        bounds,
        args=(40000, False),  # 40kN load, not verbose
        x0=np.array(initial_guess),
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=0.01,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=42,
        workers=1,
        updating='deferred',
        disp=True,
        atol=0.01,
        polish=False  # Disable polishing to avoid numerical issues
    )

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)

    # Evaluate best design with verbose output
    print("\nBest design found:")
    print("-"*80)
    best_obj = evaluate_crane_design(result.x, max_load=40000, verbose=True)

    # Extract and visualize best design
    n_segments_opt = int(round(result.x[0]))
    boom_height_opt = result.x[1]
    connectivity_pattern_opt = result.x[2:5]

    X_opt, C_opt, types_opt, tower_base, boom_top_nodes, boom_bot_nodes = \
        design_parametric_crane(n_segments_opt, boom_height_opt, connectivity_pattern_opt)

    # Visualize
    plot_optimized_crane(X_opt, C_opt, types_opt, result.x[5:],
                        n_segments_opt, boom_height_opt, connectivity_pattern_opt)

    return result

def plot_optimized_crane(X, C, element_types, thickness_params,
                        n_segments, boom_height, connectivity_pattern):
    '''Plot the optimized crane design'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Crane geometry with member types
    n_elements = C.shape[0]

    colors = {
        'top_strut': 'blue',
        'bot_strut': 'green',
        'vertical': 'purple',
        'diagonal': 'orange',
        'support': 'red'
    }

    for iEl in range(n_elements):
        etype = element_types[iEl]
        color = colors.get(etype, 'gray')
        d_outer = thickness_params[2*iEl] * 1000  # Convert to mm
        linewidth = max(1, d_outer / 10)  # Scale linewidth with thickness

        ax1.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                [X[C[iEl,0],1], X[C[iEl,1],1]],
                color=color, linewidth=linewidth, alpha=0.7)

    ax1.scatter(X[:,0], X[:,1], c='black', s=30, zorder=5)
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_title(f'Optimized Crane Design\n{n_segments} segments, h={boom_height:.2f}m',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend(handles=[plt.Line2D([0], [0], color=c, linewidth=2, label=t)
                       for t, c in colors.items()], loc='upper right')

    # Plot 2: Member thickness distribution
    thicknesses_outer = [thickness_params[2*i]*1000 for i in range(n_elements)]
    thicknesses_inner = [thickness_params[2*i+1]*1000 for i in range(n_elements)]
    wall_thickness = [thicknesses_outer[i] - thicknesses_inner[i] for i in range(n_elements)]

    x_pos = np.arange(n_elements)
    ax2.bar(x_pos, thicknesses_outer, label='Outer diameter', alpha=0.7)
    ax2.bar(x_pos, thicknesses_inner, label='Inner diameter', alpha=0.7)
    ax2.axhline(y=50, color='r', linestyle='--', linewidth=2, label='Max outer diameter (50mm)')
    ax2.set_xlabel('Element index', fontsize=12)
    ax2.set_ylabel('Diameter (mm)', fontsize=12)
    ax2.set_title('Member Thickness Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save final optimized design instead of showing
    global output_dir
    final_filename = output_dir / "FINAL_optimized_design.png"
    plt.savefig(final_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFinal design saved to: {final_filename}")

    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZED DESIGN SUMMARY")
    print("="*80)
    diag_names = {
        0: "Alternating (Warren)",
        1: "All positive",
        2: "All negative",
        3: "X-pattern",
        4: "Fan from bottom",
        5: "Fan from top",
        6: "Long-span",
        7: "Mixed span",
        8: "Concentrated",
        9: "Progressive fan",
        10: "Full connectivity"
    }

    print(f"Segments: {n_segments}")
    print(f"Boom height: {boom_height:.3f} m")
    diag_num = int(connectivity_pattern[0])
    print(f"Diagonal type: {diag_num} - {diag_names.get(diag_num, 'Unknown')}")
    print(f"Vertical spacing: every {int(connectivity_pattern[1])} nodes")
    print(f"Support cables: {int(connectivity_pattern[2])}")
    print(f"\nTotal elements: {n_elements}")
    print(f"Total nodes: {X.shape[0]}")
    print(f"\nMember thickness range:")
    print(f"  Outer diameter: {min(thicknesses_outer):.1f} - {max(thicknesses_outer):.1f} mm")
    print(f"  Wall thickness: {min(wall_thickness):.1f} - {max(wall_thickness):.1f} mm")
    print("="*80)

def save_best_crane(result, filename='best_crane.pkl'):
    '''
    Save the best crane design to a file for later testing

    Parameters:
    -----------
    result : OptimizeResult
        Result object from differential_evolution
    filename : str
        Name of file to save design to
    '''
    global output_dir

    # Extract design parameters
    n_segments_opt = int(round(result.x[0]))
    boom_height_opt = result.x[1]
    connectivity_pattern_opt = result.x[2:5]
    taper_ratio_opt = result.x[5]

    # Regenerate geometry
    X_opt, C_opt, types_opt, tower_base, boom_top_nodes, boom_bot_nodes = \
        design_parametric_crane(n_segments_opt, boom_height_opt, connectivity_pattern_opt, taper_ratio_opt)

    # Extract thickness parameters
    n_elements = C_opt.shape[0]
    thickness_params = result.x[6:6+n_elements*2]

    # Package everything
    crane_data = {
        'design_vector': result.x,
        'n_segments': n_segments_opt,
        'boom_height': boom_height_opt,
        'connectivity_pattern': connectivity_pattern_opt,
        'taper_ratio': taper_ratio_opt,
        'X': X_opt,
        'C': C_opt,
        'element_types': types_opt,
        'tower_base': tower_base,
        'boom_top_nodes': boom_top_nodes,
        'boom_bot_nodes': boom_bot_nodes,
        'thickness_params': thickness_params,
        'objective': result.fun
    }

    # Save to output directory
    filepath = output_dir / filename
    with open(filepath, 'wb') as f:
        pickle.dump(crane_data, f)

    print(f"\nBest crane design saved to: {filepath}")
    return filepath

def load_crane(filename='best_crane.pkl'):
    '''
    Load a saved crane design

    Parameters:
    -----------
    filename : str or Path
        Path to saved crane file

    Returns:
    --------
    crane_data : dict
        Dictionary containing all crane design data
    '''
    with open(filename, 'rb') as f:
        crane_data = pickle.load(f)

    print(f"Loaded crane design from: {filename}")
    print(f"  Segments: {crane_data['n_segments']}")
    print(f"  Elements: {crane_data['C'].shape[0]}")
    print(f"  Nodes: {crane_data['X'].shape[0]}")
    print(f"  Objective: {crane_data['objective']:.2f}")

    return crane_data

def analyze_moving_load(crane_data, load_magnitudes=None):
    '''
    Analyze crane performance with varying loads at different positions
    Tests loads at different positions along the bottom chord

    Parameters:
    -----------
    crane_data : dict
        Crane design data from load_crane()
    load_magnitudes : array-like, optional
        Load magnitudes to test (default: 0 to 40000 N in 5 steps)
    '''
    print("\n" + "="*70)
    print("MOVING LOAD ANALYSIS")
    print("="*70)

    # Extract crane data
    X = crane_data['X']
    C = crane_data['C']
    tower_base = crane_data['tower_base']
    boom_top_nodes = crane_data['boom_top_nodes']
    boom_bot_nodes = crane_data['boom_bot_nodes']
    thickness_params = crane_data['thickness_params']

    # Material properties
    E = 200e9
    rho_steel = 7850
    g = 9.81

    # Build element areas from thickness parameters
    element_areas = []
    for iEl in range(C.shape[0]):
        d_outer = thickness_params[2*iEl]
        d_inner = thickness_params[2*iEl + 1]

        # Ensure valid dimensions
        min_wall = 0.002
        if d_inner >= d_outer - min_wall:
            d_inner = d_outer - min_wall
        if d_inner < 0.001:
            d_inner = 0.001

        A, I = hollow_circular_section(d_outer, d_inner)
        element_areas.append(A)

    # Boundary conditions
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True
    bc[tower_base, :] = True

    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Assembly global stiffness matrix
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)
    for iEl in range(C.shape[0]):
        A = element_areas[iEl]
        dof = dof_el(C[iEl,0], C[iEl,1])
        k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    # Test parameters
    if load_magnitudes is None:
        load_magnitudes = np.linspace(0, 40000, 5)
    test_positions = boom_bot_nodes

    # Storage for results
    max_deflections = np.zeros((len(load_magnitudes), len(test_positions)))
    max_stresses = np.zeros((len(load_magnitudes), len(test_positions)))
    max_tensions = np.zeros((len(load_magnitudes), len(test_positions)))
    max_compressions = np.zeros((len(load_magnitudes), len(test_positions)))

    print(f"\nTesting {len(load_magnitudes)} load magnitudes at {len(test_positions)} positions...")
    print(f"Load range: {load_magnitudes[0]:.0f} to {load_magnitudes[-1]:.0f} N")

    # Iterate through loads and positions
    for i, load_mag in enumerate(load_magnitudes):
        for j, pos_node in enumerate(test_positions):
            # Apply load at current position
            loads = np.zeros([n_nodes, 2], float)
            loads[pos_node, 1] = -load_mag

            # Add self-weight
            for iEl in range(C.shape[0]):
                n1, n2 = C[iEl]
                L = np.linalg.norm(X[n2] - X[n1])
                A = element_areas[iEl]
                volume = A * L
                mass = rho_steel * volume
                weight = mass * g
                loads[n1, 1] -= weight / 2
                loads[n2, 1] -= weight / 2

            load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

            # Solve
            displacements = np.zeros([2*n_nodes], float)
            displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
            D = displacements.reshape(n_nodes, 2)

            # Calculate maximum deflection
            max_deflections[i, j] = np.max(np.abs(D))

            # Calculate stresses
            element_stresses = []
            for iEl in range(C.shape[0]):
                n1, n2 = C[iEl]
                d_el = np.concatenate([D[n1], D[n2]])
                A = element_areas[iEl]

                k_el = element_stiffness(X[n1], X[n2], A, E)
                f_el = k_el @ d_el

                d_vec = X[n2] - X[n1]
                L = np.linalg.norm(d_vec)

                if L < 1e-10:
                    element_stresses.append(0)
                    continue

                u_vec = d_vec / L
                du = np.array([d_el[2] - d_el[0], d_el[3] - d_el[1]])
                epsilon = np.dot(du, u_vec) / L
                stress = E * epsilon
                element_stresses.append(stress)

            element_stresses = np.array(element_stresses)
            max_stresses[i, j] = np.max(np.abs(element_stresses))
            max_tensions[i, j] = np.max(element_stresses)
            max_compressions[i, j] = np.min(element_stresses)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    positions_x = X[boom_bot_nodes, 0]

    # Plot 1: Max deflection vs position
    ax1 = axes[0, 0]
    for i, load_mag in enumerate(load_magnitudes):
        ax1.plot(positions_x, max_deflections[i, :] * 1000, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax1.set_xlabel('Position along crane (m)', fontsize=12)
    ax1.set_ylabel('Maximum deflection (mm)', fontsize=12)
    ax1.set_title('Maximum Deflection vs Load Position', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Max stress vs position
    ax2 = axes[0, 1]
    for i, load_mag in enumerate(load_magnitudes):
        ax2.plot(positions_x, max_stresses[i, :] / 1e6, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax2.axhline(y=100, color='r', linestyle='--', label='Allowable (100 MPa)')
    ax2.set_xlabel('Position along crane (m)', fontsize=12)
    ax2.set_ylabel('Maximum stress (MPa)', fontsize=12)
    ax2.set_title('Maximum Stress vs Load Position', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Max tensile stress vs position
    ax3 = axes[1, 0]
    for i, load_mag in enumerate(load_magnitudes):
        ax3.plot(positions_x, max_tensions[i, :] / 1e6, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax3.axhline(y=100, color='r', linestyle='--', label='Allowable (100 MPa)')
    ax3.set_xlabel('Position along crane (m)', fontsize=12)
    ax3.set_ylabel('Maximum tensile stress (MPa)', fontsize=12)
    ax3.set_title('Maximum Tensile Stress vs Load Position', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Max compressive stress vs position
    ax4 = axes[1, 1]
    for i, load_mag in enumerate(load_magnitudes):
        ax4.plot(positions_x, max_compressions[i, :] / 1e6, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax4.axhline(y=-100, color='r', linestyle='--', label='Allowable (-100 MPa)')
    ax4.set_xlabel('Position along crane (m)', fontsize=12)
    ax4.set_ylabel('Maximum compressive stress (MPa)', fontsize=12)
    ax4.set_title('Maximum Compressive Stress vs Load Position', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"Overall maximum deflection: {np.max(max_deflections)*1000:.2f} mm")
    print(f"Overall maximum stress: {np.max(max_stresses)/1e6:.2f} MPa")
    print(f"Maximum tensile stress: {np.max(max_tensions)/1e6:.2f} MPa")
    print(f"Maximum compressive stress: {np.min(max_compressions)/1e6:.2f} MPa")
    print("="*70)

    return max_deflections, max_stresses, positions_x

def animate_moving_load(crane_data, load_magnitude=30000, scale_factor=100, interval=200):
    '''
    Create an animated time-lapse of crane deformation as load moves along bottom chord

    Parameters:
    -----------
    crane_data : dict
        Crane design data from load_crane()
    load_magnitude : float
        Magnitude of the moving load in Newtons (default: 30000 N = 30 kN)
    scale_factor : float
        Scale factor for deformation visualization (default: 100)
    interval : int
        Time between frames in milliseconds (default: 200 ms)
    '''
    print("\n" + "="*70)
    print("ANIMATED MOVING LOAD ANALYSIS")
    print("="*70)
    print(f"Load magnitude: {load_magnitude/1000:.1f} kN")
    print(f"Deformation scale: {scale_factor}x")
    print("Creating animation...")

    # Extract crane data
    X = crane_data['X']
    C = crane_data['C']
    tower_base = crane_data['tower_base']
    boom_top_nodes = crane_data['boom_top_nodes']
    boom_bot_nodes = crane_data['boom_bot_nodes']
    thickness_params = crane_data['thickness_params']

    # Material properties
    E = 200e9
    rho_steel = 7850
    g = 9.81

    # Build element areas
    element_areas = []
    for iEl in range(C.shape[0]):
        d_outer = thickness_params[2*iEl]
        d_inner = thickness_params[2*iEl + 1]

        min_wall = 0.002
        if d_inner >= d_outer - min_wall:
            d_inner = d_outer - min_wall
        if d_inner < 0.001:
            d_inner = 0.001

        A, I = hollow_circular_section(d_outer, d_inner)
        element_areas.append(A)

    # Boundary conditions
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True
    bc[tower_base, :] = True

    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Assembly global stiffness matrix
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)
    for iEl in range(C.shape[0]):
        A = element_areas[iEl]
        dof = dof_el(C[iEl,0], C[iEl,1])
        k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    # Test positions
    test_positions = boom_bot_nodes

    # Calculate deformations for each position
    deformations = []
    max_deflections_list = []

    for pos_node in test_positions:
        loads = np.zeros([n_nodes, 2], float)
        loads[pos_node, 1] = -load_magnitude

        # Add self-weight
        for iEl in range(C.shape[0]):
            n1, n2 = C[iEl]
            L = np.linalg.norm(X[n2] - X[n1])
            A = element_areas[iEl]
            volume = A * L
            mass = rho_steel * volume
            weight = mass * g
            loads[n1, 1] -= weight / 2
            loads[n2, 1] -= weight / 2

        load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

        # Solve
        displacements = np.zeros([2*n_nodes], float)
        displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
        D = displacements.reshape(n_nodes, 2)

        deformations.append(D)
        tip_deflection = abs(D[boom_top_nodes[-1], 1])
        max_deflections_list.append(tip_deflection)

    # Create animation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    def init():
        ax1.clear()
        ax2.clear()
        return []

    def update(frame):
        ax1.clear()
        ax2.clear()

        pos_node = test_positions[frame]
        D = deformations[frame]
        X_deformed = X + D * scale_factor

        # Plot 1: Original and deformed structure
        for iEl in range(C.shape[0]):
            ax1.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                    [X[C[iEl,0],1], X[C[iEl,1],1]],
                    'gray', linewidth=1, alpha=0.3, linestyle='--')

        for iEl in range(C.shape[0]):
            ax1.plot([X_deformed[C[iEl,0],0], X_deformed[C[iEl,1],0]],
                    [X_deformed[C[iEl,0],1], X_deformed[C[iEl,1],1]],
                    'b-', linewidth=2)

        # Mark load position
        load_pos = X[pos_node]
        ax1.arrow(load_pos[0], load_pos[1] + 0.5, 0, -0.3,
                 head_width=0.5, head_length=0.2, fc='red', ec='red', linewidth=3)
        ax1.plot(load_pos[0], load_pos[1], 'ro', markersize=15, label='Load Position')

        # Mark supports
        ax1.plot(X[boom_bot_nodes[0], 0], X[boom_bot_nodes[0], 1], 'g^',
                markersize=12, label='Pin Support')
        ax1.plot(X[tower_base, 0], X[tower_base, 1], 'g^', markersize=12)

        ax1.set_xlabel('x (m)', fontsize=12)
        ax1.set_ylabel('y (m)', fontsize=12)
        ax1.set_title(f'Crane Deformation - Load at x = {X[pos_node, 0]:.2f} m (Scale: {scale_factor}x)\n'
                     f'Load: {load_magnitude/1000:.1f} kN | Tip Deflection: {max_deflections_list[frame]*1000:.2f} mm',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        ax1.legend(loc='upper right')

        # Plot 2: Tip deflection profile
        positions_x = X[boom_bot_nodes, 0]
        ax2.plot(positions_x, np.array(max_deflections_list) * 1000, 'b-o', linewidth=2, markersize=8)
        ax2.plot(X[pos_node, 0], max_deflections_list[frame] * 1000, 'ro', markersize=15)
        ax2.axvline(x=X[pos_node, 0], color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Load Position along crane (m)', fontsize=12)
        ax2.set_ylabel('Tip Deflection (mm)', fontsize=12)
        ax2.set_title('Tip Deflection vs Load Position', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return []

    anim = FuncAnimation(fig, update, init_func=init, frames=len(test_positions),
                        interval=interval, blit=False, repeat=True)

    print("Animation created! Close the window to continue...")
    plt.show()

    print("="*70)

    return anim

if __name__ == '__main__':
    import sys

    # Parse command line arguments
    maxiter = 100  # Default
    popsize = 15   # Default

    if len(sys.argv) > 1:
        try:
            maxiter = int(sys.argv[1])
            print(f"Using maxiter from command line: {maxiter}")
        except ValueError:
            print(f"Invalid maxiter argument: {sys.argv[1]}, using default: {maxiter}")

    if len(sys.argv) > 2:
        try:
            popsize = int(sys.argv[2])
            print(f"Using popsize from command line: {popsize}")
        except ValueError:
            print(f"Invalid popsize argument: {sys.argv[2]}, using default: {popsize}")

    # Run optimization
    result = optimize_crane(maxiter=10, popsize=5)

    # Save the best design
    save_best_crane(result)
