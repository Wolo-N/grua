import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import time
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import NonlinearConstraint

def element_stiffness(n1, n2, A, E):
    '''
    Returns Rod Elemental Stiffness Matrix in global coordinates

    Parameters:
    -----------
    n1 : array-like
        Coordinates of first node [x1, y1]
    n2 : array-like
        Coordinates of second node [x2, y2]
    A : float
        Cross-sectional area of the element
    E : float
        Young's modulus of the material

    Returns:
    --------
    k_structural : ndarray (4x4)
        Element stiffness matrix in global coordinates
        DOF order: [u1, v1, u2, v2] where u=x-displacement, v=y-displacement
    '''
    # Calculate element vector and length
    d = n2 - n1  # Element vector from node 1 to node 2
    L = np.linalg.norm(d)  # Element length

    # Check for zero-length elements to avoid division by zero
    if L < 1e-10:
        return np.zeros((4, 4), dtype=float)

    # Direction cosines (unit vector components)
    c = d[0] / L  # cos(θ) - x-direction cosine
    s = d[1] / L  # sin(θ) - y-direction cosine

    # Local element stiffness matrix (1D rod element)
    # In local coordinates: k_local = (AE/L) * [1 -1; -1 1]
    k_local = (A * E / L) * np.array([[ 1, -1],
                                      [-1,  1]], dtype=float)

    # Transformation matrix from local to global coordinates
    # T maps local displacements [u_local] to global displacements [u1, v1, u2, v2]
    T = np.array([[ c, s, 0, 0],  # Local displacement mapped to global DOFs
                  [ 0, 0, c, s]], dtype=float)

    # Transform stiffness matrix to global coordinates: K_global = T^T * K_local * T
    k_structural = np.matmul(T.T, np.matmul(k_local, T))

    return k_structural

def dof_el(nnod1, nnod2):
    '''Returns Elemental DOF for Assembly'''
    return [2*(nnod1+1)-2,2*(nnod1+1)-1,2*(nnod2+1)-2,2*(nnod2+1)-1]

def plot_truss(X, C, title="Truss Structure", scale_factor=1, show_nodes=True):
    '''Plot truss structure with optional node numbering'''
    plt.figure(figsize=(15, 8))

    # Plot elements
    for iEl in range(C.shape[0]):
        plt.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                [X[C[iEl,0],1], X[C[iEl,1],1]],
                'b-', linewidth=2)

    # Plot nodes
    if show_nodes:
        plt.scatter(X[:,0], X[:,1], c='red', s=50, zorder=5)
        for i in range(X.shape[0]):
            plt.annotate(f'{i}', (X[i,0], X[i,1]), xytext=(5,5),
                        textcoords='offset points', fontsize=8)

    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

def plot_stress_heatmap(X, C, stresses, element_areas, title="Stress Distribution Heatmap"):
    '''
    Plot truss structure with stress heatmap visualization

    Parameters:
    -----------
    X : ndarray
        Node coordinates
    C : ndarray
        Connectivity matrix
    stresses : ndarray
        Stress values for each element (Pa)
    element_areas : list
        Cross-sectional areas for each element
    title : str
        Plot title
    '''
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create line segments for each element
    segments = []
    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        segments.append([(X[n1, 0], X[n1, 1]), (X[n2, 0], X[n2, 1])])

    # Normalize stress values for colormap
    stress_mpa = stresses / 1e6  # Convert to MPa
    norm = Normalize(vmin=np.min(stress_mpa), vmax=np.max(stress_mpa))

    # Create colormap (blue for compression, red for tension)
    cmap = cm.RdBu_r  # Reversed so red is tension, blue is compression

    # Create line collection with colors based on stress
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4)
    lc.set_array(stress_mpa)

    # Add line collection to plot
    line = ax.add_collection(lc)

    # Add colorbar
    cbar = fig.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label('Stress (MPa)\n← Compression | Tension →', rotation=270, labelpad=25)

    # Plot nodes
    ax.scatter(X[:, 0], X[:, 1], c='black', s=30, zorder=5, alpha=0.5)

    # Labels and formatting
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Auto-scale to fit data
    ax.autoscale()

    plt.tight_layout()
    plt.show()

    # Print stress statistics
    print("\n" + "="*60)
    print("STRESS ANALYSIS SUMMARY")
    print("="*60)
    print(f"Maximum Tensile Stress:     {np.max(stress_mpa):8.2f} MPa")
    print(f"Maximum Compressive Stress: {np.min(stress_mpa):8.2f} MPa")
    print(f"Average Stress Magnitude:   {np.mean(np.abs(stress_mpa)):8.2f} MPa")
    print("="*60)

def design_crane_geometry():
    '''Design 30m crane truss geometry'''

    # Design parameters
    crane_length = 30.0  # m
    tower_height = 0.0  # m
    boom_height = 1.0    # m boom depth
    n_boom_segments = 12  # Number of segments along boom

    # Calculate positions for horizontal boom
    boom_x = np.linspace(0, crane_length, n_boom_segments + 1)
    boom_y_top = np.full(n_boom_segments + 1, tower_height + boom_height)  # Top chord horizontal
    boom_y_bot = np.full(n_boom_segments + 1, tower_height)  # Bottom chord horizontal

    # Initialize node coordinates
    total_nodes = 1 + 2 * (n_boom_segments + 1)  # Tower + all top and bottom boom nodes
    X = np.zeros([total_nodes, 2], float)

    node_idx = 0

    # Tower base and top
    X[0] = [0, tower_height + 5]  # Tower base
    tower_base = 0
    node_idx += 1

    # Boom top chord nodes (keep all nodes for trapezoidal shape)
    boom_top_nodes = []
    for i in range(n_boom_segments + 1):
        X[node_idx] = [boom_x[i], boom_y_top[i]]
        boom_top_nodes.append(node_idx)
        node_idx += 1

    # Boom bottom chord nodes (keep all)
    boom_bot_nodes = []
    for i in range(n_boom_segments + 1):
        X[node_idx] = [boom_x[i], boom_y_bot[i]]
        boom_bot_nodes.append(node_idx)
        node_idx += 1

    return X, tower_base, boom_top_nodes, boom_bot_nodes

def create_crane_connectivity(tower_base, boom_top_nodes, boom_bot_nodes):
    '''Create element connectivity matrix for crane'''

    elements = []

    # Boom top chord elements (all top nodes form a continuous chord)
    for i in range(len(boom_top_nodes) - 1):
        elements.append([boom_top_nodes[i], boom_top_nodes[i+1]])

    # Boom bottom chord elements (all bottom nodes)
    for i in range(len(boom_bot_nodes) - 1):
        elements.append([boom_bot_nodes[i], boom_bot_nodes[i+1]])

    # Boom vertical members (connect each top node to corresponding bottom node)
    # Both arrays have the same length, so direct 1-to-1 mapping
    for i in range(len(boom_top_nodes)):
        elements.append([boom_top_nodes[i], boom_bot_nodes[i]])

    # Diagonal members (alternating pattern throughout)
    for i in range(len(boom_top_nodes) - 1):
        if i % 2 == 0:  # Even indices: bottom-left to top-right
            elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
        else:  # Odd indices: top-left to bottom-right
            elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])

    # Support cable from tower (node 0) to rightmost top node
    elements.append([tower_base, boom_top_nodes[-2]])

    return np.array(elements, dtype=int)

def crane_simulation():
    '''Main crane simulation function'''

    print("Designing 30m Crane Structure...")

    # Create geometry
    X, tower_base, boom_top_nodes, boom_bot_nodes = design_crane_geometry()
    C = create_crane_connectivity(tower_base, boom_top_nodes, boom_bot_nodes)

    print(f"Structure created with {X.shape[0]} nodes and {C.shape[0]} elements")

    # Debug: Print connectivity info
    print(f"\nNode 0 (tower_base) position: {X[0]}")
    print(f"boom_top_nodes: {boom_top_nodes}")
    print(f"boom_bot_nodes: {boom_bot_nodes}")
    print(f"\nFirst 10 elements:")
    for i in range(min(10, C.shape[0])):
        print(f"  Element {i}: nodes {C[i,0]} to {C[i,1]}")
    print(f"\nLast 5 elements:")
    for i in range(max(0, C.shape[0]-5), C.shape[0]):
        print(f"  Element {i}: nodes {C[i,0]} to {C[i,1]}")

    # Plot original structure
    plot_truss(X, C, "30m Crane Truss Structure - Original", show_nodes=True)

    # Material and section properties
    E = 200e9  # Steel Young's modulus (Pa)
    A_main = 0.01  # Main structural members cross-sectional area (m�)
    A_secondary = 0.005  # Secondary members cross-sectional area (m�)

    # Boundary conditions
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    # Fix first bottom boom node (leftmost) - fully fixed (pin support)
    bc[boom_bot_nodes[0], :] = True  # Fixed support at first bottom boom node (leftmost)
    # Fix tower base (cable anchor point) - fully fixed
    bc[tower_base, :] = True  # Fixed support at tower base (node 0)


    print(f"\nBoundary conditions:")
    print(f"  Node {tower_base} (tower): fully fixed at {X[tower_base]}")
    print(f"  Node {boom_bot_nodes[0]} (left boom): fully fixed at {X[boom_bot_nodes[0]]}")

    # Loads - simulate load at crane tip
    loads = np.zeros([n_nodes, 2], float)
    tip_load = 5000  # N (50 kN vertical load at tip)
    loads[boom_top_nodes[-1], 1] = -tip_load  # Vertical load at boom tip

    # Self-weight (approximate)
    for node in boom_top_nodes + boom_bot_nodes:
        loads[node, 1] -= 1000  # Approximate self-weight

    print(f"Applied {tip_load/1000:.0f} kN load at crane tip")

    # Prepare for FE analysis
    bc_mask = bc.reshape(1, 2*n_nodes).ravel()
    load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

    # Assembly global stiffness matrix
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)

    for iEl in range(C.shape[0]):
        # Use different cross-sections for different member types
        # Main boom chords (top and bottom)
        num_chord_elements = 2 * (len(boom_top_nodes) - 1)
        if iEl < num_chord_elements:  # Main boom chords
            A = A_main
        else:  # Secondary elements (verticals, diagonals, cables)
            A = A_secondary

        dof = dof_el(C[iEl,0], C[iEl,1])
        k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

    # Solve system
    k_reduced = k_global[~bc_mask][:, ~bc_mask]
    displacements = np.zeros([2*n_nodes], float)
    displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)

    # Reshape displacements
    D = displacements.reshape(n_nodes, 2)

    print(f"Maximum displacement: {np.max(np.abs(D)):.4f} m")
    print(f"Tip deflection: {D[boom_top_nodes[-1], 1]:.4f} m")

    # Plot deformed shape (scaled for visibility)
    scale_factor = 100
    plot_truss(X + D * scale_factor, C,
              f"30m Crane - Deformed Shape (Scale: {scale_factor}x)",
              show_nodes=False)

    # Calculate member forces and stresses
    print("\n=� Member Force Analysis:")
    print("=" * 50)

    max_tension = 0
    max_compression = 0
    element_forces = np.zeros(C.shape[0])
    element_stresses = np.zeros(C.shape[0])
    element_areas = []

    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        d_el = np.concatenate([D[n1], D[n2]])

        # Element properties
        num_chord_elements = 2 * (len(boom_top_nodes) - 1)
        if iEl < num_chord_elements:  # Main boom chords
            A = A_main
            member_type = "Boom Chord"
        else:  # Secondary elements (verticals, diagonals, cables)
            A = A_secondary
            member_type = "Secondary"

        element_areas.append(A)

        # Calculate element force
        k_el = element_stiffness(X[n1], X[n2], A, E)
        f_el = k_el @ d_el

        # Calculate axial force using element vector
        d_vec = X[n2] - X[n1]
        L = np.linalg.norm(d_vec)

        # Skip zero-length elements
        if L < 1e-10:
            element_forces[iEl] = 0
            element_stresses[iEl] = 0
            continue

        # Unit vector
        e = d_vec / L
        # Project forces onto element axis
        # Positive = tension, Negative = compression
        axial_force = -np.dot([f_el[2] - f_el[0], f_el[3] - f_el[1]], e)

        # Calculate stress (force/area)
        stress = axial_force / A

        element_forces[iEl] = axial_force
        element_stresses[iEl] = stress

        if axial_force > max_tension:
            max_tension = axial_force
        if axial_force < max_compression:
            max_compression = axial_force

        if iEl < 10 or abs(axial_force) > 1000:  # Show first 10 elements and significant forces
            print(f"Element {iEl:2d} ({member_type:12s}): {axial_force:8.0f} N")

    print(f"\nMaximum Tension:     {max_tension:8.0f} N")
    print(f"Maximum Compression: {max_compression:8.0f} N")

    # Safety factors and recommendations
    yield_strength = 250e6  # Pa (typical steel)
    safety_factor = 2.5

    max_stress_tension = max_tension / A_main
    max_stress_compression = abs(max_compression) / A_main

    print(f"\n=Structural Analysis Summary:")
    print("=" * 50)
    print(f"Max tensile stress:     {max_stress_tension/1e6:.1f} MPa")
    print(f"Max compressive stress: {max_stress_compression/1e6:.1f} MPa")
    print(f"Allowable stress:       {yield_strength/safety_factor/1e6:.1f} MPa")

    if max_stress_tension < yield_strength/safety_factor and max_stress_compression < yield_strength/safety_factor:
        print("Structure is SAFE under applied loads")
    else:
        print("Structure may be OVERSTRESSED - consider increasing member sizes")

    # Plot stress heatmap
    plot_stress_heatmap(X, C, element_stresses, element_areas,
                       title="Crane Stress Distribution - Color-coded by Stress Level")

    # Analyze cost and security factors
    cost, FS_tension, FS_pandeo, mass = analyze_cost_and_security(
        X, C, element_stresses, element_forces, boom_top_nodes)

    return X, C, D, loads, cost, FS_tension, FS_pandeo, mass

def calculate_cost(m, n_elementos, n_uniones, m0=1000, n_elementos_0=50, n_uniones_0=100):
    '''
    Calculate the cost of the crane structure

    Parameters:
    -----------
    m : float
        Total mass of the structure (kg)
    n_elementos : int
        Number of elements in the structure
    n_uniones : int
        Number of unions/joints in the structure
    m0 : float
        Reference mass (kg), default = 1000 kg
    n_elementos_0 : int
        Reference number of elements, default = 50
    n_uniones_0 : int
        Reference number of unions, default = 100

    Returns:
    --------
    cost : float
        Total cost based on the given formula
    '''
    cost = (m / m0) + 1.5 * (n_elementos / n_elementos_0) + 2 * (n_uniones / n_uniones_0)
    return cost

def calculate_security_factors(sigma_max, P_max, sigma_adm=100e6, P_critica=None):
    '''
    Calculate security factors for the crane structure

    Parameters:
    -----------
    sigma_max : float
        Maximum stress in the structure (Pa)
    P_max : float
        Maximum load/force in the structure (N)
    sigma_adm : float
        Admissible/allowable stress (Pa), default = 100 MPa
    P_critica : float
        Critical buckling load (N), optional

    Returns:
    --------
    FS_tension : float
        Safety factor for tension (σ_adm / σ_max)
    FS_pandeo : float or None
        Safety factor for buckling (P_critica / P_max), if P_critica is provided
    '''
    FS_tension = sigma_adm / abs(sigma_max)
    FS_pandeo = None

    if P_critica is not None:
        FS_pandeo = P_critica / abs(P_max)

    return FS_tension, FS_pandeo

def estimate_structure_mass(X, C, A_main=0.01, A_secondary=0.005, rho_steel=7850):
    '''
    Estimate the total mass of the structure

    Parameters:
    -----------
    X : ndarray
        Node coordinates
    C : ndarray
        Connectivity matrix
    A_main : float
        Cross-sectional area of main members (m²)
    A_secondary : float
        Cross-sectional area of secondary members (m²)
    rho_steel : float
        Density of steel (kg/m³), default = 7850 kg/m³

    Returns:
    --------
    total_mass : float
        Total mass of the structure (kg)
    '''
    total_mass = 0

    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        d_vec = X[n2] - X[n1]
        L = np.linalg.norm(d_vec)

        # Determine cross-section (simplified: first elements are main members)
        if iEl < 24:  # Approximate number of chord elements
            A = A_main
        else:
            A = A_secondary

        # Mass = density * volume = density * area * length
        element_mass = rho_steel * A * L
        total_mass += element_mass

    return total_mass

def analyze_cost_and_security(X, C, element_stresses, element_forces, boom_top_nodes):
    '''
    Analyze cost and security factors for the crane structure

    Parameters:
    -----------
    X : ndarray
        Node coordinates
    C : ndarray
        Connectivity matrix
    element_stresses : ndarray
        Stress in each element (Pa)
    element_forces : ndarray
        Force in each element (N)
    boom_top_nodes : list
        List of top boom node indices

    Returns:
    --------
    cost : float
        Total cost of the structure
    FS_tension : float
        Safety factor for tension
    FS_pandeo : float or None
        Safety factor for buckling
    mass : float
        Total mass of structure (kg)
    '''
    # Calculate structure mass
    mass = estimate_structure_mass(X, C)

    # Number of elements
    n_elementos = C.shape[0]

    # Number of unions (nodes that connect multiple elements)
    # Approximate: each node is a union
    n_uniones = X.shape[0]

    # Calculate cost
    cost = calculate_cost(mass, n_elementos, n_uniones)

    # Find maximum stress and force
    sigma_max = np.max(np.abs(element_stresses))
    P_max = np.max(np.abs(element_forces))

    # Allowable stress (typical structural steel)
    sigma_adm = 100e6  # 100 MPa

    # Estimate critical buckling load (simplified Euler buckling)
    # For the most critical compression member
    E = 200e9  # Steel Young's modulus
    A_main = 0.01
    I_min = A_main * (0.1)**2 / 12  # Simplified moment of inertia for square section

    # Find longest compressed member
    max_compressed_length = 0
    for iEl in range(C.shape[0]):
        if element_forces[iEl] < 0:  # Compression
            n1, n2 = C[iEl]
            L = np.linalg.norm(X[n2] - X[n1])
            if L > max_compressed_length:
                max_compressed_length = L

    # Euler buckling load: P_cr = π²EI/L²
    if max_compressed_length > 0:
        P_critica = (np.pi**2 * E * I_min) / (max_compressed_length**2)
    else:
        P_critica = None

    # Calculate safety factors
    FS_tension, FS_pandeo = calculate_security_factors(sigma_max, P_max, sigma_adm, P_critica)

    # Print results
    print("\n" + "="*70)
    print("COST AND SECURITY FACTOR ANALYSIS")
    print("="*70)
    print(f"\nStructure Properties:")
    print(f"  Total mass:           {mass:.2f} kg")
    print(f"  Number of elements:   {n_elementos}")
    print(f"  Number of unions:     {n_uniones}")
    print(f"\nCost Analysis:")
    print(f"  Total Cost:           {cost:.4f}")
    print(f"\nSecurity Factors:")
    print(f"  Maximum stress:       {sigma_max/1e6:.2f} MPa")
    print(f"  Allowable stress:     {sigma_adm/1e6:.2f} MPa")
    print(f"  FS_tension:           {FS_tension:.2f}")
    if FS_tension > 2:
        print(f"  → SAFE (FS > 2)")
    else:
        print(f"  → WARNING: FS < 2 (UNSAFE)")

    if FS_pandeo is not None:
        print(f"\n  Maximum compression:  {abs(P_max)/1000:.2f} kN")
        print(f"  Critical buckling:    {P_critica/1000:.2f} kN")
        print(f"  FS_pandeo:            {FS_pandeo:.2f}")
        if FS_pandeo > 2:
            print(f"  → SAFE against buckling (FS > 2)")
        else:
            print(f"  → WARNING: Risk of buckling (FS < 2)")

    print("="*70)

    return cost, FS_tension, FS_pandeo, mass

def analyze_moving_load():
    '''
    Analyze crane performance with varying loads at different positions
    Tests loads from 0 to 40000 N at different positions along the bottom chord
    '''
    print("\n" + "="*70)
    print("MOVING LOAD ANALYSIS")
    print("="*70)

    # Create geometry once
    X, tower_base, boom_top_nodes, boom_bot_nodes = design_crane_geometry()
    C = create_crane_connectivity(tower_base, boom_top_nodes, boom_bot_nodes)

    # Material and section properties
    E = 200e9  # Steel Young's modulus (Pa)
    A_main = 0.01  # Main structural members (m²)
    A_secondary = 0.005  # Secondary members (m²)

    # Boundary conditions
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True
    bc[tower_base, :] = True

    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Assembly global stiffness matrix (only once)
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)
    for iEl in range(C.shape[0]):
        num_chord_elements = 2 * (len(boom_top_nodes) - 1)
        if iEl < num_chord_elements:
            A = A_main
        else:
            A = A_secondary

        dof = dof_el(C[iEl,0], C[iEl,1])
        k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    # Test parameters
    load_magnitudes = np.linspace(0, 40000, 5)  # 0 to 40000 N in 5 steps
    test_positions = boom_bot_nodes  # Test at each bottom node

    # Storage for results
    max_deflections = np.zeros((len(load_magnitudes), len(test_positions)))
    max_stresses = np.zeros((len(load_magnitudes), len(test_positions)))
    max_tensions = np.zeros((len(load_magnitudes), len(test_positions)))
    max_compressions = np.zeros((len(load_magnitudes), len(test_positions)))

    print(f"\nTesting {len(load_magnitudes)} load magnitudes at {len(test_positions)} positions...")
    print(f"Load range: 0 to 40000 N")
    print(f"Positions: Along bottom chord (nodes {boom_bot_nodes[0]} to {boom_bot_nodes[-1]})")

    # Iterate through loads and positions
    for i, load_mag in enumerate(load_magnitudes):
        for j, pos_node in enumerate(test_positions):
            # Apply load at current position
            loads = np.zeros([n_nodes, 2], float)
            loads[pos_node, 1] = -load_mag  # Vertical downward load

            # Add self-weight
            for node in boom_top_nodes + boom_bot_nodes:
                loads[node, 1] -= 1000

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

                num_chord_elements = 2 * (len(boom_top_nodes) - 1)
                if iEl < num_chord_elements:
                    A = A_main
                else:
                    A = A_secondary

                k_el = element_stiffness(X[n1], X[n2], A, E)
                f_el = k_el @ d_el

                d_vec = X[n2] - X[n1]
                L = np.linalg.norm(d_vec)

                if L < 1e-10:
                    element_stresses.append(0)
                    continue

                e = d_vec / L
                axial_force = -np.dot([f_el[2] - f_el[0], f_el[3] - f_el[1]], e)
                stress = axial_force / A
                element_stresses.append(stress)

            element_stresses = np.array(element_stresses)
            max_stresses[i, j] = np.max(np.abs(element_stresses))
            max_tensions[i, j] = np.max(element_stresses)
            max_compressions[i, j] = np.min(element_stresses)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Position along crane (x-coordinates of bottom nodes)
    positions_x = X[boom_bot_nodes, 0]

    # Plot 1: Max deflection vs position for different loads
    ax1 = axes[0, 0]
    for i, load_mag in enumerate(load_magnitudes):
        ax1.plot(positions_x, max_deflections[i, :] * 1000, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax1.set_xlabel('Position along crane (m)', fontsize=12)
    ax1.set_ylabel('Maximum deflection (mm)', fontsize=12)
    ax1.set_title('Maximum Deflection vs Load Position', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Max stress vs position for different loads
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
    print(f"  Occurs at: {load_magnitudes[np.unravel_index(np.argmax(max_deflections), max_deflections.shape)[0]]/1000:.1f} kN")
    print(f"  Position:  {positions_x[np.unravel_index(np.argmax(max_deflections), max_deflections.shape)[1]]:.2f} m")
    print(f"\nOverall maximum stress: {np.max(max_stresses)/1e6:.2f} MPa")
    print(f"  Occurs at: {load_magnitudes[np.unravel_index(np.argmax(max_stresses), max_stresses.shape)[0]]/1000:.1f} kN")
    print(f"  Position:  {positions_x[np.unravel_index(np.argmax(max_stresses), max_stresses.shape)[1]]:.2f} m")
    print(f"\nMaximum tensile stress: {np.max(max_tensions)/1e6:.2f} MPa")
    print(f"Maximum compressive stress: {np.min(max_compressions)/1e6:.2f} MPa")
    print("="*70)

    return max_deflections, max_stresses, positions_x

def animate_moving_load(load_magnitude=30000, scale_factor=100, interval=200):
    '''
    Create an animated time-lapse of crane deformation as load moves along bottom chord

    Parameters:
    -----------
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

    # Create geometry
    X, tower_base, boom_top_nodes, boom_bot_nodes = design_crane_geometry()
    C = create_crane_connectivity(tower_base, boom_top_nodes, boom_bot_nodes)

    # Material and section properties
    E = 200e9
    A_main = 0.01
    A_secondary = 0.005

    # Boundary conditions
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True
    bc[tower_base, :] = True

    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Assembly global stiffness matrix
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)
    for iEl in range(C.shape[0]):
        num_chord_elements = 2 * (len(boom_top_nodes) - 1)
        if iEl < num_chord_elements:
            A = A_main
        else:
            A = A_secondary

        dof = dof_el(C[iEl,0], C[iEl,1])
        k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    # Test positions (bottom nodes)
    test_positions = boom_bot_nodes

    # Calculate deformations for each position
    deformations = []
    max_deflections_list = []

    for pos_node in test_positions:
        # Apply load at current position
        loads = np.zeros([n_nodes, 2], float)
        loads[pos_node, 1] = -load_magnitude

        # Add self-weight
        for node in boom_top_nodes + boom_bot_nodes:
            loads[node, 1] -= 1000

        load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

        # Solve
        displacements = np.zeros([2*n_nodes], float)
        displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
        D = displacements.reshape(n_nodes, 2)

        deformations.append(D)
        max_deflections_list.append(np.max(np.abs(D)))

    # Create animation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Initialize plots
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
        # Original structure (gray)
        for iEl in range(C.shape[0]):
            ax1.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                    [X[C[iEl,0],1], X[C[iEl,1],1]],
                    'gray', linewidth=1, alpha=0.3, linestyle='--')

        # Deformed structure (blue)
        for iEl in range(C.shape[0]):
            ax1.plot([X_deformed[C[iEl,0],0], X_deformed[C[iEl,1],0]],
                    [X_deformed[C[iEl,0],1], X_deformed[C[iEl,1],1]],
                    'b-', linewidth=2)

        # Mark load position with red arrow
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
                     f'Load: {load_magnitude/1000:.1f} kN | Max Deflection: {max_deflections_list[frame]*1000:.2f} mm',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        ax1.legend(loc='upper right')

        # Plot 2: Deflection profile
        positions_x = X[boom_bot_nodes, 0]
        ax2.plot(positions_x, np.array(max_deflections_list) * 1000, 'b-o', linewidth=2, markersize=8)
        ax2.plot(X[pos_node, 0], max_deflections_list[frame] * 1000, 'ro', markersize=15)
        ax2.axvline(x=X[pos_node, 0], color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Position along crane (m)', fontsize=12)
        ax2.set_ylabel('Maximum deflection (mm)', fontsize=12)
        ax2.set_title('Deflection Profile', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return []

    # Create animation
    anim = FuncAnimation(fig, update, init_func=init, frames=len(test_positions),
                        interval=interval, blit=False, repeat=True)

    print("Animation created! Close the window to continue...")
    plt.show()

    print("="*70)

    return anim

def optimize_crane_design(crane_length=30.0, max_load=40000, min_FS=2.0):
    '''
    Optimize crane design to minimize cost while maintaining safety requirements

    Design variables:
    - boom_height: Height of the trapezoidal boom at the left end
    - n_boom_segments: Number of segments along the boom
    - A_main: Cross-sectional area of main members
    - A_secondary: Cross-sectional area of secondary members

    Objective:
    - Minimize cost function

    Constraints:
    - FS_tension >= min_FS (safety factor for tension)
    - FS_pandeo >= min_FS (safety factor for buckling)
    - Maximum deflection < crane_length / 200 (deflection limit)
    '''

    print("\n" + "="*70)
    print("CRANE DESIGN OPTIMIZATION")
    print("="*70)
    print(f"Target: Minimize cost while maintaining safety")
    print(f"Crane length: {crane_length} m")
    print(f"Maximum load: {max_load/1000:.1f} kN")
    print(f"Minimum safety factor: {min_FS}")
    print("="*70)

    # Store optimization history
    iteration = [0]
    cost_history = []
    fs_tension_history = []
    fs_pandeo_history = []
    design_history = []

    def evaluate_design(x):
        '''Evaluate a crane design and return cost, safety factors, and max deflection'''
        boom_height, n_segments, A_main, A_secondary = x

        # Round n_segments to integer
        n_segments = int(np.round(n_segments))
        n_segments = max(6, min(20, n_segments))  # Limit to reasonable range

        # Ensure areas are within bounds
        A_main = max(0.005, min(0.020, A_main))
        A_secondary = max(0.002, min(0.010, A_secondary))

        try:
            # Create geometry with current design (simplified - no tower)
            boom_x = np.linspace(0, crane_length, n_segments + 1)
            boom_y_top = np.linspace(boom_height, 0, n_segments + 1)
            boom_y_bot = np.full(n_segments + 1, 0.0)

            total_nodes = 2 * (n_segments + 1)
            X = np.zeros([total_nodes, 2], float)

            node_idx = 0

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

            # Create connectivity (simplified - without tower)
            elements = []

            # Top chord
            for i in range(len(boom_top_nodes) - 1):
                elements.append([boom_top_nodes[i], boom_top_nodes[i+1]])

            # Bottom chord
            for i in range(len(boom_bot_nodes) - 1):
                elements.append([boom_bot_nodes[i], boom_bot_nodes[i+1]])

            # Vertical members
            for i in range(len(boom_top_nodes)):
                elements.append([boom_top_nodes[i], boom_bot_nodes[i]])

            # Diagonal members
            for i in range(len(boom_top_nodes) - 1):
                if i % 2 == 0:
                    elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
                else:
                    elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])

            C = np.array(elements, dtype=int)

            # Material properties
            E = 200e9

            # Boundary conditions (no tower)
            n_nodes = X.shape[0]
            bc = np.full((n_nodes, 2), False)
            bc[boom_bot_nodes[0], :] = True  # Pin support at left bottom
            bc[boom_bot_nodes[-1], 1] = True  # Roller support at right bottom

            bc_mask = bc.reshape(1, 2*n_nodes).ravel()

            # Assembly global stiffness matrix
            k_global = np.zeros([2*n_nodes, 2*n_nodes], float)
            for iEl in range(C.shape[0]):
                num_chord_elements = 2 * (len(boom_top_nodes) - 1)
                if iEl < num_chord_elements:
                    A = A_main
                else:
                    A = A_secondary

                dof = dof_el(C[iEl,0], C[iEl,1])
                k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
                k_global[np.ix_(dof, dof)] += k_elemental

            # Apply load at tip
            loads = np.zeros([n_nodes, 2], float)
            loads[boom_top_nodes[-1], 1] = -max_load

            # Add self-weight
            for node in boom_top_nodes + boom_bot_nodes:
                loads[node, 1] -= 1000

            load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

            # Solve
            k_reduced = k_global[~bc_mask][:, ~bc_mask]

            # Check if matrix is singular
            cond_num = np.linalg.cond(k_reduced)
            if cond_num > 1e10:
                print(f"DEBUG: Singular matrix detected. Condition number: {cond_num}")
                return None, None, None, None, None

            displacements = np.zeros([2*n_nodes], float)
            displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
            D = displacements.reshape(n_nodes, 2)

            max_deflection = np.max(np.abs(D))

            # Calculate stresses and forces
            element_stresses = []
            element_forces = []

            for iEl in range(C.shape[0]):
                n1, n2 = C[iEl]
                d_el = np.concatenate([D[n1], D[n2]])

                num_chord_elements = 2 * (len(boom_top_nodes) - 1)
                if iEl < num_chord_elements:
                    A = A_main
                else:
                    A = A_secondary

                k_el = element_stiffness(X[n1], X[n2], A, E)
                f_el = k_el @ d_el

                d_vec = X[n2] - X[n1]
                L = np.linalg.norm(d_vec)

                if L < 1e-10:
                    element_stresses.append(0)
                    element_forces.append(0)
                    continue

                e = d_vec / L
                axial_force = -np.dot([f_el[2] - f_el[0], f_el[3] - f_el[1]], e)
                stress = axial_force / A

                element_stresses.append(stress)
                element_forces.append(axial_force)

            element_stresses = np.array(element_stresses)
            element_forces = np.array(element_forces)

            # Calculate mass and cost
            mass = estimate_structure_mass(X, C, A_main, A_secondary)
            n_elementos = C.shape[0]
            n_uniones = X.shape[0]
            cost = calculate_cost(mass, n_elementos, n_uniones)

            # Calculate safety factors
            sigma_max = np.max(np.abs(element_stresses))
            P_max = np.max(np.abs(element_forces))
            sigma_adm = 100e6

            # Euler buckling
            I_min = A_main * (0.1)**2 / 12
            max_compressed_length = 0
            for iEl in range(C.shape[0]):
                if element_forces[iEl] < 0:
                    n1, n2 = C[iEl]
                    L = np.linalg.norm(X[n2] - X[n1])
                    if L > max_compressed_length:
                        max_compressed_length = L

            if max_compressed_length > 0:
                P_critica = (np.pi**2 * E * I_min) / (max_compressed_length**2)
            else:
                P_critica = 1e10

            FS_tension, FS_pandeo = calculate_security_factors(sigma_max, P_max, sigma_adm, P_critica)

            return cost, FS_tension, FS_pandeo, max_deflection, (X, C, boom_top_nodes, boom_bot_nodes)

        except Exception as e:
            print(f"DEBUG: Exception in evaluate_design: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None

    def objective(x):
        '''Objective function: minimize cost'''
        iteration[0] += 1
        cost, FS_tension, FS_pandeo, max_deflection, _ = evaluate_design(x)

        if cost is None:
            return 1e10

        # Store history
        cost_history.append(cost)
        fs_tension_history.append(FS_tension)
        fs_pandeo_history.append(FS_pandeo if FS_pandeo is not None else 0)
        design_history.append(x.copy())

        # Add penalties for constraint violations
        penalty = 0

        if FS_tension < min_FS:
            penalty += 1000 * (min_FS - FS_tension)**2

        if FS_pandeo is not None and FS_pandeo < min_FS:
            penalty += 1000 * (min_FS - FS_pandeo)**2

        deflection_limit = crane_length / 200
        if max_deflection > deflection_limit:
            penalty += 1000 * (max_deflection - deflection_limit)**2

        total_cost = cost + penalty

        if iteration[0] % 10 == 0:
            print(f"Iteration {iteration[0]:3d}: Cost={cost:.4f}, FS_t={FS_tension:.2f}, "
                  f"FS_p={FS_pandeo:.2f if FS_pandeo else 0:.2f}, Def={max_deflection*1000:.2f}mm")

        return total_cost

    # Design variable bounds
    # [boom_height, n_segments, A_main, A_secondary]
    bounds = [
        (0.5, 2.5),      # boom_height (m)
        (8, 16),         # n_segments
        (0.005, 0.020),  # A_main (m²) - more conservative range
        (0.002, 0.010)   # A_secondary (m²) - more conservative range
    ]

    # Initial guess (baseline design)
    x0 = [1.0, 12, 0.01, 0.005]

    # Evaluate initial design first to ensure it works
    print("\nEvaluating initial design...")
    init_cost, init_FS_t, init_FS_p, init_def, init_geom = evaluate_design(x0)

    if init_cost is not None:
        print(f"Initial design works: Cost={init_cost:.4f}, FS_t={init_FS_t:.2f}, FS_p={init_FS_p:.2f if init_FS_p else 0:.2f}")
    else:
        print("WARNING: Initial design failed. Cannot proceed with optimization.")
        print("Returning baseline design from gruita2...")
        # Return gruita2 baseline
        return np.array([1.0, 12, 0.01, 0.005]), None, None, None, None

    print("\nStarting optimization...")
    print("Using differential evolution algorithm...")

    # Use differential evolution for global optimization
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=20,
        popsize=8,
        tol=0.01,
        mutation=(0.3, 0.7),  # Smaller mutations for stability
        recombination=0.7,
        seed=42,
        disp=True,
        polish=False,  # Disable polishing to stay within bounds
        init='latinhypercube',
        atol=0.001,
        workers=1
    )

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)

    # Extract optimal design and ensure it's within bounds
    optimal_x = result.x.copy()

    # Clamp to bounds
    optimal_x[0] = max(bounds[0][0], min(bounds[0][1], optimal_x[0]))  # boom_height
    optimal_x[1] = max(bounds[1][0], min(bounds[1][1], optimal_x[1]))  # n_segments
    optimal_x[2] = max(bounds[2][0], min(bounds[2][1], optimal_x[2]))  # A_main
    optimal_x[3] = max(bounds[3][0], min(bounds[3][1], optimal_x[3]))  # A_secondary

    boom_height_opt, n_segments_opt, A_main_opt, A_secondary_opt = optimal_x
    n_segments_opt = int(np.round(n_segments_opt))

    # Evaluate optimal design
    print(f"\nEvaluating optimal design with parameters:")
    print(f"  x = {optimal_x}")
    cost_opt, FS_t_opt, FS_p_opt, def_opt, geometry_opt = evaluate_design(optimal_x)

    print(f"\nOptimal Design Parameters:")
    print(f"  Boom height:           {boom_height_opt:.3f} m")
    print(f"  Number of segments:    {n_segments_opt}")
    print(f"  Main member area:      {A_main_opt*10000:.2f} cm²")
    print(f"  Secondary member area: {A_secondary_opt*10000:.2f} cm²")

    if cost_opt is not None:
        print(f"\nOptimal Performance:")
        print(f"  Total cost:            {cost_opt:.4f}")
        print(f"  FS_tension:            {FS_t_opt:.2f}")
        print(f"  FS_pandeo:             {FS_p_opt:.2f if FS_p_opt else 'N/A'}")
        print(f"  Max deflection:        {def_opt*1000:.2f} mm")
        print(f"  Deflection limit:      {crane_length/200*1000:.2f} mm")
    else:
        print("\nWARNING: Optimal design evaluation failed. Retrying with constrained parameters...")
        # Try with more reasonable values
        optimal_x[2] = min(0.03, max(0.002, optimal_x[2]))  # Constrain A_main
        optimal_x[3] = min(0.015, max(0.001, optimal_x[3]))  # Constrain A_secondary
        cost_opt, FS_t_opt, FS_p_opt, def_opt, geometry_opt = evaluate_design(optimal_x)

        if cost_opt is not None:
            print(f"\nOptimal Performance (adjusted):")
            print(f"  Total cost:            {cost_opt:.4f}")
            print(f"  FS_tension:            {FS_t_opt:.2f}")
            print(f"  FS_pandeo:             {FS_p_opt:.2f if FS_p_opt else 'N/A'}")
            print(f"  Max deflection:        {def_opt*1000:.2f} mm")
        else:
            print("ERROR: Could not evaluate optimal design")

    print("="*70)

    # Plot optimization history
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    iterations = np.arange(1, len(cost_history) + 1)

    ax1 = axes[0, 0]
    ax1.plot(iterations, cost_history, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.set_title('Cost Evolution')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(iterations, fs_tension_history, 'g-', linewidth=2, label='FS_tension')
    ax2.plot(iterations, fs_pandeo_history, 'r-', linewidth=2, label='FS_pandeo')
    ax2.axhline(y=min_FS, color='k', linestyle='--', label=f'Min FS = {min_FS}')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Safety Factor')
    ax2.set_title('Safety Factors Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    boom_heights = [d[0] for d in design_history]
    n_segments_hist = [int(np.round(d[1])) for d in design_history]
    ax3.plot(iterations, boom_heights, 'purple', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Boom Height (m)')
    ax3.set_title('Boom Height Evolution')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    A_main_hist = [d[2]*10000 for d in design_history]
    A_sec_hist = [d[3]*10000 for d in design_history]
    ax4.plot(iterations, A_main_hist, 'b-', linewidth=2, label='Main members')
    ax4.plot(iterations, A_sec_hist, 'r-', linewidth=2, label='Secondary members')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Cross-sectional Area (cm²)')
    ax4.set_title('Member Sizes Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Plot optimal design
    if geometry_opt is not None:
        X_opt, C_opt, boom_top_opt, boom_bot_opt = geometry_opt
        plot_truss(X_opt, C_opt, f"Optimized Crane Design (Cost={cost_opt:.4f})", show_nodes=True)

    return optimal_x, cost_opt, FS_t_opt, FS_p_opt, geometry_opt

if __name__ == "__main__":
    print("="*70)
    print("GRUITA 3.0 - SELF-OPTIMIZING CRANE DESIGN")
    print("="*70)

    # Run optimization to find best design
    optimal_design, cost_opt, FS_t, FS_p, geometry = optimize_crane_design(
        crane_length=30.0,
        max_load=40000,
        min_FS=2.0
    )

    print("\nOptimization complete! The crane has designed itself.")
    print("Close the visualization windows to exit.")