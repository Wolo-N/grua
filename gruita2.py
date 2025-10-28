import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import time

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

def hollow_circular_section(d_outer, d_inner):
    '''
    Calculate cross-sectional area and moment of inertia for hollow circular section

    Parameters:
    -----------
    d_outer : float
        Outer diameter (m)
    d_inner : float
        Inner diameter (m)

    Returns:
    --------
    A : float
        Cross-sectional area (m²)
    I : float
        Second moment of area (m⁴)
    '''
    r_outer = d_outer / 2
    r_inner = d_inner / 2

    # Cross-sectional area: A = π(r_outer² - r_inner²)
    A = np.pi * (r_outer**2 - r_inner**2)

    # Second moment of area: I = π/4 * (r_outer⁴ - r_inner⁴)
    I = np.pi / 4 * (r_outer**4 - r_inner**4)

    return A, I

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

    # STRUTS (compression members - rigid structural elements)

    # Top strut elements (horizontal top members)
    for i in range(len(boom_top_nodes) - 1):
        elements.append([boom_top_nodes[i], boom_top_nodes[i+1]])

    # Bottom strut elements (horizontal bottom members)
    for i in range(len(boom_bot_nodes) - 1):
        elements.append([boom_bot_nodes[i], boom_bot_nodes[i+1]])

    # Vertical struts (connect each top node to corresponding bottom node)
    # Both arrays have the same length, so direct 1-to-1 mapping
    for i in range(len(boom_top_nodes)):
        elements.append([boom_top_nodes[i], boom_bot_nodes[i]])

    # CABLES (tension members - flexible cables)

    # Diagonal cables (alternating pattern throughout)
    for i in range(len(boom_top_nodes) - 1):
        if i % 2 == 0:  # Even indices: bottom-left to top-right
            elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
        else:  # Odd indices: top-left to bottom-right
            elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])

    # Support cables from tower (node 0) to boom
    elements.append([tower_base, boom_top_nodes[-2]])
    elements.append([tower_base, boom_top_nodes[-8]])  # Second cable for stability

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

    # Hollow circular cross-section specifications (in meters)
    # Struts (compression members: top/bottom/vertical)
    d_outer_strut = 0.050  # 50 mm outer diameter (max allowed)
    d_inner_strut = 0.040  # 40 mm inner diameter (5 mm wall thickness)
    A_strut, I_strut = hollow_circular_section(d_outer_strut, d_inner_strut)

    # Cables (tension members: diagonals/supports)
    d_outer_cable = 0.050  # 25 mm outer diameter
    d_inner_cable = 0.000  # 20 mm inner diameter (2.5 mm wall thickness)
    A_cable, I_cable = hollow_circular_section(d_outer_cable, d_inner_cable)

    print(f"\nCross-section properties:")
    print(f"  Struts: D_outer={d_outer_strut*1000:.1f}mm, D_inner={d_inner_strut*1000:.1f}mm, A={A_strut*1e4:.2f}cm², I={I_strut*1e8:.2f}cm⁴")
    print(f"  Cables: D_outer={d_outer_cable*1000:.1f}mm, D_inner={d_inner_cable*1000:.1f}mm, A={A_cable*1e4:.2f}cm², I={I_cable*1e8:.2f}cm⁴")

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
    tip_load = 5 # N (50 kN vertical load at tip)
    loads[boom_top_nodes[-1], 1] = -tip_load  # Vertical load at boom tip

    # Calculate self-weight based on element mass: Weight = ρ × V × g
    # Distribute each element's weight equally to its two nodes
    rho_steel = 7850  # kg/m³ - density of steel
    g = 9.81  # m/s² - gravitational acceleration

    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]

        # Element length
        L = np.linalg.norm(X[n2] - X[n1])

        # Element cross-sectional area (struts vs cables)
        # Struts: top + bottom + vertical = 2*(n-1) + n = 3n - 2
        num_strut_elements = 3 * len(boom_top_nodes) - 2
        A = A_strut if iEl < num_strut_elements else A_cable

        # Calculate element weight
        volume = A * L  # m³
        mass = rho_steel * volume  # kg
        weight = mass * g  # N

        # Distribute weight equally to both nodes
        loads[n1, 1] -= weight / 2
        loads[n2, 1] -= weight / 2

    print(f"Applied {tip_load/1000:.0f} kN load at crane tip")

    # Prepare for FE analysis
    bc_mask = bc.reshape(1, 2*n_nodes).ravel()
    load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

    # Assembly global stiffness matrix
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)

    for iEl in range(C.shape[0]):
        # Use different cross-sections for struts vs cables
        # Struts: top + bottom + vertical = 3n - 2
        num_strut_elements = 3 * len(boom_top_nodes) - 2
        if iEl < num_strut_elements:  # Struts (rigid compression members)
            A = A_strut
        else:  # Cables (tension members: diagonals + support cables)
            A = A_cable

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
        num_strut_elements = 3 * len(boom_top_nodes) - 2
        if iEl < num_strut_elements:  # Struts (rigid compression members)
            A = A_strut
            member_type = "Strut"
        else:  # Cables (tension members)
            A = A_cable
            member_type = "Cable"

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

    max_stress_tension = max_tension / A_strut
    max_stress_compression = abs(max_compression) / A_strut

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

    # Estimate critical buckling load (Euler buckling)
    # For the most critical compression member
    E = 200e9  # Steel Young's modulus

    # Hollow circular cross-section specifications
    d_outer_strut = 0.050  # 50 mm
    d_inner_strut = 0.046  # 46 mm
    A_strut, I_strut = hollow_circular_section(d_outer_strut, d_inner_strut)

    d_outer_cable = 0.020  # 20 mm
    d_inner_cable = 0.016  # 16 mm
    A_cable, I_cable = hollow_circular_section(d_outer_cable, d_inner_cable)

    # Find longest compressed member and its moment of inertia
    max_compressed_length = 0
    I_critical = I_strut  # Default to strut inertia
    for iEl in range(C.shape[0]):
        if element_forces[iEl] < 0:  # Compression
            n1, n2 = C[iEl]
            L = np.linalg.norm(X[n2] - X[n1])
            if L > max_compressed_length:
                max_compressed_length = L
                # Determine if this is a strut or cable
                num_strut_elements = 3 * len(boom_top_nodes) - 2
                I_critical = I_strut if iEl < num_strut_elements else I_cable

    # Euler buckling load: P_cr = π²EI/L²
    if max_compressed_length > 0:
        P_critica = (np.pi**2 * E * I_critical) / (max_compressed_length**2)
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

    # Hollow circular cross-section specifications (in meters)
    d_outer_strut = 0.050  # 50 mm outer diameter
    d_inner_strut = 0.046  # 46 mm inner diameter
    A_strut, I_strut = hollow_circular_section(d_outer_strut, d_inner_strut)

    d_outer_cable = 0.020  # 20 mm outer diameter
    d_inner_cable = 0.016  # 16 mm inner diameter
    A_cable, I_cable = hollow_circular_section(d_outer_cable, d_inner_cable)

    # Boundary conditions
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True
    bc[tower_base, :] = True

    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Assembly global stiffness matrix (only once)
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)
    for iEl in range(C.shape[0]):
        num_strut_elements = 3 * len(boom_top_nodes) - 2
        if iEl < num_strut_elements:
            A = A_strut
        else:
            A = A_cable

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

            # Calculate self-weight based on element mass: Weight = ρ × V × g
            # Distribute each element's weight equally to its two nodes
            rho_steel = 7850  # kg/m³ - density of steel
            g = 9.81  # m/s² - gravitational acceleration

            for iEl in range(C.shape[0]):
                n1, n2 = C[iEl]
                L = np.linalg.norm(X[n2] - X[n1])
                num_strut_elements = 3 * len(boom_top_nodes) - 2
                A = A_strut if iEl < num_strut_elements else A_cable
                volume = A * L  # m³
                mass = rho_steel * volume  # kg
                weight = mass * g  # N
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

                num_strut_elements = 3 * len(boom_top_nodes) - 2
                if iEl < num_strut_elements:
                    A = A_strut
                else:
                    A = A_cable

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

    # Hollow circular cross-section specifications (in meters)
    d_outer_strut = 0.050  # 50 mm outer diameter
    d_inner_strut = 0.046  # 46 mm inner diameter
    A_strut, I_strut = hollow_circular_section(d_outer_strut, d_inner_strut)

    d_outer_cable = 0.020  # 20 mm outer diameter
    d_inner_cable = 0.016  # 16 mm inner diameter
    A_cable, I_cable = hollow_circular_section(d_outer_cable, d_inner_cable)

    # Boundary conditions
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True
    bc[tower_base, :] = True

    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Assembly global stiffness matrix
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)
    for iEl in range(C.shape[0]):
        num_strut_elements = 3 * len(boom_top_nodes) - 2
        if iEl < num_strut_elements:
            A = A_strut
        else:
            A = A_cable

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

        # Calculate self-weight properly based on element mass
        # Distribute each element's weight equally to its two nodes
        rho_steel = 7850  # kg/m³
        g = 9.81  # m/s²

        for iEl in range(C.shape[0]):
            n1, n2 = C[iEl]

            # Element length
            L = np.linalg.norm(X[n2] - X[n1])

            # Element cross-section
            num_strut_elements = 3 * len(boom_top_nodes) - 2
            A = A_strut if iEl < num_strut_elements else A_cable

            # Element volume and mass
            volume = A * L  # m³
            mass = rho_steel * volume  # kg
            weight = mass * g  # N

            # Distribute weight equally to both nodes
            loads[n1, 1] -= weight / 2
            loads[n2, 1] -= weight / 2

        load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

        # Solve
        displacements = np.zeros([2*n_nodes], float)
        displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
        D = displacements.reshape(n_nodes, 2)

        deformations.append(D)
        # Calculate deflection at crane tip (last top node)
        tip_deflection = abs(D[boom_top_nodes[-1], 1])  # Vertical displacement at tip
        max_deflections_list.append(tip_deflection)

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

    # Create animation
    anim = FuncAnimation(fig, update, init_func=init, frames=len(test_positions),
                        interval=interval, blit=False, repeat=True)

    print("Animation created! Close the window to continue...")
    plt.show()

    print("="*70)

    return anim

if __name__ == "__main__":
    # Run basic crane simulation
    crane_simulation()

    # Run moving load analysis
    analyze_moving_load()

    # # Run animated moving load visualization
    # animate_moving_load(load_magnitude=0, scale_factor=100, interval=200)