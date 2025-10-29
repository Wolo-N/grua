import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

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
    L : float
        Element length (m)
    '''
    # Calculate element vector and length
    d = n2 - n1  # Element vector from node 1 to node 2
    L = np.linalg.norm(d)  # Element length

    # Check for zero-length elements to avoid division by zero
    if L < 1e-10:
        return np.zeros((4, 4), dtype=float), 0.0

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

    return k_structural, L

def tubular_section_properties(D_outer, thickness):
    '''
    Calculate properties of tubular cross-section

    Parameters:
    -----------
    D_outer : float
        Outer diameter (m)
    thickness : float
        Wall thickness (m)

    Returns:
    --------
    A : float
        Cross-sectional area (m²)
    I : float
        Second moment of area (m⁴)
    '''
    D_inner = D_outer - 2 * thickness
    A = np.pi / 4 * (D_outer**2 - D_inner**2)
    I = np.pi / 64 * (D_outer**4 - D_inner**4)
    return A, I

def euler_buckling_load(E, I, L, end_conditions='pinned-pinned'):
    '''
    Calculate Euler critical buckling load

    Parameters:
    -----------
    E : float
        Young's modulus (Pa)
    I : float
        Second moment of area (m⁴)
    L : float
        Element length (m)
    end_conditions : str
        Boundary conditions ('pinned-pinned', 'fixed-free', 'fixed-fixed', 'fixed-pinned')

    Returns:
    --------
    P_cr : float
        Critical buckling load (N)
    '''
    # Effective length factors
    K_factors = {
        'pinned-pinned': 1.0,
        'fixed-free': 2.0,
        'fixed-fixed': 0.5,
        'fixed-pinned': 0.7
    }

    K = K_factors.get(end_conditions, 1.0)
    L_eff = K * L

    if L_eff < 1e-10:
        return np.inf

    P_cr = (np.pi**2 * E * I) / (L_eff**2)
    return P_cr

def dof_el(nnod1, nnod2):
    '''Returns Elemental DOF for Assembly'''
    return [2*(nnod1+1)-2,2*(nnod1+1)-1,2*(nnod2+1)-2,2*(nnod2+1)-1]

def calculate_cost(mass, n_elements, n_nodes, m0=1000, nelementos0=50, nuniones0=30):
    '''
    Calculate cost function according to TP2 specification

    C(m, n_elementos, n_uniones) = m/m0 + 1.5*(n_elementos/nelementos0) + 2*(n_uniones/nuniones0)

    Parameters:
    -----------
    mass : float
        Total mass of structure (kg)
    n_elements : int
        Number of elements
    n_nodes : int
        Number of nodes (unions)
    m0, nelementos0, nuniones0 : float
        Normalization factors

    Returns:
    --------
    cost : float
        Normalized cost function value
    '''
    cost = (mass / m0) + 1.5 * (n_elements / nelementos0) + 2 * (n_nodes / nuniones0)
    return cost

def plot_truss(X, C, title="Truss Structure", scale_factor=1, show_nodes=True):
    '''Plot truss structure with optional node numbering'''
    plt.figure(figsize=(15, 12))

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

def plot_tension_heatmap(X, C, stresses, title="Tension Stress Distribution"):
    '''Plot truss structure with tension stress heatmap (only positive stresses)'''
    fig, ax = plt.subplots(figsize=(16, 12))

    segments = []
    tension_stresses = []
    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        segments.append([(X[n1, 0], X[n1, 1]), (X[n2, 0], X[n2, 1])])
        tension_stresses.append(max(0, stresses[iEl]))

    tension_stresses = np.array(tension_stresses)
    tension_mpa = tension_stresses / 1e6
    max_tension = np.max(tension_mpa)

    if max_tension > 0:
        norm = Normalize(vmin=0, vmax=max_tension)
        cmap = cm.Reds
    else:
        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.Reds

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4)
    lc.set_array(tension_mpa)
    line = ax.add_collection(lc)

    cbar = fig.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label('Tension Stress (MPa)', rotation=270, labelpad=25)

    ax.scatter(X[:, 0], X[:, 1], c='black', s=30, zorder=5, alpha=0.5)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.autoscale()

    plt.tight_layout()
    plt.show()

def plot_compression_heatmap(X, C, stresses, title="Compression Stress Distribution"):
    '''Plot truss structure with compression stress heatmap (only negative stresses)'''
    fig, ax = plt.subplots(figsize=(16, 12))

    segments = []
    compression_stresses = []
    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        segments.append([(X[n1, 0], X[n1, 1]), (X[n2, 0], X[n2, 1])])
        compression_stresses.append(abs(min(0, stresses[iEl])))

    compression_stresses = np.array(compression_stresses)
    compression_mpa = compression_stresses / 1e6
    max_compression = np.max(compression_mpa)

    if max_compression > 0:
        norm = Normalize(vmin=0, vmax=max_compression)
        cmap = cm.Blues
    else:
        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.Blues

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4)
    lc.set_array(compression_mpa)
    line = ax.add_collection(lc)

    cbar = fig.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label('Compression Stress (MPa)', rotation=270, labelpad=25)

    ax.scatter(X[:, 0], X[:, 1], c='black', s=30, zorder=5, alpha=0.5)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.autoscale()

    plt.tight_layout()
    plt.show()

def plot_deformation_heatmap(X, C, D, title="Deformation Magnitude Distribution"):
    '''Plot truss structure with deformation magnitude heatmap'''
    fig, ax = plt.subplots(figsize=(16, 12))

    segments = []
    deformations = []
    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        segments.append([(X[n1, 0], X[n1, 1]), (X[n2, 0], X[n2, 1])])
        avg_def = (np.linalg.norm(D[n1]) + np.linalg.norm(D[n2])) / 2
        deformations.append(avg_def)

    deformations = np.array(deformations)
    deformation_mm = deformations * 1000
    norm = Normalize(vmin=0, vmax=np.max(deformation_mm))

    cmap = cm.YlOrRd

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4)
    lc.set_array(deformation_mm)
    line = ax.add_collection(lc)

    cbar = fig.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label('Deformation Magnitude (mm)', rotation=270, labelpad=25)

    ax.scatter(X[:, 0], X[:, 1], c='black', s=30, zorder=5, alpha=0.5)
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.autoscale()

    plt.tight_layout()
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
    fig, ax = plt.subplots(figsize=(16, 12))

    # Create line segments for each element
    segments = []
    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        segments.append([(X[n1, 0], X[n1, 1]), (X[n2, 0], X[n2, 1])])

    # Normalize stress values for colormap
    stress_mpa = stresses / 1e6  # Convert to MPa

    # Filter out nan/inf values for normalization
    valid_stress = stress_mpa[np.isfinite(stress_mpa)]

    if len(valid_stress) == 0:
        print("Warning: No valid stress values to plot")
        return

    # Use symmetric scale centered at zero for better visualization
    # Option 1: Use max absolute value for symmetric scale
    max_abs_stress = np.max(np.abs(valid_stress))

    # Option 2: Use percentiles to avoid extreme outliers (uncomment to use)
    # percentile = 95
    # max_abs_stress = np.percentile(np.abs(valid_stress), percentile)

    norm = Normalize(vmin=-max_abs_stress, vmax=max_abs_stress)

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

    # Print stress statistics (using valid stresses only)
    print("\n" + "="*60)
    print("STRESS ANALYSIS SUMMARY")
    print("="*60)
    print(f"Maximum Tensile Stress:     {np.max(valid_stress):8.2f} MPa")
    print(f"Maximum Compressive Stress: {np.min(valid_stress):8.2f} MPa")
    print(f"Average Stress Magnitude:   {np.mean(np.abs(valid_stress)):8.2f} MPa")
    print(f"Color scale range:          ±{max_abs_stress:.2f} MPa")
    print("="*60)

def design_tower_crane_geometry():
    '''
    Design tower crane with vertical mast, horizontal boom, and counterweight

    Returns:
    --------
    X : ndarray
        Node coordinates
    tower_base_nodes : list
        Ground anchor nodes
    tower_nodes : list
        Vertical tower nodes
    jib_top_nodes : list
        Top chord of jib (boom) nodes
    jib_bot_nodes : list
        Bottom chord of jib nodes
    counterwt_top_nodes : list
        Top chord of counterweight arm nodes
    counterwt_bot_nodes : list
        Bottom chord of counterweight arm nodes
    '''

    # Design parameters
    tower_height = 40.0      # m - vertical mast height
    jib_length = 30.0        # m - main boom length
    counterwt_length = 10.0  # m - counterweight arm length
    truss_depth = 1.0        # m - depth of horizontal trusses
    crane_drop = 5.0         # m - how far below tower top the crane arms sit

    n_tower_segments = 8     # Number of segments in tower
    n_jib_segments = 10      # Number of segments in jib
    n_counterwt_segments = 4 # Number of segments in counterweight arm

    # Tower base width for stability
    base_width = 2.0  # m

    node_list = []
    node_idx = 0

    # Ground base nodes (4 corners for stability)
    ground_base_nodes = []
    base_positions = [
        [base_width/2, 0],
        [-base_width/2, 0]
    ]
    for pos in base_positions:  # Only use 2 for simplicity
        node_list.append(pos)
        ground_base_nodes.append(node_idx)
        node_idx += 1

    # Vertical tower nodes (creating a truss tower)
    tower_heights = np.linspace(0, tower_height, n_tower_segments + 1)
    tower_left_nodes = []
    tower_right_nodes = []
    tower_width = 2.0  # m - tower width

    for h in tower_heights:
        # Left side of tower
        node_list.append([-tower_width/2, h])
        tower_left_nodes.append(node_idx)
        node_idx += 1

        # Right side of tower
        node_list.append([tower_width/2, h])
        tower_right_nodes.append(node_idx)
        node_idx += 1

    # Top of tower - connection point for boom
    tower_top_y = tower_height
    tower_center = 0.0
    crane_height = tower_height - crane_drop  # Height where crane arms are positioned

    # JIB (boom) - extends horizontally from tower, below tower top
    jib_x = np.linspace(tower_center, jib_length, n_jib_segments + 1)
    jib_top_nodes = []
    jib_bot_nodes = []

    for x in jib_x:
        # Top chord
        node_list.append([x, crane_height + truss_depth])
        jib_top_nodes.append(node_idx)
        node_idx += 1

        # Bottom chord
        node_list.append([x, crane_height])
        jib_bot_nodes.append(node_idx)
        node_idx += 1

    # COUNTERWEIGHT ARM - extends in opposite direction
    counterwt_x = np.linspace(tower_center, -counterwt_length, n_counterwt_segments + 1)
    counterwt_top_nodes = []
    counterwt_bot_nodes = []

    for x in counterwt_x:
        # Top chord
        node_list.append([x, crane_height + truss_depth])
        counterwt_top_nodes.append(node_idx)
        node_idx += 1

        # Bottom chord
        node_list.append([x, crane_height])
        counterwt_bot_nodes.append(node_idx)
        node_idx += 1

    X = np.array(node_list, dtype=float)

    return (X, ground_base_nodes, tower_left_nodes, tower_right_nodes,
            jib_top_nodes, jib_bot_nodes, counterwt_top_nodes, counterwt_bot_nodes)

def create_tower_crane_connectivity(ground_base_nodes, tower_left_nodes, tower_right_nodes,
                                   jib_top_nodes, jib_bot_nodes,
                                   counterwt_top_nodes, counterwt_bot_nodes):
    '''Create element connectivity matrix for tower crane'''

    elements = []

    # Ground anchors to tower base
    elements.append([ground_base_nodes[0], tower_left_nodes[0]])
    elements.append([ground_base_nodes[1], tower_right_nodes[0]])
    elements.append([ground_base_nodes[0], tower_right_nodes[0]])
    elements.append([ground_base_nodes[1], tower_left_nodes[0]])

    # Vertical tower structure
    for i in range(len(tower_left_nodes) - 1):
        # Left vertical members
        elements.append([tower_left_nodes[i], tower_left_nodes[i+1]])
        # Right vertical members
        elements.append([tower_right_nodes[i], tower_right_nodes[i+1]])
        # Horizontal cross-bracing
        elements.append([tower_left_nodes[i], tower_right_nodes[i]])
        elements.append([tower_left_nodes[i+1], tower_right_nodes[i+1]])
        # Diagonal bracing (X-pattern)
        elements.append([tower_left_nodes[i], tower_right_nodes[i+1]])
        elements.append([tower_right_nodes[i], tower_left_nodes[i+1]])

    # Tower top nodes for cable connections
    tower_top_left = tower_left_nodes[-1]
    tower_top_right = tower_right_nodes[-1]

    # Rigidly connect horizontal arms to tower at crane height level
    # Find tower node closest to crane height (32m with 8 segments means node at 30m or 35m)
    # With 8 segments from 0-40m, we have nodes at: 0, 5, 10, 15, 20, 25, 30, 35, 40m
    # Crane arms are at 32m, so connect to tower nodes around that height
    tower_connection_idx = len(tower_left_nodes) - 2  # Second from top (35m)

    # Connect horizontal arm center to tower at crane height
    elements.append([tower_left_nodes[tower_connection_idx], jib_bot_nodes[0]])
    elements.append([tower_right_nodes[tower_connection_idx], jib_bot_nodes[0]])
    elements.append([tower_left_nodes[tower_connection_idx], jib_top_nodes[0]])
    elements.append([tower_right_nodes[tower_connection_idx], jib_top_nodes[0]])

    elements.append([tower_left_nodes[tower_connection_idx], counterwt_bot_nodes[0]])
    elements.append([tower_right_nodes[tower_connection_idx], counterwt_bot_nodes[0]])
    elements.append([tower_left_nodes[tower_connection_idx], counterwt_top_nodes[0]])
    elements.append([tower_right_nodes[tower_connection_idx], counterwt_top_nodes[0]])

    # SUPPORT CABLES from tower top to jib (multiple points for support)
    # Adjust these lists to control number of cables:
    # More cables = more support but more complex
    # Fewer cables = simpler but less support
    jib_cable_points = [len(jib_top_nodes)//2, -1]  # Only cable to tip: [-1]
                              # Two cables: [len(jib_top_nodes)//2, -1]
                              # Three cables: [len(jib_top_nodes)//3, 2*len(jib_top_nodes)//3, -1]

    for cable_idx in jib_cable_points:
        elements.append([tower_top_right, jib_top_nodes[cable_idx]])


    # SUPPORT CABLES from tower top to counterweight arm
    counterwt_cable_points = [-1]  # Only cable to tip: [-1]
                                    # Two cables: [len(counterwt_top_nodes)//2, -1]

    for cable_idx in counterwt_cable_points:
        elements.append([tower_top_left, counterwt_top_nodes[cable_idx]])

    # JIB top chord
    for i in range(len(jib_top_nodes) - 1):
        elements.append([jib_top_nodes[i], jib_top_nodes[i+1]])

    # JIB bottom chord
    for i in range(len(jib_bot_nodes) - 1):
        elements.append([jib_bot_nodes[i], jib_bot_nodes[i+1]])

    # JIB vertical members
    for i in range(len(jib_top_nodes)):
        elements.append([jib_top_nodes[i], jib_bot_nodes[i]])

    # JIB diagonal bracing
    for i in range(len(jib_top_nodes) - 1):
        if i % 2 == 0:
            elements.append([jib_top_nodes[i], jib_bot_nodes[i+1]])
        else:
            elements.append([jib_bot_nodes[i], jib_top_nodes[i+1]])

    # COUNTERWEIGHT ARM top chord
    for i in range(len(counterwt_top_nodes) - 1):
        elements.append([counterwt_top_nodes[i], counterwt_top_nodes[i+1]])

    # COUNTERWEIGHT ARM bottom chord
    for i in range(len(counterwt_bot_nodes) - 1):
        elements.append([counterwt_bot_nodes[i], counterwt_bot_nodes[i+1]])

    # COUNTERWEIGHT ARM vertical members
    for i in range(len(counterwt_top_nodes)):
        elements.append([counterwt_top_nodes[i], counterwt_bot_nodes[i]])

    # COUNTERWEIGHT ARM diagonal bracing
    for i in range(len(counterwt_top_nodes) - 1):
        if i % 2 == 0:
            elements.append([counterwt_top_nodes[i], counterwt_bot_nodes[i+1]])
        else:
            elements.append([counterwt_bot_nodes[i], counterwt_top_nodes[i+1]])

    return np.array(elements, dtype=int)

def tower_crane_simulation():
    '''Main tower crane simulation function'''

    print("="*70)
    print("TOWER CRANE WITH COUNTERWEIGHT - STRUCTURAL ANALYSIS")
    print("="*70)

    # Create geometry
    (X, ground_base_nodes, tower_left_nodes, tower_right_nodes,
     jib_top_nodes, jib_bot_nodes, counterwt_top_nodes, counterwt_bot_nodes) = design_tower_crane_geometry()

    C = create_tower_crane_connectivity(ground_base_nodes, tower_left_nodes, tower_right_nodes,
                                       jib_top_nodes, jib_bot_nodes,
                                       counterwt_top_nodes, counterwt_bot_nodes)

    print(f"Structure created with {X.shape[0]} nodes and {C.shape[0]} elements")
    print(f"Tower height: 40m")
    print(f"Jib length: 30m")
    print(f"Counterweight arm: 10m")
    print(f"Crane arms positioned 8m below tower top")
    print(f"Support cables connect tower top to crane arms")

    # Plot original structure
    plot_truss(X, C, "Tower Crane with Counterweight - Original Structure", show_nodes=True)

    # Material and section properties - TP2 SPECIFICATIONS
    E = 200e9  # Steel Young's modulus (Pa) - TP2: 200 GPa
    rho = 7800  # Steel density (kg/m³) - TP2: 7800 kg/m³
    sigma_adm = 250e6  # Admissible stress (Pa) - TP2: 250 MPa

    # Tubular sections - TP2: D_external_max = 50mm
    D_outer_tower = 0.050  # 50mm outer diameter for tower
    D_outer_main = 0.040   # 40mm outer diameter for main members
    D_outer_secondary = 0.030  # 30mm outer diameter for secondary members
    D_outer_cable = 0.020  # 20mm outer diameter for cables
    thickness = 0.005  # 5mm wall thickness

    # Calculate cross-sectional areas and moments of inertia
    A_tower, I_tower = tubular_section_properties(D_outer_tower, thickness)
    A_main, I_main = tubular_section_properties(D_outer_main, thickness)
    A_secondary, I_secondary = tubular_section_properties(D_outer_secondary, thickness)
    A_cable, I_cable = tubular_section_properties(D_outer_cable, thickness)

    # Boundary conditions - fix ground base nodes and tower base nodes
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    for node in ground_base_nodes:
        bc[node, :] = True  # Fixed support at ground
    # Also fix the tower base nodes (first node of each tower side)
    bc[tower_left_nodes[0], :] = True  # Fix tower left base
    bc[tower_right_nodes[0], :] = True  # Fix tower right base

    # Loads
    loads = np.zeros([n_nodes, 2], float)

    # Load at jib tip (crane lifting load)
    # In a real crane, the load hangs from the bottom chord (trolley with hook)
    jib_load = 2000  # N (100 kN)
    loads[jib_bot_nodes[-1], 1] = -jib_load  # Load hangs from bottom chord only

    # Counterweight (simulates heavy concrete blocks)
    # Counterweight sits on top of the counterweight arm platform
    counterweight = 150000  # N (150 kN) - balances the jib load
    loads[counterwt_bot_nodes[-1], 1] = -counterweight  # Weight on bottom chord

    # Self-weight will be calculated during assembly based on element properties

    print(f"\nApplied jib load: {jib_load/1000:.0f} kN")
    print(f"Counterweight: {counterweight/1000:.0f} kN")

    # Prepare for FE analysis
    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Assembly global stiffness matrix and calculate self-weight
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)

    n_tower_elements = len(tower_left_nodes) - 1
    element_type_list = []
    element_lengths = np.zeros(C.shape[0])
    element_masses = np.zeros(C.shape[0])

    # Get tower top nodes for cable identification
    tower_top_left = tower_left_nodes[-1]
    tower_top_right = tower_right_nodes[-1]

    for iEl in range(C.shape[0]):
        # Determine element type for cross-section assignment
        n1, n2 = C[iEl]

        # Tower elements get largest cross-section
        if n1 in tower_left_nodes + tower_right_nodes and n2 in tower_left_nodes + tower_right_nodes:
            A = A_tower
            element_type_list.append('tower')
        # Main chords
        elif ((n1 in jib_top_nodes and n2 in jib_top_nodes) or
              (n1 in jib_bot_nodes and n2 in jib_bot_nodes) or
              (n1 in counterwt_top_nodes and n2 in counterwt_top_nodes) or
              (n1 in counterwt_bot_nodes and n2 in counterwt_bot_nodes)):
            A = A_main
            element_type_list.append('main')
        # Support cables from tower top to crane arms
        elif ((n1 in [tower_top_left, tower_top_right] and n2 in jib_top_nodes + counterwt_top_nodes) or
              (n2 in [tower_top_left, tower_top_right] and n1 in jib_top_nodes + counterwt_top_nodes)):
            A = A_cable
            element_type_list.append('cable')
        else:
            A = A_secondary
            element_type_list.append('secondary')

        dof = dof_el(C[iEl,0], C[iEl,1])
        k_elemental, L = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

        # Store element length and calculate mass
        element_lengths[iEl] = L
        element_masses[iEl] = A * L * rho  # mass = volume * density

    # Add self-weight to loads (distribute element mass to nodes)
    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        element_weight = element_masses[iEl] * 9.81  # Weight in N
        # Distribute half to each node
        loads[n1, 1] -= element_weight / 2
        loads[n2, 1] -= element_weight / 2

    # Total structure mass
    total_mass = np.sum(element_masses)
    print(f"Total structure mass: {total_mass:.2f} kg")

    # Now update load vector with self-weight included
    load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

    # Solve system
    k_reduced = k_global[~bc_mask][:, ~bc_mask]
    displacements = np.zeros([2*n_nodes], float)
    displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)

    # Reshape displacements
    D = displacements.reshape(n_nodes, 2)

    print(f"\nMaximum displacement: {np.max(np.abs(D)):.4f} m")
    print(f"Jib tip deflection: {D[jib_bot_nodes[-1], 1]:.4f} m")
    print(f"Tower top deflection: {np.max(np.abs(D[tower_left_nodes[-1]])):.4f} m")

    # Debug: Check for any extreme displacements
    if np.max(np.abs(D)) > 100:
        print("\n⚠ WARNING: Displacements are extremely large!")
        print("This might indicate a structural instability or calculation error.")

    # Calculate member forces and stresses
    print("\n" + "="*70)
    print("MEMBER FORCE ANALYSIS")
    print("="*70)

    max_tension = 0
    max_compression = 0
    element_forces = np.zeros(C.shape[0])
    element_stresses = np.zeros(C.shape[0])
    element_areas = []

    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        d_el = np.concatenate([D[n1], D[n2]])

        # Get element cross-section
        if element_type_list[iEl] == 'tower':
            A = A_tower
        elif element_type_list[iEl] == 'main':
            A = A_main
        elif element_type_list[iEl] == 'cable':
            A = A_cable
        else:
            A = A_secondary

        element_areas.append(A)

        # Calculate element force
        k_el, L = element_stiffness(X[n1], X[n2], A, E)
        f_el = k_el @ d_el

        # Calculate axial force
        d_vec = X[n2] - X[n1]
        e = d_vec / L
        axial_force = -np.dot([f_el[2] - f_el[0], f_el[3] - f_el[1]], e)

        # Calculate stress
        stress = axial_force / A

        element_forces[iEl] = axial_force
        element_stresses[iEl] = stress

        if axial_force > max_tension:
            max_tension = axial_force
        if axial_force < max_compression:
            max_compression = axial_force

    print(f"\nMaximum Tension:     {max_tension/1000:8.1f} kN")
    print(f"Maximum Compression: {max_compression/1000:8.1f} kN")

    # Safety analysis - TP2 Requirements: FS > 2
    print(f"\n" + "="*70)
    print("STRUCTURAL SAFETY ANALYSIS - TP2")
    print("="*70)

    # Calculate safety factors for all elements
    tension_safety_factors = np.zeros(C.shape[0])
    buckling_safety_factors = np.zeros(C.shape[0])

    for iEl in range(C.shape[0]):
        stress = element_stresses[iEl]
        force = element_forces[iEl]
        L = element_lengths[iEl]

        # Determine element type to get correct I value
        element_type = element_type_list[iEl]
        if element_type == 'tower':
            I = I_tower
            A_elem = A_tower
        elif element_type == 'main':
            I = I_main
            A_elem = A_main
        elif element_type == 'cable':
            I = I_cable
            A_elem = A_cable
        else:
            I = I_secondary
            A_elem = A_secondary

        # Tension safety factor: FS_tension = sigma_adm / sigma_max
        if stress > 0:  # Tension
            tension_safety_factors[iEl] = sigma_adm / stress
        else:  # Compression (also check against sigma_adm)
            tension_safety_factors[iEl] = sigma_adm / abs(stress)

        # Buckling safety factor: FS_buckling = P_critical / P_max (only for compression)
        if force < 0:  # Compression
            P_cr = euler_buckling_load(E, I, L, end_conditions='pinned-pinned')
            buckling_safety_factors[iEl] = P_cr / abs(force)
        else:
            buckling_safety_factors[iEl] = np.inf  # No buckling in tension

    # Find minimum safety factors
    min_tension_sf = np.min(tension_safety_factors)
    min_buckling_sf = np.min(buckling_safety_factors[buckling_safety_factors < np.inf])
    min_overall_sf = min(min_tension_sf, min_buckling_sf)

    print(f"Admissible stress (σ_adm):     {sigma_adm/1e6:.1f} MPa")
    print(f"Max tensile stress:            {np.max(element_stresses)/1e6:.1f} MPa")
    print(f"Max compressive stress:        {abs(np.min(element_stresses))/1e6:.1f} MPa")
    print(f"\nSafety Factors:")
    print(f"  Minimum tension SF:          {min_tension_sf:.2f}")
    print(f"  Minimum buckling SF:         {min_buckling_sf:.2f}")
    print(f"  Minimum overall SF:          {min_overall_sf:.2f}")
    print(f"  Required SF (TP2):           2.00")

    if min_overall_sf >= 2.0:
        print("\n✓ Structure PASSES safety requirements (FS > 2)")
    else:
        print("\n✗ WARNING: Structure FAILS safety requirements (FS < 2)")
        print("  Recommendation: Increase member sizes or reduce loads")

    # Calculate cost function
    cost = calculate_cost(total_mass, C.shape[0], X.shape[0])
    print(f"\nCost Function (TP2):")
    print(f"  C = {cost:.3f}")

    # Plot stress heatmap on deformed shape
    scale_factor = 50
    X_deformed = X + D * scale_factor
    plot_stress_heatmap(X_deformed, C, element_stresses, element_areas,
                       title=f"Tower Crane Stress Distribution - Deformed Shape (Scale: {scale_factor}x)")

    # Plot separate tension, compression, and deformation heatmaps
    print("\nGenerating detailed stress and deformation visualizations...")
    plot_tension_heatmap(X, C, element_stresses,
                        title="Tower Crane - Tension Stress Distribution")
    plot_compression_heatmap(X, C, element_stresses,
                            title="Tower Crane - Compression Stress Distribution")
    plot_deformation_heatmap(X, C, D,
                            title="Tower Crane - Deformation Magnitude Distribution")

    print("\n" + "="*70)

    return X, C, D, loads

if __name__ == "__main__":
    tower_crane_simulation()
