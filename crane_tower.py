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
    truss_depth = 3.0        # m - depth of horizontal trusses
    crane_drop = 8.0         # m - how far below tower top the crane arms sit

    n_tower_segments = 8     # Number of segments in tower
    n_jib_segments = 10      # Number of segments in jib
    n_counterwt_segments = 4 # Number of segments in counterweight arm

    # Tower base width for stability
    base_width = 4.0  # m

    node_list = []
    node_idx = 0

    # Ground base nodes (4 corners for stability)
    ground_base_nodes = []
    base_positions = [
        [-base_width/2, 0],
        [base_width/2, 0],
        [base_width/2, 0],
        [-base_width/2, 0]
    ]
    for pos in base_positions[:2]:  # Only use 2 for simplicity
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

    # Connect tower to crane arms at base (near tower)
    elements.append([tower_top_left, jib_bot_nodes[0]])
    elements.append([tower_top_right, jib_bot_nodes[0]])
    elements.append([tower_top_left, jib_top_nodes[0]])
    elements.append([tower_top_right, jib_top_nodes[0]])

    elements.append([tower_top_left, counterwt_bot_nodes[0]])
    elements.append([tower_top_right, counterwt_bot_nodes[0]])
    elements.append([tower_top_left, counterwt_top_nodes[0]])
    elements.append([tower_top_right, counterwt_top_nodes[0]])

    # SUPPORT CABLES from tower top to jib (multiple points for support)
    # Adjust these lists to control number of cables:
    # More cables = more support but more complex
    # Fewer cables = simpler but less support
    jib_cable_points = [-1]  # Only cable to tip: [-1]
                              # Two cables: [len(jib_top_nodes)//2, -1]
                              # Three cables: [len(jib_top_nodes)//3, 2*len(jib_top_nodes)//3, -1]

    for cable_idx in jib_cable_points:
        elements.append([tower_top_left, jib_top_nodes[cable_idx]])
        elements.append([tower_top_right, jib_top_nodes[cable_idx]])

    # SUPPORT CABLES from tower top to counterweight arm
    counterwt_cable_points = [-1]  # Only cable to tip: [-1]
                                    # Two cables: [len(counterwt_top_nodes)//2, -1]

    for cable_idx in counterwt_cable_points:
        elements.append([tower_top_left, counterwt_top_nodes[cable_idx]])
        elements.append([tower_top_right, counterwt_top_nodes[cable_idx]])

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

    # Material and section properties
    E = 200e9  # Steel Young's modulus (Pa)
    A_tower = 0.02  # Tower members - larger cross-section (m²)
    A_main = 0.01  # Main structural members (m²)
    A_secondary = 0.005  # Secondary/bracing members (m²)
    A_cable = 0.003  # Support cables - smaller cross-section (m²)

    # Boundary conditions - fix ground base nodes
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    for node in ground_base_nodes:
        bc[node, :] = True  # Fixed support at ground

    # Loads
    loads = np.zeros([n_nodes, 2], float)

    # Load at jib tip (crane lifting load)
    # In a real crane, the load hangs from the bottom chord (trolley with hook)
    jib_load = 10000  # N (100 kN)
    loads[jib_bot_nodes[-1], 1] = -jib_load  # Load hangs from bottom chord only

    # Counterweight (simulates heavy concrete blocks)
    # Counterweight sits on top of the counterweight arm platform
    counterweight = 150000  # N (150 kN) - balances the jib load
    loads[counterwt_bot_nodes[-1], 1] = -counterweight  # Weight on bottom chord

    # Self-weight approximation
    for node in jib_top_nodes + jib_bot_nodes:
        loads[node, 1] -= 500  # Jib self-weight
    for node in counterwt_top_nodes + counterwt_bot_nodes:
        loads[node, 1] -= 500  # Counterweight arm self-weight
    for node in tower_left_nodes + tower_right_nodes:
        loads[node, 1] -= 800  # Tower self-weight

    print(f"\nApplied jib load: {jib_load/1000:.0f} kN")
    print(f"Counterweight: {counterweight/1000:.0f} kN")

    # Prepare for FE analysis
    bc_mask = bc.reshape(1, 2*n_nodes).ravel()
    load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

    # Assembly global stiffness matrix
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)

    n_tower_elements = len(tower_left_nodes) - 1
    element_type_list = []

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
        k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

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

    # Plot deformed shape
    scale_factor = 50
    plot_truss(X + D * scale_factor, C,
              f"Tower Crane - Deformed Shape (Scale: {scale_factor}x)",
              show_nodes=False)

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
        k_el = element_stiffness(X[n1], X[n2], A, E)
        f_el = k_el @ d_el

        # Calculate axial force
        d_vec = X[n2] - X[n1]
        L = np.linalg.norm(d_vec)
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

    # Safety analysis
    yield_strength = 250e6  # Pa (typical steel)
    safety_factor = 2.5

    max_stress_tension = max_tension / A_tower
    max_stress_compression = abs(max_compression) / A_tower

    print(f"\n" + "="*70)
    print("STRUCTURAL SAFETY ANALYSIS")
    print("="*70)
    print(f"Max tensile stress:     {max_stress_tension/1e6:.1f} MPa")
    print(f"Max compressive stress: {max_stress_compression/1e6:.1f} MPa")
    print(f"Allowable stress:       {yield_strength/safety_factor/1e6:.1f} MPa")

    if max_stress_tension < yield_strength/safety_factor and max_stress_compression < yield_strength/safety_factor:
        print("\n✓ Structure is SAFE under applied loads")
    else:
        print("\n✗ WARNING: Structure may be OVERSTRESSED")
        print("  Recommendation: Increase member sizes or reduce loads")

    # Plot stress heatmap
    plot_stress_heatmap(X, C, element_stresses, element_areas,
                       title="Tower Crane Stress Distribution - With Counterweight")

    print("\n" + "="*70)

    return X, C, D, loads

if __name__ == "__main__":
    tower_crane_simulation()
