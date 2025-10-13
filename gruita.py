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
    tower_height = 25.0  # m
    boom_height = 4.0    # m boom depth
    n_boom_segments = 10  # Number of segments along boom

    # Calculate positions for horizontal boom
    boom_x = np.linspace(0, crane_length, n_boom_segments + 1)
    boom_y_top = np.full(n_boom_segments + 1, tower_height + boom_height)  # Top chord horizontal
    boom_y_bot = np.full(n_boom_segments + 1, tower_height)  # Bottom chord horizontal

    # Initialize node coordinates
    total_nodes = 1 + 2 * (n_boom_segments + 1)  # Tower top + boom nodes
    X = np.zeros([total_nodes, 2], float)

    node_idx = 0

    # Tower base and top
    X[0] = [0, 35]  # Tower base
    tower_base = 0
    node_idx += 1

    # Boom top chord nodes (starting from tower top)
    boom_top_nodes = []
    for i in range(n_boom_segments + 1):
        X[node_idx] = [boom_x[i], boom_y_top[i]]
        boom_top_nodes.append(node_idx)
        node_idx += 1

    # Boom bottom chord nodes
    boom_bot_nodes = []
    for i in range(n_boom_segments + 1):
        X[node_idx] = [boom_x[i], boom_y_bot[i]]
        boom_bot_nodes.append(node_idx)
        node_idx += 1

    return X, tower_base, boom_top_nodes, boom_bot_nodes

def create_crane_connectivity(tower_base, boom_top_nodes, boom_bot_nodes):
    '''Create element connectivity matrix for crane'''

    elements = []

    # Tower element (from base to boom connection)
    elements.append([tower_base, boom_bot_nodes[0]])  # Tower vertical member

    # Boom top chord elements
    for i in range(len(boom_top_nodes) - 1):
        elements.append([boom_top_nodes[i], boom_top_nodes[i+1]])

    # Boom bottom chord elements
    for i in range(len(boom_bot_nodes) - 1):
        elements.append([boom_bot_nodes[i], boom_bot_nodes[i+1]])

    # Boom vertical members
    for i in range(len(boom_top_nodes)):
        elements.append([boom_top_nodes[i], boom_bot_nodes[i]])

    # Boom diagonal members (alternating)
    for i in range(len(boom_top_nodes) - 1):
        if i % 2 == 0:
            elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
        else:
            elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])

    # Support ties from tower to boom
    elements.append([tower_base, boom_top_nodes[6]])  # Support tie 1
    elements.append([tower_base, boom_top_nodes[10]])  # Support tie 2

    return np.array(elements, dtype=int)

def crane_simulation():
    '''Main crane simulation function'''

    print("Designing 30m Crane Structure...")

    # Create geometry
    X, tower_base, boom_top_nodes, boom_bot_nodes = design_crane_geometry()
    C = create_crane_connectivity(tower_base, boom_top_nodes, boom_bot_nodes)

    print(f"Structure created with {X.shape[0]} nodes and {C.shape[0]} elements")

    # Plot original structure
    plot_truss(X, C, "30m Crane Truss Structure - Original", show_nodes=True)

    # Material and section properties
    E = 200e9  # Steel Young's modulus (Pa)
    A_main = 0.01  # Main structural members cross-sectional area (m�)
    A_secondary = 0.005  # Secondary members cross-sectional area (m�)

    # Boundary conditions
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    # Fix all nodes at x=0 (tower base and left-side boom nodes)
    bc[tower_base, :] = True  # Fixed support at tower base
    bc[boom_top_nodes[0], :] = True  # Fixed support at top left boom node
    bc[boom_bot_nodes[0], :] = True  # Fixed support at bottom left boom node

    # Loads - simulate load at crane tip
    loads = np.zeros([n_nodes, 2], float)
    tip_load = 500000  # N (50 kN vertical load at tip)
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
        if iEl == 0:  # Tower element
            A = A_main
        elif iEl < 1 + 2 * (len(boom_top_nodes) - 1):  # Main boom chords
            A = A_main
        else:  # Secondary elements
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
        if iEl == 0:  # Tower element
            A = A_main
            member_type = "Tower"
        elif iEl < 1 + 2 * (len(boom_top_nodes) - 1):  # Main boom chords
            A = A_main
            member_type = "Boom Chord"
        else:  # Secondary elements
            A = A_secondary
            member_type = "Secondary"

        element_areas.append(A)

        # Calculate element force
        k_el = element_stiffness(X[n1], X[n2], A, E)
        f_el = k_el @ d_el

        # Calculate axial force using element vector
        d_vec = X[n2] - X[n1]
        L = np.linalg.norm(d_vec)
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

    return X, C, D, loads

if __name__ == "__main__":
    crane_simulation()