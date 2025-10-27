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
        return np.zeros((4, 4), dtype=float), 0.0

    # Direction cosines (unit vector components)
    c = d[0] / L  # cos(θ) - x-direction cosine
    s = d[1] / L  # sin(θ) - y-direction cosine

    # Local element stiffness matrix (1D rod element)
    k_local = (A * E / L) * np.array([[ 1, -1],
                                      [-1,  1]], dtype=float)

    # Transformation matrix from local to global coordinates
    T = np.array([[ c, s, 0, 0],
                  [ 0, 0, c, s]], dtype=float)

    # Transform stiffness matrix to global coordinates: K_global = T^T * K_local * T
    k_structural = np.matmul(T.T, np.matmul(k_local, T))

    return k_structural, L

def dof_el(nnod1, nnod2):
    '''Returns Elemental DOF for Assembly'''
    return [2*(nnod1+1)-2, 2*(nnod1+1)-1, 2*(nnod2+1)-2, 2*(nnod2+1)-1]

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

def plot_truss(X, C, title="Truss Structure", scale_factor=1, show_nodes=True, filename=None):
    '''Plot truss structure with optional node numbering'''
    plt.figure(figsize=(16, 8))

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

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_stress_heatmap(X, C, stresses, title="Stress Distribution", filename=None):
    '''Plot truss structure with stress heatmap visualization'''
    fig, ax = plt.subplots(figsize=(18, 8))

    # Create line segments for each element
    segments = []
    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        segments.append([(X[n1, 0], X[n1, 1]), (X[n2, 0], X[n2, 1])])

    # Normalize stress values for colormap (in MPa)
    stress_mpa = stresses / 1e6
    norm = Normalize(vmin=np.min(stress_mpa), vmax=np.max(stress_mpa))

    # Create colormap (blue for compression, red for tension)
    cmap = cm.RdBu_r

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
    ax.autoscale()

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_tension_compression_combined(X, C, stresses, title="Tension and Compression Stress Distribution", filename=None):
    '''Plot truss structure with tension and compression in side-by-side subplots'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

    # Create line segments for each element
    segments = []
    tension_stresses = []
    compression_stresses = []
    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        segments.append([(X[n1, 0], X[n1, 1]), (X[n2, 0], X[n2, 1])])
        # Separate tension and compression
        tension_stresses.append(max(0, stresses[iEl]))
        compression_stresses.append(abs(min(0, stresses[iEl])))

    tension_stresses = np.array(tension_stresses)
    compression_stresses = np.array(compression_stresses)

    # --- TENSION PLOT (LEFT) ---
    tension_mpa = tension_stresses / 1e6
    max_tension = np.max(tension_mpa)

    if max_tension > 0:
        norm_t = Normalize(vmin=0, vmax=max_tension)
    else:
        norm_t = Normalize(vmin=0, vmax=1)

    lc_tension = LineCollection(segments, cmap=cm.Reds, norm=norm_t, linewidths=4)
    lc_tension.set_array(tension_mpa)
    line_t = ax1.add_collection(lc_tension)

    cbar_t = fig.colorbar(line_t, ax=ax1, pad=0.02)
    cbar_t.set_label('Tension Stress (MPa)', rotation=270, labelpad=25)

    ax1.scatter(X[:, 0], X[:, 1], c='black', s=30, zorder=5, alpha=0.5)
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_title('Tension Stress', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.autoscale()

    # --- COMPRESSION PLOT (RIGHT) ---
    compression_mpa = compression_stresses / 1e6
    max_compression = np.max(compression_mpa)

    if max_compression > 0:
        norm_c = Normalize(vmin=0, vmax=max_compression)
    else:
        norm_c = Normalize(vmin=0, vmax=1)

    lc_compression = LineCollection(segments, cmap=cm.Blues, norm=norm_c, linewidths=4)
    lc_compression.set_array(compression_mpa)
    line_c = ax2.add_collection(lc_compression)

    cbar_c = fig.colorbar(line_c, ax=ax2, pad=0.02)
    cbar_c.set_label('Compression Stress (MPa)', rotation=270, labelpad=25)

    ax2.scatter(X[:, 0], X[:, 1], c='black', s=30, zorder=5, alpha=0.5)
    ax2.set_xlabel('x (m)', fontsize=12)
    ax2.set_ylabel('y (m)', fontsize=12)
    ax2.set_title('Compression Stress', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.autoscale()

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_deformation_heatmap(X, C, D, title="Deformation Magnitude Distribution", filename=None):
    '''Plot truss structure with deformation magnitude heatmap'''
    fig, ax = plt.subplots(figsize=(18, 8))

    # Create line segments for each element
    segments = []
    deformations = []
    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        segments.append([(X[n1, 0], X[n1, 1]), (X[n2, 0], X[n2, 1])])
        # Calculate average deformation magnitude for element (both nodes)
        avg_def = (np.linalg.norm(D[n1]) + np.linalg.norm(D[n2])) / 2
        deformations.append(avg_def)

    deformations = np.array(deformations)

    # Normalize deformation values for colormap (in mm)
    deformation_mm = deformations * 1000
    norm = Normalize(vmin=0, vmax=np.max(deformation_mm))

    # Create colormap (green to yellow to red for increasing deformation)
    cmap = cm.YlOrRd  # Yellow-Orange-Red colormap

    # Create line collection with colors based on deformation
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4)
    lc.set_array(deformation_mm)

    # Add line collection to plot
    line = ax.add_collection(lc)

    # Add colorbar
    cbar = fig.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label('Deformation Magnitude (mm)', rotation=270, labelpad=25)

    # Plot nodes
    ax.scatter(X[:, 0], X[:, 1], c='black', s=30, zorder=5, alpha=0.5)

    # Labels and formatting
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.autoscale()

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def design_tower_crane_geometry(tower_height=5.0, counterweight_span=10.0, boom_length=40.0,
                                 boom_height=1.0, n_boom_segments=20):
    '''
    Design tower crane geometry with counterweight
    FLIPPED ORIENTATION: Boom on x-axis (horizontal), cables connect from top

    Parameters:
    -----------
    tower_height : float
        Height of tower from base to boom level (m)
    counterweight_span : float
        Distance from tower to counterweight (m)
    boom_length : float
        Total boom length from tower to tip (m)
    boom_height : float
        Vertical distance between top and bottom boom chords (m)
    n_boom_segments : int
        Number of segments along boom

    Returns:
    --------
    X : ndarray
        Node coordinates
    node_indices : dict
        Dictionary containing node indices for different parts
    '''

    # Calculate positions
    # Counterweight side: from -counterweight_span to 0 (excluding 0)
    # Boom side: from 0 to boom_length (excluding 0)
    counterweight_x = np.linspace(-counterweight_span, 0, n_boom_segments // 4 + 1)[:-1]  # Exclude x=0
    boom_x = np.linspace(0, boom_length, n_boom_segments + 1)[1:]  # Exclude x=0

    # Initialize node list
    nodes = []
    node_indices = {}

    # Tower base (fixed support at origin)
    tower_base_idx = len(nodes)
    nodes.append([0, 0])

    # Tower top (at tower_height)
    tower_top_idx = len(nodes)
    nodes.append([0, tower_height])

    # Counterweight side - bottom chord (on x-axis level, y=0)
    cw_bot_nodes = []
    for x in counterweight_x:
        cw_bot_nodes.append(len(nodes))
        nodes.append([x, 0])

    # Counterweight side - top chord (at boom_height above x-axis)
    cw_top_nodes = []
    for x in counterweight_x:
        cw_top_nodes.append(len(nodes))
        nodes.append([x, boom_height])

    # Boom - bottom chord (on x-axis level, y=0)
    boom_bot_nodes = []
    for x in boom_x:
        boom_bot_nodes.append(len(nodes))
        nodes.append([x, 0])

    # Boom - top chord (at boom_height above x-axis)
    boom_top_nodes = []
    for x in boom_x:
        boom_top_nodes.append(len(nodes))
        nodes.append([x, boom_height])

    X = np.array(nodes, dtype=float)

    node_indices = {
        'tower_base': tower_base_idx,
        'tower_top': tower_top_idx,
        'cw_bot': cw_bot_nodes,
        'cw_top': cw_top_nodes,
        'boom_bot': boom_bot_nodes,
        'boom_top': boom_top_nodes
    }

    return X, node_indices

def create_tower_crane_connectivity(node_indices):
    '''
    Create element connectivity matrix for tower crane with counterweight
    Configuration: Boom on x-axis, cables from tower TOP to boom top chord
    '''

    elements = []

    # Add tower element connecting base to top (provides vertical support structure)
    elements.append([node_indices['tower_base'], node_indices['tower_top']])

    # Counterweight side - bottom chord
    cw_bot = node_indices['cw_bot']
    for i in range(len(cw_bot) - 1):
        elements.append([cw_bot[i], cw_bot[i+1]])

    # Counterweight side - top chord
    cw_top = node_indices['cw_top']
    for i in range(len(cw_top) - 1):
        elements.append([cw_top[i], cw_top[i+1]])

    # Counterweight side - verticals
    for i in range(len(cw_top)):
        elements.append([cw_bot[i], cw_top[i]])

    # Counterweight side - diagonals (alternating pattern)
    for i in range(len(cw_top) - 1):
        if i % 2 == 0:
            elements.append([cw_bot[i], cw_top[i+1]])
        else:
            elements.append([cw_top[i], cw_bot[i+1]])

    # Boom - bottom chord
    boom_bot = node_indices['boom_bot']
    for i in range(len(boom_bot) - 1):
        elements.append([boom_bot[i], boom_bot[i+1]])

    # Boom - top chord
    boom_top = node_indices['boom_top']
    for i in range(len(boom_top) - 1):
        elements.append([boom_top[i], boom_top[i+1]])

    # Boom - verticals
    for i in range(len(boom_top)):
        elements.append([boom_bot[i], boom_top[i]])

    # Boom - diagonals (alternating pattern)
    for i in range(len(boom_top) - 1):
        if i % 2 == 0:
            elements.append([boom_bot[i], boom_top[i+1]])
        else:
            elements.append([boom_top[i], boom_bot[i+1]])

    # Connect tower base to boom and counterweight structures
    # Since we excluded x=0 from boom/counterweight, connect to nearest nodes
    elements.append([node_indices['tower_base'], cw_bot[-1]])  # Tower base to last counterweight bottom node
    elements.append([node_indices['tower_base'], boom_bot[0]])  # Tower base to first boom bottom node
    elements.append([node_indices['tower_base'], cw_top[-1]])  # Tower base to last counterweight top node
    elements.append([node_indices['tower_base'], boom_top[0]])  # Tower base to first boom top node

    # Support cables from tower TOP to boom TOP chord (cables on top)
    elements.append([node_indices['tower_top'], boom_top[len(boom_top)//2]])  # Cable 1
    elements.append([node_indices['tower_top'], boom_top[-1]])  # Cable 2 (to tip)
    elements.append([node_indices['tower_top'], cw_top[0]])  # Cable to counterweight

    return np.array(elements, dtype=int)

def calculate_cost(mass, n_elements, n_nodes, m0=1000, nelementos0=50, nuniones0=30):
    '''
    Calculate cost function according to TP2 specification

    C(m, n_elementos, n_uniones) = m/m0 + 1.5*(n_elementos/nelementos0) + 2*(n_uniones/nuniones0)
    '''
    cost = (mass / m0) + 1.5 * (n_elements / nelementos0) + 2 * (n_nodes / nuniones0)
    return cost

def tower_crane_analysis(X, C, node_indices, load_position_x, load_magnitude,
                         counterweight_mass, D_outer, thickness, E, rho, sigma_adm):
    '''
    Perform FEA analysis of tower crane for a given load position

    Parameters:
    -----------
    X : ndarray
        Node coordinates
    C : ndarray
        Connectivity matrix
    node_indices : dict
        Dictionary with node indices
    load_position_x : float
        Position of load along boom (m)
    load_magnitude : float
        Magnitude of applied load (N)
    counterweight_mass : float
        Counterweight mass (kg)
    D_outer : float
        Outer diameter of tubular sections (m)
    thickness : float
        Wall thickness (m)
    E : float
        Young's modulus (Pa)
    rho : float
        Material density (kg/m³)
    sigma_adm : float
        Admissible stress (Pa)

    Returns:
    --------
    results : dict
        Analysis results including displacements, stresses, forces, safety factors
    '''

    n_nodes = X.shape[0]
    n_elements = C.shape[0]

    # Calculate section properties
    A, I = tubular_section_properties(D_outer, thickness)

    # Initialize global stiffness matrix
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)

    # Store element lengths for later use
    element_lengths = np.zeros(n_elements)
    element_masses = np.zeros(n_elements)

    # Assemble global stiffness matrix
    for iEl in range(n_elements):
        n1, n2 = C[iEl]
        k_el, L = element_stiffness(X[n1], X[n2], A, E)
        element_lengths[iEl] = L
        element_masses[iEl] = A * L * rho

        dof = dof_el(n1, n2)
        k_global[np.ix_(dof, dof)] += k_el

    # Boundary conditions
    bc = np.full((n_nodes, 2), False)
    # Fix tower base as pin support at ground (fully fixed)
    bc[node_indices['tower_base'], :] = True  # Fix both x and y

    # Apply loads
    loads = np.zeros([n_nodes, 2], float)

    # Self-weight (distributed as nodal loads)
    for iNode in range(n_nodes):
        # Find elements connected to this node
        connected_elements = np.where((C[:, 0] == iNode) | (C[:, 1] == iNode))[0]
        nodal_mass = 0
        for iEl in connected_elements:
            nodal_mass += element_masses[iEl] / 2  # Half of element mass to each node
        loads[iNode, 1] -= nodal_mass * 9.81  # Gravity load

    # Counterweight (at leftmost node of counterweight side)
    counterweight_node = node_indices['cw_bot'][0]  # Leftmost counterweight node
    loads[counterweight_node, 1] -= counterweight_mass * 9.81

    # Applied load at specified position along boom
    # Find nearest boom bottom node to load position
    boom_bot_nodes = node_indices['boom_bot']
    load_node = boom_bot_nodes[0]
    min_dist = abs(X[load_node, 0] - load_position_x)
    for node in boom_bot_nodes:
        dist = abs(X[node, 0] - load_position_x)
        if dist < min_dist:
            min_dist = dist
            load_node = node
    loads[load_node, 1] -= load_magnitude

    # Solve system
    bc_mask = bc.reshape(1, 2*n_nodes).ravel()
    load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    # Check for singular matrix
    cond_num = np.linalg.cond(k_reduced)
    if cond_num > 1e10:
        print(f"WARNING: Stiffness matrix is poorly conditioned (cond={cond_num:.2e})")
        print(f"This may indicate structural instability or duplicate/disconnected nodes")

    displacements = np.zeros([2*n_nodes], float)
    try:
        displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
    except np.linalg.LinAlgError as e:
        print(f"ERROR: Failed to solve system - {e}")
        print("Structure may be unstable or have mechanisms")

    D = displacements.reshape(n_nodes, 2)

    # Calculate element forces and stresses
    element_forces = np.zeros(n_elements)
    element_stresses = np.zeros(n_elements)
    buckling_safety_factors = np.zeros(n_elements)
    tension_safety_factors = np.zeros(n_elements)

    for iEl in range(n_elements):
        n1, n2 = C[iEl]
        d_el = np.concatenate([D[n1], D[n2]])

        k_el, L = element_stiffness(X[n1], X[n2], A, E)
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

        # Calculate safety factors
        # Tension safety factor
        if stress > 0:  # Tension
            tension_safety_factors[iEl] = sigma_adm / stress
        else:  # Compression
            tension_safety_factors[iEl] = sigma_adm / abs(stress)

        # Buckling safety factor (only for compression)
        if axial_force < 0:  # Compression (negative force)
            P_cr = euler_buckling_load(E, I, L, end_conditions='pinned-pinned')
            buckling_safety_factors[iEl] = P_cr / abs(axial_force)
        else:
            buckling_safety_factors[iEl] = np.inf

    # Calculate total mass
    total_mass = np.sum(element_masses)

    # Calculate reactions at supports
    reaction_forces = k_global @ displacements
    reaction_at_base = reaction_forces[2*node_indices['tower_base']:2*node_indices['tower_base']+2]

    # Compile results
    results = {
        'displacements': D,
        'max_displacement': np.max(np.abs(D)),
        'tip_deflection': D[node_indices['boom_bot'][-1], 1],
        'element_forces': element_forces,
        'element_stresses': element_stresses,
        'max_stress': np.max(np.abs(element_stresses)),
        'max_tension_stress': np.max(element_stresses),
        'max_compression_stress': np.min(element_stresses),
        'tension_safety_factors': tension_safety_factors,
        'buckling_safety_factors': buckling_safety_factors,
        'min_tension_sf': np.min(tension_safety_factors),
        'min_buckling_sf': np.min(buckling_safety_factors[buckling_safety_factors < np.inf]),
        'min_overall_sf': min(np.min(tension_safety_factors),
                             np.min(buckling_safety_factors[buckling_safety_factors < np.inf])),
        'total_mass': total_mass,
        'element_masses': element_masses,
        'element_lengths': element_lengths,
        'reaction_forces': reaction_at_base,
        'load_node': load_node,
        'load_position_actual': X[load_node, 0]
    }

    return results

def main():
    '''Main function for tower crane design and optimization'''

    print("="*80)
    print("TOWER CRANE DESIGN - TP2 (WITH COUNTERWEIGHT)")
    print("="*80)

    # Material properties
    E = 200e9  # Young's modulus (Pa)
    rho = 7800  # Density (kg/m³)
    sigma_adm = 250e6  # Admissible stress (Pa)

    # Geometry parameters
    tower_height = 5.0  # m
    counterweight_span = 10.0  # m (TP2: 10m counterweight arm)
    boom_length = 40.0  # m
    boom_height = 1.0  # m (H parameter from assignment)
    n_boom_segments = 20

    # Section properties (tubular)
    D_outer = 0.050  # 50mm outer diameter (maximum allowed)
    thickness = 0.005  # 5mm wall thickness (initial guess)

    # Loads
    counterweight_mass = 15000  # kg (TP2: 15000 kg counterweight)
    max_load = 20000  # N (example working load)

    print(f"\nGeometry:")
    print(f"  Tower height: {tower_height} m")
    print(f"  Counterweight span: {counterweight_span} m")
    print(f"  Boom length: {boom_length} m")
    print(f"  Boom height: {boom_height} m")

    print(f"\nMaterial Properties:")
    print(f"  Young's modulus: {E/1e9} GPa")
    print(f"  Density: {rho} kg/m³")
    print(f"  Admissible stress: {sigma_adm/1e6} MPa")

    print(f"\nSection Properties:")
    print(f"  Outer diameter: {D_outer*1000} mm")
    print(f"  Wall thickness: {thickness*1000} mm")

    print(f"\nLoads:")
    print(f"  Counterweight mass: {counterweight_mass} kg")
    print(f"  Max working load: {max_load/1000} kN")

    # Create geometry
    print(f"\nCreating geometry...")
    X, node_indices = design_tower_crane_geometry(tower_height, counterweight_span,
                                                   boom_length, boom_height, n_boom_segments)
    C = create_tower_crane_connectivity(node_indices)

    print(f"  Nodes: {X.shape[0]}")
    print(f"  Elements: {C.shape[0]}")

    # Plot original structure
    plot_truss(X, C, "Tower Crane Structure - Original Geometry (With Counterweight)", show_nodes=True)

    # Analysis for no load condition
    print(f"\n" + "="*80)
    print("ANALYSIS 1: NO EXTERNAL LOAD (Self-weight + Counterweight only)")
    print("="*80)

    results_no_load = tower_crane_analysis(X, C, node_indices, load_position_x=boom_length,
                                           load_magnitude=0, counterweight_mass=counterweight_mass,
                                           D_outer=D_outer, thickness=thickness, E=E, rho=rho,
                                           sigma_adm=sigma_adm)

    print(f"\nDisplacements:")
    print(f"  Maximum displacement: {results_no_load['max_displacement']*1000:.2f} mm")
    print(f"  Boom tip deflection: {results_no_load['tip_deflection']*1000:.2f} mm")

    print(f"\nStresses:")
    print(f"  Maximum tensile stress: {results_no_load['max_tension_stress']/1e6:.2f} MPa")
    print(f"  Maximum compressive stress: {abs(results_no_load['max_compression_stress'])/1e6:.2f} MPa")

    print(f"\nSafety Factors:")
    print(f"  Minimum tension SF: {results_no_load['min_tension_sf']:.2f}")
    print(f"  Minimum buckling SF: {results_no_load['min_buckling_sf']:.2f}")
    print(f"  Minimum overall SF: {results_no_load['min_overall_sf']:.2f}")

    print(f"\nMass:")
    print(f"  Total structure mass: {results_no_load['total_mass']:.2f} kg")

    print(f"\nReactions at base:")
    print(f"  Rx = {results_no_load['reaction_forces'][0]:.2f} N")
    print(f"  Ry = {results_no_load['reaction_forces'][1]:.2f} N")

    # Check force equilibrium
    total_weight = (results_no_load['total_mass'] + counterweight_mass) * 9.81
    print(f"\nForce Equilibrium Check:")
    print(f"  Total weight (structure + counterweight): {total_weight:.2f} N")
    print(f"  Reaction force (Ry): {abs(results_no_load['reaction_forces'][1]):.2f} N")
    print(f"  Difference: {abs(total_weight - abs(results_no_load['reaction_forces'][1])):.2f} N")

    # Plot stress distributions for no load case
    plot_stress_heatmap(X, C, results_no_load['element_stresses'],
                       "Stress Distribution - No External Load")
    plot_tension_compression_combined(X, C, results_no_load['element_stresses'],
                                     "Tension and Compression Stress - No External Load")
    plot_deformation_heatmap(X, C, results_no_load['displacements'],
                            "Deformation - No External Load")

    # Analysis for load at boom tip
    print(f"\n" + "="*80)
    print("ANALYSIS 2: MAXIMUM LOAD AT BOOM TIP")
    print("="*80)

    load_position = boom_length  # Load at the tip
    print(f"\n--- Load Position: x = {load_position:.1f} m (TIP) ---")

    critical_results = tower_crane_analysis(X, C, node_indices, load_position_x=load_position,
                                           load_magnitude=max_load, counterweight_mass=counterweight_mass,
                                           D_outer=D_outer, thickness=thickness, E=E, rho=rho,
                                           sigma_adm=sigma_adm)

    all_results = [critical_results]  # Keep for compatibility
    critical_position = load_position

    print(f"  Max displacement: {critical_results['max_displacement']*1000:.2f} mm")
    print(f"  Tip deflection: {critical_results['tip_deflection']*1000:.2f} mm")
    print(f"  Max tensile stress: {critical_results['max_tension_stress']/1e6:.2f} MPa")
    print(f"  Max compressive stress: {abs(critical_results['max_compression_stress'])/1e6:.2f} MPa")
    print(f"  Min tension SF: {critical_results['min_tension_sf']:.2f}")
    print(f"  Min buckling SF: {critical_results['min_buckling_sf']:.2f}")
    print(f"  Min overall SF: {critical_results['min_overall_sf']:.2f}")

    if critical_results['min_overall_sf'] < 2.0:
        print(f"  WARNING: Safety factor below required minimum (FS > 2)")

    print(f"\n" + "="*80)
    print("CRITICAL LOAD CASE")
    print("="*80)
    print(f"Critical load position: x = {critical_position:.1f} m")
    print(f"Minimum overall safety factor: {critical_results['min_overall_sf']:.2f}")

    # Plot deformed shape for critical case
    scale_factor = 500
    X_deformed = X + critical_results['displacements'] * scale_factor
    plot_truss(X_deformed, C,
              f"Tower Crane - Deformed Shape (Load at x={critical_position:.1f}m, Scale: {scale_factor}x)",
              show_nodes=False)

    # Plot all stress distributions for critical case
    plot_stress_heatmap(X, C, critical_results['element_stresses'],
                       f"Stress Distribution - Critical Case (Load at x={critical_position:.1f}m)")
    plot_tension_compression_combined(X, C, critical_results['element_stresses'],
                                     f"Tension and Compression Stress - Critical Case (Load at x={critical_position:.1f}m)")
    plot_deformation_heatmap(X, C, critical_results['displacements'],
                            f"Deformation - Critical Case (Load at x={critical_position:.1f}m)")

    # Calculate cost
    n_elements = C.shape[0]
    n_nodes = X.shape[0]
    cost = calculate_cost(critical_results['total_mass'], n_elements, n_nodes)

    print(f"\n" + "="*80)
    print("COST ANALYSIS")
    print("="*80)
    print(f"Total mass: {critical_results['total_mass']:.2f} kg")
    print(f"Number of elements: {n_elements}")
    print(f"Number of nodes (unions): {n_nodes}")
    print(f"Cost function: C = {cost:.3f}")

    print(f"\n" + "="*80)
    print("SUMMARY - OPTIMAL DESIGN")
    print("="*80)
    print(f"Structure meets safety requirements: {critical_results['min_overall_sf'] >= 2.0}")
    print(f"Total mass: {critical_results['total_mass']:.2f} kg")
    print(f"Cost: {cost:.3f}")
    print(f"Max deflection (all cases): {max([r['max_displacement'] for r in all_results])*1000:.2f} mm")
    print(f"Max stress (all cases): {max([r['max_stress'] for r in all_results])/1e6:.2f} MPa")
    print(f"Min safety factor (all cases): {min([r['min_overall_sf'] for r in all_results]):.2f}")

    return X, C, node_indices, all_results, critical_results

if __name__ == "__main__":
    X, C, node_indices, all_results, critical_results = main()
