#!/usr/bin/env python3
"""
TEST GRUITA3 CRANE WITH GRUITA2 TESTS
======================================
This script loads the best crane from gruita3 and runs all the Gruita2 tests on it.

The gruita3 crane has individually optimized member thicknesses, so this script
adapts the Gruita2 testing framework to work with variable cross-sections.

Usage:
------
python test_gruita3_with_gruita2_tests.py [path/to/best_crane.pkl]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# Import functions from gruita3
from gruita3 import (
    load_crane,
    element_stiffness,
    dof_el,
    hollow_circular_section
)

def run_full_simulation_on_gruita3_crane(crane_data, max_load=40000):
    """
    Run complete structural simulation on the gruita3 crane
    Similar to Gruita2's crane_simulation() but adapted for variable thicknesses

    Parameters:
    -----------
    crane_data : dict
        Crane design data from load_crane()
    max_load : float
        Applied load at tip in Newtons (default: 40 kN)

    Returns:
    --------
    results : dict
        Dictionary containing all simulation results
    """
    print("\n" + "="*80)
    print("FULL STRUCTURAL SIMULATION (GRUITA2-STYLE)")
    print("="*80)

    # Extract crane data
    X = crane_data['X']
    C = crane_data['C']
    tower_base = crane_data['tower_base']
    boom_top_nodes = crane_data['boom_top_nodes']
    boom_bot_nodes = crane_data['boom_bot_nodes']
    thickness_params = crane_data['thickness_params']
    element_types = crane_data['element_types']

    # Material properties
    E = 200e9  # Pa
    rho_steel = 7850  # kg/m³
    g = 9.81  # m/s²

    n_nodes = X.shape[0]
    n_elements = C.shape[0]

    print(f"\nStructure Details:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Elements: {n_elements}")
    print(f"  Load: {max_load/1000:.1f} kN at tip")

    # Build element areas and inertias from optimized thicknesses
    element_areas = []
    element_inertias = []
    element_masses = []
    total_mass = 0

    for iEl in range(n_elements):
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
        element_inertias.append(I)

        # Calculate mass
        n1, n2 = C[iEl]
        L = np.linalg.norm(X[n2] - X[n1])
        volume = A * L
        mass = rho_steel * volume
        element_masses.append(mass)
        total_mass += mass

    print(f"  Total mass: {total_mass:.2f} kg")

    # Boundary conditions
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True  # Pin support at boom base
    bc[tower_base, :] = True  # Fixed tower

    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Assemble global stiffness matrix
    print("\nAssembling global stiffness matrix...")
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)

    for iEl in range(n_elements):
        A = element_areas[iEl]
        dof = dof_el(C[iEl,0], C[iEl,1])
        k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    # Apply loads
    print("Applying loads (tip load + self-weight)...")
    loads = np.zeros([n_nodes, 2], float)
    loads[boom_top_nodes[-1], 1] = -max_load  # Tip load

    # Add self-weight
    for iEl in range(n_elements):
        n1, n2 = C[iEl]
        weight = element_masses[iEl] * g
        loads[n1, 1] -= weight / 2
        loads[n2, 1] -= weight / 2

    load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

    # Solve
    print("Solving system...")
    displacements = np.zeros([2*n_nodes], float)
    displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
    D = displacements.reshape(n_nodes, 2)

    # Calculate stresses and forces
    print("Calculating stresses and forces...")
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

    P_critica = 1e12
    critical_element = -1
    for iEl in range(n_elements):
        if element_forces[iEl] < -1:  # Compression
            n1, n2 = C[iEl]
            L = np.linalg.norm(X[n2] - X[n1])
            I = element_inertias[iEl]
            P_cr = (np.pi**2 * E * I) / (L**2)
            if P_cr < P_critica:
                P_critica = P_cr
                critical_element = iEl

    if P_max_compression < 1e-6:
        FS_pandeo = 1000
    else:
        FS_pandeo = P_critica / P_max_compression

    # Calculate cost
    m0 = 1000
    n_elementos_0 = 50
    n_uniones_0 = 25

    cost = (total_mass / m0) + 1.5 * (n_elements / n_elementos_0) + 2 * (n_nodes / n_uniones_0)

    # Print results
    print("\n" + "="*80)
    print("SIMULATION RESULTS")
    print("="*80)
    print(f"\nDeflections:")
    print(f"  Maximum deflection: {np.max(np.abs(D))*1000:.2f} mm")
    print(f"  Tip deflection: {abs(D[boom_top_nodes[-1], 1])*1000:.2f} mm")

    print(f"\nStresses:")
    print(f"  Maximum stress: {sigma_max/1e6:.2f} MPa")
    print(f"  Maximum tension: {np.max(element_stresses)/1e6:.2f} MPa")
    print(f"  Maximum compression: {np.min(element_stresses)/1e6:.2f} MPa")

    print(f"\nForces:")
    print(f"  Maximum tensile force: {np.max(element_forces)/1000:.2f} kN")
    print(f"  Maximum compressive force: {abs(np.min(element_forces))/1000:.2f} kN")

    print(f"\nSafety Factors:")
    print(f"  Tension FS: {FS_tension:.2f}")
    print(f"  Buckling FS: {FS_pandeo:.2f}")
    if critical_element >= 0:
        print(f"  Critical buckling element: {critical_element}")

    print(f"\nCost Analysis:")
    print(f"  Total mass: {total_mass:.2f} kg")
    print(f"  Elements: {n_elements}")
    print(f"  Joints: {n_nodes}")
    print(f"  Total cost: {cost:.2f}")

    print("="*80)

    # Package results
    results = {
        'X': X,
        'C': C,
        'D': D,
        'element_forces': element_forces,
        'element_stresses': element_stresses,
        'element_areas': element_areas,
        'element_types': element_types,
        'total_mass': total_mass,
        'FS_tension': FS_tension,
        'FS_pandeo': FS_pandeo,
        'cost': cost,
        'boom_top_nodes': boom_top_nodes,
        'boom_bot_nodes': boom_bot_nodes,
        'tower_base': tower_base
    }

    return results

def plot_stress_heatmap_gruita3(results):
    """
    Create stress heatmap visualization (Gruita2-style)
    """
    X = results['X']
    C = results['C']
    stresses = results['element_stresses']
    element_areas = results['element_areas']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Plot 1: Stress magnitude heatmap
    sigma_abs = np.abs(stresses)
    vmax = max(np.max(sigma_abs), 1e-6)

    for iEl in range(len(C)):
        stress_normalized = sigma_abs[iEl] / vmax
        color = plt.cm.hot(stress_normalized)
        linewidth = 3 * (1 + 2*stress_normalized)

        ax1.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                [X[C[iEl,0],1], X[C[iEl,1],1]],
                color=color, linewidth=linewidth, alpha=0.8)

    ax1.scatter(X[:,0], X[:,1], c='black', s=40, zorder=5, edgecolors='white', linewidths=1)
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_title('Stress Magnitude Heatmap\n(Color: stress level, Width: intensity)',
                 fontsize=14, fontweight='bold')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hot,
                               norm=plt.Normalize(vmin=0, vmax=vmax/1e6))
    sm.set_array([])
    cbar1 = plt.colorbar(sm, ax=ax1)
    cbar1.set_label('Stress (MPa)', fontsize=11)

    # Plot 2: Tension/Compression visualization
    for iEl in range(len(C)):
        stress = stresses[iEl]
        if stress > 0:  # Tension
            color = 'red'
            intensity = min(stress / (100e6), 1.0)
        else:  # Compression
            color = 'blue'
            intensity = min(abs(stress) / (100e6), 1.0)

        linewidth = 2 + 4*intensity
        alpha = 0.4 + 0.6*intensity

        ax2.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                [X[C[iEl,0],1], X[C[iEl,1],1]],
                color=color, linewidth=linewidth, alpha=alpha)

    ax2.scatter(X[:,0], X[:,1], c='black', s=40, zorder=5, edgecolors='white', linewidths=1)
    ax2.set_xlabel('x (m)', fontsize=12)
    ax2.set_ylabel('y (m)', fontsize=12)
    ax2.set_title('Tension/Compression Distribution\n(Red: tension, Blue: compression)',
                 fontsize=14, fontweight='bold')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='Tension'),
        Line2D([0], [0], color='blue', lw=3, label='Compression')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.show()

def analyze_moving_load_gruita3(crane_data, load_magnitudes=None):
    """
    Gruita2-style moving load analysis
    """
    from gruita3 import analyze_moving_load
    return analyze_moving_load(crane_data, load_magnitudes, scale_factor=1)

def animate_moving_load_gruita3(crane_data, load_magnitude=60000):
    """
    Gruita2-style moving load animation
    """
    from gruita3 import animate_moving_load
    return animate_moving_load(crane_data, load_magnitude, scale_factor=1)

def main():
    print("="*80),
    print("GRUITA3 CRANE - GRUITA2 TESTING FRAMEWORK")
    print("="*80)

    # Get crane file path
    if len(sys.argv) > 1:
        crane_file = Path(sys.argv[1])
    else:
        # Find most recent gruita3 iterations directory
        iteration_dirs = sorted(Path('.').glob('gruita3_iterations_*'))
        if not iteration_dirs:
            print("\nERROR: No gruita3_iterations_* directories found!")
            print("Please run gruita3.py first or specify a crane file path.")
            return

        latest_dir = iteration_dirs[-1]
        crane_file = latest_dir / 'best_crane.pkl'

        if not crane_file.exists():
            print(f"\nERROR: No best_crane.pkl found in {latest_dir}")
            return

    print(f"\nLoading crane from: {crane_file}")
    print("-"*80)

    # Load crane
    crane_data = load_crane(crane_file)

    # Menu
    while True:
        print("\n" + "="*80)
        print("TESTING MENU (GRUITA2-STYLE TESTS)")
        print("="*80)
        print("1. Full structural simulation (40 kN)")
        print("2. Stress heatmap visualization")
        print("3. Moving load analysis (0-40 kN)")
        print("4. Animate moving load (30 kN)")
        print("5. Custom load simulation")
        print("6. Run all tests")
        print("7. Exit")
        print("="*80)

        choice = input("\nEnter your choice (1-7): ").strip()

        if choice == '1':
            results = run_full_simulation_on_gruita3_crane(crane_data, max_load=40000)

        elif choice == '2':
            results = run_full_simulation_on_gruita3_crane(crane_data, max_load=40000)
            plot_stress_heatmap_gruita3(results)

        elif choice == '3':
            analyze_moving_load_gruita3(crane_data)

        elif choice == '4':
            animate_moving_load_gruita3(crane_data)

        elif choice == '5':
            try:
                load = float(input("Enter load in kN: "))
                load_N = load * 1000
                results = run_full_simulation_on_gruita3_crane(crane_data, max_load=load_N)

                show_heatmap = input("Show stress heatmap? (y/n): ").strip().lower()
                if show_heatmap == 'y':
                    plot_stress_heatmap_gruita3(results)

            except ValueError:
                print("Invalid input!")

        elif choice == '6':
            print("\nRunning all tests...\n")

            # Test 1: Full simulation
            print("\n[1/4] Running full structural simulation...")
            results = run_full_simulation_on_gruita3_crane(crane_data, max_load=40000)

            # Test 2: Stress heatmap
            print("\n[2/4] Generating stress heatmap...")
            plot_stress_heatmap_gruita3(results)

            # Test 3: Moving load analysis
            print("\n[3/4] Running moving load analysis...")
            analyze_moving_load_gruita3(crane_data)

            # Test 4: Animation
            print("\n[4/4] Creating animation...")
            animate_moving_load_gruita3(crane_data)

            print("\n" + "="*80)
            print("ALL TESTS COMPLETED!")
            print("="*80)

        elif choice == '7':
            print("\nExiting. Goodbye!")
            break

        else:
            print("Invalid choice! Please enter 1-7.")

if __name__ == '__main__':
    main()