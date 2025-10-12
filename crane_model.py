import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import eigh

class CraneModel:
    """
    Tower Crane Structural Analysis Model

    Geometry:
    - Counter-jib: 10m horizontal
    - Vertical tower: 5m
    - Main jib: 30m horizontal (with 1m height at end)
    - Load: 15000kg at jib end

    Material Properties:
    - E = 200 GPa (steel)
    - ρ = 7800 kg/m³
    - σ_adm = 250 MPa
    """

    def __init__(self):
        # Material properties
        self.E = 200e9  # Pa
        self.rho = 7800  # kg/m³
        self.sigma_adm = 250e6  # Pa
        self.g = 9.81  # m/s²

        # Geometry
        self.counter_jib_length = 10  # m
        self.tower_height = 5  # m
        self.main_jib_length = 30  # m
        self.jib_end_height = 1  # m
        self.load_mass = 15000  # kg

        # Design variables (initial guess for cross-sectional areas in m²)
        self.A_counter = 0.01  # counter-jib
        self.A_tower = 0.02  # tower
        self.A_main = 0.01  # main jib
        self.A_cables = 0.001  # cables

    def get_nodes(self):
        """Define node coordinates"""
        # Node 0: Counter-jib end (ground level)
        # Node 1: Tower base (origin)
        # Node 2: Tower top
        # Node 3: Main jib end
        nodes = np.array([
            [-10, 0],  # Node 0: counter-jib end
            [0, 0],    # Node 1: tower base
            [0, 5],    # Node 2: tower top
            [30, 6]    # Node 3: main jib end (5m + 1m height difference)
        ])
        return nodes

    def get_elements(self):
        """Define elements (member connections)"""
        # Each row: [node_i, node_j, area, member_type]
        # member_type: 0=counter-jib, 1=tower, 2=main-jib, 3=cable
        elements = [
            [0, 1, self.A_counter, 0],  # Counter-jib (horizontal beam)
            [1, 2, self.A_tower, 1],    # Tower (vertical)
            [2, 3, self.A_main, 2],     # Main jib (horizontal beam)
            [0, 2, self.A_cables, 3],   # Cable from counter-jib end to tower top
            [0, 3, self.A_cables, 3],   # Cable from counter-jib end to main jib end
        ]
        return elements

    def assemble_stiffness_matrix(self, nodes, elements):
        """Assemble global stiffness matrix"""
        n_nodes = len(nodes)
        n_dofs = 2 * n_nodes  # 2 DOFs per node (x, y)
        K = np.zeros((n_dofs, n_dofs))

        for elem in elements:
            i, j, A, _ = elem
            xi, yi = nodes[i]
            xj, yj = nodes[j]

            # Element length
            L = np.sqrt((xj - xi)**2 + (yj - yi)**2)

            # Direction cosines
            c = (xj - xi) / L
            s = (yj - yi) / L

            # Element stiffness in global coordinates
            k_elem = (self.E * A / L) * np.array([
                [c**2, c*s, -c**2, -c*s],
                [c*s, s**2, -c*s, -s**2],
                [-c**2, -c*s, c**2, c*s],
                [-c*s, -s**2, c*s, s**2]
            ])

            # Assembly indices
            dofs = [2*i, 2*i+1, 2*j, 2*j+1]

            # Add to global matrix
            for ii, global_i in enumerate(dofs):
                for jj, global_j in enumerate(dofs):
                    K[global_i, global_j] += k_elem[ii, jj]

        return K

    def apply_boundary_conditions(self, K, F):
        """Apply fixed boundary conditions at tower base and counter-jib end"""
        # Fix node 0 (counter-jib end): DOFs 0 and 1
        # Fix node 1 (tower base): DOFs 2 and 3
        fixed_dofs = [0, 1, 2, 3]

        # Reduce system
        all_dofs = np.arange(len(F))
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

        K_reduced = K[np.ix_(free_dofs, free_dofs)]
        F_reduced = F[free_dofs]

        return K_reduced, F_reduced, free_dofs

    def solve_displacements(self):
        """Solve for nodal displacements"""
        nodes = self.get_nodes()
        elements = self.get_elements()

        # Assemble stiffness matrix
        K = self.assemble_stiffness_matrix(nodes, elements)

        # Applied loads (only load at node 3)
        n_dofs = 2 * len(nodes)
        F = np.zeros(n_dofs)
        F[6] = 0  # x-direction at node 3
        F[7] = -self.load_mass * self.g  # y-direction at node 3 (downward)

        # Apply boundary conditions
        K_reduced, F_reduced, free_dofs = self.apply_boundary_conditions(K, F)

        # Solve for displacements
        U = np.zeros(n_dofs)
        U[free_dofs] = np.linalg.solve(K_reduced, F_reduced)

        return U, nodes

    def calculate_stresses(self, U):
        """Calculate stresses in each member"""
        nodes = self.get_nodes()
        elements = self.get_elements()
        stresses = []

        for elem in elements:
            i, j, A, member_type = elem
            xi, yi = nodes[i]
            xj, yj = nodes[j]

            # Element length
            L = np.sqrt((xj - xi)**2 + (yj - yi)**2)

            # Direction cosines
            c = (xj - xi) / L
            s = (yj - yi) / L

            # Nodal displacements
            ui, vi = U[2*i], U[2*i+1]
            uj, vj = U[2*j], U[2*j+1]

            # Axial strain
            epsilon = (c * (uj - ui) + s * (vj - vi)) / L

            # Stress
            sigma = self.E * epsilon

            stresses.append({
                'member': f"Element {i}-{j}",
                'type': ['counter-jib', 'tower', 'main-jib', 'cable'][member_type],
                'stress': sigma / 1e6,  # Convert to MPa
                'axial_force': sigma * A / 1000  # kN
            })

        return stresses

    def calculate_weight(self):
        """Calculate total weight of structure"""
        nodes = self.get_nodes()
        elements = self.get_elements()
        total_weight = 0

        for elem in elements:
            i, j, A, _ = elem
            xi, yi = nodes[i]
            xj, yj = nodes[j]

            # Element length
            L = np.sqrt((xj - xi)**2 + (yj - yi)**2)

            # Weight of element
            weight = self.rho * A * L * self.g
            total_weight += weight

        return total_weight / 1000  # Convert to kN

    def buckling_analysis(self):
        """Calculate buckling load factor"""
        nodes = self.get_nodes()
        elements = self.get_elements()

        # Geometric stiffness matrix (simplified)
        K = self.assemble_stiffness_matrix(nodes, elements)

        # For buckling, we need geometric stiffness matrix
        # Simplified approach: check Euler buckling for compression members
        buckling_factors = []

        U, _ = self.solve_displacements()
        stresses = self.calculate_stresses(U)

        for idx, elem in enumerate(elements):
            i, j, A, member_type = elem
            xi, yi = nodes[i]
            xj, yj = nodes[j]

            # Element length
            L = np.sqrt((xj - xi)**2 + (yj - yi)**2)

            # Moment of inertia (assuming circular cross-section)
            # A = π*r² => r = sqrt(A/π)
            # I = π*r⁴/4 = A²/(4π)
            I = A**2 / (4 * np.pi)

            # Euler buckling load
            P_cr = (np.pi**2 * self.E * I) / L**2

            # Applied load from stress analysis
            P_applied = abs(stresses[idx]['axial_force'] * 1000)  # Convert back to N

            if P_applied > 0:
                buckling_factor = P_cr / P_applied
                buckling_factors.append({
                    'member': stresses[idx]['member'],
                    'P_critical': P_cr / 1000,  # kN
                    'P_applied': P_applied / 1000,  # kN
                    'buckling_factor': buckling_factor
                })

        return buckling_factors

    def optimize_design(self):
        """Optimize cross-sectional areas to minimize weight while satisfying constraints"""

        def objective(x):
            """Weight to minimize"""
            self.A_counter, self.A_tower, self.A_main, self.A_cables = x
            return self.calculate_weight()

        def stress_constraint(x):
            """Ensure stresses are below allowable"""
            self.A_counter, self.A_tower, self.A_main, self.A_cables = x
            U, _ = self.solve_displacements()
            stresses = self.calculate_stresses(U)

            # Return negative if constraint violated
            max_stress = max([abs(s['stress']) for s in stresses])
            return self.sigma_adm / 1e6 - max_stress

        def buckling_constraint(x):
            """Ensure buckling factor > 2.0"""
            self.A_counter, self.A_tower, self.A_main, self.A_cables = x
            buckling_factors = self.buckling_analysis()

            if buckling_factors:
                min_bf = min([bf['buckling_factor'] for bf in buckling_factors])
                return min_bf - 2.0
            return 0

        # Initial guess
        x0 = [0.01, 0.02, 0.01, 0.001]

        # Bounds (areas must be positive)
        bounds = [(0.001, 0.1), (0.001, 0.1), (0.001, 0.1), (0.0001, 0.01)]

        # Constraints
        constraints = [
            {'type': 'ineq', 'fun': stress_constraint},
            {'type': 'ineq', 'fun': buckling_constraint}
        ]

        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        if result.success:
            self.A_counter, self.A_tower, self.A_main, self.A_cables = result.x
            return result
        else:
            return None

    def plot_deformed_shape(self, U, scale=100):
        """Plot undeformed and deformed structure"""
        nodes = self.get_nodes()
        elements = self.get_elements()

        # Deformed nodes
        deformed_nodes = nodes.copy()
        for i in range(len(nodes)):
            deformed_nodes[i, 0] += U[2*i] * scale
            deformed_nodes[i, 1] += U[2*i+1] * scale

        plt.figure(figsize=(14, 8))

        # Plot undeformed
        for elem in elements:
            i, j, _, member_type = elem
            plt.plot([nodes[i, 0], nodes[j, 0]],
                    [nodes[i, 1], nodes[j, 1]],
                    'b--', linewidth=2, alpha=0.5, label='Undeformed' if elem == elements[0] else '')

        # Plot deformed
        for elem in elements:
            i, j, _, member_type = elem
            color = ['orange', 'red', 'green', 'gray'][member_type]
            plt.plot([deformed_nodes[i, 0], deformed_nodes[j, 0]],
                    [deformed_nodes[i, 1], deformed_nodes[j, 1]],
                    color=color, linewidth=2,
                    label=['Counter-jib', 'Tower', 'Main jib', 'Cable'][member_type]
                          if i == elem[0] and j == elem[1] else '')

        # Plot nodes
        plt.scatter(nodes[:, 0], nodes[:, 1], c='blue', s=100, zorder=5, alpha=0.5)
        plt.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], c='red', s=100, zorder=5)

        # Annotations
        for i, node in enumerate(nodes):
            plt.annotate(f'N{i}', (node[0], node[1]), fontsize=12, ha='right')

        plt.xlabel('X (m)', fontsize=12)
        plt.ylabel('Y (m)', fontsize=12)
        plt.title(f'Crane Deformation (Scale: {scale}x)', fontsize=14)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('crane_deformation.png', dpi=150)
        plt.close()

    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("="*70)
        print("TOWER CRANE STRUCTURAL ANALYSIS REPORT")
        print("="*70)

        # Solve
        U, nodes = self.solve_displacements()
        stresses = self.calculate_stresses(U)
        weight = self.calculate_weight()
        buckling = self.buckling_analysis()

        # Deformations
        print("\n1. EXTREME DEFORMATIONS")
        print("-" * 70)
        max_disp = 0
        max_node = -1
        for i in range(len(nodes)):
            disp = np.sqrt(U[2*i]**2 + U[2*i+1]**2)
            print(f"Node {i}: Ux = {U[2*i]*1000:.2f} mm, Uy = {U[2*i+1]*1000:.2f} mm, |U| = {disp*1000:.2f} mm")
            if disp > max_disp:
                max_disp = disp
                max_node = i
        print(f"\nMaximum displacement: {max_disp*1000:.2f} mm at Node {max_node}")

        # Stresses
        print("\n2. STRESSES")
        print("-" * 70)
        for s in stresses:
            status = "✓ OK" if abs(s['stress']) < self.sigma_adm/1e6 else "✗ FAIL"
            print(f"{s['member']:20s} ({s['type']:12s}): σ = {s['stress']:8.2f} MPa, F = {s['axial_force']:8.2f} kN [{status}]")

        max_stress = max([abs(s['stress']) for s in stresses])
        print(f"\nMaximum stress: {max_stress:.2f} MPa (Allowable: {self.sigma_adm/1e6:.2f} MPa)")

        # Weight
        print(f"\n3. WEIGHT")
        print("-" * 70)
        print(f"Total structural weight: {weight:.2f} kN ({weight/self.g:.2f} kg)")

        # Buckling
        print(f"\n4. BUCKLING LOAD FACTORS")
        print("-" * 70)
        for bf in buckling:
            status = "✓ OK" if bf['buckling_factor'] > 2.0 else "✗ FAIL"
            print(f"{bf['member']:20s}: λ = {bf['buckling_factor']:.2f} [{status}]")

        print("\n" + "="*70)

        # Plot
        self.plot_deformed_shape(U, scale=100)
        print("\nDeformation plot saved as 'crane_deformation.png'")

        return U, stresses, weight, buckling


# Run analysis
if __name__ == "__main__":
    crane = CraneModel()

    print("INITIAL DESIGN ANALYSIS")
    print("="*70)
    crane.generate_report()

    # Try different cross-sections
    print("\n\nTESTING IMPROVED DESIGN WITH LARGER MEMBERS...")
    print("="*70)
    crane.A_main = 0.05  # Increase main jib area
    crane.A_cables = 0.01  # Increase cable area significantly
    crane.A_counter = 0.02  # Increase counter-jib
    crane.A_tower = 0.03  # Increase tower

    print(f"Modified areas:")
    print(f"  Counter-jib: {crane.A_counter*1e4:.2f} cm²")
    print(f"  Tower: {crane.A_tower*1e4:.2f} cm²")
    print(f"  Main jib: {crane.A_main*1e4:.2f} cm²")
    print(f"  Cables: {crane.A_cables*1e4:.2f} cm²")

    crane.generate_report()

    # Optimal design analysis
    print("\n\nOPTIMAL DESIGN (Balanced stress/buckling)...")
    print("="*70)
    crane.A_main = 0.08  # Further increase for buckling
    crane.A_cables = 0.012
    crane.A_counter = 0.03
    crane.A_tower = 0.04

    print(f"Optimal areas:")
    print(f"  Counter-jib: {crane.A_counter*1e4:.2f} cm²")
    print(f"  Tower: {crane.A_tower*1e4:.2f} cm²")
    print(f"  Main jib: {crane.A_main*1e4:.2f} cm²")
    print(f"  Cables: {crane.A_cables*1e4:.2f} cm²")

    crane.generate_report()
