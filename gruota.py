# -*- coding: utf-8 -*-
"""
Template para Diseño de Grúa Torre
Modelado usando Método de Elementos Finitos
"""

import numpy as np
import matplotlib.pyplot as plt

def element_stiffness(n1, n2, A, E):
    '''Returns Rod Elemental Matrix'''
    d = n2 - n1
    L = np.linalg.norm(d)
    c = d[0] / L
    s = d[1] / L
    k_local = (A * E / L) * np.array([[ 1, -1],[-1,  1]], dtype=float)
    T = np.array([[ c, s, 0, 0],[ 0, 0, c, s]], dtype=float)
    k_structural = np.matmul(T.T, np.matmul(k_local, T))
    return k_structural

def dof_el(nnod1, nnod2):
    '''Returns Elemental dof for Assembly'''
    return [2*(nnod1+1)-2,2*(nnod1+1)-1,2*(nnod2+1)-2,2*(nnod2+1)-1]

def plot_crane(X, C, bc=None, title="Estructura de Grúa", scale_factor=1):
    """Plot crane structure with different colors for different parts"""
    try:
        plt.figure(figsize=(14, 8))

        # Definir elementos por tipo según la imagen
        base_tower = [0, 1, 2]  # Base/Torre (azul, grueso)
        cuerda_superior = list(range(3, 10))  # Cuerda superior (verde, media)
        diagonales = list(range(10, 17))  # Diagonales (naranja/amarillo)
        montantes = list(range(17, 25))  # Montantes (morado)
        cables_soporte = list(range(25, C.shape[0]))  # Cables de soporte (negro punteado)

        # Plot elementos con estilos según tipo
        plotted_labels = set()

        for iEl in range(C.shape[0]):
            if iEl in base_tower:
                color = 'blue'
                linewidth = 4
                linestyle = '-'
                label = 'Base/Torre'
            elif iEl in cuerda_superior:
                color = 'green'
                linewidth = 2.5
                linestyle = '-'
                label = 'Cuerda Superior'
            elif iEl in diagonales:
                color = 'orange'
                linewidth = 1.5
                linestyle = '-'
                label = 'Diagonales'
            elif iEl in montantes:
                color = 'purple'
                linewidth = 1.5
                linestyle = '-'
                label = 'Montantes'
            else:  # cables_soporte
                color = 'black'
                linewidth = 1.5
                linestyle = '--'
                label = 'Cables Soporte'

            # Solo agregar label si no ha sido agregado antes
            if label in plotted_labels:
                label = ""
            else:
                plotted_labels.add(label)

            plt.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                    [X[C[iEl,0],1], X[C[iEl,1],1]],
                    color=color, linewidth=linewidth, linestyle=linestyle, label=label)

        # Plot nodos según tipo (fijos = rojos cuadrados, libres = verdes círculos)
        if bc is not None:
            # Nodos fijos (ambas coordenadas fijas)
            fixed_nodes = np.where(np.all(bc, axis=1))[0]
            free_nodes = np.where(~np.all(bc, axis=1))[0]

            # Nodos fijos (rojos, cuadrados)
            if len(fixed_nodes) > 0:
                plt.scatter(X[fixed_nodes,0], X[fixed_nodes,1],
                           c='red', s=150, marker='s', edgecolors='black',
                           linewidths=1.5, zorder=5, label='Nodos Fijos (Apoyos)')

            # Nodos libres (verdes, círculos)
            if len(free_nodes) > 0:
                plt.scatter(X[free_nodes,0], X[free_nodes,1],
                           c='lime', s=120, marker='o', edgecolors='black',
                           linewidths=1.5, zorder=5, label='Nodos Libres')
        else:
            # Si no hay info de bc, plot todos como círculos negros
            plt.scatter(X[:,0], X[:,1], c='black', s=50, zorder=5)

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error al mostrar gráfico: {e}")
        print("Información de la estructura:")
        print(f"Nodos (x, y):")
        for i, node in enumerate(X):
            print(f"  Nodo {i}: ({node[0]:.1f}, {node[1]:.1f})")
        print(f"Elementos (nodo1, nodo2):")
        for i, elem in enumerate(C):
            print(f"  Elemento {i}: {elem[0]} -> {elem[1]}")

def create_crane_geometry():
    """Crear geometría de grúa torre tipo truss según imagen"""

    # Parámetros de diseño basados en la imagen
    tower_base_y = 0.0          # Base en el suelo
    tower_top_y = 13.0          # Altura de la torre ~13m
    tower_x_left = 0.0          # Posición x izquierda
    tower_x_right = 3.0         # Ancho de la torre ~3m

    # Dimensiones del brazo horizontal (jib)
    jib_start_x = tower_x_right
    jib_length = 24.0           # Longitud total ~24m
    jib_y_lower = 11.5          # Cuerda inferior (diagonales)
    jib_y_upper = 14.0          # Cuerda superior
    num_panels = 7              # Número de paneles en el brazo

    # Matriz de coordenadas de nodos (x, y)
    nodes = []

    # NODOS DE LA TORRE (0-3)
    nodes.append([tower_x_left, tower_base_y])      # 0: base izquierda (fijo)
    nodes.append([tower_x_right, tower_base_y])     # 1: base derecha (fijo)
    nodes.append([tower_x_left, tower_top_y])       # 2: tope izquierdo
    nodes.append([tower_x_right, tower_top_y])      # 3: tope derecho

    # NODOS DEL BRAZO - Cuerda superior (4-10)
    for i in range(num_panels + 1):
        x = jib_start_x + i * (jib_length / num_panels)
        nodes.append([x, jib_y_upper])              # 4-10: cuerda superior

    # NODOS DEL BRAZO - Cuerda inferior / diagonales (11-17)
    for i in range(num_panels + 1):
        x = jib_start_x + i * (jib_length / num_panels)
        nodes.append([x, jib_y_lower])              # 11-17: cuerda inferior

    X = np.array(nodes)

    # Matriz de conectividad de elementos (nodo1, nodo2)
    elements = []

    # === ELEMENTOS BASE/TORRE (0-2) - AZUL ===
    elements.append([0, 1])  # 0: base horizontal
    elements.append([0, 2])  # 1: torre izquierda vertical
    elements.append([1, 3])  # 2: torre derecha vertical

    # === ELEMENTOS CUERDA SUPERIOR (3-9) - VERDE ===
    for i in range(num_panels):
        elements.append([4 + i, 4 + i + 1])  # 3-9: cuerda superior

    # === ELEMENTOS DIAGONALES (10-16) - NARANJA ===
    for i in range(num_panels):
        elements.append([11 + i, 11 + i + 1])  # 10-16: cuerda inferior/diagonales

    # === ELEMENTOS MONTANTES (17-24) - MORADO ===
    # Montantes verticales conectando cuerda superior e inferior
    for i in range(num_panels + 1):
        elements.append([4 + i, 11 + i])  # 17-24: montantes verticales (8 elementos)

    # === ELEMENTOS CABLES SOPORTE (25+) - NEGRO PUNTEADO ===
    # Diagonales tipo Warren truss
    for i in range(num_panels):
        if i % 2 == 0:
            # Diagonal ascendente
            elements.append([11 + i, 4 + i + 1])  # diagonal /
        else:
            # Diagonal descendente
            elements.append([4 + i, 11 + i + 1])  # diagonal \

    # Conexiones adicionales de soporte
    elements.append([2, 4])  # Conexión torre-brazo izquierda superior
    elements.append([3, 4])  # Conexión torre-brazo derecha superior

    C = np.array(elements)

    return X, C

def define_materials_and_sections():
    """Definir materiales y secciones"""
    
    # Acero estructural
    E = 200e9  # Módulo de elasticidad [Pa]
    
    # Secciones transversales [m²]
    sections = {
        'base': 0.01,      # Elementos base - sección grande
        'tower': 0.008,    # Torre - sección media
        'jib': 0.005,      # Brazo - sección media
        'support': 0.003,  # Soportes - sección pequeña
        'cable': 0.001     # Cables - sección muy pequeña
    }
    
    return E, sections

def apply_boundary_conditions_and_loads(num_nodes):
    """Aplicar condiciones de borde y cargas"""

    # Condiciones de borde: False = libre, True = fijo
    bc = np.full((num_nodes, 2), False)

    # Fijar solo los dos nodos de la base (0 y 1)
    bc[0, :] = True  # Nodo 0 fijo en x e y
    bc[1, :] = True  # Nodo 1 fijo en x e y

    # Matriz de cargas: Fx, Fy por nodo [N]
    loads = np.zeros([num_nodes, 2])

    # Carga en la punta del brazo (simular carga levantada)
    # El último nodo de la cuerda superior es el nodo 10
    loads[10, 1] = -50000  # 50 kN hacia abajo en la punta

    # Cargas distribuidas en los nodos del brazo
    for i in range(4, 11):  # Nodos de la cuerda superior
        loads[i, 1] = -5000  # 5 kN por nodo (peso propio)

    return bc, loads

def main():
    """Función principal para análisis de la grúa"""

    print("=== ANÁLISIS ESTRUCTURAL DE GRÚA TORRE ===\n")

    # 1. Crear geometría
    print("1. Creando geometría de la grúa...")
    X, C = create_crane_geometry()
    print(f"   Nodos: {X.shape[0]}")
    print(f"   Elementos: {C.shape[0]}")

    # 2. Condiciones de borde y cargas
    bc, loads = apply_boundary_conditions_and_loads(X.shape[0])
    print(f"   Nodos fijos: {np.sum(np.all(bc, axis=1))}")
    print(f"   Carga total aplicada: {np.sum(loads[:, 1])/1000:.0f} kN")

    # 3. Mostrar estructura con nodos fijos y libres diferenciados
    plot_crane(X, C, bc, "Grúa Torre - Nodos Fijos (Rojos) y Libres (Verdes)")

    # 4. Definir materiales
    E, sections = define_materials_and_sections()
    print(f"   Módulo de elasticidad: {E/1e9:.0f} GPa")

    # 5. Preparar para análisis FEM
    bc_mask = bc.reshape(-1)
    load_vector = loads.reshape(-1)[~bc_mask]

    print("\n=== ESTRUCTURA CREADA EXITOSAMENTE ===")
    print("Próximos pasos:")
    print("- Implementar el análisis FEM completo")
    print("- Añadir verificación de resistencia")
    print("- Optimizar el diseño")

    return X, C, E, sections, bc, loads

if __name__ == "__main__":
    X, C, E, sections, bc, loads = main()
