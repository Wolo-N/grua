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

def plot_crane(X, C, title="Estructura de Grúa", scale_factor=1):
    """Plot crane structure with different colors for different parts"""
    try:
        plt.figure(figsize=(12, 8))
        
        # Definir colores para diferentes partes
        base_elements = [0, 1, 2, 3]  # Elementos de la base
        tower_elements = [4, 5, 6, 7, 8, 9, 10]  # Torre
        jib_elements = [11, 12, 13, 14]  # Brazo horizontal
        support_elements = list(range(15, C.shape[0]))  # Elementos de soporte
        
        # Plot diferentes partes con diferentes colores
        for iEl in range(C.shape[0]):
            if iEl in base_elements:
                color = 'blue'
                linewidth = 3
                label = 'Base' if iEl == base_elements[0] else ""
            elif iEl in tower_elements:
                color = 'red'
                linewidth = 2
                label = 'Torre' if iEl == tower_elements[0] else ""
            elif iEl in jib_elements:
                color = 'green'
                linewidth = 2
                label = 'Brazo' if iEl == jib_elements[0] else ""
            else:
                color = 'orange'
                linewidth = 1
                label = 'Soporte' if iEl == support_elements[0] else ""
            
            plt.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                    [X[C[iEl,0],1], X[C[iEl,1],1]], 
                    color=color, linewidth=linewidth, label=label)
        
        # Plot nodos
        plt.scatter(X[:,0], X[:,1], c='black', s=30, zorder=5)
        
        # Numeración de nodos
        for i in range(X.shape[0]):
            plt.annotate(f'{i}', (X[i,0], X[i,1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
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
    """Crear geometría básica de grúa torre"""
    
    # Parámetros de diseño
    base_width = 4.0        # Ancho de la base [m]
    tower_height = 20.0     # Altura de la torre [m]
    jib_length = 15.0       # Longitud del brazo [m]
    jib_height = 22.0       # Altura del brazo [m]
    
    # Matriz de coordenadas de nodos (x, y)
    nodes = []
    
    # Nodos de la base (0-3) - cuadrado en el suelo
    nodes.append([0.0, 0.0])                    # 0: esquina base
    nodes.append([base_width, 0.0])             # 1: esquina base
    nodes.append([base_width, base_width])      # 2: esquina base
    nodes.append([0.0, base_width])             # 3: esquina base
    
    # Nodos de la torre (4-7) - cada 5m de altura
    center_x = base_width / 2
    center_y = base_width / 2
    for i in range(4):
        height = (i + 1) * 5.0
        nodes.append([center_x, height])         # 4,5,6,7: nodos torre
    
    # Nodos del brazo horizontal (8-11)
    jib_y = jib_height
    for i in range(4):
        x_pos = center_x + (i + 1) * (jib_length / 4)
        nodes.append([x_pos, jib_y])            # 8,9,10,11: nodos brazo
    
    # Nodos de soporte del brazo (12-13)
    nodes.append([center_x + jib_length/2, jib_y + 3])  # 12: soporte superior
    nodes.append([center_x + jib_length, jib_y + 2])    # 13: soporte punta
    
    X = np.array(nodes)
    
    # Matriz de conectividad de elementos (nodo1, nodo2)
    elements = []
    
    # Elementos de la base (0-3)
    elements.append([0, 1])  # 0
    elements.append([1, 2])  # 1
    elements.append([2, 3])  # 2
    elements.append([3, 0])  # 3
    
    # Elementos de la torre (4-9)
    elements.append([4, 5])  # 4: torre vertical
    elements.append([5, 6])  # 5: torre vertical
    elements.append([6, 7])  # 6: torre vertical
    
    # Conexiones base-torre (7-9)
    elements.append([0, 4])  # 7
    elements.append([1, 4])  # 8
    elements.append([2, 4])  # 9
    elements.append([3, 4])  # 10
    
    # Elementos del brazo horizontal (10-14)
    elements.append([7, 8])   # 11: inicio brazo
    elements.append([8, 9])   # 12: brazo
    elements.append([9, 10])  # 13: brazo
    elements.append([10, 11]) # 14: brazo
    
    # Elementos de soporte del brazo (15-18)
    elements.append([7, 12])  # 15: soporte torre-arriba
    elements.append([12, 9])  # 16: soporte diagonal
    elements.append([12, 11]) # 17: soporte diagonal
    elements.append([11, 13]) # 18: soporte punta
    elements.append([7, 13])  # 19: cable soporte principal
    
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
    
    # Fijar la base completamente
    bc[0:4, :] = True  # Nodos 0-3 fijos en x e y
    
    # Matriz de cargas: Fx, Fy por nodo [N]
    loads = np.zeros([num_nodes, 2])
    
    # Carga en la punta del brazo (simular carga levantada)
    loads[11, 1] = -50000  # 50 kN hacia abajo en nodo 11
    
    # Peso propio del brazo (distribuido)
    for i in range(8, 12):  # Nodos del brazo
        loads[i, 1] = -5000  # 5 kN por nodo
    
    # Peso de la torre
    for i in range(4, 8):  # Nodos de la torre
        loads[i, 1] = -10000  # 10 kN por nodo
    
    return bc, loads

def main():
    """Función principal para análisis de la grúa"""
    
    print("=== ANÁLISIS ESTRUCTURAL DE GRÚA TORRE ===\n")
    
    # 1. Crear geometría
    print("1. Creando geometría de la grúa...")
    X, C = create_crane_geometry()
    print(f"   Nodos: {X.shape[0]}")
    print(f"   Elementos: {C.shape[0]}")
    
    # 2. Mostrar estructura
    plot_crane(X, C, "Geometría Inicial de la Grúa")
    
    # 3. Definir materiales
    E, sections = define_materials_and_sections()
    print(f"   Módulo de elasticidad: {E/1e9:.0f} GPa")
    
    # 4. Condiciones de borde y cargas
    bc, loads = apply_boundary_conditions_and_loads(X.shape[0])
    print(f"   Nodos fijos: {np.sum(bc)}")
    print(f"   Carga total aplicada: {np.sum(loads[:, 1])/1000:.0f} kN")
    
    # 5. Preparar para análisis FEM
    bc_mask = bc.reshape(-1)
    load_vector = loads.reshape(-1)[~bc_mask]
    
    print("\n=== TEMPLATE CREADO EXITOSAMENTE ===")
    print("Próximos pasos:")
    print("- Refinar la geometría según necesidades específicas")
    print("- Implementar el análisis FEM completo")
    print("- Añadir verificación de resistencia")
    print("- Optimizar el diseño")
    
    return X, C, E, sections, bc, loads

if __name__ == "__main__":
    X, C, E, sections, bc, loads = main()
