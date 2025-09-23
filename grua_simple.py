# -*- coding: utf-8 -*-
"""
Template Básico para Diseño de Grúa Torre
Modelado usando Método de Elementos Finitos
"""

import numpy as np

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
    
    # Conexiones base-torre (7-10)
    elements.append([0, 4])  # 7
    elements.append([1, 4])  # 8
    elements.append([2, 4])  # 9
    elements.append([3, 4])  # 10
    
    # Elementos del brazo horizontal (11-14)
    elements.append([7, 8])   # 11: inicio brazo
    elements.append([8, 9])   # 12: brazo
    elements.append([9, 10])  # 13: brazo
    elements.append([10, 11]) # 14: brazo
    
    # Elementos de soporte del brazo (15-19)
    elements.append([7, 12])  # 15: soporte torre-arriba
    elements.append([12, 9])  # 16: soporte diagonal
    elements.append([12, 11]) # 17: soporte diagonal
    elements.append([11, 13]) # 18: soporte punta
    elements.append([7, 13])  # 19: cable soporte principal
    
    C = np.array(elements)
    
    return X, C

def print_crane_structure(X, C):
    """Imprimir información de la estructura"""
    print("\n=== ESTRUCTURA DE LA GRÚA ===")
    print(f"Total de nodos: {X.shape[0]}")
    print(f"Total de elementos: {C.shape[0]}")
    
    print("\nCOORDENADAS DE NODOS:")
    print("Nodo |   X   |   Y   | Descripción")
    print("-" * 40)
    descriptions = [
        "Base esquina 1", "Base esquina 2", "Base esquina 3", "Base esquina 4",
        "Torre 5m", "Torre 10m", "Torre 15m", "Torre 20m",
        "Brazo 1", "Brazo 2", "Brazo 3", "Brazo punta",
        "Soporte superior", "Soporte punta"
    ]
    
    for i, (node, desc) in enumerate(zip(X, descriptions)):
        print(f"{i:4d} | {node[0]:5.1f} | {node[1]:5.1f} | {desc}")
    
    print("\nCONECTIVIDAD DE ELEMENTOS:")
    print("Elem | Nodo1 | Nodo2 | Tipo")
    print("-" * 30)
    
    element_types = [
        "Base", "Base", "Base", "Base",
        "Torre", "Torre", "Torre",
        "Base-Torre", "Base-Torre", "Base-Torre", "Base-Torre",
        "Brazo", "Brazo", "Brazo", "Brazo",
        "Soporte", "Soporte", "Soporte", "Soporte", "Cable"
    ]
    
    for i, (elem, tipo) in enumerate(zip(C, element_types)):
        print(f"{i:4d} | {elem[0]:5d} | {elem[1]:5d} | {tipo}")

def calculate_element_lengths(X, C):
    """Calcular longitudes de elementos"""
    print("\nLONGITUDES DE ELEMENTOS:")
    print("Elem | Longitud [m] | Tipo")
    print("-" * 30)
    
    element_types = [
        "Base", "Base", "Base", "Base",
        "Torre", "Torre", "Torre",
        "Base-Torre", "Base-Torre", "Base-Torre", "Base-Torre",
        "Brazo", "Brazo", "Brazo", "Brazo",
        "Soporte", "Soporte", "Soporte", "Soporte", "Cable"
    ]
    
    total_length = 0
    for i, (elem, tipo) in enumerate(zip(C, element_types)):
        n1, n2 = elem
        length = np.linalg.norm(X[n2] - X[n1])
        total_length += length
        print(f"{i:4d} | {length:8.2f}   | {tipo}")
    
    print(f"\nLongitud total de estructura: {total_length:.2f} m")

def define_loads_and_constraints():
    """Definir cargas y restricciones"""
    print("\nCONDICIONES DE BORDE:")
    print("- Nodos 0-3 (base): Completamente fijos")
    print("- Resto de nodos: Libres")
    
    print("\nCARGAS APLICADAS:")
    print("- Nodo 11 (punta brazo): 50 kN hacia abajo")
    print("- Nodos 8-11 (brazo): 5 kN/nodo peso propio")
    print("- Nodos 4-7 (torre): 10 kN/nodo peso propio")
    print("- Carga total: 110 kN")

def main():
    """Función principal"""
    print("=" * 50)
    print("    TEMPLATE PARA DISEÑO DE GRÚA TORRE")
    print("=" * 50)
    
    # Crear geometría
    X, C = create_crane_geometry()
    
    # Mostrar información
    print_crane_structure(X, C)
    calculate_element_lengths(X, C)
    define_loads_and_constraints()
    
    print("\n" + "=" * 50)
    print("PRÓXIMOS PASOS PARA EL DESARROLLO:")
    print("=" * 50)
    print("1. Implementar análisis FEM completo")
    print("2. Calcular desplazamientos y fuerzas")
    print("3. Verificar resistencia de elementos")
    print("4. Optimizar secciones transversales")
    print("5. Añadir análisis de estabilidad")
    print("6. Considerar efectos dinámicos")
    
    return X, C

if __name__ == "__main__":
    X, C = main()
