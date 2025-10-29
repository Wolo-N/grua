"""
Template Básico para Diseño de Grúa Torre
Sin dependencias externas - Solo Python nativo
"""

import math

def create_crane_geometry():
    """Crear geometría básica de grúa torre"""
    
    # Parámetros de diseño
    base_width = 4.0        # Ancho de la base [m]
    tower_height = 20.0     # Altura de la torre [m]
    jib_length = 15.0       # Longitud del brazo [m]
    jib_height = 22.0       # Altura del brazo [m]
    
    # Coordenadas de nodos (x, y)
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
    
    # Conectividad de elementos (nodo1, nodo2)
    elements = []
    
    # Elementos de la base (0-3)
    elements.append([0, 1])  # 0
    elements.append([1, 2])  # 1
    elements.append([2, 3])  # 2
    elements.append([3, 0])  # 3
    
    # Elementos de la torre (4-6)
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
    
    return nodes, elements

def calculate_distance(node1, node2):
    """Calcular distancia entre dos nodos"""
    dx = node2[0] - node1[0]
    dy = node2[1] - node1[1]
    return math.sqrt(dx*dx + dy*dy)

def print_crane_structure(nodes, elements):
    """Imprimir información de la estructura"""
    print("\n" + "="*60)
    print("           ESTRUCTURA DE LA GRÚA TORRE")
    print("="*60)
    print(f"Total de nodos: {len(nodes)}")
    print(f"Total de elementos: {len(elements)}")
    
    print(f"\nCOORDENADAS DE NODOS:")
    print("Nodo |   X   |   Y   | Descripción")
    print("-" * 50)
    descriptions = [
        "Base esquina SW", "Base esquina SE", "Base esquina NE", "Base esquina NW",
        "Torre 5m", "Torre 10m", "Torre 15m", "Torre 20m",
        "Brazo 1/4", "Brazo 2/4", "Brazo 3/4", "Brazo punta",
        "Soporte superior", "Soporte punta"
    ]
    
    for i, (node, desc) in enumerate(zip(nodes, descriptions)):
        print(f"{i:4d} | {node[0]:5.1f} | {node[1]:5.1f} | {desc}")
    
    print(f"\nCONECTIVIDAD DE ELEMENTOS:")
    print("Elem | Nodo1 | Nodo2 | Longitud | Tipo")
    print("-" * 45)
    
    element_types = [
        "Base S", "Base E", "Base N", "Base W",
        "Torre", "Torre", "Torre",
        "Base-Torre", "Base-Torre", "Base-Torre", "Base-Torre",
        "Brazo", "Brazo", "Brazo", "Brazo",
        "Soporte", "Soporte Diag", "Soporte Diag", "Soporte", "Cable Principal"
    ]
    
    total_length = 0
    max_length = 0
    for i, (elem, tipo) in enumerate(zip(elements, element_types)):
        n1, n2 = elem
        length = calculate_distance(nodes[n1], nodes[n2])
        total_length += length
        max_length = max(max_length, length)
        print(f"{i:4d} | {n1:5d} | {n2:5d} | {length:8.2f} | {tipo}")
    
    print(f"\nRESUMEN DE LA ESTRUCTURA:")
    print(f"- Longitud total: {total_length:.2f} m")
    print(f"- Elemento más largo: {max_length:.2f} m")

def analyze_crane_components(nodes, elements):
    """Analizar componentes de la grúa"""
    print(f"\nANÁLISIS POR COMPONENTES:")
    print("-" * 40)
    
    # Análisis de la base
    base_elements = elements[0:4]
    base_length = sum(calculate_distance(nodes[e[0]], nodes[e[1]]) for e in base_elements)
    print(f"BASE:")
    print(f"  - 4 elementos, perímetro: {base_length:.2f} m")
    print(f"  - Área base: {nodes[1][0] * nodes[2][1]:.2f} m²")
    
    # Análisis de la torre
    tower_elements = elements[4:7]
    tower_length = sum(calculate_distance(nodes[e[0]], nodes[e[1]]) for e in tower_elements)
    print(f"TORRE:")
    print(f"  - {len(tower_elements)} elementos verticales")
    print(f"  - Altura total: {tower_length:.2f} m")
    print(f"  - 4 elementos de conexión a la base")
    
    # Análisis del brazo
    jib_elements = elements[11:15]
    jib_length = sum(calculate_distance(nodes[e[0]], nodes[e[1]]) for e in jib_elements)
    print(f"BRAZO HORIZONTAL:")
    print(f"  - {len(jib_elements)} elementos principales")
    print(f"  - Longitud total: {jib_length:.2f} m")
    print(f"  - Altura de trabajo: {nodes[8][1]:.1f} m")
    
    # Análisis de soportes
    support_elements = elements[15:]
    print(f"SISTEMA DE SOPORTE:")
    print(f"  - {len(support_elements)} elementos de soporte")
    print(f"  - Incluye tirantes y cables de estabilización")

def define_loads_and_materials():
    """Definir cargas típicas y materiales"""
    print(f"\nCARGAS DE DISEÑO:")
    print("-" * 30)
    print("CARGAS OPERACIONALES:")
    print(f"  - Carga nominal en punta: 50 kN")
    print(f"  - Peso propio brazo: 20 kN (distribuido)")
    print(f"  - Peso propio torre: 40 kN (distribuido)")
    print(f"  - Carga de viento: Por calcular según norma")
    
    print(f"\nMATERIALES PROPUESTOS:")
    print("-" * 25)
    print(f"ACERO ESTRUCTURAL:")
    print(f"  - Módulo de elasticidad: 200 GPa")
    print(f"  - Resistencia a fluencia: 355 MPa")
    print(f"  - Densidad: 7850 kg/m³")
    
    print(f"\nSECCIONES TRANSVERSALES:")
    print("-" * 30)
    print(f"  - Elementos base: Perfiles HEB o tubos □200x200")
    print(f"  - Torre: Perfiles HEB200 o tubos □150x150")
    print(f"  - Brazo: Perfiles IPE300 o UPN200")
    print(f"  - Soportes: Perfiles L100x100 o tubos ⌀89")
    print(f"  - Cables: Cables de acero ⌀20mm")

def next_steps():
    """Próximos pasos en el desarrollo"""
    print(f"\n" + "="*60)
    print("         PRÓXIMOS PASOS PARA EL DESARROLLO")
    print("="*60)
    
    steps = [
        "1. ANÁLISIS ESTRUCTURAL COMPLETO",
        "   - Implementar método de elementos finitos",
        "   - Calcular desplazamientos y rotaciones",
        "   - Determinar fuerzas internas en elementos",
        "",
        "2. VERIFICACIÓN DE RESISTENCIA",
        "   - Verificar tensiones admisibles",
        "   - Controlar deflexiones máximas",
        "   - Verificar estabilidad local y global",
        "",
        "3. OPTIMIZACIÓN DEL DISEÑO",
        "   - Optimizar secciones transversales",
        "   - Minimizar peso total",
        "   - Cumplir restricciones de seguridad",
        "",
        "4. ANÁLISIS AVANZADOS",
        "   - Análisis dinámico (resonancia)",
        "   - Efectos de viento y sismo",
        "   - Análisis de fatiga",
        "",
        "5. IMPLEMENTACIÓN Y VALIDACIÓN",
        "   - Crear modelo detallado en software CAE",
        "   - Validar con resultados analíticos",
        "   - Generar planos de construcción"
    ]
    
    for step in steps:
        print(step)

def main():
    """Función principal"""
    print("*" * 60)
    print("    TEMPLATE PARA DISEÑO DE GRÚA TORRE")
    print("    Análisis Estructural Preliminar")
    print("*" * 60)
    
    # Crear geometría
    nodes, elements = create_crane_geometry()
    
    # Mostrar información completa
    print_crane_structure(nodes, elements)
    analyze_crane_components(nodes, elements)
    define_loads_and_materials()
    next_steps()
    
    print(f"\n" + "*" * 60)
    print("    TEMPLATE COMPLETADO EXITOSAMENTE")
    print("*" * 60)
    
    return nodes, elements

if __name__ == "__main__":
    nodes, elements = main()
