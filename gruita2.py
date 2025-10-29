import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import time

# Importar utilidades compartidas
from fem_utils import *
from analysis_utils import *

def plot_truss(X, C, title="Truss Structure", scale_factor=1, show_nodes=True):
    '''Graficar estructura de cercha con numeración opcional de nodos'''
    plt.figure(figsize=(15, 8))

    # Graficar elementos
    for iEl in range(C.shape[0]):
        plt.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                [X[C[iEl,0],1], X[C[iEl,1],1]],
                'b-', linewidth=2)

    # Graficar nodos
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
    Graficar estructura de cercha con visualización de mapa de calor de tensiones

    Parámetros:
    -----------
    X : ndarray
        Coordenadas de nodos
    C : ndarray
        Matriz de conectividad
    stresses : ndarray
        Valores de tensión para cada elemento (Pa)
    element_areas : list
        Áreas de sección transversal para cada elemento
    title : str
        Título del gráfico
    '''
    fig, ax = plt.subplots(figsize=(16, 10))

    # Crear segmentos de línea para cada elemento
    segments = []
    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        segments.append([(X[n1, 0], X[n1, 1]), (X[n2, 0], X[n2, 1])])

    # Normalizar valores de tensión para mapa de colores
    stress_mpa = stresses / 1e6  # Convertir a MPa
    norm = Normalize(vmin=np.min(stress_mpa), vmax=np.max(stress_mpa))

    # Crear mapa de colores (azul para compresión, rojo para tracción)
    cmap = cm.RdBu_r  # Invertido para que rojo sea tracción, azul sea compresión

    # Crear colección de líneas con colores basados en tensión
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=4)
    lc.set_array(stress_mpa)

    # Agregar colección de líneas al gráfico
    line = ax.add_collection(lc)

    # Agregar barra de color
    cbar = fig.colorbar(line, ax=ax, pad=0.02)
    cbar.set_label('Stress (MPa)\n← Compression | Tension →', rotation=270, labelpad=25)

    # Graficar nodos
    ax.scatter(X[:, 0], X[:, 1], c='black', s=30, zorder=5, alpha=0.5)

    # Etiquetas y formato
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Auto-escalar para ajustar datos
    ax.autoscale()

    plt.tight_layout()
    plt.show()

    # Imprimir estadísticas de tensión
    print("\n" + "="*60)
    print("RESUMEN DE ANÁLISIS DE TENSIONES")
    print("="*60)
    print(f"Tensión de Tracción Máxima:     {np.max(stress_mpa):8.2f} MPa")
    print(f"Tensión de Compresión Máxima: {np.min(stress_mpa):8.2f} MPa")
    print(f"Magnitud de Tensión Promedio:   {np.mean(np.abs(stress_mpa)):8.2f} MPa")
    print("="*60)

def design_crane_geometry():
    '''Diseñar geometría de cercha de grúa de 30m'''

    # Parámetros de diseño
    crane_length = 30.0  # m
    tower_height = 0.0  # m
    boom_height = 1.0    # m profundidad del brazo
    n_boom_segments = 12  # Número de segmentos a lo largo del brazo

    # Calcular posiciones para brazo horizontal
    boom_x = np.linspace(0, crane_length, n_boom_segments + 1)
    boom_y_top = np.full(n_boom_segments + 1, tower_height + boom_height)  # Cuerda superior horizontal
    boom_y_bot = np.full(n_boom_segments + 1, tower_height)  # Cuerda inferior horizontal

    # Inicializar coordenadas de nodos
    total_nodes = 1 + 2 * (n_boom_segments + 1)  # Torre + todos los nodos superior e inferior del brazo
    X = np.zeros([total_nodes, 2], float)

    node_idx = 0

    # Base y parte superior de la torre
    X[0] = [0, tower_height + 5]  # Base de la torre
    tower_base = 0
    node_idx += 1

    # Nodos de cuerda superior del brazo (mantener todos los nodos para forma trapezoidal)
    boom_top_nodes = []
    for i in range(n_boom_segments + 1):
        X[node_idx] = [boom_x[i], boom_y_top[i]]
        boom_top_nodes.append(node_idx)
        node_idx += 1

    # Nodos de cuerda inferior del brazo (mantener todos)
    boom_bot_nodes = []
    for i in range(n_boom_segments + 1):
        X[node_idx] = [boom_x[i], boom_y_bot[i]]
        boom_bot_nodes.append(node_idx)
        node_idx += 1

    return X, tower_base, boom_top_nodes, boom_bot_nodes

def create_crane_connectivity(tower_base, boom_top_nodes, boom_bot_nodes):
    '''Crear matriz de conectividad de elementos para la grúa'''

    elements = []

    # PUNTALES (miembros en compresión - elementos estructurales rígidos)

    # Elementos de puntal superior (miembros horizontales superiores)
    for i in range(len(boom_top_nodes) - 1):
        elements.append([boom_top_nodes[i], boom_top_nodes[i+1]])

    # Elementos de puntal inferior (miembros horizontales inferiores)
    for i in range(len(boom_bot_nodes) - 1):
        elements.append([boom_bot_nodes[i], boom_bot_nodes[i+1]])

    # Puntales verticales (conectan cada nodo superior al nodo inferior correspondiente)
    # Ambos arrays tienen la misma longitud, así que mapeo directo 1 a 1
    for i in range(len(boom_top_nodes)):
        elements.append([boom_top_nodes[i], boom_bot_nodes[i]])

    # CABLES (miembros en tracción - cables flexibles)

    # Cables diagonales (patrón alternado en todo)
    for i in range(len(boom_top_nodes) - 1):
        if i % 2 == 0:  # Índices pares: inferior-izquierda a superior-derecha
            elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
        else:  # Índices impares: superior-izquierda a inferior-derecha
            elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])

    # Cables de soporte desde la torre (nodo 0) al brazo
    elements.append([tower_base, boom_top_nodes[-2]])
    elements.append([tower_base, boom_top_nodes[-8]])  # Segundo cable para estabilidad

    return np.array(elements, dtype=int)

def crane_simulation():
    '''Función principal de simulación de grúa'''

    print("Diseñando Estructura de Grúa de 30m...")

    # Crear geometría
    X, tower_base, boom_top_nodes, boom_bot_nodes = design_crane_geometry()
    C = create_crane_connectivity(tower_base, boom_top_nodes, boom_bot_nodes)

    print(f"Estructura creada con {X.shape[0]} nodos y {C.shape[0]} elementos")

    # Depuración: Imprimir información de conectividad
    print(f"\nPosición Nodo 0 (tower_base): {X[0]}")
    print(f"boom_top_nodes: {boom_top_nodes}")
    print(f"boom_bot_nodes: {boom_bot_nodes}")
    print(f"\nPrimeros 10 elementos:")
    for i in range(min(10, C.shape[0])):
        print(f"  Elemento {i}: nodos {C[i,0]} a {C[i,1]}")
    print(f"\nÚltimos 5 elementos:")
    for i in range(max(0, C.shape[0]-5), C.shape[0]):
        print(f"  Elemento {i}: nodos {C[i,0]} a {C[i,1]}")

    # Graficar estructura original
    plot_truss(X, C, "30m Crane Truss Structure - Original", show_nodes=True)

    # Propiedades de material y sección
    E = 200e9  # Módulo de Young del acero (Pa)

    # Especificaciones de sección transversal circular hueca (en metros)
    # Puntales (miembros en compresión: superior/inferior/vertical)
    d_outer_strut = 0.050  # 50 mm diámetro exterior (máximo permitido)
    d_inner_strut = 0.040  # 40 mm diámetro interior (5 mm espesor de pared)
    A_strut, I_strut = hollow_circular_section(d_outer_strut, d_inner_strut)

    # Cables (miembros en tracción: diagonales/soportes)
    d_outer_cable = 0.050  # 25 mm diámetro exterior
    d_inner_cable = 0.000  # 20 mm diámetro interior (2.5 mm espesor de pared)
    A_cable, I_cable = hollow_circular_section(d_outer_cable, d_inner_cable)

    print(f"\nPropiedades de sección transversal:")
    print(f"  Puntales: D_ext={d_outer_strut*1000:.1f}mm, D_int={d_inner_strut*1000:.1f}mm, A={A_strut*1e4:.2f}cm², I={I_strut*1e8:.2f}cm⁴")
    print(f"  Cables: D_ext={d_outer_cable*1000:.1f}mm, D_int={d_inner_cable*1000:.1f}mm, A={A_cable*1e4:.2f}cm², I={I_cable*1e8:.2f}cm⁴")

    # Condiciones de borde
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    # Fijar primer nodo inferior del brazo (más a la izquierda) - completamente fijo (apoyo de pasador)
    bc[boom_bot_nodes[0], :] = True  # Apoyo fijo en primer nodo inferior del brazo (más a la izquierda)
    # Fijar base de la torre (punto de anclaje del cable) - completamente fijo
    bc[tower_base, :] = True  # Apoyo fijo en base de la torre (nodo 0)


    print(f"\nCondiciones de borde:")
    print(f"  Nodo {tower_base} (torre): completamente fijo en {X[tower_base]}")
    print(f"  Nodo {boom_bot_nodes[0]} (brazo izquierdo): completamente fijo en {X[boom_bot_nodes[0]]}")

    # Cargas - simular carga en la punta de la grúa
    loads = np.zeros([n_nodes, 2], float)
    tip_load = 50000  # N (50 kN carga vertical en la punta)
    loads[boom_top_nodes[-1], 1] = -tip_load  # Carga vertical en la punta del brazo

    # Calcular peso propio basado en masa del elemento: Peso = ρ × V × g
    # Distribuir el peso de cada elemento equitativamente a sus dos nodos
    rho_steel = 7850  # kg/m³ - densidad del acero
    g = 9.81  # m/s² - aceleración gravitacional

    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]

        # Longitud del elemento
        L = np.linalg.norm(X[n2] - X[n1])

        # Área de sección transversal del elemento (puntales vs cables)
        # Puntales: superior + inferior + vertical = 2*(n-1) + n = 3n - 2
        num_strut_elements = 3 * len(boom_top_nodes) - 2
        A = A_strut if iEl < num_strut_elements else A_cable

        # Calcular peso del elemento
        volume = A * L  # m³
        mass = rho_steel * volume  # kg
        weight = mass * g  # N

        # Distribuir peso equitativamente a ambos nodos
        loads[n1, 1] -= weight / 2
        loads[n2, 1] -= weight / 2

    print(f"Aplicada carga de {tip_load/1000:.0f} kN en la punta de la grúa")

    # Preparar para análisis de elementos finitos
    bc_mask = bc.reshape(1, 2*n_nodes).ravel()
    load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

    # Ensamblar matriz de rigidez global
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)

    for iEl in range(C.shape[0]):
        # Usar diferentes secciones transversales para puntales vs cables
        # Puntales: superior + inferior + vertical = 3n - 2
        num_strut_elements = 3 * len(boom_top_nodes) - 2
        if iEl < num_strut_elements:  # Puntales (miembros rígidos en compresión)
            A = A_strut
        else:  # Cables (miembros en tracción: diagonales + cables de soporte)
            A = A_cable

        dof = dof_el(C[iEl,0], C[iEl,1])
        k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

    # Resolver sistema
    k_reduced = k_global[~bc_mask][:, ~bc_mask]
    displacements = np.zeros([2*n_nodes], float)
    displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)

    # Redimensionar desplazamientos
    D = displacements.reshape(n_nodes, 2)

    print(f"Desplazamiento máximo: {np.max(np.abs(D)):.4f} m")
    print(f"Deflexión en la punta: {D[boom_top_nodes[-1], 1]:.4f} m")

    # Graficar forma deformada (escalada para visibilidad)
    scale_factor = 1
    plot_truss(X + D * scale_factor, C,
              f"30m Crane - Deformed Shape (Scale: {scale_factor}x)",
              show_nodes=False)

    # Calcular fuerzas y tensiones de los miembros
    print("\n=� Análisis de Fuerzas de Miembros:")
    print("=" * 50)

    max_tension = 0
    max_compression = 0
    element_forces = np.zeros(C.shape[0])
    element_stresses = np.zeros(C.shape[0])
    element_areas = []
    element_inertias = []

    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        d_el = np.concatenate([D[n1], D[n2]])

        # Propiedades del elemento
        num_strut_elements = 3 * len(boom_top_nodes) - 2
        if iEl < num_strut_elements:  # Puntales (miembros rígidos en compresión)
            A = A_strut
            I = I_strut
            member_type = "Puntal"
        else:  # Cables (miembros en tracción)
            A = A_cable
            I = I_cable
            member_type = "Cable"

        element_areas.append(A)
        element_inertias.append(I)

        # Calcular fuerza del elemento
        k_el = element_stiffness(X[n1], X[n2], A, E)
        f_el = k_el @ d_el

        # Calcular fuerza axial usando vector del elemento
        d_vec = X[n2] - X[n1]
        L = np.linalg.norm(d_vec)

        # Saltar elementos de longitud cero
        if L < 1e-10:
            element_forces[iEl] = 0
            element_stresses[iEl] = 0
            continue

        # Vector unitario
        e = d_vec / L
        # Proyectar fuerzas sobre el eje del elemento
        # Positivo = tracción, Negativo = compresión
        axial_force = -np.dot([f_el[2] - f_el[0], f_el[3] - f_el[1]], e)

        # Calcular tensión (fuerza/área)
        stress = axial_force / A

        element_forces[iEl] = axial_force
        element_stresses[iEl] = stress

        if axial_force > max_tension:
            max_tension = axial_force
        if axial_force < max_compression:
            max_compression = axial_force

        if iEl < 10 or abs(axial_force) > 1000:  # Mostrar primeros 10 elementos y fuerzas significativas
            print(f"Elemento {iEl:2d} ({member_type:12s}): {axial_force:8.0f} N")

    print(f"\nTracción Máxima:     {max_tension:8.0f} N")
    print(f"Compresión Máxima: {max_compression:8.0f} N")

    # Factores de seguridad y recomendaciones
    yield_strength = 250e6  # Pa (acero típico)
    safety_factor = 2.5

    max_stress_tension = max_tension / A_strut
    max_stress_compression = abs(max_compression) / A_strut

    print(f"\n=Resumen de Análisis Estructural:")
    print("=" * 50)
    print(f"Tensión de tracción máx:     {max_stress_tension/1e6:.1f} MPa")
    print(f"Tensión de compresión máx: {max_stress_compression/1e6:.1f} MPa")
    print(f"Tensión admisible:       {yield_strength/safety_factor/1e6:.1f} MPa")

    if max_stress_tension < yield_strength/safety_factor and max_stress_compression < yield_strength/safety_factor:
        print("La estructura es SEGURA bajo cargas aplicadas")
    else:
        print("La estructura puede estar SOBRECARGADA - considere aumentar tamaños de miembros")

    # Graficar mapa de calor de tensiones
    plot_stress_heatmap(X, C, element_stresses, element_areas,
                       title="Crane Stress Distribution - Color-coded by Stress Level")

    # Analizar costo y factores de seguridad
    cost, FS_tension, FS_pandeo, mass = analyze_cost_and_security(
        X, C, element_stresses, element_forces, element_inertias, boom_top_nodes)

    return X, C, D, loads, cost, FS_tension, FS_pandeo, mass


def analyze_moving_load():
    '''
    Analizar rendimiento de la grúa con cargas variables en diferentes posiciones
    Prueba cargas de 0 a 40000 N en diferentes posiciones a lo largo de la cuerda inferior
    '''
    print("\n" + "="*70)
    print("ANÁLISIS DE CARGA MÓVIL")
    print("="*70)

    # Crear geometría una vez
    X, tower_base, boom_top_nodes, boom_bot_nodes = design_crane_geometry()
    C = create_crane_connectivity(tower_base, boom_top_nodes, boom_bot_nodes)

    # Propiedades de material y sección
    E = 200e9  # Módulo de Young del acero (Pa)

    # Especificaciones de sección transversal circular hueca (en metros)
    d_outer_strut = 0.050  # 50 mm diámetro exterior
    d_inner_strut = 0.020  # 46 mm diámetro interior
    A_strut, I_strut = hollow_circular_section(d_outer_strut, d_inner_strut)

    d_outer_cable = 0.020  # 20 mm diámetro exterior
    d_inner_cable = 0.016  # 16 mm diámetro interior
    A_cable, I_cable = hollow_circular_section(d_outer_cable, d_inner_cable)

    # Condiciones de borde
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True
    bc[tower_base, :] = True

    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Ensamblar matriz de rigidez global (solo una vez)
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)
    for iEl in range(C.shape[0]):
        num_strut_elements = 3 * len(boom_top_nodes) - 2
        if iEl < num_strut_elements:
            A = A_strut
        else:
            A = A_cable

        dof = dof_el(C[iEl,0], C[iEl,1])
        k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    # Parámetros de prueba
    load_magnitudes = np.linspace(0, 40000, 5)  # 0 a 40000 N en 5 pasos
    test_positions = boom_bot_nodes  # Probar en cada nodo inferior

    # Almacenamiento para resultados
    max_deflections = np.zeros((len(load_magnitudes), len(test_positions)))
    max_stresses = np.zeros((len(load_magnitudes), len(test_positions)))
    max_tensions = np.zeros((len(load_magnitudes), len(test_positions)))
    max_compressions = np.zeros((len(load_magnitudes), len(test_positions)))

    print(f"\nProbando {len(load_magnitudes)} magnitudes de carga en {len(test_positions)} posiciones...")
    print(f"Rango de carga: 0 a 40000 N")
    print(f"Posiciones: A lo largo de la cuerda inferior (nodos {boom_bot_nodes[0]} a {boom_bot_nodes[-1]})")

    # Iterar a través de cargas y posiciones
    for i, load_mag in enumerate(load_magnitudes):
        for j, pos_node in enumerate(test_positions):
            # Aplicar carga en posición actual
            loads = np.zeros([n_nodes, 2], float)
            loads[pos_node, 1] = -load_mag  # Carga vertical hacia abajo

            # Calcular peso propio basado en masa del elemento: Peso = ρ × V × g
            # Distribuir el peso de cada elemento equitativamente a sus dos nodos
            rho_steel = 7850  # kg/m³ - densidad del acero
            g = 9.81  # m/s² - aceleración gravitacional

            for iEl in range(C.shape[0]):
                n1, n2 = C[iEl]
                L = np.linalg.norm(X[n2] - X[n1])
                num_strut_elements = 3 * len(boom_top_nodes) - 2
                A = A_strut if iEl < num_strut_elements else A_cable
                volume = A * L  # m³
                mass = rho_steel * volume  # kg
                weight = mass * g  # N
                loads[n1, 1] -= weight / 2
                loads[n2, 1] -= weight / 2

            load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

            # Resolver
            displacements = np.zeros([2*n_nodes], float)
            displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
            D = displacements.reshape(n_nodes, 2)

            # Calcular deflexión máxima
            max_deflections[i, j] = np.max(np.abs(D))

            # Calcular tensiones
            element_stresses = []
            for iEl in range(C.shape[0]):
                n1, n2 = C[iEl]
                d_el = np.concatenate([D[n1], D[n2]])

                num_strut_elements = 3 * len(boom_top_nodes) - 2
                if iEl < num_strut_elements:
                    A = A_strut
                else:
                    A = A_cable

                k_el = element_stiffness(X[n1], X[n2], A, E)
                f_el = k_el @ d_el

                d_vec = X[n2] - X[n1]
                L = np.linalg.norm(d_vec)

                if L < 1e-10:
                    element_stresses.append(0)
                    continue

                e = d_vec / L
                axial_force = -np.dot([f_el[2] - f_el[0], f_el[3] - f_el[1]], e)
                stress = axial_force / A
                element_stresses.append(stress)

            element_stresses = np.array(element_stresses)
            max_stresses[i, j] = np.max(np.abs(element_stresses))
            max_tensions[i, j] = np.max(element_stresses)
            max_compressions[i, j] = np.min(element_stresses)

    # Graficar resultados
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Posición a lo largo de la grúa (coordenadas x de nodos inferiores)
    positions_x = X[boom_bot_nodes, 0]

    # Gráfico 1: Deflexión máx vs posición para diferentes cargas
    ax1 = axes[0, 0]
    for i, load_mag in enumerate(load_magnitudes):
        ax1.plot(positions_x, max_deflections[i, :] * 1000, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax1.set_xlabel('Position along crane (m)', fontsize=12)
    ax1.set_ylabel('Maximum deflection (mm)', fontsize=12)
    ax1.set_title('Maximum Deflection vs Load Position', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Tensión máx vs posición para diferentes cargas
    ax2 = axes[0, 1]
    for i, load_mag in enumerate(load_magnitudes):
        ax2.plot(positions_x, max_stresses[i, :] / 1e6, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax2.axhline(y=100, color='r', linestyle='--', label='Allowable (100 MPa)')
    ax2.set_xlabel('Position along crane (m)', fontsize=12)
    ax2.set_ylabel('Maximum stress (MPa)', fontsize=12)
    ax2.set_title('Maximum Stress vs Load Position', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Gráfico 3: Tensión de tracción máx vs posición
    ax3 = axes[1, 0]
    for i, load_mag in enumerate(load_magnitudes):
        ax3.plot(positions_x, max_tensions[i, :] / 1e6, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax3.axhline(y=100, color='r', linestyle='--', label='Allowable (100 MPa)')
    ax3.set_xlabel('Position along crane (m)', fontsize=12)
    ax3.set_ylabel('Maximum tensile stress (MPa)', fontsize=12)
    ax3.set_title('Maximum Tensile Stress vs Load Position', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Gráfico 4: Tensión de compresión máx vs posición
    ax4 = axes[1, 1]
    for i, load_mag in enumerate(load_magnitudes):
        ax4.plot(positions_x, max_compressions[i, :] / 1e6, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax4.axhline(y=-100, color='r', linestyle='--', label='Allowable (-100 MPa)')
    ax4.set_xlabel('Position along crane (m)', fontsize=12)
    ax4.set_ylabel('Maximum compressive stress (MPa)', fontsize=12)
    ax4.set_title('Maximum Compressive Stress vs Load Position', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Imprimir resumen
    print("\n" + "="*70)
    print("RESUMEN DE ANÁLISIS")
    print("="*70)
    print(f"Deflexión máxima general: {np.max(max_deflections)*1000:.2f} mm")
    print(f"  Ocurre en: {load_magnitudes[np.unravel_index(np.argmax(max_deflections), max_deflections.shape)[0]]/1000:.1f} kN")
    print(f"  Posición:  {positions_x[np.unravel_index(np.argmax(max_deflections), max_deflections.shape)[1]]:.2f} m")
    print(f"\nTensión máxima general: {np.max(max_stresses)/1e6:.2f} MPa")
    print(f"  Ocurre en: {load_magnitudes[np.unravel_index(np.argmax(max_stresses), max_stresses.shape)[0]]/1000:.1f} kN")
    print(f"  Posición:  {positions_x[np.unravel_index(np.argmax(max_stresses), max_stresses.shape)[1]]:.2f} m")
    print(f"\nTensión de tracción máxima: {np.max(max_tensions)/1e6:.2f} MPa")
    print(f"Tensión de compresión máxima: {np.min(max_compressions)/1e6:.2f} MPa")
    print("="*70)

    return max_deflections, max_stresses, positions_x

def animate_moving_load(load_magnitude=30000, scale_factor=1, interval=200):
    '''
    Crear un lapso de tiempo animado de la deformación de la grúa a medida que la carga se mueve a lo largo de la cuerda inferior

    Parámetros:
    -----------
    load_magnitude : float
        Magnitud de la carga móvil en Newtons (por defecto: 30000 N = 30 kN)
    scale_factor : float
        Factor de escala para visualización de deformación (por defecto: 100)
    interval : int
        Tiempo entre cuadros en milisegundos (por defecto: 200 ms)
    '''
    print("\n" + "="*70)
    print("ANÁLISIS DE CARGA MÓVIL ANIMADO")
    print("="*70)
    print(f"Magnitud de carga: {load_magnitude/1000:.1f} kN")
    print(f"Escala de deformación: {scale_factor}x")
    print("Creando animación...")

    # Crear geometría
    X, tower_base, boom_top_nodes, boom_bot_nodes = design_crane_geometry()
    C = create_crane_connectivity(tower_base, boom_top_nodes, boom_bot_nodes)

    # Propiedades de material y sección
    E = 200e9

    # Especificaciones de sección transversal circular hueca (en metros)
    d_outer_strut = 0.050  # 50 mm diámetro exterior
    d_inner_strut = 0.046  # 46 mm diámetro interior
    A_strut, I_strut = hollow_circular_section(d_outer_strut, d_inner_strut)

    d_outer_cable = 0.020  # 20 mm diámetro exterior
    d_inner_cable = 0.016  # 16 mm diámetro interior
    A_cable, I_cable = hollow_circular_section(d_outer_cable, d_inner_cable)

    # Condiciones de borde
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True
    bc[tower_base, :] = True

    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Ensamblar matriz de rigidez global
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)
    for iEl in range(C.shape[0]):
        num_strut_elements = 3 * len(boom_top_nodes) - 2
        if iEl < num_strut_elements:
            A = A_strut
        else:
            A = A_cable

        dof = dof_el(C[iEl,0], C[iEl,1])
        k_elemental = element_stiffness(X[C[iEl,0],:], X[C[iEl,1],:], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    # Posiciones de prueba (nodos inferiores)
    test_positions = boom_bot_nodes

    # Calcular deformaciones para cada posición
    deformations = []
    max_deflections_list = []

    for pos_node in test_positions:
        # Aplicar carga en posición actual
        loads = np.zeros([n_nodes, 2], float)
        loads[pos_node, 1] = -load_magnitude

        # Calcular peso propio correctamente basado en masa del elemento
        # Distribuir el peso de cada elemento equitativamente a sus dos nodos
        rho_steel = 7850  # kg/m³
        g = 9.81  # m/s²

        for iEl in range(C.shape[0]):
            n1, n2 = C[iEl]

            # Longitud del elemento
            L = np.linalg.norm(X[n2] - X[n1])

            # Sección transversal del elemento
            num_strut_elements = 3 * len(boom_top_nodes) - 2
            A = A_strut if iEl < num_strut_elements else A_cable

            # Volumen y masa del elemento
            volume = A * L  # m³
            mass = rho_steel * volume  # kg
            weight = mass * g  # N

            # Distribuir peso equitativamente a ambos nodos
            loads[n1, 1] -= weight / 2
            loads[n2, 1] -= weight / 2

        load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

        # Resolver
        displacements = np.zeros([2*n_nodes], float)
        displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
        D = displacements.reshape(n_nodes, 2)

        deformations.append(D)
        # Calcular deflexión en la punta de la grúa (último nodo superior)
        tip_deflection = abs(D[boom_top_nodes[-1], 1])  # Desplazamiento vertical en la punta
        max_deflections_list.append(tip_deflection)

    # Crear animación
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Inicializar gráficos
    def init():
        ax1.clear()
        ax2.clear()
        return []

    def update(frame):
        ax1.clear()
        ax2.clear()

        pos_node = test_positions[frame]
        D = deformations[frame]
        X_deformed = X + D * scale_factor

        # Gráfico 1: Estructura original y deformada
        # Estructura original (gris)
        for iEl in range(C.shape[0]):
            ax1.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                    [X[C[iEl,0],1], X[C[iEl,1],1]],
                    'gray', linewidth=1, alpha=0.3, linestyle='--')

        # Estructura deformada (azul)
        for iEl in range(C.shape[0]):
            ax1.plot([X_deformed[C[iEl,0],0], X_deformed[C[iEl,1],0]],
                    [X_deformed[C[iEl,0],1], X_deformed[C[iEl,1],1]],
                    'b-', linewidth=2)

        # Marcar posición de carga con flecha roja
        load_pos = X[pos_node]
        ax1.arrow(load_pos[0], load_pos[1] + 0.5, 0, -0.3,
                 head_width=0.5, head_length=0.2, fc='red', ec='red', linewidth=3)
        ax1.plot(load_pos[0], load_pos[1], 'ro', markersize=15, label='Load Position')

        # Marcar soportes
        ax1.plot(X[boom_bot_nodes[0], 0], X[boom_bot_nodes[0], 1], 'g^',
                markersize=12, label='Pin Support')
        ax1.plot(X[tower_base, 0], X[tower_base, 1], 'g^', markersize=12)

        ax1.set_xlabel('x (m)', fontsize=12)
        ax1.set_ylabel('y (m)', fontsize=12)
        ax1.set_title(f'Crane Deformation - Load at x = {X[pos_node, 0]:.2f} m (Scale: {scale_factor}x)\n'
                     f'Load: {load_magnitude/1000:.1f} kN | Tip Deflection: {max_deflections_list[frame]*1000:.2f} mm',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        ax1.legend(loc='upper right')

        # Gráfico 2: Perfil de deflexión en la punta
        positions_x = X[boom_bot_nodes, 0]
        ax2.plot(positions_x, np.array(max_deflections_list) * 1000, 'b-o', linewidth=2, markersize=8)
        ax2.plot(X[pos_node, 0], max_deflections_list[frame] * 1000, 'ro', markersize=15)
        ax2.axvline(x=X[pos_node, 0], color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Load Position along crane (m)', fontsize=12)
        ax2.set_ylabel('Tip Deflection (mm)', fontsize=12)
        ax2.set_title('Tip Deflection vs Load Position', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return []

    # Crear animación
    anim = FuncAnimation(fig, update, init_func=init, frames=len(test_positions),
                        interval=interval, blit=False, repeat=True)

    print("¡Animación creada! Cierre la ventana para continuar...")
    plt.show()

    print("="*70)

    return anim

if __name__ == "__main__":
    # Ejecutar simulación básica de grúa
    crane_simulation()

    # Ejecutar análisis de carga móvil
    analyze_moving_load()

    # Ejecutar visualización animada de carga móvil
    animate_moving_load(load_magnitude=60000, scale_factor=1, interval=200)