import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import os
from pathlib import Path
from datetime import datetime
import pickle
import time

# Import utility modules
from fem_utils import *
from analysis_utils import *
from plotting_utils import *

# Variables globales para seguimiento de iteraciones
iteration_counter = 0
best_objective = float('inf')
best_design_vector = None  # Almacenar el mejor vector de diseño
last_save_time = None  # Seguir cuándo guardamos por última vez
output_dir = None  # Se establecerá en optimize_crane() con marca de tiempo

def design_parametric_crane(n_segments, boom_height, connectivity_pattern, taper_ratio=1.0):
    '''
    Diseñar geometría de grúa con parámetros variables

    Parámetros:
    -----------
    n_segments : int
        Número de segmentos a lo largo del brazo de 30m (afecta el conteo de nodos)
    boom_height : float
        Profundidad del brazo de cercha en la base (m)
    taper_ratio : float
        Relación de altura de punta a altura de base (0.0 a 1.0)
        1.0 = rectangular (sin ahusamiento), 0.0 = ahusamiento completo a altura cero en la punta
    connectivity_pattern : array
        Codificación de patrón: [diag_type, vertical_spacing, support_cables]
        - diag_type:
            0 = Alternado (cercha Warren)
            1 = Toda pendiente positiva
            2 = Toda pendiente negativa
            3 = Diagonales dobles (patrón X)
            4 = Abanico desde abajo (múltiples diagonales por nodo inferior)
            5 = Abanico desde arriba (múltiples diagonales por nodo superior)
            6 = Tramo largo (saltar 2 segmentos)
            7 = Tramo mixto (asimétrico, adyacente + largo)
            8 = Concentrado (más denso en extremos, escaso en centro)
            9 = Abanico progresivo (más conexiones hacia la punta)
            10 = Conectividad completa (CADA nodo inferior a CADA nodo superior)

    Retorna:
    --------
    X : ndarray
        Coordenadas de nodos
    C : ndarray
        Matriz de conectividad
    element_types : list
        Tipo de cada elemento ('top_strut', 'bot_strut', 'vertical', 'diagonal', 'support')
    '''
    crane_length = 30.0  # Requisito fijo de 30m
    tower_height = 0.0

    # Crear nodos del brazo
    boom_x = np.linspace(0, crane_length, n_segments + 1)

    # Control de ahusamiento: taper_ratio determina altura de punta relativa a altura de base
    # taper_ratio = 1.0 → rectangular (altura constante)
    # taper_ratio = 0.0 → completamente ahusado (altura cero en la punta)
    tip_height = boom_height * taper_ratio
    boom_y_top = np.linspace(tower_height + boom_height, tower_height + tip_height, n_segments + 1)
    boom_y_bot = np.full(n_segments + 1, tower_height)

    # Inicializar nodos
    total_nodes = 1 + 2 * (n_segments + 1)  # Torre + superior + inferior
    X = np.zeros([total_nodes, 2], float)

    node_idx = 0

    # Torre
    X[0] = [0, tower_height + 5]
    tower_base = 0
    node_idx += 1

    # Nodos superiores
    boom_top_nodes = []
    for i in range(n_segments + 1):
        X[node_idx] = [boom_x[i], boom_y_top[i]]
        boom_top_nodes.append(node_idx)
        node_idx += 1

    # Nodos inferiores
    boom_bot_nodes = []
    for i in range(n_segments + 1):
        X[node_idx] = [boom_x[i], boom_y_bot[i]]
        boom_bot_nodes.append(node_idx)
        node_idx += 1

    # Crear conectividad basada en patrón
    elements = []
    element_types = []

    # Puntales superiores
    for i in range(len(boom_top_nodes) - 1):
        elements.append([boom_top_nodes[i], boom_top_nodes[i+1]])
        element_types.append('top_strut')

    # Puntales inferiores
    for i in range(len(boom_bot_nodes) - 1):
        elements.append([boom_bot_nodes[i], boom_bot_nodes[i+1]])
        element_types.append('bot_strut')

    # Miembros verticales (controlados por parámetro vertical_spacing)
    vertical_spacing = int(connectivity_pattern[1])  # Cada enésimo nodo obtiene una vertical
    vertical_spacing = max(1, min(vertical_spacing, n_segments))

    for i in range(0, len(boom_top_nodes), vertical_spacing):
        elements.append([boom_top_nodes[i], boom_bot_nodes[i]])
        element_types.append('vertical')

    # Patrón diagonal
    diag_type = int(connectivity_pattern[0])

    if diag_type == 0:  # Alternado (cercha Warren clásica)
        for i in range(len(boom_top_nodes) - 1):
            if i % 2 == 0:
                elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
                element_types.append('diagonal')
            else:
                elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
                element_types.append('diagonal')

    elif diag_type == 1:  # Toda pendiente positiva
        for i in range(len(boom_top_nodes) - 1):
            elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
            element_types.append('diagonal')

    elif diag_type == 2:  # Toda pendiente negativa
        for i in range(len(boom_top_nodes) - 1):
            elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
            element_types.append('diagonal')

    elif diag_type == 3:  # Diagonales dobles (patrón X, simétrico)
        for i in range(len(boom_top_nodes) - 1):
            elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
            elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
            element_types.append('diagonal')
            element_types.append('diagonal')

    elif diag_type == 4:  # Patrón de abanico desde nodos inferiores (múltiples diagonales desde un solo nodo)
        for i in range(len(boom_bot_nodes)):
            # Cada nodo inferior se conecta a múltiples nodos superiores (abanico)
            for j in range(max(0, i-1), min(len(boom_top_nodes), i+3)):
                if j != i:  # Saltar vertical directa (ya manejada)
                    elements.append([boom_bot_nodes[i], boom_top_nodes[j]])
                    element_types.append('diagonal')

    elif diag_type == 5:  # Patrón de abanico desde nodos superiores (múltiples diagonales desde un solo nodo)
        for i in range(len(boom_top_nodes)):
            # Cada nodo superior se conecta a múltiples nodos inferiores (abanico)
            for j in range(max(0, i-1), min(len(boom_bot_nodes), i+3)):
                if j != i:  # Saltar vertical directa
                    elements.append([boom_top_nodes[i], boom_bot_nodes[j]])
                    element_types.append('diagonal')

    elif diag_type == 6:  # Diagonales de tramo largo (saltar 2 segmentos)
        for i in range(len(boom_top_nodes) - 2):
            elements.append([boom_bot_nodes[i], boom_top_nodes[i+2]])
            element_types.append('diagonal')
        for i in range(len(boom_bot_nodes) - 2):
            elements.append([boom_top_nodes[i], boom_bot_nodes[i+2]])
            element_types.append('diagonal')

    elif diag_type == 7:  # Tramo mixto (adyacente y tramo largo, asimétrico)
        for i in range(len(boom_top_nodes) - 1):
            # Tramo adyacente
            if i % 2 == 0:
                elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
                element_types.append('diagonal')
            # Tramo largo (cada 3er segmento)
            if i % 3 == 0 and i + 2 < len(boom_top_nodes):
                elements.append([boom_top_nodes[i], boom_bot_nodes[i+2]])
                element_types.append('diagonal')

    elif diag_type == 8:  # Diagonales concentradas (más denso en extremos, escaso en medio)
        for i in range(len(boom_top_nodes) - 1):
            # Más diagonales cerca de los extremos (mayor corte)
            n_mid = len(boom_top_nodes) // 2
            dist_from_center = abs(i - n_mid)

            if dist_from_center > n_mid * 0.6:  # 40% exterior - diagonales dobles
                elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
                elements.append([boom_top_nodes[i], boom_bot_nodes[i+1]])
                element_types.append('diagonal')
                element_types.append('diagonal')
            elif dist_from_center > n_mid * 0.3:  # 30% medio - diagonal simple
                elements.append([boom_bot_nodes[i], boom_top_nodes[i+1]])
                element_types.append('diagonal')
            # Centro 30% - sin diagonales (región de menor corte)

    elif diag_type == 9:  # Abanico progresivo (más conexiones hacia la punta)
        for i in range(len(boom_bot_nodes)):
            # El número de conexiones aumenta hacia la punta
            progress = i / max(1, len(boom_bot_nodes) - 1)
            max_span = int(1 + progress * 3)  # 1 a 4 conexiones

            for j in range(max(0, i - max_span), min(len(boom_top_nodes), i + max_span + 1)):
                if abs(j - i) > 0 and abs(j - i) <= max_span:
                    elements.append([boom_bot_nodes[i], boom_top_nodes[j]])
                    element_types.append('diagonal')

    elif diag_type == 10:  # Conectividad completa (cualquier nodo inferior a cualquier nodo superior)
        # Conectar CADA nodo inferior a CADA nodo superior (excluyendo verticales directas)
        # Esto crea una estructura completamente triangulada
        for i in range(len(boom_bot_nodes)):
            for j in range(len(boom_top_nodes)):
                if i != j:  # Saltar conexiones verticales directas (ya manejadas)
                    elements.append([boom_bot_nodes[i], boom_top_nodes[j]])
                    element_types.append('diagonal')

    # Cables de soporte desde la torre (controlados por connectivity_pattern[2])
    num_support_cables = int(connectivity_pattern[2])
    num_support_cables = max(1, min(num_support_cables, len(boom_top_nodes) - 1))

    # Distribuir cables de soporte uniformemente
    support_indices = np.linspace(len(boom_top_nodes)//2, len(boom_top_nodes)-2,
                                  num_support_cables, dtype=int)

    for idx in support_indices:
        elements.append([tower_base, boom_top_nodes[idx]])
        element_types.append('support')

    return X, np.array(elements, dtype=int), element_types, tower_base, boom_top_nodes, boom_bot_nodes

def save_iteration_graph(X, C, element_types, thickness_params, objective, iteration_num, n_segments, boom_height):
    '''
    Guardar un gráfico del diseño actual sin mostrarlo
    (Simplificado - solo para registro durante optimización)
    '''
    global output_dir

    try:
        n_elements = C.shape[0]

        colors = {
            'top_strut': 'blue',
            'bot_strut': 'green',
            'vertical': 'purple',
            'diagonal': 'orange',
            'support': 'red'
        }

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        # Graficar elementos con ancho de línea basado en espesor
        for iEl in range(n_elements):
            if 2*iEl < len(thickness_params):
                d_outer = thickness_params[2*iEl] * 1000  # mm
                lw = max(0.5, min(5, d_outer / 10))
            else:
                lw = 1.5

            etype = element_types[iEl] if iEl < len(element_types) else 'diagonal'
            color = colors.get(etype, 'gray')

            ax.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                   [X[C[iEl,0],1], X[C[iEl,1],1]],
                   color=color, linewidth=lw, alpha=0.7)

        ax.scatter(X[:,0], X[:,1], c='black', s=30, zorder=5)
        ax.set_xlabel('x (m)', fontsize=10)
        ax.set_ylabel('y (m)', fontsize=10)
        ax.set_title(f'Iteración {iteration_num} - Objetivo: {objective:.2f}\n' +
                    f'{n_segments} segmentos, h={boom_height:.2f}m, {n_elements} elementos, {X.shape[0]} nodos',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Agregar leyenda
        legend_elements = [plt.Line2D([0], [0], color=c, lw=2, label=t)
                          for t, c in colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()

        # Guardar con objetivo en nombre de archivo para ordenar fácilmente
        filename = output_dir / f"iter_{iteration_num:04d}_obj_{objective:.2f}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close(fig)

    except Exception as e:
        # Si el gráfico falla, no bloquear la optimización
        plt.close('all')
        pass

def save_intermediate_design(design_vector, objective, iteration_num):
    '''
    Guardar un diseño intermedio de grúa durante la optimización
    Esto se llama periódicamente (cada N minutos) para evitar perder progreso

    Parámetros:
    -----------
    design_vector : array
        Vector de diseño completo
    objective : float
        Valor objetivo
    iteration_num : int
        Número de iteración actual
    '''
    global output_dir

    try:
        # Extraer parámetros de diseño
        n_segments = int(round(design_vector[0]))
        boom_height = design_vector[1]
        connectivity_pattern = design_vector[2:5]
        taper_ratio = design_vector[5]

        # Regenerar geometría
        X, C, types, tower_base, boom_top_nodes, boom_bot_nodes = \
            design_parametric_crane(n_segments, boom_height, connectivity_pattern, taper_ratio)

        # Extraer parámetros de espesor
        n_elements = C.shape[0]
        thickness_params = design_vector[6:6+n_elements*2]

        # Empaquetar todo
        crane_data = {
            'design_vector': design_vector,
            'n_segments': n_segments,
            'boom_height': boom_height,
            'connectivity_pattern': connectivity_pattern,
            'taper_ratio': taper_ratio,
            'X': X,
            'C': C,
            'element_types': types,
            'tower_base': tower_base,
            'boom_top_nodes': boom_top_nodes,
            'boom_bot_nodes': boom_bot_nodes,
            'thickness_params': thickness_params,
            'objective': objective,
            'iteration': iteration_num
        }

        # Guardar con marca de tiempo en nombre de archivo
        timestamp = datetime.now().strftime("%H%M%S")
        filepath = output_dir / f'checkpoint_iter{iteration_num:04d}_obj{objective:.2f}_{timestamp}.pkl'

        with open(filepath, 'wb') as f:
            pickle.dump(crane_data, f)

        print(f"  [CHECKPOINT] Diseño intermedio guardado en: {filepath.name}")

    except Exception as e:
        # No bloquear optimización si falla el guardado
        print(f"  [WARNING] Fallo al guardar diseño intermedio: {e}")

def evaluate_crane_design(design_vector, max_load=40000, verbose=False):
    '''
    Evaluar un diseño de grúa y devolver costo con penalizaciones por violaciones de seguridad

    Codificación del vector de diseño:
    [0]: n_segments (8-20)
    [1]: boom_height (0.5-2.0 m)
    [2]: diag_type (0-9)
    [3]: vertical_spacing (1-3)
    [4]: num_support_cables (1-3)
    [5]: taper_ratio (0.0-1.0)
    [6:]: espesores de miembros - pares de (d_outer, d_inner) para cada elemento

    Retorna:
    --------
    objective : float
        Costo + penalizaciones (minimizar esto)
    '''
    # Decodificar diseño
    n_segments = int(round(design_vector[0]))
    boom_height = design_vector[1]
    connectivity_pattern = design_vector[2:5]
    taper_ratio = design_vector[5]

    # Generar geometría
    try:
        X, C, element_types, tower_base, boom_top_nodes, boom_bot_nodes = \
            design_parametric_crane(n_segments, boom_height, connectivity_pattern, taper_ratio)
    except Exception as e:
        if verbose:
            print(f"Generación de geometría falló: {e}")
        return 1e9  # Penalización enorme por geometría inválida

    n_elements = C.shape[0]
    n_nodes = X.shape[0]

    # Verificar si tenemos suficientes parámetros de espesor
    expected_thickness_params = n_elements * 2  # 2 por elemento (d_outer, d_inner)
    if len(design_vector) < 6 + expected_thickness_params:
        if verbose:
            print(f"No hay suficientes parámetros de espesor: se necesitan {expected_thickness_params}, se obtuvieron {len(design_vector)-6}")
        return 1e9

    # Extraer espesores de miembros
    thickness_params = design_vector[6:6+expected_thickness_params]

    # Propiedades del material
    E = 200e9
    rho_steel = 7850
    g = 9.81

    # Construir áreas e inercias de elementos
    element_areas = []
    element_inertias = []
    total_mass = 0

    for iEl in range(n_elements):
        d_outer = thickness_params[2*iEl]
        d_inner = thickness_params[2*iEl + 1]

        # Asegurar interior < exterior con espesor de pared mínimo de 2mm
        min_wall = 0.002  # 2mm espesor de pared mínimo
        if d_inner >= d_outer - min_wall:
            d_inner = d_outer - min_wall

        # Asegurar diámetro interior positivo
        if d_inner < 0.001:
            d_inner = 0.001

        A, I = hollow_circular_section(d_outer, d_inner)
        element_areas.append(A)
        element_inertias.append(I)

        # Calcular masa
        n1, n2 = C[iEl]
        L = np.linalg.norm(X[n2] - X[n1])
        volume = A * L
        mass = rho_steel * volume
        total_mass += mass

    # Condiciones de borde
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True  # Apoyo de pasador
    bc[tower_base, :] = True  # Torre fija

    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Ensamblar matriz de rigidez global usando función de utilidades
    k_global = assemble_global_stiffness(X, C, element_areas, E)

    # Verificar matriz singular
    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    try:
        # Verificar condicionamiento de matriz
        cond = np.linalg.cond(k_reduced)
        if cond > 1e12:
            if verbose:
                print(f"Matriz mal condicionada: cond={cond:.2e}")
            return 1e8
    except:
        return 1e8

    # Aplicar cargas (carga en punta + peso propio)
    loads = np.zeros([n_nodes, 2], float)
    loads[boom_top_nodes[-1], 1] = -max_load

    # Agregar peso propio usando función de utilidades
    apply_self_weight(X, C, element_areas, loads, rho_steel, g)

    load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

    # Resolver
    try:
        displacements = np.zeros([2*n_nodes], float)
        displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
        D = displacements.reshape(n_nodes, 2)
    except np.linalg.LinAlgError:
        if verbose:
            print("Matriz singular durante resolución")
        return 1e8

    # Calcular tensiones y fuerzas usando función de utilidades
    element_forces, element_stresses = calculate_element_stresses(X, C, D, element_areas, E)

    # Calcular factores de seguridad
    sigma_max = np.max(np.abs(element_stresses))
    sigma_adm = 100e6  # 100 MPa

    if sigma_max < 1e-6:
        FS_tension = 1000
    else:
        FS_tension = sigma_adm / sigma_max

    # Verificación de pandeo usando función de utilidades
    P_critica, max_compressed_length = calculate_buckling_load(
        X, C, element_forces, element_inertias, E, boom_top_nodes
    )

    P_max_compression = abs(np.min(element_forces))
    if P_max_compression < 1e-6:
        FS_pandeo = 1000
    else:
        FS_pandeo = P_critica / P_max_compression

    # Calcular costo usando función de utilidades
    n_uniones = n_nodes
    cost = calculate_cost(total_mass, n_elements, n_uniones)

    # Penalizaciones por violaciones de seguridad
    penalty = 0
    min_FS = 2.0

    if FS_tension < min_FS:
        penalty += 1000 * (min_FS - FS_tension)**2

    if FS_pandeo < min_FS:
        penalty += 1000 * (min_FS - FS_pandeo)**2

    # Penalización por deflexión (máx 200mm en punta)
    tip_deflection = abs(D[boom_top_nodes[-1], 1])
    max_deflection = 0.200  # 200mm
    if tip_deflection > max_deflection:
        penalty += 500 * (tip_deflection - max_deflection)**2

    objective = cost + penalty

    # Rastrear iteraciones y guardar gráficos SOLO para diseños que mejoran
    global iteration_counter, best_objective, best_design_vector, last_save_time

    iteration_counter += 1

    # Guardar gráfico SOLO si este es un mejor diseño
    if objective < best_objective and objective < 1e7:  # Solo guardar diseños válidos que mejoran
        # Extraer parámetros de espesor
        expected_thickness_params = n_elements * 2
        if len(design_vector) >= 6 + expected_thickness_params:
            thickness_params = design_vector[6:6+expected_thickness_params]
        else:
            thickness_params = []

        save_iteration_graph(X, C, element_types, thickness_params,
                           objective, iteration_counter, n_segments, boom_height)

        best_objective = objective
        best_design_vector = design_vector.copy()  # Almacenar el mejor diseño
        print(f"[Iter {iteration_counter}] NUEVO MEJOR! Objective: {objective:.2f} (Cost: {cost:.2f}, Penalty: {penalty:.2f})")

        # Verificar si pasó suficiente tiempo para guardar un checkpoint intermedio
        current_time = time.time()
        if last_save_time is None:
            last_save_time = current_time

        time_since_last_save = current_time - last_save_time
        save_interval = 300  # Guardar cada 5 minutos (300 segundos)

        if time_since_last_save >= save_interval:
            save_intermediate_design(design_vector, objective, iteration_counter)
            last_save_time = current_time

    if verbose:
        print(f"Design: n_seg={n_segments}, h={boom_height:.2f}m, n_el={n_elements}, n_nodes={n_nodes}")
        print(f"  Mass: {total_mass:.1f} kg, Cost: {cost:.2f}, Penalty: {penalty:.2f}")
        print(f"  FS_tension: {FS_tension:.2f}, FS_pandeo: {FS_pandeo:.2f}")
        print(f"  Tip deflection: {tip_deflection*1000:.1f} mm")
        print(f"  Objective: {objective:.2f}")

    return objective

def optimize_crane(maxiter=100, popsize=15):
    '''
    Rutina principal de optimización

    Parámetros:
    -----------
    maxiter : int
        Número máximo de iteraciones para evolución diferencial (default: 100)
    popsize : int
        Tamaño de población para evolución diferencial (default: 15)
    '''
    global output_dir, iteration_counter, best_objective

    # Crear directorio de salida con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"gruita3_iterations_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"Guardando gráficos de iteración en: {output_dir.absolute()}")

    # Reiniciar contadores
    iteration_counter = 0
    best_objective = float('inf')

    print("="*80)
    print("GRÚA AUTO-OPTIMIZANTE (GRUITA 3)")
    print("="*80)
    print("\nLa grúa se diseñará a sí misma optimizando:")
    print("  1. Número de segmentos (distribución de nodos)")
    print("  2. Altura del brazo (profundidad de cercha)")
    print("  3. Patrón de conectividad (diagonales, verticales, cables de soporte)")
    print("  4. Espesores individuales de miembros (cada elemento optimizado)")
    print("\nObjetivo: Minimizar costo manteniendo FS >= 2.0")
    print("\nLos gráficos se guardarán SOLO cuando se encuentre un mejor diseño!")
    print("="*80)

    # Comenzar con una línea base razonable para estimar cantidad de elementos
    baseline_segments = 12
    baseline_height = 1.0
    baseline_pattern = [0, 1, 2]  # Diagonales alternadas, vertical en cada nodo, 2 cables de soporte

    X_base, C_base, types_base, _, _, _ = design_parametric_crane(
        baseline_segments, baseline_height, baseline_pattern)

    n_elements_base = C_base.shape[0]
    print(f"\nEl diseño base tiene {n_elements_base} elementos")

    # Crear límites para optimización
    # [n_segments, boom_height, diag_type, vertical_spacing, num_support_cables, taper_ratio,
    # d_outer_0, d_inner_0, d_outer_1, d_inner_1, ...]

    bounds = [
        (8, 20),      # n_segments
        (0.5, 2.0),   # boom_height
        (0, 10),      # diag_type (ahora 0-10 incluyendo conectividad completa)
        (1, 3),       # vertical_spacing
        (1, 3),       # num_support_cables
        (0.0, 1.0),   # taper_ratio (0=ahusamiento completo, 1=rectangular)
    ]

    # Agregar límites de espesor para máxima cantidad posible de elementos
    # Tipo 10 (conectividad completa) con 20 segmentos crea ~500 elementos
    max_possible_elements = 600  # Sobrestimación conservadora para tipo 10

    for i in range(max_possible_elements):
        bounds.append((0.010, 0.050))  # d_outer: 10mm a 50mm (restricción máx)
        bounds.append((0.005, 0.045))  # d_inner: 5mm a 45mm

    print(f"\nVariables de optimización: {len(bounds)}")
    print("\nIniciando optimización global (evolución diferencial)...")
    print("Esto puede tomar varios minutos...\n")

    # Crear estimación inicial con diseño razonable
    initial_guess = []
    initial_guess.append(12)   # n_segments
    initial_guess.append(1.0)  # boom_height
    initial_guess.append(0)    # diag_type (alternado)
    initial_guess.append(1)    # vertical_spacing (cada nodo)
    initial_guess.append(2)    # num_support_cables
    initial_guess.append(0.0)  # taper_ratio (0=ahusamiento completo como el original)

    # Agregar espesores iniciales: puntales más gruesos que cables
    for i in range(max_possible_elements):
        if i < 40:  # Primeros elementos probablemente sean puntales
            initial_guess.append(0.040)  # 40mm exterior
            initial_guess.append(0.030)  # 30mm interior (10mm pared)
        else:  # Elementos posteriores probablemente sean cables
            initial_guess.append(0.025)  # 25mm exterior
            initial_guess.append(0.020)  # 20mm interior (2.5mm pared)

    print("\nEvaluando diseño inicial...")
    initial_obj = evaluate_crane_design(np.array(initial_guess), max_load=40000, verbose=True)

    # Usar evolución diferencial para optimización global
    print(f"\nIniciando optimización con evolución diferencial...")
    print(f"  Máx iteraciones: {maxiter}")
    print(f"  Tamaño de población: {popsize}")
    result = differential_evolution(
        evaluate_crane_design,
        bounds,
        args=(40000, False),  # Carga de 40kN, no verboso
        x0=np.array(initial_guess),
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=0.01,
        mutation=(0.5, 1.5),
        recombination=0.7,
        seed=42,
        workers=1,
        updating='deferred',
        disp=True,
        atol=0.01,
        polish=False  # Deshabilitar pulido para evitar problemas numéricos
    )

    print("\n" + "="*80)
    print("OPTIMIZACIÓN COMPLETA")
    print("="*80)

    # Evaluar mejor diseño con salida verbosa
    print("\nMejor diseño encontrado:")
    print("-"*80)
    best_obj = evaluate_crane_design(result.x, max_load=40000, verbose=True)

    # Extraer y visualizar mejor diseño
    n_segments_opt = int(round(result.x[0]))
    boom_height_opt = result.x[1]
    connectivity_pattern_opt = result.x[2:5]

    X_opt, C_opt, types_opt, tower_base, boom_top_nodes, boom_bot_nodes = \
        design_parametric_crane(n_segments_opt, boom_height_opt, connectivity_pattern_opt)

    # El mejor diseño ya fue guardado durante la optimización
    print("\n" + "="*80)
    print("Diseño óptimo completado!")
    print("Use el archivo .pkl guardado para análisis y visualización posteriores.")
    print("="*80)

    return result


def save_best_crane(result, filename='best_crane.pkl'):
    '''
    Guardar el mejor diseño de grúa a un archivo para pruebas posteriores

    Parámetros:
    -----------
    result : OptimizeResult
        Objeto de resultado de differential_evolution
    filename : str
        Nombre del archivo para guardar el diseño
    '''
    global output_dir

    # Extraer parámetros de diseño
    n_segments_opt = int(round(result.x[0]))
    boom_height_opt = result.x[1]
    connectivity_pattern_opt = result.x[2:5]
    taper_ratio_opt = result.x[5]

    # Regenerar geometría
    X_opt, C_opt, types_opt, tower_base, boom_top_nodes, boom_bot_nodes = \
        design_parametric_crane(n_segments_opt, boom_height_opt, connectivity_pattern_opt, taper_ratio_opt)

    # Extraer parámetros de espesor
    n_elements = C_opt.shape[0]
    thickness_params = result.x[6:6+n_elements*2]

    # Empaquetar todo
    crane_data = {
        'design_vector': result.x,
        'n_segments': n_segments_opt,
        'boom_height': boom_height_opt,
        'connectivity_pattern': connectivity_pattern_opt,
        'taper_ratio': taper_ratio_opt,
        'X': X_opt,
        'C': C_opt,
        'element_types': types_opt,
        'tower_base': tower_base,
        'boom_top_nodes': boom_top_nodes,
        'boom_bot_nodes': boom_bot_nodes,
        'thickness_params': thickness_params,
        'objective': result.fun
    }

    # Guardar en directorio de salida
    filepath = output_dir / filename
    with open(filepath, 'wb') as f:
        pickle.dump(crane_data, f)

    print(f"\nMejor diseño de grúa guardado en: {filepath}")
    return filepath


if __name__ == '__main__':
    import sys

    # Parsear argumentos de línea de comandos
    maxiter = 100  # Default
    popsize = 15   # Default

    if len(sys.argv) > 1:
        try:
            maxiter = int(sys.argv[1])
            print(f'Usando maxiter de línea de comandos: {maxiter}')
        except ValueError:
            print(f'Argumento maxiter inválido: {sys.argv[1]}, usando default: {maxiter}')

    if len(sys.argv) > 2:
        try:
            popsize = int(sys.argv[2])
            print(f'Usando popsize de línea de comandos: {popsize}')
        except ValueError:
            print(f'Argumento popsize inválido: {sys.argv[2]}, usando default: {popsize}')

    # Ejecutar optimización
    result = optimize_crane(maxiter=maxiter, popsize=popsize)

    # Guardar el mejor diseño
    pkl_file = save_best_crane(result)

    print()
    print('='*80)
    print('OPTIMIZACIÓN COMPLETADA')
    print('='*80)
    print(f'Para ejecutar análisis:')
    print(f'  python grua_analizador.py {pkl_file}')
    print('='*80)

