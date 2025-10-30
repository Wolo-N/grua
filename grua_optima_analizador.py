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

def load_crane(filename='best_crane.pkl'):
    '''
    Cargar un diseño de grúa guardado

    Parámetros:
    -----------
    filename : str or Path
        Ruta al archivo de grúa guardado

    Retorna:
    --------
    crane_data : dict
        Diccionario conteniendo todos los datos del diseño de grúa
    '''
    with open(filename, 'rb') as f:
        crane_data = pickle.load(f)

    print(f"Diseño de grúa cargado desde: {filename}")
    print(f"  Segmentos: {crane_data['n_segments']}")
    print(f"  Elementos: {crane_data['C'].shape[0]}")
    print(f"  Nodos: {crane_data['X'].shape[0]}")
    print(f"  Objetivo: {crane_data['objective']:.2f}")

    return crane_data

def run_crane_analysis(crane_pkl_file):
    '''
    Cargar un diseño de grúa optimizado y ejecutar todos los análisis usando plotting_utils

    Parámetros:
    -----------
    crane_pkl_file : str or Path
        Ruta al archivo .pkl del diseño de grúa
    '''

    # Cargar diseño
    crane_data = load_crane(crane_pkl_file)

    # Extraer datos
    X = crane_data['X']
    C = crane_data['C']
    element_types = crane_data['element_types']
    thickness_params = crane_data['thickness_params']
    n_segments = crane_data['n_segments']
    boom_height = crane_data['boom_height']
    connectivity_pattern = crane_data['connectivity_pattern']
    tower_base = crane_data['tower_base']
    boom_top_nodes = crane_data['boom_top_nodes']
    boom_bot_nodes = crane_data['boom_bot_nodes']

    # 1. Graficar diseño optimizado
    print("\n" + "="*80)
    print("GRAFICANDO DISEÑO OPTIMIZADO")
    print("="*80)
    plot_optimized_crane(X, C, element_types, thickness_params,
                        n_segments, boom_height, connectivity_pattern)

    # 2. Análisis de tensiones con carga en punta
    print("\n" + "="*80)
    print("ANÁLISIS DE TENSIONES")
    print("="*80)

    # Construir áreas de elementos
    element_areas = []
    element_inertias = []
    for iEl in range(C.shape[0]):
        d_outer = thickness_params[2*iEl]
        d_inner = thickness_params[2*iEl + 1]

        min_wall = 0.002
        if d_inner >= d_outer - min_wall:
            d_inner = d_outer - min_wall
        if d_inner < 0.001:
            d_inner = 0.001

        A, I = hollow_circular_section(d_outer, d_inner)
        element_areas.append(A)
        element_inertias.append(I)

    # Propiedades del material
    E = 200e9
    rho_steel = 7800  # kg/m³ (consigna TP2)
    g = 9.81

    # Condiciones de borde
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True
    bc[tower_base, :] = True
    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Ensamblar matriz de rigidez
    k_global = assemble_global_stiffness(X, C, element_areas, E)
    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    # Aplicar carga en punta
    loads = np.zeros([n_nodes, 2], float)
    loads[boom_top_nodes[-1], 1] = -40000  # 40 kN en punta
    apply_self_weight(X, C, element_areas, loads, rho_steel, g)

    load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

    # Resolver
    displacements = np.zeros([2*n_nodes], float)
    displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
    D = displacements.reshape(n_nodes, 2)

    # Calcular tensiones
    element_forces, element_stresses = calculate_element_stresses(X, C, D, element_areas, E)

    # Graficar mapa de calor de tensiones
    plot_stress_heatmap(X, C, element_stresses, element_areas,
                       title="Distribución de Tensiones - Carga de 40kN en Punta")

    # 3. Análisis de carga móvil
    analyze_moving_load_internal(crane_data)

    # 4. Animación de carga móvil
    print("\n" + "="*80)
    print("CREANDO ANIMACIÓN DE CARGA MÓVIL")
    print("="*80)
    animate_moving_load(X, C, element_areas, boom_top_nodes, boom_bot_nodes,
                       tower_base, E, rho_steel, g,
                       load_magnitude=30000, scale_factor=1, interval=200)

    print("\n" + "="*80)
    print("ANÁLISIS COMPLETO")
    print("="*80)


def analyze_moving_load_internal(crane_data, load_magnitudes=None):
    '''
    Analizar rendimiento de grúa con cargas variables en diferentes posiciones
    Prueba cargas en diferentes posiciones a lo largo de la cuerda inferior

    Parámetros:
    -----------
    crane_data : dict
        Datos de diseño de grúa de load_crane()
    load_magnitudes : array-like, opcional
        Magnitudes de carga a probar (default: 0 a 40000 N en 5 pasos)
    '''
    print("\n" + "="*70)
    print("ANÁLISIS DE CARGA MÓVIL")
    print("="*70)

    # Extraer datos de grúa
    X = crane_data['X']
    C = crane_data['C']
    tower_base = crane_data['tower_base']
    boom_top_nodes = crane_data['boom_top_nodes']
    boom_bot_nodes = crane_data['boom_bot_nodes']
    thickness_params = crane_data['thickness_params']

    # Propiedades del material
    E = 200e9
    rho_steel = 7800  # kg/m³ (consigna TP2)
    g = 9.81

    # Construir áreas de elementos desde parámetros de espesor
    element_areas = []
    for iEl in range(C.shape[0]):
        d_outer = thickness_params[2*iEl]
        d_inner = thickness_params[2*iEl + 1]

        # Asegurar dimensiones válidas
        min_wall = 0.002
        if d_inner >= d_outer - min_wall:
            d_inner = d_outer - min_wall
        if d_inner < 0.001:
            d_inner = 0.001

        A, I = hollow_circular_section(d_outer, d_inner)
        element_areas.append(A)

    # Condiciones de borde
    n_nodes = X.shape[0]
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True
    bc[tower_base, :] = True

    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Ensamblar matriz de rigidez global usando función de utilidades
    k_global = assemble_global_stiffness(X, C, element_areas, E)
    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    # Parámetros de prueba
    if load_magnitudes is None:
        load_magnitudes = np.linspace(0, 40000, 5)
    test_positions = boom_bot_nodes

    # Almacenamiento para resultados
    max_deflections = np.zeros((len(load_magnitudes), len(test_positions)))
    max_stresses = np.zeros((len(load_magnitudes), len(test_positions)))
    max_tensions = np.zeros((len(load_magnitudes), len(test_positions)))
    max_compressions = np.zeros((len(load_magnitudes), len(test_positions)))

    print(f"\nProbando {len(load_magnitudes)} magnitudes de carga en {len(test_positions)} posiciones...")
    print(f"Rango de carga: {load_magnitudes[0]:.0f} a {load_magnitudes[-1]:.0f} N")

    # Iterar a través de cargas y posiciones
    for i, load_mag in enumerate(load_magnitudes):
        for j, pos_node in enumerate(test_positions):
            # Aplicar carga en posición actual
            loads = np.zeros([n_nodes, 2], float)
            loads[pos_node, 1] = -load_mag

            # Agregar peso propio usando función de utilidades
            apply_self_weight(X, C, element_areas, loads, rho_steel, g)

            load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

            # Resolver
            displacements = np.zeros([2*n_nodes], float)
            displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
            D = displacements.reshape(n_nodes, 2)

            # Calcular deflexión máxima
            max_deflections[i, j] = np.max(np.abs(D))

            # Calcular tensiones usando función de utilidades
            element_forces, element_stresses = calculate_element_stresses(X, C, D, element_areas, E)
            max_stresses[i, j] = np.max(np.abs(element_stresses))
            max_tensions[i, j] = np.max(element_stresses)
            max_compressions[i, j] = np.min(element_stresses)

    # Graficar resultados usando función de utilidades
    from plotting_utils import plot_moving_load_analysis

    positions_x = X[boom_bot_nodes, 0]
    plot_moving_load_analysis(max_deflections, max_stresses, max_tensions,
                              max_compressions, positions_x, load_magnitudes)

    return max_deflections, max_stresses, positions_x


def generate_complete_report(crane_pkl_file, load_range=(0, 30000)):
    '''
    Generar reporte completo de la estructura optimizada con todos los gráficos y tablas

    Parámetros:
    -----------
    crane_pkl_file : str or Path
        Ruta al archivo .pkl del diseño de grúa optimizado
    load_range : tuple
        Rango de cargas a analizar (min, max) en Newtons (default: 0 a 30 kN)

    Retorna:
    --------
    report_dict : dict
        Diccionario con todos los resultados del análisis
    '''
    # Cargar diseño de grúa optimizado
    crane_data = load_crane(crane_pkl_file)

    # Extraer datos
    X = crane_data['X']
    C = crane_data['C']
    tower_base = crane_data['tower_base']
    boom_top_nodes = crane_data['boom_top_nodes']
    boom_bot_nodes = crane_data['boom_bot_nodes']
    thickness_params = crane_data['thickness_params']

    # Propiedades de material y sección
    E = 200e9  # Módulo de Young del acero (Pa)
    rho_steel = 7800  # kg/m³ (según consigna TP2)
    g = 9.81  # m/s²

    # Construir listas de áreas e inercias de elementos desde thickness_params
    element_areas = []
    element_inertias = []

    for iEl in range(C.shape[0]):
        d_outer = thickness_params[2*iEl]
        d_inner = thickness_params[2*iEl + 1]

        # Asegurar dimensiones válidas
        min_wall = 0.002
        if d_inner >= d_outer - min_wall:
            d_inner = d_outer - min_wall
        if d_inner < 0.001:
            d_inner = 0.001

        A, I = hollow_circular_section(d_outer, d_inner)
        element_areas.append(A)
        element_inertias.append(I)

    # Generar reporte comprensivo usando la función de plotting_utils
    report_dict = generate_comprehensive_report(
        X, C, element_areas, element_inertias,
        boom_top_nodes, boom_bot_nodes, tower_base,
        E, rho_steel, g, load_range
    )

    return report_dict


if __name__ == '__main__':
    import sys

    run_crane_analysis("gruita3_iterations_20251029_225648/best_crane.pkl")
    generate_complete_report("gruita3_iterations_20251029_225648/best_crane.pkl", load_range=(0, 30000))
