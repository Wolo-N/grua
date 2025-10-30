import numpy as np
from fem_utils import hollow_circular_section


def calculate_cost(m, n_elementos, n_uniones, m0=1523, n_elementos_0=51, n_uniones_0=27):
    '''
    Calcular el costo de la estructura de la grúa

    Parámetros:
    -----------
    m : float
        Masa total de la estructura (kg)
    n_elementos : int
        Número de elementos en la estructura
    n_uniones : int
        Número de uniones/juntas en la estructura
    m0 : float
        Masa de referencia (kg), por defecto = 1000 kg
    n_elementos_0 : int
        Número de referencia de elementos, por defecto = 50
    n_uniones_0 : int
        Número de referencia de uniones, por defecto = 100

    Retorna:
    --------
    cost : float
        Costo total basado en la fórmula dada
    '''
    cost = (m / m0) + 1.5 * (n_elementos / n_elementos_0) + 2 * (n_uniones / n_uniones_0)
    return cost


def calculate_security_factors(sigma_max, P_max, sigma_adm=250e6, P_critica=None):
    '''
    Calcular factores de seguridad para la estructura de la grúa

    Parámetros:
    -----------
    sigma_max : float
        Tensión máxima en la estructura (Pa)
    P_max : float
        Carga/fuerza máxima en la estructura (N)
    sigma_adm : float
        Tensión admisible/permitida (Pa), por defecto = 100 MPa
    P_critica : float
        Carga crítica de pandeo (N), opcional

    Retorna:
    --------
    FS_tension : float
        Factor de seguridad para tracción (σ_adm / σ_max)
    FS_pandeo : float o None
        Factor de seguridad para pandeo (P_critica / P_max), si se proporciona P_critica
    '''
    FS_tension = sigma_adm / abs(sigma_max)
    FS_pandeo = None

    if P_critica is not None:
        FS_pandeo = P_critica / abs(P_max)

    return FS_tension, FS_pandeo


def estimate_structure_mass(X, C, element_areas=None, A_main=0.01, A_secondary=0.005, rho_steel=7800):
    '''
    Estimar la masa total de la estructura

    Parámetros:
    -----------
    X : ndarray
        Coordenadas de nodos
    C : ndarray
        Matriz de conectividad
    element_areas : list o array, opcional
        Áreas de cada elemento (m²). Si se proporciona, se usa en lugar de A_main/A_secondary
    A_main : float
        Área de sección transversal de miembros principales (m²) - solo si element_areas=None
    A_secondary : float
        Área de sección transversal de miembros secundarios (m²) - solo si element_areas=None
    rho_steel : float
        Densidad del acero (kg/m³), por defecto = 7800 kg/m³ (según consigna TP2)

    Retorna:
    --------
    total_mass : float
        Masa total de la estructura (kg)
    '''
    total_mass = 0

    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        d_vec = X[n2] - X[n1]
        L = np.linalg.norm(d_vec)

        # Determinar área de sección transversal
        if element_areas is not None:
            # Usar áreas reales de los elementos
            A = element_areas[iEl]
        else:
            # Usar valores por defecto (simplificado: primeros elementos son miembros principales)
            if iEl < 24:  # Número aproximado de elementos de cuerda
                A = A_main
            else:
                A = A_secondary

        # Masa = densidad * volumen = densidad * área * longitud
        element_mass = rho_steel * A * L
        total_mass += element_mass

    return total_mass


def calculate_buckling_load(X, C, element_forces, element_inertias, E, boom_top_nodes=None):
    '''
    Calcular carga crítica de pandeo para miembros en compresión

    Parámetros:
    -----------
    X : ndarray
        Coordenadas de nodos
    C : ndarray
        Matriz de conectividad
    element_forces : ndarray
        Fuerzas en cada elemento
    element_inertias : list o array
        Momentos de inercia de cada elemento
    E : float
        Módulo de Young del material
    boom_top_nodes : list, opcional
        Lista de nodos superiores del brazo (para determinar tipo de elemento)

    Retorna:
    --------
    P_critica : float
        Carga crítica de pandeo mínima (N)
    max_compressed_length : float
        Longitud del miembro comprimido más crítico (m)
    '''
    max_compressed_length = 0
    P_critica = 1e12

    for iEl in range(C.shape[0]):
        if element_forces[iEl] < -1:  # Compresión
            n1, n2 = C[iEl]
            L = np.linalg.norm(X[n2] - X[n1])
            I = element_inertias[iEl]

            # Carga de pandeo de Euler: P_cr = π²EI/L²
            P_cr = (np.pi**2 * E * I) / (L**2)

            if P_cr < P_critica:
                P_critica = P_cr
                max_compressed_length = L

    return P_critica, max_compressed_length


def analyze_cost_and_security(X, C, element_stresses, element_forces, element_inertias,
                              boom_top_nodes=None, element_areas=None, verbose=True):
    '''
    Analizar costo y factores de seguridad para la estructura de la grúa

    Parámetros:
    -----------
    X : ndarray
        Coordenadas de nodos
    C : ndarray
        Matriz de conectividad
    element_stresses : ndarray
        Tensión en cada elemento (Pa)
    element_forces : ndarray
        Fuerza en cada elemento (N)
    element_inertias : list o array
        Momentos de inercia de cada elemento
    boom_top_nodes : list, opcional
        Lista de índices de nodos superiores del brazo
    element_areas : list o array, opcional
        Áreas de cada elemento (m²). Si no se proporciona, usa valores por defecto
    verbose : bool
        Si True, imprime resultados

    Retorna:
    --------
    cost : float
        Costo total de la estructura
    FS_tension : float
        Factor de seguridad para tracción
    FS_pandeo : float o None
        Factor de seguridad para pandeo
    mass : float
        Masa total de la estructura (kg)
    '''
    # Calcular masa de la estructura usando las áreas reales de los elementos
    mass = estimate_structure_mass(X, C, element_areas=element_areas)

    # Número de elementos y uniones
    n_elementos = C.shape[0]
    n_uniones = X.shape[0]

    # Calcular costo
    cost = calculate_cost(mass, n_elementos, n_uniones)

    # Encontrar tensión y fuerza máximas
    sigma_max = np.max(np.abs(element_stresses))
    P_max = np.max(np.abs(element_forces))

    # Tensión admisible
    sigma_adm = 250e6  # 250 MPa (según consigna TP2)

    # Estimar carga crítica de pandeo
    E = 200e9
    P_critica, max_compressed_length = calculate_buckling_load(
        X, C, element_forces, element_inertias, E, boom_top_nodes
    )

    # Calcular factores de seguridad
    FS_tension, FS_pandeo = calculate_security_factors(sigma_max, P_max, sigma_adm, P_critica)

    if verbose:
        # Imprimir resultados
        print("\n" + "="*70)
        print("ANÁLISIS DE COSTO Y FACTORES DE SEGURIDAD")
        print("="*70)
        print(f"\nPropiedades de la Estructura:")
        print(f"  Masa total:           {mass:.2f} kg")
        print(f"  Número de elementos:   {n_elementos}")
        print(f"  Número de uniones:     {n_uniones}")
        print(f"\nAnálisis de Costo:")
        print(f"  Costo Total:           {cost:.4f}")
        print(f"\nFactores de Seguridad:")
        print(f"  Tensión máxima:       {sigma_max/1e6:.2f} MPa")
        print(f"  Tensión admisible:     {sigma_adm/1e6:.2f} MPa")
        print(f"  FS_tension:           {FS_tension:.2f}")
        if FS_tension > 2:
            print(f"  → SEGURO (FS > 2)")
        else:
            print(f"  → ADVERTENCIA: FS < 2 (INSEGURO)")

        if FS_pandeo is not None:
            print(f"\n  Compresión máxima:  {abs(P_max)/1000:.2f} kN")
            print(f"  Pandeo crítico:    {P_critica/1000:.2f} kN")
            print(f"  FS_pandeo:            {FS_pandeo:.2f}")
            if FS_pandeo > 2:
                print(f"  → SEGURO contra pandeo (FS > 2)")
            else:
                print(f"  → ADVERTENCIA: Riesgo de pandeo (FS < 2)")

        print("="*70)

    return cost, FS_tension, FS_pandeo, mass
