import numpy as np


def element_stiffness(n1, n2, A, E):
    '''
    Devuelve la Matriz de Rigidez Elemental de Barra en coordenadas globales

    Parámetros:
    -----------
    n1 : array-like
        Coordenadas del primer nodo [x1, y1]
    n2 : array-like
        Coordenadas del segundo nodo [x2, y2]
    A : float
        Área de sección transversal del elemento
    E : float
        Módulo de Young del material

    Retorna:
    --------
    k_structural : ndarray (4x4)
        Matriz de rigidez del elemento en coordenadas globales
        Orden DOF: [u1, v1, u2, v2] donde u=desplazamiento-x, v=desplazamiento-y
    '''
    # Calcular vector del elemento y longitud
    d = n2 - n1  # Vector del elemento desde nodo 1 al nodo 2
    L = np.linalg.norm(d)  # Longitud del elemento

    # Verificar elementos de longitud cero para evitar división por cero
    if L < 1e-10:
        return np.zeros((4, 4), dtype=float)

    # Cosenos directores (componentes del vector unitario)
    c = d[0] / L  # cos(θ) - coseno dirección x
    s = d[1] / L  # sin(θ) - coseno dirección y

    # Matriz de rigidez local del elemento (elemento barra 1D)
    # En coordenadas locales: k_local = (AE/L) * [1 -1; -1 1]
    k_local = (A * E / L) * np.array([[ 1, -1],
                                      [-1,  1]], dtype=float)

    # Matriz de transformación de coordenadas locales a globales
    # T mapea desplazamientos locales [u_local] a desplazamientos globales [u1, v1, u2, v2]
    T = np.array([[ c, s, 0, 0],  # Desplazamiento local mapeado a DOFs globales
                  [ 0, 0, c, s]], dtype=float)

    # Transformar matriz de rigidez a coordenadas globales: K_global = T^T * K_local * T
    k_structural = np.matmul(T.T, np.matmul(k_local, T))

    return k_structural


def dof_el(nnod1, nnod2):
    '''Devuelve los DOF Elementales para Ensamblaje'''
    return [2*(nnod1+1)-2, 2*(nnod1+1)-1, 2*(nnod2+1)-2, 2*(nnod2+1)-1]


def hollow_circular_section(d_outer, d_inner):
    '''
    Calcular área de sección transversal y momento de inercia para sección circular hueca

    Parámetros:
    -----------
    d_outer : float
        Diámetro exterior (m)
    d_inner : float
        Diámetro interior (m)

    Retorna:
    --------
    A : float
        Área de sección transversal (m²)
    I : float
        Segundo momento de área (m⁴)
    '''
    r_outer = d_outer / 2
    r_inner = d_inner / 2

    # Área de sección transversal: A = π(r_outer² - r_inner²)
    A = np.pi * (r_outer**2 - r_inner**2)

    # Segundo momento de área: I = π/4 * (r_outer⁴ - r_inner⁴)
    I = np.pi / 4 * (r_outer**4 - r_inner**4)

    return A, I


def assemble_global_stiffness(X, C, element_areas, E):
    '''
    Ensamblar matriz de rigidez global

    Parámetros:
    -----------
    X : ndarray
        Coordenadas de nodos
    C : ndarray
        Matriz de conectividad
    element_areas : list o array
        Áreas de sección transversal de cada elemento
    E : float
        Módulo de Young del material

    Retorna:
    --------
    k_global : ndarray
        Matriz de rigidez global
    '''
    n_nodes = X.shape[0]
    n_elements = C.shape[0]
    k_global = np.zeros([2*n_nodes, 2*n_nodes], float)

    for iEl in range(n_elements):
        A = element_areas[iEl]
        dof = dof_el(C[iEl, 0], C[iEl, 1])
        k_elemental = element_stiffness(X[C[iEl, 0], :], X[C[iEl, 1], :], A, E)
        k_global[np.ix_(dof, dof)] += k_elemental

    return k_global


def calculate_element_stresses(X, C, D, element_areas, E):
    '''
    Calcular tensiones en todos los elementos

    Parámetros:
    -----------
    X : ndarray
        Coordenadas de nodos
    C : ndarray
        Matriz de conectividad
    D : ndarray
        Desplazamientos nodales
    element_areas : list o array
        Áreas de sección transversal de cada elemento
    E : float
        Módulo de Young del material

    Retorna:
    --------
    element_forces : ndarray
        Fuerzas axiales en cada elemento
    element_stresses : ndarray
        Tensiones en cada elemento
    '''
    n_elements = C.shape[0]
    element_forces = np.zeros(n_elements)
    element_stresses = np.zeros(n_elements)

    for iEl in range(n_elements):
        n1, n2 = C[iEl]
        d_el = np.concatenate([D[n1], D[n2]])
        A = element_areas[iEl]

        k_el = element_stiffness(X[n1], X[n2], A, E)
        f_el = k_el @ d_el

        d_vec = X[n2] - X[n1]
        L = np.linalg.norm(d_vec)

        if L < 1e-10:
            element_forces[iEl] = 0
            element_stresses[iEl] = 0
            continue

        u_vec = d_vec / L
        du = np.array([d_el[2] - d_el[0], d_el[3] - d_el[1]])
        epsilon = np.dot(du, u_vec) / L
        stress = E * epsilon
        force = stress * A

        element_forces[iEl] = force
        element_stresses[iEl] = stress

    return element_forces, element_stresses


def apply_self_weight(X, C, element_areas, loads, rho_steel=7850, g=9.81):
    '''
    Agregar peso propio de la estructura a las cargas

    Parámetros:
    -----------
    X : ndarray
        Coordenadas de nodos
    C : ndarray
        Matriz de conectividad
    element_areas : list o array
        Áreas de sección transversal de cada elemento
    loads : ndarray
        Matriz de cargas nodales (se modifica in-place)
    rho_steel : float
        Densidad del acero (kg/m³)
    g : float
        Aceleración gravitacional (m/s²)

    Retorna:
    --------
    None (modifica loads in-place)
    '''
    for iEl in range(C.shape[0]):
        n1, n2 = C[iEl]
        L = np.linalg.norm(X[n2] - X[n1])
        A = element_areas[iEl]
        volume = A * L
        mass = rho_steel * volume
        weight = mass * g
        loads[n1, 1] -= weight / 2
        loads[n2, 1] -= weight / 2
