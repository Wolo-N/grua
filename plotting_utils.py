import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

"""
PLOTTING UTILITIES FOR CRANE ANALYSIS
======================================
Unified plotting functions for visualizing crane structures and analysis results.
"""

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


def plot_optimized_crane(X, C, element_types, thickness_params,
                        n_segments, boom_height, connectivity_pattern):
    '''Graficar el diseño optimizado de la grúa'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Gráfico 1: Geometría de grúa con tipos de miembros
    n_elements = C.shape[0]

    colors = {
        'top_strut': 'blue',
        'bot_strut': 'green',
        'vertical': 'purple',
        'diagonal': 'orange',
        'support': 'red'
    }

    for iEl in range(n_elements):
        etype = element_types[iEl]
        color = colors.get(etype, 'gray')
        if 2*iEl < len(thickness_params):
            d_outer = thickness_params[2*iEl] * 1000  # Convertir a mm
            linewidth = max(1, d_outer / 10)  # Escalar ancho de línea con espesor
        else:
            linewidth = 1.5

        ax1.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                [X[C[iEl,0],1], X[C[iEl,1],1]],
                color=color, linewidth=linewidth, alpha=0.7)

    ax1.scatter(X[:,0], X[:,1], c='black', s=30, zorder=5)
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_title(f'Diseño de Grúa Optimizado\n{n_segments} segmentos, h={boom_height:.2f}m',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend(handles=[plt.Line2D([0], [0], color=c, linewidth=2, label=t)
                       for t, c in colors.items()], loc='upper right')

    # Gráfico 2: Distribución de espesores de miembros
    thicknesses_outer = [thickness_params[2*i]*1000 for i in range(min(n_elements, len(thickness_params)//2))]
    thicknesses_inner = [thickness_params[2*i+1]*1000 for i in range(min(n_elements, len(thickness_params)//2))]
    wall_thickness = [thicknesses_outer[i] - thicknesses_inner[i] for i in range(len(thicknesses_outer))]

    x_pos = np.arange(len(thicknesses_outer))
    ax2.bar(x_pos, thicknesses_outer, label='Diámetro exterior', alpha=0.7)
    ax2.bar(x_pos, thicknesses_inner, label='Diámetro interior', alpha=0.7)
    ax2.axhline(y=50, color='r', linestyle='--', linewidth=2, label='Diámetro ext máx (50mm)')
    ax2.set_xlabel('Índice de elemento', fontsize=12)
    ax2.set_ylabel('Diámetro (mm)', fontsize=12)
    ax2.set_title('Distribución de Espesores de Miembros', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    # Imprimir resumen
    print("\n" + "="*80)
    print("RESUMEN DE DISEÑO OPTIMIZADO")
    print("="*80)
    diag_names = {
        0: "Alternado (Warren)",
        1: "Todos positivos",
        2: "Todos negativos",
        3: "Patrón X",
        4: "Abanico desde abajo",
        5: "Abanico desde arriba",
        6: "Largo alcance",
        7: "Alcance mixto",
        8: "Concentrado",
        9: "Abanico progresivo",
        10: "Conectividad completa"
    }

    print(f"Segmentos: {n_segments}")
    print(f"Altura del brazo: {boom_height:.3f} m")
    diag_num = int(connectivity_pattern[0])
    print(f"Tipo de diagonal: {diag_num} - {diag_names.get(diag_num, 'Desconocido')}")
    print(f"Espaciado vertical: cada {int(connectivity_pattern[1])} nodos")
    print(f"Cables de soporte: {int(connectivity_pattern[2])}")
    print(f"\nTotal de elementos: {n_elements}")
    print(f"Total de nodos: {X.shape[0]}")
    if thicknesses_outer:
        print(f"\nRango de espesores de miembros:")
        print(f"  Diámetro exterior: {min(thicknesses_outer):.1f} - {max(thicknesses_outer):.1f} mm")
        print(f"  Espesor de pared: {min(wall_thickness):.1f} - {max(wall_thickness):.1f} mm")
    print("="*80)


def plot_moving_load_analysis(max_deflections, max_stresses, max_tensions,
                               max_compressions, positions_x, load_magnitudes):
    '''
    Graficar resultados del análisis de carga móvil

    Parámetros:
    -----------
    max_deflections : ndarray
        Deflexiones máximas para cada combinación de carga/posición
    max_stresses : ndarray
        Tensiones máximas para cada combinación
    max_tensions : ndarray
        Tensiones de tracción máximas
    max_compressions : ndarray
        Tensiones de compresión máximas
    positions_x : ndarray
        Posiciones x a lo largo de la grúa
    load_magnitudes : ndarray
        Magnitudes de carga probadas
    '''
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Gráfico 1: Deflexión máx vs posición
    ax1 = axes[0, 0]
    for i, load_mag in enumerate(load_magnitudes):
        ax1.plot(positions_x, max_deflections[i, :] * 1000, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax1.set_xlabel('Posición a lo largo de la grúa (m)', fontsize=12)
    ax1.set_ylabel('Deflexión máxima (mm)', fontsize=12)
    ax1.set_title('Deflexión Máxima vs Posición de Carga', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Tensión máx vs posición
    ax2 = axes[0, 1]
    for i, load_mag in enumerate(load_magnitudes):
        ax2.plot(positions_x, max_stresses[i, :] / 1e6, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax2.axhline(y=100, color='r', linestyle='--', label='Admisible (100 MPa)')
    ax2.set_xlabel('Posición a lo largo de la grúa (m)', fontsize=12)
    ax2.set_ylabel('Tensión máxima (MPa)', fontsize=12)
    ax2.set_title('Tensión Máxima vs Posición de Carga', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Gráfico 3: Tensión de tracción máx vs posición
    ax3 = axes[1, 0]
    for i, load_mag in enumerate(load_magnitudes):
        ax3.plot(positions_x, max_tensions[i, :] / 1e6, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax3.axhline(y=100, color='r', linestyle='--', label='Admisible (100 MPa)')
    ax3.set_xlabel('Posición a lo largo de la grúa (m)', fontsize=12)
    ax3.set_ylabel('Tensión de tracción máxima (MPa)', fontsize=12)
    ax3.set_title('Tensión de Tracción Máxima vs Posición de Carga', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Gráfico 4: Tensión de compresión máx vs posición
    ax4 = axes[1, 1]
    for i, load_mag in enumerate(load_magnitudes):
        ax4.plot(positions_x, max_compressions[i, :] / 1e6, 'o-',
                label=f'{load_mag/1000:.1f} kN', linewidth=2, markersize=6)
    ax4.axhline(y=-100, color='r', linestyle='--', label='Admisible (-100 MPa)')
    ax4.set_xlabel('Posición a lo largo de la grúa (m)', fontsize=12)
    ax4.set_ylabel('Tensión de compresión máxima (MPa)', fontsize=12)
    ax4.set_title('Tensión de Compresión Máxima vs Posición de Carga', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Imprimir resumen
    print("\n" + "="*70)
    print("RESUMEN DE ANÁLISIS DE CARGA MÓVIL")
    print("="*70)
    print(f"Deflexión máxima general: {np.max(max_deflections)*1000:.2f} mm")
    print(f"Tensión máxima general: {np.max(max_stresses)/1e6:.2f} MPa")
    print(f"Tensión de tracción máxima: {np.max(max_tensions)/1e6:.2f} MPa")
    print(f"Tensión de compresión máxima: {np.min(max_compressions)/1e6:.2f} MPa")
    print("="*70)


def animate_moving_load(X, C, element_areas, boom_top_nodes, boom_bot_nodes,
                        tower_base, E, rho_steel, g,
                        load_magnitude=30000, scale_factor=100, interval=200):
    '''
    Crear un time-lapse animado de deformación de grúa mientras la carga se mueve a lo largo de la cuerda inferior

    Parámetros:
    -----------
    X : ndarray
        Coordenadas de nodos
    C : ndarray
        Matriz de conectividad
    element_areas : list
        Áreas de elementos
    boom_top_nodes : list
        Índices de nodos superiores
    boom_bot_nodes : list
        Índices de nodos inferiores
    tower_base : int
        Índice del nodo base de la torre
    E : float
        Módulo de Young
    rho_steel : float
        Densidad del acero
    g : float
        Aceleración gravitacional
    load_magnitude : float
        Magnitud de la carga móvil en Newtons (default: 30000 N = 30 kN)
    scale_factor : float
        Factor de escala para visualización de deformación (default: 100)
    interval : int
        Tiempo entre cuadros en milisegundos (default: 200 ms)
    '''
    from fem_utils import assemble_global_stiffness, apply_self_weight

    print("\n" + "="*70)
    print("ANÁLISIS DE CARGA MÓVIL ANIMADO")
    print("="*70)
    print(f"Magnitud de carga: {load_magnitude/1000:.1f} kN")
    print(f"Escala de deformación: {scale_factor}x")
    print("Creando animación...")

    n_nodes = X.shape[0]

    # Condiciones de borde
    bc = np.full((n_nodes, 2), False)
    bc[boom_bot_nodes[0], :] = True
    bc[tower_base, :] = True
    bc_mask = bc.reshape(1, 2*n_nodes).ravel()

    # Ensamblar matriz de rigidez global
    k_global = assemble_global_stiffness(X, C, element_areas, E)
    k_reduced = k_global[~bc_mask][:, ~bc_mask]

    # Posiciones de prueba
    test_positions = boom_bot_nodes

    # Calcular deformaciones para cada posición
    deformations = []
    max_deflections_list = []

    for pos_node in test_positions:
        loads = np.zeros([n_nodes, 2], float)
        loads[pos_node, 1] = -load_magnitude

        # Agregar peso propio
        apply_self_weight(X, C, element_areas, loads, rho_steel, g)

        load_vector = loads.reshape(1, 2*n_nodes).ravel()[~bc_mask]

        # Resolver
        displacements = np.zeros([2*n_nodes], float)
        displacements[~bc_mask] = np.linalg.solve(k_reduced, load_vector)
        D = displacements.reshape(n_nodes, 2)

        deformations.append(D)
        tip_deflection = abs(D[boom_top_nodes[-1], 1])
        max_deflections_list.append(tip_deflection)

    # Crear animación
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

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
        for iEl in range(C.shape[0]):
            ax1.plot([X[C[iEl,0],0], X[C[iEl,1],0]],
                    [X[C[iEl,0],1], X[C[iEl,1],1]],
                    'gray', linewidth=1, alpha=0.3, linestyle='--')

        for iEl in range(C.shape[0]):
            ax1.plot([X_deformed[C[iEl,0],0], X_deformed[C[iEl,1],0]],
                    [X_deformed[C[iEl,0],1], X_deformed[C[iEl,1],1]],
                    'b-', linewidth=2)

        # Marcar posición de carga
        load_pos = X[pos_node]
        ax1.arrow(load_pos[0], load_pos[1] + 0.5, 0, -0.3,
                 head_width=0.5, head_length=0.2, fc='red', ec='red', linewidth=3)
        ax1.plot(load_pos[0], load_pos[1], 'ro', markersize=15, label='Posición de Carga')

        # Marcar apoyos
        ax1.plot(X[boom_bot_nodes[0], 0], X[boom_bot_nodes[0], 1], 'g^',
                markersize=12, label='Apoyo de Pasador')
        ax1.plot(X[tower_base, 0], X[tower_base, 1], 'g^', markersize=12)

        ax1.set_xlabel('x (m)', fontsize=12)
        ax1.set_ylabel('y (m)', fontsize=12)
        ax1.set_title(f'Deformación de Grúa - Carga en x = {X[pos_node, 0]:.2f} m (Escala: {scale_factor}x)\n'
                     f'Carga: {load_magnitude/1000:.1f} kN | Deflexión en Punta: {max_deflections_list[frame]*1000:.2f} mm',
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        ax1.legend(loc='upper right')

        # Gráfico 2: Perfil de deflexión en punta
        positions_x = X[boom_bot_nodes, 0]
        ax2.plot(positions_x, np.array(max_deflections_list) * 1000, 'b-o', linewidth=2, markersize=8)
        ax2.plot(X[pos_node, 0], max_deflections_list[frame] * 1000, 'ro', markersize=15)
        ax2.axvline(x=X[pos_node, 0], color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Posición de Carga a lo largo de la grúa (m)', fontsize=12)
        ax2.set_ylabel('Deflexión en Punta (mm)', fontsize=12)
        ax2.set_title('Deflexión en Punta vs Posición de Carga', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return []

    anim = FuncAnimation(fig, update, init_func=init, frames=len(test_positions),
                        interval=interval, blit=False, repeat=True)

    print("Animación creada! Cierre la ventana para continuar...")
    plt.show()

    print("="*70)

    return anim
