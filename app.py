# @title Laboratorio de Clustering Interactivo 🔬
# @markdown Ejecuta esta celda para iniciar el laboratorio.

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

# --- Configuración Visual ---
plt.style.use('ggplot')
COLORS = ['#ef4444', '#8b5cf6', '#10b981', '#f59e0b', '#06b6d4', '#ec4899', '#6366f1', '#84cc16']
COLOR_NOISE = '#cbd5e1'

# --- 1. Generación de Datos ---
def get_dataset(name, n_samples=600):
    if name == 'Marketing (Blobs)':
        X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=0.6, random_state=42)
    elif name == 'Sensor (Lunas)':
        X, _ = make_moons(n_samples=n_samples, noise=0.08, random_state=42)
    elif name == 'Geo (Anillos)':
        X, _ = make_circles(n_samples=n_samples, factor=0.4, noise=0.06, random_state=42)
    
    # Normalizamos para que los parámetros visuales (epsilon) sean consistentes
    return MinMaxScaler().fit_transform(X)

# --- 2. Lógica de Interpretación ---
def get_interpretation(algo, data_name):
    # Mensajes didácticos idénticos a tu versión web
    key = f"{data_name}-{algo}"
    if 'Marketing' in data_name:
        return "✅ <b>IDEAL:</b> Los datos forman grupos compactos ('globos'). K-Means encuentra el centro fácilmente."
    
    if 'Sensor' in data_name: # Lunas
        if algo == 'K-Means': return "❌ <b>ERROR CRÍTICO:</b> K-Means corta las lunas con líneas rectas. Asume que los grupos son redondos."
        if algo == 'DBSCAN': return "✅ <b>EXCELENTE:</b> DBSCAN sigue la curvatura uniendo puntos densos y descarta el ruido."
        return "⚠️ <b>REGULAR:</b> El Jerárquico puede funcionar, pero depende mucho del método de enlace."

    if 'Geo' in data_name: # Anillos
        if algo == 'K-Means': return "❌ <b>FALLO VISUAL:</b> Divide los anillos como un pastel. No entiende que un grupo esté dentro de otro."
        if algo == 'DBSCAN': return "✅ <b>EL MEJOR:</b> Navega por el anillo sin saltar al vacío del centro."
        return "✅ <b>INTERESANTE:</b> El Jerárquico suele lograr cerrar los anillos uniendo eslabones."
    
    return "Selecciona una combinación."

# --- 3. Interfaz Gráfica (Widgets) ---

# Controles
dd_dataset = widgets.Dropdown(options=['Geo (Anillos)', 'Sensor (Lunas)', 'Marketing (Blobs)'], value='Geo (Anillos)', description='Datos:')
dd_algo = widgets.ToggleButtons(options=['K-Means', 'DBSCAN', 'Jerárquico'], value='K-Means', description='Algoritmo:', style={'button_width': '100px'})

# Sliders
slider_k = widgets.IntSlider(value=3, min=2, max=8, step=1, description='K Clusters:')
slider_eps = widgets.FloatSlider(value=0.06, min=0.02, max=0.2, step=0.01, description='Radio (Eps):', readout_format='.2f')
slider_min = widgets.IntSlider(value=4, min=2, max=10, description='Min Pts:')
check_lines = widgets.Checkbox(value=True, description='Ver Conexiones (Solo K-Means)')

# Salida Gráfica
out_plot = widgets.Output()
out_info = widgets.HTML()

# --- 4. Función Principal de Renderizado ---
def update_view(change=None):
    # Obtener valores
    dataset_name = dd_dataset.value
    algo_name = dd_algo.value
    k = slider_k.value
    eps = slider_eps.value
    min_samples = slider_min.value
    show_lines = check_lines.value

    # Gestionar visibilidad de controles según algoritmo
    if algo_name == 'DBSCAN':
        slider_k.layout.display = 'none'
        check_lines.layout.display = 'none'
        slider_eps.layout.display = 'flex'
        slider_min.layout.display = 'flex'
    else:
        slider_k.layout.display = 'flex'
        check_lines.layout.display = 'flex' if algo_name == 'K-Means' else 'none'
        slider_eps.layout.display = 'none'
        slider_min.layout.display = 'none'

    # Procesar Datos
    X = get_dataset(dataset_name)
    
    # Ejecutar Algoritmo
    centroids = None
    if algo_name == 'K-Means':
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(X)
        centroids = model.cluster_centers_
    elif algo_name == 'DBSCAN':
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
    else:
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X)

    # --- DIBUJAR ---
    with out_plot:
        clear_output(wait=True) # Evita parpadeo excesivo
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Colores
        colors = [COLOR_NOISE if l == -1 else COLORS[l % len(COLORS)] for l in labels]
        
        # 1. Líneas de conexión (K-Means Didáctico)
        if algo_name == 'K-Means' and show_lines and centroids is not None:
            for i, point in enumerate(X):
                c = centroids[labels[i]]
                ax.plot([point[0], c[0]], [point[1], c[1]], color=COLORS[labels[i] % len(COLORS)], alpha=0.1, linewidth=0.5, zorder=1)

        # 2. Puntos
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=40, edgecolors='white', linewidth=0.5, alpha=0.8, zorder=2)

        # 3. Centroides
        if centroids is not None:
            ax.scatter(centroids[:, 0], centroids[:, 1], c='white', s=200, marker='X', edgecolors='black', linewidth=2, zorder=3)

        # Estética
        ax.set_title(f"{algo_name} en {dataset_name}", fontsize=14, fontweight='bold', color='#333')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        
        # Borde redondeado simulado
        for spine in ax.spines.values():
            spine.set_edgecolor('#ddd')
            
        plt.show()

    # --- TEXTO DIDÁCTICO ---
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    html_content = f"""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 15px; margin-top: 10px; font-family: sans-serif;">
        <div style="display: flex; gap: 20px; margin-bottom: 10px; color: #64748b; font-size: 0.9em; font-weight: bold; text-transform: uppercase;">
            <span>🧩 Clusters: <span style="color: #4f46e5; font-size: 1.2em;">{n_clusters}</span></span>
            <span style="border-left: 1px solid #cbd5e1; padding-left: 20px;">🗑️ Ruido: <span style="color: {'#ef4444' if n_noise > 0 else '#64748b'}; font-size: 1.2em;">{n_noise}</span></span>
        </div>
        <div style="background-color: #e0e7ff; color: #3730a3; padding: 10px; border-radius: 8px; border-left: 5px solid #4f46e5;">
            {get_interpretation(algo_name, dataset_name)}
        </div>
    </div>
    """
    out_info.value = html_content

# Conectar eventos
dd_dataset.observe(update_view, names='value')
dd_algo.observe(update_view, names='value')
slider_k.observe(update_view, names='value')
slider_eps.observe(update_view, names='value')
slider_min.observe(update_view, names='value')
check_lines.observe(update_view, names='value')

# Layout final
ui = widgets.VBox([
    widgets.HTML("<h2 style='color:#4f46e5; margin-bottom:5px;'>🧪 Laboratorio de Clustering Pro</h2>"),
    widgets.HBox([dd_dataset, dd_algo]),
    widgets.HBox([slider_k, slider_eps, slider_min]),
    check_lines,
    out_plot,
    out_info
])

# Iniciar
update_view()
display(ui)
