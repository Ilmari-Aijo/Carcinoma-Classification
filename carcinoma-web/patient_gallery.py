
import os
import re
import json
from PIL import Image
 
# --- CONFIGURACION: ajusta estas dos rutas a tu maquina ---
DATASET_PATH = r"C:\Users\ilmar\Downloads\archive"
OUTPUT_DIR = r"C:\Users\ilmar\Documents\proyectos\Carcinoma Project\carcinoma-web\static\gallery"
 
PATCH_SIZE = 50   # cada parche del dataset es 50x50 px
GRID_CELLS = 30   # ventana de 10x10 parches = canvas de 500x500 px (solo se usa si FULL_COVERAGE = False)
 
# FULL_COVERAGE = True  -> reconstruye TODA el area donde el paciente tiene
#                          parches, sin recortar a una ventana fija. Util
#                          para explorar/entender como se ve el tejido
#                          completo (vas a ver huecos blancos donde Kaggle
#                          no genero parche, eso es normal).
# FULL_COVERAGE = False -> busca la ventana mas densa de GRID_CELLS x
#                          GRID_CELLS (modo liviano, pensado para el
#                          gallery final que va al repo/hosting).
FULL_COVERAGE = False
 
# Los 10 pacientes elegidos del test set real (seed=42 sobre los 18
# pacientes que imprimio find_test_patients.py), para que la seleccion
# sea reproducible y no un capricho manual.
SELECTED_PATIENTS = [
    '10254', '10272', '10300', '12890', '12954',
    '14082', '14305', '15472', '15839', '9250',
]
 
# Los nombres de archivo del dataset siguen el patron:
#   <patient_id>_idx5_x<X>_y<Y>_class<C>.png
# Esta regex extrae X, Y y la clase (C) sin importar el resto del nombre.
FILENAME_RE = re.compile(r"_x(\d+)_y(\d+)_class(\d+)")
 
 
def parse_patches(patient_id):
    """Recorre las carpetas 0/ y 1/ de un paciente y devuelve una lista de
    diccionarios: {x, y, label, filepath} para cada parche encontrado."""
    patches = []
    patient_path = os.path.join(DATASET_PATH, patient_id)
 
    for label in ['0', '1']:
        label_path = os.path.join(patient_path, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            match = FILENAME_RE.search(fname)
            if not match:
                continue  # archivo que no matchea el patron esperado, se ignora
            x_str, y_str, cls_str = match.groups()
            patches.append({
                'x': int(x_str),
                'y': int(y_str),
                'label': int(cls_str),
                'filepath': os.path.join(label_path, fname),
            })
 
    return patches
 
 
def find_densest_window(patches, grid_cells=GRID_CELLS, patch_size=PATCH_SIZE):
    """Convierte las coordenadas absolutas (x, y) en indices de grilla
    (columna, fila) dividiendo por el tamano del parche, y despues prueba
    todas las ventanas posibles de grid_cells x grid_cells para encontrar
    la que contiene mas parches reales (menos huecos)."""
 
    for p in patches:
        p['gx'] = p['x'] // patch_size
        p['gy'] = p['y'] // patch_size
 
    min_gx = min(p['gx'] for p in patches)
    max_gx = max(p['gx'] for p in patches)
    min_gy = min(p['gy'] for p in patches)
    max_gy = max(p['gy'] for p in patches)
 
    best_count = -1
    best_origin = (min_gx, min_gy)
 
    # Fuerza bruta: para un paciente individual la grilla es chica
    # (tipicamente decenas x decenas de celdas), asi que esto corre
    # en segundos, no hace falta optimizarlo.
    for origin_x in range(min_gx, max_gx - grid_cells + 2):
        for origin_y in range(min_gy, max_gy - grid_cells + 2):
            count = sum(
                1 for p in patches
                if origin_x <= p['gx'] < origin_x + grid_cells
                and origin_y <= p['gy'] < origin_y + grid_cells
            )
            if count > best_count:
                best_count = count
                best_origin = (origin_x, origin_y)
 
    return best_origin, best_count
 
 
def compute_full_bounds(patches, patch_size=PATCH_SIZE):
    """Variante de find_densest_window para el modo FULL_COVERAGE: en vez
    de buscar una ventana fija, calcula el rectangulo minimo que cubre
    TODOS los parches del paciente. Devuelve el origen (esquina superior
    izquierda en coordenadas de grilla) y el ancho/alto en celdas."""
 
    for p in patches:
        p['gx'] = p['x'] // patch_size
        p['gy'] = p['y'] // patch_size
 
    min_gx = min(p['gx'] for p in patches)
    max_gx = max(p['gx'] for p in patches)
    min_gy = min(p['gy'] for p in patches)
    max_gy = max(p['gy'] for p in patches)
 
    width_cells = max_gx - min_gx + 1
    height_cells = max_gy - min_gy + 1
 
    return (min_gx, min_gy), width_cells, height_cells
 
 
def build_composite(patches, origin, width_cells, height_cells, patch_size=PATCH_SIZE):
    """Pega los parches reales dentro del area elegida (ventana fija o
    cobertura completa, segun el modo) en un lienzo blanco. Devuelve la
    imagen compuesta y la metadata de cada celda ocupada (fila, columna,
    posicion en pixeles, label)."""
 
    origin_x, origin_y = origin
    canvas_w = width_cells * patch_size
    canvas_h = height_cells * patch_size
    canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
 
    cell_metadata = []
 
    for p in patches:
        rel_gx = p['gx'] - origin_x
        rel_gy = p['gy'] - origin_y
 
        if 0 <= rel_gx < width_cells and 0 <= rel_gy < height_cells:
            patch_img = Image.open(p['filepath']).convert('RGB')
            px = rel_gx * patch_size
            py = rel_gy * patch_size
            canvas.paste(patch_img, (px, py))
 
            cell_metadata.append({
                'row': rel_gy,
                'col': rel_gx,
                'pixel_x': px,
                'pixel_y': py,
                'label': p['label'],
            })
 
    return canvas, cell_metadata
 
 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = {}
 
    for patient_id in SELECTED_PATIENTS:
        patches = parse_patches(patient_id)
 
        if not patches:
            print(f"Paciente {patient_id}: no se encontraron parches, se omite.")
            continue
 
        if FULL_COVERAGE:
            origin, width_cells, height_cells = compute_full_bounds(patches)
        else:
            origin, _ = find_densest_window(patches)
            width_cells = height_cells = GRID_CELLS
 
        composite, cells = build_composite(patches, origin, width_cells, height_cells)
 
        image_filename = f"{patient_id}.png"
        composite.save(os.path.join(OUTPUT_DIR, image_filename))
 
        manifest[patient_id] = {
            'image': image_filename,
            'patch_size': PATCH_SIZE,
            'grid_width': width_cells,
            'grid_height': height_cells,
            'cells': cells,
        }
 
        total_cells = width_cells * height_cells
        canvas_w, canvas_h = composite.size
        print(
            f"Paciente {patient_id}: {len(cells)}/{total_cells} celdas con tejido real "
            f"(canvas {canvas_w}x{canvas_h} px)."
        )
        if canvas_w > 3000 or canvas_h > 3000:
            print(
                f"  Aviso: el composite de {patient_id} es grande "
                f"({canvas_w}x{canvas_h} px, PNG puede pesar varios MB). "
                f"Esta bien para explorar, pero conviene achicarlo (FULL_COVERAGE=False) "
                f"antes de subirlo al repo."
            )
 
    manifest_path = os.path.join(OUTPUT_DIR, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
 
    print(f"\nListo. Composites y manifest.json guardados en:\n  {OUTPUT_DIR}")
 
 
if __name__ == '__main__':
    main()