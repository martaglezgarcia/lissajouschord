import numpy as np
import trimesh

# Diccionario de frecuencias de notas (en Hz) para la cuarta octava como base
# Puedes expandir esto para incluir más octavas si es necesario.
# Por simplicidad, nos centraremos en la cuarta octava (C4 a B4) y una octava superior.
NOTE_FREQUENCIES = {
    'C4': 261.63,
    'C#4': 277.18, 'Db4': 277.18,
    'D4': 293.66,
    'D#4': 311.13, 'Eb4': 311.13,
    'E4': 329.63,
    'F4': 349.23,
    'F#4': 369.99, 'Gb4': 369.99,
    'G4': 392.00,
    'G#4': 415.30, 'Ab4': 415.30,
    'A4': 440.00,
    'A#4': 466.16, 'Bb4': 466.16,
    'B4': 493.88,
    'C5': 523.25, # Una octava más arriba
    'D5': 587.33,
    'E5': 659.25,
    'F5': 698.46,
    'G5': 783.99,
    'A5': 880.00,
    'B5': 987.77
}

def get_note_frequency(note_name):
    """
    Obtiene la frecuencia de una nota dada su nombre (ej. 'C4', 'E4').
    Convierte el nombre a mayúsculas y maneja los sostenidos/bemoles.
    """
    note_name = note_name.upper().replace('B#', 'C').replace('E#', 'F').replace('Cb', 'B').replace('Fb', 'E')
    # Ajusta para sostenidos/bemoles si el usuario no usa # o b.
    if len(note_name) > 2 and note_name[1] == '#':
        note_name = note_name[0] + '#' + note_name[2:]
    elif len(note_name) > 2 and note_name[1] == 'B': # Para Db, Eb, Gb, Ab, Bb
        note_name = note_name[0] + 'b' + note_name[2:]

    return NOTE_FREQUENCIES.get(note_name)

def create_lissajous_from_notes_3d(
    note1_name, note2_name, note3_name,
    duration=0.05, # Duración por defecto para una visualización compacta
    amplitude_x=1,
    amplitude_y=1,
    amplitude_z=1,
    phase_x=0,
    phase_y=0, 
    phase_z=0,
    tube_radius=0.05,
    cylinder_segments=16,
    output_filename="lissajous_3d_custom_notes.stl"
):
    """
    Genera una curva de Lissajous 3D basada en tres notas especificadas
    y la exporta como un archivo STL creando y uniendo segmentos de cilindro.

    Args:
        note1_name (str): Nombre de la primera nota (ej. 'C4').
        note2_name (str): Nombre de la segunda nota (ej. 'E4').
        note3_name (str): Nombre de la tercera nota (ej. 'G4').
        duration (float): Duración de la simulación en segundos.
        amplitude_x (float): Amplitud de la onda en el eje X.
        amplitude_y (float): Amplitud de la onda en el eje Y.
        amplitude_z (float): Amplitud de la onda en el eje Z.
        phase_x (float): Fase inicial de la onda X en radianes.
        phase_y (float): Fase inicial de la onda Y en radianes.
        phase_z (float): Fase inicial de la onda Z en radianes.
        tube_radius (float): Radio del "tubo" que forma la curva.
        cylinder_segments (int): Número de segmentos para cada cilindro.
        output_filename (str): Nombre del archivo STL de salida.
    """

    # Obtener frecuencias de las notas
    omega_x = get_note_frequency(note1_name)
    omega_y = get_note_frequency(note2_name)
    omega_z = get_note_frequency(note3_name)

    # Validar que todas las notas fueron encontradas
    if None in [omega_x, omega_y, omega_z]:
        print(f"Error: Una o más notas no son válidas. Notas válidas: {list(NOTE_FREQUENCIES.keys())}")
        return None

    print(f"Generando Lissajous 3D para {note1_name} ({omega_x:.2f} Hz), {note2_name} ({omega_y:.2f} Hz), {note3_name} ({omega_z:.2f} Hz)")

    # --- Creación del vector 't' usando np.arange para definir el paso ---
    start_time = 0.0
    end_time = duration
    sample_rate = 44100.0
    step_size = 1.0 / sample_rate
    t = np.arange(start_time, end_time + step_size / 2, step_size)

    # Ecuaciones de la curva de Lissajous
    x = amplitude_x * np.sin(2 * np.pi * omega_x * t + phase_x)
    y = amplitude_y * np.sin(2 * np.pi * omega_y * t + phase_y)
    z = amplitude_z * np.sin(2 * np.pi * omega_z * t + phase_z)

    points = np.vstack([x, y, z]).T

    # --- Configuración para los segmentos de cilindro ---
    all_meshes = []

    # Itera a través de los puntos para crear un cilindro entre cada par de puntos consecutivos
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i+1]

        segment_vector = end_point - start_point
        segment_length = np.linalg.norm(segment_vector)

        if segment_length < 1e-6: # Evitar segmentos de longitud cero
            continue

        cylinder_segment = trimesh.creation.cylinder(radius=tube_radius,
                                                     height=segment_length,
                                                     sections=cylinder_segments)

        # Transformación para alinear y posicionar el cilindro
        from_v = np.array([0, 0, 1]) # Vector original del cilindro (alineado con Z)
        to_v = segment_vector / segment_length # Vector al que queremos alinear el cilindro

        # Calcular matriz de rotación
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.arccos(np.dot(from_v, to_v)), np.cross(from_v, to_v)
        )

        mid_point = (start_point + end_point) / 2

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix[:3, :3]
        transform_matrix[:3, 3] = mid_point

        transformed_cylinder = cylinder_segment.apply_transform(transform_matrix)
        all_meshes.append(transformed_cylinder)

    if not all_meshes:
        print("Advertencia: No se generaron segmentos de curva. La duración o el sample_rate podrían ser demasiado bajos.")
        final_mesh = trimesh.creation.icosphere(radius=tube_radius / 2) # Crear una esfera pequeña como fallback
    else:
        final_mesh = trimesh.util.concatenate(all_meshes)

    final_mesh.export(output_filename)
    print(f"Curva de Lissajous guardada como {output_filename}")
    return final_mesh


def main_interactive_lissajous_creator():
    """
    Función principal para interactuar con el usuario, pedir notas
    y generar la figura 3D de Lissajous.
    """
    print("\n--- Generador de Curvas de Lissajous 3D a partir de Triadas de Notas ---")
    print("Notas disponibles (Octava 4 y 5): C, C#, D, D#, E, F, F#, G, G#, A, A#, B (ej. C4, E4, B5)")
    print("También puedes usar Db, Eb, Gb, Ab, Bb en lugar de los sostenidos.")

    notes = []
    for i in range(1, 4):
        while True:
            note_input = input(f"Introduce la nota {i} (ej. C4, E4): ").strip()
            # Normalizar la entrada para búsqueda
            normalized_note = note_input.upper().replace('B#', 'C').replace('E#', 'F').replace('CB', 'B').replace('FB', 'E')
            if len(normalized_note) > 1 and normalized_note[-1].isdigit(): # Simple check for note + octave
                if normalized_note in NOTE_FREQUENCIES:
                    notes.append(normalized_note)
                    break
                # Try adjusting for single char notes like "C" assuming default octave
                elif len(normalized_note) == 1 and normalized_note + '4' in NOTE_FREQUENCIES:
                    print(f"Asumiendo octava 4 para '{normalized_note}'. Usando {normalized_note}4.")
                    notes.append(normalized_note + '4')
                    break
                elif len(normalized_note) == 2 and normalized_note[0].isalpha() and normalized_note[1].isdigit():
                    # For cases like 'C4' or 'A5' where user might type 'C4' as is.
                    # Or 'CS' 'DB' for C# or Db.
                    # This branch tries to catch simple typos or alternative notations.
                    found = False
                    for key in NOTE_FREQUENCIES:
                        if key.upper() == normalized_note:
                            notes.append(key)
                            found = True
                            break
                    if found:
                        break
            
            print(f"Nota '{note_input}' no reconocida o formato inválido. Por favor, usa el formato Nota+Octava (ej. C4, D#4, Bb4).")

    # Pedir duración (opcional, con valor por defecto)
    while True:
        try:
            duration_input = input(f"Introduce la duración de la curva en segundos (ej. 0.05 para una forma compacta, Enter para {0.05}s por defecto): ")
            if duration_input == "":
                duration_val = 0.05
                break
            duration_val = float(duration_input)
            if duration_val <= 0:
                print("La duración debe ser un número positivo.")
            else:
                break
        except ValueError:
            print("Entrada inválida. Por favor, introduce un número.")

    # Pedir el radio del tubo (opcional, con valor por defecto)
    while True:
        try:
            radius_input = input(f"Introduce el radio del tubo para la curva (ej. 0.05, Enter para {0.05} por defecto): ")
            if radius_input == "":
                radius_val = 0.05
                break
            radius_val = float(radius_input)
            if radius_val <= 0:
                print("El radio debe ser un número positivo.")
            else:
                break
        except ValueError:
            print("Entrada inválida. Por favor, introduce un número.")

    # Pedir nombre del archivo (opcional, con valor por defecto)
    filename_input = input(f"Introduce el nombre del archivo STL de salida (ej. mi_curva.stl, Enter para 'lissajous_3d_custom_notes.stl' por defecto): ")
    output_filename = filename_input if filename_input else "lissajous_3d_custom_notes.stl"

    # Llamar a la función de creación de la figura
    created_mesh = create_lissajous_from_notes_3d(
        note1_name=notes[0],
        note2_name=notes[1],
        note3_name=notes[2],
        duration=duration_val,
        tube_radius=radius_val,
        output_filename=output_filename
    )

    if created_mesh:
        print(f"\n¡La figura 3D de Lissajous ha sido creada con éxito para las notas {notes[0]}, {notes[1]}, {notes[2]} y guardada como '{output_filename}'!")
    else:
        print("\nNo se pudo crear la figura. Por favor, revisa las notas ingresadas.")


# --- Punto de entrada del programa ---
if __name__ == "__main__":
    main_interactive_lissajous_creator()