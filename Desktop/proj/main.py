import os
import json
import cv2
import numpy as np
import trimesh
import mediapipe as mp
import scipy.sparse
import matplotlib.pyplot as plt

# === Config ===
IGNORE_Z = False     # Set to True to ignore z-axis in morphing
Z_SCALE = 1.0        # If not ignoring, optionally scale z-axis delta
STRENGTH = 0.2       # Morphing strength
SMOOTH_ITERS = 500
SMOOTH_LAMBDA = 0.06
TARGET_SIZE = (256, 256)

# === Load landmark-to-vertex map ===
with open('landmark_to_vertex_map.json', 'r') as f:
    landmark_to_vertex_map = json.load(f)
    landmark_to_vertex_map = {int(k): v for k, v in landmark_to_vertex_map.items()}

# === Initialize MediaPipe Face Mesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

def get_aligned_landmarks(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        raise ValueError(f"No face detected in {image_path}")
    
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
    h, w = image.shape[:2]

    landmarks_px = landmarks.copy()
    landmarks_px[:, 0] *= w
    landmarks_px[:, 1] *= h

    min_x, max_x = landmarks_px[:, 0].min(), landmarks_px[:, 0].max()
    min_y, max_y = landmarks_px[:, 1].min(), landmarks_px[:, 1].max()
    margin = 0.1
    min_x = max(0, min_x - margin * (max_x - min_x))
    max_x = min(w, max_x + margin * (max_x - min_x))
    min_y = max(0, min_y - margin * (max_y - min_y))
    max_y = min(h, max_y + margin * (max_y - min_y))

    cropped = image[int(min_y):int(max_y), int(min_x):int(max_x)]
    cropped_resized = cv2.resize(cropped, target_size)

    scale_x = target_size[0] / (max_x - min_x)
    scale_y = target_size[1] / (max_y - min_y)
    aligned = landmarks_px.copy()
    aligned[:, 0] = (aligned[:, 0] - min_x) * scale_x
    aligned[:, 1] = (aligned[:, 1] - min_y) * scale_y

    aligned[:, 0] /= target_size[0]
    aligned[:, 1] /= target_size[1]

    return aligned

def laplacian_smooth(vertices, adjacency, iterations, lamb):
    n = len(vertices)
    V = vertices.copy()

    row_idx, col_idx, data = [], [], []
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            row_idx.append(i)
            col_idx.append(j)
            data.append(1)

    A = scipy.sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
    degrees = np.array(A.sum(axis=1)).flatten().reshape(-1, 1)

    for _ in range(iterations):
        neighbor_sum = A.dot(V)
        avg = np.divide(neighbor_sum, degrees, where=degrees != 0)
        V += lamb * (avg - V)
        lamb *= 0.99

    return V

def normalize_to_unit_space(vertices):
    center = vertices.mean(axis=0)
    max_dist = np.linalg.norm(vertices - center, axis=1).max()
    return (vertices - center) / max_dist

def compute_error_map(ref_landmarks, input_landmarks):
    errors = []
    for lm_idx in landmark_to_vertex_map:
        if lm_idx >= len(ref_landmarks): continue
        d = input_landmarks[lm_idx] - ref_landmarks[lm_idx]
        if IGNORE_Z:
            d[2] = 0
        else:
            d[2] *= Z_SCALE
        errors.append(np.linalg.norm(d))
    return np.array(errors)

def paint_vertex_colors(vertices, landmark_errors):
    colors = np.zeros((len(vertices), 3))
    norm_errors = (landmark_errors - landmark_errors.min()) / (landmark_errors.ptp() + 1e-8)

    for i, (lm_idx, vert_idx) in enumerate(landmark_to_vertex_map.items()):
        if i >= len(norm_errors): break
        heat = norm_errors[i]
        colors[vert_idx] = [heat, 0, 1.0 - heat]  # Red-blue map
    return (colors * 255).astype(np.uint8)

def morph_model(base_mesh_path, output_path, colored_output_path, ref_landmarks, input_landmarks):
    mesh = trimesh.load(base_mesh_path)
    vertices = mesh.vertices.copy()
    vertices = normalize_to_unit_space(vertices)

    # === Morph using deltas ===
    for lm_idx, vert_idx in landmark_to_vertex_map.items():
        if lm_idx >= len(ref_landmarks) or vert_idx >= len(vertices):
            continue
        delta = input_landmarks[lm_idx] - ref_landmarks[lm_idx]
        if IGNORE_Z:
            delta[2] = 0
        else:
            delta[2] *= Z_SCALE
        vertices[vert_idx] += STRENGTH * delta

    vertices = laplacian_smooth(vertices, mesh.vertex_neighbors, SMOOTH_ITERS, SMOOTH_LAMBDA)
    mesh.vertices = vertices

    # === Export base morphed model ===
    mesh.export(output_path)
    print(f"Exported morphed model to: {output_path}")

    # === Compute and color heatmap ===
    error_map = compute_error_map(ref_landmarks, input_landmarks)
    colors = paint_vertex_colors(vertices, error_map)
    mesh.visual.vertex_colors = colors
    mesh.export(colored_output_path)
    print(f"Exported colored model to: {colored_output_path}")

    return error_map.mean(), error_map

def main():
    img_dir = "images"
    os.makedirs("outputs", exist_ok=True)

    reference_landmarks = get_aligned_landmarks("photoa.png")
    image_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]

    all_errors = []
    error_table = {}

    for img_name in image_names:
        input_path = os.path.join(img_dir, img_name)
        input_landmarks = get_aligned_landmarks(input_path)

        output_path = os.path.join("outputs", f"morphed_{os.path.splitext(img_name)[0]}.obj")
        colored_path = os.path.join("outputs", f"colored_{os.path.splitext(img_name)[0]}.obj")

        avg_error, landmark_errors = morph_model("altmodel.obj", output_path, colored_path, reference_landmarks, input_landmarks)
        all_errors.append(landmark_errors)
        error_table[img_name] = avg_error
        print(f"Average landmark error for {img_name}: {avg_error:.4f}")

    # === Plot error distribution ===
    all_errors = np.array(all_errors)
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_errors.T, showfliers=False)
    plt.xlabel("Landmark Index (mapped)")
    plt.ylabel("Morphing Error (L2 distance)")
    plt.title("Landmark Error Distribution Across All Images")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/error_distribution.png")
    plt.show()
    print("Saved boxplot to outputs/error_distribution.png")

    # Optionally, save error summary
    with open("outputs/error_summary.json", "w") as f:
        json.dump(error_table, f, indent=2)
        print("Saved average error summary to outputs/error_summary.json")

if __name__ == "__main__":
    main()
