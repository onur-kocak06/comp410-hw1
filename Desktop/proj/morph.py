import os
import json
import cv2
import numpy as np
import trimesh
import mediapipe as mp
import scipy.sparse

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

    # Convert normalized to pixel coordinates
    landmarks_px = np.copy(landmarks)
    landmarks_px[:, 0] *= w
    landmarks_px[:, 1] *= h

    # Calculate bounding box
    min_x = np.min(landmarks_px[:, 0])
    max_x = np.max(landmarks_px[:, 0])
    min_y = np.min(landmarks_px[:, 1])
    max_y = np.max(landmarks_px[:, 1])

    # Slightly expand the box for context
    margin = 0.1
    min_x = max(0, min_x - margin * (max_x - min_x))
    max_x = min(w, max_x + margin * (max_x - min_x))
    min_y = max(0, min_y - margin * (max_y - min_y))
    max_y = min(h, max_y + margin * (max_y - min_y))

    # Crop and resize the face
    cropped = image[int(min_y):int(max_y), int(min_x):int(max_x)]
    cropped_resized = cv2.resize(cropped, target_size)

    # Update landmark coordinates relative to the cropped+resized face
    box_w = max_x - min_x
    box_h = max_y - min_y
    scale_x = target_size[0] / box_w
    scale_y = target_size[1] / box_h

    aligned_landmarks = np.copy(landmarks_px)
    aligned_landmarks[:, 0] = (aligned_landmarks[:, 0] - min_x) * scale_x
    aligned_landmarks[:, 1] = (aligned_landmarks[:, 1] - min_y) * scale_y

    # Re-normalize to 0-1 for compatibility if needed
    aligned_landmarks[:, 0] /= target_size[0]
    aligned_landmarks[:, 1] /= target_size[1]

    return aligned_landmarks  # still (468, 3)

def laplacian_smooth(vertices, adjacency, iterations, lamb):
    n = len(vertices)
    V = vertices.copy()

    # Build sparse adjacency matrix
    row_idx = []
    col_idx = []
    data = []
    for i, neighbors in enumerate(adjacency):
        for j in neighbors:
            row_idx.append(i)
            col_idx.append(j)
            data.append(1)

    A = scipy.sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))

    # Compute degree (number of neighbors for each vertex)
    degrees = np.array(A.sum(axis=1)).flatten().reshape(-1, 1)  # shape (n,1)

    for _ in range(iterations):
        neighbor_sum = A.dot(V)
        avg = np.divide(neighbor_sum, degrees, where=degrees != 0)
        V += lamb * (avg - V)
        lamb *= 0.99  # optional decay

    return V

def morph_model(base_mesh_path, output_path, ref_landmarks, input_landmarks, strength=0.2, smoothing_iters=500, smoothing_lambda=0.06):
    mesh = trimesh.load(base_mesh_path)
    vertices = mesh.vertices.copy()

    # === Morph using landmarks ===
    for lm_idx, vert_idx in landmark_to_vertex_map.items():
        if lm_idx >= len(ref_landmarks) or vert_idx >= len(vertices):
            continue
        delta = input_landmarks[lm_idx] - ref_landmarks[lm_idx]
   
        vertices[vert_idx] += strength * delta

    # === Smoothing ===
    adjacency = mesh.vertex_neighbors
    vertices = laplacian_smooth(vertices, adjacency, iterations=smoothing_iters, lamb=smoothing_lambda)

    # === Save morphed model ===
    mesh.vertices = vertices
    mesh.export(output_path)
    print(f"Exported morphed model to: {output_path}")

# === Paths ===
img_dir = "images"
reference_image_path = os.path.join("photoa.png")
input_image_names = [f for f in os.listdir(img_dir)]

# === Get reference landmarks ===
reference_landmarks = get_aligned_landmarks(reference_image_path)

# === Morph for each input image ===
for img_name in input_image_names:
    input_path = os.path.join(img_dir, img_name)
    output_path = os.path.join("outputs", f"morphed_{os.path.splitext(img_name)[0]}.obj")

    input_landmarks = get_aligned_landmarks(input_path)
    morph_model("altmodel.obj", output_path, reference_landmarks, input_landmarks)
