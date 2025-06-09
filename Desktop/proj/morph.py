import os
import json
import cv2
import numpy as np
import trimesh
import mediapipe as mp
import scipy.sparse
from PIL import Image

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

    # Normalize to [0, 1]
    aligned_landmarks[:, 0] /= target_size[0]
    aligned_landmarks[:, 1] /= target_size[1]

    return aligned_landmarks  # shape: (468, 3)

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

def compute_new_uvs(mesh, old_vertices, new_vertices):
    if mesh.visual.uv is None or mesh.faces is None:
        raise ValueError("Mesh must have UVs and face definitions for texture warping.")

    uvs = mesh.visual.uv.copy()
    faces = mesh.faces

    new_uvs = np.zeros_like(uvs)

    for i, face in enumerate(faces):
        verts_old = old_vertices[face]
        verts_new = new_vertices[face]
        uvs_face = uvs[face]

        for j in range(3):
            vi = face[j]
            v = old_vertices[vi]

            # Compute barycentric coordinates
            A = verts_old[0]
            B = verts_old[1]
            C = verts_old[2]
            P = v

            v0 = B - A
            v1 = C - A
            v2 = P - A

            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)

            denom = d00 * d11 - d01 * d01
            if denom == 0:
                continue
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w

            # Interpolate new UV using new geometry
            new_uv = u * uvs_face[0] + v * uvs_face[1] + w * uvs_face[2]
            new_uvs[vi] = new_uv

    return new_uvs

def morph_model(base_mesh_path, output_obj_path, output_tex_path, ref_landmarks, input_landmarks, texture_image_path, strength=0.2, smoothing_iters=500, smoothing_lambda=0.06):
    mesh = trimesh.load(base_mesh_path, process=False)
    if mesh.visual.kind != 'texture':
        raise ValueError("Mesh must have texture coordinates and a texture image.")

    vertices = mesh.vertices.copy()
    original_vertices = vertices.copy()

    # === Morph landmarks ===
    for lm_idx, vert_idx in landmark_to_vertex_map.items():
        if lm_idx >= len(ref_landmarks) or vert_idx >= len(vertices):
            continue
        delta = input_landmarks[lm_idx] - ref_landmarks[lm_idx]
        vertices[vert_idx] += strength * delta

    # === Smooth mesh ===
    adjacency = mesh.vertex_neighbors
    vertices = laplacian_smooth(vertices, adjacency, iterations=smoothing_iters, lamb=smoothing_lambda)

    # === Update UVs ===
    new_uvs = compute_new_uvs(mesh, original_vertices, vertices)
    mesh.vertices = vertices
    mesh.visual.uv = new_uvs

    # === Export morphed model ===
    mesh.export(output_obj_path)
    print(f"Exported morphed model: {output_obj_path}")

    # Copy the same texture image
    if os.path.exists(texture_image_path):
        img = Image.open(texture_image_path)
        img.save(output_tex_path)
        print(f"Copied texture to: {output_tex_path}")
    else:
        print(f"Texture image not found: {texture_image_path}")

# === Paths ===
img_dir = "images"
reference_image_path = "photoa.png"
base_mesh_path = "textured.obj"
base_texture_path = "textured.png"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# === Get reference landmarks ===
reference_landmarks = get_aligned_landmarks(reference_image_path)

# === Process each image ===
for img_name in os.listdir(img_dir):
    input_path = os.path.join(img_dir, img_name)
    if not input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    input_landmarks = get_aligned_landmarks(input_path)
    basename = os.path.splitext(img_name)[0]

    output_obj = os.path.join(output_dir, f"morphed_{basename}.obj")
    output_tex = os.path.join(output_dir, f"morphed_{basename}.png")

    morph_model(
        base_mesh_path=base_mesh_path,
        output_obj_path=output_obj,
        output_tex_path=output_tex,
        ref_landmarks=reference_landmarks,
        input_landmarks=input_landmarks,
        texture_image_path=base_texture_path
    )
