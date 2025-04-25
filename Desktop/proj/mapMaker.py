import cv2
import numpy as np
import mediapipe as mp
import trimesh
import json

# === CONFIGURATION ===
OBJ_PATH = "free_head.obj"
MAPPING_OUTPUT = "landmark_vertex_map.json"

LANDMARK_IDS = range(1,468)  # You can expand this list for more detail

# === LOAD MODEL IMAGE ===
# This image should be a front-facing photo/render of your base mesh
IMG_PATH = "photo.png"
model_img = cv2.imread(IMG_PATH)
if model_img is None:
    raise FileNotFoundError(f"Model image not found: {IMG_PATH}")
image_h, image_w = model_img.shape[:2]

# === DETECT LANDMARKS ON MODEL IMAGE ===
mp_face = mp.solutions.face_mesh
with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise RuntimeError("No face found in the model image!")
    face_landmarks = results.multi_face_landmarks[0]

landmark_2d_positions = {
    idx: np.array([
        face_landmarks.landmark[idx].x * image_w,
        face_landmarks.landmark[idx].y * image_h
    ], dtype=np.float32)
    for idx in LANDMARK_IDS
}
mp_face = mp.solutions.face_mesh
with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise RuntimeError("No face found!")
    face_landmarks = results.multi_face_landmarks[0]

image_h, image_w = model_img.shape[:2]

landmark_2d_positions = {
    idx: np.array([
        face_landmarks.landmark[idx].x * image_w,
        face_landmarks.landmark[idx].y * image_h
    ]) for idx in LANDMARK_IDS
}

# === LOAD MODEL ===
print("[*] Loading base OBJ...")
scene = trimesh.load(OBJ_PATH, process=False, force='scene')
if isinstance(scene, trimesh.Scene):
    mesh = trimesh.util.concatenate([g for g in scene.geometry.values()])
else:
    mesh = scene
vertices = mesh.vertices.copy()

# === PROJECT MESH VERTICES TO 2D ===
# Orthographic projection for front-facing mesh
proj = mesh.vertices.copy()
projected_2d = proj[:, :2].astype(np.float32)
# Normalize and align projected to image space
# Center data
mean_proj = projected_2d.mean(axis=0)
mean_land = np.array(list(landmark_2d_positions.values())).mean(axis=0)
# Scale factor matching std dev
std_proj = projected_2d.std(axis=0)
std_land = np.array(list(landmark_2d_positions.values())).std(axis=0)
scale = std_land / (std_proj + 1e-8)
projected_2d = (projected_2d - mean_proj) * scale + mean_land

# === FIND NEAREST MESH VERTEX FOR EACH LANDMARK ===
print("[*] Computing landmark-vertex map...")
landmark_vertex_map = {}
for lm_idx, lm_pos in landmark_2d_positions.items():
    dists = np.linalg.norm(projected_2d - lm_pos, axis=1)
    closest_vertex = int(np.argmin(dists))
    landmark_vertex_map[str(lm_idx)] = closest_vertex
    print(f"Landmark {lm_idx} → Vertex {closest_vertex}")

# === SAVE MAPPING ===
with open(MAPPING_OUTPUT, "w") as f:
    json.dump(landmark_vertex_map, f, indent=2)
print(f"[+] Mapping saved to {MAPPING_OUTPUT}")
print("[*] Finding closest vertices...")
landmark_vertex_map = {}
for lm_idx, lm_pos in landmark_2d_positions.items():
    dists = np.linalg.norm(projected_2d - lm_pos, axis=1)
    closest_vertex = np.argmin(dists)
    landmark_vertex_map[str(lm_idx)] = int(closest_vertex)
    print(f"Landmark {lm_idx} → Vertex {closest_vertex}")

# === SAVE MAPPING ===
with open(MAPPING_OUTPUT, "w") as f:
    json.dump(landmark_vertex_map, f, indent=2)
print(f"[+] Saved mapping to {MAPPING_OUTPUT}")
