import cv2
import numpy as np
import mediapipe as mp
import trimesh
import json

# === CONFIGURATION ===
OBJ_PATH = "model3.obj"  # Path to your 3D model
IMG_PATH = "photo3b.png"  # Path to your input image with face
MAPPING_OUTPUT = "landmark_vertex_map3.json"  # Output mapping file
COLORED_MESH_PATH = "highlighted_landmarks.ply"  # Path to export colored mesh

# Fixed landmark-to-vertex correspondences (manually verified)
FIXED_LANDMARKS = {
    1: 2967,     # Nose tip
    473: 7999,   # Right eye
    468: 1,      # Left eye
    12: 2831,    # Mouth center
    10: 3866,    # Forehead center
    152: 16039   # Chin
}

# === LOAD IMAGE ===
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMG_PATH}")
h, w = img.shape[:2]

# === DETECT LANDMARKS ===
print("[*] Detecting facial landmarks...")
mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise RuntimeError("No face detected.")
    face_landmarks = results.multi_face_landmarks[0].landmark

# === PREPARE LANDMARKS IN IMAGE SPACE ===
landmarks_2d = {
    i: np.array([face_landmarks[i].x * w, face_landmarks[i].y * h], dtype=np.float32)
    for i in FIXED_LANDMARKS.keys()
}

# === LOAD MESH ===
print("[*] Loading 3D mesh...")
scene = trimesh.load(OBJ_PATH, process=False, force='scene')
mesh = trimesh.util.concatenate(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else scene
vertices = mesh.vertices.copy()

# === SOLVE CAMERA POSE ===
object_points = np.array([vertices[FIXED_LANDMARKS[i]] for i in FIXED_LANDMARKS], dtype=np.float32)
image_points = np.array([landmarks_2d[i] for i in FIXED_LANDMARKS], dtype=np.float32)

focal_length = w
camera_matrix = np.array([
    [focal_length, 0, w / 2],
    [0, focal_length, h / 2],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
if not success:
    raise RuntimeError("solvePnP failed to find a valid pose.")

# === PROJECT MESH TO IMAGE ===
projected_2d, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
projected_2d = projected_2d.reshape(-1, 2)

# === FILTER TO FRONTAL VERTICES ONLY ===
print("[*] Filtering frontal face region...")
camera_direction = np.array([0, 1, 1])
mesh_center = np.mean(vertices, axis=0)
vertex_dirs = vertices - mesh_center
vertex_dirs = vertex_dirs / np.linalg.norm(vertex_dirs, axis=1, keepdims=True)
frontal_mask = np.dot(vertex_dirs, camera_direction) > 0.3  # Keep ~face front

frontal_indices = np.where(frontal_mask)[0]

# === DETECT ALL LANDMARKS FOR MAPPING ===
all_landmarks_2d = {
    i: np.array([face_landmarks[i].x * w, face_landmarks[i].y * h], dtype=np.float32)
    for i in range(468)
}

# === MAP LANDMARKS TO NEAREST VISIBLE VERTEX ===
print("[*] Mapping MediaPipe landmarks to mesh vertices...")
landmark_vertex_map = {str(k): v for k, v in FIXED_LANDMARKS.items()}
used_vertices = set(FIXED_LANDMARKS.values())

for lid, pos in all_landmarks_2d.items():
    if str(lid) in landmark_vertex_map:
        continue  # Skip fixed ones

    dists = np.linalg.norm(projected_2d[frontal_indices] - pos, axis=1)
    sorted_idxs = np.argsort(dists)

    for i in sorted_idxs:
        candidate_vidx = frontal_indices[i]
        if candidate_vidx not in used_vertices:
            landmark_vertex_map[str(lid)] = int(candidate_vidx)
            used_vertices.add(candidate_vidx)
            break

# === SAVE MAPPING ===
with open(MAPPING_OUTPUT, "w") as f:
    json.dump(landmark_vertex_map, f, indent=2)
print(f"[+] Saved landmark-vertex mapping to {MAPPING_OUTPUT}")

# === HIGHLIGHT MAPPED VERTICES ===
colors = np.ones((len(vertices), 3)) * 0.8  # light gray
for vidx in landmark_vertex_map.values():
    colors[vidx] = np.array([1.0, 0.0, 0.0])  # red

colored_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces, vertex_colors=colors)
colored_mesh.export(COLORED_MESH_PATH)
print(f"[+] Exported colored mesh to {COLORED_MESH_PATH}")

# === VISUALIZE OVERLAY (OPTIONAL) ===
vis_img = img.copy()
for lid, vidx in landmark_vertex_map.items():
    pt = projected_2d[int(vidx)].astype(int)
    cv2.circle(vis_img, tuple(pt), 2, (0, 255, 0), -1)

cv2.imshow("Landmark to Vertex Overlay", vis_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
