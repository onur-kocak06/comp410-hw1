import cv2
import numpy as np
import mediapipe as mp
import trimesh
import json
from scipy.spatial import cKDTree
import os

# === Config ===
IMG_PATH = 'photoa.png'
OBJ_PATH = 'altmodel.obj'
MAP_PATH = 'landmark_vertex_map.json'
PLY_PATH = 'highlighted_landmarks.ply'

CONFIDENCE_THRESHOLD = 0.95
CONFIDENCE_SCALE = 0.05  # Tune this for confidence falloff curve

# Radius (in mesh units) to restrict search for known landmarks
NEIGHBOR_RADIUS = 0.02

# === Step 1: Load image and detect face landmarks ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

image = cv2.imread(IMG_PATH)
if image is None:
    raise FileNotFoundError(f"Image '{IMG_PATH}' not found.")
h, w = image.shape[:2]

results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
if not results.multi_face_landmarks:
    raise RuntimeError("No face detected!")

landmarks = results.multi_face_landmarks[0].landmark
landmarks_2d = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

# === Step 2: Load 3D model and vertices ===
mesh = trimesh.load_mesh(OBJ_PATH, process=False)
vertices_3d = np.array(mesh.vertices, dtype=np.float32)

# === Known landmark to vertex correspondences (indices) ===
known_lmk_to_vert = {
    1: 5710, 
    387: 2965, 
    159: 480,  
    12: 5264,  
    10: 2843,   
    152: 3818
}

# === Step 3: Prepare data for solvePnP ===
object_points = np.array([vertices_3d[idx] for idx in known_lmk_to_vert.values()], dtype=np.float32)
image_points = np.array([landmarks_2d[idx] for idx in known_lmk_to_vert.keys()], dtype=np.float32)

# === Step 4: Estimate camera intrinsics (approximate) ===
focal_length = w  # Approximate focal length in pixels; can be adjusted
center = (w / 2, h / 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.zeros((4, 1))  # No distortion assumed

# === Step 5: Solve PnP for pose estimation ===
success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
if not success:
    raise RuntimeError("Initial pose estimation failed!")

# Optional: Refine pose using Levenberg-Marquardt optimization if available
try:
    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec)
except AttributeError:
    # Older OpenCV versions may not have solvePnPRefineLM
    pass

# === Step 6: Project all vertices to 2D image points ===
projected_2d, _ = cv2.projectPoints(vertices_3d, rvec, tvec, camera_matrix, dist_coeffs)
projected_2d = projected_2d.squeeze()

# === Step 7: Build KD-tree for fast 3D neighbor search ===
kdtree = cKDTree(vertices_3d)

# === Step 8: Map each landmark to closest vertex ===
landmark_to_vertex = {}
for i, lm_2d in enumerate(landmarks_2d):
    if i in known_lmk_to_vert:
        # Restrict candidates to neighbors in 3D around known vertex
        anchor_vertex_idx = known_lmk_to_vert[i]
        anchor_pos = vertices_3d[anchor_vertex_idx]

        candidate_indices = kdtree.query_ball_point(anchor_pos, NEIGHBOR_RADIUS)
        if not candidate_indices:
            # Fallback to global search if no neighbors found
            candidate_indices = np.arange(len(vertices_3d))
        candidate_proj = projected_2d[candidate_indices]

        # Compute 2D distances to projected vertices
        dists = np.linalg.norm(candidate_proj - lm_2d, axis=1)
        min_idx_local = np.argmin(dists)
        min_dist = dists[min_idx_local]
        closest_idx = candidate_indices[min_idx_local]
    else:
        # For unknown landmarks, do global search (less precise)
        dists = np.linalg.norm(projected_2d - lm_2d, axis=1)
        closest_idx = np.argmin(dists)
        min_dist = dists[closest_idx]

    # Confidence scoring based on 2D reprojection distance
    confidence = float(np.exp(-min_dist * CONFIDENCE_SCALE))

    if confidence >= CONFIDENCE_THRESHOLD:
        landmark_to_vertex[str(i + 1)] = {
            "vertex": int(closest_idx),
            "confidence": confidence
        }

print(f"[+] Mapped {len(landmark_to_vertex)} high-confidence landmarks")

# === Step 9: Save mapping to JSON ===
with open(MAP_PATH, 'w') as f:
    json.dump(landmark_to_vertex, f, indent=2)
print(f"[+] Saved landmark-vertex map to '{MAP_PATH}'")

# === Step 10: Export PLY with highlighted landmarks ===
scene = trimesh.Scene()
scene.add_geometry(mesh)

for data in landmark_to_vertex.values():
    center = vertices_3d[data["vertex"]]
    sphere = trimesh.creation.icosphere(radius=0.005, subdivisions=2)
    sphere.apply_translation(center)

    # Color gradient: green (high confidence) to red (low confidence)
    c = data["confidence"]
    color = [int(255 * (1 - c)), int(255 * c), 0, 255]  # RGBA
    sphere.visual.vertex_colors = color

    scene.add_geometry(sphere)

scene.export(PLY_PATH)
print(f"[+] Exported colored mesh to '{PLY_PATH}'")

# === Optional Step 11: Visualize projected points and landmarks on the image ===
def visualize_overlay(img, landmarks_2d, projected_2d, landmark_to_vertex):
    vis_img = img.copy()
    # Draw detected landmarks (blue)
    for pt in landmarks_2d:
        cv2.circle(vis_img, tuple(pt.astype(int)), 2, (255, 0, 0), -1)

    # Draw projected matched vertices (green)
    for v in landmark_to_vertex.values():
        proj_pt = projected_2d[v["vertex"]]
        cv2.circle(vis_img, tuple(proj_pt.astype(int)), 3, (0, 255, 0), -1)

    # Draw lines between matched pairs
    for idx_str, v in landmark_to_vertex.items():
        lmk_idx = int(idx_str) - 1
        pt1 = landmarks_2d[lmk_idx].astype(int)
        pt2 = projected_2d[v["vertex"]].astype(int)
        cv2.line(vis_img, tuple(pt1), tuple(pt2), (0, 255, 255), 1)

    cv2.imshow("Landmark to Vertex Mapping", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Uncomment to enable visualization
# visualize_overlay(image, landmarks_2d, projected_2d, landmark_to_vertex)
