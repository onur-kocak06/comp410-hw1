import cv2
import numpy as np
import mediapipe as mp
import trimesh
import json
import os
import argparse
from scipy.interpolate import Rbf

# === CONFIGURATION ===
OBJ_PATH = "free_head.obj"        # Your base mesh
OUTPUT_OBJ_PATH = "morphed.obj"   # Output path
LANDMARK_VERTEX_MAP_FILE = 'landmark_vertex_map.json'  # Landmark to vertex map JSON file

# === LOAD LANDMARK TO VERTEX MAP ===
def load_landmark_vertex_map():
    if os.path.exists(LANDMARK_VERTEX_MAP_FILE):
        with open(LANDMARK_VERTEX_MAP_FILE, 'r') as f:
            return json.load(f)
    else:
        print(f"[!] {LANDMARK_VERTEX_MAP_FILE} not found!")
        return {}

LANDMARK_VERTEX_MAP_raw = load_landmark_vertex_map()
LANDMARK_VERTEX_MAP = {int(k): v for k, v in LANDMARK_VERTEX_MAP_raw.items()}
# === SMOOTHING PARAMETERS ===
XY_SCALE   = 0.5    # Try values from 0.1 (very subtle) up to 2.0 (very strong)
Z_SCALE    = 0.2    # Controls how much the landmark’s z-value moves the mesh in depth
SMOOTH_RADIUS = 2   # As before: how wide the smoothing “brush” is

# === FUNCTION TO SMOOTH VERTICES ===
def smooth_vertices(vertices, radius=SMOOTH_RADIUS):
    smoothed_vertices = vertices.copy()
    for i in range(len(vertices)):
        distances = np.linalg.norm(vertices - vertices[i], axis=1)
        nearby_vertices = np.where(distances < radius)[0]
        if len(nearby_vertices) > 0:
            smoothed_vertices[i] = np.mean(vertices[nearby_vertices], axis=0)
    return smoothed_vertices

# === CAMERA PHOTO CAPTURE ===
def capture_photo():
    cap = cv2.VideoCapture(0)
    print("[*] Press SPACE to capture photo.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture Face", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            photo = frame.copy()
            break
    cap.release()
    cv2.destroyAllWindows()
    return photo

# === MORPHING FOR A SINGLE IMAGE ===
def morph_single_image(photo):
    # === MEDIAPIPE LANDMARKS ===
    
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            raise RuntimeError("No face found in the captured image!")
        face_landmarks = results.multi_face_landmarks[0]

    image_h, image_w, _ = photo.shape
    landmark_positions = {
        idx: np.array([
            face_landmarks.landmark[idx].x * image_w,
            face_landmarks.landmark[idx].y * image_h,
            face_landmarks.landmark[idx].z * 1000
        ]) for idx in LANDMARK_VERTEX_MAP.keys()
    }

    # === LOAD MESH ===
    print("[*] Loading base OBJ...")
    scene_or_mesh = trimesh.load(OBJ_PATH, process=False, force='scene')
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([geometry for geometry in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh

    vertices = mesh.vertices.copy()
    faces = mesh.faces

    # === AUTOMATICALLY MAP LANDMARKS TO VERTICES ===
    # scale X/Y and Z separately
    
    print("[*] Mapping landmarks to mesh vertices...")
    for lm_idx, v_idx in LANDMARK_VERTEX_MAP.items():
        if lm_idx not in landmark_positions:
            continue
        cam_pos = landmark_positions[lm_idx]
        base_pos = vertices[v_idx]
        dx = ((cam_pos[0] - image_w/2) / image_w) * XY_SCALE
        dy = (-(cam_pos[1] - image_h/2) / image_h) * XY_SCALE
        dz = (cam_pos[2] * Z_SCALE * 0.001)
        delta = np.array([dx, dy, dz], dtype=np.float32)
        vertices[v_idx] += delta

    # === SMOOTHING IF NEEDED ===
    print(f"[*] Smoothing mesh with radius {SMOOTH_RADIUS}...")
    vertices = smooth_vertices(vertices, SMOOTH_RADIUS)

    # === EXPORT MORPHED MESH ===
    print(f"[*] Writing morphed mesh to {OUTPUT_OBJ_PATH}")
    with open(OUTPUT_OBJ_PATH, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print("[+] Done. Check morphed.obj in your folder.")

# === PROCESS MULTIPLE IMAGES ===
def process_images_in_folder(image_folder="images"):
    images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        print("[!] No images found in the folder.")
        return

    for image_file in images:
        image_path = os.path.join(image_folder, image_file)
        print(f"[*] Processing {image_file}...")
        photo = cv2.imread(image_path)
        morph_single_image(photo)

# === MAIN ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Morph 3D model based on face landmarks.")
    parser.add_argument('-d', '--directory', action='store_true', help="Process all images in the images folder.")
    args = parser.parse_args()

    if args.directory:
        process_images_in_folder()
    else:
        photo = capture_photo()
        morph_single_image(photo)
