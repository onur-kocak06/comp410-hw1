import os
import cv2
import numpy as np
import mediapipe as mp
import trimesh
import json
import argparse
import scipy.sparse

# ========== DEFAULT CONFIG ==========
DEFAULT_OBJ_PATH = "model3.obj"
DEFAULT_LANDMARK_MAP = "landmark_vertex_map3.json"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_IMAGE_DIR = "images"

DEFAULT_XY_SCALE = 0.2
DEFAULT_Z_SCALE = 0.05
DEFAULT_INTENSITY = 0.2
DEFAULT_MAX_DELTA = 0.01
DEFAULT_SMOOTH_ITER = 500
DEFAULT_SMOOTH_LAMBDA = 0.001


# ========== UTILITY FUNCTIONS ==========

def build_adjacency(faces, num_vertices):
    """Build adjacency list for Laplacian smoothing."""
    adjacency = [set() for _ in range(num_vertices)]
    for a, b, c in faces:
        adjacency[a].update([b, c])
        adjacency[b].update([a, c])
        adjacency[c].update([a, b])
    return [list(neighbors) for neighbors in adjacency]

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
        # Sum of neighbor vertices: sparse matrix multiply (n,n) @ (n,3) -> (n,3)
        neighbor_sum = A.dot(V)

        # Compute average of neighbors, avoiding division by zero
        avg = np.divide(neighbor_sum, degrees, where=degrees != 0)

        # Update vertices
        V += lamb * (avg - V)

        # Optional decay of lamb
        lamb *= 0.99

    return V


def load_mesh(path):
    """Load a Trimesh mesh, handling both Scene and Trimesh cases."""
    scene_or_mesh = trimesh.load(path, process=False, force='scene')
    return trimesh.util.concatenate(scene_or_mesh.geometry.values()) if isinstance(scene_or_mesh, trimesh.Scene) else scene_or_mesh


# ========== MORPHING FUNCTION ==========


def morph_face(image, mesh, landmark_map, output_path,
               xy_scale=DEFAULT_XY_SCALE, z_scale=DEFAULT_Z_SCALE,
               intensity=DEFAULT_INTENSITY, max_delta=DEFAULT_MAX_DELTA,
               smooth_iter=DEFAULT_SMOOTH_ITER, smooth_lambda=DEFAULT_SMOOTH_LAMBDA,
               morph_iterations=50):
 
    # Step 1: Detect face landmarks
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        raise RuntimeError("No face detected.")

    landmark_data = results.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]

    # Step 2: Extract normalized landmark positions in 3D
    landmarks = {}
    for idx, vertex_idx in landmark_map.items():
        if idx >= len(landmark_data):
            continue
        lm = landmark_data[idx]
        landmarks[idx] = np.array([
            lm.x - 0.5,
            -(lm.y - 0.5),
            -lm.z  # MediaPipe z is inward (negative)
        ], dtype=np.float32)

    face_center = np.mean(list(landmarks.values()), axis=0)

    # Step 3: Prepare mesh data
    vertices = mesh.vertices.copy()
    faces = mesh.faces
    num_vertices = len(vertices)
    bbox_scale = mesh.bounds[1] - mesh.bounds[0]

    # Step 4: Compute full displacements for each mapped vertex
    total_deltas = {}
    for lm_idx, v_idx in landmark_map.items():
        if lm_idx not in landmarks or v_idx >= num_vertices:
            continue

        offset = landmarks[lm_idx] - face_center
        scaled = offset * bbox_scale * np.array([xy_scale, xy_scale, z_scale]) * intensity
        delta = np.clip(scaled, -max_delta, max_delta)
        total_deltas[v_idx] = delta

        # Print debug info for selected landmarks only once
        label = {
            1: "Nose tip",
            10: "Forehead",
            12: "Mouth center",
            468: "Left eye",
            473: "Right eye"
        }.get(lm_idx, None)
        if label:
            print(f"[INFO] Landmark {lm_idx} ({label}) total displacement for vertex {v_idx} is {delta.round(6)}")

    vertices = mesh.vertices.copy()
    faces = mesh.faces
    num_vertices = len(vertices)
    bbox_scale = mesh.bounds[1] - mesh.bounds[0]
    adjacency = build_adjacency(faces, num_vertices)

    # Compute total deltas per vertex (same as before)
    total_deltas = {}
    for lm_idx, v_idx in landmark_map.items():
        if lm_idx not in landmarks or v_idx >= num_vertices:
            continue
        offset = landmarks[lm_idx] - face_center
        scaled = offset * bbox_scale * np.array([xy_scale, xy_scale, z_scale]) * intensity
        delta = np.clip(scaled, -max_delta, max_delta)
        total_deltas[v_idx] = delta

    fraction = 1.0 / morph_iterations
    print(f"[INFO] Morphing with {morph_iterations} iterations, smoothing {smooth_iter} per iteration...")

    for i in range(morph_iterations):
        # Morph step
        for v_idx, delta in total_deltas.items():
            vertices[v_idx] += fraction * delta
        
        # Smoothing step
        vertices = laplacian_smooth(vertices, adjacency, smooth_iter, smooth_lambda)

    # Optional: final smoothing (if desired)
    # vertices = laplacian_smooth(vertices, adjacency, smooth_iter, smooth_lambda)

    # Export final mesh
    morphed = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    morphed.export(output_path)
    print(f"[+] Morphed mesh saved to {output_path}")

# ========== MAIN EXECUTION ==========

def main():
    parser = argparse.ArgumentParser(description="Morph a 3D face mesh based on MediaPipe landmarks.")
    parser.add_argument("-i", "--image", help="Input image file")
    parser.add_argument("-d", "--directory", help="Process all images in folder")
    parser.add_argument("--model", default=DEFAULT_OBJ_PATH, help="Path to base .obj model")
    parser.add_argument("--map", default=DEFAULT_LANDMARK_MAP, help="Path to landmark→vertex map JSON")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output folder")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load landmark map
    with open(args.map, 'r') as f:
        landmark_map = {int(k): v for k, v in json.load(f).items()}

    # Load mesh once
    mesh = load_mesh(args.model)

    def process_image_file(path):
        image = cv2.imread(path)
        if image is None:
            print(f"[-] Failed to read image: {path}")
            return
        base_name = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join(args.output, f"morphed_{base_name}.obj")
        try:
            morph_face(image, mesh, landmark_map, output_path)
        except Exception as e:
            print(f"[!] Error processing {path}: {e}")

    if args.directory:
        for file in os.listdir(args.directory):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                process_image_file(os.path.join(args.directory, file))
    elif args.image:
        process_image_file(args.image)
    else:
        # Webcam capture fallback
        cam = cv2.VideoCapture(0)
        print("Press SPACE to capture frame...")
        while True:
            ret, frame = cam.read()
            if not ret:
                print("[-] Failed to capture frame.")
                break
            cv2.imshow("Capture", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
        cam.release()
        cv2.destroyAllWindows()
        output_path = os.path.join(args.output, "morphed_capture.obj")
        morph_face(frame, mesh, landmark_map, output_path)


if __name__ == "__main__":
    main()
