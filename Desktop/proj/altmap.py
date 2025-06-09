import cv2
import mediapipe as mp
import numpy as np
import pyvista as pv
import json

def load_obj_vertices(filename):
    vertices = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
    return np.array(vertices)

def get_face_landmarks(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image {image_path} not found or cannot be read.")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        print("No face detected.")
        return None
    
    face_landmarks = results.multi_face_landmarks[0]

    height, width, _ = image.shape

    landmarks = []
    for lm in face_landmarks.landmark:
        x = lm.x * width
        y = lm.y * height
        z = lm.z * width
        landmarks.append([x, y, z])
    return np.array(landmarks)

def similarity_transform(src_pts, dst_pts):
    assert src_pts.shape == dst_pts.shape
    n = src_pts.shape[0]

    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)

    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    H = src_centered.T @ dst_centered / n

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    var_src = np.sum(src_centered ** 2) / n
    scale = np.sum(S) / var_src

    t = dst_mean - scale * R @ src_mean

    return scale, R, t

def apply_transform(points, scale, R, t):
    return scale * (points @ R.T) + t

def find_closest_vertex(landmark_coord, vertices):
    distances = np.linalg.norm(vertices - landmark_coord, axis=1)
    idx = np.argmin(distances)
    return idx, distances[idx]

def visualize(vertices, mapped_vertex_indices, manual_vertex_indices):
    plotter = pv.Plotter()
    
    cloud = pv.PolyData(vertices)
    plotter.add_points(cloud, color='lightgray', point_size=5, render_points_as_spheres=True, label='Model vertices')
    
    mapped_points = vertices[mapped_vertex_indices]
    mapped_cloud = pv.PolyData(mapped_points)
    plotter.add_points(mapped_cloud, color='blue', point_size=12, render_points_as_spheres=True, label='Mapped landmarks')

    manual_points = vertices[manual_vertex_indices]
    manual_cloud = pv.PolyData(manual_points)
    plotter.add_points(manual_cloud, color='red', point_size=20, render_points_as_spheres=True, label='Manual matched')

    plotter.add_legend()
    plotter.show(title="3D Model Vertices and Landmark Mappings")

def main():
    vertices = load_obj_vertices('textured.obj')
    image_path = "textured.png"  # Replace with your own image path
    media_pipe_landmarks = get_face_landmarks(image_path)
    if media_pipe_landmarks is None:
        print("No landmarks found, exiting.")
        return

    manual_mapping = {
        1: 5710,
        387: 2965,
        159: 480,
        12: 5264,
        10: 2843,
        152: 3818
    }

    model_points = np.array([vertices[v_idx] for v_idx in manual_mapping.values()])
    mp_points = np.array([media_pipe_landmarks[lm_id] for lm_id in manual_mapping.keys()])

    scale, R, t = similarity_transform(mp_points, model_points)
    print(f"Estimated scale: {scale}")
    print(f"Estimated rotation matrix:\n{R}")
    print(f"Estimated translation vector: {t}")

    transformed_landmarks = apply_transform(media_pipe_landmarks, scale, R, t)

    landmark_to_vertex_map = {}
    mapped_vertex_indices = []
    for i, lm_coord in enumerate(transformed_landmarks):
        idx, dist = find_closest_vertex(lm_coord, vertices)
        landmark_to_vertex_map[i] = idx
        mapped_vertex_indices.append(idx)
    landmark_to_vertex_map_clean = {int(k): int(v) for k, v in landmark_to_vertex_map.items()}

    with open("textured_landmark_to_vertex_map.json", "w") as f:
        json.dump(landmark_to_vertex_map_clean, f, indent=2)
    print("\nSaved landmark_to_vertex_map.json")

    visualize(vertices, mapped_vertex_indices, list(manual_mapping.values()))

if __name__ == '__main__':
    main()
