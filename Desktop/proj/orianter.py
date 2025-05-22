# preprocess_model_reference.py
import cv2, mediapipe as mp, json

image = cv2.imread("photo3a.png")
h, w = image.shape[:2]

with mp.solutions.face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmark_data = results.multi_face_landmarks[0].landmark

ref_landmarks = {
    idx: [(lm.x - 0.5), -(lm.y - 0.5), -lm.z]
    for idx, lm in enumerate(landmark_data)
}
with open("reference_landmarks.json", "w") as f:
    json.dump(ref_landmarks, f)
