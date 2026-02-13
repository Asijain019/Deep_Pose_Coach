import cv2
import mediapipe as mp
import numpy as np
import math

#Real-time performance confirmed now convert a human pose → numbers.

#Your final feature vector (146 dimensions):Raw landmarks (132) 33 landmarks × (x, y, z, visibility),Joint angles (9),Foot & toe features (5)

#Feature engineering is now frozen

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

POSE_HOLD_FRAMES = 60


def extract_landmark_vector(landmarks):
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(features, dtype=np.float32)


def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def foot_direction_angle(heel, toe):
    v = np.array([toe.x - heel.x, toe.y - heel.y])
    return np.degrees(np.arctan2(v[1], v[0]))


def extract_joint_angles(lm):
    angles = []
    angles.append(calculate_angle(lm[11], lm[13], lm[15]))  # L elbow
    angles.append(calculate_angle(lm[12], lm[14], lm[16]))  # R elbow
    angles.append(calculate_angle(lm[13], lm[11], lm[23]))  # L shoulder
    angles.append(calculate_angle(lm[14], lm[12], lm[24]))  # R shoulder
    angles.append(calculate_angle(lm[23], lm[25], lm[27]))  # L knee
    angles.append(calculate_angle(lm[24], lm[26], lm[28]))  # R knee
    angles.append(calculate_angle(lm[11], lm[23], lm[25]))  # L hip
    angles.append(calculate_angle(lm[12], lm[24], lm[26]))  # R hip
    angles.append(calculate_angle(lm[11], lm[23], lm[24]))  # Spine
    return np.array(angles, dtype=np.float32)


def extract_foot_features(lm):
    features = []

    # Foot direction angles
    features.append(foot_direction_angle(lm[29], lm[31]))  # Left foot
    features.append(foot_direction_angle(lm[30], lm[32]))  # Right foot

    # Ankle bend angles
    features.append(calculate_angle(lm[25], lm[27], lm[31]))  # Left ankle
    features.append(calculate_angle(lm[26], lm[28], lm[32]))  # Right ankle

    # Feet spread distance
    dist = math.dist(
        [lm[27].x, lm[27].y],
        [lm[28].x, lm[28].y]
    )
    features.append(dist)

    return np.array(features, dtype=np.float32)


cap = cv2.VideoCapture(0)
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

buffer = []
count = 0

print("Hold a pose...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = res.pose_landmarks.landmark

        base = extract_landmark_vector(lm)
        angles = extract_joint_angles(lm)
        feet = extract_foot_features(lm)

        full = np.concatenate([base, angles, feet])
        buffer.append(full)
        count += 1

        cv2.putText(frame, f"Holding pose: {count}/{POSE_HOLD_FRAMES}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if count >= POSE_HOLD_FRAMES:
            break

    cv2.imshow("Pose Capture (FULL)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

if buffer:
    avg = np.mean(buffer, axis=0)
    print("FINAL feature vector shape:", avg.shape)
    print("Foot features:", avg[-5:])
else:
    print("No pose detected.")
