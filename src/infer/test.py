import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque
import math

# =========================
# CONFIG
# =========================
MODEL_PATH = "models/svm_pose_classifier.pkl"
CONF_THRESHOLD = 0.25
HISTORY_SIZE = 25

# =========================
# LOAD MODEL
# =========================
model = joblib.load(MODEL_PATH)
print(f"[INFO] Pipeline loaded. Expecting 146 features")

# =========================
# MEDIAPIPE
# =========================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# =========================
# **COMPLETE 146-FEATURE EXTRACTION** (MATCHES TRAINING)
# =========================
def extract_landmark_vector(landmarks):
    """132 features: 33 landmarks × (x,y,z,visibility)"""
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(features, dtype=np.float32)

def calculate_angle(a, b, c):
    """2D angle calculation - EXACTLY like training"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def foot_direction_angle(heel, toe):
    """Exact foot direction from training"""
    v = np.array([toe.x - heel.x, toe.y - heel.y])
    return np.degrees(np.arctan2(v[1], v[0]))

def extract_joint_angles(lm):
    """9 angles - EXACT order from pose_feature_extractor.py"""
    angles = []
    # Left elbow, Right elbow
    angles.append(calculate_angle(lm[11], lm[13], lm[15]))
    angles.append(calculate_angle(lm[12], lm[14], lm[16]))
    # Left shoulder, Right shoulder
    angles.append(calculate_angle(lm[13], lm[11], lm[23]))
    angles.append(calculate_angle(lm[14], lm[12], lm[24]))
    # Left knee, Right knee
    angles.append(calculate_angle(lm[23], lm[25], lm[27]))
    angles.append(calculate_angle(lm[24], lm[26], lm[28]))
    # Left hip, Right hip
    angles.append(calculate_angle(lm[11], lm[23], lm[25]))
    angles.append(calculate_angle(lm[12], lm[24], lm[26]))
    # Spine angle
    angles.append(calculate_angle(lm[11], lm[23], lm[24]))
    return np.array(angles, dtype=np.float32)

def extract_foot_features(lm):
    """5 foot features - EXACT from training"""
    features = []
    # Foot direction angles
    features.append(foot_direction_angle(lm[29], lm[31]))  # Left foot
    features.append(foot_direction_angle(lm[30], lm[32]))  # Right foot
    # Ankle angles
    features.append(calculate_angle(lm[25], lm[27], lm[31]))  # Left ankle
    features.append(calculate_angle(lm[26], lm[28], lm[32]))  # Right ankle
    # Feet distance
    dist = math.dist([lm[27].x, lm[27].y], [lm[28].x, lm[28].y])
    features.append(dist)
    return np.array(features, dtype=np.float32)

def extract_features(landmarks):
    """🎯 146 TOTAL FEATURES: 132 landmarks + 9 angles + 5 feet"""
    landmarks_flat = extract_landmark_vector(landmarks)  # 132
    angles = extract_joint_angles(landmarks)             # 9
    feet = extract_foot_features(landmarks)              # 5
    
    features = np.concatenate([landmarks_flat, angles, feet])
    print(f"[DEBUG] Feature shape: {features.shape}")  # Should be (146,)
    assert features.shape[0] == 146, f"WRONG SHAPE: {features.shape[0]}"
    return features

# =========================
# STABLE PREDICTION
# =========================
history = deque(maxlen=HISTORY_SIZE)

def get_best_pose():
    if len(history) < 10:
        return "Hold steady...", 0.0
    
    # Best confidence in recent history
    best = max(history, key=lambda x: x[1])
    label = best[0].replace("_", " ").title()
    return (label[:20] + "..") if len(label) > 20 else label, best[1]

# =========================
# DISPLAY
# =========================
def draw_pose_info(frame, label, confidence):
    h, w = frame.shape[:2]
    
    # Background
    cv2.rectangle(frame, (40, 40), (w-40, 180), (20, 20, 40), -1)
    
    color = (0, 255, 0) if confidence > 0.3 else (0, 255, 255)
    
    # Main pose name
    cv2.putText(frame, f"🧘 POSE: {label}", (60, 85), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    
    # Confidence
    cv2.putText(frame, f"CONF: {confidence:.0%}", (60, 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    # Instructions
    cv2.putText(frame, "SIDE VIEW • HOLD 5s • BRIGHT LIGHT", (60, 165), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

# =========================
# MAIN LOOP
# =========================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("✅ FIXED: 146-FEATURE PIPELINE")
print("📐 Stand SIDEWAYS, hold poses 5+ seconds")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    label, confidence = "Detecting pose...", 0.0
    
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 🔥 CRITICAL: FULL 146 FEATURES
        feat = extract_features(results.pose_landmarks.landmark)
        
        # Pipeline handles scaling automatically
        pred = model.predict(feat.reshape(1, -1))[0]
        
        try:
            probs = model.predict_proba(feat.reshape(1, -1))[0]
            conf = probs.max()
        except:
            conf = 0.3
        
        history.append((pred, conf))
        label, confidence = get_best_pose()
        
        frame_count += 1
        if frame_count % 20 == 0:  # Print every 20 frames
            print(f"[{frame_count}] {label} | {confidence:.1%}")
    
    draw_pose_info(frame, label, confidence)
    cv2.imshow("🧘 YOGA POSE DETECTOR - 146 FEATURES FIXED", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
print("✅ COMPLETE!")
