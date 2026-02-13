import cv2
import numpy as np
import mediapipe as mp
import joblib
from collections import deque, Counter

MODEL_PATH = "models/svm_pose_classifier.pkl"

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

model = joblib.load(MODEL_PATH)
scaler = model.named_steps['scaler']
svm = model.named_steps['svm']

def extract_features(lm):
    """Exact 146 features - PROVEN working"""
    lm_copy = list(lm)
    
    # Normalize
    xs = np.array([p.x for p in lm_copy])
    ys = np.array([p.y for p in lm_copy])
    cx, cy = xs.mean(), ys.mean()
    for p in lm_copy:
        p.x -= cx
        p.y -= cy
    
    # 132 landmarks - FIXED explicit loop
    landmarks = []
    for p in lm_copy:
        landmarks.extend([p.x, p.y, p.z, p.visibility])
    landmarks = np.array(landmarks, dtype=np.float32)
    
    # 9 angles
    def calc_angle(a_idx, b_idx, c_idx):
        try:
            a = lm_copy[a_idx]
            b = lm_copy[b_idx] 
            c = lm_copy[c_idx]
            a_pos = np.array([a.x, a.y])
            b_pos = np.array([b.x, b.y])
            c_pos = np.array([c.x, c.y])
            ba = a_pos - b_pos
            bc = c_pos - b_pos
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
            return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        except:
            return 90.0  # Safe default
    
    angles = np.array([
        calc_angle(11, 13, 15), calc_angle(12, 14, 16),
        calc_angle(13, 11, 23), calc_angle(14, 12, 24),
        calc_angle(23, 25, 27), calc_angle(24, 26, 28),
        calc_angle(11, 23, 25), calc_angle(12, 24, 26),
        calc_angle(11, 23, 24)
    ], dtype=np.float32)
    
    feet = np.array([0.0, 0.0, 90.0, 90.0, 0.1], dtype=np.float32)
    
    return np.concatenate([landmarks, angles, feet])

# 🔥 CONTINUOUS DETECTION SYSTEM
pose_buffer = deque(maxlen=6)  # Fast response
conf_buffer = deque(maxlen=6)
STABLE_THRESHOLD = 0.08
CHANGE_THRESHOLD = 0.25  # Higher confidence needed to change pose

current_pose = "STAND CLEAR"
current_conf = 0.0

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("🧘 CONTINUOUS POSE DETECTOR")
print("📐 SIDE VIEW - Switch poses freely!")
print("✅ Tadasana → Tree → Chair → etc")

frame_count = 0
pose_changes = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    frame_count += 1
    
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Predict
        feat_raw = extract_features(results.pose_landmarks.landmark).reshape(1, -1)
        feat_scaled = scaler.transform(feat_raw)
        pred = svm.predict(feat_scaled)[0]
        conf = svm.predict_proba(feat_scaled)[0].max()
        
        # 🔥 CONTINUOUS LOGIC
        if conf > STABLE_THRESHOLD:
            pose_buffer.append(pred)
            conf_buffer.append(conf)
            
            # Check for stable pose
            if len(pose_buffer) >= 4:
                common, count = Counter(pose_buffer).most_common(1)[0]
                ratio = count / len(pose_buffer)
                avg_conf = np.mean([c for p,c in zip(pose_buffer, conf_buffer) if p == common])
                
                # CHANGE POSE LOGIC
                if ratio >= 0.7 and avg_conf > CHANGE_THRESHOLD:
                    new_pose = common.replace("_", " ").title()
                    
                    # Only change if confident OR different pose
                    if new_pose != current_pose and avg_conf > 0.12:
                        pose_changes += 1
                        print(f"🔄 CHANGED: {current_pose} → {new_pose} ({avg_conf:.1%})")
                        current_pose = new_pose
                        current_conf = avg_conf
                        pose_buffer.clear()  # Reset for next pose
                        conf_buffer.clear()
                    elif new_pose == current_pose:
                        current_conf = avg_conf
                        
        # Debug top predictions every 30 frames
        if frame_count % 30 == 0:
            probs = svm.predict_proba(feat_scaled)[0]
            top_idx = np.argsort(probs)[-3:][::-1]
            top_poses = [model.classes_[i] for i in top_idx]
            top_confs = probs[top_idx]
            print(f"[{frame_count}] TOP: {top_poses[0]}({top_confs[0]:.1%}) | Current: {current_pose}")
    
    # 🔥 PROFESSIONAL CONTINUOUS UI
    h, w = frame.shape[:2]
    
    # Dynamic background
    if current_conf > 0.25:
        bg_color = (0, 50, 0)
        color = (0, 255, 0)
        status = "✅ STABLE"
    elif current_conf > 0.12:
        bg_color = (0, 50, 50)
        color = (0, 255, 255)
        status = "🟡 STABILIZING"
    else:
        bg_color = (50, 50, 50)
        color = (0, 165, 255)
        status = "🔄 DETECTING"
    
    cv2.rectangle(frame, (10, 10), (w-10, 120), bg_color, -1)
    cv2.rectangle(frame, (10, 10), (w-10, 120), (255, 255, 255), 2)
    
    cv2.putText(frame, f"🧘 {current_pose}", (25, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
    cv2.putText(frame, f"CONF: {current_conf:.0%}", (25, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, status, (25, 105), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Instructions
    cv2.putText(frame, f"Changes: {pose_changes} | SIDE VIEW", (25, h-60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "TADASANA → TREE → CHAIR → QUIT=Q", (25, h-35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 200), 1)
    
    cv2.imshow("🧘 CONTINUOUS POSE DETECTOR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
print(f"\n🎉 Session complete!")
print(f"Total pose changes: {pose_changes}")
print(f"Final pose: {current_pose} ({current_conf:.1%})")
