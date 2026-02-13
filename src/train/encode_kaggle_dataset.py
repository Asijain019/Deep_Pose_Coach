import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from pathlib import Path
import math

mp_pose = mp.solutions.pose

# =========================
# NORMALIZATION (CRITICAL)
# =========================
def normalize_landmarks(lm):
    """Centroid normalization - center pose at origin"""
    xs = np.array([p.x for p in lm])
    ys = np.array([p.y for p in lm])
    cx, cy = xs.mean(), ys.mean()
    for p in lm:
        p.x -= cx
        p.y -= cy

# =========================
# FEATURE FUNCTIONS (EXACT MATCH)
# =========================
def extract_landmark_vector(landmarks):
    feats = []
    for lm in landmarks:
        feats.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(feats, dtype=np.float32)

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def foot_direction_angle(h, t):
    v = np.array([t.x - h.x, t.y - h.y])
    return np.degrees(np.arctan2(v[1], v[0]))

def extract_joint_angles(lm):
    return np.array([
        calculate_angle(lm[11], lm[13], lm[15]),  # L elbow
        calculate_angle(lm[12], lm[14], lm[16]),  # R elbow
        calculate_angle(lm[13], lm[11], lm[23]),  # L shoulder
        calculate_angle(lm[14], lm[12], lm[24]),  # R shoulder
        calculate_angle(lm[23], lm[25], lm[27]),  # L knee
        calculate_angle(lm[24], lm[26], lm[28]),  # R knee
        calculate_angle(lm[11], lm[23], lm[25]),  # L hip
        calculate_angle(lm[12], lm[24], lm[26]),  # R hip
        calculate_angle(lm[11], lm[23], lm[24])   # spine
    ], dtype=np.float32)

def extract_foot_features(lm):
    return np.array([
        foot_direction_angle(lm[29], lm[31]),
        foot_direction_angle(lm[30], lm[32]),
        calculate_angle(lm[25], lm[27], lm[31]),
        calculate_angle(lm[26], lm[28], lm[32]),
        math.dist([lm[27].x, lm[27].y], [lm[28].x, lm[28].y])
    ], dtype=np.float32)

# =========================
# DATASET ENCODING
# =========================
DATASET_DIR = Path("data/images")
OUTPUT_FILE = Path("data/pose_features.npz")

# FIXED: Create pose instance AFTER functions defined
pose = mp_pose.Pose(
    static_image_mode=True, 
    model_complexity=2, 
    min_detection_confidence=0.7
)

X, y = [], []

print("🔥 Encoding Kaggle yoga dataset...")
print(f"📁 Dataset: {DATASET_DIR}")

# FIXED: Proper directory handling
for pose_name in sorted(os.listdir(DATASET_DIR)):
    pose_dir = DATASET_DIR / pose_name
    if not pose_dir.is_dir():
        print(f"⏭️ Skipping {pose_name} (not directory)")
        continue

    print(f"📂 Processing: {pose_name}")
    
    for img_path in tqdm(list(pose_dir.glob("*jpg*")) + list(pose_dir.glob("*png*")), desc=pose_name):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # FIXED: Consistent image preprocessing
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            continue

        # 🔥 CRITICAL: Copy landmarks before normalization
        lm = list(res.pose_landmarks.landmark)
        normalize_landmarks(lm)

        features = np.concatenate([
            extract_landmark_vector(lm),
            extract_joint_angles(lm),
            extract_foot_features(lm)
        ])

        if features.shape[0] == 146:
            X.append(features)
            y.append(pose_name)

X = np.array(X, dtype=np.float32)
y = np.array(y)

# FIXED: Ensure output directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUTPUT_FILE, X=X, y=y)

print("✅ DONE!")
print(f"📊 Samples: {X.shape[0]}")
print(f"📐 Features: {X.shape[1]}")
print(f"🏷️  Classes: {len(np.unique(y))}")
print(f"💾 Output: {OUTPUT_FILE}")
