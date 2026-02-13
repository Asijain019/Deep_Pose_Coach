🧘 Deep Pose Coach
Real-Time Yoga Pose Detection using MediaPipe + SVM (146 Biomechanical Features)

A machine learning-based real-time yoga pose classification system that uses human pose landmarks, joint angles, and foot geometry to detect yoga poses from webcam input.

Built using MediaPipe + Scikit-learn SVM (RBF Kernel).

🚀 Project Overview

Deep Pose Coach converts a human pose into a 146-dimensional biomechanical feature vector and classifies it using a trained SVM pipeline.

The system supports:

✅ Real-time webcam detection

✅ Stable prediction smoothing

✅ Body-centered normalization

✅ Angle-based biomechanical features

✅ Continuous pose switching

🧠 Feature Engineering (146 Dimensions)

Each pose is represented using:

🔹 1. Raw Landmarks (132 Features)

33 body landmarks × (x, y, z, visibility)

🔹 2. Joint Angles (9 Features)

Left & Right elbow

Left & Right shoulder

Left & Right knee

Left & Right hip

Spine angle

🔹 3. Foot Geometry (5 Features)

Left & Right foot direction

Left & Right ankle angles

Feet spread distance

All landmarks are body-centered normalized to remove camera bias.

🏗 Project Structure
Deep_Pose_Coach/
│
├── src/
│   ├── encode_kaggle_dataset.py
│   ├── train_svm.py
│   ├── test.py
│   ├── pose_feature_extractor.py
│   └── mediapipe_pose_test.py
│
├── data/              # (Not included – dataset folder)
├── models/            # (Not included – trained model)
│
├── README.md
├── requirements.txt
└── .gitignore


⚠️ Dataset and trained models are not included due to size.

🛠 Installation
1️⃣ Clone Repository
git clone https://github.com/Asijain019/Deep_Pose_Coach.git
cd Deep_Pose_Coach

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt


If requirements file not available:

pip install opencv-python mediapipe numpy scikit-learn joblib tqdm

📊 Training the Model
Step 1 — Prepare Dataset

Place dataset in:

data/images/
    ├── tree_pose/
    ├── warrior_ii/
    ├── downward_dog/
    └── ...


Each folder should contain images of that pose.

Step 2 — Encode Dataset
python src/encode_kaggle_dataset.py


This generates:

data/pose_features.npz

Step 3 — Train SVM Model
python src/train_svm.py


This creates:

models/svm_pose_classifier.pkl

🎥 Real-Time Pose Detection

Run:

python src/test.py

Instructions:

Ensure full body is visible

Use good lighting

Hold pose for 1–2 seconds

Press Q to exit

⚙️ Model Details
Component	Description
Algorithm	SVM (RBF Kernel)
Preprocessing	StandardScaler
Feature Size	146
Class Weighting	Balanced
Stability	Sliding window voting
🧪 Debug Tools
🔹 Skeleton Check
python src/mediapipe_pose_test.py


Verifies MediaPipe tracking.

🔹 Feature Extraction Test
python src/pose_feature_extractor.py


Confirms 146-dimensional feature generation.

📈 Why This Approach?

Instead of using raw images, this system uses:

Pose geometry

Biomechanical angles

Relative body structure

This improves:

Camera invariance

Lighting robustness

Computational efficiency

🔮 Future Improvements

Pose correctness scoring (0–100%)

Joint-level correction feedback

Deep learning comparison (MLP vs SVM)

ONNX optimization for real-time inference

Front vs side view classifier

🧩 Tech Stack

Python

OpenCV

MediaPipe

NumPy

Scikit-learn

Joblib

👩‍💻 Author

Asi Jain
B.Tech Computer Science
Deep Learning Project

📜 License

This project is open-source and available under the MIT License.
