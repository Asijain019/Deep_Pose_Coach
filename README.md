# 🧘 Deep Pose Coach  
### Real-Time Yoga Pose Detection using MediaPipe + SVM

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/MediaPipe-Pose-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-SVM-yellow?style=for-the-badge&logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" />
</p>

---

## 🚀 Overview

Deep Pose Coach is a real-time yoga pose classification system built using:

- 🧍 MediaPipe Pose Estimation  
- 📐 Biomechanical Feature Engineering  
- 🧠 Support Vector Machine (RBF Kernel)  

Instead of training directly on raw images, this project converts each human pose into a structured **146-dimensional geometric feature vector**, and classifies it using a machine learning pipeline.

This approach is:

- Lightweight  
- Interpretable  
- CPU-friendly  
- Fully reproducible  

---

# 🧠 How the System Works

## Step 1 — Pose Estimation

MediaPipe extracts **33 body landmarks** per frame.

Each landmark provides:

- x coordinate  
- y coordinate  
- z coordinate  
- visibility score  

---

## Step 2 — Feature Engineering (146 Features)

Each pose is converted into:

### 🔹 132 Raw Landmark Features  
33 landmarks × (x, y, z, visibility)

### 🔹 9 Joint Angle Features  
- Left & Right elbows  
- Left & Right shoulders  
- Left & Right knees  
- Left & Right hips  
- Spine angle  

Angles are computed using vector geometry.

### 🔹 5 Foot Geometry Features  
- Left & Right foot direction  
- Left & Right ankle angles  
- Feet spread distance  

All landmarks are **body-centered normalized** before training.

---

## Step 3 — Model Training

A Scikit-learn Pipeline is used:

1. StandardScaler  
2. SVM (RBF Kernel)  

The scaler ensures consistent feature scaling during both training and inference.

The trained model is saved as:

```
models/svm_pose_classifier.pkl
```

---

# 📦 Dataset

## 📥 How to Download the Dataset (Kaggle)

This project uses a yoga pose image dataset from Kaggle.

You can search on Kaggle for:

```
Yoga Pose Image Classification Dataset
```

Example:
https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset

---

## 🔽 Option 1 — Download Manually

1. Create a Kaggle account  
2. Open dataset page  
3. Click **Download**
4. Extract the dataset  
5. Place images inside:

```
data/images/
    ├── tree_pose/
    ├── warrior_ii/
    ├── downward_dog/
    └── ...
```

Each folder should contain images of that pose.

---

## 🔽 Option 2 — Download via Kaggle API

### 1️⃣ Install Kaggle CLI

```bash
pip install kaggle
```

### 2️⃣ Add Kaggle API key

- Go to Kaggle → Account  
- Click **Create API Token**
- Download `kaggle.json`
- Place it in:

Windows:
```
C:\Users\YourUsername\.kaggle\
```

Linux/Mac:
```
~/.kaggle/
```

### 3️⃣ Download Dataset

```bash
kaggle datasets download -d niharika41298/yoga-poses-dataset
```

### 4️⃣ Extract

```bash
unzip yoga-poses-dataset.zip -d data/images
```

---

# 📂 What Is Excluded from This Repository?

To keep the repository clean and lightweight, the following are excluded:

```
data/
models/
venv/
```

### Why?

- Datasets are large
- Trained models are binary files
- Virtual environments are system-specific
- GitHub file limit: 100MB

---

# 🏗 How to Reproduce Everything From Scratch

Follow these steps to rebuild everything locally.

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Asijain019/Deep_Pose_Coach.git
cd Deep_Pose_Coach
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Download Dataset

Follow Kaggle instructions above.

Ensure structure:

```
data/images/
```

---

## 5️⃣ Encode Dataset into Features

```bash
python src/encode_kaggle_dataset.py
```

This generates:

```
data/pose_features.npz
```

---

## 6️⃣ Train Model

```bash
python src/train_svm.py
```

This generates:

```
models/svm_pose_classifier.pkl
```

---

## 7️⃣ Run Real-Time Detection

```bash
python src/test.py
```

---

# 🎥 Real-Time Instructions

- Ensure full body is visible  
- Use good lighting  
- Hold pose for ~1–2 seconds  
- Press **Q** to exit  

---

# ⚙️ Model Details

| Component | Description |
|-----------|------------|
| Algorithm | SVM (RBF Kernel) |
| Preprocessing | StandardScaler |
| Feature Size | 146 |
| Runtime | Real-time |
| Hardware | CPU |

---

# 🔍 Debug Utilities

### Skeleton Test

```bash
python src/mediapipe_pose_test.py
```

### Feature Extraction Test

```bash
python src/pose_feature_extractor.py
```

---

# 📈 Why This Approach?

Compared to deep CNN models:

- Requires smaller dataset  
- Faster training  
- Lower computational cost  
- Interpretable geometric features  
- Stable real-time performance  

---

# 🔮 Future Improvements

- Pose correctness scoring (0–100%)
- Joint-level correction feedback
- Multi-person support
- Deep learning comparison
- Web deployment

---

# 🧩 Tech Stack

- Python  
- OpenCV  
- MediaPipe  
- NumPy  
- Scikit-learn  
- Joblib  

---

# 👩‍💻 Author

**Asi Jain**  
B.Tech Computer Science  
Deep Learning Project  

---

# 📜 License

MIT License

