# 🚦 Intelligent Traffic Management System (AI)

An end-to-end, AI-powered traffic management architecture that monitors feeds in real time, computes intersection densities, tracks queues, triggers emergency vehicle priority preemption, logs driving violations, and dynamically scales signal configurations using Machine Learning.

---

## 🧩 1. Problem Statement

Traditional traffic management architectures use fixed-interval timers. This rigid implementation introduces severe industrial and urban inefficiencies:

* **Asymmetric Latency:** Heavy traffic columns get restricted to the same ~30-second window as completely clear road corridors.
* **Emergency Impediment:** First-responder vehicles (ambulances, fire trucks) get trapped in static queues, risking lives.
* **Environmental & Economic Cost:** Idle waiting loops lead to massive queue extensions, fuel wastage, and high carbon emissions.

**The Solution:** This project utilizes real-time Computer Vision and Adaptive AI algorithms to dynamically modify green light durations based on live traffic densities and queue states.

---

## 🔁 2. End-to-End System Pipeline

The core framework processes input telemetry frames through the following architectural hierarchy:

```text
Traffic Video / CCTV Stream
      │
      ▼
[ Frame Extraction & Preprocessing ] ──► (Resize, Noise Filters, Color Spaces)
      │
      ▼
[ Vehicle Detection (YOLOv8) ] ────────► (Cars, Trucks, Buses, Motorcycles)
      │
      ▼
[ Object Tracking (ByteTrack) ] ───────► (Persistent ID Association across frames)
      │
      ▼
[ Spatial Virtual Counting Line ] ─────► (Incremental Counters / No Double Counting)
      │
      ▼
[ Feature Extraction Engine ] ─────────► (Traffic Density, Queue Length, Speed Tracking)
      │
      ▼
[ Cognitive Decision Rules / ML ] ─────► (Adaptive Green Signal Optimization)
      │
      ├───► [ Priority Preemption ] ──► (Emergency Vehicles Detection Override)
      │
      └───► [ Enforcement Engine ] ───► (Red Light & Lane Violation Tracking + ANPR OCR)
            │
            ▼
[ SQLite Database Logging ] ───────────► (Structured Audits & Historical Data Store)
      │
      ▼
[ Streamlit UI Dashboard ] ────────────► (Real-Time Visualizations & Operational Analytics)
```
## ✨ 3. Detailed Component Features

### 🎞️ Video Processing & Frame Preprocessing
* Captures high-frame-rate inputs using OpenCV VideoCapture streams from CCTV, drones, or prototype video samples (`traffic1.mp4`, `junction.mp4`).
* Applies frame resizing, localized noise removal, and brightness/contrast adjustments to maintain high inference accuracy in low-light or adverse weather states.

### 🚗 Vehicle Detection & Tracking
* Utilizes **YOLOv8** to segment and bound distinct object classes: Car, Truck, Bus, Motorcycle, and Bicycle.
* Integrates object tracking (ByteTrack / DeepSORT) to lock unique identifiers (ID $x$) to specific bounding objects across consecutive frames, eliminating duplication.

### 🧮 Quantitative Metrics (Counting, Density & Queue Length)
* **Virtual Count Line:** Tracks the center-point intersections of tracked IDs across configured image thresholds (Count++).
* **Traffic Density Index:** Computes lane saturation levels dynamically:
---
$$\text{Density} = \frac{\text{Vehicles Present}}{\text{Maximum Lane Capacity}}$$
-
* **Queue Length Mapping:** Measures pixel distance from the stop-line layout boundary back to the furthest detected upstream vehicle, converting the value to a real-world metric (meters).
* **Velocity Estimation:** Computes travel times across calibrated virtual distances to screen for overspeeding violations:
---
$$\text{Speed} = \frac{\text{Distance}}{\text{Time}}$$
-

### 🚨 Emergency Preemption & Violation Control
* **Dedicated Tracking Logic:** Scans for priority vehicle targets (Ambulance, Fire Truck, Police). Detection applies an immediate safe override, maximizing green phase cycles along that lane.
* **Enforcement Checks:** Cross-references active signal matrices with bounding vector overlaps. If a vehicle crosses the virtual stop line while the signal state is RED, a traffic violation is captured with an image snapshot, timestamp, and unique tracking ID.
* **ANPR Integration:** Crops localized vehicle license plates and utilizes an OCR engine (EasyOCR / Tesseract) to convert license geometry into indexable text strings.

### 📊 AI Signal Decision Engine
The controller alternates between two core execution topologies:

* **Deterministic Rule Engine:** Operates fallback conditional algorithms:
  * Density > 80% or Long Queue → Green = **60s**
  * Density > 60% → Green = **45s**
  * Density > 40% → Green = **35s**
  * Default Minimum Baseline → Green = **20s**
* **Machine Learning Model:** Implements an optimized Random Forest Classifier that predicts timing profiles by evaluating combined vectors: vehicle arrays, traffic densities, maximum queue distances, and average road speeds.

---

## 🗂️ 4. Suggested Folder Structure

```text
Traffic_AI/
│
├── videos/                  # Raw source footage repository (e.g., junction.mp4)
├── models/                  # Stored model weights files (yolov8n.pt)
├── outputs/                 # Processed visual outputs and diagnostic clips
├── violations/              # Image captures of red-light violations and ANPR cropped plates
├── database/                # SQLite storage databases (traffic_logs.db)
├── dashboard/               # Streamlit application orchestration source files
│
├── detection.py             # Core YOLOv8 object detection module
├── tracking.py              # ByteTrack framework integration configurations
├── counting.py              # Virtual boundary calculation logic
├── density.py               # Calculation engines for density metrics and queue trackers
├── signal.py                # Rule-based and Random Forest adaptive timing scripts
├── violation.py             # Enforces red-light checkpoints and violation logging
├── plate.py                 # ANPR extraction pipeline and EasyOCR bindings
├── app.py                   # Streamlit web-app UI dashboard frontend
├── requirements.txt         # Project production library stack dependencies
└── README.md                # General system documentation manual
```
## 🛠️ 5. Technologies Used

| Module | Core Technology |
| :--- | :--- |
| **Video Ingestion Engine** | OpenCV (`cv2`) |
| **Object Localization / Classification** | Ultralytics YOLOv8 |
| **Identity Persistence Framework** | ByteTrack / DeepSORT Tracker |
| **Mathematical Analytics Engine** | NumPy, Pandas, Scikit-Learn |
| **Optical Character Recognition (ANPR)** | EasyOCR / Tesseract OCR |
| **Structured Relational Storage** | SQLite / MySQL |
---
🚀Methods to use
---
**1.** Running with a webcam:
```bash
python main.py
```
**2.** Running with a specific video file (e.g., traffic1.mp4 ):
```bash
python main.py traffic1.mp4
```
