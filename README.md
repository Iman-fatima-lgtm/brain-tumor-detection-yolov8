# 🧠 Brain Tumor Detection using YOLOv8

This repository contains a deep learning-based Brain Tumor Detection System implemented in Google Colab using the YOLOv8 object detection model on MRI images. The system automates the identification and localization of brain tumors, providing fast and accurate results to assist healthcare professionals in early diagnosis.

## 📄 Colab Notebook
Click below to launch the project in Google Colab:  
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-lOMSZx0PPA--6UkdXrVGyhpFWLw8ljI)

---

## 📌 Project Highlights

- ✅ **Real-time brain tumor detection** in MRI scans using YOLOv8
- 🖼️ **Gradio-powered web interface** for uploading images and viewing results
- ⚡ **Fast inference** (under 5 seconds per image)
- 🔍 Accurate and consistent tumor localization with bounding boxes
- 🌐 Easily **deployable on Colab, local systems, or cloud servers**

---

## 🎯 Project Objectives

- Collect and preprocess annotated MRI images
- Train YOLOv8 for tumor detection
- Build a real-time web interface using Gradio
- Evaluate system accuracy and performance
- Enable portability, scalability, and ethical deployment

---

## 🧪 Tools & Technologies

| Category         | Tools/Frameworks                                      |
|------------------|--------------------------------------------------------|
| Language         | Python 3.x                                             |
| Deep Learning    | Ultralytics YOLOv8, PyTorch                            |
| Image Processing | OpenCV, NumPy                                          |
| UI / Deployment  | Gradio (Frontend), Google Colab (Runtime)             |
| Version Control  | Git, GitHub                                            |
| Optional         | Flask, Docker, AWS/GCP for scaling                    |

---

## 📁 Dataset Structure

Ensure your data is in the YOLO format:

data/
├── images/
│ ├── train/
│ └── val/
├── labels/
│ ├── train/
│ └── val/
└── dataset.yaml


Example `dataset.yaml`:
```yaml
train: data/images/train
val: data/images/val

nc: 1
names: ['tumor']
🚀 Usage Instructions
🔧 1. Install Requirements
bash
Copy
Edit
pip install ultralytics opencv-python gradio numpy
🧠 2. Train the Model
python
Copy
Edit
from ultralytics import YOLO
model = YOLO("yolov8n.yaml")
model.train(data="data/dataset.yaml", epochs=50, imgsz=640)
🧪 3. Run Detection
python
Copy
Edit
model = YOLO("runs/detect/train/weights/best.pt")
results = model("data/images/test/sample.jpg", save=True)
🌐 4. Launch Gradio Interface
python
Copy
Edit
import gradio as gr
def detect(image):
    results = model(image)
    return results[0].plot()

gr.Interface(fn=detect, inputs="image", outputs="image").launch(share=True)
🎨 System Workflow
Upload MRI image via Gradio interface.

Image is preprocessed using OpenCV.

YOLOv8 performs detection.

Output image is returned with bounding boxes around tumors.

Results are shown in real time and downloadable.

✅ Testing
Test Case	Description
✅ Upload MRI & Detect	Uploads a valid image and verifies tumor bounding boxes.
✅ View Results	Ensures annotated results display properly.
✅ Error Handling	Verifies correct response to invalid/corrupted images.

🔐 Legal & Ethical Considerations
All images are anonymized to protect patient privacy.

The system is not a replacement for professional diagnosis.

Adheres to GDPR/HIPAA-aligned ethical AI practices.

👩‍⚕️ Stakeholders
Medical professionals & radiologists

Hospitals and diagnostic labs

AI/ML researchers in medical imaging

Patients and healthcare organizations

Students & educators in computer vision

📈 Future Work
Expand to 3D MRI image support

Add tumor classification (e.g., glioma, meningioma)

Improve small tumor detection sensitivity

Integrate with hospital PACS or EMR systems

🙌 Acknowledgements
Ultralytics YOLOv8

University of Agriculture Faisalabad – Department of CS

Supervisor: Dr. Kareem Ullah

📚 References
Jiang et al. (2022) – YOLOv5-based brain tumor detection, Computers in Biology and Medicine

Abdel-Basset et al. (2021) – Fusion-based brain tumor detection, Pattern Recognition Letters

Ultralytics Documentation – https://docs.ultralytics.com

Created by: Iman Fatima – 2021-ag-8053
Degree: BS Software Engineering – University of Agriculture Faisalabad
