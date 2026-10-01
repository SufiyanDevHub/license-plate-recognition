# 🚘 License Plate Recognition System

An AI-powered **License Plate Recognition System** built with **YOLO, EasyOCR, OpenCV, and Streamlit**.

The application detects vehicle license plates from uploaded images and videos, draws bounding boxes around detected plates, and automatically reads the text written on the license plate using OCR.

## 🌐 Live Demo

👉 **[Try the Live App](https://license-plate-recognitions.streamlit.app/)**

## ✨ Features

* 🚘 License plate detection using YOLO
* 🔍 Automatic license plate number recognition using EasyOCR
* 📷 Image upload and processing
* 🎥 Video upload and processing
* 📦 Bounding boxes around detected license plates
* 📝 Extracted license plate text displayed on the result
* 📊 OCR confidence score
* ⚡ Streamlit-based interactive web interface

## 🧠 How It Works

```text
User Uploads Image / Video
          ↓
     YOLO Detection
          ↓
   License Plate Crop
          ↓
     Image Processing
          ↓
        EasyOCR
          ↓
 License Plate Number
          ↓
  Bounding Box + OCR Text
```

## 🛠️ Tech Stack

* **Python**
* **YOLO (Ultralytics)**
* **EasyOCR**
* **OpenCV**
* **Streamlit**
* **NumPy**
* **Pillow**
* **PyTorch**

## 📸 Example Workflow

### Image

```text
Upload Image
     ↓
Detect License Plate
     ↓
Draw Bounding Box
     ↓
Read Plate Number
     ↓
Display Detected Number + Confidence
```

### Video

```text
Upload Video
     ↓
Process Video Frames
     ↓
Detect License Plates
     ↓
Read Plate Numbers
     ↓
Generate Processed Video
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/SufiyanDevHub/license-plate-recognition.git
cd license-plate-recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📂 Project Structure

```text
license-plate-recognition/
│
├── app.py
├── best.pt
├── requirements.txt
├── .gitignore
└── README.md
```

## 🎯 Project Goal

The goal of this project is to build an end-to-end **Computer Vision and OCR pipeline** capable of detecting license plates and extracting the text written on them.

This project combines **object detection, image processing, OCR, and web deployment** into a single application.

## 👨‍💻 Author

**Sufiyan Ali**

AI & Data Science Student
Computer Vision | Machine Learning | Python | Data Science

---

⭐ If you find this project useful, consider giving the repository a star!
