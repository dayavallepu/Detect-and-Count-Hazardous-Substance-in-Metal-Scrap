# Detect and Count the Hazardous Substance in Metal Scrap

This project leverages deep learning (YOLO, Faster R-CNN) to detect and count hazardous substances in metal scrap images. The workflow follows the CRISP-ML(Q) methodology for robust, explainable AI development.

---

## CRISP-ML(Q) Workflow

### 1. Business and Data Understanding

- **Business Problem:** Automate detection and counting of hazardous substances in metal scrap for safety and compliance.
- **Objectives:** Maximize workers safety, Minimize maintenance costs
- **Constraints:** Minimize cost,  Maximize realtime detections.
- **ML Success Criteria:** Achieve detection accuracy of at least 85%
- **Stakeholders:** Scrap yard operators, environmental agencies, safety inspectors.
- **Project Charter & Architecture:** Defined using CRISP-ML(Q) forms and diagrams. architecture created using (`Draw.io`) 
  [View Project Architecture image](Project_265.png)

#### Image Classes
1. Gas Cylinders
2. Aerosol Cans/Paint Tins
3. Fire Extinguishers
4. Hydraulic/Pneumatic Cylinders
5. Compressors
6. Oil Filters
7. Fluid Tanks/Cans
8. Batteries
9. Unexploded Ordnance/Ammunition Shells

#### Challenges & Solutions
- **Data Availability:** Used video frames, Google Lens for data collection.
- **Watermarks:** Applied watermark removal tools.

---

### 2. Data Preparation

- **Annotation:** Used Digital Sreeni for polygon annotation (80–90 per class), saved in COCO JSON format.
- **Augmentation:** Each image expanded to 15 using rotation, zoom, blur, brightness, sharpening, flipping, grayscale, and histogram equalization.
- **Splitting:** 70% train, 20% validation, 10% test.

---

### 3. Model Building

- **Models Tried:** YOLOv8, YOLOv5, YOLOv9, YOLOv11, Faster R-CNN.
- **Final Model:** YOLOv9s, YOLOv8s selected based on performance.

---

### 4. Model Evaluation

- **Metrics:** Precision, Recall, mAP@50, mAP@50:90.

---

### 5. Model Deployment

- **Deployment:** Flask web app for inference.

---

### 6. Monitoring and Maintenance

- **Status:** Not started.

---

### Documentation Deliverables

- All documentation and deliverables prepared by 19/05/2025.

---

## Setup

1. Clone the repository and install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. (Optional) If using Google Colab, mount your Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

---

## Usage

1. **Data Conversion:** Convert COCO annotations to YOLO format using provided scripts/notebooks.
2. **YAML Creation:** Generate `dataset.yaml` for YOLO training.
3. **Training:** Train the YOLOv9s, YOLOv8s,YOLO11, YOLOv5m and Faster R-CNNs using the `train.py` script.
 model using your dataset.
4. **Validation & Testing:** Evaluate model performance and visualize results.
5. **Inference:** Detect and count objects in new images.

---

## Results

- Training, validation, and test metrics are logged and visualized.
- Output images and plots are saved in the `all model visuals/` directory.

---

## Requirements

See [requirements.txt](requirements.txt) for all dependencies.

---

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PyCOCOTools](https://github.com/cocodataset/cocoapi)