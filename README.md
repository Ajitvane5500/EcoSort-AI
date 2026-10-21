# EcoSort AI — Waste Classification Web App

**Project type:** Machine Learning mini project  
**Approach:** Transfer learning with MobileNetV2 (CNN-based image classifier)  
**Interface:** Streamlit  
**Language:** Python

## 1. Project objective
EcoSort AI classifies an uploaded waste image into one of six dataset categories: cardboard, glass, metal, paper, plastic, or trash. It then displays a basic disposal suggestion. This is an educational prototype, not a certified waste-management system.

## 2. Project structure
```text
EcoSort_AI_Project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── dataset-resized/       # Add dataset here; not bundled
├── models/
│   └── (trained model files are created here)
└── reports/
    └── Project_Report.docx
```

## 3. Install on Windows
Install Python 3.10 or 3.11 (64-bit) and select **Add Python to PATH** during installation. TensorFlow compatibility depends on your Python/OS version.

Open Command Prompt in the project folder and run:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation fails, check the TensorFlow install guide for your Python version and operating system before changing package versions.

## 4. Get a dataset
Use the public **TrashNet** dataset (six categories: cardboard, glass, metal, paper, plastic, trash). Search for the official TrashNet repository on GitHub and follow its current dataset instructions/licence. This project does not bundle the dataset.

After downloading and extracting the image dataset, arrange it so that each class has its own folder, for example:
```text
data/dataset-resized/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```
Folder names determine the model's class labels. Keep the folder names consistent with the categories you want.

## 5. Train the model
From the activated virtual environment:
```bat
python train_model.py
```
The first training run downloads MobileNetV2 pretrained weights, so an internet connection is needed. Training time depends on your computer. The script creates `models/ecosort_mobilenetv2.keras` and `models/class_names.json`. Validation accuracy is measured on a held-out split; do not claim a fixed accuracy before running your own experiment.

## 6. Run the web application
```bat
python -m streamlit run app.py
```
A local browser tab should open. Upload a JPG, JPEG, PNG, or WEBP image. If the model has not been trained, the app will explain that model files are missing.

## 7. How the system works
1. User uploads an image.
2. Pillow opens the image and converts it to RGB.
3. Image is resized to 224 × 224 pixels.
4. MobileNetV2 preprocessing scales the image for the pretrained network.
5. The model produces class probabilities.
6. The app displays the top predicted category, confidence, class probabilities, and disposal guidance.

## 8. Testing checklist
- Upload a clear image of a single plastic bottle.
- Test paper/cardboard, metal, glass, and general trash images.
- Test a blurry image and note whether confidence falls.
- Upload a non-image file (the uploader should reject unsupported file types).
- Compare predictions against true labels in a small manually checked test set.
Record your actual results in the report; never invent accuracy values.

## 9. Important limitations
- Dataset images may not represent real-world lighting, backgrounds, local waste categories, or contaminated items.
- A model prediction is not a guarantee that an item is recyclable locally.
- The six labels are dataset categories, not a complete waste-sorting standard.
- Avoid touching hazardous or broken waste; follow local municipal guidance.
- Use a separate test set for a more rigorous evaluation before presenting results.

## 10. Viva quick facts
- **ML technique:** Transfer learning / image classification.
- **Base model:** MobileNetV2, a CNN architecture.
- **Why transfer learning?** It reuses visual features learned from a large dataset and reduces training cost.
- **Output layer:** Softmax, one probability per class.
- **Loss:** Sparse categorical cross-entropy.
- **Optimizer:** Adam.
- **Evaluation:** Validation loss and accuracy; ideally also precision, recall, F1-score, and confusion matrix.
