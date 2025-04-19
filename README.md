# 🐾 Multi-Class Animal Classification

This project is a deep learning-powered image classification system capable of identifying **90 different animal species** from uploaded images. Using **MobileNetV2** and transfer learning, the model delivers high-accuracy predictions and is deployed as a clean and responsive **Streamlit web application**.

![Demo Screenshot](https://github.com/rithvikchandan/Multi_Class_Animal_Classification/blob/main/Screenshot%202025-04-19%20153958.png)

---

## 🚀 Features

- ✅ Classifies animal images into **90 species**
- 🧠 Uses **MobileNetV2** pretrained on ImageNet
- 📷 Accepts `.jpg`, `.jpeg`, `.png` formats
- 🌐 Deployed via **Streamlit** for real-time interaction
- 📈 Returns predicted class with **confidence score**

---

## 🎯 Learning Objectives

- Understand multi-class classification using CNNs  
- Apply transfer learning using pretrained models  
- Use `ImageDataGenerator` for preprocessing and augmentation  
- Build and deploy a real-time prediction web app with Streamlit  
- Visualize and interpret model predictions  

---

## 🛠️ Tools & Technologies Used

- **Python**  
- **TensorFlow / Keras**  
- **MobileNetV2** (Transfer Learning)  
- **Streamlit**  
- **NumPy**, **Pillow**, **Matplotlib**  
- **Kaggle Dataset** (90 Animal Species)

---

## 🧪 Methodology

1. **Dataset Preparation**  
   - 90 animal categories from Kaggle (folder-structured)

2. **Image Preprocessing**  
   - Resized to 224×224, normalized, and augmented using `ImageDataGenerator`

3. **Model Architecture**  
   - MobileNetV2 as base  
   - GlobalAveragePooling → Dropout(0.3) → Dense(90, softmax)

4. **Training & Evaluation**  
   - Trained with Adam optimizer and categorical crossentropy  
   - Accuracy evaluated and saved as `MCAR.keras`

5. **Deployment**  
   - Real-time prediction app built using Streamlit  

---

## ❗ Problem Statement

Identifying animal species from images is complex due to high intra-class similarity, inter-class variation, and image quality diversity.  
This project solves that by using a deep learning model capable of classifying images into 90 categories with high accuracy, deployed via a simple user interface.

---

## ✅ Solution

- Used **MobileNetV2** with transfer learning for efficient and accurate feature extraction  
- Custom classification layers added for the 90-class problem  
- Images preprocessed and fed through a lightweight CNN  
- Real-time predictions served via a web interface using **Streamlit**

---

## 📁 Project Structure

```
Multi_Class_Animal_Classification/
├── app.py                   # Streamlit App
├── MCAR.keras               # Trained Model
├── requirements.txt         # Dependencies
├── Screenshot.png           # UI Preview
└── animal_classifier_ppt.pptx  # Project Presentation
```

---

## 🔧 How to Run Locally

1. Clone the repository  
```bash
git clone https://github.com/rithvikchandan/Multi_Class_Animal_Classification.git
cd Multi_Class_Animal_Classification
```

2. Install dependencies  
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app  
```bash
streamlit run app.py
```

---

## 📦 Model Summary

- 📐 Input Shape: (224, 224, 3)  
- ⚙️ Architecture: MobileNetV2 + Custom Dense Layers  
- 🔢 Output: 90-class softmax  
- 💾 Saved as: `MCAR.keras`

---

## 🏁 Conclusion

This project demonstrates how deep learning and transfer learning can be effectively used to build a multi-class animal classifier.  
With real-time deployment via Streamlit, it bridges the gap between research and usability — making AI accessible to end-users.

---

## ✨ Improvisations Done by Me

- Customized MobileNetV2 for multi-class classification  
- Built a real-time web app with a user-friendly interface  
- Enhanced prediction output with class mapping and visual feedback

---

## 🤝 Credits

- Dataset: [Kaggle - 90 Animal Species](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)  
- UI Inspiration: Clean, minimal Streamlit design  
- Developed by: **Rithvik Chandan**

---
