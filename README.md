# Integrated Machine Learning Models for Smoker Status & Forest Cover Type Classification

## Project Description
This project implements and compares machine learning models on **two classification tasks**:

1. **Binary Classification – Smoker Status Prediction**  
   Predict whether an individual is a smoker or non-smoker using physiological and clinical indicators.

2. **Multiclass Classification – Forest Cover Type Prediction**  
   Predict one of **7 forest cover types** using environmental and geographical features from the UCI Forest Cover Type Dataset.

The study includes exploratory data analysis, feature engineering, model selection, hyperparameter tuning, and performance evaluation.  
The detailed report is provided in the repository as **AIT_511_ML_2.pdf**.

---

## Dataset Information

### **1. Smoker Status Dataset**
- **Records:** 38,984 samples  
- **Numerical Features:** Height, Weight, BMI, Hemoglobin, Cholesterol levels, Blood pressure indicators, etc.  
- **Categorical Features:** Gender, Hearing issues, Dental caries, Urine protein, etc.  
- **Target Variable:** Smoker (1) or Non-smoker (0)

### **2. Forest Cover Type Dataset**
- **Records:** 581,012 samples  
- **Features:** 54 cartographic variables  
  - Elevation, Slope, Hillshade measures  
  - Soil type indicators  
  - Hydrology distances  
- **Target Variable:** 7 forest cover types  
  (Spruce-Fir, Lodgepole Pine, Ponderosa Pine, Cottonwood/Willow, Aspen, Douglas-fir, Krummholz)

---

## Models Implemented

### **A. Smoker Status Prediction**

#### 1. Logistic Regression
- Baseline linear classifier  
- **Accuracy:** ~72.96%

#### 2. Support Vector Machine (Best Performing)
- **Best Parameters:** RBF kernel, C=1.5, γ=0.5  
- **Accuracy:** ~77.97%

#### 3. Neural Network
- **Architecture:** (256 → 128 → 64) with ReLU  
- **Accuracy:** ~76.48%

---

### **B. Forest Cover Type Classification**

#### 1. Softmax Regression
- Baseline multiclass model  
- **Accuracy:** ~72.34%

#### 2. SVM with PCA
- PCA reduced dimensions for computational efficiency  
- **Accuracy:** ~90.20%

#### 3. Multilayer Perceptron (Best Performing)
- **Architecture:** (512 → 256 → 128 → 64), ReLU activations  
- **Accuracy:** ~94.62%

---

## Key Findings

### **Smoker Status**
- SVM with RBF kernel achieved the best performance.  
- Outlier handling (IQR + winsorization) significantly improved model stability.  
- BMI, Hemoglobin, and Cholesterol were strong predictors of smoking status.

### **Forest Cover**
- Neural Network achieved the highest accuracy.  
- PCA drastically improved SVM training time without major performance loss.  
- Elevation, Hillshade, and Hydrology distance were top contributing features.

---

## Project Structure
- `data/` - Dataset files and preprocessing scripts
- `notebooks/` - Exploratory data analysis and model implementation

## Technologies Used
- Python
- Scikit-learn
- PyTorch
- Pandas, NumPy
- Matplotlib, Seaborn
