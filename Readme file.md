# 🎗 Breast Cancer Classifier — Medical AI Desktop App

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Tkinter-GUI-ff4e8e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-Wisconsin%20BC-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Ready-brightgreen?style=for-the-badge"/>
</p>

> **A professional desktop application** that classifies breast tumors as **Malignant (M)** or **Benign (B)** using three machine learning algorithms — built with Python Tkinter and a modern dark-themed GUI.

---

## 📁 Project Files

```
breast-cancer-classifier/
│
├── Breast_Cancer_classification.ipynb   ← Research notebook (data analysis + model training)
├── breast_cancer_gui.py                 ← Main GUI application (run this file)
└── README.md                            ← You are here
```

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 🏠 **Dashboard** | Live metric cards showing dataset stats and model accuracy after training |
| 📊 **Dataset Explorer** | Browse 569 rows of the Wisconsin dataset in an interactive table |
| 🤖 **Model Training** | Train 3 classifiers with one click — real-time training log |
| 🔬 **Predict** | Enter 5 tumor features and get instant Malignant / Benign diagnosis |
| 📈 **Visualize** | 4 interactive charts: Class Distribution, Radius Histogram, Heatmap, Box Plot |

---

## 🧠 Machine Learning Models

Three classifiers are trained and compared:

| Model | Algorithm | Notes |
|-------|-----------|-------|
| **Logistic Regression** | Linear model with sigmoid function | Best for linearly separable data |
| **KNN Classifier** | K-Nearest Neighbors (k=1) | Distance-based classification |
| **Decision Tree** | Information gain splitting | Interpretable tree structure |

**Prediction logic:** All 3 models vote — majority (2 out of 3) wins the final diagnosis.

---

## 📦 Dataset

- **Name:** Wisconsin Breast Cancer Dataset
- **Source:** UCI ML Repository via Kaggle (`uciml/breast-cancer-wisconsin-data`)
- **Samples:** 569 total — 212 Malignant, 357 Benign
- **Features:** 30 numeric cell nucleus measurements
- **Target:** `diagnosis` — M (Malignant) or B (Benign)

**Key features used for prediction:**

| Feature | Range | Description |
|---------|-------|-------------|
| `radius_mean` | 8.0 – 28.0 | Mean radius of cell nuclei |
| `texture_mean` | 9.0 – 40.0 | Standard deviation of gray-scale values |
| `perimeter_mean` | 43.0 – 190.0 | Mean size of the core tumor |
| `area_mean` | 140.0 – 2500 | Mean area of cell nuclei |
| `smoothness_mean` | 0.05 – 0.17 | Local variation in radius lengths |

---

## 🚀 Installation & Setup

### Step 1 — Prerequisites

Make sure you have **Python 3.8 or higher** installed.

```bash
python --version
```

### Step 2 — Install Required Libraries

```bash
pip install kagglehub pandas numpy scikit-learn matplotlib seaborn
```

### Step 3 — Kaggle API Setup

This project downloads the dataset automatically via `kagglehub`. You need a Kaggle account:

1. Go to [https://www.kaggle.com](https://www.kaggle.com) → Account → **Create New Token**
2. Download `kaggle.json`
3. Place it in the correct location:

```bash
# Windows
C:\Users\<YourName>\.kaggle\kaggle.json

# Linux / Mac
~/.kaggle/kaggle.json
```

### Step 4 — Run the Application

```bash
python breast_cancer_gui.py
```

---

## 🖥️ How to Use the App

### 1. Load Dataset & Train Models
- Click **"⬇ Load Dataset & Train All Models"** on the Dashboard
- The app will automatically download the dataset from Kaggle and train all 3 models
- Watch the real-time training log in the **Train Models** page

### 2. Explore the Dataset
- Navigate to **📊 Dataset** from the sidebar
- Browse 100 rows of data with all 15 visible columns
- See counts of Malignant vs Benign samples

### 3. Make a Prediction
- Go to **🔬 Predict** in the sidebar
- Enter 5 tumor feature values manually, **or** click:
  - `Malignant Sample` — fills a known malignant case
  - `Benign Sample` — fills a known benign case
- Click **"Run Prediction"** to see the result
- All 3 model votes are shown alongside the final diagnosis

### 4. Visualize Data
- Go to **📈 Visualize**
- Choose from 4 chart types:
  - 📊 Class Distribution
  - 📈 Radius Mean Distribution (Malignant vs Benign)
  - 🔥 Correlation Heatmap (first 10 features)
  - 📦 Box Plot (Radius Mean by class)

---

## 🗂️ Notebook Overview (`Breast_Cancer_classification.ipynb`)

The Jupyter notebook contains the full data science workflow:

```
1. Import Libraries          → kagglehub, pandas, matplotlib, seaborn
2. Load Dataset              → Download via kagglehub API
3. Data Cleaning             → Drop 'Unnamed: 32' and 'id' columns
4. Null & Duplicate Check    → isnull().sum(), duplicated()
5. Exploratory Analysis      → describe(), value_counts()
6. Encode Labels             → M → 1, B → 0
7. Train/Test Split          → 80% train, 20% test (random_state=2)
8. Data Visualization        → distplot, barplot, lineplot, histplot, boxplot
9. Logistic Regression       → Train + accuracy on train & test
10. KNN Classifier           → n_neighbors=1, accuracy on test
11. Decision Tree            → Default params, accuracy on test
12. Correlation Heatmap      → 15×15 feature correlation matrix
13. Count Plot + Pairplot    → Label distribution and feature relationships
```

---

## 🏗️ GUI Code Structure (`breast_cancer_gui.py`)

```
BreastCancerApp (tk.Tk)
│
├── _setup_styles()          → ttk dark theme, colors, fonts
├── _build_ui()              → Header + Sidebar + Pages + Status bar
│
├── Pages
│   ├── _build_page_home()       → Dashboard with metric & accuracy cards
│   ├── _build_page_data()       → Treeview dataset explorer
│   ├── _build_page_train()      → Model cards + training log
│   ├── _build_page_predict()    → Input form + result display
│   └── _build_page_visualize()  → Matplotlib chart canvas
│
└── Logic
    ├── _load_and_train()        → Starts training thread
    ├── _train_thread()          → Downloads data, trains 3 models
    ├── _predict()               → Runs prediction with majority vote
    └── _draw_chart()            → Renders selected visualization
```

---

## 🎨 Design System

| Element | Color | Usage |
|---------|-------|-------|
| Background | `#0d1117` | Main dark background |
| Card | `#161b22` | Panel and card backgrounds |
| Pink Accent | `#ff4e8e` | Buttons, Malignant results |
| Blue Accent | `#58a6ff` | Logistic Regression, headings |
| Green Accent | `#3fb950` | Benign results, KNN model |
| Gold Accent | `#d29922` | Decision Tree model |
| Text Primary | `#e6edf3` | Main readable text |
| Text Muted | `#8b949e` | Labels, subtitles |

---

## ⚠️ Important Disclaimer

> This application is built for **educational and learning purposes only**.  
> It is **not** a medical device and should **never** be used for actual clinical diagnosis.  
> Always consult a qualified medical professional for any health concerns.

---

## 📋 Requirements Summary

```txt
python >= 3.8
kagglehub
pandas
numpy
scikit-learn
matplotlib
seaborn
tkinter  (built-in with Python)
```

---

## 👤 Author

**Student Project** — Breast Cancer Classification with Machine Learning GUI  
Dataset: [UCI Wisconsin Breast Cancer](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

<p align="center">Made with ❤️ using Python, Tkinter & scikit-learn</p>
