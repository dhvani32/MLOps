# MLOps Course

## Overview
This repository contains all course materials, lab exercises, and project files for the MLOps course. The primary objective of this course is to practice standardizing development processes, managing dependencies, and structuring machine learning projects effectively.

## Project Structure
```
mlops-course/  
│── mlops/             # Main project folder
│── models/            # Folder for storing trained models
│── requirements.txt   # Dependencies for the project
│── README.md          # Documentation
│── .DS_Store          
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/dhvani32/MLOps
cd mlops-course
```

### 2. Create and Activate Virtual Environment
Using `venv`:
```bash
python -m venv mlops  
source mlops/bin/activate  # On macOS/Linux  
mlops\Scripts\activate     # On Windows  
```

Using `conda`:
```bash
conda create --name mlops python=3.9  
conda activate mlops  
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt  
```

### 4. Verify Installation
```bash
pip list  
```

### 5. Push Initial Changes
```bash
git add .  
git commit -m "Initial commit: project setup"  
git push origin main  
```

## Dependencies
The following packages are included in `requirements.txt`:
- `mlflow==2.15.1`  
- `numpy==1.26.4`  
- `pandas==2.2.2`  
- `scikit-learn==1.5.1`  

