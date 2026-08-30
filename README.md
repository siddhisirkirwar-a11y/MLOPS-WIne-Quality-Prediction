# 🍷 Wine Quality Prediction — End-to-End MLOps Project

> **An end-to-end Machine Learning + MLOps project that predicts wine quality and demonstrates the complete ML lifecycle — from data ingestion and validation to model training, evaluation, AWS model deployment, Dockerization, CI/CD, and production inference.**

---

## 🚀 Project Overview

This project implements a **production-oriented Wine Quality Prediction system** using modern Machine Learning and MLOps practices.

The system automatically takes wine-related features, processes the data through an ML pipeline, predicts the **wine quality**, and serves the prediction through a web application.

The project is designed to demonstrate how a machine learning model can move beyond experimentation and be transformed into a **reproducible, automated, and deployable ML system**.

### 🎯 Key Objectives

* Build a complete end-to-end ML pipeline
* Store and retrieve datasets using **MongoDB Atlas**
* Perform automated data validation and transformation
* Handle class imbalance during preprocessing
* Train and evaluate machine learning models
* Track model performance and decide whether a new model should be deployed
* Store trained models in **AWS S3**
* Containerize the application using **Docker**
* Automate deployment using **GitHub Actions**
* Deploy the application on an **AWS EC2** instance
* Provide a web interface for real-time wine quality prediction
* Provide a dedicated `/training` route for model training

---

# 🛠️ Tech Stack

### Programming & Machine Learning

* **Python 3.10**
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib
* Jupyter Notebook

### MLOps

* Modular ML Pipeline
* Data Ingestion
* Data Validation
* Data Transformation
* Model Training
* Model Evaluation
* Model Pusher
* Prediction Pipeline
* Logging
* Custom Exception Handling

### Databases & Storage

* **MongoDB Atlas**
* **AWS S3**

### DevOps & Cloud

* **Docker**
* **AWS ECR**
* **AWS EC2**
* **AWS IAM**
* **GitHub Actions**
* **GitHub Self-Hosted Runner**

### Application

* Flask
* HTML
* CSS
* Static assets
* Templates

---

# 🔬 Machine Learning Workflow

The ML pipeline follows a structured workflow:

```text
Raw Dataset
     ↓
Data Ingestion
     ↓
Data Validation
     ↓
Data Transformation
     ↓
Feature Engineering
     ↓
Handling Class Imbalance
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Acceptance / Rejection
     ↓
Model Registry (AWS S3)
     ↓
Prediction Pipeline
     ↓
Wine Quality Prediction
```

---

# 🧹 Data Validation

The **Data Validation** component ensures that incoming data satisfies the expected schema before it enters the ML pipeline.

Validation includes:

* Expected column validation
* Data type validation
* Dataset structure validation
* Train/test consistency checks
* Schema-based validation
* Validation report generation

This prevents invalid or unexpected data from silently reaching the model-training stage.

---

# ⚙️ Data Transformation

The transformation pipeline prepares raw data for machine learning.

It supports:

* Feature preprocessing
* Numerical feature scaling
* Train/test transformation
* Handling imbalanced target classes
* Reusable preprocessing pipelines
* Serialization of transformation objects

For numerical features, preprocessing techniques such as **StandardScaler** can be integrated into the transformation pipeline.

---

# ⚖️ Handling Imbalanced Data

Wine quality datasets can contain uneven distributions across quality classes.

To address this, the project supports imbalance-handling techniques such as:

**SMOTEENN**

```text
Original Training Data
        ↓
      SMOTE
        ↓
Synthetic Minority Samples
        ↓
      ENN
        ↓
Cleaned Balanced Dataset
        ↓
   Model Training
```

This helps the model learn minority classes instead of being dominated by the majority class.

> The imbalance strategy can be adjusted depending on the distribution of the dataset and the selected model.

---

# 🤖 Model Training

The project follows a modular model-training architecture where machine learning algorithms can be plugged into the training pipeline.

Example classifier:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    random_state=42
)
```

The trained model is serialized and passed to the model evaluation stage.

---

# 📊 Model Evaluation

The **Model Evaluation** component determines whether the newly trained model performs sufficiently well before it is promoted for deployment.

A configurable performance threshold is used:

```python
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02
```

The idea is to avoid replacing a deployed model with a new model unless the new model provides a meaningful improvement.

```text
New Model
    ↓
Evaluate Performance
    ↓
Compare with Existing Model
    ↓
 ┌───────────────┐
 │ Improvement?  │
 └───────┬───────┘
         │
    ┌────┴────┐
   YES        NO
    ↓          ↓
Deploy       Reject
```

---

# ☁️ AWS Model Registry

Accepted models are pushed to **Amazon S3**.

Configured model storage:

```text
S3 Bucket
└── my-model-mlopsproj
    └── model-registry
        └── trained model
```

The project includes an S3 estimator/storage layer responsible for:

* Uploading models
* Downloading models
* Managing model artifacts
* Loading models for prediction

---

# 🐳 Dockerized Application

The application is containerized using Docker to provide a consistent runtime environment.

```text
Source Code
     ↓
Dockerfile
     ↓
Docker Build
     ↓
Docker Image
     ↓
AWS ECR
     ↓
AWS EC2
     ↓
Running Application
```

This makes the application portable and simplifies deployment across environments.

---

# 🔄 CI/CD Pipeline

The project implements a **CI/CD workflow using GitHub Actions**.

### Pipeline

```text
Developer
    │
    ▼
Git Push
    │
    ▼
GitHub Repository
    │
    ▼
GitHub Actions
    │
    ▼
Self-Hosted Runner
    │
    ▼
Build Docker Image
    │
    ▼
Push Image to AWS ECR
    │
    ▼
Deploy on AWS EC2
    │
    ▼
Production Application
```

The CI/CD workflow is triggered when new changes are committed and pushed to GitHub.

---

# 🏃 GitHub Self-Hosted Runner

Instead of relying entirely on GitHub-hosted infrastructure, the project uses an **AWS EC2 instance as a GitHub Actions self-hosted runner**.

```text
GitHub Actions
      │
      ▼
Self-Hosted Runner
      │
      ▼
AWS EC2
      │
      ├── Docker
      ├── Application
      └── Deployment
```

This demonstrates practical experience with custom CI/CD infrastructure.

---

# ☁️ AWS Infrastructure

The deployment architecture uses multiple AWS services:

| AWS Service | Purpose                                |
| ----------- | -------------------------------------- |
| **EC2**     | Hosts the production application       |
| **S3**      | Stores trained ML models               |
| **ECR**     | Stores Docker images                   |
| **IAM**     | Manages authentication and permissions |

### Deployment Architecture

```text
                  AWS Cloud
                     │
        ┌────────────┴────────────┐
        │                         │
       S3                        ECR
        │                         │
   ML Models                Docker Images
        │                         │
        └────────────┬────────────┘
                     │
                     ▼
                    EC2
                     │
                 Docker App
                     │
                     ▼
               Wine Prediction
```

---

# 🌐 Prediction Application

The project includes a web-based prediction application.

The user provides wine-related feature values through the UI.

```text
User Input
    ↓
Flask Application
    ↓
Prediction Pipeline
    ↓
Preprocessing
    ↓
Trained Model
    ↓
Prediction
    ↓
Wine Quality Result
```

The application also includes a dedicated:

```text
/training
```

route that can be used to trigger the model-training pipeline.

---

# 📝 Logging & Exception Handling

The project implements centralized logging and custom exception handling.

### Logging

Pipeline execution generates logs for important operations such as:

```text
Data Ingestion Started
        ↓
Data Validation Completed
        ↓
Data Transformation Started
        ↓
Model Training Started
        ↓
Model Evaluation Completed
        ↓
Model Pushed
```

### Custom Exceptions

A custom exception layer provides meaningful error information while maintaining a clean application structure.

---

# 💡 Why This Project Matters

This project goes beyond simply training a machine learning model.

It demonstrates how to build an ML system that can:

```text
📥 Ingest Data
      ↓
🔍 Validate Data
      ↓
⚙️ Transform Data
      ↓
🤖 Train Model
      ↓
📊 Evaluate Model
      ↓
☁️ Register Model
      ↓
🐳 Containerize Application
      ↓
🔄 Automate Deployment
      ↓
☁️ Deploy to AWS
      ↓
🌐 Serve Predictions
```

This represents the transition from **"ML model in a notebook"** to a **deployable MLOps application**.

## 🔄 End-to-End MLOps Pipeline

The project implements a complete production-oriented machine learning lifecycle:

### 1. 📥 Data Pipeline

**Data Ingestion** ➔ **Data Validation** ➔ **Data Transformation**

* Dataset ingestion from **MongoDB Atlas**
* Schema and data consistency validation
* Feature preprocessing and scaling
* Handling of imbalanced data

### 2. 🤖 Machine Learning Pipeline

**Model Training** ➔ **Model Evaluation** ➔ **Model Deployment**

* Train machine learning classification models
* Evaluate model performance against configurable thresholds
* Store accepted models in **AWS S3**
* Load the approved model through the prediction pipeline

### 3. 🚀 CI/CD & Cloud Deployment

**GitHub Push** ➔ **GitHub Actions** ➔ **Self-Hosted EC2 Runner** ➔ **Docker Build** ➔ **AWS ECR** ➔ **AWS EC2**

* Automated CI/CD pipeline using **GitHub Actions**
* Docker-based application packaging
* Docker images stored in **Amazon ECR**
* Production deployment on **Amazon EC2**
* Self-hosted GitHub Actions runner for deployment automation

---

## 📌 Project Summary

This project demonstrates how a machine learning model can be transformed into a **complete, automated, and cloud-deployable MLOps solution**.

It brings together:

**Machine Learning** + **Data Engineering** + **MLOps** + **Docker** + **CI/CD** + **AWS Cloud**

from raw data ingestion all the way to a deployed wine quality prediction application.

## 💬 Connect

If you found this project interesting or have questions about the implementation, feel free to connect with me on GitHub.

🔗 **GitHub:** [Siddhi Sirkirwar](https://github.com/siddhisirkirwar-a11y)

---