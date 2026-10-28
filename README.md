Iris Machine Learning Model
A simple machine learning project built with Python and scikit-learn as part of my MLOps learning plan.
Project Overview
This project trains a Random Forest classifier on the Iris dataset, evaluates its accuracy, and saves the trained model using joblib.
Technologies
- Python 3.12
- scikit-learn
- joblib
- Jupyter Notebook
- Git
Project Files
- 01_first_model.ipynb — initial model training in Jupyter Notebook
- train.py — standalone training script
- iris_model.joblib — saved trained model
- requirements.txt — Python dependencies
- README.md — project documentation

Setup
Create and activate a virtual environment:
python3.12 -m venv .venv
source .venv/bin/activate
Install dependencies:
python -m pip install -r requirements.txt
Train the Model
Run:
python train.py
The script trains the classifier, prints its test accuracy, and saves the model to iris_model.joblib.