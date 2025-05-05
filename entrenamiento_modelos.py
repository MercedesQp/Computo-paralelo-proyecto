import os
import csv
import numpy as np
import pickle
from hmmlearn import hmm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

LABELS_CSV = "data/labels.csv"
FEATURES_DIR = "data/processed"
HMM_MODEL_PATH = "models/hmm/hmm_models.pkl"
SVM_MODEL_PATH = "models/svm/svm_model.pkl"

def train_hmm():
    data_by_class = {}
    with open(LABELS_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for fname, label in reader:
            path = os.path.join(FEATURES_DIR, fname.replace(".wav", ".npy"))
            if os.path.exists(path):
                mfcc = np.load(path)
                data_by_class.setdefault(label, []).append(mfcc)

    models = {}
    for label, sequences in data_by_class.items():
        X = np.vstack(sequences)
        lengths = [len(s) for s in sequences]
        model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=100)
        model.fit(X, lengths)
        models[label] = model

    os.makedirs("models/hmm", exist_ok=True)
    with open(HMM_MODEL_PATH, "wb") as f:
        pickle.dump(models, f)

    print("\nModelos HMM guardados en models/hmm/hmm_models.pkl")

def train_svm():
    X, y = [], []
    with open(LABELS_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for fname, label in reader:
            path = os.path.join(FEATURES_DIR, fname.replace(".wav", ".npy"))
            if os.path.exists(path):
                mfcc = np.load(path)
                X.append(np.mean(mfcc, axis=0))
                y.append(label)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nReporte SVM:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    os.makedirs("models/svm", exist_ok=True)
    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le}, f)

    print("Modelo SVM guardado en models/svm/svm_model.pkl")
