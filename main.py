from datos_preprocesamiento import prepare_labels_from_transcript, extract_mfcc
from entrenamiento_modelos import train_hmm, train_svm
import pickle
import os
import csv
import numpy as np

HMM_MODEL_PATH = "models/hmm/hmm_models.pkl"
SVM_MODEL_PATH = "models/svm/svm_model.pkl"
LABELS_CSV = "data/labels.csv"
FEATURES_DIR = "data/processed"

def evaluate_models():
    with open(HMM_MODEL_PATH, "rb") as f:
        hmm_models = pickle.load(f)
    with open(SVM_MODEL_PATH, "rb") as f:
        svm_data = pickle.load(f)
        svm_model = svm_data["model"]
        le = svm_data["label_encoder"]

    correct_hmm = 0
    correct_svm = 0
    total = 0

    with open(LABELS_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for fname, true_label in reader:
            path = os.path.join(FEATURES_DIR, fname.replace(".wav", ".npy"))
            if not os.path.exists(path):
                continue
            mfcc = np.load(path)

            # HMM
            scores = {label: model.score(mfcc) for label, model in hmm_models.items()}
            hmm_pred = max(scores, key=scores.get)
            if hmm_pred == true_label:
                correct_hmm += 1

            # SVM
            vec = np.mean(mfcc, axis=0).reshape(1, -1)
            pred_svm_idx = svm_model.predict(vec)[0]
            pred_svm_label = le.inverse_transform([pred_svm_idx])[0]
            if pred_svm_label == true_label:
                correct_svm += 1

            total += 1

    acc_hmm = (correct_hmm / total) * 100
    acc_svm = (correct_svm / total) * 100

    print("\nComparativa de Accuracy de Modelos")
    print("+--------------+----------------------+----------------+")
    print("| Modelo       | Correctos / Total    | Accuracy (%)    |")
    print("+--------------+----------------------+----------------+")
    print(f"| HMM          |  {correct_hmm} / {total:<17} |      {acc_hmm:.2f}%     |")
    print(f"| SVM          |  {correct_svm} / {total:<17} |      {acc_svm:.2f}%     |")
    print("+--------------+----------------------+----------------+")

def run_all():
    prepare_labels_from_transcript()

    print("\nProcesamiento SECUENCIAL...")
    t_seq = extract_mfcc(mode='sequential')

    print("\nProcesamiento PARALELO...")
    t_par = extract_mfcc(mode='parallel')

    print("\nComparativa de Tiempos de Extracción de Características")
    print("+----------------------+------------------+")
    print("| Método               | Tiempo (segundos)|")
    print("+----------------------+------------------+")
    print(f"| Secuencial           | {t_seq:>17} |")
    print(f"| Paralelo (multiproc) | {t_par:>17} |")
    print("+----------------------+------------------+")

    train_hmm()
    train_svm()
    evaluate_models()

if __name__ == "__main__":
    run_all()
