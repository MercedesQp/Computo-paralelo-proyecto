import pickle
from hmmlearn import hmm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from multiprocessing import Pool
import time

HMM_MODEL_PATH = "models/hmm/hmm_models.pkl"
SVM_MODEL_PATH = "models/svm/svm_model.pkl"
NUM_CORES = 1

# Entrenar HMM
def train_hmm_for_label(args):
    label, sequences = args
    X = np.vstack(sequences)
    lengths = [len(s) for s in sequences]
    model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=100)
    model.fit(X, lengths)
    return (label, model)

def train_hmm():
    import csv
    data_by_class = {}
    with open("data/labels.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for fname, label in reader:
            path = os.path.join("data/processed", fname.replace(".wav", ".npy"))
            if os.path.exists(path):
                mfcc = np.load(path)
                data_by_class.setdefault(label, []).append(mfcc)

    with Pool(NUM_CORES) as pool:
        results = pool.map(train_hmm_for_label, data_by_class.items())

    models = dict(results)
    os.makedirs("models/hmm", exist_ok=True)
    with open(HMM_MODEL_PATH, "wb") as f:
        pickle.dump(models, f)

# Entrenar SVM
def load_and_vectorize_svm(args):
    fname, label = args
    path = os.path.join("data/processed", fname.replace(".wav", ".npy"))
    if os.path.exists(path):
        mfcc = np.load(path)
        return (np.mean(mfcc, axis=0), label)
    return None

def train_svm():
    import csv
    with open("data/labels.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        tasks = [(fname, label) for fname, label in reader]

    with Pool(NUM_CORES) as pool:
        results = pool.map(load_and_vectorize_svm, tasks)

    filtered = [r for r in results if r is not None]
    X, y = zip(*filtered)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    os.makedirs("models/svm", exist_ok=True)
    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le}, f)
