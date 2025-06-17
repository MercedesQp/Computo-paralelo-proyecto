from preprocesamiento import prepare_labels_from_transcript, extract_mfcc
from entrenamiento import train_hmm, train_svm, train_combined_hmm_svm
from evaluacion import evaluate_models

def run_all():
    prepare_labels_from_transcript()
    extract_mfcc()
    train_combined_hmm_svm()
    train_hmm()
    train_svm()
    evaluate_models()

if __name__ == "__main__":
    run_all()
