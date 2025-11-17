import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

# Funzione per caricare i file pkl
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Percorsi ai file .pkl (assicurati di averli correttamente indicati)
files_before = [
    'Resnet_base/runs/0_poison_subsampled/gan2:pre&gan3:pre&sdXL:pre&real:pre/scores/gan2:pre&gan3:pre&sdXL:pre&real:pre_vs_gan2:pre.pkl',
    #'Resnet_base/runs/0_poison_subsampled/gan2:pre&gan3:pre&sdXL:pre&real:pre/scores/gan2:pre&gan3:pre&sdXL:pre&real:pre_vs_gan3:pre.pkl',
    'Resnet_base/runs/0_poison_subsampled/gan2:pre&gan3:pre&sdXL:pre&real:pre/scores/gan2:pre&gan3:pre&sdXL:pre&real:pre_vs_real:pre.pkl',
    #'Resnet_base/runs/0_poison_subsampled/gan2:pre&gan3:pre&sdXL:pre&real:pre/scores/gan2:pre&gan3:pre&sdXL:pre&real:pre_vs_sdXL:pre.pkl',
]

files_after = [
    'Resnet_base/runs/20_poison_subsampled/gan2:pre&gan3:pre&sdXL:pre&real:pre/scores/gan2:pre&gan3:pre&sdXL:pre&real:pre_vs_gan2:pre.pkl',
    #'Resnet_base/runs/20_poison_subsampled/gan2:pre&gan3:pre&sdXL:pre&real:pre/scores/gan2:pre&gan3:pre&sdXL:pre&real:pre_vs_gan3:pre.pkl',
    'Resnet_base/runs/20_poison_subsampled/gan2:pre&gan3:pre&sdXL:pre&real:pre/scores/gan2:pre&gan3:pre&sdXL:pre&real:pre_vs_real:pre.pkl',
    #'Resnet_base/runs/20_poison_subsampled/gan2:pre&gan3:pre&sdXL:pre&real:pre/scores/gan2:pre&gan3:pre&sdXL:pre&real:pre_vs_sdXL:pre.pkl',
]

# Variabili per accumulare i risultati
y_true_before = []
y_pred_before = []
y_true_after = []
y_pred_after = []

# Carica i dati e accumula i risultati
for file_before, file_after in zip(files_before, files_after):
    data_before = load_pkl(file_before)
    data_after = load_pkl(file_after)
    
    y_true_before.extend(data_before['y_true'].cpu().numpy())
    y_pred_before.extend(data_before['y_pred'].cpu().numpy())
    
    y_true_after.extend(data_after['y_true'].cpu().numpy())
    y_pred_after.extend(data_after['y_pred'].cpu().numpy())

# Calcola le metriche di performance per "before" e "after"
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return accuracy, precision, recall, f1, cm

# Calcola le metriche prima e dopo unlearning
accuracy_before, precision_before, recall_before, f1_before, cm_before = calculate_metrics(y_true_before, y_pred_before)
accuracy_after, precision_after, recall_after, f1_after, cm_after = calculate_metrics(y_true_after, y_pred_after)

# Stampa i risultati
print("### Performance 0% poison ###")
print(f"Accuracy: {accuracy_before*100:.2f}%")
print(f"Precision: {precision_before*100:.2f}%")
print(f"Recall: {recall_before*100:.2f}%")
print(f"F1 Score: {f1_before*100:.2f}%")
print("Confusion Matrix:")
print(pd.DataFrame(cm_before, index=["Real", "Fake"], columns=["Real", "Fake"]))

print("\n### Performance 20% poison ###")
print(f"Accuracy: {accuracy_after*100:.2f}%")
print(f"Precision: {precision_after*100:.2f}%")
print(f"Recall: {recall_after*100:.2f}%")
print(f"F1 Score: {f1_after*100:.2f}%")
print("Confusion Matrix:")
print(pd.DataFrame(cm_after, index=["Real", "Fake"], columns=["Real", "Fake"]))

# Differenza netta per classe
diff_accuracy = accuracy_after - accuracy_before
diff_precision = precision_after - precision_before
diff_recall = recall_after - recall_before
diff_f1 = f1_after - f1_before

print("\n### Net Difference (After - Before) ###")
print(f"Accuracy Difference: {diff_accuracy*100:.2f}%")
print(f"Precision Difference: {diff_precision*100:.2f}%")
print(f"Recall Difference: {diff_recall*100:.2f}%")
print(f"F1 Score Difference: {diff_f1*100:.2f}%")
