import matplotlib.pyplot as plt

# Supponiamo di avere questi dati da ogni epoca di training:
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Esempio di epoche
train_accuracy = [0.933, 0.933, 1.00, 0.867, 1.00, 0.80, 0.933, 1.00, 0.933, 1.00]  # Esempio di accuratezza del training
val_accuracy = [0.918, 0.916, 0.933, 0.928, 0.937, 0.934, 0.945, 0.941, 0.948, 0.939]  # Esempio di accuratezza di validazione

train_loss = [0.25, 0.259, 0.0858, 0.34, 0.0543, 0.348, 0.12, 0.0209, 0.0778, 0.0398]  # Esempio di loss del training
val_loss = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Esempio di loss di validazione

# Plotting accuratezza
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, label="Train Accuracy")
plt.plot(epochs, val_accuracy, label="Validation Accuracy")
plt.title("Accuracy during Training")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.title("Loss during Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("train_val_plot.png")
