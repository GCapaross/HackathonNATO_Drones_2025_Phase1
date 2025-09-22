import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# TODO: Do it for all folds, not seperate files

# Load CSV
df = pd.read_csv("Results_Fold8.csv", header=None, names=["predicted", "true"])

# Overall accuracy
accuracy = accuracy_score(df["true"], df["predicted"])
print(f"Overall Accuracy: {accuracy*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(df["true"], df["predicted"])
num_classes = len(np.unique(df[["true", "predicted"]].values))
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(num_classes),
            yticklabels=range(num_classes))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Per-class accuracy
per_class_acc = cm.diagonal() / cm.sum(axis=1)
for i, acc in enumerate(per_class_acc):
    print(f"Class {i} Accuracy: {acc*100:.2f}%")



# TODO: Uncomment for the other type of file, taht isn't just (int, int) but (int, float, float, float, ...)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, accuracy_score

# # Load probabilities
# probs = pd.read_csv("results.csv", header=None).values  # shape: (num_samples, num_classes)

# # Predicted class = index of max probability
# predicted = np.argmax(probs, axis=1)

# # If you have the true labels in another CSV or array
# true = pd.read_csv("true_labels.csv", header=None).values.flatten()  # shape: (num_samples,)

# # Accuracy
# accuracy = accuracy_score(true, predicted)
# print(f"Overall Accuracy: {accuracy*100:.2f}%")

# # Confusion matrix
# cm = confusion_matrix(true, predicted)
# plt.figure(figsize=(10,8))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()
