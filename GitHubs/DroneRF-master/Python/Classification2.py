#########################################################################
#
# Copyright 2018 Mohammad Al-Sa'd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#########################################################################

#########################################################################
# RF-based drone detection and identification
# Improved training with evaluation metrics
#########################################################################

############################## Libraries ################################
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

############################## Functions ###############################
def decode(datum):
    """Decode one-hot encoded labels back to integers."""
    y = np.zeros((datum.shape[0], 1))
    for i in range(datum.shape[0]):
        y[i] = np.argmax(datum[i])
    return y

def encode(datum):
    """One-hot encode integer labels."""
    return to_categorical(datum)

############################# Parameters ###############################
np.random.seed(1)
K                    = 10
number_epoch         = 200
batch_length         = 32
show_inter_results   = 1

############################### Loading ################################
print("Loading Data ...")
# Use relative path from Python directory
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../../../Data_aggregated/RF_Data.csv")
print(f"Loading from: {data_path}")

if not os.path.exists(data_path):
    print("\nERROR: RF_Data.csv not found!")
    print("You need to run the MATLAB scripts first:")
    print("  1. Main_1_Data_aggregation.m")
    print("  2. Main_2_Data_labeling.m")
    exit(1)

Data = np.loadtxt(data_path, delimiter=",")

############################## Splitting ################################
print("Preparing Data ...")
x = np.transpose(Data[0:2047, :])
Label_3 = np.transpose(Data[2050:2051, :]).astype(int)
y = encode(Label_3)

# Normalize input features for more stable training
scaler = StandardScaler()
x = scaler.fit_transform(x)

################################ Main ###################################
cvscores = []
cnt = 0

# For storing predictions across folds
all_y_true = []
all_y_pred = []

kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)

for train, test in kfold.split(x, decode(y)):
    cnt += 1
    print(f"\nTraining Fold {cnt}/{K}...")

    # Split training into training + validation
    x_train, x_val, y_train, y_val = train_test_split(
        x[train], y[train], test_size=0.1, random_state=1
    )

    # --------------------- OLD MODEL (commented) ---------------------
    # model = Sequential()
    # for i in range(3):
    #     model.add(Dense(128, input_dim=x.shape[1], activation='relu'))
    # model.add(Dense(y.shape[1], activation='sigmoid'))
    # model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    #
    # Issues:
    # - Wrong loss for classification (mse instead of crossentropy).
    # - Wrong output activation (sigmoid instead of softmax for multi-class).
    # - No regularization (risk of overfitting).
    # ----------------------------------------------------------------

    # --------------------- NEW MODEL (improved) ---------------------
    model = Sequential()
    model.add(Dense(512, input_dim=x.shape[1], activation='relu'))
    model.add(Dropout(0.3))                 # regularization: prevents overfitting
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))                 # another dropout layer
    model.add(Dense(y.shape[1], activation='softmax'))  # multi-class output

    model.compile(
        loss='categorical_crossentropy',     # correct for multi-class
        optimizer='adam',                    # adaptive optimizer
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,                        # stop if no improvement for 10 epochs
        restore_best_weights=True
    )

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=number_epoch,
        batch_size=batch_length,
        verbose=show_inter_results,
        callbacks=[early_stop]
    )
    # ----------------------------------------------------------------

    # Evaluate on test set
    scores = model.evaluate(x[test], y[test], verbose=0)
    print(f"Fold {cnt} Accuracy: {scores[1]*100:.2f}%")
    cvscores.append(scores[1]*100)

    # Predictions
    y_pred = model.predict(x[test])
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y[test], axis=1)

    # Collect for confusion matrix later
    all_y_true.extend(y_true_classes)
    all_y_pred.extend(y_pred_classes)

    # Save predictions per fold
    np.savetxt(f"Results_Fold{cnt}.csv", np.column_stack((y_true_classes, y_pred_classes)), delimiter=",", fmt='%s')

# ----------------------------------------------------------------------
# Final evaluation across all folds
print(f"\nAverage Accuracy over {K} folds: {np.mean(cvscores):.2f}%")

# Confusion matrix
cm = confusion_matrix(all_y_true, all_y_pred)
labels = np.unique(all_y_true)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix across all folds")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(all_y_true, all_y_pred))

#########################################################################
# Key Explanations:
#
# 1. Softmax vs Sigmoid:
#    - Sigmoid is for binary classification (yes/no, 0/1).
#    - Softmax is for multi-class problems (e.g., multiple drone types).
#      It outputs probabilities that sum to 1 across all classes.
#
# 2. Loss Function:
#    - MSE (mean squared error) is not suitable for classification.
#    - categorical_crossentropy is designed for multi-class classification
#      and works with softmax output.
#
# 3. Layers and Neurons:
#    - More neurons (512, 256) allow the network to learn complex patterns.
#    - Too many neurons can overfit, so dropout is added.
#
# 4. Dropout:
#    - Randomly "drops" neurons during training.
#    - Forces the network not to rely on specific connections, improving generalization.
#
# 5. Early Stopping:
#    - Prevents overfitting by stopping training when validation loss stops improving.
#
# 6. Confusion Matrix:
#    - Shows per-class performance.
#    - Diagonal = correct predictions.
#    - Off-diagonal = misclassifications (which classes get confused).
#
# 7. Classification Report:
#    - Precision: Out of predicted positives, how many were correct.
#    - Recall: Out of actual positives, how many were detected.
#    - F1-score: Balance of precision and recall.
#
# Together, these metrics give a clearer picture than accuracy alone,
# especially if some classes have more samples than others.
#########################################################################


# Dense layers (fully connected): Every input is connected to every output with weights.

# Activation functions:

# relu (rectified linear unit) in hidden layers introduces non-linearity and avoids vanishing gradients.

# softmax in the output layer ensures outputs are valid probabilities for multi-class.

# Layer sizes:

# Larger hidden layers (512 → 256) increase model capacity.

# Stacking multiple layers allows learning hierarchical features.



# If this don't work we can do CNN based model.