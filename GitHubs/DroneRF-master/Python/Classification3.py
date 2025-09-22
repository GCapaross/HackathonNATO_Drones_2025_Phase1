# For differnt classification tasks (opt 1, 2, 3)
#########################################################################

import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split

def decode(datum):
    y = np.zeros((datum.shape[0],1))
    for i in range(datum.shape[0]):
        y[i] = np.argmax(datum[i])
    return y

def encode(datum):
    return to_categorical(datum)

# ----------------- Parameters -----------------
np.random.seed(1)
K = 10
inner_activation_fun = 'relu'
outer_activation_fun = 'sigmoid'
optimizer_loss_fun   = 'mse'
optimizer_algorithm  = 'adam'
number_inner_layers  = 3
number_inner_neurons = 256
number_epoch         = 200
batch_length         = 10
show_inter_results   = 1
early_stop_patience  = 10

# ----------------- Load Data -----------------
Data = np.loadtxt("G:/Programing/HackathonNATO_Drones_2025/Data_aggregated/RF_Data.csv", delimiter=",")
x = np.transpose(Data[0:2047,:])
Label_1 = np.transpose(Data[2048:2049,:]).astype(int)  # Drone presence
Label_2 = np.transpose(Data[2049:2050,:]).astype(int)  # Drone type
Label_3 = np.transpose(Data[2050:2051,:]).astype(int)  # Mode/type (opt3)

# ----------------- Function to train & save -----------------
def train_and_save(y_labels, opt):
    y = encode(y_labels)
    cvscores = []
    cnt = 0
    kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)
    
    for train_idx, test_idx in kfold.split(x, decode(y)):
        cnt += 1
        print(f"\nTraining Fold {cnt}/{K} for opt {opt} ...")

        x_train, x_val, y_train, y_val = train_test_split(
            x[train_idx], y[train_idx], test_size=0.1, random_state=1
        )

        model = Sequential()
        model.add(Dense(number_inner_neurons, input_dim=x.shape[1], activation=inner_activation_fun))
        for _ in range(number_inner_layers - 1):
            model.add(Dense(number_inner_neurons, activation=inner_activation_fun))
        model.add(Dense(y.shape[1], activation=outer_activation_fun))

        model.compile(loss=optimizer_loss_fun, optimizer=optimizer_algorithm, metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)

        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=number_epoch,
            batch_size=batch_length,
            verbose=show_inter_results,
            callbacks=[early_stop]
        )

        scores = model.evaluate(x[test_idx], y[test_idx], verbose=0)
        print(f"Fold {cnt} Accuracy: {scores[1]*100:.2f}%")
        cvscores.append(scores[1]*100)

        y_pred = model.predict(x[test_idx])
        # Save CSV in the correct format for MATLAB
        np.savetxt(f"Results_{opt}{cnt}.csv", np.column_stack((y[test_idx], y_pred)), delimiter=",", fmt='%s')

    print(f"\nAverage Accuracy over {K} folds for opt {opt}: {np.mean(cvscores):.2f}%\n")

# ----------------- Train for opt 1 -----------------
train_and_save(Label_1, opt=1)

# ----------------- Train for opt 2 -----------------
# For opt 2, concatenate Label_1 and Label_2 horizontally
Label_12 = np.hstack((Label_1, Label_2))
train_and_save(Label_12, opt=2)

# ----------------- Train for opt 3 -----------------
train_and_save(Label_3, opt=3)
