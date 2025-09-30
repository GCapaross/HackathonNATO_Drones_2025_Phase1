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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Mohammad F. Al-Sa'd (mohammad.al-sad@tut.fi)
#          Amr Mohamed         (amrm@qu.edu.qa)
#          Abdulla Al-Ali
#          Tamer Khattab
#
# The following reference should be cited whenever this script is used:
#     M. Al-Sa'd et al. "RF-based drone detection and identification using
#     deep learning approaches: an initiative towards a large open source
#     drone database", 2018.
#
# Last Modification: 19-11-2018
#########################################################################




# Altered code

############################## Libraries ################################
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split

############################## Functions ###############################
def decode(datum):
    y = np.zeros((datum.shape[0],1))
    for i in range(datum.shape[0]):
        y[i] = np.argmax(datum[i])
    return y

def encode(datum):
    return to_categorical(datum)

############################# Parameters ###############################
np.random.seed(1)
K                    = 10
inner_activation_fun = 'relu'
outer_activation_fun = 'sigmoid'
optimizer_loss_fun   = 'mse'
optimizer_algorithm  = 'adam'
number_inner_layers  = 3
number_inner_neurons = 256
number_epoch         = 200
batch_length         = 10
show_inter_results   = 1
early_stop_patience  = 10  # Stop if no improvement in 10 epochs

############################### Loading ##################################
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

############################## Splitting #################################
print("Preparing Data ...")
x = np.transpose(Data[0:2047,:])
Label_1 = np.transpose(Data[2048:2049,:]).astype(int)
Label_2 = np.transpose(Data[2049:2050,:]).astype(int)
Label_3 = np.transpose(Data[2050:2051,:]).astype(int)
y = encode(Label_3)

################################ Main ####################################
cvscores = []
cnt = 0

kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=1)

for train_idx, test_idx in kfold.split(x, decode(y)):
    cnt += 1
    print(f"\nTraining Fold {cnt}/{K}...")

    # Split off a small validation set from training for early stopping
    x_train, x_val, y_train, y_val = train_test_split(
        x[train_idx], y[train_idx], test_size=0.1, random_state=1
    )

    model = Sequential()
    model.add(Dense(number_inner_neurons, input_dim=x.shape[1], activation=inner_activation_fun))
    for _ in range(number_inner_layers - 1):
        model.add(Dense(number_inner_neurons, activation=inner_activation_fun))
    model.add(Dense(y.shape[1], activation=outer_activation_fun))

    model.compile(loss=optimizer_loss_fun, optimizer=optimizer_algorithm, metrics=['accuracy'])

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=early_stop_patience,
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

    scores = model.evaluate(x[test_idx], y[test_idx], verbose=show_inter_results)
    print(f"Fold {cnt} Accuracy: {scores[1]*100:.2f}%")
    cvscores.append(scores[1]*100)

    y_pred = model.predict(x[test_idx])
    np.savetxt(f"Results_3{cnt}.csv", np.column_stack((y[test_idx], y_pred)), delimiter=",", fmt='%s')

print(f"\nAverage Accuracy over {K} folds: {np.mean(cvscores):.2f}%")

