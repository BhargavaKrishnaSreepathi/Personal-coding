import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.core import Dropout
from keras import optimizers
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import model_from_json



payment_data = pd.read_csv(r'C:\Users\krish\Desktop\AI Lab - Technical Interview\AI Lab - Technical Interview\Anomaly Detection\payment_data_ratio20.csv')

# payment_data.info()
# payment_data.describe()
# payment_data.head(10)


customer_data = pd.read_csv(r'C:\Users\krish\Desktop\AI Lab - Technical Interview\AI Lab - Technical Interview\Anomaly Detection\customer_data_ratio20.csv')
# customer_data.info()

credit = customer_data.drop(["fea_2", "fea_4", "fea_8", "fea_10", "fea_11", "id"],axis=1)

# credit = customer_data.drop(["fea_2",  "id"],axis=1)


scaler = MinMaxScaler()
scaler.fit_transform(credit)

train, test = train_test_split(credit, test_size = 0.1)

test_features = credit.drop('label', axis=1)
test_targets = credit['label']

x_train = train.drop('label', axis=1)

y_train = train['label']

x_test =  test.drop('label', axis=1)

y_test = test['label']

sm = SMOTE(random_state=12, ratio = 1.0)  # using SMOTE to oversample
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

model = Sequential()
sgd = optimizers.SGD(lr=0.01, decay=1e-6)
model.add(Dense(256, input_dim=(6), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))

model.add(Dense(1,  activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer= sgd, metrics=['accuracy'])

checkpoint = ModelCheckpoint("best_weights.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=200, verbose=1, mode='auto')

model.fit(np.array(x_train_res), np.array(y_train_res), validation_data=(np.array(x_test), np.array(y_test)), epochs=500, verbose=1, callbacks = [checkpoint, early])

model.evaluate(np.array(x_test), np.array(y_test))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_weights.h5")
print("Saved model to disk")

json_file = open(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning\OCBC\model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
predictive_model = model_from_json(loaded_model_json)
# load weights into new model
predictive_model.load_weights(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning\OCBC\best_weights.h5")

y_pred=predictive_model.predict(np.array(x_test))

precision_score(y_test, y_pred.round(), average='weighted')
precision=precision_score(y_test, y_pred.round(), average='weighted')
print ('precision')
print (precision)

recall=recall_score(y_test, y_pred.round(), average='weighted')
print ('recall')
print (recall)
F1 = 2 * (precision * recall) / (precision + recall)
print ('F1 score')
print (F1)

