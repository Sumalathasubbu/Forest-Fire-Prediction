
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
# converting to is weekend
df['day'] = ((df['day'] == 'sun') | (df['day'] == 'sat'))
# renaming column
df = df.rename(columns = {'day' : 'is_weekend'})
# visualizing
sns.countplot(df['is_weekend'])
plt.title('Count plot of weekend vs weekday')
# fit a model and plot learning curve
def fit_model(trainX, trainy, validX, validy, n_epoch):
# define model
model = Sequential()
model.add(Dense(6, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(6, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'relu'))
# compile model
model.compile(optimizer ='adam', metrics=['accuracy'], loss = 'binary_crossentropy')
# fit model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=n_epoch, verbose=0, batch_size=6)
# plot learning curves
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('epoch='+str(n_epoch))
plt.legend()
# Create learning curves for different batch sizes
epochs = [20, 50, 100, 120, 150, 200, 300, 400]
plt.figure(figsize=(10,15))
for i in range(len(batch_sizes)):
# Determine the Plot Number
plot_no = 420 + (i+1)
plt.subplot(plot_no)
# Fit model and plot learning curves for a batch size
fit_model(X_train, y_train, X_test, y_test, epochs[i])
# Show learning curves
plt.show()
