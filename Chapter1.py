import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

#Holds the dense layer
l0 = Dense(units=1, input_shape=[1])
#Defining model strucutre
model = Sequential([l0])
#Compile Model with sgd and mean squared error for loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

#Defining data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#training the data
model.fit(xs, ys, epochs=500)
#Prediction from the model
print(model.predict([10.0]))
#Print the weights in which you learned
print("Here is what I learned: {}".format(l0.get_weights()))