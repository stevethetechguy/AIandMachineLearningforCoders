# import tensorflow as tf

# data = tf.keras.datasets.fashion_mnist

# (training_images, training_labels), (test_images, test_labels) = data.load_data()

# training_images = training_images.reshape(60000, 28, 28, 1)
# training_images = training_images / 255.0
# test_images = test_images.reshape(10000, 28, 28, 1)
# test_images = test_images / 255.0

# model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2,2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation=tf.nn.relu),
#         tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

# model.compile(
#             optimizer='adam', 
#             loss='sparse_categorical_crossentropy', 
#             metrics=['accuracy'])

# model.fit(training_images, training_labels, epochs=5)

# model.evaluate(test_images, test_labels)

# classifications = model.predict(test_images)
# print(classifications[0])
# print(test_labels[0])

# model.summary()

# import urllib.request
# import zipfile

# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# print('step 1')
# url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
# print('step 2')
# file_name = "horse-or-human.zip"
# training_dir = 'horse-or-human/training/'
# print('step 3')
# urllib.request.urlretrieve(url, file_name)
# print('step 4')
# zip_ref = zipfile.ZipFile(file_name, 'r')
# zip_ref.extractall(training_dir)
# zip_ref.close()
import tensorflow as tf
import urllib.request
import zipfile
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


print('Step 1: Define URL and File Name')
url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"
validationurl = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"

file_name = "horse-or-human.zip"
validation_file_name = "validation-horse-or-human.zip"
training_dir = 'horse-or-human/training/'
validation_dir = 'horse-or-human/validaton/'
urllib.request.urlretrieve(url,file_name)
urllib.request.urlretrieve(validationurl, validation_file_name)

zip_ref = zipfile.ZipFile(file_name, 'r')
zip_ref.extractall(training_dir)
zip_ref.close()

zip_ref = zipfile.ZipFile(validation_file_name, 'r')
zip_ref.extractall(validation_dir)
zip_ref.close()
#All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode='binary'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                           input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

history = model.fit_generator(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)

model.summary()