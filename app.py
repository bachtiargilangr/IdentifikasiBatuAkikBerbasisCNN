import os
from flask import Flask, render_template, request
from flask import send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
#import cv2
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape=(64,64,3), activation="relu"))
model.add(Conv2D(16, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3,3), activation="relu"))
model.add(Conv2D(32, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(7, activation="softmax"))


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
   
model = load_model('percobaan1.h5')
indices = {0: 'Batu Kali', 1: 'Badar Besi', 2: 'Batu lumut', 3: 'Giok Hijau', 4: 'Giok Biru', 5: 'Giok Merah', 6: 'Kecubung Ungu'}

# call model to predict an image
def predict_label(full_path):
    print(0)
    data =  tf.compat.v1.keras.preprocessing.image.load_img(full_path, target_size=(64, 64, 3))
    print(1)
    data = np.expand_dims(data, axis=0)
    print(2)
    data = data * 1.0 / 255
    print(3)
    predicted = model.predict(data)
    print(4)
    return predicted

@app.route("/")
def main():
	return render_template('index.html')


@app.route("/identifikasi")
def identifikasi():
	return render_template('identifikasi.html')

#proses upload dan identifikasi batuan
@app.route('/upload', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('identifikasi.html')
    else:
        file = request.files['image']
        full_name = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        print('C', full_name)
        file.save(full_name)
        print('A')
        result = predict_label(full_name)
        print('B')
        predicted_class = np.asscalar(np.argmax(result, axis=1))
        accuracy = round(result[0][predicted_class] * 100, 2)
        label = indices[predicted_class]
    return render_template('identifikasi.html', uploaded_image = file.filename, prediction = label, accuracy = accuracy)

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__=="__main__":
	app.run(debug=True)
