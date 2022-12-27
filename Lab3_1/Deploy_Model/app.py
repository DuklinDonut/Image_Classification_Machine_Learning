#All the import statements goes here:
import os
import numpy as np
from PIL import Image
import tensorflow as tf

from resizeimage import resizeimage
from flask import Flask, flash, request, make_response, render_template, send_from_directory

UPLOAD_FOLDER = 'static/uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        name= file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], name))

        image_file = Image.open("static/uploads/{}".format(name)) # open colour image
        image_file = image_file.convert('1') # convert image to black and white
        image_file.save('static/uploads/{}'.format(name))

        with open('static/uploads/{}'.format(name), 'r+b') as f:
            with Image.open(f) as image:
                cover = resizeimage.resize_cover(image, [28, 28])
                cover.save('static/uploads/{}'.format(name), image.format)


        img = Image.open('static/uploads/{}'.format(name))
        array = np.array(img)

        new_model = tf.keras.models.load_model("static/model/FatiModel1.h5")

        pred = new_model.predict(np.array([array]))

        pred = np.argmax(pred)

        class_names = ['T-shirt/top', 'Trouser', 'Sneaker', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        val = class_names[pred]

        return render_template('predict.html', image_file_name=file.filename, val=val)
    else:
        return render_template("index.html")

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True,
            use_reloader=False
            )
#debug=True