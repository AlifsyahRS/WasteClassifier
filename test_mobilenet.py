import numpy as np
import tensorflow as tf
import os



# Get the directory of the Python file
directory = os.path.dirname(os.path.abspath(__file__))




model_path = directory + '\model\model_mobilenet.h5' # Directory of model
#model_path = directory + '/model/model.h5' # Use this for Unix systems


# Put your image file here
file_path = os.path.join(directory,'cardboard.jpg') # Name of your image file


model = tf.keras.models.load_model(model_path) # Load the model
class_name = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic', 'trash'] # Class names

# Processing the input image
img = tf.keras.preprocessing.image.load_img(file_path, target_size=(256, 256))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

predictions = model.predict(img_array) # Predicting the result using the model

max_index = np.argmax(predictions[0])
print(f"Prediction: {class_name[max_index]} ({max(predictions[0]*100)}%)")