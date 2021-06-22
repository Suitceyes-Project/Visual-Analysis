import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from data import *
from tensorflow.keras.models import load_model

# Add directory holding utility functions to path to allow importing utility functions

sess = tf.Session()

#Path to scene model
_SCL_PATH_TO_MODEL = '/Path_to_location/scene_model.pb'

def preprocess_image(img_np):
    img = tf.keras.preprocessing.image.array_to_img(img_np)
    
    img = img.resize(size=(224,224))
    x = tf.keras.preprocessing.image.img_to_array(img)
   
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')
    return x
   
   
with tf.variable_scope('keras_graph'):
    scene_model = load_model(_SCL_PATH_TO_MODEL)
scene_model._make_predict_function()
print("Done")
def process_scene(image_np):
    
    #LOAD IMAGE
    height = image_np.shape[0]
    width = image_np.shape[1]
    
    #Predict the scene class
    scene_preds = scene_model.predict_on_batch(preprocess_image(image_np))
    scene_class_pred = np.argmax(scene_preds[0])
    scene_class_score = scene_preds[0][scene_class_pred]
    print (scene_class_pred, cls_sc[scene_class_pred], scene_class_score)
    #Threshold to keep scenes only with high confidence
    if scene_class_score > 0.2:
        scene_ret = cls_sc[scene_class_pred]
        fin_score = scene_class_score
    else:
        scene_ret = "scene_unknown"
        fin_score = 0
    return scene_ret,fin_score




