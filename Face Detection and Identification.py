#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import package
from matplotlib import pyplot
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from numpy import load
from numpy import expand_dims
from numpy import asarray
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib.patches import Rectangle
import os

# Face Detection

# defining function to take images and get face data and convert into arrays to process
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array
# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face      
        face = extract_face(path)
        # store
        faces.append(face)
    return faces
# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

trainX, trainy = load_dataset('/storage/try/dataset1/')
print(trainX.shape, trainy.shape)
#Saving compressed numpy error in a single file for later use
savez_compressed('face_recognition_data.npz', trainX, trainy)
>loaded 118 examples for class: msd
>loaded 117 examples for class: virat
>loaded 114 examples for class: jaddu
>loaded 115 examples for class: umesh
>loaded 111 examples for class: shikhar
>loaded 110 examples for class: ashwin
(685, 160, 160, 3) (685,)
In [4]:
# loading array of facial data
data = load('face_recognition_data.npz')
trainX, trainy = data['arr_0'], data['arr_1']
print('Loaded: ', trainX.shape, trainy.shape)
Loaded:  (685, 160, 160, 3) (685,)

## this function will take the training images array and convert into embeddings using facenet weights
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global) - this is a pre-processing step that is required before embedding
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

import os
os.chdir('/storage')

#loading keras weights
model = load_model('facenet_keras.h5')
print('Loaded Model')
Loaded Model
/usr/local/lib/python3.6/dist-packages/keras/engine/saving.py:341: UserWarning: No training configuration found in save file: the model was *not* compiled. Compile it manually.
  warnings.warn('No training configuration found in save file: '

newTrainX = list()
for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
newTrainX = asarray(newTrainX)
print(newTrainX.shape)
(685, 128)

## Saving the embeddings which can be used in our SVM layer later
savez_compressed('face_recognition.npz', newTrainX, trainy)

data = load('face_recognition.npz')
trainX, trainy = data['arr_0'], data['arr_1']
print('Dataset: train=%d' % (trainX.shape[0]))
Dataset: train=685

# normalizing training data
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)

# encoding the target variable
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)

# creating SVM model and fitting
model1 = SVC(kernel='linear',probability = True)

model1.fit(trainX, trainy)

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=True, random_state=None,
    shrinking=True, tol=0.001, verbose=False)

 #Testing¶

os.chdir('/storage/test_image/test/')

# Taking face features of test image, creating embedding and predicting with SVM.
import numpy as np
pyplot.figure(figsize=(120,120))
test = pyplot.imread('indianteam_1.jpg')
required_size = (160,160)
pyplot.imshow(test)
ax = pyplot.gca()
detector = MTCNN()
faces = detector.detect_faces(test)
ax = pyplot.gca()
z = 1
for face in faces:
    x,y,width,height = face['box']
    x2,y2 = x+width,y+height
    face = test[y:y2, x:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    #print(face_array.shape)
    
    embedding = get_embedding(model, face_array)
    #print(embedding.shape)
    embedding = np.reshape(embedding, (1,-1))
    #print(embedding.shape)
    in_encoder = Normalizer(norm='l2')
    
    testX = in_encoder.transform(embedding)
    model1.probability = True
    yhat_test = model1.predict(testX)
    yhat_prob = model1.predict_proba(testX)
    class_index = yhat_test
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_test)
    if (class_probability < 90):
        predict_names = 'Unknown'
    rect = Rectangle((x,y),width,height,color = 'Green',fill = False,visible = True,linewidth = 20)
    ax.add_patch(rect)
    ax.text(0.48*(x+(x+width)), 0.58*(y+(y+height)),predict_names,fontsize=120, color='Black')
    
pyplot.show()
