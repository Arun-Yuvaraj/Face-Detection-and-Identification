#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import package
from matplotlib import pyplot
import cv2
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from numpy import expand_dims
from PIL import Image
from numpy import asarray


# In[10]:


# using MTCNN model to fetch face in a image
pyplot.figure(figsize=(20,5))
test = pyplot.imread('108128.jpg')
pyplot.imshow(test)
detector = MTCNN()
faces = detector.detect_faces(test)
x1, y1, width, height = faces[0]['box']
x2, y2 = x1 + width, y1 + height 
face = test[y1:y2, x1:x2]
image = Image.fromarray(face) 
req_size = (224,224)
image = image.resize(req_size) 
face_array = asarray(image)
pyplot.imshow(test[y1:y2,x1:x2])


# ## Face Identification

# In[18]:


model = VGGFace(model='vgg16') 


# In[19]:


# using VGG16 Model to identify the face.
face_array = face_array.astype('float32')
samples = expand_dims(face_array, axis=0) 
samples = preprocess_input(samples, version=2) 
yhat = model.predict(samples)
results = decode_predictions(yhat) 
                    


# In[20]:


results[0]


# In[44]:





# In[ ]:




