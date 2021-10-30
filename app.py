import numpy as np
import os
import imutils
import time
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import tensorflow as tf                
from tqdm import tqdm
from keras import backend as K
import streamlit as st
import cv2 as cv
import tempfile
import pickle
import string
import numpy as np

st.markdown("<h1 style='text-align: center; color: White;background-color:teal'>CMR Project</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: teal;'>Drop in the video below</h3>", unsafe_allow_html=True)
# st.sidebar.header("What is this Project about?")
# st.sidebar.text("Lorem Ipsum Text.")
# st.sidebar.header("What tools where used to make this?")
# st.sidebar.text("Lorem Ipsum Text.")
f = st.file_uploader("Upload file")
tfile = tempfile.NamedTemporaryFile(delete=False) 
if f is not None:
    tfile.write(f.read())


vf = cv.VideoCapture(tfile.name)


if(os.path.exists("abnormal.avi")):
        os.remove("abnormal.avi")
if(os.path.exists("normal.avi")):
    os.remove("normal.avi")

stframe = st.empty()
ret, frame = vf.read()

if st.button('Predict'):
# if frame is read correctly ret is True
    #vf = cv.VideoCapture('cancer.avi')
    class_names = ['Abnormal', 'Normal', 'Uninformative']
    class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
    nb_classes = len(class_names)
    IMAGE_SIZE = (256, 256)
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    simpmodel = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (256, 256, 3),padding='same'), 
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu',padding='same'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(3, activation=tf.nn.softmax)
    ])
    #Change location/name of the model below
    simpmodel.load_weights("model/simpmodel.h5")

    output1 = 'normal.avi'
    output2 = 'abnormal.avi'
    # codec ='avc1'
    codec='mp4v'

    fps = int(vf.get(5))
    wi =int(vf.get(3) )  # float `width`
    he = int(vf.get(4) )
    print(fps,wi,he)
    time.sleep(2.0)
    fourcc = cv.VideoWriter_fourcc(*codec)
    writer1 = None
    writer2 = None
    (h, w) = (None, None)
    zeros = None
    curframe=1
    if writer1 is None:
        (h, w) = (he,wi) #frame.shape[:2]
        writer1 = cv.VideoWriter(output1, fourcc,fps,(wi , he),True)    	
        zeros = np.zeros((h, w), dtype="uint8")
    if writer2 is None:
        (h, w) = (he,wi) #frame.shape[:2]    	
        writer2 = cv.VideoWriter(output2, fourcc,fps,(wi , he),True)
        zeros = np.zeros((h, w), dtype="uint8")
        # output1 = np.zeros((h , w , 3), dtype="uint8")
        # output1[0:h, 0:w] = frame
        # writer.write(output1)
    while True:
        ret, frame = vf.read()
        if ret==True:
            frame = imutils.resize(frame, width=wi)
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image = cv.resize(image, IMAGE_SIZE)
            image = np.array(image, dtype = 'float32')
            image=image.reshape(1,IMAGE_SIZE[0],IMAGE_SIZE[1],3)
            predictions = simpmodel.predict(image)     # Vector of probabilities
            pred_labels = np.argmax(predictions, axis = 1) 
            print(pred_labels)# We take the highest probability
            
        # check if the writer is None
        else:
            break
            #writer2 = cv.VideoWriter(output2, fourcc, fps, (w , h) , True)
        
        if(pred_labels==1):
            print("frame is normal=", curframe)
            output1 = np.zeros((h , w , 3), dtype="uint8")
            output1[0:h, 0:w] = frame
            writer1.write(output1)
        elif(pred_labels==0):
            print("frame is abnormal=", curframe)
            output2 = np.zeros((h , w , 3), dtype="uint8")
            output2[0:h, 0:w] = frame
            writer2.write(output2)

        curframe+=1
if not ret:
    if(os.path.exists("normal.avi")):
        video_normal = open('normal.avi', 'rb')
        video_bytes_normal = video_normal.read()
        st.text('Normal:')
        st.video(video_bytes_normal)
    if(os.path.exists("abnormal.avi")):
        video_abnormal = open('abnormal.avi', 'rb')
        video_bytes_abnormal = video_abnormal.read()
        st.text('Abnormal:')
        st.video(video_bytes_abnormal)
    



cv.destroyAllWindows()
vf.release()
