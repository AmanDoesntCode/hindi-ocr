# from keras.preprocessing.image import img_to_array
# import numpy as np
# import cv2

# ''' ["ka","kha","ga","gha","kna","cha","chha","ja","jha","yna","t`a","t`ha","d`a","d`ha","adna","ta","tha","da","dha","na","pa","pha","ba","bha","ma","yaw","ra","la","waw","sha","shat","sa","ha","aksha","tra","gya","0","1","2","3","4","5","6","7","8","9"]
# labels =['yna', 't`aa', 't`haa', 'd`aa', 'd`haa', 'a`dna', 'ta', 'tha', 'da', 'dha', 'ka', 'na', 'pa', 'pha', 'ba', 'bha', 'ma', 'yaw', 'ra', 'la', 'waw', 'kha', 'sha', 'shat', 'sa', 'ha', 'aksha', 'tra', 'gya', 'ga', 'gha', 'kna', 'cha', 'chha', 'ja', 'jha', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# '''
# labels = [u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939','ksha','tra','gya',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f']
# #
# import numpy as np
# from keras.preprocessing import image
# test_image = cv2.imread("2512.png")
# image = cv2.resize(test_image, (32,32))
# image = image.astype("float") / 255.0
# image = img_to_array(image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = np.expand_dims(image, axis=0)
# image = np.expand_dims(image, axis=3)
# print("[INFO] loading network...")
# import tensorflow as tf
# model = tf.keras.models.load_model("HindiModel2.h5")
# lists = model.predict(image)[0]
# print("The letter is ",labels[np.argmax(lists)])

import streamlit as st
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("HiM2.h5")

# Define your labels
labels = [u'\u091E',u'\u091F',u'\u0920',u'\u0921',u'\u0922',u'\u0923',u'\u0924',u'\u0925',u'\u0926',u'\u0927',u'\u0915',u'\u0928',u'\u092A',u'\u092B',u'\u092c',u'\u092d',u'\u092e',u'\u092f',u'\u0930',u'\u0932',u'\u0935',u'\u0916',u'\u0936',u'\u0937',u'\u0938',u'\u0939','ksha','tra','gya',u'\u0917',u'\u0918',u'\u0919',u'\u091a',u'\u091b',u'\u091c',u'\u091d',u'\u0966',u'\u0967',u'\u0968',u'\u0969',u'\u096a',u'\u096b',u'\u096c',u'\u096d',u'\u096e',u'\u096f']

def predict_letter(image):
    image = cv2.resize(image, (32,32))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)
    prediction = model.predict(image)[0]
    return labels[np.argmax(prediction)]

st.title("Hindi Letter Predictor")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if len(image.shape) != 2 :
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Predicted Letter: ", predict_letter(image))
