import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

model = load_model('plant_disease_model.h5')

CLASS_NAMES = ["bacterial_leaf_blight","bacterial_leaf_streak","bacterial_panicle_blight","blast","brown_spot",
               "dead_heart","downy_mildew","hispa","normal","tungro"]

st.title("🌾Paddy Disease identification app🌾")
st.markdown("Choose an option:")

option = st.radio('', ('Upload Image from device 📁', 'Take Photo from camera 📷'))

if option == 'Upload Image from device 📁':
    plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    submit = st.button("Predict Disease")

    if submit:
        if plant_image is not None:
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes,   1)

            st.image(opencv_image, channels="BGR")
            st.write(opencv_image.shape)
            opencv_image = cv2.resize(opencv_image, (256, 256))
            opencv_image = opencv_image.reshape((1, 256, 256, 3))

            y_pred = model.predict(opencv_image)
            result = CLASS_NAMES[np.argmax(y_pred)]
            st.title("Predicted Disease: " + result)
        else:
            st.warning("Please upload an image.")

elif option == 'Take Photo from camera 📷':
    if st.button("Capture Photo"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="BGR")
        st.write(frame.shape)

        frame = cv2.resize(frame, (256, 256))
        frame = frame.reshape((1, 256, 256, 3))

        y_pred = model.predict(frame)
        result = CLASS_NAMES[np.argmax(y_pred)]
        st.title("Predicted Disease: " + result)
