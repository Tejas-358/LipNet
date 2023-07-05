import streamlit as st
import os 
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
import numpy as np

st.set_page_config(layout = 'wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipsBuddy')
    st.info('This application is originally developed from LipNet deep learning model')


st.title('Lipnet Full Stack App')

options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

col1, col2 = st.columns(2)

if options:
    with col1:
        st.info('The video below displays the converted video in mp4 format')

        ffmpeg_path = r"D:\ffmpeg\ffmpeg-2023-07-02-git-50f34172e0-full_build\bin\ffmpeg"
        file_path = os.path.join('..', 'data', 's1', selected_video)
        command = f'{ffmpeg_path} -i "{file_path}" -vcodec libx264 test_video.mp4 -y'
        os.system(command)    

        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info('This is the preprocess GIF ML model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))

        video1 = video.numpy()
        video_reshaped = np.reshape(video1, (75, 46, 140))
        video_normalized = (video_reshaped * 255).astype(np.uint8)
        imageio.mimsave('animation.gif', video_normalized, duration = 100)
        st.image('animation.gif', width = 400)

        st.info('These is the output tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy = True)[0][0].numpy()
        st.text(decoder)

        st.info('Decode tokens')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)


        