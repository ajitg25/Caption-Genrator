import streamlit as st
from PIL import Image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pickle import load
from keras.models import Model, load_model
from keras.applications.xception import Xception #to get pre-trained model Xception
from keras_preprocessing.sequence import pad_sequences

st.title("Welcome to Caption Generator (LISTED)")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

def extract_features(filename, model):
    try:
        # image = Image.open(filename)
        image = filename
    except:
        print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
    image = image.resize((299,299))
    image = np.array(image)
    # for images that has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4: 
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    in_text = in_text.split(" ")
    in_text = in_text[1:-1]
    in_text =" ".join(in_text)
    return in_text



if uploaded_file is not None:
    img_path = Image.open(uploaded_file)
    st.image(img_path, caption='Uploaded Image')

    # img_path = "D:\college\PY\ds\project\CaptionGenerator\hack.jpg"
    max_length = 32
    tokenizer = load(open("tokenizer.p","rb"))
    model = load_model('models/model_9.h5')
    xception_model = Xception(include_top=False, pooling="avg")

    photo = extract_features(img_path, xception_model)
    # img = Image.open(img_path)
    img = img_path
    description = generate_desc(model, tokenizer, photo, max_length)
    print("\n\n")
    print(description)
    st.markdown("Suggested Caption : ")
    st.header(description)
    
    plt.imshow(img)







