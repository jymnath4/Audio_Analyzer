import streamlit as st
import assemblyai as aai
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from PIL import Image
#import torch

image = Image.open(r'ggs.jpg')
st.image(image, width=150)
st.title('CALL RECORDING ANALYZER')

Audiofile = st.text_input("PLEASE PROVIDE CALL RECORDING LINK & PRESS ENTER")
st.write("URL:", Audiofile)

aai.settings.api_key = "f9e702924bb4481d997cdc25834eb040"

FILE_URL = Audiofile

# adding audio button for hear conversation
button = st.button('Play Call Recording', key=None, help=None, on_click=None, args=None, kwargs=None, disabled=False)

# if click on audio button audio controls and playback
if button:
    st.audio(FILE_URL, format='audio/mp3', start_time=0)

#volume = st.slider("Volume", 0.0, 1.0, 0.5, step=0.1)

#st.audio(FILE_URL, format='audio/mp3', start_time=0, volume=volume)

#existing_tensor = torch.randn(1, 514)  # Example existing tensor
#target_size = (1, 3470)

# Reshape the existing tensor to match the target size
#resized_tensor = existing_tensor.view(*target_size)


transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_URL)
t1 = transcript.text
st.write(t1)

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

encoded_text = tokenizer(t1, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'Negative' : scores[0],
    'Neutral' : scores[1],
    'Positive' : scores[2]
}

streamlit.markdown("SENTIMENT SCORES:")
st.write(scores_dict)