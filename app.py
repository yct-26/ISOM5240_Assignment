# Importing necessary packages
import streamlit as st
from transformers import pipeline

# Setting Headers and Titles for the Application
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🧠")
st.header("Turn Your Image to Audio Story")

# Adding a file uploader to the application
uploaded_file = st.file_uploader("Select an Image...")


# Defining a function to transform image to text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", 
                                   model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# Defining a function to generate a story from text
def text2story(text):
    story_pipe = pipeline("text-generation", 
                            model="pranavpsv/genre-story-generator-v2")
    story_results = story_pipe(text)
    story = story_results[0]['generated_text']
    return story_text

# Defining a function to transform story to speech/audio
def text2audio(story_text):
    audio_pipe = pipeline("text-to-audio", 
                          model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story)
    return audio_data


# Defining a main function to execute all the sub-functions
def main():    
    scenario = img2text(uploaded_file.name)
    st.write(f"**Scenario:** {scenario}")
    story_text = text2story(scenario)
    st.write(f"**Story:** {story}")
    audio_data = text2audio(story_text)


# Execute main() if there is an uploaded file
if uploaded_file is not None:
    main()


# Setting up a Play button
if st.button("Play Audio"):
    audio_array = audio_data["audio"]
    sample_rate = audio_data["sampling_rate"]
    st.audio(audio_array, sample_rate=sample_rate)
   












