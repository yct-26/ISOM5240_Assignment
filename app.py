# Importing necessary packages
import streamlit as st
from transformers import pipeline

# Setting Headers and Titles for the Application
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🧠")
st.header("Turn Your Image to Audio Story")

# Adding a file uploader to the application
uploaded_file = st.file_uploader("Select an Image...")


# Defining a function to transform image to text (caption)
def img2text(url):
    image_to_text_model = pipeline("image-to-text", 
                                   model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# Defining a function to generate a story from text
def text2story(text):
    story_pipe = pipeline("text-generation", 
                            model="pranavpsv/genre-story-generator-v2")

    prompt = f"Write a story about {text}. The story should be suitable for children between the ages of 3-10."
    story_results = story_pipe(prompt, 
                               min_new_tokens = 60,
                               max_new_tokens = 120,
                               temperature = 0.8)        # Temperature set to 0.8 to make the story more focused on caption.
    story = story_results[0]['generated_text']
    return story

# Defining a function to transform story to speech/audio
def text2audio(story_text):
    audio_pipe = pipeline("text-to-audio", 
                          model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story_text)
    return audio_data


# Defining a main function to execute all the sub-functions
def main():    
    # Saving the uploaded file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # 1) Image to Text
    scenario = img2text(uploaded_file.name)
    st.write(f"**Scenario:** {scenario}")

    # 2) Text to Story
    story_text = text2story(scenario)
    st.write(f"**Story:** {story_text}")

    # 3) Story to Audio
    audio_data = text2audio(story_text)
    
    return audio_data


# Execute main() if there is an uploaded file
if uploaded_file is not None:
    audio_data = main()
    # Setting up a Play button
    if st.button("Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)



   












