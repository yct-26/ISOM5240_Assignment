# Importing necessary packages for application
import streamlit as st
from transformers import pipeline

# Setting headers and titles for the application
st.set_page_config(page_title="Transform an Image to an Audio Story", page_icon="🧠")
st.header("Transform Your Image into an Audio Story")

# Adding a file uploader to the application
uploaded_file = st.file_uploader("Select an Image.")

# Defining a function to transform image to text (caption)
def img2text(url):
    img2text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = img2text_model(url)[0]["generated_text"]
    return text

# Defining a function to generate a story from the extracted text
def text2story(text):
    story_model = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    prompt = f"Children's story (ages 3-10) about {text}. Focus on topic."
    story_results = story_model(prompt, 
                               min_new_tokens = 60, 
                               max_new_tokens = 120,
                               temperature = 0.8)
    
    full_text = story_results[0]['generated_text']
    story = full_text[len(prompt):].strip()
    return story

# Defining a function to transform the generated story to speech/audio format
def text2audio(story_text):
    audio_model = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_model(story_text)
    return audio_data

# Defining a main function to group all the functions together
def main():    
    st.image(uploaded_file, use_container_width=True)

    # Check if we already have the data to avoid re-running everything on button click
    if 'audio_data' not in st.session_state:
        # Saving the uploaded file locally
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        # 1) Image to Text
        scenario = img2text(uploaded_file.name)
        st.session_state['scenario'] = scenario

        # 2) Text to Story
        story_text = text2story(scenario)
        st.session_state['story_text'] = story_text

        # 3) Story to Audio
        audio_data = text2audio(story_text)
        st.session_state['audio_data'] = audio_data

    # Displaying results from session state
    st.write(f"**Scenario:** {st.session_state['scenario']}")
    st.write(f"**Story:** {st.session_state['story_text']}")
    
    return st.session_state['audio_data']


# Execute main() if there is an uploaded file
if uploaded_file is not None:
    audio_data = main()

    # The conditional Play Audio button
    if st.button("Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
