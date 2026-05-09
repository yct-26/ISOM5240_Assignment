## Setup
# Importing necessary packages for application
import streamlit as st
from transformers import pipeline

# Setting headers and titles for the application
st.set_page_config(page_title="Transform an Image to an Audio Story", 
                   page_icon="🧠")
st.header("Transform Your Image into an Audio Story 📖")

# Adding a file uploader to the application
uploaded_file = st.file_uploader("Select an Image.")



## Defining Functions
# Defining a function to transform image to text (caption)
def img2text(url):
    img2text_model = pipeline("image-to-text",
                              model="Salesforce/blip-image-captioning-base")
    text = img2text_model(url)[0]["generated_text"]
    return text

# Defining a function to generate a story from the extracted text
def text2story(text):
   story_model = pipeline("text-generation", 
                          model="pranavpsv/genre-story-generator-v2")
   # Edit the prompt so that the generated story is suitable for children 
   prompt = f"Write a story for children about {text}:"
   story_results = story_model(prompt,
                              min_new_tokens=70,
                              max_new_tokens=120,
                              do_sample=True)
  # Removing the prompt from the generated output
   full_text = story_results[0]['generated_text']
   story = full_text.replace(prompt, "").strip()
   return story

# Defining a function to transform the generated story to speech/audio format
def text2audio(story_text):
    # This pipeline occasionally crashes when there is an error in its input. This condition implements an error-handling feature when this occurs.
    if not story_text or len(story_text.strip()) < 5:
        return None
    audio_model = pipeline("text-to-audio", 
                           model="Matthijs/mms-tts-eng")
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



## Execution
# Run main() if there is an uploaded image file
if uploaded_file is not None:
    audio_data = main()
  
    # Conditional Play Audio button
    # Ensure that the story generation process was successful and audio data actually exists. If so, play the audio. Otherwise, display an error message.
    if st.button("Play Audio"):
        if audio_data is not None:
            audio_array = audio_data["audio"]
            sample_rate = audio_data["sampling_rate"]
            st.audio(audio_array, sample_rate=sample_rate)
        else:
            st.error("There was an error when generating the story. Please refresh and try again.")
