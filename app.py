# Importing necessary packages for application
import streamlit as st
from transformers import pipeline

# Setting headers and titles for the application
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

# Defining a function to generate a story from the extracted text
def text2story(text):
    story_pipe = pipeline("text-generation", 
                            model="pranavpsv/genre-story-generator-v2")

    # Writing a prompt to ensure that the generated story is suitable for the target audience
    prompt = f"Write a story suitable for children between the ages of 3-10 years old about {text}."
    story_results = story_pipe(prompt, 
                               min_new_tokens = 60,      # An output range of 60-120 tokens generally results in a story that is 50-100 words long.
                               max_new_tokens = 120,
                               temperature = 0.8)        # Temperature set to 0.8 to make the story more focused on caption.
    
    # Extracting the generated output
    full_text = story_results[0]['generated_text']
    
    # Remove the prompt from the output so only the story remains
    story = full_text[len(prompt):].strip()
    return story

# Defining a function to transform the generated story to speech/audio format
def text2audio(story_text):
    audio_pipe = pipeline("text-to-audio", 
                          model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story_text)
    return audio_data


# Defining a main function to execute all the sub-functions
def main():    
    st.image(uploaded_file, use_container_width=True)
        # Check if we already have the data in session state to avoid re-running
    if 'audio_data' not in st.session_state:
        # Saving the uploaded file locally
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        # 1) Image to Text
        scenario = img2text(uploaded_file.name)
        st.session_state['scenario'] = scenario
        st.write(f"**Scenario:** {st.session_state['scenario']}")

        # 2) Text to Story
        story_text = text2story(scenario)
        st.session_state['story_text'] = story_text
        st.write(f"**Story:** {st.session_state['story_text']}")

        # 3) Story to Audio
        audio_data = text2audio(story_text)
        st.session_state['audio_data'] = audio_data    

    # Returning audio_data from Session State
    return st.session_state['audio_data']


# Defining a function to clear session state safely
def clear_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


# Execute main() if there is an uploaded file
if uploaded_file is not None:
    audio_data = main()
    
    # Setting up a Play button for the generated text
    if st.button("Play Audio"):
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)

    # Setting up an optional button to re-generate a story
    st.button("Start Over", on_click = clear_state)



   












