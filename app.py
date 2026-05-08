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
    image_to_text_model = pipeline("image-to-text", 
                                   model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# Defining a function to generate a story from the extracted text
def text2story(text):
    story_pipe = pipeline("text-generation", 
                          model="pranavpsv/genre-story-generator-v2")

    # Writing a prompt to ensure that the generated story is suitable for the target audience
    prompt = f"Children's story (ages 3-10) about {text}. Focus on topic."
    story_results = story_pipe(prompt, 
                               min_new_tokens = 75,      # Accounting for the tokens in the prompt, a range of 75-150 tokens should result in a story that is 50-100 words long.
                               max_new_tokens = 150,
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


# Defining a main function to group all the functions together
def main():    
    # Show the image immediately after file is uploaded
    st.image(uploaded_file, use_container_width=True)

    # Saving the uploaded file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    # 1) Image to Text
    scenario = img2text(uploaded_file.name)
    st.write(f"**Scenario:** {scenario}")

    # 2) Text to Story
    story_text = text2story(scenario)
    st.write(f"**Story:** {story_text}")

    # 3) Story to Audio
    audio_data = text2audio(story_text)


# Execute main() if there is an uploaded file
if uploaded_file is not None:
    main()
    if st.button("Play Audio")
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
   












