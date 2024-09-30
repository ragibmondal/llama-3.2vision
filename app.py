import os
import streamlit as st
from typing import Generator
from groq import Groq
from dotenv import load_dotenv
import yaml
import base64
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def load_config():
    """Load and validate the configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        if "models" not in config:
            config["models"] = {}

        # Add the new models to the configuration
        new_models = [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-3.2-11b-text-preview",
            "llama-3.2-11b-vision-preview",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
            "llama-3.2-90b-text-preview",
            "llama-guard-3-8b",
            "llama3-70b-8192",
            "llama3-8b-8192"
        ]

        for model in new_models:
            if model not in config["models"]:
                config["models"][model] = {
                    "name": model,
                    "developer": "Meta/Groq",
                    "description": f"Llama model: {model}",
                    "tokens": 8192,  # Default max tokens, adjust as needed
                    "vision_capable": "vision" in model
                }
        
        if "default_max_tokens" not in config:
            config["default_max_tokens"] = 1024
        
        if "prompt_templates" not in config:
            config["prompt_templates"] = {}
        
        return config
    except FileNotFoundError:
        st.error("config.yaml file not found. Please ensure it exists in the same directory as this script.")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"Error reading config.yaml: {e}")
        st.stop()

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Set up the Streamlit page
st.set_page_config(page_icon="💬", layout="wide", page_title="Llama Chat App with Multi-Image Vision")

# Load configuration
config = load_config()

# Load Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable not found. Please set it in the .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Initialize chat history and uploaded images
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# Model selection
model_options = list(config["models"].keys())
model_option = st.sidebar.selectbox("Select Model", options=model_options, index=0)

model_info = config["models"][model_option]

# Display model information
st.sidebar.header("Model Information")
st.sidebar.markdown(f"**Name:** {model_info['name']}")
st.sidebar.markdown(f"**Developer:** {model_info['developer']}")
st.sidebar.markdown(f"**Description:** {model_info['description']}")
st.sidebar.markdown(f"**Vision Capable:** {'Yes' if model_info['vision_capable'] else 'No'}")

max_tokens_range = model_info.get("tokens", 8192)  # Default to 8192 if not specified

# Adjust max_tokens slider
max_tokens = st.sidebar.slider(
    "Max Tokens:",
    min_value=512,
    max_value=max_tokens_range,
    value=min(config["default_max_tokens"], max_tokens_range),
    step=512,
    help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
)

# Add multiple image upload option
uploaded_files = st.sidebar.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
        encoded_image = encode_image_to_base64(image)
        st.session_state.uploaded_images.append({
            "name": uploaded_file.name,
            "data": f"data:image/png;base64,{encoded_image}"
        })

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        if isinstance(message["content"], list):
            for content in message["content"]:
                if content["type"] == "text":
                    st.markdown(content["text"])
                elif content["type"] == "image_url":
                    st.image(content["image_url"]["url"], caption=content.get("name", "Uploaded Image"))
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Enter your prompt here..."):
    # Prepare the message content
    message_content = [{"type": "text", "text": prompt}]
    
    # If there are uploaded images and the model is vision capable, include them
    if st.session_state.uploaded_images and model_info["vision_capable"]:
        for image in st.session_state.uploaded_images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": image["data"]},
                "name": image["name"]
            })
        # Clear the uploaded images to avoid reusing them in future messages
        st.session_state.uploaded_images = []
    
    st.session_state.messages.append({"role": "user", "content": message_content})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)
        for content in message_content[1:]:
            st.image(content["image_url"]["url"], caption=content["name"])

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=1,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(f"Error: {e}", icon="🚨")
    else:
        # Append the full response to session_state.messages
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response if isinstance(full_response, str) else "\n".join(str(item) for item in full_response)}
        )

# Add a clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.uploaded_images = []

# Add a download chat history button
if st.sidebar.button("Download Chat History"):
    chat_history = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
    st.download_button(
        label="Download Chat History",
        data=chat_history,
        file_name="chat_history.txt",
        mime="text/plain",
    )

# Add a prompt templates section
st.sidebar.header("Prompt Templates")
template_options = list(config["prompt_templates"].keys())
if template_options:
    selected_template = st.sidebar.selectbox("Choose a prompt template", options=template_options)
    if st.sidebar.button("Load Template"):
        prompt_template = config["prompt_templates"][selected_template]
        st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": prompt_template}]})
        st.experimental_rerun()
else:
    st.sidebar.info("No prompt templates available.")
