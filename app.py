import streamlit as st
from groq import Groq
from PIL import Image
import base64
from io import BytesIO

# Initialize Groq client
client = Groq()

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_image(image, question):
    encoded_image = encode_image(image)
    image_url = f"data:image/jpeg;base64,{encoded_image}"
    
    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content

def main():
    st.title("Image Analysis with Groq API")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Question input
    question = st.text_input("Enter your question about the image:", "What's in this image?")

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Analyze the image
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                analysis = analyze_image(image, question)
            st.subheader("Analysis Results:")
            st.write(analysis)

if __name__ == "__main__":
    main()
