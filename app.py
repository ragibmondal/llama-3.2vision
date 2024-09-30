import os
import streamlit as st
from PIL import Image
import io
import base64
from together import Together

# Set up Together AI client
together_api_key = os.environ.get('TOGETHER_API_KEY')
client = Together(api_key=together_api_key)

def encode_image(image):
    """Encode image to base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_image(image):
    """Analyze the image using Together AI API"""
    encoded_image = encode_image(image)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image and describe what you see in detail."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>", "<|eom_id|>"]
    )
    return response.choices[0].message.content

def main():
    st.set_page_config(page_title="Advanced Image Analysis App", layout="wide")
    st.title("Advanced Image Analysis App")

    # Sidebar for app mode selection
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Single Image Analysis", "Image Comparison"])

    if app_mode == "Single Image Analysis":
        single_image_analysis()
    else:
        image_comparison()

def single_image_analysis():
    st.header("Single Image Analysis")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                analysis = analyze_image(image)
                st.subheader("Image Analysis")
                st.write(analysis)

def image_comparison():
    st.header("Image Comparison")
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file1 = st.file_uploader("Choose first image...", type=["jpg", "jpeg", "png"])
    with col2:
        uploaded_file2 = st.file_uploader("Choose second image...", type=["jpg", "jpeg", "png"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        image1 = Image.open(uploaded_file1)
        image2 = Image.open(uploaded_file2)

        col1.image(image1, caption="First Image", use_column_width=True)
        col2.image(image2, caption="Second Image", use_column_width=True)

        if st.button("Compare Images"):
            with st.spinner("Analyzing images..."):
                analysis1 = analyze_image(image1)
                analysis2 = analyze_image(image2)

                st.subheader("Image 1 Analysis")
                st.write(analysis1)

                st.subheader("Image 2 Analysis")
                st.write(analysis2)

                st.subheader("Comparison")
                comparison_prompt = f"Compare and contrast the following two image descriptions:\n\nImage 1: {analysis1}\n\nImage 2: {analysis2}\n\nProvide a detailed comparison highlighting similarities and differences."
                comparison = client.chat.completions.create(
                    model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                    messages=[
                        {"role": "user", "content": comparison_prompt}
                    ],
                    max_tokens=512,
                    temperature=0.7
                )
                st.write(comparison.choices[0].message.content)

if __name__ == "__main__":
    main()
