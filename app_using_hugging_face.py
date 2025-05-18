import streamlit as st
st.set_page_config(page_title="🧠 Caption, Summarize & Sentiment", layout="wide")

import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import pipeline
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load image captioning model
@st.cache_resource
def load_caption_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, processor, tokenizer

caption_model, caption_processor, caption_tokenizer = load_caption_model()

# Load summarizer
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# Load sentiment analysis
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis")

sentiment_model = load_sentiment_model()

# Generate caption from image
def generate_description_from_image(image):
    try:
        image = image.convert("RGB")
        pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values
        output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=4)
        caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logger.error(f"Captioning error: {str(e)}")
        st.error("❌ Captioning failed.")
        return None

# Summarize text
def summarize_text(text):
    try:
        result = summarizer(text, max_length=45, min_length=10, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        st.error("❌ Summarization failed.")
        return None

# Sentiment analysis
def analyze_sentiment(text):
    try:
        result = sentiment_model(text)
        label = result[0]['label']
        score = round(result[0]['score'], 2)
        return f"{label} (confidence: {score})"
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        st.error("❌ Sentiment analysis failed.")
        return None

# UI
st.title("🧠 Image Captioning + Summarization + Sentiment")
st.write("Upload an image to get a caption, a summarized description, and a sentiment analysis.")

uploaded_image = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("1️⃣ Scene Description")
    with st.spinner("Generating caption..."):
        caption = generate_description_from_image(image)
    if caption:
        st.success(f"🖼️ **Caption**: {caption}")

        st.subheader("2️⃣ Summary")
        with st.spinner("Summarizing..."):
            summary = summarize_text(f"The image shows: {caption}")
        if summary:
            st.info(f"📝 **Summary**: {summary}")

            st.subheader("3️⃣ Sentiment Analysis")
            with st.spinner("Analyzing sentiment..."):
                sentiment = analyze_sentiment(summary)
            if sentiment:
                st.warning(f"💡 **Sentiment**: {sentiment}")
