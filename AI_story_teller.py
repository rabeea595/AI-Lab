import streamlit as st
from PIL import Image
from transformers import pipeline
st.set_page_config(page_title="AI Storyteller from Images", layout="centered")
st.title("📖 AI Storyteller from Images")
st.write("Upload an image, ask a question, and let AI describe and summarize a story.")
@st.cache_resource(show_spinner=True)
def load_models():
    try:
        vqa_pipe = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
        text_generator = pipeline("text-generation", model="gpt2")
        summarizer = pipeline("summarization", model="t5-small")
        return vqa_pipe, text_generator, summarizer
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None
vqa_pipe, text_generator, summarizer = load_models()
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
question = st.text_input("❓ Ask a question about the image", "What is in the image?")
if uploaded_file and question and vqa_pipe:
    try:
        image = Image.open(uploaded_file).convert("RGB").resize((256, 256))
        st.image(image, caption="Uploaded Image", use_column_width=False)

        st.subheader("1️⃣ Visual Question Answering")
        with st.spinner("Answering your question..."):
            vqa_result = vqa_pipe(image=image, question=question, top_k=1)
            answer = vqa_result[0]['answer']
            st.success(f"💬 **Answer**: {answer}")
        st.subheader("2️⃣ Story Generation")
        with st.spinner("Generating a short story..."):
            story_prompt = f"A picture shows {answer}. This image shows"
            generated_story = text_generator(story_prompt, max_length=100, num_return_sequences=1)
            story_text = generated_story[0]['generated_text']
            st.info(f"📘 **Story**:\n\n{story_text}")
        st.subheader("3️⃣ Summary")
        with st.spinner("Summarizing the story..."):
            summary = summarizer(story_text, max_length=30, min_length=10, do_sample=False)
            st.success(f"📝 **Summary**: {summary[0]['summary_text']}")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
else:
    st.warning("Please upload an image and type a question to proceed.")
