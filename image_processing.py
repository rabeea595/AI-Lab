import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Load Haar cascades for object detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'cars.xml')
cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')

def detect_objects(image, cascade, scale_factor=1.1, min_neighbors=5, color=(255, 0, 0), thickness=2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    for (x, y, w, h) in objects:
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
    return image

def apply_gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def main():
    st.title("Image Processing with OpenCV")
    st.write("Upload an image and apply various computer vision operations")

    # Sidebar controls
    st.sidebar.title("Options")
    operation = st.sidebar.selectbox(
        "Select Operation",
        ["Original", "Gaussian Blur", "Detect Faces", "Detect Cars", "Detect Cats"]
    )

    # Parameters based on selected operation
    params = {}
    if operation == "Gaussian Blur":
        params["kernel_size"] = st.sidebar.slider("Kernel Size", 1, 25, 5, step=2)
    elif operation in ["Detect Faces", "Detect Cars", "Detect Cats"]:
        params["scale_factor"] = st.sidebar.slider("Scale Factor", 1.01, 1.5, 1.1, step=0.01)
        params["min_neighbors"] = st.sidebar.slider("Min Neighbors", 1, 10, 5)
        params["thickness"] = st.sidebar.slider("Rectangle Thickness", 1, 10, 2)

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and convert image
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        # Convert RGBA to RGB if needed
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display original image
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Process image based on selected operation
        processed_image = image.copy()
        
        if operation == "Gaussian Blur":
            processed_image = apply_gaussian_blur(processed_image, params["kernel_size"])
            st.image(processed_image, caption="Blurred Image", use_column_width=True)
        
        elif operation == "Detect Faces":
            processed_image = detect_objects(
                processed_image, 
                face_cascade,
                params["scale_factor"],
                params["min_neighbors"],
                (0, 255, 0),  # Green
                params["thickness"]
            )
            st.image(processed_image, caption="Face Detection", use_column_width=True)
        
        elif operation == "Detect Cars":
            processed_image = detect_objects(
                processed_image, 
                car_cascade,
                params["scale_factor"],
                params["min_neighbors"],
                (255, 0, 0),  # Blue
                params["thickness"]
            )
            st.image(processed_image, caption="Car Detection", use_column_width=True)
        
        elif operation == "Detect Cats":
            processed_image = detect_objects(
                processed_image, 
                cat_cascade,
                params["scale_factor"],
                params["min_neighbors"],
                (0, 0, 255),  # Red
                params["thickness"]
            )
            st.image(processed_image, caption="Cat Detection", use_column_width=True)
        
        # Download button for processed image
        if operation != "Original":
            buf = BytesIO()
            processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            processed_pil.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name=f"processed_{operation.lower().replace(' ', '_')}.jpg",
                mime="image/jpeg"
            )

if __name__ == "__main__":
    main()