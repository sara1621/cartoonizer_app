import streamlit as st
import cv2
import numpy as np
from PIL import Image
st.set_page_config(page_title="Cartoonizer", page_icon="🎨", layout="centered")


  #Titre de l'application
title_html = """
<h1 style="margin: 0; font-family: Tahoma, sans-serif;margin-left: 5px;">
<span style="color: #F5F5DC;">Cartooni</span><span style="color: #F5F5DC;">zer</span>
</h1>
"""
st.title("Transform your photos into amazing cartoon-style images!")
st.markdown(
    "<h3 style='text-align: left; color:rgb(122, 122, 122);'>Upload an image, choose your cartoonization method, and see the magic!</h3>",
    unsafe_allow_html=True
)
#st.write("Transform your photos into amazing cartoon-style images!")

st.markdown("""
    <style>
    .title {
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

image_path = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRre7HcSQGP0PhCdsoqEN7Y6j0tfygxOC_0BQ&s"

#import os
#if os.path.exists(image_path):
   # print("Image found!")
#else:
   # print("Image not found!")
#image_path = "https://t3.ftcdn.net/jpg/08/98/22/00/360_F_898220026_YpEtXl3GCaJM39rPLux8t0acxy3wpsQN.jpg"
#st.image(image_path, caption="Cartoonize Your Life", use_container_width=True)

palestine_path='https://img2.freepng.fr/20190628/o/kisspng-palestinian-national-authority-flag-of-palestine-c-stop-the-war-palestine-peace-dove-clipart-full-5d16b0ac1a97c0.8831415215617681081089.jpg'




bannerh_html = f"""
<div style="position: fixed; left: 0; top: 0; width: 100%; padding: 2px; background-image: url('{image_path}'); background-size: cover;z-index: 1000">
 {title_html}
</div>
"""

st.markdown(bannerh_html, unsafe_allow_html=True)


# Ajoutez le CSS pour définir l'image comme arrière-plan
page_bg_img = """
<style>
@keyframes gradient {
    0% { background: #5F2757; }
    50% { background: #493131; }
    100% { background: #5F2757; }
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(to top, #5F2757, #F5F5DC);
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    animation: gradient 5s ease infinite;
}
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.5);
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

#




uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])













def cartoonize_option_1(image):
    """Cartoonize using the first method (k-means and edges)."""
    # Step 1: Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=7
    )

    # Step 2: Color quantization using k-means clustering
    small_img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    Z = small_img.reshape((-1, 3))
    Z = np.float32(Z)

    K = 12  # Number of clusters for color quantization
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()]
    quantized_image = quantized_image.reshape(small_img.shape)

    # Resize back to original size
    quantized_image = cv2.resize(quantized_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Step 3: Combine quantized image with edges
    cartoon = cv2.bitwise_and(quantized_image, quantized_image, mask=edges)

    # Step 4: Apply a slight blur to enhance the cartoon effect
    cartoon = cv2.medianBlur(cartoon, 5)

    return cartoon

def cartoonize_option_2(image):
    """Cartoonize using the second method (bilateral filtering)."""
    # Step 1: Apply bilateral filter multiple times to smooth colors
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2
    )

    # Step 3: Combine the edges with the smoothed image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    cartoon = cv2.bitwise_and(smoothed, edges_colored)

    return cartoon

def cartoonize_option_3(image):
    """Cartoonize using the third method (k-means with enhanced edges and sharpening)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=11, C=7
    )

    small_img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    Z = small_img.reshape((-1, 3))
    Z = np.float32(Z)

    K = 12  # Increased number of colors for more vibrancy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()]
    quantized_image = quantized_image.reshape(small_img.shape)

    quantized_image = cv2.resize(quantized_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    cartoon = cv2.bitwise_and(quantized_image, quantized_image, mask=edges)
    cartoon = cv2.medianBlur(cartoon, 5)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    cartoon = cv2.filter2D(cartoon, -1, kernel)

    return cartoon



if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    if image.shape[-1] == 4:  # Convert RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    st.image(image, caption="Original Image", use_container_width=True)

    method = st.selectbox("Select Cartoonization Method", 

                          ["Option 1: K-Means & Edges", 
                           "Option 2: Bilateral Filtering", 
                           "Option 3: K-Means Enhanced"])

    if method == "Option 1: K-Means & Edges":
        cartoon = cartoonize_option_1(image)
    elif method == "Option 2: Bilateral Filtering":
        cartoon = cartoonize_option_2(image)
    else:
        cartoon = cartoonize_option_3(image)
    cartoon_image = Image.fromarray(cartoon) 
    st.image(cartoon_image, caption="Cartoonized Image", use_container_width=True)

    from io import BytesIO
    buffer = BytesIO()
    cartoon_image.save(buffer, format="PNG")  # Sauvegarder comme PNG
    buffer.seek(0)  # Réinitialiser le pointeur
    st.download_button(
     label="Download Cartoon Image",
     data=buffer,
     file_name="cartoon_image.png",
     mime="image/png",
      )   