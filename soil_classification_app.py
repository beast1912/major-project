import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps # For image processing
import numpy as np

crop_recommendations = {
    "alluvial_soil": ["Rice", "Sugarcane", "Wheat"],
    "black_soil": ["Cotton", "Wheat", "Soybeans"],
    "clay_soil": ["Potatoes", "Cabbage", "Lettuce"],
    "red_soil": ["Rice", "Wheat", "Millets"],
}

def get_crop_recommendations(soil_type):
    return crop_recommendations.get(soil_type, [])


def preprocess_image(image_data):
  """
  Preprocesses an image for model input.

  Args:
      image_data: The image data as a Streamlit file uploader object.

  Returns:
      A NumPy array representing the preprocessed RGB image with shape (1, 224, 224, 3).
  """

  # Read image using PIL (consistent with Streamlit uploads)
  img = Image.open(image_data)

  # Convert to RGB if necessary (accounting for OpenCV BGR format)
  if img.mode != 'RGB':
    img = img.convert('RGB')  # Handle RGBA, grayscale, and other modes

  # Resize using PIL for consistency
  img = img.resize((224, 224), Image.ANTIALIAS)

  # Convert to NumPy array
  img_arr = np.array(img)

  # Normalize pixel values
  img_arr = img_arr / 255.0

  # Reshape to add batch dimension
  img_reshape = img_arr[np.newaxis, ...]

  return img_reshape


def main():
  
  loaded_model = tf.keras.models.load_model('mobile_net_model_final1.h5')

  # User interface
  uploaded_file = st.file_uploader("Upload an image here", type=["jpg","png"])
  if uploaded_file is not None:
      image = preprocess_image(uploaded_file)
      # image = np.expand_dims(image, axis=0)  # Add batch dimension
      print(image.shape)
      prediction = loaded_model.predict(image)
      class_names = ['alluvial_soil', 'black_soil', 'clay_soil', 'red_soil']
      predicted_soil_type_class= class_names[np.argmax(prediction[0])]
      st.image(image, caption="Uploaded Image")
      recommended_crops = get_crop_recommendations(predicted_soil_type_class)
      predicted_soil_type = predicted_soil_type_class.replace("_", " ").upper()
      st.write(f"Predicted Class: {predicted_soil_type}")
      if recommended_crops:
        st.write("Recommended Crops:")
        for crop in recommended_crops:
          st.write(f"- {crop}")
      else:
          st.write("No crop recommendations available for this soil type.")


if __name__ == "__main__":

    main()
