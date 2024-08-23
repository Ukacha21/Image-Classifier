
img_height = 96
img_width = 96

from image_classifier import Classifier
from tensorflow.keras import models

cat_dog_model = models.load_model("cat_dog_classifier-40_epochs_x96px.keras")

img = "data/x96.jpg"

get_prediction = Classifier.predict_image(img, cat_dog_model)

print(get_prediction)

