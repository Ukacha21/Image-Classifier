
from tkinter import *
from tkinter import filedialog
from image_classifier import Classifier
from tensorflow.keras import models

img_height = 96
img_width = 96
#loading the model might take some time, 
#make it the first thing you do
cat_dog_model = models.load_model("cat_dog_classifier-40_epochs_x96px.keras")
def browse_file(label):
    image = filedialog.askopenfilename(
        title="Please choose a valid image file",
        filetypes=[
        ("PNG files", "*.png"), 
        ("JPG files", "*.jpg"),
        ("Any", "*.*")
        ])
    label.config(text="Please wait...")
    '''
    image.open will give error if you browse with askopenfile,
    you should browse the file with askopenfilename isntead
    '''
    get_prediction = Classifier.predict_image(image, cat_dog_model)
    print(get_prediction)
    label.config(text=get_prediction)
def run():
    root = Tk()
    root.geometry("400x400")
    pred_label = Label(root, text="Choose an image to predict", font=("arial", 16))
    pred_label.pack(pady=25)

    select_button = Button(root, text="Browse", command= lambda: browse_file(pred_label))
    select_button.pack(pady=20)

    root.mainloop()

run()
