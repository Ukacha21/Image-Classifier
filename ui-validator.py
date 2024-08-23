from tkinter import *
from tkinter import filedialog
from image_classifier import Classifier
from tensorflow.keras import models

cat_dog_model = models.load_model("cat_dog_classifier-40_epochs_x96px.keras")

def browse(label):
    image = filedialog.askopenfilename(
        title="choose an valid image",
        filetypes=[
            ("PNG files", "*.png"),
            ("JPG files", "*.jpg"),
            ("Any files", "*.*"),
        ]
    )
    '''
    image = Image.open(image)  # This will give error if you browse with askopenfile,
    because we need the name and the dynamic path,
    you should browse the file with askopenfilename isntead
    '''
    
    label.config(text="please wait.......")

    get_prediction = Classifier.predict_image(image, cat_dog_model)
    label.config(text=get_prediction)

def run():
    root = Tk()
    root.geometry("400x400")
    pred_label = Label(root, text="choose an image", font=("ariat", 16))
    pred_label.pack(pady=25)

    button = Button(root, text="Browse", command= lambda: browse(pred_label))
    button.pack(pady=25)

    root.mainloop()

run()