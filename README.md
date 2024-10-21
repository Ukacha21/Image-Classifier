### Image Classifire with Tensorflow.Keras

This classifier identifies Cats and Dog in general. Unlike ciphar10 or ciphar100, this classifier model was trained on a custom local dataset of nearly 4800 images for each classname, which means, dont expect it to perform too well.

For **testing** use [validator](validator.py) or [ui-validator](ui-validator.py)

**Requirements:**

* Numpy
* Matplotlib
* Tensorflow
* OpenCV
* Pillow(PIL)

**What you can improve**

If you have the time and resources you can improve the training using a larger dataset, increasing number of batch sizes, epochs, image resolution, etc.

In both 3 and 40 epoch models a resolution of 96x96 pixels is used

```python
    img_height, img_width = 96, 96
```

**What you must know**

1- The resolutiion you use for training must stay consistent in testing as well, for instance if you train a model with a resolutiion of x100 and test with one of x200 or any different, it will raise an error. To avoid that, I resized every image bebfore testing and stored in a temp folder created locally.

```python
#create a temporary folder to store optimized files
try:
    print("making temporary dir")
    os.mkdir('temps')
except FileExistsError:
    print("temporary file already exists")

#open and resize to the same width, height as in training
image = PILImage.open(path)
resized_image = image.resize((img_width, img_height),
)

path = "temps/adjusted.jpg"

print(f'temp path: {path}')

resized_image.save(path)
```

2- Only trained models are included in this repo,

    * cat_dog_classifier-3_epochs_x96px.keras 

and

    * cat_dog_classifier-40_epochs_x96px.keras

the dataset isnt a availabe, but you can find even larger datasets in places like kaggle.

Other issues are mentioned as docstrings within the 
[image_classifier.py](image_classifier.py) file.



