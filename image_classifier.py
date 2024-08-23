import cv2  as cv
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, preprocessing
from PIL import Image as PILImage

img_height = 96
img_width = 96
color_channels = 3 # 3 for rgb, 1 for grayscale
batch_size = 2


# Normalize the pixel values

# training_images = training_images / 255
# testing_images = testing_images / 255

# Convert class vectors to binary class matrices
class_names = ['Cat', 'Dog']

# ds_train = None
# ds_val = None
class Classifier:
    #load custom train dataset
    def load_dataset(required_dataset):
        ds_train = preprocessing.image_dataset_from_directory(
            "data/train_set/",#this will be the dynamic path for your dataset
            labels = "inferred",
            label_mode = "int", # int, binary, categorical
            class_names = class_names,
            color_mode = "rgb", #"rgb", "rgba", "grayscale"
            batch_size = batch_size,
            image_size = (img_height, img_width), #reshape if not in this size
            shuffle = True, #shuffle and get randomized order
            seed = 123,
            validation_split = .1, #separate 10% of the data for validation
            subset = "training"
        )

        ds_val = preprocessing.image_dataset_from_directory(
            "data/train_set/",#you can use dif datasets, train & validation
            labels = "inferred",
            label_mode = "int", # int, binary, categorical
            class_names = class_names,
            color_mode = "rgb", #rgb
            batch_size = batch_size,
            image_size = (img_height, img_width), #reshape if not in this size
            shuffle = True, #shuffle and get randomized order
            seed = 123,
            validation_split = .1, #separate 10% of the data for validation
            subset = "validation"
        )

        '''
        #if we need to augment
        def augment(x, y):
            image = tf.image.random_brightness(x, max_delta=.05)
            return image, y
        ds_train = ds_train.map( augment)
        '''

        if required_dataset == "training":
            return ds_train
        elif required_dataset == "validation":
            return ds_val
        else:
            raise ValueError("required dataset must be either \'training\' or \'validation\'")

    #----------------------------------------------------------------

    #show available images and label with plt
    #igonre this, its for me
    '''for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.xticks([])
        plt.yticks([])
        #plt.grid(False) # not included in tutorial
        plt.imshow(training_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[training_labels[i][0]])
    plt.show()'''

    # Build the CNN model
    def train_model(train_data, val_data, class_names, width, height, color_channels):
        model = models.Sequential()

        # inputs = Input(shape=(img_height, img_width, 3))
        # x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)

        #input layer
        model.add(layers.Input((height,width, color_channels)))
        model.add(layers.Conv2D(32, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (2, 2), activation='relu'))

        #flatten the 2D output from the last layer into a 1D vector
        model.add(layers.Flatten())

        #dens layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(len(class_names), activation='softmax')) #2 possible classifications, len(class_names)


        """

        input_shape=(32,32, 3)
            meaninf the image resolution should be is 32x32 pixels and 3 color channe;s

        Convolutional layer filters patterns or features such as:
            horse has long legs
            airplane has a wing
            cat has pointy ears
            truck>>>Car
            ...

        Maxpolling reduces the image to essential information

        Output layer gets percentages or probabilites for the individual classificaction


        """

        #compute the model
        model.compile(optimizer='adam',
            loss = 'sparse_categorical_crossentropy',
            metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=5)

        model.fit(train_data, epochs=3,  #100,10-20 is adequate for basic
            validation_data=val_data, 
            #callbacks=[early_stop]
            )

        """
        Epochs defines how many time its going to see the same data over and over again
        """

        loss, accuracy = model.evaluate(load_dataset("validation"))
        print(f"loss: {loss} \naccuracy: {accuracy}")

        #save model
        # model.save("image_classifier.model")
        model.save("cat_dog_classifier-3_epochs_x96px.keras")


    #use this to start the training process, 
    #the model saves  automatically after successful training
    '''train_model(
        load_dataset("training"),
        load_dataset("validation"), class_names, 
        img_width, img_height, 
        color_channels
        )'''

    #------------------------------------------------------------------
    #implement predict method

    def predict_image(path, model):
        print("Getting prediction...")

        #create a temporary folder to store optimized files
        try:
            print("making temporary dir")
            os.mkdir('temps')
        except FileExistsError:
            print("temporary file already exists")
        
        #open and resize to the same width, height as in training
        image = PILImage.open(path)
        resized_image = image.resize((img_width, img_height),
        #PILImage.ANTIALIAS
        )

        path = "temps/adjusted.jpg"

        print(f'temp path: {path}')
        
        resized_image.save(path)

        image = cv.imread(path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # plt.imshow(image, cmap=plt.cm.binary)

        """
        if the used image for prediction has a resolution different from
        the image_height and image_width used int the trainig tensorflow will raise an error

        ------------------------

        here is an example of the error of a picture with 960x720:

        inputs=tf.Tensor(shape=(1, 720, 960, 3), dtype=uint8)

        """

        prediction = model.predict(np.array([image]))
        index = np.argmax(prediction)

        result = f"Prediction is {class_names[index]}"

        return result


    """
    try to use no more  that one subforlder when loading 
    the image dynamic  path. 

    if not, it will give an error of cv.cvtColor which has  nothing to do
    and you wont know why
    """

    #in case you just need the resize func only
    def resize(image, width, height):
        try:
            print("making temporary dir")
            os.mkdir('temps')
        except FileExistsError:
            print("temporary file already exists")
        image = PILImage.open(image)
        resized_image = image.resize((width, height),
        #PILImage.ANTIALIAS
        )

        saved_path = "temps/adjusted.jpg"

        print(f'temp path: {saved_path}')
        
        resized_image.save(saved_path)
        
        return saved_path

if __name__ == '__main__':

    cat_dog_model = models.load_model("cat_dog_classifier-40_epochs_x96px.keras")

    img = "data/x96.jpg"#use your image or else error

    get_prediction = Classifier.predict_image(img, cat_dog_model)

    print(get_prediction)
