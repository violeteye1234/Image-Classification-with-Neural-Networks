#tutorial: https://www.youtube.com/watch?v=jztwpsIzEGc
#virtual environment: https://www.youtube.com/watch?v=IAvAlS0CuxI&t=21s

#1) Building a Data Pipeline
#2) Preprocessing Images for Deep Learning
#3) Creating a deep Neural Network Classifier
#4) Evaluating Model Performance
#5) Saving the Model for Deployment


#Setup & Load Data
#1.1 Install Dependencies & Setup 
#Install: tensorflow, tensorflow-gpu, opencv-python, matplotlib

import tensorflow
import os #navigate through file structures

#limit tensor from using all the VRAM (Prevent Out Of Memory (OOM) Error)
"""
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
numGpus = len(gpus)
if numGpus > 0:
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU Devices Available. Using CPU")
"""

#1.2 Remove Dodgy Images
import cv2
import imghdr #checking file extensions 
dataDir = 'data/'
imgExts = ['jpeg','jpg','bmp','png']

for imageClass in os.listdir(dataDir):
    if imageClass.startswith("."):
        continue
    for image in os.listdir(os.path.join(dataDir, imageClass)):
        imagePath = os.path.join(dataDir, imageClass, image) #joining the entire directory path of the image
        try:
            img = cv2.imread(imagePath) #reading image using OpenCV
            tip = imghdr.what(imagePath)
            if tip not in imgExts:
                print("Image not in ext list {}".format(imagePath))
                os.remove(imagePath)
        except Exception as e:
            print("Issue with Image {}".format(imagePath))

#1.3 Load Data
#Create a Data Pipeline
#API AccesS: tensorflow.data.Dataset
import numpy 
from matplotlib import pyplot
#Building Image Dataset (Will preprocess for you - resize etc.)
#Building data pipeline
data = tensorflow.keras.utils.image_dataset_from_directory('data')
#allowing to access data pipeline - allowing to loop through it
dataIterator = data.as_numpy_iterator() 
#accessing data pipeline
#2 batches - 1.Images 2.Labels
batch = dataIterator.next()
#images represented as numpy arrays
batch[0].shape
#Class 1 = Sad; Class 0 = Happy
batch[1]

fig, ax = pyplot.subplots(ncols = 4, figsize = (20,20))
for index, img in enumerate(batch[0][:4]):
    ax[index].imshow(img.astype(int))
    ax[index].title.set_text(batch[1][index])
pyplot.show() #shows the graph
#--------------------Part 1 Complete-------------------------
