#tutorial: https://www.youtube.com/watch?v=jztwpsIzEGc
#virtual environment: https://www.youtube.com/watch?v=IAvAlS0CuxI&t=21s

#1) Building a Data Pipeline
#2) Preprocessing Images for Deep Learning
#3) Creating a Deep Neural Network Classifier
#4) Evaluating Model Performance
#5) Saving the Model for Deployment

'''
Flowchart:
A: Install Dependencies & Setup 
    A1: Install tensorflow, tensorflow-gpu, opencv-python, matplotlib
    A2: Limit tensor from using all the VRAM & avoid OOM error
B: Remove Dodgy Images
    B1: Check Image Extensions
    B2: Remove Images not in expected extensions
C: Load Data
    C1: Create image dataset
    C2: Create data iterator to access the dataset
    C3: Visualise sample images and their labels
D: Preprocess Data
    D1: Scale data to 0-1 range
    D2: Split data into training, validation, and test sets 
E: Build Deep Learning Model
    E1: Create a Sequential Model
    E2: Add Convolutional, Max Pooling and Fully Connected Layers
    E3: Compile the model
F: Train Model
    F1: Train the model using fit method
    F2: Log Training and Validation Loss, Accuracy using TensorBoard
G: Evaluate Model Performance
    G1: Evaluate model performance using Precision, Recall, and Accuracy Metrics
    G2: Test the model with a sample image
H: Save Model for Deployment 
    H1: Save the trained model using save method
    H2: Load the saved model using load_method
    H3: Test the loaded model with the same sample image
'''

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

#Scanning through 'data/' directory to remove any images that are not specified in imgExts (Image Extensions)
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
#Loading data to create a data pipeline 
#API Access: tensorflow.data.Dataset
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

#Preprocess Data
#2.1 Scale Data
#Transformation to smallest values within the pipeline 
scaledData = data.map(lambda x,y: (x/255, y))
#iterating through datasets to grab the next batch
scaledIterator = scaledData.as_numpy_iterator() 
#setting batch between 0 and 1
batch = scaledIterator.next()

#2.2 Split Data
#Data to train the DLM
trainingSize = int(len(scaledData) *0.7)
#Data to evaluate model while training
validationSize = int(len(scaledData) *0.2)
#Data used after final evaluation state
testSize = int(len(scaledData) *0.1)+1

train = data.take(trainingSize)
#skip first 4
val = data.skip(trainingSize).take(validationSize)
test = data.skip(trainingSize+validationSize).take(testSize)
#--------------------Part 2 Complete-------------------------

#Deep Model
#3.1 Building the Deep Learning Model
#importing libraries & API
from tensorflow.keras.models import Sequential
#importing layers for images & Convulate (Recognise Patterns)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
#form an architecture (multiple layers) to build the neural network
model = Sequential() #instance of sequential class
#adding Convulational Layer 
#16 filters with 3x3px with a stride = 1. Activation = relu (Output is converted to 0 if negative __/ ) 256 height x 256 wide x 3 channels deep
model.add(Conv2D(16, (3,3), 1, activation ='relu', input_shape=(256,256,3)))
#adding Max Pooling Layer - Condense image data by selecting max
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3),1,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3),1,activation = 'relu'))
model.add(MaxPooling2D())

#Single Dimension
model.add(Flatten())

#Fully connected layers with 256 neurons and relu activation
model.add(Dense(256, activation = 'relu'))
#Single output - 0 or 1 
model.add(Dense(1, activation = 'sigmoid'))

#compile
model.compile('adam', loss=tensorflow.losses.BinaryCrossentropy(), metrics=['accuracy'])
summary = model.summary()
#print(summary)

#3.2 Train
logDir  = 'logs/'
#save and log checkpoints
tensorBoard_callBack = tensorflow.keras.callbacks.TensorBoard(log_dir=logDir)
#fit - take in training data 
#train data (4*32), epochs (no. of runs), run evaluation on validation data, and finally log info
history = model.fit(train, epochs=20, validation_data = val, callbacks = [tensorBoard_callBack]) 

figure1 = pyplot.figure()
pyplot.plot(history.history['loss'],color='green', label='loss')
pyplot.plot(history.history['val_loss'], color='red', label='val_loss')
figure1.suptitle('Loss',fontsize=20)
pyplot.legend(loc='upper left')
pyplot.show()

accuracyFigure = pyplot.figure()
pyplot.plot(history.history['accuracy'],color='green', label='accuracy')
pyplot.plot(history.history['val_accuracy'], color='red', label='val_accuracy')
figure1.suptitle('Accuracy',fontsize=20)
pyplot.legend(loc='upper left')
pyplot.show()
#--------------------Part 3 Complete-------------------------

#Evaluating Model Performance - Precision, Recall, Accuracy 
#4.1 Evaluate 
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
#creating instances
pre = Precision()
rec = Recall()
acc = BinaryAccuracy()

#loop through testing data batch
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    rec.update_state(y, yhat)
    acc.update_state(y, yhat)
print(f'Precision: {pre.result().numpy()}, Recall = {rec.result().numpy()}, Accuracy = {acc.result().numpy()}')

#4.2 test
img = cv2.imread('happyFaceTest.png')
pyplot.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pyplot.show()

#resize to 256x256x3
resize = tensorflow.image.resize(img, (256,256))
pyplot.imshow(resize.numpy().astype(int))
pyplot.show()

#encapsulate shape 
numpy.expand_dims(resize,0)
yhat = model.predict(numpy.expand_dims(resize/255,0))

#round down if 50% for binary classification 
if yhat >0.5:
    print("Predicted Class is Sad")
else:
    print("Predicted Class is Happy")

#--------------------Part 4 Complete-------------------------

#Saving Model for Deployment 
#5.1 Save the model 
from tensorflow.keras.models import load_model
#saving inside models folder
model.save(os.path.join('models','happysadmodel.h5'))
newModel = load_model(os.path.join('models','happysadmodel.h5'))
yhatnew = newModel.predict(numpy.expand_dims(resize/255,0))
if yhatnew >0.5:
    print("Predicted Class is Sad")
else:
    print("Predicted Class is Happy")

#--------------------Part 5 Complete-------------------------
