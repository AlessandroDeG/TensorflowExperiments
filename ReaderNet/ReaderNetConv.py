#ReaderNet

import pygame
import random
import os.path
from os import walk
import inspect
import threading
import time
import sys
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
import time
from datetime import datetime
import csv


#Importing Image module from PIL package  
from PIL import Image  
import PIL  
 
 
np.set_printoptions(threshold=np.inf)
#################################NETS

class Net:
    def __init__(self, datasetImagesSize=28, outputClassesSize=10, hiddenLayerSize=200):  #hiddenLayerSize=128
        #if(hiddenLayerSize==None):
           # hiddenLayerSize=(datasetImagesSize**2+outputClassesSize)//2
            
        self.loss=1.0  
        self.datasetImagesSize=datasetImagesSize
        self.hiddenLayerSize=hiddenLayerSize
        self.outputClassesSize=outputClassesSize
              
        #fixed number of layers?
        self.defineLayers()
        self.preprocessTrainingSet()      
        self.compile()
   
        
     
    def defineLayers(self, hiddenLayerSize=128):
        self.model= keras.Sequential([  
                    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(13, 13, 64)),
                    tf.keras.layers.MaxPooling2D(2, 2),
                    #tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(5, 5, 64)),
                    #tf.keras.layers.MaxPooling2D(2, 2),
                    keras.layers.Flatten(),
                    #keras.layers.Flatten(input_shape=(self.datasetImagesSize,self.datasetImagesSize)),
                    keras.layers.Dense(hiddenLayerSize, activation=tf.nn.relu),    #default 128                 
                    keras.layers.Dense(self.outputClassesSize, activation=tf.nn.softmax)     #default 10   
        ])
    
    
    def preprocessTrainingSet(self):
        print("\n\n**PREPROCESSING TRAINING DATASET:**")
        (self.default_train_images,self.default_train_labels), (self.default_test_images,self.default_test_labels) = tf.keras.datasets.mnist.load_data()
         
        self.default_test_images=self.binarizeImages(self.default_test_images)
        self.default_train_images=self.binarizeImages(self.default_train_images)
        #self.default_train_images = self.default_train_images/255   #lol    array/255
        #self.default_test_images = self.default_test_images/255

         #add dynamic dataset
        print("\n**LOADING USER TRAINING DATASET:**")
        (self.dynamic_images, self.dynamic_labels) = self.loadDynamicDataset()
        self.default_train_images = np.concatenate((self.default_train_images,self.dynamic_images))
        self.default_train_labels = np.concatenate((self.default_train_labels,self.dynamic_labels))
        
        
        
        random_image = self.default_train_images[random.randint(0,self.default_train_images.shape[2]),:,:]
        print("**RANDOMLY PICKED TRAINING DATASET IMAGE:**\n"+str(np.squeeze(random_image)))
        
        self.default_train_images=self.default_train_images.reshape(-1,28,28,1)
        self.default_test_images=self.default_test_images.reshape(-1,28,28,1)
        
        #reshape for later evaluation
        self.dynamic_images=self.dynamic_images.reshape(-1,28,28,1)
       
        #self.default_train_labels=self.default_train_labels.reshape(-1,28,28,1)
         
        print("\n**TRAINING DATASET:**"+str(self.default_train_images.shape))
        
    def loadDynamicDataset(self):
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path = os.path.dirname(os.path.abspath(filename))
        parentDirectory="\\Resources"
        dynamicDatasetDirectory="\\DynamicDataset"
        
        dynamic_images=[]
        dynamic_labels=[]
        
        ##Silly arrow
        sillyArrow=False
        countAll=0
        for (dirpath, dirnames, filenames) in walk(path+parentDirectory+dynamicDatasetDirectory):
            for dir in dirnames:
                files = os.listdir(dirpath+"\\"+dir)                
                for file in files:
                    countAll+=1
                                   
        if(countAll>=100):
            sillyArrow=True
        
        if(sillyArrow):
            print("**PROCESSING "+str(countAll)+" IMAGES**")
            percent= countAll//100
            rest = countAll%100
            arrow=[]
            for t in range(0,100):
                arrow.append("-")
        
        
        ###########
        i=0
        for (dirpath, dirnames, filenames) in walk(path+parentDirectory+dynamicDatasetDirectory):
            for dir in dirnames:
                files = os.listdir(dirpath+"\\"+dir)
                for file in files:
                    dynamic_images.append(np.squeeze(self.preprocessImage(dirpath+"\\"+dir+"\\"+file)))
                    dynamic_labels.append(np.uint8(int(dir)))
                    ###silly arrow :D
                    if(sillyArrow):
                        if(i%percent==0 and not (countAll-i <= rest)):
                            if(i//percent != 0 ):    arrow[(i//percent)-1]="-"
                            arrow[i//percent]=">"
                            print(str(i//percent)+"%" , end = "")
                            for s in arrow:
                                print(s, end=""), 
                            print("100%\r", end=""), 
                    i+=1
                    #print(i)                    
            ##############
                     
        return (np.array(dynamic_images),np.array(dynamic_labels))
        
    
    def preprocessImage(self,image_path, show=False):
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_arr = keras.preprocessing.image.img_to_array(image)   ###ITS A 3D NUMPY ARRAY!!!!! for rgb channels!! 
        #image_arr = np.array([image_arr])
        """
        print("TEST IMAGE:\n\n\n"+ str(image_arr)+"\n\n\n")
        print(image_arr.shape)
        image_arr = np.squeeze(image_arr)
        """
        image_arr = tf.image.rgb_to_grayscale(image_arr)
        #print("TEST IMAGE:\n\n\n"+ str(image_arr)+"\n\n\n")
        #print(image_arr.shape)
        image_arr = np.array([image_arr])
        #image_arr = np.squeeze(image_arr)
        
        
        #print("\nTEST IMAGE:\n\n\n"+ str(np.squeeze(image_arr))+"\n\n\n")
        #print(image_arr.shape)
        image_arr=self.binarizeImages(image_arr)
        
        if(show):
            print("TEST IMAGE:\n\n"+ str(np.squeeze(image_arr))+"\n\n\n")
            print(image_arr.shape)
             
        image_arr = image_arr.reshape(1, 28,28, 1)
              
        return image_arr
        
    def compile(self):
        print("\n***TRAINING NEW NET***")
        self.model.summary()       
        self.model.compile(tf.optimizers.Adam()  ,     #check optimizers implementation https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam               
              loss= "sparse_categorical_crossentropy")
        
       
        
          
    def startTraining(self, epochsN=10, train_images=None, train_labels=None): #returns evaluation array
        ##apparently self is defined only inside the func so cant use it directly to set defaults in args... so:
        if(train_images==None):    train_images=self.default_train_images
        if(train_labels==None):    train_labels=self.default_train_labels
        ##
      
        self.model.fit(train_images, train_labels,epochs=epochsN)
        print("*EVALUATION on default_dataset*")
        evaluation = self.model.evaluate(self.default_test_images,self.default_test_labels)
        #print(type(self.default_test_images))
        #print(type(self.default_test_images[0]))
        #print(type(self.default_test_labels))
        #print(type(self.default_test_labels[0]))
        print("*EVALUATION on user_dataset*")
        #print(type(self.dynamic_images))
        #print(type(self.dynamic_images[0]))
        #print(type(self.dynamic_labels))
        #print(type(self.dynamic_labels[0]))
        
        evaluation = self.model.evaluate(self.dynamic_images,self.dynamic_labels)
        #print(self.model.metrics_names)
        #self.loss=evaluation["loss"]
        #return evaluation["loss"]
        print("\n**FINISHED TRAINING***\n")
        return evaluation
        
        
    def predict(self,test_images):  ##returns classification array
        #plt.imshow(np.squeeze(test_images))
        #plt.show()
        #print("TEST IMAGE:\n\n\n"+ str(test_images)+"\n\n\n")
        return self.model.predict(test_images)
    
    def binarizeImages(self, image_arr):  ##0 or 1
        
         ###silly arrow :D
        sillyArrow=False
        if(len(image_arr)>=100):
            sillyArrow=True
        
        if(sillyArrow):
            print("\n**PROCESSING "+str(len(image_arr))+" IMAGES**")
            percent= len(image_arr)//100
            arrow=[]
            for t in range(0,100):
                arrow.append("-")
        ##########################
            
        for i in range(0,len(image_arr)):
            
            ###silly arrow :D
            if(sillyArrow):
                if(i%percent==0):
                    if(i//percent != 0 ):    arrow[(i//percent)-1]="-"
                    arrow[i//percent]=">"
                    print(str(i//percent)+"%" , end = "")
                    for s in arrow:
                        print(s, end=""), 
                    print("100%\r", end=""),    
            ##############
             
            image=image_arr[i]
            #print("image:"+str(image))
            #print("shape:"+str(image.shape))
            #print("shape[0]:"+str(image.shape[0]))
            #print("shape[1]"+str(image.shape[1]))
            for x in range(0,image.shape[0]):
                for y in range(0, image.shape[1]):
                    if(image[x,y]>0):
                        image[x,y]=1
                        image_arr[i]=image
        return image_arr
        
#######################################CMD INPUTS
class AskForInput(threading.Thread):
    # active = True
   
    yn = ""
    label=""
    predictedLabel=""
    validLabels=["0","1","2","3","4","5","6","7","8","9"]
    delete ="del"
    
    def __init__(self, predictedLabel):
        super(AskForInput, self).__init__()
        AskForInput.yn = ""
        AskForInput.label=""
        AskForInput.predictedLabel=predictedLabel
        
        
        
    def run(self):
        print("CORRECT ?")
        
        AskForInput.yn=""
        print("y = yes , n = no")
        AskForInput.yn = input()
        while(AskForInput.yn != "y" and AskForInput.yn != "n"):
            print("y = yes , n = no")
            AskForInput.yn = input()
            
        
        if(AskForInput.yn=="n"):
            while(AskForInput.label not in AskForInput.validLabels or AskForInput==AskForInput.delete):
                 print("CORRECT = ")
                 AskForInput.label = input()
        else:
            AskForInput.label= AskForInput.predictedLabel
                
            
        #print("closed")





######################


##initialize and trains net with all default values
net = Net()
net.startTraining()

delay=10
   
   
pygame.init()
pygame.font.init()


filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

print(path)
print(filename)

#pygame.image.save(screen,"test.jpg")

datasetSize=28   ##size of images n*n
scalingFactor=10
screenSize=datasetSize*scalingFactor

screen = pygame.display.set_mode((screenSize, screenSize), 0)





running = True


mouseDrawingPositions=[]
#print(type(mouseDrawingPositions))

drawing=False

WHITE = (255,255,255)



             



print("\nDRAW\n")
while(running):
    screen.fill(0)

    for pos in mouseDrawingPositions:
         pygame.draw.rect(screen,WHITE,(pos[0]-scalingFactor,pos[1]-scalingFactor,scalingFactor*2,scalingFactor*2))
    

    for event in pygame.event.get():
        #print(event)
        
        if(event.type == pygame.QUIT):
             pygame.display.quit()
             pygame.quit()
             sys.exit()
             
        if(event.type == pygame.MOUSEBUTTONDOWN):
            drawing=True
        
        if(event.type == pygame.MOUSEBUTTONUP):
            drawing=False        
        
        if(event.type == pygame.MOUSEMOTION and drawing):
            mouseDrawingPositions.append(pygame.mouse.get_pos())
            
        
        if(event.type == pygame.KEYDOWN):
            #print(event)
            if(event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE or event.key == pygame.K_CLEAR  ):
                print("clear")
                #print(len(mouseDrawingPositions))
                #del mouseDrawingPositions[:]
                mouseDrawingPositions.clear()
                #print(len(mouseDrawingPositions))
            if(event.key == pygame.K_KP_ENTER or event.key == pygame.K_RETURN):
                
                
                    
                    #pygame.image.save(screen,path+"\\test.jpg")
                    screenshot = screen.copy()         
                    screenshot = pygame.transform.scale(screenshot, (screenSize//scalingFactor, screenSize//scalingFactor))
                
                    
                    ###TODO REFACTOR NAMING PROCESS TO OBJECT
                    parentDirectory="\\Resources"
                    dynamicDatasetDirectory="\\DynamicDataset"
                    userInputDirectory="\\UserInputs"                    
                    #labelDirectory=    #make !=dir for every label
                    newFileName="\\testReaderNetNEWFILE"               
                    newFileFormat=".png"
                    newpath=""
                    exist=True
                    counter=0
                 
                    while(exist):
                        exist = os.path.isfile(path+parentDirectory+userInputDirectory+newFileName+str(counter)+newFileFormat)
                        ##print(str(exist)+str(newFileName))
                        if(exist):                   
                           counter+=1
                           #print("newname"+newFileName)
                        else:               
                           print("saving :"+path+parentDirectory+userInputDirectory+newFileName+str(counter)+newFileFormat)
                           pygame.image.save(screenshot,path+parentDirectory+userInputDirectory+newFileName+str(counter)+newFileFormat)
                           exist=False
                           
                    originalUserImage=Image.open(path+parentDirectory+userInputDirectory+newFileName+str(counter)+newFileFormat)
                    
                    testImage = net.preprocessImage(path+parentDirectory+userInputDirectory+newFileName+str(counter)+newFileFormat, True)
                    result = net.predict(testImage)
                    print(result)
                    max=0
                    n =0
                    ncounter=0
                    for i in result[0]:
                        if(i>max):
                            max=i
                            n=ncounter   
                        ncounter+=1   
                    print("\nPREDICTED="+str(n)+"  with confidence=" + str(max)+"\n")
                    
                    askforinput = AskForInput(str(n))            
                    askforinput.daemon=True
                    askforinput.start()
                    yn = askforinput.yn
                    label = askforinput.label
                    while(yn != "y" and yn != "n" or (label not in askforinput.validLabels)):  ##waiting for input and prevent stupid pygame event to crash
                        yn = askforinput.yn
                        label = askforinput.label
                        for event in pygame.event.get():
                            if(event.type == pygame.QUIT):
                                pygame.display.quit()
                                pygame.quit()
                                sys.exit()
                    
                    if(label!=AskForInput.delete):                    
                        print("saving : "+path+parentDirectory+dynamicDatasetDirectory+"\\"+label+newFileName+str(counter)+newFileFormat)
                        originalUserImage.save(path+parentDirectory+dynamicDatasetDirectory+"\\"+label+newFileName+str(counter)+newFileFormat)
                    else:
                        print("DELETED")
                    
                    
                    
                    print("\nDRAW\n")
                    
                    
                         
                 #pygame.image.save(screenshot,path+"\\test.png")
        
      
                
                 
        pygame.display.flip()
        #pygame.display.update()
        pygame.time.delay(delay)