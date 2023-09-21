#ReaderNet

import pygame
import random
import os.path
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


np.set_printoptions(threshold=np.inf)
#################################NETS

class Net:
    def __init__(self, datasetImagesSize=28, outputClassesSize=10, hiddenLayerSize=128):  #hiddenLayerSize=128
        #if(hiddenLayerSize==None):
           # hiddenLayerSize=(datasetImagesSize**2+outputClassesSize)//2
            
        self.loss=1.0
    
        self.datasetImagesSize=datasetImagesSize
        self.hiddenLayerSize=hiddenLayerSize
        self.outputClassesSize=outputClassesSize
        
        
        #fixed number of layers?
        self.defineLayers()
        """
        self.model= keras.Sequential([  
                    keras.layers.Flatten(input_shape=(self.datasetImagesSize,self.datasetImagesSize)),
                    keras.layers.Dense(self.hiddenLayerSize, activation=tf.nn.relu),    #default 128                 
                    keras.layers.Dense(self.outputClassesSize, activation=tf.nn.softmax)     #default 10   
        ])
        """
        
        self.preprocessTrainingSet()
        """
        ##self.defaultDataset = 
        print("**PREPROCESSING TRAINING SET:**")
        (self.default_train_images,self.default_train_labels), (self.default_test_images,self.default_test_labels) = tf.keras.datasets.mnist.load_data()
        self.default_test_images=self.binarizeImages(self.default_test_images)
        self.default_train_images=self.binarizeImages(self.default_train_images)
        self.default_train_images = self.default_train_images/255   #lol    array/255
        self.default_test_images = self.default_test_images/255 
        
        
        
        
        print("**TRAINING SET:**"+str(self.default_train_images.shape))
        random_image = self.default_train_images[random.randint(0,self.default_train_images.shape[2]),:,:]
        print("**RANDOMLY PICKED TRAINING SET IMAGE:**\n"+str(random_image))
        #plt.imshow(random_image)
        #plt.show()
        """
        
        
        
        
        self.compile()
        """
        print("***NEW NET***")
        self.model.summary()
        
        self.model.compile(tf.optimizers.Adam()  ,     #check optimizers implementation https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam               
              loss= "sparse_categorical_crossentropy")   
        """
        
     
    def defineLayers(self, hiddenLayerSize=128):
        self.model= keras.Sequential([  
                    keras.layers.Flatten(input_shape=(self.datasetImagesSize,self.datasetImagesSize)),
                    keras.layers.Dense(hiddenLayerSize, activation=tf.nn.relu),    #default 128                 
                    keras.layers.Dense(self.outputClassesSize, activation=tf.nn.softmax)     #default 10   
        ])
    
    
    def preprocessTrainingSet(self):
        print("**PREPROCESSING TRAINING SET:**")
        (self.default_train_images,self.default_train_labels), (self.default_test_images,self.default_test_labels) = tf.keras.datasets.mnist.load_data()
        self.default_test_images=self.binarizeImages(self.default_test_images)
        self.default_train_images=self.binarizeImages(self.default_train_images)
        self.default_train_images = self.default_train_images/255   #lol    array/255
        self.default_test_images = self.default_test_images/255        
        print("**TRAINING SET:**"+str(self.default_train_images.shape))
        random_image = self.default_train_images[random.randint(0,self.default_train_images.shape[2]),:,:]
        print("**RANDOMLY PICKED TRAINING SET IMAGE:**\n"+str(random_image))
        
    def compile(self):
        print("***NEW NET***")
        self.model.summary()       
        self.model.compile(tf.optimizers.Adam()  ,     #check optimizers implementation https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam               
              loss= "sparse_categorical_crossentropy")
        
       
        
          
    def startTraining(self, epochsN=20, train_images=None, train_labels=None): #returns evaluation array
        ##apparently self is defined only inside the func so cant use it directly to set defaults in args... so:
        if(train_images==None):    train_images=self.default_train_images
        if(train_labels==None):    train_labels=self.default_train_labels
        ##
      
        self.model.fit(train_images, train_labels,epochs=epochsN)
        evaluation = self.model.evaluate(self.default_test_images,self.default_test_labels)
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
    
    def binarizeImages(self, image_arr):  ##0 or 255
        #print("\n**BINARIZE**\n")
        for i in range(0,len(image_arr)):
            image=image_arr[i]
            #print("image:"+str(image))
            #print("shape:"+str(image.shape))
            #print("shape[0]:"+str(image.shape[0]))
            #print("shape[1]"+str(image.shape[1]))
            for x in range(0,image.shape[0]):
                for y in range(0, image.shape[1]):
                    if(image[x,y]>0):
                        image[x,y]=255
                        image_arr[i]=image
        return image_arr
        
    #def optimizeStructure:
    
    
class OptimizerNet:#Experiment
    def __init__(self, net):
        self.net=net
        
        self.model = keras.Sequential([
                     keras.layers.Flatten(input_shape=[2]),     #in = hiddenlayerSize,epoch
                     keras.layers.Dense(512, activation=tf.nn.tanh),
                     keras.layers.Dense(128, activation=tf.nn.relu),
                     keras.layers.Dense(64, activation=tf.nn.tanh),
                     
                     #keras.layers.Dense(124, activation=tf.nn.tanh),             
                     keras.layers.Dense(units=1)  #out = loss
        ])
              
        self.model.compile(optimizer="sgd" ,loss = "mean_squared_error")
        
        self.logger = Logger()
        
        self.trainingList=[]    #tuples(hiddenLayerSize,epochs)
        self.trainingLabelsList=[]      #loss
        self.testList=[]    #tuples(hiddenLayerSize,epochs)
        self.testLabelsList=[]
        
        
        print("\n\n***CREATING OPTIMIZER DATASET***")
        #self.fillOptimizerTrainingSet()
        
        
        parsedData= self.logger.parseNetSetCSV("C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\Optimizer\\Nets\\NetsetsTraining\\netsetTraining2020-09-19-01-10-03-963943.txt")
        self.trainingList=parsedData[0]
        self.trainingLabelsList=parsedData[1]
        
        parsedData= self.logger.parseNetSetCSV("C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\Optimizer\\Nets\\NetsetsTesting\\netsetTesting2020-09-19-01-10-03-963943.txt")
        self.testList=parsedData[0]
        self.testLabelsList=parsedData[1]
        
        
        self.trainingList=np.array(self.trainingList)
        self.trainingLabelsList=np.array(self.trainingLabelsList)
        self.testList=np.array(self.testList)
        self.testLabelsList=np.array(self.testLabelsList)
        
        print("\n\n***TRAINING OPTIMIZER***")
        self.model.summary()  
        self.loss= self.startTraining()
        
        
        self.predictions=[]
        self.hiddenValues=[]
        self.epochValues=[]
        
        print("\n***PREDICTING OPTIMIZED NET***")
        """
        self.logger.setType(Logger.TYPE_OPTIMIZER_PREDICTIONS)
        #for hidden in range(1,28**2):
        for hidden in range(1,300):
            for epoch in range(1,15):
                self.hiddenValues.append(hidden)
                self.epochValues.append(epoch)
                #arr=arr.reshape(2)
                self.predicted=self.model.predict((np.array([(hidden,epoch)])))
                self.predictions.append(self.predicted[0])
                self.logger.log(str(hidden)+";"+str(epoch)+";"+str(self.predicted[0]))
                print(str(hidden)+";"+str(epoch)+";"+str(self.predicted[0]))
        """
        parsedData= self.logger.parseNetSetCSV("C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\Optimizer\\Nets\\OptimizerPredictions\\optimizerPredictions2020-09-19-11-47-45-535611.txt")
        self.predictedList=parsedData[0]
        self.predictedLabelsList=parsedData[1]
        
        self.hiddenValues = self.predictedList[0]
        self.epochValues = self.predictedList[1]
        self.predictions = self.predictedLabelsList
        
        print(str(self.predictedList)+"  "+str(self.predictedLabelsList))
        
                
        
        x=self.trainingList[0]
        y=self.trainingList[1]
        loss=self.trainingLabelsList
        px=self.hiddenValues
        py=self.epochValues
        ploss= self.predictions
        
                 
        print(str(px)+"  "+str(py)+"  "+str(ploss))
                
        ax = plt.axes(projection='3d')

        # Data for a three-dimensional line
        zline = np.array(ploss)
        xline = np.array(px)
        yline = np.array(py)
        ax.plot3D(xline, yline, zline, 'gray')

        # Data for three-dimensional scattered points
        zdata = np.array(loss)
        xdata = np.array(x)
        ydata = np.array(y)
        ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');
        
        ax.show()
        
        
                
        self.minLoss=1
        self.index=0
        self.minIndex=0
        
        for predicted in self.predictions:
            if(predicted[0] < self.minLoss):
                self.minLoss=predicted[0]
                self.minIndex=self.index
            self.index+=1
        
        print("\n\n***OPTIMIZED VALUES***")        
        print("loss:"+str(self.minLoss)+" hidden="+str(self.hiddenValues[self.minIndex])+"epoch="+str(self.epochValues[self.minIndex]))
        
        print("\n***TESTING OPTIMIZED NET***")
        
        self.net.defineLayers(self.hiddenValues[self.minIndex])
        self.net.compile()
        self.testedOptimized=self.net.startTraining(self.epochValues[self.minIndex])
        
        self.logger.setType(Logger.TYPE_OPTIMIZED_NET)
        self.logger.log(str(self.hiddenValues[self.minIndex])+";"+str(self.epochValues[self.minIndex])+";"+str(self.predictions[self.minIndex])+";"+str(self.testedOptimized))
        
        
    def fillOptimizerTrainingSet(self): #15 #5
        self.logger.setType(Logger.TYPE_NETSET_TRAINING)
        for hidden in range(1, 200, 10):    #trainingSet
            self.net.defineLayers(hidden)
            self.net.compile()
            for epoch in range(1,10,1):
                self.trainingList.append((hidden,epoch))
                trainingLabel=self.net.startTraining(epoch)
                self.trainingLabelsList.append(trainingLabel)
                self.logger.log(str(hidden)+";"+str(epoch)+";"+str(trainingLabel))
                
        self.logger.setType(Logger.TYPE_NETSET_TESTING)  
        for hidden in range(25, 950, 100):     #testSet
            self.net.defineLayers(hidden)
            self.net.compile()
            for epoch in range(2,15,3):
                self.testList.append((hidden,epoch))
                testLabel=self.net.startTraining(epoch)
                self.testLabelsList.append(testLabel)
                self.logger.log(str(hidden)+";"+str(epoch)+";"+str(testLabel))
                
                             
    def startTraining(self):           
        self.model.fit(self.trainingList, self.trainingLabelsList,epochs=10)
        evaluation = self.model.evaluate(self.testList,self.testLabelsList)
        #print(self.model.metrics_names)
        #return evaluation["loss"]
        print("**EVALUTAION**"+str(evaluation))
        return evaluation
                
    
                
                
                
                
                
    
        
            
            
class NetThread(threading.Thread):
    def __init__(self, net=Net()):
        super(NetThread, self).__init__()
        self.net=net
        self.netReady=False
        
    def run(self):
        #self.net.startTraining()
        self.netReady=True
        
        
        
    def preprocessImage(self,image_path):
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
        image_arr = image_arr.reshape(1, 28,28)
        
        print("\nTEST IMAGE:\n\n\n"+ str(image_arr)+"\n\n\n")
        print(image_arr.shape)
        image_arr=self.net.binarizeImages(image_arr)/255
        print("TEST IMAGE BINARIZED:\n\n\n"+ str(image_arr)+"\n\n\n")
        print(image_arr.shape)
        
        return image_arr
        
        
class Logger:
    TYPE_GENERAL=0
    TYPE_NETSET_TRAINING=1
    TYPE_NETSET_TESTING=2
    TYPE_OPTIMIZED_NET=3
    TYPE_OPTIMIZER_PREDICTIONS=4
    
    def __init__(self):
         
        #THIS FILE.py NAME
        self.filename = inspect.getframeinfo(inspect.currentframe()).filename    
        #DIRECTORIES
        self.path=os.path.dirname(os.path.abspath(self.filename))    
        self.parentDirectory="\\Resources\\Optimizer"
        self.logsDirectory="\\Logs"#n
        self.netsDirectory="\\Nets"
        self.netSetTrainingDirectory="\\NetsetsTraining"  #n
        self.netSetTestingDirectory="\\NetsetsTesting"  #n
        self.optimizedPredictionsDirectory="\\OptimizerPredictions"  #n
        
        #NAMES 
        self.netSetTrainingName="\\netsetTraining"  #modify this
        self.netSetTestingName="\\netsetTesting"  #modify this
        self.optimizerPredictionsName="\\optimizerPredictions"
        self.optimizedNetsName="\\OptimizedNets" 
        self.OptimizedNetName="\\net"
        #FORMATS        
        self.fileFormat=".txt"
                  
        self.now = str(datetime.now()).replace(".","-").replace(" ","-").replace(":","-")
        
             
    #LOG INTO         
    def setType(self,type): #retains initialized timeNOW 
        self.newPath=""
        if(type==self.TYPE_NETSET_TRAINING):
            self.newPath+=(self.path+self.parentDirectory+self.netsDirectory+self.netSetTrainingDirectory+self.netSetTrainingName+self.now+self.fileFormat)
        if(type==self.TYPE_NETSET_TESTING):
            self.newPath+=(self.path+self.parentDirectory+self.netsDirectory+self.netSetTestingDirectory+self.netSetTestingName+self.now+self.fileFormat)
        if(type==self.TYPE_OPTIMIZER_PREDICTIONS):
            self.newPath+=(self.path+self.parentDirectory+self.netsDirectory+self.optimizedPredictionsDirectory+self.optimizerPredictionsName+self.now+self.fileFormat)
        if(type==self.TYPE_OPTIMIZED_NET):
            self.newPath+=(self.path+self.parentDirectory+self.netsDirectory+self.optimizedNetsName+self.fileFormat)
    
        
        
        
    def log(self,dataString):
        dataString+="\n"
              
        try:
            fileWriter = open(self.newPath, "x")
            fileWriter.close()
            fileWriter = open(self.newPath, "a+")
            fileWriter.write(str(dataString))
            fileWriter.close()
        except BaseException as e:
            # print(e)
            fileWriter = open(self.newPath, "a+")
            fileWriter.write(str(dataString))
            fileWriter.close()
            
    
    def parseNetSetCSV(self, path):  
        trainingList=[]
        trainingLabelsList=[] 
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            line_count = 0
            for row in csv_reader:
                trainingList.append((float((row[0])),float(row[1])))
                trainingLabelsList.append(float(row[2]))           
                line_count += 1
            print("NETSETSIZE ="+str(line_count))
        return(trainingList,trainingLabelsList)
    
    
   
            
    
            
           
        
 

#######################################


##initialize and trains net with all default values
netThread = NetThread() 
netThread.daemon=True
netThread.start()

netOptimizer = OptimizerNet(netThread.net)

delay=10
   
while(not netThread.netReady):
    time.sleep(delay)           #N of seconds!!??
   
if(netThread.netReady):
    delay=15                   #ms


print("NET READY:"+str(netThread.netReady))



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
            if(event.key == pygame.K_KP_ENTER or event.key == pygame.K_RETURN and netThread.netReady):
                
                if(netThread.netReady):
                    
                    #pygame.image.save(screen,path+"\\test.jpg")
                    screenshot = screen.copy()         
                    screenshot = pygame.transform.scale(screenshot, (screenSize//scalingFactor, screenSize//scalingFactor))
                
                    
                    ###TODO REFACTOR NAMING PROCESS TO OBJECT
                    parentDirectory="\\Resources"
                    newDirectory="\\DynamicDataset"                     
                    #labelDirectory=    #make !=dir for every label
                    newFileName="\\testReaderNetNEWFILE"               
                    newFileFormat=".png"
                    newpath=""
                    exist=True
                    counter=0
                 
                    while(exist):
                        exist = os.path.isfile(path+parentDirectory+newDirectory+newFileName+str(counter)+newFileFormat)
                        ##print(str(exist)+str(newFileName))
                        if(exist):                   
                           counter+=1
                           #print("newname"+newFileName)
                        else:               
                           print("saving :"+path+parentDirectory+newDirectory+newFileName+str(counter)+newFileFormat)
                           pygame.image.save(screenshot,path+parentDirectory+newDirectory+newFileName+str(counter)+newFileFormat)
                           exist=False
                           
                
                    testImage = netThread.preprocessImage(path+parentDirectory+newDirectory+newFileName+str(counter)+newFileFormat)
                    result = netThread.net.predict(testImage)
                    print(result)
                    max=0
                    n =0
                    counter=0
                    for i in result[0]:
                        if(i>max):
                            max=i
                            n=counter   
                        counter+=1   
                    print("predicted="+str(n)+"  with confidence=" + str(max))
                
                else:  #netThread not ready
                    print("WAIT! ...Net still is training...")
              
                       
                      
                
                 #pygame.image.save(screenshot,path+"\\test.png")
        
        if(not netThread.netReady):
            delay=1000
        else:
            delay=15
                
        
           
            
        pygame.display.flip()
        #pygame.display.update()
        pygame.time.delay(delay)
        
