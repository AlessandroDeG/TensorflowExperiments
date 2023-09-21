
import time

import random
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from tensorflow import keras



#Importing Image module from PIL package  
from PIL import Image  
import PIL  



def convolve3x3(image,filter):

    imagef = np.copy(image)

    size_x = image.shape[0]
    size_y = image.shape[1]
   
    weight=1
    sum=0
    for i in range(0,filter.shape[0]):
        for j in range(0,filter.shape[1]):
            sum+=filter[i][j]
    
    if(sum!=1 and sum!=0):
        weight = 1/sum    #normalize
        
    #print("w="+str(weight))
    
     
        ##Silly arrow
    sillyArrow=False
    
    size=size_y*size_x                            
    if(size>=100):
        sillyArrow=True
        
    if(sillyArrow):
        
        print("**PROCESSING IMAGE OF SIZE : "+str(size)+"**")
        percent= size//100
        rest = size%100
        arrow=[]
        for t in range(0,100):
            arrow.append("-")
        i=0
        
        ###########
    
    
    for x in range(1,size_x-1):
      for y in range(1,size_y-1):
          convolution = 0.0
          convolution+= (image[x - 1, y-1] * filter[0][0])
          convolution+= (image[x, y-1] * filter[0][1])
          convolution+= (image[x + 1, y-1] * filter[0][2])
          convolution+= (image[x-1, y] * filter[1][0])
          convolution+= (image[x, y] * filter[1][1])
          convolution+=(image[x+1, y] * filter[1][2])
          convolution+=(image[x-1, y+1] * filter[2][0])
          convolution+=(image[x, y+1] * filter[2][1])
          convolution+=(image[x+1, y+1] * filter[2][2])
          convolution = convolution * weight
          #print(convolution)
          if(convolution<0):
            convolution = 0
          if(convolution>255):
            convolution= 255
          imagef[x, y] = convolution
        
           ###silly arrow :D
          if(sillyArrow):
              if(i%percent==0 and not (size-i <= rest)):
                  if(i//percent != 0 ):    arrow[(i//percent)-1]="-"
                  arrow[i//percent]=">"
                  print(str(i//percent)+"%" , end = "")
                  for s in arrow:
                      print(s, end=""), 
                  print("100%\r", end=""), 
          i+=1
          
     
    return imagef
    
def convolveNxN(image,filter, filterName="FILTER"):

    imagef = np.copy(image)

    size_x = image.shape[0]
    size_y = image.shape[1]
    
    filter_size_x=filter.shape[0]
    filter_size_y=filter.shape[1]
    filter_center__index_x=filter_size_x//2
    filter_center_index_y=filter_size_y//2
   
    weight=1
    sum=0
    for i in range(0,filter.shape[0]):
        for j in range(0,filter.shape[1]):
            sum+=filter[i][j]
    
    if(sum!=1 and sum!=0):
        weight = 1/sum    #normalize
        
    print("w="+str(weight))
    
     
        ##Silly arrow
    sillyArrow=False
    
    size=size_y*size_x                            
    if(size>=100):
        sillyArrow=True
        
    if(sillyArrow):
        
        print("**"+filterName+" - IMAGE SIZE : "+str(size)+"**")
        percent= size//100
        rest = size%100
        arrow=[]
        for t in range(0,100):
            arrow.append("-")
        i=0
        
        ###########
    
    
    for x in range(filter_center__index_x,size_x-filter_center__index_x):
        for y in range(filter_center_index_y,size_y-filter_center_index_y):
            convolution = 0.0
            
            for fx in range(0, filter_size_x):
                for fy in range(0,filter_size_y):
                    convolution+= (image[x +(fx-filter_center__index_x), y+(fy-filter_center_index_y)] * filter[fx][fy])
                
            convolution = convolution * weight
            #print(convolution)
            if(convolution<0):
                convolution = 0
            if(convolution>255):
                convolution= 255
                
            imagef[x, y] = convolution
        
            ###silly arrow :D
            if(sillyArrow):
                if(i%percent==0 and not (size-i <= rest)):
                    if(i//percent != 0 ):    arrow[(i//percent)-1]="-"
                    arrow[i//percent]=">"
                    print(str(i//percent)+"%" , end = "")
                    for s in arrow:
                        print(s, end=""), 
                    print("100%\r", end=""), 
                i+=1
          
    
    return imagef
          

          

def loadImage(image_path, show=False):
    
    image = tf.keras.preprocessing.image.load_img(image_path)
    image_arr = keras.preprocessing.image.img_to_array(image)  
    
    print("**LOADING IMAGE...**")
    
    image_arr = tf.image.rgb_to_grayscale(image_arr)
    image_arr = np.squeeze(np.array([image_arr]))
    
    #image_arr = image_arr.reshape(image_arr.shape[0],image_arr.shape[1])
    #image_arr= binarizeImage(image_arr)
    
   
    if(show):
        #print("TEST IMAGE:\n\n"+ str(np.squeeze(image_arr))+"\n\n\n")
        print(image_arr.shape)
             
    
              
    return image_arr

       
def binarizeImage(image):  ##0 or 255                    
    for x in range(0,image.shape[0]):
        for y in range(0, image.shape[1]):
            if(image[x,y]>0):
                image[x,y]=1
               
    return image
    

    
    
    
#def sumImages(image1,image2)

#size_x = image.shape[0]
#size_y = image.shape[1]


    
    
#C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\DynamicDataset\\0\\TestReaderNetNEWFILE175
#C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\DynamicDataset\\1\\
#C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\DynamicDataset\\2\\
#C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\DynamicDataset\\3\\
#C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\DynamicDataset\\4\\
#C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\DynamicDataset\\5\\
#C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\DynamicDataset\\6\\
#C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\DynamicDataset\\7\\
#C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\DynamicDataset\\8\\
#C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\DynamicDataset\\9\\

#image = loadImage("C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\DynamicDataset\\0\\TestReaderNetNEWFILE175.png",True)
#zimage = loadImage("C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\test_convolve.png")
mari="\\mari.jpg"
topo="\\topoGrayScale.png"
image = loadImage("C:\\Users\\Ale\\Documents\\Python\\TensorFlow\\ReaderNet\\Resources\\"+mari)

#print("TEST IMAGE:\n\n"+ str(image)+"\n\n\n")
print(image.shape)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(image)
plt.show()

"""
sobelfilterh = [ [-1,-2, -1],[0,0,0],[1,2, 1],]
sobelfilterh=np.array(sobelfilterh)

print(sobelfilterh.shape)
          
sobelImageh = convolve3x3(image,sobelfilterh)

#print("TEST IMAGE:\n\n"+ str(sobelImage)+"\n\n\n")
print(sobelImageh.shape)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(sobelImageh)
plt.show()
"""


mediumLogfilter = [ [0,1,1,2,1,1,0],
                    [1,2,3,5,3,2,1],
                    [1,3,-8,-16,-8,3,1],
                    [2,5,-16,-32,-16,5,2],
                    [1,3,-8,-16,-8,3,1],
                    [1,2,3,5,3,2,1],
                    [0,1,1,2,1,1,0],
                
 ]
mediumLogfilter=np.array(mediumLogfilter)
mediumLogImage = convolveNxN(image,mediumLogfilter, "mediumLog")

#print("TEST IMAGE:\n\n"+ str(mediumLogImage)+"\n\n\n")
print(mediumLogImage.shape)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(mediumLogImage)
plt.show()
#######################################################
sobelfilterh = [ [-1,-2, -1],[0,0,0],[1,2, 1],]
sobelfilterh=np.array(sobelfilterh)

print(sobelfilterh.shape)
          
sobelImageh = convolveNxN(image,sobelfilterh,"SOBEL H")

#print("TEST IMAGE:\n\n"+ str(sobelImage)+"\n\n\n")
print(sobelImageh.shape)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(sobelImageh)
plt.show()
###########################################


sobelfilterv = [ [-1,0, 1],[-2,0,2],[-1,0, 1]]
sobelfilterv=np.array(sobelfilterv)

print(sobelfilterv.shape)
          
sobelImagev = convolveNxN(image,sobelfilterv,"SOBEL V")

#print("TEST IMAGE:\n\n"+ str(sobelImage)+"\n\n\n")
print(sobelImagev.shape)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(sobelImagev)
plt.show()
####################################################

gaussianfilter = [ [0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]]
gaussianfilter=np.array(gaussianfilter)
gaussianImage = convolveNxN(image,gaussianfilter, "LOG")

#print("TEST IMAGE:\n\n"+ str(gaussianImage)+"\n\n\n")
print(gaussianImage.shape)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(gaussianImage)
plt.show()

binGaussianImage = binarizeImage(gaussianImage)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(binGaussianImage)
plt.show()
###############################################################

sobel2filter = [ [-2,-2,0],[-2,0,2],[0,2,2]]
sobel2filter=np.array(sobel2filter)
sobel2Image = convolveNxN(image,sobel2filter,"SUM OF SOBELS?")

#print("TEST IMAGE:\n\n"+ str(sobel2Image)+"\n\n\n")
print(sobel2Image.shape)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(sobel2Image)
plt.show()
#############################################################################
sobelTestfilter = [ [-1,-1,1],[-1,1,3],[1,3,3]]
sobelTestfilter=np.array(sobelTestfilter)
sobelTestImage = convolveNxN(image,sobelTestfilter, "SUM OF SOBELS +1?")

#print("TEST IMAGE:\n\n"+ str(sobelTestImage)+"\n\n\n")
print(sobelTestImage.shape)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(sobelTestImage)
plt.show()
####################################################


Laplacianfilter = [ [-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
Laplacianfilter=np.array(Laplacianfilter)
LaplacianImage = convolveNxN(image,Laplacianfilter, "Laplacian")

#print("TEST IMAGE:\n\n"+ str(LaplacianImage)+"\n\n\n")
print(LaplacianImage.shape)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(LaplacianImage)
plt.show()
###############################################################

Laplacian1filter = [ [1,1,1],[1,-8,1],[1,1,1]]
Laplacian1filter=np.array(Laplacian1filter)
Laplacian1Image = convolveNxN(image,Laplacian1filter, "Laplacian1")

#print("TEST IMAGE:\n\n"+ str(Laplacian1Image)+"\n\n\n")
print(Laplacian1Image.shape)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(Laplacian1Image)
plt.show()

binLaplacianImage = binarizeImage(LaplacianImage)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(binLaplacianImage)
plt.show()


###########################################################

BigLogfilter = [ [0,1,1,2,2,2,1,1,0],
                 [1,2,4,5,5,5,4,2,1],
                 [1,4,5,3,0,3,5,4,1],
                 [2,5,3,-12,-24,-12,3,5,2],
                 [2,5,0,-24,-40,-24,0,5,2],
                 [2,5,3,-12,-24,-12,3,5,2],
                 [1,4,5,3,0,3,5,4,1],
                 [1,2,4,5,5,5,4,2,1],
                 [0,1,1,2,2,2,1,1,0]
 ]
BigLogfilter=np.array(BigLogfilter)
BigLogImage = convolveNxN(image,BigLogfilter, "BigLog")

#print("TEST IMAGE:\n\n"+ str(BigLogImage)+"\n\n\n")
print(BigLogImage.shape)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(BigLogImage)
plt.show()


############################################














    

   