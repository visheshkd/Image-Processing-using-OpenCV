#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMAGE PROCESSING WITH PYTHON AND OpenCV
import cv2 #Opencv library
import urllib.request
img_url = "https://upload.wikimedia.org/wikipedia/commons/7/76/Donald_Trump_Justin_Trudeau_2017-02-13_02.jpg"
img_name = "image_test.jpg"

urllib.request.urlretrieve(img_url,img_name)


# In[2]:


#checking if the downaloaded file is in the directory
import os
os.listdir(os.curdir)#showing all file in current directory
#print(file_name in os.listdir(os.curdir))--> to check if the particular file is in directory or not


# In[6]:


#plotting image
from matplotlib import pyplot as plt
img_plot = cv2.imread(img_name)
plt.imshow(img_plot)


# In[8]:


#Fixing colors of above image doing image processing using OpenCV
correct_plot = cv2.cvtColor(img_plot , cv2.COLOR_BGR2RGB)
plt.imshow(correct_plot)


# In[9]:


#remove axis ticks from the image plot
plt.axis("off")
plt.imshow(correct_plot)


# In[11]:


#changing size of the display image
from pylab import rcParams
rcParams['figure.figsize'] = 10 , 10
plt.axis("off")
plt.imshow(correct_plot)


# In[13]:


#convert to grayscale
gray_img = cv2.cvtColor(correct_plot , cv2.COLOR_BGR2GRAY)
plt.axis("off")
plt.imshow(gray_img , cmap="gray")
plt.title("Grayscale image ")


# In[18]:


#Canny Edge Detection
#Algorithm used to detect edges in an image
rcParams['figure.figsize'] = 10 ,12
edge = cv2.Canny(correct_plot , threshold1=100 , threshold2=200) #threshold --> hysteresis thresholding
plt.imshow(edge , cmap = "gray")
plt.title("Edge Image") , plt.xticks([]),plt.yticks([])


# In[19]:


#Analyzing images using histogram 0-255 scale 
import numpy as np
rcParams["figure.figsize"]= 8 , 4
plt.hist(gray_img.ravel() , 256 , [0,256]) #ravel()-->returns a flattened one-dimensional array.
plt.title("histogram of grayscale image")
plt.show()
#lot of images around 0-50 are darkest pixels and around 200-255 are brightest pixels.


# In[23]:


#RGB histogram
color = ('b','g','r')
for i , col in enumerate(color):
    histr = cv2.calcHist([correct_plot] , [i] , None , [256],[0,256])
    plt.plot(histr,color=col)
    plt.xlim([0,256])
plt.show()
#You can see that how blue , green and red pixels are distributed where they are lighter and darker.


# In[ ]:





# In[ ]:




