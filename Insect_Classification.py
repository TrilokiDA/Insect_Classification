#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 14:28:44 2018

@author: mtech06
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 15:32:54 2018

@author: mtech11
"""


import numpy as np
from numpy import array
import cv2
import matplotlib.pyplot as plt
import mahotas as mt
import argparse
import math
import os
from os import listdir
from os.path import isfile, join
from sklearn.svm import SVC
from skimage.feature import greycomatrix, greycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
#from matplotlib import pyplot as plt

class Image:
    def LoadImage(self,path):
        img = cv2.imread(path)
        
        return(img)
    def kmeans(self,K,img):
           Z = img.reshape((-1,3))
           Z = np.float32(Z)
           
           # define criteria, number of clusters(K) and apply kmeans()
           criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
           ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
           # Now convert back into uint8, and make original image
           center = np.uint8(center)
           res = center[label.flatten()]
           res2 = res.reshape((img.shape))
           
           return(res2)
    def showimage(self,img):
       
        cv2.imshow('img',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
class Features:
    def extract_texture(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = greycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
        cont = greycoprops(glcm, 'contrast')
        diss = greycoprops(glcm, 'dissimilarity')
        homo = greycoprops(glcm, 'homogeneity')
        eng = greycoprops(glcm, 'energy')
        corr = greycoprops(glcm, 'correlation')
        ASM = greycoprops(glcm, 'ASM')
        features=np.hstack((cont, diss, homo, eng, corr, ASM))
        features=np.vstack(features).squeeze()
        return (features)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # calculate haralick texture features for 4 types of adjacency
        
        textures = mt.features.haralick(gray)
        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean
        """

    def colourhistogram(self,img):
        
        (means, stds) = cv2.meanStdDev(img)
        means=np.vstack(means).squeeze()
        stds=np.vstack(stds).squeeze()
        features=np.hstack((means,stds))
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features=np.vstack(hist).squeeze()
        """
        return(features)
         
    def shapeextract(self,img):
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       blurred = cv2.GaussianBlur(gray, (5, 5), 0)
       thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
       
       i,cnts,h = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        	cv2.CHAIN_APPROX_SIMPLE)
       maxcnt=0
       index=-1
       
       for (i, c) in enumerate(cnts):
           #print("\tSize of contour %d: %d" % (i, len(c)))
           #print(c)
           length=len(c)
           if maxcnt < length:
               maxcnt=length
               index=i
               
       if(maxcnt<64):
           return(0,0,0)
       else:
           return(index,maxcnt,cnts) 
    
        
        
    def drawcontour(self,index,maxcnt,cnts):          
           #print(index)
           #print(maxcnt)        
           cv2.drawContours(img,cnts,index,(255,0,0),3)
           cv2.imshow('boundary',img)
           cv2.waitKey(0)
           cv2.destroyAllWindows()
    def centroiddistance(self,cnts,index):
         maxcnt=cnts[index]
         M = cv2.moments(maxcnt)
         
         maxcnt = np.vstack(maxcnt).squeeze()
         cx = int(M['m10']/M['m00'])
         cy = int(M['m01']/M['m00'])
         N=64
         L=len(maxcnt)
         interval=L/N
         coordinates = np.arange(N,dtype=np.float32)
         index=0
         for i in range(0,N):
             x=maxcnt[i][0]
             y=maxcnt[i][1]
             dist=math.sqrt(math.pow((x-cx),2)+math.pow((y-cy),2))
             #dist=math.ceil(dist)
             coordinates[index]=dist
             i+=int(interval)
             index+=1
            
         return(coordinates)
    def dft(self,coordinates):
        f=np.zeros(64)
        dft = np.abs(cv2.dft(coordinates))
        for i in range(1,len(dft)):
            f[i-1]=dft[i]/dft[0]
        return(f)
        #dft_shift = np.fft.fftshift(dft)
        #magnitude_spectrum = 20*np.log(np.abs(dft_shift))
        #print(magnitude_spectrum)

class Model:
     def __init__(self):
        self.feature_matrix=np.zeros(0)
        
     def train(self,path):
        ImageObj=Image()
        FeatObj=Features()
        train_dir = [os.path.join(path,f) for f in os.listdir(path)]
        out=0
        for t in train_dir:
            files = [os.path.join(t,f) for f in os.listdir(t)]
            for fi in files:
                label=os.path.basename(os.path.dirname(fi))
                #print(fi)
                img=ImageObj.LoadImage(fi)
                #ImageObj.showimage(img)
                res2=ImageObj.kmeans(2,img)
                #ImageObj.showimage(res2)
                
                index,maxcnt,cnts=FeatObj.shapeextract(res2)
                #FeatObj.drawcontour(index,maxcnt,cnts)
                #print(fi)
                if(maxcnt==0):
                    print("deleted:%s"%label)
                    continue
                    
                coordinates=FeatObj.centroiddistance(cnts,index)
                shapefeatures=FeatObj.dft(coordinates)
                colorfeature=FeatObj.colourhistogram(img)
                texturefeature=FeatObj.extract_texture(res2)
                features=np.hstack((shapefeatures,colorfeature,texturefeature,out))
                #features=np.hstack((shapefeatures,colorfeature,out))
                """
                1.Cifuna locuples
                2.Tettigella viridis
                3.Colposcelis signata
                4.Maruca testulalis
                5.Atractomorpha sinensis
                """
                if(label=='Cifunalocuples'):
                    out=1.0
                elif(label=='Tettigellaviridis'):
                    out=2.0
                elif(label=='Colposcelissignata'):
                    out=3.0
                elif(label=='Marucatestulalis'):
                    out=4.0
                elif(label=='Atractomorphasinensis'):
                    out=5.0
                features[-1] = out
                
               
                if self.feature_matrix.size==0:
                        self.feature_matrix=np.append(self.feature_matrix,features)
                else:
                        self.feature_matrix=np.vstack([self.feature_matrix,features])
        #print(self.feature_matrix)
        return(self.feature_matrix) 
                     
        
        
class Classification:
    def svm(self,feature_matrix1,feature_matrix2):
        Train=feature_matrix1[:,:-1]
        Target=np.uint8(feature_matrix1[:,-1:]).ravel()
        #print(Target)
        Test=feature_matrix2[:,:-1]
        Actual=np.uint8(feature_matrix2[:,-1:]).ravel()
        clf = SVC()
        clf.fit(Train, Target)
        prediction=clf.predict(Test)
        #print(prediction)
        result=np.hstack((prediction.reshape(-1,1),Actual.reshape(-1,1)))
        return(result)
    def random_forest(self,feature_matrix1,feature_matrix2):
        Train=feature_matrix1[:,:-1]
        Target=np.uint8(feature_matrix1[:,-1:]).ravel()
        #print(Target)
        Test=feature_matrix2[:,:-1]
        Actual=np.uint8(feature_matrix2[:,-1:]).ravel()
        clf = RandomForestClassifier()
        clf.fit(Train, Target)
        prediction=clf.predict(Test)
        #print(prediction)
        result=np.hstack((prediction.reshape(-1,1),Actual.reshape(-1,1)))
        return(result)
    def knn(self,feature_matrix1,feature_matrix2):
        Train=feature_matrix1[:,:-1]
        Target=np.uint8(feature_matrix1[:,-1:]).ravel()
        #print(Target)
        Test=feature_matrix2[:,:-1]
        Actual=np.uint8(feature_matrix2[:,-1:]).ravel()
        clf = KNeighborsClassifier()
        clf.fit(Train, Target)
        prediction=clf.predict(Test)
        #print(prediction)
        result=np.hstack((prediction.reshape(-1,1),Actual.reshape(-1,1)))
        return(result)
    def naive_bayes(self,feature_matrix1,feature_matrix2):
        Train=feature_matrix1[:,:-1]
        Target=np.uint8(feature_matrix1[:,-1:]).ravel()
        #print(Target)
        Test=feature_matrix2[:,:-1]
        Actual=np.uint8(feature_matrix2[:,-1:]).ravel()
        clf = GaussianNB()
        clf.fit(Train, Target)
        prediction=clf.predict(Test)
        #print(prediction)
        result=np.hstack((prediction.reshape(-1,1),Actual.reshape(-1,1)))
        return(result)
    def accuracy(self,result):
        #print(result)
        correct=0
        for tt in result:
            if(tt[0]==tt[1]):
                correct+=1
        accuracy=float(correct/len(result)) 
        print("Accuracy=%.5f"%(accuracy*100))


def main():
    train="/home/mtech11/Desktop/insect classification/TRAIN2"
    test="/home/mtech11/Desktop/insect classification/TEST2"
    model=Model()
    feature_matrix1=model.train(train)
    feature_matrix2=model.train(test)
    clasify=Classification()
    result=clasify.svm(feature_matrix1,feature_matrix2)
    print("------------SVM RESULT-----------------")
    clasify.accuracy(result)
    result=clasify.knn(feature_matrix1,feature_matrix2)
    print("------------KNN RESULT-----------------")
    clasify.accuracy(result)
    result=clasify.random_forest(feature_matrix1,feature_matrix2)
    print("-------Random Forest RESULT------------")
    clasify.accuracy(result)
    result=clasify.naive_bayes(feature_matrix1,feature_matrix2)
    print("---------Naive Bayes RESULT-------------")
    clasify.accuracy(result)
    
if __name__== "__main__":
  main()

           
