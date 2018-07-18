# ***3D-IRCADb 01 Data Set***  

This project is to do some segmentation works on the [***3D-IRCADb 01***](https://www.ircad.fr/research/3d-ircadb-01/) data set.  
This data set contains the original CT scans of 20 patients. Meanwhile, corresponding mask for each organ is also provided.  

## ***Liver Segmentation***

### **1. Original Data & 3D Reconstruction of Liver.**  


![](https://github.com/NusLuoKe/tf_dicom/blob/master/readme_img/ori_mask_1.png)
![](https://github.com/NusLuoKe/tf_dicom/blob/master/readme_img/ori_mask_2.png)    
Above two images are the 3D reconstruction results of the ground truth.  

---
### **2. Model prediction on training set.(Patient 1) -- DenseUnet**    


![](https://github.com/NusLuoKe/IRCADb/blob/master/readme_img/dense_unet_p1.png){:height="280px" width="400px"}
Above image shows the prediction result on the training set(Patient 1). 
Prediction result is given by the DenseUnet.    
  
 * When training the DenseUnet model. I crop the slices by batches. For each batch, I choose the slice which has the 
 largest area of liver to draw a box, and apply the size of this box to other slices in the batch, making sure the 
 cropped image contains all the information (liver here) needed to feed into the model. When calculate the inference, I 
 just feed the original slice into the model and there are many noises in the output. 


---
### **3. Model prediction on training set with a very simple post processing -- crop.(Patient 1) -- DenseUnet**  


![](https://github.com/NusLuoKe/IRCADb/blob/master/readme_img/dense_unet_p1_post_processing.png){:height="280px" width="400px"}
Above image shows the prediction result of one slice on the training set(Patient 1) in the ITK-SNAP. 
Prediction result is given by the DenseUnet with post processing.  

* The method to do post processing on the prediction result is very simple, just crop the output slice by a box 
which has a coordinate of a approximately location of the liver in the original slice. So the result is not good as well.
  

---   
### **4. Model prediction on training set. (Patient 1) -- Mask-rcnn**


***NOTE: Implemention of Mask-rcnn is based on [https://github.com/matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN).*** 
  
![](https://github.com/NusLuoKe/IRCADb/blob/master/readme_img/mrcnn_p1.png){:height="280px" width="400px"}  
Above image shows the prediction result of one slice in the ITK-SNAP.  
  
       
![](https://github.com/NusLuoKe/IRCADb/blob/master/readme_img/mrcnn_p1_01.png)
![](https://github.com/NusLuoKe/IRCADb/blob/master/readme_img/mrcnn_p1_02.png)  
Above image shows the prediction result on the training set(Patient 1). 
Prediction result is given by the Mask-rcnn.  

* Comparing to the result given by DenseUet, the segmentation effect is much better.The dice coefficient is around 0.97.

---
### **5. Model prediction on validation set. (Patient 20) -- Mask-rcnn**


![](https://github.com/NusLuoKe/IRCADb/blob/master/readme_img/mrcnn_p20_01.png)
![](https://github.com/NusLuoKe/IRCADb/blob/master/readme_img/mrcnn_p20_02.png)  
Above image shows the prediction result on the test set(Patient 20). 
Prediction result is given by the Mask-rcnn.  

* The segmentation effect on test set is good as well.  
Do not consider those slices without liver but predicted to have liver(noises),the dice coefficient is around 0.94. 

--- 
 

# ***Can do furthur post processing works to eliminate noises.***  
 ***eg: Connected domain algorithm***



