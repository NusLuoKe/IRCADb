# ***3D-IRCADb 01 Data Set***
This project is to do some segmentation works on the [***3D-IRCADb 01***](https://www.ircad.fr/research/3d-ircadb-01/) data set.  
This data set contains the original CT scans of 20 patients. Meanwhile, corresponding mask for each organ is also provided.  

## ***Liver Segmentation***

### **1. Original Data & 3D Reconstruction of Liver.**  

![](https://github.com/NusLuoKe/tf_dicom/blob/master/readme_img/ori_mask_1.png)
![](https://github.com/NusLuoKe/tf_dicom/blob/master/readme_img/ori_mask_2.png)    
Above two images are the 3D reconstruction results of the ground truth.  

### **2. Model prediction on training set.(Patient 1) -- DenseUnet**  

![](https://github.com/NusLuoKe/IRCADb/blob/master/readme_img/dense_unet_p1.png)  
Above image shows the prediction result on the training set(Patient 1) of the DenseUnet.  
  
 When training the DenseUnet model. I crop the slices by batches. For each batch, I choose the slice which has the 
 largest area of liver to draw a box, and apply the size of this box to other slices in the batch, making sure the 
 cropped image contains all the information (liver here) needed to feed into the model. When calculate the inference, I 
 just feed the original slice into the model and there are many noises in the output. 


### **3. Model prediction on training set with a very simple post processing -- crop.(Patient 1) -- DenseUnet**  
  
![]()
![]()  

### **4. Model prediction on training set with a very simple post processing -- crop.(Patient 1) -- DenseUnet**  
  
![]()
![]()  



