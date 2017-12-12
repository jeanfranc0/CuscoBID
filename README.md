# Cusco Building Image Dataset (CuscoBID)

<p align="center"><img width="50%" src="nips_2017.jpg" /></p>

We worked on a project that aims to recognize building historic from images in the city of Cusco in Peru. Building recognition from images is a challenging task since pictures can be taken from different angles and under different illumination conditions. An additional challenge is to differentiate buildings with a similar architectural design.

We compare the baseline method Bag-of-Words (We using SIFT and SURF to feature extraction) and the proposed CNN-based method (We propose a transfer learning approach using the models Vgg16, Vgg19, Inception-V3 and Xception to feature extraction).

Contributions are welcome. If you went to Cusco you can send us your photos to increase our dataset. Send to email 120885@unsaac.edu.pe

## Contents

- [Requirements](#requirements)

- [Data Preparation](#data-preparation)

## Requirements

- **To Transfer Learning**
  - Python 2.7
  
  - Tensorflow  1.0.0
 
  - Keras  2.0.2 
  
  - Matplotlib 2.0.0.
  
- **To Bag-of-Words**
  
  - Opencv  2.4.11

  - NumPy  1.12.0
  
  - SciPy 0.18.1
  
  - SciKitLearn 0.18.1

## Data Preparation

- **Format**

  Change format to class_imagenumber.jpg. If you use Ubuntu you can execute the following sentence:
      
  //enter to folder catedral and run; output 01_0001.jpg, 01_0002.jpg, ....
      
  ls *.jpg | awk 'BEGIN{ class=1; photo=1; }{ printf "mv \"%s\" %02d_%04d.jpg\n", $0, class, photo++ }' | bash
      
  //enter to folder coricancha and run; output 02_0001.jpg, 02_0002.jpg, ....
   
  ls *.jpg | awk 'BEGIN{ class=2; photo=1; }{ printf "mv \"%s\" %02d_%04d.jpg\n", $0, class, photo++ }' | bash
      
  //enter to folder garcilaso and run; output 03_0001.jpg, 01_0003.jpg, ....
      
  ls *.jpg | awk 'BEGIN{ class=3; photo=1; }{ printf "mv \"%s\" %02d_%04d.jpg\n", $0, class, photo++ }' | bash
  
  ...
  
- **Join Data**
  
  Copy all the images of the folders to a new folder (where we will leave all the images), we recommend the name of "dataset_cus".

  
              
