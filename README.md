# Cusco Building Image Dataset (CuscoBID)

<p align="center"><img width="50%" src="nips_2017.jpg" /></p>

We worked on a project that aims to recognize building historic from images in the city of Cusco in Peru. Building recognition from images is a challenging task since pictures can be taken from different angles and under different illumination conditions. An additional challenge is to differentiate buildings with a similar architectural design.

We compare the baseline method Bag-of-Words (We using SIFT and SURF to feature extraction) and the proposed CNN-based method (We propose a transfer learning approach using the models Vgg16, Vgg19, Inception-V3 and Xception to feature extraction).

Contributions are welcome. If you went to Cusco you can send us your photos to increase our dataset. Send to email 120885@unsaac.edu.pe

## Contents

- [Requirements](#requirements)

- [Data Preparation](#data-preparation)

- [Bag-of-Words](#bag-of-words)

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
  
- **Split Train and Test**

  We created N folders and randomdly split the dataset in train and test. Run script
  
  python split_dataset.py ~/Path-to-original-dataset/ nsplits perc_train ~/Path-to-output-dataset/
  
  Where:
  
    - ~/Path-to-original-dataset/ : Directory of input images(dataset_cus)
    
    - nsplits : Number of folders splits
    
    - perc_train : Percentage of train samples
    
    - ~/Path-to-output-dataset/ : Output path
    
## Bag-of-Words
    
- **Build Codebook**

  We used SURF to feature extraction. However, if you want to use another algorithm (for example: SIFT). You must change the line of code 'desc_method = cv2.SURF()' for 'desc_method = cv2.SIFT()' in script bovw_utils.py. Finally, to create the codebook, you need to run.
  
  python codebook.py ~/Path-to-train-dataset/ codebook_size codebook_method ~/Path-to-output-dataset/
  
  Where:
  
    - ~/Path-to-train-dataset/ : Directory of input images(train)
    
    - codebook_size : Size of the dictionary
    
    - codebook_method : Codebook method ('random, kmeans, st_kmeans, fast_st_kmeans). We we recommend using fast_st_kmeans because it is faster
    
    - codebook_filename : Output file(*.npy)

- **Build Bag-of-visual-Words**

  This script needs to be executed twice. one for the train data and the other for the test data.
    
  python bovw.py ~/Path-to-train-dataset/ codebook_filename output_bovw_filename output_labels_filename
   
  Where:
  
    - ~/Path-to-train-dataset/ : Directory of input images(train and test)
    
    - codebook_filename : Codebook file (*.npy)
    
    - output_bovw_filename : Output file(*.npy) with visual words
    
    - output_labels_filename : Output file(*.npy) with labels of visual words

- **Classification**

  
  
              
