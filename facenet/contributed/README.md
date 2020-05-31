# Real_time_face_detection and recognition
This is a real-time/recorded-video face detection and recognition project base  on opencv/tensorflow/mtcnn/facenet.    
Face detection is based on [MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html).   
Face embedding is based on [Facenet](https://arxiv.org/abs/1503.03832).   

## Workflow
![](https://github.com/shanren7/real_time_face_recognition/blob/master/images/real%20time%20face%20detection%20and%20recognition.jpg)   

## 1. Install Dependencies
In the below description it is assumed that   
- Tensorflow has been [installed](https://github.com/davidsandberg/facenet/wiki#1-install-tensorflow)   
- the facenet [repo](https://github.com/davidsandberg/facenet) has been cloned, and   
- the [required python modules](https://github.com/davidsandberg/facenet/blob/master/requirements.txt) has been installed.   

## 2. Download the dataset
**1. Download the unaligned images from internet**    
In this case, create your own datset for training your own classifier and place it as a raw folder in a local directory`~/datasets`   
**2. Download the unaligned LFW images from [here](http://vis-www.cs.umass.edu/lfw/lfw.tgz)**   
In this case, extract it to the local directory`~/datasets`   
```
cd ~/datasets   
mkdir -p lfw/raw   
tar xvf ~/Downloads/lfw.tgz -C lfw/raw --strip-components=1   
```
## 3. Set the python path
Set the environment variable `PYTHONPATH` to point to the `src` directory of the cloned repo. This is typically done something like this   
```
export PYTHONPATH=[...]/facenet/src   
```
where `[...]` should be replaced with the directory where the cloned facenet repo resides.   

## 4. Align the dataset
For example, Alignment of the LFW dataset can be done using align_dataset_mtcnn in the align module.   
    
Alignment of the LFW dataset is done something like this:    
```
for N in {1..4}; do \   
python src/align/align_dataset_mtcnn.py \   
~/datasets/lfw/raw \   
~/datasets/lfw/lfw_mtcnnpy_160 \    
--image_size 160 \    
--margin 32 \    
--random_order \    
--gpu_memory_fraction 0.25 \    
& done    
```
The parameter `margin` controls how much wider aligned image should be cropped compared to the bounding box given by the face detector. 32 pixels with an image size of 160 pixels corresponds to a margin of 44 pixels with an image size of 182, which is the image size that has been used for training of the model below.    

## 5. Download pre-trained model (optional)
If you don not have your own trained model that you would like to test and easy way forward is to download a pre-trained model to run the test on. One such model can be found [here](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-). Download and extract the model and place in your favorite models directory (in this example we use `~/models/facenet/`). After extracting the archive there should be a new folder `20180402-114759` with the contents    
```
20180402-114759.pb   
model-20180402-114759.ckpt-275.data-00000-of-00001    
model-20180402-114759.ckpt-275.index    
model-20180402-114759.meta    
```

## 6. Train a classifier
This tutorial describes how to train your own classifier on your own dataset. Here it is assumed that you have followed e.g. the guide Validate on LFW to install dependencies, clone the FaceNet repo, set the python path etc and aligned the LFW dataset (at least for the LFW experiment). In the examples below the frozen model `20170216-091149` is used. Using a frozen graph significantly speeds up the loading of the model.  
    
**1. Train a classifier on LFW:**    
For this experiment we train a classifier using a subset of the LFW images. The LFW dataset is split into a training and a test set. Then a pretrained model is loaded, and this model is then used to generate features for the selected images. The pretrained model is typically trained on a much larger dataset in order to give decent performance (in this case a subset of the MS-Celeb-1M dataset).    
- Split the dataset into train and test sets    
- Load a pretrained model for feature extraction    
- Calculate embeddings for images in the dataset    
- mode=TRAIN:    
 > - Train the classifier using embeddings from the train part of a dataset   
 > - Save the trained classification model as a python pickle    
- mode=CLASSIFY:   
 > - Load a classification model    
 > - Test the classifier using embeddings from the test part of a dataset   

**- Training a classifier on the training set part of the dataset is done as:**     
`python src/classifier.py TRAIN /home/david/datasets/lfw/lfw_mtcnnalign_160`   
`/home/david/models/model-20170216-091149.pb ~/models/lfw_classifier.pkl --batch_size 1000`   
`--min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset`    
   
The output from the training is shown below:    
```
Number of classes: 19   
Number of images: 665   
Loading feature extraction model    
Model filename: /home/david/models/model-20170216-091149.pb    
Calculating features for images    
Training classifier    
Saved classifier model to file "/home/david/models/lfw_classifier.pkl"    
```
   
**- The trained classifier can later be used for classification using the test set:**       
`python src/classifier.py CLASSIFY ~/datasets/lfw/lfw_mtcnnalign_160 ~/models/model-`   
`20170216-091149.pb ~/models/lfw_classifier.pkl --batch_size 1000 --`   
`min_nrof_images_per_class 40 --nrof_train_images_per_class 35 --use_split_dataset` 
   
Here the test set part of the dataset is used for classification and the classification result together with the classification probability is shown. The classification accuracy for this subset is ~0.98.    
   
```
Number of classes: 19   
Number of images: 1202   
Loading feature extraction model   
Model filename: /home/david/models/export/model-20170216-091149.pb   
Calculating features for images   
Testing classifier   
Loaded classifier model from file "/home/david/lfw_classifier.pkl"   
   0  Ariel Sharon: 0.583   
   1  Ariel Sharon: 0.611   
   2  Ariel Sharon: 0.670   
...   
...   
...   
1198  Vladimir Putin: 0.588   
1199  Vladimir Putin: 0.623   
1200  Vladimir Putin: 0.566   
1201  Vladimir Putin: 0.651   
Accuracy: 0.978   
```
   
**2. Train a classifier on your own dataset:**    
So maybe you want to automatically categorize your private photo collection. Or you have a security camera that you want to automatically recognize the members of your family. Then it's likely that you would like to train a classifier on your own dataset. In this case `classifier.py` program can be used also for this. I have created my own train and test datasets and aligned the datset as discussed above.   
   
**- The training of the classifier is done in a similar way as before:**    
`python src/classifier.py TRAIN ~/datasets/my_dataset/train/ ~/models/model-20170216-`   
`091149.pb ~/models/my_classifier.pkl --batch_size 1000` 
   
The training of the classifier takes a few seconds (after loading the pre-trained model) and the output is shown below. Since this is a very simple dataset the accuracy is very good.   
   
```
Number of classes: 10   
Number of images: 50   
Loading feature extraction model   
Model filename: /home/david/models/model-20170216-091149.pb    
Calculating features for images    
Training classifier   
Saved classifier model to file "/home/david/models/my_classifier.pkl"
```
**- Classification on the test set can be ran using:**       
`python src/classifier.py CLASSIFY ~/datasets/my_dataset/test/ ~/models/model-20170216-`       
`091149.pb ~/models/my_classifier.pkl --batch_size 1000`       
```
Number of classes: 10   
Number of images: 50   
Loading feature extraction model   
Model filename: /home/david/models/model-20170216-091149.pb   
Calculating features for images   
Testing classifier   
Loaded classifier model from file "/home/david/models/my_classifier.pkl"   
   0  Ariel Sharon: 0.452   
   1  Ariel Sharon: 0.376   
   2  Ariel Sharon: 0.426   
...   
...  
...  
  47  Vladimir Putin: 0.418   
  48  Vladimir Putin: 0.453   
  49  Vladimir Putin: 0.378   
Accuracy: 1.000   
```
   
This code is aimed to give some inspiration and ideas for how to use the face recognizer, but it is by no means a useful application by itself. Some additional things that could be needed for a real life application include:   
- Include face detection in a face detection and classification pipe line   
- Use a threshold for the classification probability to find unknown people instead of just using the class with the highest probability   

## 7. Run the application
The face detection and recognition model can thus be used to detect faces on live video or a video file. It can be ran using:   
`python contributed/real_time_face_recognition.py`   

