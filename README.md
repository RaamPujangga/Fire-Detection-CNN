# Fire Detection With Image Processing Using Convolutional Neural Network Algorithm
## Introduction
This repository is my mini-thesis with my partner @iqbal757, where he uses ANN to detect fire. Also, fully supported by master of Python @vajrayudhar. This tutorial will tell you how to use TensorFlow to detect a fire on Windows 10. I haven't tried yet on Windows 8, 7 and Linux.
I'm using TensorFlow-GPU version 1.9.0

This readme describes every step required to get going with fire detection:

1. [Installing Anaconda CUDA, and cuDNN](https://github.com/RaamPujangga/Fire-Detection-CNN/tree/master#1-install-anaconda-cuda-and-cudnn)
2. [Setting up the directory](https://github.com/RaamPujangga/Fire-Detection-CNN/blob/master/README.md#2-set-up-directory)
3. [Download pictures / dataset](https://github.com/RaamPujangga/Fire-Detection-CNN/blob/master/README.md#3-download-dataset)
4. [Preparing Telegram Bot](https://github.com/RaamPujangga/Fire-Detection-CNN/blob/master/README.md#4-prepare-telegram-bot)
5. [Setting Up TensorFlow and Preparing Necessary Packages](https://github.com/RaamPujangga/Fire-Detection-CNN/blob/master/README.md#5-set-up-tensorflow--prepare-necessary-packages)
6. [Training](https://github.com/RaamPujangga/Fire-Detection-CNN/blob/master/README.md#6-training)
7. [Testing](https://github.com/RaamPujangga/Fire-Detection-CNN/blob/master/README.md#7-testing)

Next time, I will make this tutorial on YouTube

# Steps
##  1. Install Anaconda CUDA, and cuDNN
I follow [this YouTube video tutorial by Mark Jay](https://www.youtube.com/watch?v=RplXYjxgZbw), which shows the process for installing CUDA, and cuDNN. You do not need to install TensorFlow as shown in the video, because we will do that later. I suggest that you choose CUDA v9.0.176 and cuDNN v7.1, cause I'm using that version. After that [download](https://docs.anaconda.com/anaconda/install/hashes/Anaconda3-5.2.0-Windows-x86_64.exe-hash/) and install Anaconda. Also, my python version is Python 3.6.5. This is my laptop specification
1.	Processor		: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80 GHz (8 CPUs)
2.	RAM		: 8192 MB
3.	OS	: Windows 10 Enterprise 64-bit
4.	GPU		: NVIDIA GeForce GTX 1050

Version 
1. TensorFlow-GPU version 1.9.0
2. Python 3.6.5
3. [CUDA v9.0.176](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64)
4. [cuDNN v7.1](https://developer.nvidia.com/rdp/cudnn-archive)
5. [Anaconda 5.2.0](https://docs.anaconda.com/anaconda/install/hashes/Anaconda3-5.2.0-Windows-x86_64.exe-hash/)

**Notes : It depends on you to use what version, but I tried several times, and it didn't work. So this is the successful combination between the Tensorflow-GPU Version, CUDA Version, cuDNN Version, Anaconda Version, and Python Version.**

## 2. Set up Directory
This tutorial also requires several additional Python packages.
1. Create a folder directly in C: and name it “tensorflow1”.
2. Create a folder "dataset" **inside tensorflow1 folder**
3. Download [cam.py](https://github.com/RaamPujangga/Fire-Detection-CNN/blob/master/cam.py) and [train.py](https://github.com/RaamPujangga/Fire-Detection-CNN/blob/master/train.py), and put it **inside tensorflow1 folder**

![image](https://user-images.githubusercontent.com/48588826/65870233-ed308600-e3a5-11e9-8281-1f469d149bd2.png)


## 3. Download dataset
TensorFlow needs hundreds of images of an object to train. Here I have 9844 Fire pictures and 8000 Non-fire pictures.
![image](https://user-images.githubusercontent.com/48588826/65851683-fdcd0600-e37d-11e9-832a-93c6f595f4d3.png)
I capture this picture with my partner @iqbal757 for our Mini-thesis. [Download](https://drive.google.com/drive/folders/1T6ZRC8EZXeYVpOdpw25sLrO4zkD8kLSs?usp=sharing) the dataset and put it **inside dataset folder**

## 4. Prepare Telegram Bot
The purpose of making this telegram bot is to notify the user, that the webcam has detected a fire.
#### 4a. Make Telegram Bot
1. Search for the “@BotFather” telegram bot (he’s the one that’ll assist you with creating and managing your bot)
2. Click on or type **/newbot** to create a new bot.
3. Follow instructions and make a new name for your bot. Although, its screen name can be whatever you like. I have chosen “test” as the screen name and “testfiredetection_bot” as its username.
4. Congratulations! You have created your first bot. You should see a new API token generated for it (for example, in the picture, you can see my newly generated token is 944415410:AAFCIgbTCjs-_ZAEPz4YciCGzg5mX_FFF9M)
5. Now you can search for your newly created bot on telegram
6. For more details, see this tutorial in [picture](https://drive.google.com/file/d/1eR0mJnltrvTEQYJQTUSkXaa4hTjNBKq_/view?usp=sharing)

#### 4b. Get Chat ID
After created Telegram Bot we need to know the **chat id** for sending notifications to the user. Follow this step to get the **chat id**
1. Type anything on the bot and send it
2. Open web browser and type
```
https://api.telegram.org/bot**YourBOTToken**/getUpdates
```
3. In this case, my Bot Token is **944415410:AAFCIgbTCjs-_ZAEPz4YciCGzg5mX_FFF9M**
4. So type 
```
https://api.telegram.org/bot944415410:AAFCIgbTCjs-_ZAEPz4YciCGzg5mX_FFF9M/getUpdates
```
5. If the process didn't work, try again from step 1
6. If you make it, you will see the **id : your id** (for example, in the picture, you can see **my id : 389776309**)
7. For more details, see this tutorial in [picture](https://drive.google.com/file/d/1ga6ddiikhoHDYwLv_vONb1P3ytHlTAO8/view?usp=sharing)

## 5. Set Up TensorFlow & Prepare Necessary Packages
Next we'll go to our directory C:\tensorflow1, and follow these instructions :
1. Go to directory C:\tensorflow1
2. Hold shift+right click mouse and click **Open Powershell window here**
3. Type **cmd** and enter
4. Create a new virtual environment called “tensorflow1” by issuing the following command:
```
C:\> conda create -n tensorflow1 pip python=3.6.5
```
5. Install tensorflow-gpu in this environment by issuing:
```
C:\> pip install tensorflow-gpu==1.9.0
```
6. Activate tensorflow by issuing:
```
C:\> activate tensorflow1
```
7. Install the other necessary packages by issuing the following commands:
```
(tensorflow1) C:\> pip install python==3.6.5
(tensorflow1) C:\> pip install pillow==5.4.1
(tensorflow1) C:\> pip install lxml==4.3.2
(tensorflow1) C:\> pip install Cython==0.28.1
(tensorflow1) C:\> pip install numpy==1.16.3
(tensorflow1) C:\> pip install matplotlib==2.1.2
(tensorflow1) C:\> pip install pandas==0.24.2
(tensorflow1) C:\> pip install opencv-python==4.0.0.21
```

## 6. Training
If everything has been set up correctly, TensorFlow will initialize the training.
1. Go to directory C:\tensorflow1
2. Hold shift+right click mouse and click **Open Powershell window here**
3. Type **cmd** and enter
4. Activate tensorflow by issuing:
```
C:\> activate tensorflow1
``` 
5. Start training by issuing:
```
(tensorflow1) C:\> python train.py --image_dir="dataset" --how_many_training_steps=200
```
**Notes : It depends on you, how many training steps (epoch) do you want, the default is 2000. I personally choose 200**

When training begins, it will look like this:

![image](https://user-images.githubusercontent.com/48588826/65884020-47d7db00-e3c2-11e9-971b-4ed8aa0f9538.png)

The first part of the training, it will automatically download [CNN Architecture Inception-v3](https://arxiv.org/pdf/1512.00567).
You can see the result of training on C:/tmp.
Training will automatically stop.

## 7. Testing
Before you test, you should open [cam.py](https://github.com/RaamPujangga/Fire-Detection-CNN/blob/master/cam.py)

1. Replace the video input **at line 12**. Adjust with your video input. If you use your Laptop Webcam, let it be **0**.
2. Replace **line 57** with your **API Token** and your **Chat ID**

If you finish, you can start test to this Fire Detection.
1. Go to directory C:\tensorflow1
2. Hold shift+right click mouse and click **Open Powershell window here**
3. Type **cmd** and enter
4. Activate tensorflow by issuing:
```
C:\> activate tensorflow1
``` 
5. Start testing by issuing:
```
(tensorflow1) C:\> python cam.py
```
6. If you want to stop, press Ctrl+C

If everything is working properly, it will look like this.

![image](https://user-images.githubusercontent.com/48588826/65883235-d21f3f80-e3c0-11e9-9cbd-07b167da22b7.png)

If the percentage above 80%, it will automatically sending notifications to Telegram Bot, like this

![image](https://user-images.githubusercontent.com/48588826/65884531-35aa6c80-e3c3-11e9-8e2c-a4ab190a3ea1.png)

You can change the percentage, by replacing [cam.py](https://github.com/RaamPujangga/Fire-Detection-CNN/blob/master/cam.py) **at line 56**.

Hope you make it till the end.
Cheers.
