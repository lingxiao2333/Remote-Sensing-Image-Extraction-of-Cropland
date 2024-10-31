What is this repository for?
Remote Sensing Image Extraction of Cropland, this code is based on the SE-ResUNet network for extracting cropland, which performs well in the extraction of cropland in mountainous areas and is capable of adapting to the extraction of land types in complex regions. 
How do I get set up?
Please follow these steps: Download the files from the code repository in a Windows 64-bit system, and use them on a computer with the TensorFlow framework installed. The code is based on Python 3.8, and you should download the corresponding packages as prompted when running the train file, including opencv, numpy, and others.
Usage
1) Please follow these steps:
•	Download Anaconda, create a virtual environment, and open the virtual environment in software like PyCharm or VSCode to perform operations.
2) First, download the images from the test folder and run the `enhancedata.py` file to enhance the data (note to create folders and replace folder paths as needed). After running, the images will be divided into three folders: train, val, and test. (The default ratio is 8:1:1).
3) Second, run the `train.py` script. The default network is SE-ResUNet (if you need to change the network, you will need to manually modify the comments in the `def train` section to select the desired network). For testing purposes, please set the `rootpath` in the code to the `test` directory within the repository. 
4) Please confirm the size and number of channels of the input image, as well as the number of land cover classes to be categorized (default image size is 256*256, 3 channels, and 3 classifications). Determine whether to generate a confusion matrix image and set the image name accordingly.
5) After the image training is completed, a `save_models` folder will be output in the `save_data` folder within the image directory. This folder contains the trained model, loss and accuracy line charts, and so on.
Who do I talk to?
School of Faculty of Geography, Yunnan Normal University, Kunming, Yunnan Province, 650,500, P. R. China; Email: 1875016877@qq.com
