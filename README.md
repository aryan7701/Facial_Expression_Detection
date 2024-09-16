# Emotion_detection_with_CNN
Developed a convolutional neural network (CNN) model to detect and classify human emotions from images and videos, utilizing 7 distinct emotion classes. Implemented the model to process input data, effectively distinguishing emotions with test data, achieving a test accuracy of more than 70.29%. Demonstrated the model's capability through significant performance metrics such as accuracy and loss during training and evaluation phases.

### Packages need to be installed
- pip install numpy
- pip install opencv-python
- pip install keras
- pip3 install --upgrade tensorflow
- pip install pillow

### download FER2013 dataset
- from below link and put in data folder under your project directory
- https://www.kaggle.com/msambare/fer2013

### Train Emotion detector
- with all face expression images in the FER2013 Dataset
- command --> python TranEmotionDetector.py

It will take several hours depends on your processor. (On Ryzen5 processor with 16 GB RAM it took me around 5 hours)
after Training , you will find the trained model structure and weights are stored in your project directory.
emotion_model.json
emotion_model.h5

copy these two files create model folder in your project directory and paste it.

### run your emotion detection test file in terminal
.\env\Scripts\Activate       
python TestEmotionDetector.py
