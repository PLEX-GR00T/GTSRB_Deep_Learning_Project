# Main Task
In  this project we will compare and contrast our new model accuracy with professor's model. Here are the results below for training graphs and statistics.
1) Compare all the models with Graphs. 
![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/Deep_learning_model%20compare%20graph.png)

2) Show statistics of thoes model in graphs.
![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/Model%20compare%20statistics.png)
# Bonus Work 2:
My code can be found here on [Google Colab](https://colab.research.google.com/drive/1RPYmuJPH5piDmX6qamFHzJl_f7qMsf6S?usp=sharing) and on [Github](https://github.com/PLEX-GR00T/Data_Mining/blob/main/Bonus_work_2.ipynb) here.
1) Use [simple model3](https://github.com/lkk688/MultiModalClassifier/blob/main/TFClassifier/myTFmodels/CNNsimplemodels.py) from Professor's GitHub repo and learn to implement in your own dataset.
2) Accelerate the model with Intel using Tensorflow. 
3) Implement TensorFlowLite Inference

# Work Done and Image References : 

### 1) Visualize the Dataset.
- Dataset source : GTSRB - [German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- Single-image, multi-class classification problem
- More than 40 classes
- More than 50,000 images in total
- Large, lifelike database
- converted image size : (28,28)

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/Dataset_img.png)

### 2) Model Training and Accuracy curve.
- Model is taken from the professor's github as metioned before.
- I had to try few models that do not overfit, and give good accurayc like below.

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/Accuracy_Model_graph.png) 

### 3) Accelerate model Training with Intel.
- Intel's integration with the Tensorflow

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/Intel_acceleration.png)

### 4) Run the inference and note the timing.
- Average 49 miliseconds for every 50 images.
- Throughput: 155 images/s

### 5) Save model predictions into Pandas DataFrame.
### 6) Load Model and confert it into TFlite and save inference predictions into Panda DataFrame.
- model.tflite : 415360 KB = 415.36 MB
### 7) Convert Tflite model into Tflite_quant and save infrerence predictions into Panda DataFrame.
- model_quant_tl.tflite : 210736 KB = 210.73 MB

### 8) Saved models looked like this in Google Colab.
![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/Files_Directory.png)

### 9) Now highlight different prediciton values to see changes easily.
![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputPandaframe.png)

### 10) Conclusion.
![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/conclusion.png)
 
# TensorRT inference
