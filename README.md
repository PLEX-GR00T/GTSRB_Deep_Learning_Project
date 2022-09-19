# Traffic sign Dataset Projects:
List of major Tasks.
1. Compare your own different models with Professor's.
2. Convert model into **TensorFlow Lite** and accelerate the training time with **Intel's acceleration**.
3. **Compare state-of-the-art models** and perfrom **TensorRT inference acceleartion** on best model.

Visualization of the Dataset used in project.
- Dataset source : GTSRB - [German Traffic Sign Recognition Benchmark](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)
- Single-image, multi-class classification problem
- More than 40 classes
- More than 50,000 images in total
- Large, lifelike database
- converted image size : (28,28)

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/Dataset_img.png)

## 1) Compare your own different models with Professor's.
- In  this project we will compare and contrast our new models' accuracy with professor's model. 
- Here are the results below for training graphs and statistics.
- You can find this code [here](https://github.com/PLEX-GR00T/Data_Mining/blob/main/Deep_learning_signboards_models_compare.ipynb).
1) Compare all the models with Graphs. 

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/Deep_learning_model%20compare%20graph.png)

2) Statistics of models above.

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/Model%20compare%20statistics.png)

## 2) Convert model into TensorFlow Lite and accelerate the training time with Intel's acceleration.
My code can be found [here](https://github.com/PLEX-GR00T/Data_Mining/blob/main/Bonus_work_2.ipynb).
1) Use [simple model3](https://github.com/lkk688/MultiModalClassifier/blob/main/TFClassifier/myTFmodels/CNNsimplemodels.py) from Professor's GitHub repo and learn to implement it in our own dataset.

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/Accuracy_Model_graph.png)

2) Accelerate the model with Intel using Tensorflow. 

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/Intel_acceleration.png)

3) Implement TensorFlowLite Inference.
Inference on regular model (Without TensorFlowLite)
- Average 49 miliseconds for every 50 images.
- Throughput: 155 images/s
- Size of original model = 600 MB

4) Convert model into TFlite and compare:
- model.tflite : 415360 KB = 415.36 MB
- model_quant_tl.tflite : 210736 KB = 210.73 MB

- Saved models looks like this in Google Colab.

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/Files_Directory.png)

- Now highlight different prediciton values to see changes easily.

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/outputPandaframe.png)

- Conclusion.

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/conclusion.png)
 
## 3) Compare state-of-the-art models and perfrom TensorRT inference acceleartion on best model.
- In this project I compared 12 differnt TensorFlow State-of-the-art models with traffic sign dataset, and you can find my code [here](https://github.com/PLEX-GR00T/Data_Mining/blob/main/TensorRT_Inference_Comparison.ipynb).
- After that I comared 2 differnt models's time and throughtput accuracy.

1) Model comparison results.

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/12%20State%20of%20the%20art%20models.png)

2) Resnet-V2-50/feature-vector
- Time

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/renetv2%20time.png)

- Throughtput

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/resnetv2%20throughput.png)

3) Mobilenet-V1-050-160/feature-vector
- Time

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/mobilenetv1%20time.png)

- Throughtput

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/mobilenetv1%20throughput.png)

4) TensorRT comparison Table

![image](https://github.com/PLEX-GR00T/Data_Mining/blob/main/outputs/TensorRT%20comparison.png)
