#**Traffic Sign Recognition** 


[//]: # (Image References)

[visualization_image]: ./writeup_images/visualization.png "Visualization"
[histogram_training]: ./writeup_images/histogram_training.png "Histogram of training data"
[histogram_validation]: ./writeup_images/histogram_validation.png "Histogram of validation data"
[histogram_test]: ./writeup_images/histogram_test.png "Histogram of test data"
[grayscaling]: ./writeup_images/grayscaling.png "Grayscaling"
[normalization]: ./writeup_images/grayscale_normalization.png "Normalization"
[webImage1]: ./webImage/c12_Priorityroad.jpg "Traffic Sign 1"
[webImage2]: ./webImage/c13_yield.jpg "Traffic Sign 2"
[webImage3]: ./webImage/c17_noentry.jpeg "Traffic Sign 3"
[webImage4]: ./webImage/c18_generalCaution.jpeg "Traffic Sign 4"
[webImage5]: ./webImage/c23_slippery_road.jpeg "Traffic Sign 5"
[webImage6]: ./webImage/c40_roundabout.jpg "Traffic Sign 6"
[webchart]: ./writeup_images/webimage_chart.png "Softmax predictions"


###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. 

![alt text][visualization_image]

The bar chart below shows how the data is distributed at each class. The distributions of training data, verification data and test data are similar.

![alt text][histogram_training]
![alt text][histogram_validation]
![alt text][histogram_test]

###Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because it helps to reduce training time. I referenced the paper, Pierre Sermanet and Yann LeCun ",Traffic Sign Recognition with Multi-Scale Convolutional Networks" and, in the paper, the models with gray scale image achived over 98% accuracy. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][grayscaling]

As a last step, I normalized the image data because some of data is too bright or dark compares to the others in the same class.

Here is an example of an original image and an augmented image:

![alt text][normalization]


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x30 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x30 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| input 400, output 120   						|
| RELU					|												|
| Dropout				| Keep probability: 0.5							|
| Fully connected		| input 120, output 84   						|
| RELU					|												|
| Dropout				| Keep probability: 0.5							|
| Fully connected		| input 84, output 43   						|
| Softmax				| prediction 									|
|						|												|


To train the model, I used an LeNet-5 which is included in the class meterial as my base model. However, the validation accuracy is barely over 0.93. To improve the model, I added dropout layer after last two fully connected layer, but the modified model acheived still under 0.94 validation accuracy. Therefore, I increased the first convolution kernel size 6 to 30, and finally, I got over 0.95 validation accuracy. To optimize the cost function, Adam Optimizer is used. 128 batch size, 10 epochs and 0.001 learning rate are suitable for this model.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.959
* test set accuracy of 0.945


###Test a Model on New Images

Here are six German traffic signs that I found on the web:

![alt text][webImage1] ![alt text][webImage2] ![alt text][webImage3] 
![alt text][webImage4] ![alt text][webImage5] ![alt text][webImage6]

Because the sizes of the images obtained from the web are different, I resize them to 32x32.

The second image might be difficult to classify because the sign is located on the right side of the image. Also, The sixth image might be hard because the class "40" in the training data has a relatively small number of data.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road  		| Priority road   								| 
| Yeild  				| Yeild 										|
| No entry				| No entry										|
| General caution 		| General caution				 				|
| Slippery road 		| Slippery road					 				|
| Roundabout mandatory	| Yeild 										|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 0.833. The accuracy is quite lower than the result of test set but considering that this result is from only 6 images, I think it is acceptable result.   

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

The image below is bar chart of softmax probabilities for each image.
For the second image, which I think is likely to be misclassified, the maximum probabilty is quite high, and it predicts the right label.
For the sixth image, which is misclassified, the prediction is failed and the second highest probability is also wrong label.
Considering that the two groups with the highest probability are "12", "13" classes and the misclassified  classes have the third and the fourth largest number of test data, data imbalance in test data might affect the result.
![alt text][webchart]



