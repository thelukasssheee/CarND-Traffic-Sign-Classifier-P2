# Project: Build a Traffic Sign Recognition Program 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
## Writeup from Michael Berner
### Student of Udacity Self-Driving Car NanoDegree 2018

[//]: # (Image References)

[image1a]: ./images_out/dataset_distribution.jpg "Visualization of label distribution"
[image1b]: ./images_out/dataset_example_images_1-train.jpg "Visualization of exemplary images TRAINING DS"
[image1c]: ./images_out/dataset_example_images_2-valid.jpg "Visualization of exemplary images TRAINING DS"
[image1d]: ./images_out/dataset_example_images_3-test.jpg "Visualization of exemplary images TRAINING DS"
[image2]: ./images_out/dataset_grayscale_conversion.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./additional_signs_original/stop_1.jpg "Stop sign 1, label 14"
[image5]: ./additional_signs_original/stop_2.jpg "Stop sign 2, label 14"
[image6]: ./additional_signs_original/stop_3.jpg "Stop sign 3, label 14"
[image7]: ./additional_signs_original/v30_1.jpg "Speed limit 30, label 1"
[image8]: ./additional_signs_original/v60_1.jpg "Speed limit 60, label 3"
[image9]: ./additional_signs_original/v80_1.jpg "Speed limit 80, label 5"
[image10]: ./additional_signs_original/vorfahrt_1.jpg "Yield, label 13"
[image11]: ./images_out/own_images_from_internet.jpg "Own images cropped and resized"
[image12]: ./images_out/own_images_from_internet_detected_too_large.jpg "Own images, detection quality, first attemp"
[image13]: ./images_out/own_images_from_internet_detected.jpg "Own images, detection quality, final attempt"

---

## Build a Traffic Sign Recognition Program

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

--- 

### Rubric Points 

This readme will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view), which were relevant for successfully passing this project. They are addressed individually in the lines below. 

---
##  I - Writeup / README

This is the README to my project. I decided to put my write up notes directly into this file, which makes accessing, understanding and using my program much easier. 

For the sake of completeness, I retained a copy of the original README file within this repository under the name [README_Udacity](https://github.com/thelukasssheee/CarND-Traffic-Sign-Classifier-Project/blob/master/README_Udacity.md).  

Here is the link to my code as a [Jupyter Notebook](https://github.com/thelukasssheee/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb). More files and the entire repository can be found on [GitHub](https://github.com/thelukasssheee/CarND-Traffic-Sign-Classifier-Project/)

The entire program code is held within a Jupyter notebook and is written in Python 3.5.2.final.0.

Some libraries are necessary to run the code and should be installed before running the notebook. The versions which I used during development are added below. This doesn't mean, that you need to downgrade or specifically install this version. If something isn't running though, they might be helpful:

* pickleshare 0.7.3
* numpy 1.12.1
* matplotlib 2.0.0
* tensorflow 0.12.1
* opencv-python 3.4.0.12

---

## II - Data Set Summary & Exploration

### 1. Basic summary of the data set

I used general python commands like `len` and the numpy function `shape` to calculate summary statistics of the traffic
signs data set and use it dynamically within the code.

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

### 2. Exploratory visualization of the dataset: histograms

Here is an exploratory visualization of the data set. 

I was particularly interested in two things:

* Representation of traffic signs in the dataset (e.g. is one traffic sign represented more often than others)
* Are all three provided datasets (test, train, validation) 
  * complete (are all labels included)
  * comparable (in terms of amount of pictures provided) 
  
As can be seen in the two bar charts, all traffic signs are included in all three datasets. This is good.

However, some traffic signs are represented significantly more often than others. This could lead to problems in the learning algorithm in such a way that it is overtrained for some traffic signs, but not trained enough for other. This might be one potential target for further optimizations. 

![Distribution of labels within provided image datasets][image1a]

### 3. Exploratory visualization of the dataset: example images

Some example images from all three datasets are shown below. It can be clearly seen, that the images contained in all datasets are comparable in terms of size (width x height) and color information.

![Example images from TRAINING dataset][image1b]
![Example images from VALIDATION dataset][image1c]
![Example images from TEST dataset][image1d]

---

## III - Design and Test a Model Architecture

### 1. Image preprocessing

*Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)*

In the final version, I have applied two major steps. 

At first, the image is **converted from RGB information to grayscale**, as can be seen below.

![Conversion to grayscale][image2]

As it turned out, the greatest benefit of using grayscale images was calculation performance. There was also a positive effect on model's accuracy, but it was very small. The speed increase was tremendous, allowing much more iterations for parameter optimization! 


The second step was **normalization**: the rgb/grayscale information was changed from `[0 ... 255]` to a range of `[-1.0 ... 1.0]`. This step had a huge impact on the neural networks accuracy, immediately improving the accuracy by 7%.  


### 2. Model architecture
*Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.*

My final model consisted of the following layers:

| Layer         			|     Description	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Input         			| 32x32x1 Grayscale image 						| 
| Convolution 5x5, layer 1  | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU						|												|
| Max pooling 2x2      		| 2x2 stride, outputs 14x14x6 					|
| Convolution 5x5, layer 2  | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU						|												|
| Max pooling 2x2			| 2x2 stride, outputs 5x5x16					|
| Flatten					| Output 400									|
| Fully connected, layer 3	| 400x512										|
| RELU & Dropout			| Keep probability 0.5							|
| Fully connected, layer 4	| 512x512										|
| RELU & Dropout			| Keep probability 0.5							|
| Fully connected, layer 5	| 512x43, 43 logits								|


### 3. Model training
*Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.*

To train the model, I started off playing around with hyper parameters. I found that the initial parameters from the LeNet neural network of LeCunn et. al. were performing well:

| Parameters       			|     Value			        					| 
|:-------------------------:|:---------------------------------------------:| 
| Start values     			| Truncated normal, mu = 0, sigma = 0.1, b = 0.0| 
| Learning rate		 		| 0.001 										|
| Epochs					| 30											|
| Batch size				| 128

During the optimization, I also stayed with the AdamOptimizer since it was doing great with the dropouts I implemented. 

### 4. Necessary steps to achieve required accuracy of >0.93
*Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.* 

My final model results were:

* training set accuracy of 99.9%
* validation set accuracy of 96.5% 
* **test set accuracy of 94.6%**

I started within this project with the LeNet architecture, which was available from the previous labs and lessons. There was a paper indicating, that LeNet architecture works fine on traffic sign recognition, therefore making it a good starting point. As it turned out, I was able to confirm this: the model was computing fast and achieved an accuracy > 0.8. However, not yet sufficient to achieve the required accuracy of 0.93.

In the next step, I worked on the hyper parameters in order to see, how much the model can be tweaked. Unfortunately, the additional training effort was mostly going in to over fitting the model, not significantly improving the test set accuracy.

Therefore, I decided to include dropouts into the neural network. This was, at first, reducing accuracy due to the small layer sizes of the LeNet architecture. I figured, that I have to make the network deeper by increasing the size of the fully connected layers and added an additional fully connected layer to the network. This helped a lot to boost accuracy.  

Below you can see, how the major code changes affected to neural networks accuracy:

| Accuracy         		| Prediction	        								| 
|:---------------------:|:-----------------------------------------------------:| 
| 87.3%       			| v0: LeNet, direct adaption to RGB						| 
| 94.4%    				| v1: as previous, but 512x512 layer 4 & dropout 0.5	|
| 94.6%					| v2: as previous, but changed to grayscale				|

The final accuracy was sufficient and I was surprised by the result. Unfortunately, I was running out of time to further tweak the code.

---

## VI - Test a Model on New Images

### 1. Identify further German traffic signs found on the web
*Choose further German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.*

I looked up several signs on the internet and came up with the following selection. The idea behind it was to start with very simple, basic signs. Other images have a different background, have stickers on them, are tilted or are somewhat damaged. 

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10]
 
The images needed to be cropped and downsized for the neural network to be able to process them. After these steps, the images looked like this:

![alt text][image11]

### 2. Overall prediction quality of further traffic signs
*Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).*

The overall results were surprisingly bad! The model was able to correctly guess only 3 of the 7 traffic signs, which gives an accuracy of only 43%.

What was possibly going on is explained in more detail in the next section.

### 3. Prediction for individual traffic signs (optional)
*Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)*

#### First attempts

| Image			        |     Comment	    	    	 |	Prediction		|  Discussion																|
|:---------------------:|:------------------------------:|:----------------:|:-------------------------------------------------------------------------:| 
| Stop Sign 1     		| Very clear picture (large)	 |	0.01			| Image probably too large, relevant areas on borders were not identified	 |
| Stop Sign 2  			| Distorted view				 |	0.57			| Image heavily skewed, but successfully detected							|
| Stop Sign 3			| Stickers on sign				 |	0.00			| Not detected at all!! Was NN fooled by stickers on sign?					|
| 30 km/h	      		| Speed sign (forest)			 |	1.00			| Clearly detected.															|
| 60 km/h				| Speed sign (urban)   			 |	0.00			| Not detected at all, although clearly visible! Too large?				  |
| 80 km/h				| Speed sign (forest) 			 |	0.01			| Not detected at all. Too large?											|
| Yield					| Yield sign (damaged) 			 |	1.00			| Clearly detected.															|

![alt text][image12]

My guess is, that the relevant features of the image files were too close to the borders - the traffic signs were too large. In the training dataset, traffic signs account for 1/2 to 2/3 of the image size (32x32px). In the case of my pictures, they were filling 3/4 or even the full dimension of the 32px. What is pointing me towards this direction is that the 30km/h speed sign was properly detected, whereas the other speed signs were not. Even the tilted / skewed stop sign was detected, but the very clearly visible first stop sign was not.

#### Final attempt

The images were cropped freshly to have a larger border. 

![alt text][image13]



## (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Unfortunately, I was running out time and could not finish the second optional task. 

However, the project was again a lot of fun! Thanks Udacity for this great experience!
