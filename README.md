# Reinforcement Learning Integrated with Supervised Learning for Training of Near Infrared Spectrum Data for Non-destructive Testing of Fruits
Yuqi Li1, Kulbir Singh Ahluwalia2 and Simarjeet S. Saini1,* <br>
Department of Electrical and Computer Engineering, University of Waterloo, 200, University Avenue West, Waterloo, ON, N2L 3G1, Canada <br>
Currently at University of Maryland, College Park, MD 20742, USA <br>
sssaini@uwaterloo.ca <br>

## Abstract
Near infrared spectroscopy (NIR) has been demonstrated for testing internal quality parameters of fruits by correlating features in the spectrum to biophysical and biochemical properties. NIR requires extensive training of the models and requires various steps like preprocessing to reduce noise, collinearity and emphasize variables with more relevant features, discriminant methods to train models, and calibration methods to predict the correlations. Various pretreatment methods used in fruit quality assessment include smoothing, offset correction, de-trending, multiplicative scatter correction (MSC), standard normal variate (SNV), derivative correction, wavelet transformation, orthogonal signal correction etc. Similarly, there are many discriminant methods like principal component analysis, partial least square discriminant analysis (PLS), linear discriminant analysis, support vector machine, and a plethora of calibration methods including multiple linear regression, principle component regression, partial least square regression, least square support vector machines and artificial neural networks. With all these options, training of the models becomes costly and time consuming. In this paper we present a novel machine learning algorithm which cycles through various options to decide on the best combination for prediction. The algorithm combines two categories of machine learning: supervised learning and reinforcement learning to reduce the training cost. To achieve the combination, a new concept of model visualization was introduced by use of climbers. Training process is split into n periods and in each period climbers are allowed to move more than one step. After the end of the period, only the top climbers are moved into the next period resulting in reinforcement learning. The algorithm was tested by training NIR spectral data for oranges to predict solid soluble content (SSC) and for kiwis to predict SSC and dry matter (DM). NIR spectra from 100 oranges and 378 kiwis was measured. We show that the results achieved were better than using previously used methods like MSC + SNV + PLS. For example, for SSC measurements of kiwis, correlation parameter, r of 0.87 and root mean square error for prediction (RMSEP) of 0.75 °Brix was achieved by MSC+SNV+PLS whereas the machine learning training algorithm achieved results for r of 0.97 and RMSEP 0.62 °Brix. Our machine learning algorithm to train models for NIR spectroscopy can help in quick adaptation of the technique for non-destructive testing of fruits. 

## About model optimization

### Introduction
Machine learning models are constructed by complex mathematical models. People always have a hard time tuning hyperparameters in those models. In this research, we are trying to achieve teaching a machine learning model we understand to tune another black-box machine learning model. In our example, we apply deep Q reinforcement learning to tune the number of components, as a hyperparameter, in partial least square (PLS) regression model. 

**Background: it is a common problem that a well-tuned ML model under one specific dataset usually requires re-tuning when the dataset is changed even data features keep idle. The re-tuning process is always a big cost for business, especially in the financial industry.**

**Our theory: After model A is able to optimize model B to a relatively high accuracy under one specific dataset, the trained model A could find an accuracy of local maximum for model B once a new dataset comes in (as long as the dataset features keep same).**

Before we explain the functionality, let's check out the result showing in the following two GIFs.

The graphs below show the PLS model accuracy VS the value of components (the hyperparameter). The mission is to build a deep Q learning model that finds the value of components to maximize the PLS model's accuracy in the shortest time.

Before deep Q learning model was trained<br>
<a href="https://imgflip.com/gif/3lr8ou"><img src="https://i.imgflip.com/3lr8ou.gif" title="made at imgflip.com"/></a> <br>
After training, we could see that the model is tuned to a higher accuracy with shorter time<br>
<a href="https://imgflip.com/gif/3lr975"><img src="https://i.imgflip.com/3lr975.gif" title="made at imgflip.com"/></a> <br>

**Why don't you use grid search since there is only one hyperparameter to tune?** <br>
- This is for the convenience of visualization. If there are two hyperparameters, 3D visualization costs a lot of computing power. Over two hyperparameters, we might not be able to show the optimization process so intuitively

### Functionality
#### Why deep Q learning? <br>
- We paraphrased the mission as building a climber machine whose goal is to find the highest position (accuracy) of the geometry <br>
#### Policy for the climber <br>
- Reward: Accuracy from PLS model
- States: The accuracies surrounding the agent and hyperparameter's value
- Actions: Increase or decrease hyperparameter's value by specific value options
- Gamma: 0.9
- Game Over: Once the agent steps to the edges at the two ends or stay idles after several steps. 

#### Accuracy measurement <br>
- We split the data into training and testing data. By inputting the hyperparameter values returned from the deep Q learning model to the PLS model, we calculate R square value as the accuracy of the PLS model.

