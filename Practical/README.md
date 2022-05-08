# CS 181 Practical
### By Christy Jestin and David Qian

***This readme is meant to serve as a very brief overview. We explained the methods in more detail and had far more analysis on this [doc](https://github.com/christyjestin/cs181-s22-homeworks/blob/481450d3d61a5ac8af8c7893e54b4b92583c192e/PDFs/Practical.pdf). The doc also includes our theories about what the results mean.***

## The Task
This project was an assignment for the class CS 181: Machine Learning. Our task was to train multiple models to classify 2 second audio clips. The 10 classes were
- air conditioner
- car horn
- children playing
- dog bark
- drilling
- engine idling
- gun shot
- jackhammer
- siren
- street music

Each sound clip was represented in two ways: an amplitude recording and a Mel spectrogram. The first captured signal amplitude over time at a sampling rate of 22.05 kHz. The second broke up each clip into 87 time windows and captured 128 audio features per window.

We were told to take the following steps:
1. Train a pair of baseline models by running Principal Component Analysis (PCA) on the two representations and then training a Logistic Regression classifier
2. Train a pair of nonlinear models with intuitive choices for hyperparameters
3. Finetune the nonlinear models via hyperparameter search

## Our Approach
We decided to train a RandomForest classifier and a convolutional neural network for tasks 2 and 3. We used the Mel representation for both models since it has already captured a set of audio features, and these features don't have to be inferred by the model itself. There was no particular reasoning behind using a RandomForest, but we thought that CNNs would do well because they're position agnostic i.e. they would be better equipped to handle the fact that a gun shot is a gun shot whether it starts 0.1 seconds into the clip or 0.7 seconds into the clip.

For hyperparameter tuning, we played around with the kernel size for the CNN and the max depth for the Random Forest. A larger kernel size would allow the CNN to examine more of the data at the same time which might help it make better predictions. On the Random Forest, we noticed that the test accuracy was significantly worse than the training accuracy, so we thought that restricting max depth would force the model to overfit less and perhaps have more insight on new data.

## Our Results
Our Logistic Regression models achieved a test accuracy of 19.8% and 30.9% for the amplitude and Mel representations, respectively. Both the RandomForest and CNN did noticeably better than the Logistic Regression models and had around 47% accuracy. Unfortunately, the hyperparameter search did not yield higher performing models. The CNN was remarkably consistent across different kernel sizes with very similar plots for training and test accuracy over epochs. The same was true for training and test loss and class specific accuracy. Restricting the max depth of the Random Forest did result in lower training accuracy scores but did not improve test accuracy. Please find more of our thoughts on these results **[here](https://github.com/christyjestin/cs181-s22-homeworks/blob/481450d3d61a5ac8af8c7893e54b4b92583c192e/PDFs/Practical.pdf)**.