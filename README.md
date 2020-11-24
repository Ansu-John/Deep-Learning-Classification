# Deep Learning Classification

# OBJECTIVE
The idea of the project is to build an image classification model that will be able to identify what class the input image belongs to.

# DATASET USED
The dataset used in this analysis is from Kaggle and can be found [here](https://www.kaggle.com/paultimothymooney/breast-histopathology-images/)

# TOOLS

Python, Keras  

# TECHNIQUES

Execute build_dataset.py to get the dataset sorted and then execute train_model.py for classification

**Why deep learning:** When the amounth of data is increased, machine learning techniques are insufficient in terms of performance and deep learning gives better performance like accuracy.

![DeepvsML](https://github.com/Ansu-John/Deep-Learning-Classification/blob/main/resources/DeepvsML.png)

**What is amounth of big:** It is hard to answer but intuitively 1 million sample is enough to say "big amounth of data"

**Usage fields of deep learning:** Speech recognition, image classification, natural language procession (nlp) or recommendation systems

**What is difference of deep learning from machine learning:**
+ Machine learning covers deep learning.
+ Features are given machine learning manually.
+ On the other hand, deep learning learns features directly from data.

![DeepvsML](https://github.com/Ansu-John/Deep-Learning-Classification/blob/main/resources/DeepvsML1.png)

## build_dataset.py

This will split the dataset into training, validation, and testing sets in the ratio - 80% for training (of that, 10% for validation) and 20% for testing. 

With the ImageDataGenerator from Keras, we will extract batches of images to avoid making space for the entire dataset in memory at once.

## train_model.py

This trains and evaluates the model. Here, we’ll import from keras, sklearn, cancernet, config, imutils, matplotlib, numpy, and os.

In this script, we will be performing the below steps: 

1. Set initial values for the number of epochs, the learning rate, and the batch size. 

2. Get the number of paths in the three directories for training, validation, and testing. 

3. Get the class weight for the training data so we can deal with the imbalance.

4. Initialize the training data augmentation object. This is a process of regularization that helps generalize the model. This is where we slightly modify the training examples to avoid the need for more training data. 

5. Initialize the validation and testing data augmentation objects.

6. Initialize the training, validation, and testing generators so they can generate batches of images of size batch_size. 

7. Initialize the model using the Adagrad optimizer and compile it with a binary_crossentropy loss function. 

8. Once these steps are done fit the model using fit_generator() method.

9. Once we train our model, evaluate the model on our testing data. 

10. We’ll reset the generator and make predictions on the data using predict_generator() method. 

11. Then, for images from the testing set, we get the indices of the labels with the corresponding largest predicted probability. 

12. Compute and display classification report and confusion matrix and get the accuracy, specificity, and sensitivity, and display all values. 

13. Finally, plot the training loss and accuracy.

