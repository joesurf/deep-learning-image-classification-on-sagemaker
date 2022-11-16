**NOTE:** This file is a template that you can use to create the README for your project. The **TODO** comments below will highlight the information you should be sure to include.

# Using Transfer Learning to Predict Dog Breed on Sagemaker


Create image classification model by applying transfer learning to existing resnet model for prediction of dog breeds.

## Project Set Up and Installation
Run notebook on AWS Sagemaker

## Dataset

### Overview
The dataset contains images from 133 categories of dog breeds, separated into train, valid and test folders. 

### Access
Data is stored in S3 and retrieved using the python sdk.

## Hyperparameter Tuning
Model
- Transfer learning on a pre-trained ResNet50 model from PyTorch + two fully connected neural network layers
- Use hyperparameter tuning to identify the best parameters for this model -> retrain the model using the best hyperparameters
- Add in profiling and debugging configuration
- Deploy the model and make test inferences

Hyperparameters
  - Learning rate - default(x) is 0.001 , so we have selected 0.01x to 100x range for the learing rate
  - eps - defaut is 1e-08 , which is acceptable in most cases so we have selected a range of 1e-09 to 1e-08
  - Weight decay - default(x) is 0.01 , so we have selected 0.1x to 10x range for the weight decay
  - Batch size - selected only two values: 64, 128
  
 Best Hyperparamters post Hyperparameter fine tuning are : 
 {'batch_size': 64, 'eps': '2.558405286621858e-09', 'lr': '0.00198254275850862', 'weight_decay': '0.06239842053247745'}

### Training Jobs Snapshots
![Training Jobs]()


## Debugging and Profiling
The debugger hook was set to track the loss metrics of the training and validation/testing jobs. The plot of cross entropy loss is shown below:

No smooth output lines -> Try different number of extra neural network layers on the resnet model and see if there are any improvements


### Processing Jobs Snapshots
![Processing Jobs]()


### Debugger Plot Snapshots
![Debugger Plot]()


## Model Deployment
Model was deployed to a "ml." instance type using "endpoint_inference.py" script and test using 3 images.

However, the model predicted only 1 image correctly.

### Deployment Snapshots
![Deployment]()