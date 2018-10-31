# Machine Learning Deployment using AWS SageMaker

Code and associated files 

This repository contains code and associated files for deploying ML models using AWS SageMaker. This repository consists of a number of tutorial notebooks for various coding exercises, mini-projects, and project files that will be used to supplement the lessons of the Nanodegree.

## Table Of Contents

### Tutorials
* [Boston Housing (Batch Transform) - High Level](https://github.com/udacity/sagemaker-deployment/tree/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Batch%20Transform)%20-%20High%20Level.ipynb) is the simplest notebook which introduces you to the SageMaker ecosystem and how everything works together. The data used is already clean and tabular so that no additional processing needs to be done. Uses the Batch Transform method to test the fit model.
* [Boston Housing (Batch Transform) - Low Level](https://github.com/udacity/sagemaker-deployment/tree/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Batch%20Transform)%20-%20Low%20Level.ipynb) performs the same analysis as the low level notebook, instead using the low level api. As a result it is a little more verbose, however, it has the advantage of being more flexible. It is a good idea to know each of the methods even if you only use one of them.
* [Boston Housing (Deploy) - High Level](https://github.com/udacity/sagemaker-deployment/blob/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Deploy)%20-%20High%20Level.ipynb) is a variation on the Batch Transform notebook of the same name. Instead of using Batch Transform to test the model, it deploys and then sends the test data to the deployed endpoint.
* [Boston Housing (Deploy) - Low Level](https://github.com/udacity/sagemaker-deployment/blob/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Deploy)%20-%20Low%20Level.ipynb) is again a variant of the Batch Transform notebook above. This time using the low level api and again deploys the model and sends the test data to it rather than using the batch transform method.
* [IMDB Sentiment Analysis - XGBoost - Web App](https://github.com/udacity/sagemaker-deployment/blob/master/Tutorials/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20-%20Web%20App.ipynb) creates a sentiment analysis model using XGBoost and deploys the model to an endpoint. Then describes how to set up AWS Lambda and API Gateway to create a simple web app that interacts with the deployed endpoint.
* [Boston Housing (Hyperparameter Tuning) - High Level](https://github.com/udacity/sagemaker-deployment/tree/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Hyperparameter%20Tuning)%20-%20High%20Level.ipynb) is an extension of the Boston Housing XGBoost model where instead of training a single model, the hyperparameter tuning functionality of SageMaker is used to train a number of different models, ultimately using the best performing model.
* [Boston Housing (Hyperparameter Tuning) - Low Level](https://github.com/udacity/sagemaker-deployment/tree/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Hyperparameter%20Tuning)%20-%20Low%20Level.ipynb) is a variation of the high level hyperparameter tuning notebook, this time using the low level api to create each of the objects involved in constructing a hyperparameter tuning job.
* [Boston Housing - Updating an Endpoint](https://github.com/udacity/sagemaker-deployment/tree/master/Tutorials/Boston%20Housing%20-%20Updating%20an%20Endpoint.ipynb) is another extension of the Boston Housing XGBoost model where in addition we construct a Linear model and switch a deployed endpoint between the two constructed models. In addition, we look at creating an endpoint which simulates performing an A/B test by sending some portion of the incoming inference requests to the XGBoost model and the rest to the Linear model.

### Mini-Projects
* [IMDB Sentiment Analysis - XGBoost (Batch Transform)](https://github.com/udacity/sagemaker-deployment/tree/master/Mini-Projects/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20(Batch%20Transform).ipynb) is a notebook that is to be completed which leads you through the steps of constructing a model using XGBoost to perform sentiment analysis on the IMDB dataset.
* [IMDB Sentiment Analysis - XGBoost (Hyperparameter Tuning)](https://github.com/udacity/sagemaker-deployment/tree/master/Mini-Projects/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20(Hyperparameter%20Tuning).ipynb) is a notebook that is to be completed and which leads you through the steps of constructing a sentiment analysis model using XGBoost and using SageMaker's hyperparameter tuning functionality to test a number of different hyperparameters.
* [IMDB Sentiment Analysis - XGBoost (Updating a Model)](https://github.com/udacity/sagemaker-deployment/tree/master/Mini-Projects/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20(Updating%20a%20Model).ipynb) is a notebook that is to be completed and which leads you through the steps of constructing a sentiment analysis model using XGBoost and then exploring what happens if something changes in the underlying distribution. After exploring a change in data over time you will construct an updated model and then update a deployed endpoint so that it makes use of the new model.

### Project

[Sentiment Analysis Web App](https://github.com/udacity/sagemaker-deployment/tree/master/Project) is a notebook and collection of Python files to be completed. The result is a deployed RNN performing sentiment analysis on movie reviews complete with publicly accessible API and a simple web page which interacts with the deployed endpoint. This project assumes that you have some familiarity with SageMaker. Completing the XGBoost Sentiment Analysis notebook should suffice.

## Setup Instructions

The notebooks provided in this repository are intended to be executed using Amazon's SageMaker platform. The following is a brief set of instructions on setting up a managed notebook instance using SageMaker, from which the notebooks can be completed and run.

### Log in to the AWS console and create a notebook instance

Log in to the AWS console and go to the SageMaker dashboard. Click on 'Create notebook instance'. The notebook name can be anything and using ml.t2.medium is a good idea as it is covered under the free tier. For the role, creating a new role works fine. Using the default options is also okay. Important to note that you need the notebook instance to have access to S3 resources, which it does by default. In particular, any S3 bucket or objectt with sagemaker in the name is available to the notebook.

### Use git to clone the repository into the notebook instance

Once the instance has been started and is accessible, click on 'open' to get the Jupyter notebook main page. We will begin by cloning the SageMaker Deployment github repository into the notebook instance. Note that we want to make sure to clone this into the appropriate directory so that the data will be preserved between sessions.

Click on the 'new' dropdown menu and select 'terminal'. By default, the working directory of the terminal instance is the home directory, however, the Jupyter notebook hub's root directory is under 'SageMaker'. Enter the appropriate directory and clone the repository as follows.

```bash
cd SageMaker
git clone https://github.com/udacity/sagemaker-deployment.git
exit
```

After you have finished, close the terminal window.

### Open and run the notebook of your choice

Now that the repository has been cloned into the notebook instance you may navigate to any of the notebooks that you wish to complete or execute and work with them. Any additional instructions are contained in their respective notebooks.
