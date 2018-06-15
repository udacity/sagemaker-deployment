# Sentiment Analysis using AWS Sagemaker

Below is a walkthrough of the various steps involved with following the tutorial. Each notebook exposes more Amazon services and at the end you will have a custom model, trained and deployed, which is accessible via a public website.

****Note**** Something important to note is that SageMaker ascribes a lot to the names of various things. It can be very easy to accidentally name something in a way that conflicts with old data, in which case you get some very difficult to diagnose errors.

## Log in to the AWS console and create a notebook instance

Log in to the AWS console, go to the SageMaker dashboard. Click on Create notebook instance. The notebook name can be anything and using ml.t2.medium is a good idea as it is covered under the free tier. For the role, creating a new role works fine. Using the default options is also okay. Important to note that you need the notebook instance to have access to S3 resources, which it does by default. In particular, any S3 bucket or object with sagemaker in the name is available to the notebook.

-   Take note of the name of the IAM role that was created for you. You will need to add a policy to this in the second notebook.

Leave the rest as is and create the notebook instance.

## Use git to clone the repo into the notebook instance

Once the instance has been started and is accessible, click on 'open' to get to the Jupyter notebook main page. We will begin by cloning the Sentiment Analysis github repository into the notebook instance. Note that we want to make sure to clone this into the appropriate directory so that the data will be preserved between sessions.

Click on the 'new' dropdown menu and select 'terminal'. The terminal is started in the home directory, however, the Jupyter instance's root directory is under 'SageMaker'. Enter the appropriate directory and clone the repo.

```bash
cd SageMaker
git clone https://github.com/udacity/sagemaker-deployment.git
exit
```

After this, close the terminal window.

## Open and run the XGBoost notebook

Now that the repo has been cloned into the notebook instance, navigate to the Sentiment Analysis directory and open up the [IMDB Sentiment Analysis Using XGBoost in SageMaker](https://github.com/udacity/sagemaker-deployment/blob/master/Sentiment%20Analysis/IMDB%20Sentiment%20Analysis%20Using%20XGBoost%20in%20SageMaker.ipynb) notebook. Step through the notebook.

Note, the pre-processing step takes over an hour (maybe 1:15).

## Open and run the PyTorch notebook

Now that we've gone through the SageMaker built-in example we can start constructing our own models. In order to do this we must first make sure that our SageMaker role has access to EC2. Go to the IAM roles dashboard and select the role that was created earlier for this notebook instance. Click on 'Attach policy' and find the 'AmazonEC2ContainerRegistryFullAccess' policy. Add this policy to the SageMaker role.

Next, open up the [IMDB Sentiment Analysis Using RNN (PyTorch) in Sagemaker](https://github.com/udacity/sagemaker-deployment/blob/master/Sentiment%20Analysis/IMDB%20Sentiment%20Analysis%20Using%20RNN%20(PyTorch)%20in%20SageMaker.ipynb) notebook. Step through the notebook. Note that creating and pushing the gpu container takes about a half hour.

## Open and run the custom model notebook

Lastly we get to set up a public website that will actually use the model we constructed. Open the [IMDB Sentiment Analysis Custom Model API in SageMaker](https://github.com/udacity/sagemaker-deployment/blob/master/Sentiment%20Analysis/IMDB%20Sentiment%20Analysis%20Custom%20Model%20API%20in%20SageMaker.ipynb) notebook and follow each of the instructions. It will be a bit of an excursion through AWS but at the end we'll have something we can show off.
