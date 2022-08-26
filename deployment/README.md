# Deploying model on Ray Serve
We use Ray Serve as a base platform for our model Web API. 

## Local Deploy

1. make sure you install ray[serve] and fastapi in your virtual environment
2. start ray with following command

    `ray start --head --dashboard-host=0.0.0.0`

3. open a python shell and run the following commands

    `from deployment import model_serve`

    `model_serve.register_task()`

## Usage
    You can call the model as web API with any platform. 
    specified uri for model is: 
    "localhost:8265/get_sentiment"
    this api gets a form data with a text and image as input