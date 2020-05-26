# Workflow Templates assignment

The goal of this assignment is to create a robust, re-usable workflow template. This template will build from previous
work you have done on the automation and docker-flask days.


## Part 1: Build API tests for previous work with iris dataset

1. Take a look at the contents of the `classification_template` folder, and familiarize yourself with the contents. This
 should all look familiar to you from the automation day! We will be building upon the work you completed that day. 
 In addition to the model and logger that you worked with last time, there is also an included 
 Flask application.

2. Train your model. You should be able to do this by running `train-model.py` with no modifications. You can change the
 type of classification model or other details in `model.py` if you prefer.
 
3. Run your flask application and familiarize yourself with how it works. Note that this flask app uses JSON instead of 
html forms to move data around, so you'll have to interact with the application in a python/ipython shell or jupyter 
notebook by posting json to the server. For example, to try a sample prediction:
    ```python
    import requests
    request_json = {'query': [[5.0, 3.8, 6, 7]],'type': 'numpy', 'mode': 'test'}
    r = requests.post('http://0.0.0.0:8080/predict', json=request_json)
    response = json.loads(r.text)
    print(response)
    ```
 
4. Create and fill out `unittests/ApiTests.py`. You'll want to test all aspects of your web application API, namely a 
robust testing of the `predict`, `train`, and `logs` endpoints. Try many different edge cases, poorly formatted inputs,
empty/missing data, etc. in your unittests to ensure that your web API is robust. If you find a flaw in the web app API, 
harden the API to correct for this.  

**Note**: You'll also need to adjust your `__init__.py` file to include your newly written ApiTests.

## Part 2: Extend your template for use on a regression problem

You now have a working template for a classification problem. Now you'll copy this template and re-use as much as
possible for a regression problem. 

We'll fit a regression model on the Boston housing dataset. This dataset was originally from kaggle, but is now a built-in dataset accessible from sklearn. You can download it using [sklearn.datasets.load_boston](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html).
This dataset is a regression dataset, which requires at minimum, a different type of model and evaluation than the Iris dataset,
where you built a classification model. 

1. Create a new folder called `regression_template`. You can choose to copy everything over from `classification_template`, 
but keep in mind some things will need to be changed. 

2. Alter the necessary functions in `model.py` incorporate using the new dataset, and training/grid-searching a 
regression model instead of a classification model. You can pick any regression model that you feel is appropriate, or 
even search over several models if you have time to do so.

3. Make necessary changes to `ModelTests.py` to accommodate the changes you made in `model.py`.

4. Make any necessary changes to your web application and API tests for the regression model.

## Extra Credit: Regression Logger
Update the `logger` module and `LoggerTests` for the regression template.
