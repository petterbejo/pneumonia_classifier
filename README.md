# pneumonia_classifier

A binary image classifier to predict whether the X-ray shows signs of pneumonia. 

Please note that this classifier has been built as side/learning project. It is not in any way intended for medical use. 

## Development plan

I intend to develop the following things in the near future, more or less in the indicated order:

1. Write a method for the classifier that can save and load the best performing model and its hyperparameters, and overwrite the existing best performing model if the performance metric - accuracy - is better.
2. Build models striving for optimal performance, possibly using transfer learning.
3. Build a web app, probably built with flask and Docker or similar tools, where a user can upload an X-ray and get a prediction.
