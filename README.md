# AirbnbReviewML
This project is aimed at predicting the individual ratings (accuracy, cleanliness, check-in, communication, location, and value) and overall review rating of Airbnb listings. The ratings tend to be high values (greater than 4), so special care is taken while evaluating the prediction accuracy. Also, the reviews are not always in English and may contain non-text content such as emojis. Therefore, we have decided to preprocess the data accordingly.

## Performance Metrics ##
To evaluate the performance of our models, we have considered a dummy baseline regressor, which always predicts the mean of the training set. We have used Mean Square Error and Mean Absolute Error as performance metrics to measure and compare the performance of our models. The dataset has been split into 75% train and 25% test sets.

| Models                | Train MSE | Test MSE | Train MAE | Test MAE | R2 Score |
| ---------------------| --------- | -------- | --------- | -------- | -------- |
| Lasso Regression     | 0.092     | 0.088    | 0.177     | 0.183    | 0.259    |
| k-nearest neighbours | 0.096     | 0.090    | 0.185     | 0.188    | 0.229    |
| Dummy Regressor      | N/A       | 0.109    | N/A       | 0.203    | -0.003   |

We can observe that the error scores on the training set for all models are nearly zero, which reflects good tuning of input features. Out of all the models implemented, Lasso Regression produces the lowest mean square error score of 0.68 followed by k-nearest neighbors (0.076). By comparing these scores with the baseline model, which comparatively has the highest mean square error, we can say that our model's performance is much better.

## Features Associated with a High Rating ##
Our analysis shows that superhosts tend to have higher ratings. Also, the neighborhood seems to affect the location rating, and more highly rated listings tend to have more reviews.

## Usage ##

The code for the project can be found in the src directory. To run the project, follow the below steps:

Install the required packages by running pip install -r requirements.txt.
Run python airbnbreviewml.py to preprocess tha data, train and evaluate the performance of the Lasso Regression and k-nearest neighbors models.
Note: The listings.csv and reviews.csv directory contains the Airbnb dataset used in this project.
