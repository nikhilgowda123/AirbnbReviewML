# AirbnbReviewML
This project is aimed at predicting the individual ratings (accuracy, cleanliness, check-in, communication, location, and value) and overall review rating of Airbnb listings. The ratings tend to be high values (greater than 4), so special care is taken while evaluating the prediction accuracy. Also, the reviews are not always in English and may contain non-text content such as emojis. Therefore, we have decided to preprocess the data accordingly.

## Performance Metrics ##
To evaluate the performance of our models, we have considered a dummy baseline regressor, which always predicts the mean of the training set. We have used Mean Square Error and Mean Absolute Error as performance metrics to measure and compare the performance of our models. The dataset has been split into 75% train and 25% test sets.

Model	              | Train Mean Square Error |	Test Mean Square Error | Train Mean Absolute Error | Test Mean Absolute Error|
---------------------------------------------------------------------------------------------------------------------------------
Lasso Regression    |	0.092                   |  0.088	               | 0.177                     | 0.18                    |
k-nearest neighbors |	0.096	                  |  0.090                 | 0.185                     | 0.18                    |
Dummy Regressor	    | -	                      |  0.109                 | 	-                        | 0.203                   |
