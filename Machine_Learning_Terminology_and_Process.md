
# Machine Learning Terminology and Process

1.  Business Problems
2.  ML Problem Framing
3.  Data Collection / Integration
4.  Data Preparation
5.  Data visualization and analysis
6.  Feature Engineering - converts raw data into higher representation

* Introduce nonlinearity into linear models with binning - break up continuous values - e.g., bin Age because Age isn't a linear relationship with salary (goes up to a point, then down)
* Quadratic features - introduce features by combining other features (e.g., years of eduction + occupation) - creates more complex features
* Log or polynomial target of the target or features
* Can use leaves of a decision tree as a feature

* Domain Specific Transformations
    * NLP
        * Stop word removal / stemming
        * Lowercasing, punctuation removal
        * Cutting off very high / low percentiles
        * TF/IDF normalization
    * Web page features
        * Multiple fields of text:  URL, in/out anchor text, title, frames, body, presence of certain HTML elements (image/table)
        * Relative style (bold, italics) and positioning

7.  Model Training

* Parameter tuning
    * Loss function
        * Square - regression, classification
        * Hinge - classification only, more robust to outliers
        * Logistic - classification only, better for more skewed distributions
    * Regularization
        * Prevent overfitting by constraining weights to be small
    * Learning parameters (e.g., decay rate)
        * Decaying too aggressively - algorithm never reaches optimum
        * Decaying too slowly - algorithm bounces around, never converges to optimum
        * NOTE: These are from the slides, but it's a poor explanation and I think they're reversed
        
 8.  Model Evaluation
 
 * Evaluation metrics for regression
    * RMSE
    * Mean Absolute Percent Error (MAPE)
    * R-squared - how much better is the model relative to just picking the best constant
        * R-squared = 1 - (model MSE / variance)
        * Check on test dataset, not training dataset
* Evaluation metrics for classification
    * confusion matrix
    * ROC curve
    * Precision-Recall
        * Precision = TP / everything you said was positive (TP + FP)
        * Recall = TP / everything that was actually positive (TP + FN)
        
9.  Business Goal Evaluation

* If data augmentation is needed, return to step 3 (Data Collection / Integration)
* If feature augmentation is needed, return to step 5 (Data Visualization and analysis)
* Monitor production data distribution, trigger model retraining if significant drift is found between original training data distribution and what's coming through now
* If that's hard or expensive, just retrain model periodically, like daily/weekly/monthly

