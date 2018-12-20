
# The Elements of Data Science

## What is Data Science?

### Introduction to Data Science

* Data Science - Processes and systems to extract knowledge or insights from data, either structured or unstructured
* For the purpose of this course - Managing, analyzing and visualizing data in support of the ML workflow
* DS provides the workflow for collecting high quality data used to train models
* Machine Learning - set of algorithms used by AI systems to improve their predictions by learning from large amounts of input data
* Data must be quality and specific to problem you're trying to solve
* ML main idea - learning = estimating underlying function f by mapping data attributes to some target value
* Training set - labeled examples where x is the input variables and f(x) is the observed target truth
* Goal - given a training set, find approximation of $ \hat{f} $ of $ f $ that best generalizes / predicts labels for new examples
    * "Best" is measured by some quality measure
    * Example: error rate, sum squared error
* Why Use ML?
    * Difficulty in writing some programs
        * Too complex (Facial recognition)
        * Too much data (stock market predictions)
        * Information only available dynamically (recommendation system)
    * Use of data for improvement
        * Humans are used to improving based on experience (data)
    * A lot of data is available
        * Product recommendations
        * Fraud detection
        * Facial recognition
        * Language comprehension

* For any business problem, we have to start with data

#### Data Science and ML Workflow

* Problem formulation
* Data collection
* Exploratory analysis
* Data preprocessing
* Feature engineering
* Model training
* Model evaluation
* Model Tuning
* Model debugging
* Productionisation

* Always an iterative process
* Feature = attribute = independent variable = predictor
* Dimensionality - number of features

### Types of Machine Learning


* Supervised learning - Models learn from training data that has been labeled.
* Semi-supervised learning - Some have true labels, some don't
* Unsupervised learning -  Models learn from test data that has not been labeled.
* Reinforcement learning - Models learn by taking actions that can earn rewards.
    * Algorithm isn't told what action is correct, but given reward or penalty based on actions it performs


### Key Issues in ML

* Data quality is the number one thing
* Consistency of data w/ problem you're trying to solve
* Accuracy of data - both labels and features
* Noisy data
* Missing data
* Outliers - even when the values are correct, might not be relevant or apply to the scope of the problem
* Bias
* Variance, etc.

* Model quality - overfitting vs. underfitting
* Overfitting - corresponds to high variance - small changes in training data lead to big changes in results (model is too flexible)
* Underfitting - corresponds to high bias - results show systematic lack of fit in certain regions (model is too simple)
* Computational speed and scalability
    * Use distributed computing systems like SageMaker or EC2 instances for training in order to
        * Increase speed
        * Solve prediction time complexity
        * Solve space complexity

### Supervised Methods: Linear Regression

* Linear models widely used in ML due to their simplicity
* Parametric methods where function learned has the form $ f(x) = \phi (w^Tx) $ where $ \phi() $ is some activation function
* Generally, learn weights by applying (stochastic) gradient descent to minimize loss function
* Simple; a good place to start for a new problem, at least as a baseline
* Methods
    * Linear regression for numeric target outcome
    * Logistic regression for categorical target outcome

#### Linear Regression (Univariate)

* Only one input and one output
* Model relation b/t single features (explanatory variable x) and a real-valued response (target variable y)
* Given data (x, y) and a line defined by $ w_0 $ (intercept) and $ w_1 $ (slope), the vertical offeset for each data point from the line is the error between the true label y and prediction pased on x
* The best line minimizes the sum of squared errors (SSE): $ \sum (y_i - \hat{y}_i) $
* We usually assume the error is Gaussian distributed with mean zero and fixed variance

#### Linear Regression (Multivariate)

* Multiple linear regression includes N explanatory variables with N >= 2:

$ y = w_0 x_0 + w_1 x_1 ... + w_m x_m $ = (sum from i=0 to N of $ w_i x_i $)

* Sensitive to correlation between features, resulting in high variance of coefficients
* scikit-learn implementation: `sklearn.linear_model.LinearRegression()`


### Supervised Learning: Logistic Regression and Linear Separability


* Response is binary (1/0)
* Estimates the probability of the input belonging to one of two classes: positive and negative
* Vulnerable to outliers in training data
* scikit-learn: `sklearn.linear_model.LogisticRegression()`
* Relationship to linear model: sigma(z) = 1 / (1 + e ** -z)
    * z is a trained multivariate linear function
    * phi is a fixed univariate function (not trained)
    * Objective function to maximize = probability of the true training labels
* Sigmoid curve is a representation of the probability (goes b/t 0 and 1)
* x - infinity to + infinity
* Model relation between features (explanatory variables x) and the binary responses (y = 1 or y = 0)
* For all the features, define a linear combination

$ z = w^T x = w_0 + w_1 x_1 + ... + w_n x_n $

* Define probability of y = 1 given x as p and find the logit of p as
* Use the logit function to transform z into value between 0 and 1

$ logit(p) = log \frac{p}{1 - p} $

* Logistic regression finds the best weight vector by fitting the training data

$ logit(p(y = 1 | x)) = z $

* Then for a new observation, you can use the logistic function $ \phi(z) $ to calculate the probability to have label 1.  If it's larger than a threshold (e.g., 0.5), you will predict the label for the new observation to be positive
* Outliers can really skew the results - even a few can mess up predictions
* Linearly separable - must be a single straight line?
* Can still use logistic regression even if data aren't linearly seperable?


## Problem Formulation and Exploratory Data Analysis

### Problem Formulation and Exploratory Data Analysis

* Articulate the business problem and why ML is the best approach
* Be sure ML will provide the best solution
* Data sources
* Open data on AWS
* Bias and sampling techniques
* Lableing components and tools and managing labelers
* How domain knowledge can help in exploring the data
* What to do when the domain is outside your own expertise
* Data sources, merges and joins
* Types of statistics and plots
* Formulas for correlations

### Problem Formulation





### Data Collection



### Data Collection: Sampling


### Data Collection: Labeling


### Exploratory Data Analysis: Domain Knowledge

Typical ML workflow:
(insert image here)


### Exploratory Data Analysis: Data Schema



### Exploratory Data Analysis: Data Statistics



### Exploratory Data Analysis: Correlations


### Exploratory Data Analysis: Data Issues




