
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

* Precisely define business problem
* What's the business metric?
* Is ML the appropriate approach?
* What data is available?
* What type of ML problem is it?
    * Frequently multiple models is the answer
* What are your goals?

### Data Collection

* Data collection - acquiring training or test data
* Motivation
    * Initial datasets for training models and measuring success
    * Adding data for tuning
    * Replacements for flawed or outdated sets
    * Data and model maintenance post production
* Sources
    * Logs
    * Databases
    * Web sites (crawling / scraping)
    * Data providers (public / private)

### Data Collection: Sampling

* Select subset of instances for training and testing
* Instance = example = data point
* Representativity - needs to be unbiased / representative of the population
    * especially important for testing and measurement sets
    * also important for training sets to get good generalization
* Random sampling - so each data point has equal probability of being selected
* Stratified sampling - apply random sampling to each subpopulation separately
* Usually, sampling probability is the same for each stratum
    * If not, weights can be used in metrics and/or directly in training
* Seasonality - time of day, day of week, time of year (seasons), holidays, special events, etc.
    * Stratified sampling across these can help minimize bias
    * Visualization can help
* Trends - patterns can shift over time, and new ones can emerge
    * To detect, try comparing models trained over different time periods
    * Visualization can help
* Consider using validation data that was gathered after your training data was gathered
* Leakage is specific to how data was prepared and created, or to whether it's available in training, testing and production
* Leakage - using information during training or validation that's not available in production
* Train / test bleed - inadvertent overlap of training and test data when sampling to create datasets
    * Sample from fresh data
    * Filter out already selected instances
    * Partition source data along some dimension (but avoid bias)
    * Be especially careful with time-series data and data with duplicate entries

### Data Collection: Labeling

* Obtaining gold-standard answers for supervised learning
* Motivation - often, lables aren't readily available in sampled data
* Examples
    * Search - results a customer wanted to receive
    * Music categorization - genre
    * Sentiment analysis
    * Digitization - transcribing handwriting
    * Object detection - localization of objects in images
* Sometimes lables can be inferred (e.g., from click-through data)
* Human labels can be preferable to minimize bias, capture subtleties, etc.
* Labeling guidelines (instructions to labelers, critical to get right, minimize ambiguity)
* Labeling tools
    * Excel spreadsheets
    * Mechanical Turk
    * Custom-built tools
* Questions
    * Human intelligence tasks (HITs) should be:
        * Simple
        * Unambiguous
* Poor design of tools or HITs can
    * Impact labeler productivity and quality
    * Introduce bias
* Managing Labelers
    * Motivation - bad labels lead to poor ML results
    * Plurality - assign each HIT to multiple labelers to identify difficult or ambiguous cases, or problematic labelers (lazy, confused, bias, etc.)
    * Gold Standard HITs - HITs with known labels mixed in to identify problematic labelers
    * Labeler Incentives - compensation, rewards, voluntary, gamification
    * Quality and productivity metrics - can help detect problems with labelers
* Sampling and treatment assignment
    * Random sampling + random assignment - ideal experiments - causal conclusion and can be generalized (rarely available in traditional analysis, but become available in online testing)
    * Random sampling without random assignment - Typical survey or observation studies - cannot establish causation, but can find correlation and can be generalized (additional work needed to infer causation)
    * Random assignment without random sampling - Most experiments - Causal conclusion for the sample only (more work is needed to generalize)
    * No random sampling, no random assignment - BAD - badly-designed survey or pooled studies - can't establish causation, can't generalize to larger population (more work is needed to draw useful conclusions)
* Random Assignment - can infer causation, otherwise, only correlation
* Random Sampling - should generalize, can't generalize without random sampling


### Exploratory Data Analysis: Domain Knowledge

Typical ML workflow (longer arrow = data augmentation, shorter arrow = feature augmentation)

![Typical ML workflow](ds001.png)

* Important to understand data as much as possible - this informs subsequent steps in the ML process
* Domain Knowledge
    * Understand how the domain works, important dynamics and relationships, constraints, how data is generated, etc.
    * Better understanding of domain leads to better features, better debugging, better metrics, etc.
* When domain is outside your realm
    * Consult domain experts (AWS ML specialist SA's, AWS professional services, AWS ML Solutions Lab)
    * Seek other resources (AWS Partner network)

### Exploratory Data Analysis: Data Schema

* Convert / connect various data sources to S3, then send to SageMaker
* Join data with Pandas `merge()` function, `how=inner`


### Exploratory Data Analysis: Data Statistics

* Rows and columns
* Univariate statistcs (mean, sd, variance, for numeric variables, and histograms, most/least frequent values, percentage, number of unique values for categorical)
    * `df.column.value_counts()` or Seaborn's `distplot()`
* Target Statistics
    * Class distribution (`df.target.value_counts()` or `np.bincount(y)`)
* Multivariate statistics - correlation, crosstabs ("contingency tables")
* `df.info()` - shows datatype for each column
* `df.describe()` - shows summary statistics for each column
* Plots

* Density plot

`df['varname'].plot.kde()`
`plt.show()`

* Histogram

`df['varname'].plot.hist()`
`plt.show()`

* Box Plot

`df.boxplot(['varname'])`
`plt.show()`

* Scatterplot

`df.plot.scatter(x='V1', y='V2')`
`plt.show()`

* Scatterplot Matrix - histogram on diagonal, pairwise scatterplot otherwise

`pd.scatter_matrix(df[['V1', 'V2', 'V3']], figsize=(15,15))`

https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

### Exploratory Data Analysis: Correlations

* Pearson correlation - covariance of x and y divided by standard deviation of x * standard deviation of y

### Exploratory Data Analysis: Data Issues

* Missing data
* Noisy data
* Might want to impute
* Messy data, data on different scales, mixed data type (e.g., 20 and twenty)
* Imbalanced data
* Sample bias
* Outliers
* Highly correlated features - can lead to multicolinearity

## Data Processing and Feature Engineering

### Data Preprocessing:  Encoding Categorical Variables


* Pandas supports a data type of "category" but still ML models need numeric inputs
* Ordinal - keep in mind relative magnitude (e.g., S, M, L, XL)
* Can use Pandas `map()` function to map a dict of text keys to numerical values - keys are values in the column
* Can use `sklearn.preprocessing.LabelEncoder()` for labels ONLY with binary variables
    * this is the WRONG solution when there is no relationship between categories with more than two categories (OK for binary data)

### Data Preprocessing: Encoding Nominal Variables

* Can use Pandas `get_dummies()` function - automatically assigns column names as well
* One-hot encoding - sometimes have to use `reshape(-1,1)` with the `OneHotEncoder()` from sklearn

`from sklearn.preprocessing import OneHotEncoder`

* Encoding with many classes - define a hierarchy structure
    * Example - for a column with ZIP code, use regions -> states -> city as the hierarchy and choose a specific level to encode the ZIP code column
* Try to group levels by similarity to reduce overall number of groups

### Data Preprocessing: Handling Missing Values

* Most ML models can't handle missing values
* Pandas - check how many missing values for each column: `df. isnull.sum()`
* Pandas - check how many missing values for each row: `df. isnull.sum(axis=1)`
* Pandas - can use `dropna()` to drop rows with null values
* Pandas - can use `dropna(axis=1)` to drop columns with null values
    * `df.dropna(how='all')`
    * `df.dropna(thresh=4)`
    * `df.dropna(subset=['Fruits'])`
* Risks of losing data - losing too much data leads to overfitting, wider confidence intervals, etc.
* Too much data dropped can lead to bias
* Risk of dropping columns - may lose information in features (underfitting)
* Before dropping or imputing missing values, ask:
    * What were the mechanisms that caused the missing values?
    * Are these missing values missing at random?
    * Are there rows or columns missing that you are not aware of?
* If we believe missing values are random, can do imputation
    * Mean or median for numeric variables
    * Most frequent for categoricals
    * Or any other estimated value
* `from sklearn.preprocessing import Imputer`
* `imputer = Imputer(strategy='mean')`
* then call `imputer.fit()` and `imputer.transform()` on an object (like an array)
* Advanced Methods for Imputing Missing values
    * MICE (multiple imputation by chained equations) - `sklearn.impute.MICEImputer` (v 0.20)
* Python (not sklearn) `fancyimpute` package
    * KNN impute
    * Soft impute
    * MICE
    * etc.

### Feature Engineering

* scikit-learn: `sklearn.feature_extraction`
* often more art than science
* rule of thumb - use intuition.  What information would a **human** use to predict this?
* Try generating many features first, **then** apply dimensionality reduction if needed
* Consider transformations of attributes - square a number, multiply two features together
* Try not to overthink or include too much manual logic

### Feature Engineering: Filtering and Scaling

* Images - remove channels from an image if color isn't important
* Audio - remove frequencies from audio if power is less than a threshold
* Many algorithms like kNN and gradient descent are sensitive to features being on different scales
* Decision trees and random forests aren't usually  sensitive to features being on different scales
* Scaling - want values between -1 and 1, or 0 and 1
* Fit the scaler to training data only, then transform both train and validation data
* Common choices for scaling in `sklearn`:
    * Mean/variance  - z-score, subtract mean, divide by standard deviation - produces mean 0 and standard deviation 1 - `sklearn.preprocessing.StandardScaler`
        * Advantages include many algorithms behave better with smaller values, and it keeps outlier information but reduces its impact
    * MinMax scaling - minimum = 0, maximum = 1, then subtract min from each value and divide by max minus min - `sklearn.preprocessing.MinMaxScaler`
        * Advantage: Robust to small standard deviations
        * Difference b/t min and max is usually larger than standard deviation
        * When SD is very small, mean/variance scaling isn't going to be robust, because we're dividing by a very small number
    * Maxabs scaling - take maximum absolute value in dataset and divide everything by that
        * Not centered, just scaled [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)
    * Robust scaling - look at a feature in training data set and find median, 25th quantile and 75th quantile.  Then subtract median and divide by difference b/t 75th and 25th quantile.  This makes the feature robust to outliers, because outliers have little effect in calculating medians and quantiles
* Scaling is for one column, normalizing is for one row
* Normalizer - rescales $ x_j $ to unit norm based on 
    * L1 norm
    * L2 norm
    * Max norm
* Normalizers used a lot in text analysis
* `sklearn.preprocessing.Normalizer`


### Feature Engineering: Transformation

* Sometimes a polynomial is a better fit than a linear feature
* `sklearn.preprocessing.PolynomialFeatures`
    * Raises things to a power (all powers up to a power) as well as the interaction (a * b) between features
* Beward of overfitting if the degree is too high
* Consider non-polynomial transforms as well, like:
    * Log transforms
    * Sigmoid transforms
* Risk of extrapolation beyond the range of data when using polynomial transformations
* Radial Basis Function - transform the data through a center, represented below by $ c $
* $ f(x) = f(\vec{x - c}) $
    * Widely used in SVM as a kernel and in Radial Basis Neural Networks (RBNNs)
    * Gaussian RBF is the most common RBF used

### Feature Engineering: Text-Based Features

* Bag-of-words model
    * Represent document as vector of numbers, one for each word (tokenize, count and normalize)
* NOTE
    * Sparse matrix implementation is typically used, ignores relative position of words
    * Can be extended to bag of n-grams of words or characters
* Count vectorizer
    * Per-word value is count (also called term frequency)
        * NOTE: Includes lowercasing and tokenization on white space and punctuation
        * scikit-learn: `sklearn.feature_extraction.text.CountVectorizer`
* `TfidfVectorizer` - Term Frequency times Inverse Document Frequency
* Per-word value is *downweighted* for terms common across documents (e.g., "the")
* `sklearn.feature_extraction.text.TfidfVectorizer`
* A more efficient way of mapping words and counting frequencies is by using `HashingVectorizer` - stateless mapper from text to term index
* `sklearn.feature_extraction.text.HashingVectorizer`


