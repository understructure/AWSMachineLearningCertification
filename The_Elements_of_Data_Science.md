
## The Elements of Data Science

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



### Supervised Learning: Logistic Regression and Linear Separability







