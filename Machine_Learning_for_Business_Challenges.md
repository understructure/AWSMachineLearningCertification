# AWS ML - ML basics

## Machine Learning for Business Challenges

In this course we join some of Amazon's own Machine Learning Scientists for discussions on how to appropriately use and think about Machine Learning. Topics include ML terminology, business problems, use cases, and examples. By the end of this course you will have a better understanding of how to consider machine learning  business challenges and decisions.

### Machine Learning for Business Leaders

* When is ML appropriate?
 
* ML can't be used to:
    * Determine causality
    * Increase revenue
    * Increase customer satisfaction
    * (Though the first statement is contradicted by the third video, and the last two are contradicted in the second video...)
* ML **can** be used 
    * When problem is persistent
    * If problem challenges progress or growth
    * If solution needs to scale
    * If problem needs personalization
    
* What does a successful ML solution require?
    * People - several skillsets
    * Time - long-term committment
    * Cost
    
* Six questions to ask:

1.  What assumptions did your team make (in data and algorithm choice)?
2.  What is the specific learning target? (variable that's the output / hypothesis) - hypothesis testing with large datasets is the basic premise of ML
3.  What type of ML problem is it?
4.  Why did you choose this algorithm? - Did you consider different baselines based on literature?
5.  How will you evaluate model performance?
6.  How confident are you that you can generalize results?

* Make sure your scientists have access to literature, time to learn latest techniques, access to conferences, etc.

### Introduction to Machine Learning

* ML is interdisciplinary - grew out of statistics, computer science, signal processing, data analysis, AI, and pattern recognition
* Over 100 Amazon systems using ML, over 100 billion predictions every day
* ML starts and ends w/ data
* Train model - better predictions means better recommendations, more satisfied customers, and more sales

### Machine Learning Business Understanding

* ML is the dominant subfield of AI
* ML can do "Causal inference such as identifying causes of supply chain disruption"

### How to Define and Scope a Machine Learning Problem

* Requirements gathering 
    * what's the problem you're trying to solve?
    * What's the existing solution
    * What are the pain points, and what's causing them?
    * What is the problem's impact?
    * How would the solution be used?
    * What's the scope?
    * How do you define success?
* Inputs gathering
    * Do we have sufficient data?
    * Labeled examples?
    * If not, how difficult is it to create/obtain data?
    * What are our features?
    * What are the most useful inputs?
    * Where is the data?
    * What's the data quality like?
* Output definition
    * What business metric defines success?
    * What are the tradeoffs?
    * Are there existing baselines?
    * If not, what's the simplest thing we can use as a baseline?
    * Any data validation set we need to use to make the case to others?
    * How important is runtime and performance, SLA, etc.
* What's the task type (regression, classification)?
* What are the risks and assumptions?

### When is Machine Learning a Good Solution?

* When you need to automate a process or DSS where it's easier to learn from data than code rules
    * Combining weak or unseen evidence - probability
    * ASIN - Amazon Specific Item Number
* When manual process isn't cost effective or doesn't scale
    * Competitive tasks that require human expertise, but volume of work is too large
    * Amazon has a long tail of products that are almost never purchased - might want to use ML to translate details page
    * Quality of solution in relation to value of the problem
* When you have ample data to learn from
* When problem can be formalizable as an ML problem
    * If you can't formalize it, you can't learn it
    * e.g., reduce to well-known ML problem like regression or classification
    * Grocery store floor placement isn't a good ML problem, at least as stated - is the goal to make the trip to the store as long as possible?  As short as possible?

### When is Machine Learning NOT a Good Solution?

* When a traditional software solution would work nearly as well
* For problems with no data or with no labels
* Need to launch quickly - timelines are extremely variable - usually better to launch a simpler version faster
* Where there's no tolerance for mistakes - takes often years to get to this point

### Machine Learning Application

* Supervised learning - translates inputs to outputs, need ground truth labels
    * Regression
    * Classification
* Unsupervised learning - no lablels, used to discover patterns
    * Clustering
    * Topic modeling
* Reinforcement learning

### Machine Learning Business Problem: Gift Wrap Eligibility

* Feature vector - properties you think are relevant to making a prediction
* When is an ML solution a good one?   When...
    * It's difficult to directly code a solution
    * Difficult to scale a code-based solution
    * We want personalized output
    * Functions change over time

### Data, Data, Data

* NNs are more possible now because of compute power and volume of data
* "Design matrix" - table of features and labels, essentially
* Text data is both high-dimensional (many different words) and sparse (only a very few words of the total possible are used in any given case)
* Set data - group of items purchased / viewed together
* ML Data Scope Questions
    * How much data is sufficient for building successful ML models?
    * How to deal with data quality issues?
        * Missing value imputation
        * Outlier detection
    * Data preparation prior to model building
        * Center numerical values with unit variance
        * One-hot encode categorical variables
        * Use stemming, remove stopwords, extract n-grams
        
### Image Classification: Vocabulary and Example

* Labeling
* Feature engineering - e.g., smaller standard deviation or range of pixel brightness might mean not much info in the image
* Accuracy of model depends on model chosen as well as input features
* The best model is the one that fits the business need best - not necessarily the one with the highest accuracy
* Some things are just too hard, like realizing sarcasm

### Reinforcement Learning: Robot Programming Example

* Consider a robot moving around a board and eating plants - some are good, some are poison
* Learn a mapping of current agent's state to a desired action (policy)
* Feature vector ~= vector representing current state
* Prediction ~= next action
* Action ~= reward
* State = encoding of the agent and current (observable) environment
* Action = move up, down, left, right, eat plant
* Past data = where robot has traveled, what robot has eaten so far, corresponding rewards received
* Future data = new parts of the grid, or new grids altogether
* Maximize cumulative reward in the future
* "Given this setting, reinforcement learning combines exploring the environment while attempting to maximize accumulated rewards to learn a policy for future use"
* In reality, we might need to encode images of poisonous and non-poisonous plants and use those to feed a poisonous plant image classifier, and take its prediction as part of a state description
* Reinforcement Learning Differences:
    * No explicit presentation of input/output pairs
    * Reward based
    * Agent needs to gather useful experiences
    * Evaluation of the system is often concurrent with learning

### Machine Learning in Action: The Pollexy Project

* This video is an introduction to The Pollexy Project, the service combining Alexa, Amazon Polly and Amazon Lex. The Pollexy Project is a Raspberry Pi and mobile-based special needs verbal assistant that lets caretakers schedule audio task prompts and messages both on a recurring schedule and/or on-demand. In this video, we introduce you to the project and showcase a specific use for the project: assisting children with autism. 
