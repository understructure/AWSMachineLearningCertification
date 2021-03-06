### CRISP-DM

* Developed in 1996
* Cross Industry Standard Process - Data Mining
* Framework for DS projects
* Six phases:

1. Business understanding
	* Is the problem appropriate for ML?
2. Data understanding (--> Business understanding)
	* Data collection
	* Data properties / EDA / visualization / standard stats
	* Quality 
3. Data preparation
	* Select dataset
	* Feature engineering
4. Modeling (--> data preparation)
	* Select & create model
	* Tune model (choose values for parameters)
5. Evaluation (--> Business understanding)
	* Model evaluation
	* Business objective evaluation
	* Final decision
6. Deployment 
	* Planning deployment
	* Maintenance and monitoring
	* Final report
	* Project review

### Business Understanding

Main tasks: 

1. Understanding business requirements
	* Form business question (specific and measurable)
	* Highlight project's critical features
2. Analyzing supporting information
	* List required resources and assumptions
	* Analyze associated risks
	* Plan for contingencies
	* Compare costs and benefits
3. Converting to a data mining problem / ML objective
	* Review ML question
	* Create technical DM / ML objective
	* Define the criteria for successful outcome of the project
4. Preparing a preliminary project plan
	* Number and duration of stages
	* Dependencies
	* Risks
	* Goals
	* Evaluation methods
	* Tools and techniques


### Data Understanding

1. Data collection
	* Detail various sources and steps to extract data
	* Analyze data for additional requirements
	* Consider other data sources
2. Data properties
	* Describe the data, amount of data used, and metadata properties
	* Find key features and relationships in the data
	* Use tools and techniques to explore data properties
3. Data quality
	* Verifying attributes
	* Identify missing data
	* Reveal inconsistencies
	* Report solution

* AWS services
	* AWS Athena - interactive query service
		* Interactive SQL on S3
		* Schema-on-read
		* Supports ANSI operators and functions
		* serverless
	* AWS Quicksight - cloud-powered BI service
		* Scale to hundreds of thousands of users
		* 1/10 cost of traditional BI solutions
		* Secure sharing and collaboration (StoryBoard)
	* AWS Glue - managed ETL service - 3 components
		* Build your data catalog
		* Generate and edit transformations
		* Schedule and run jobs

### Data Preparation and Modeling
* Final dataset selection - analyze constraints
	* Total size
	* Included and excluded columns
	* Record selection
	* Data type
* Preparing data for modeling
	* Cleaning
		* How is missing data handled?
			* Drop rows with missing values
			* Add default / mean value for missing data
			* Use statistical methods (e.g., regression) to calculate the value
		* Clean attributes with corrupt data or variable noise
	* Transforming / derive additional attributes from original data
	* Merging
			* Recommended to revisit data understanding phase afterwards
	* Formatting (to properly work with model)
		* Rearrange attributes
		* Shuffle data
		* Remove constraints of modeling tool (e.g., remove unicode data)
* Modeling - think of preparation and modeling as a single phase because one influences the other so much - data prep will be vastly different for different models / goals / etc.
	* Model selection and creation - Identify:
		* Modeling technique
		* Constraints  of modeling technique and tool
		* Ways in which constraints tie back to data preparation phase
	* Model testing plan
		* Train test split - 70-85% data for training, but if not enough data, can do k-fold cross validation
		* Select a model evaluation criterion
	* Parameter tuning/testing
		* Spark on EMR
			* Spark MLlib - dataframe-based API for ML
			* Use Jupyter/ iPython / Zeppelin notebooks or R Studio
			* Scala, Python, Java, R, SQL supported
			* Leverage spot instances for training jobs, run off-peak
		* EC2
			* Python - preinstalled GPU CUDA support for training, frameworks include:
				* MXNet
				* TensorFlow
				* Caffe2
				* Torch
				* Keras
				* Theano
			* Also includes Anaconda
			* R Studio - [install R Studio on EC2](https://aws.amazon.com/blogs/big-data/running-r-on-aws/)

		
### Evaluation

1. Evaluate how model is performing relative to business goals
2. Make final decision to deploy or not

* Evaluation the model - (mostly) objective assessment of how model is performing based on assessment criteria you determined earlier
* Evaluation depends on:
    * Accuracy of model
    * Model generalization on seen/unseen data
    * Evaluation of the model using the business success criteria
* Rank candidate models and write up summary of performance
* Review the project
    * Assess steps taken in each phase - anything overlooked?
    * Perform QA checks
        * Model performance using the determined data
        * Is the data available for future training?
* If you decide to deploy, copy model to S3 and terminate all resources used for training
* DEMO: Launching Jupyter notebook on EC2 using AWS deep learning AMI (he used G-series for graphics) (--no-browser flag, he opened up a new "screen"?)


### Deployment

1. Planning deployment - choose runtime, e.g., EC2, ECS, Lambda (no Sagemaker??)
    * Deployment - CodeDeploy, OpsWorks, Elastic Beanstalk
    * Infrastructure - CloudFormation, OpsWorks, Elastic Beanstalk
2. Maintenance and monitoring
    * Code Management - CodeCommit, CodePipeline, Elastic Beanstalk
    * Monitoring - CloudWatch, CloudTrail, Elastic Beanstalk

3. Final report
    * highlight processes used in the project
    * analyze if all project goals were met
    * detail findings of project
    * Identify and explain the model used and reason behind using the model
    * Identify customer groups to target with this model
4. Project review

    * Assess the outcomes of the project
    * Summarize results, write documentation
        * Common pitfalls
        * Choosing the right ML solution
    * Generalize the whole process to make it useful for the next iteration

[DEMO - deploy MXNet to Lambda](https://docs.aws.amazon.com/lambda/latest/dg/lambda-python-how-to-create-deployment-package.html)
