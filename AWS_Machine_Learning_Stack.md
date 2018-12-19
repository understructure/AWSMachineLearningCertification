# AWS ML - ML basics

## AWS Machine Learning Stack

https://www.aws.training/transcript/curriculumplayer?transcriptId=Af9SbtLa5EeVRGCTzixUMQ2

* Application Services 
	* Polly - 25 languages, 52 voices, custom lexicons, SSML tags, cache files created and play mutliple times
	* Lex - speech recognition in English & SPanish, natural language understanding, builds conversational interactions in voice and text, powered by same deep learning as Alexa
	* Rekognition - image and video analysis, text in images, person tracking, facial recognition, unsafe content detection
	* Transcribe - speech to text - Spanish & English, multiple speaker recognition, can be used for exception handling or analytics, integrate with Comprehend for contextual analysis
	* Translate - realtime and batch
	* Comprehend - NLP NN discovers entities, key phrases, different languages, sentiment analysis
	* Can use Transcribe, Translate and Comprehend with Lambda, S3 and Athena to build QuickSight dashboard for near realtime sentiment analysis
* Platform Services 
	* Sagemaker
		* Single environment to train, prepare data, choose & optimize algorithm, set up environments, tune and train, deploy model in production and scale and manage the prod environment
		* 12 algorithms out of the box, six "infinitely scalable", six support data streaming - train in single pass, produces rich format that can be used with multiple sets of hyperparameters, may be able to tune hyperparameters without retraining the model
	* DeepLens
* Frameworks & Interfaces - mostly done with GPU instances
	* Deep Learning AMI - Amazon Linux & Ubuntu - no additional charges to use these AMIs
		* MXNet
		* TensorFlow
		* Caffe
		* Caffe2
		* Keras
		* Theano
		* Torch
		* Microsoft Cognitive Toolkit
		* Gluon - high-level DL interface - improves speed, flexibility, accessibility of DL technology to developers, supports multi-frameworks, simple, easy-to-use code, flexible, imperative structure, high performance (only in MXNet right now)
		* ONNX - Open Neural Network Exchange - PyTorch, MSCT, MXNet, Caffe2 - developers choose framework that best fits their needs - provides portability across environments - AWS + Facebook + Microsoft collaboration
* Physical Infrastructure
	* Up to 8 NVIDIA V100 GPUs in a single instance
	* Support 16xlarge size, which provides:
		* Combined 128GB of GPU memory, more than 40,000 CUDA cores
		* More than 125 teraflops of single-precision floating point performance
		* More than 62 teraflops of double-precision floating point performance
		* ~14X faster than P2
* IoT Edge Devices - Greengrass - great for remote environments w/out constant cloud connection, tight security constraints, or safety reasons (e.g., when safety issue requires instant shutdown), great where there are privacy, regulatory or compliance requirements	

Benefits
1 - make best use of DS time - e.g., use Athena and Glue
2 - convert power of ML into business value
3 - embed ML into the business fabric

