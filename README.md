## End to End Deep learning  Project

# Cat vs Dog Image Classification

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Run](#run)
  * [Deployement on render](#deployement-on-render)
  * [Directory Tree](#directory-tree)
  * [To Do](#to-do)
  * [Bug / Feature Request](#bug---feature-request)
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [License](#license)
  * [Credits](#credits)


<!-- ## Demo -->
<!-- Link: [](https://ipcc.rohitswami.com) -->

<!-- [![](https://imgur.com/s4FWb9b)](https://ipcc.rohitswami.com) -->
## Cats vs. Dogs Image Classification
![Dog vs Cat](images/dogvscat.jpg)

## Overview
This project implements a deep learning model to classify images of dogs and cats. The model uses Convolutional Neural Networks (CNN) to distinguish between dog and cat images based on the features extracted from the images. This is a classic image classification problem typically used to showcase deep learning capabilities



## Motivation
The ability to classify images into categories, such as identifying whether an image contains a dog or a cat, is a fundamental task in the field of computer vision. This project aims to explore the power of deep learning, specifically Convolutional Neural Networks (CNNs), to solve image classification problems.


#Key Motivations for this Project:
Real-World Applications: Image classification is widely used in real-world applications like facial recognition, medical image analysis, and autonomous driving. The Dogs vs. Cats problem provides a simple but effective introduction to the challenges and methods used in the field of image classification.

Understanding Deep Learning: Convolutional Neural Networks (CNNs) are a cornerstone of modern computer vision tasks. This project serves as a hands-on exercise to understand CNN architecture, how they process images, and how to tune them for optimal performance.

Skill Development: By working through this project, the goal is to gain practical experience with various deep learning concepts including:

Image preprocessing
Data augmentation
Training and fine-tuning deep learning models
Model evaluation and performance analysis
Exploration of Transfer Learning: In addition to building the model from scratch, this project also explores the possibility of improving performance through transfer learning, where a pre-trained model is fine-tuned for the task at hand. This is especially useful when limited labeled data is available.

Benchmarking Deep Learning Performance: The Dogs vs. Cats dataset provides a simple benchmark for comparing different deep learning architectures and training techniques. It serves as an ideal starting point for individuals looking to understand how different models can be applied to image classification.

## Technical Aspect

This project is divided into two major parts:

Training Deep Learning Models:

We train Deep learning algorithms(CNN) on the Dog vs cat dataset.
All models are implemented using TensorFlow with Keras , a Python library for Deep leaning.

<br>

Building and Hosting a Flask Web App on Render:

A Flask web application is built to interact with the trained models and make real-time predictions based on user input.
The app is deployed using Render to provide easy access via the web.
Users can submit their data via a simple web interface and receive mental health predictions
    - 

## Installation
The Code is written in Python 3.10. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:

# To clone the repository

```bash

gh repo clone Creator-Turbo/Dog-vs-cat-clf-

```
# Install dependencies: (all lib)
```bash
pip install -r requirements.txt
```



## Run
To train the Deep leaning models:
 To run the Flask web app locally
```bash
python app.py

```
# Deployment on Render

## To deploy the Flask web app on Render:
Push your code to GitHub.<br>
Go to Render and create a new web service.<br>
Connect your GitHub repository to Render.<br>
Set up the environment variables if required (e.g., API keys, database credentials).<br>
Deploy and your app will be live!



## Directory Tree 
```
.
├── model
├── static
├── templates
├── .gitignore
├── app.py
├── dog_vs_cat_model.pkl
├── README.md
├── requirements.txt
└── tempCodeRunnerFile.py
```

## To Do




## Bug / Feature Request
If you encounter any bugs or want to request a new feature, please open an issue on GitHub. We welcome contributions!




## Technologies Used
Python 3.10<br> 
scikit-learn<br>
TensorFlow <br>
Flask (for web app development)  <br>
Render (for hosting and deployment)  <br>
pandas (for data manipulation) <br>
numpy (for numerical operations)  <br>
matplotlib (for visualizations) <br>



![](https://forthebadge.com/images/badges/made-with-python.svg)


[<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/260px-Scikit_learn_logo_small.svg.png" width=170>](https://pandas.pydata.org/docs/)
[<img target="_blank" src="https://miro.medium.com/v2/resize:fit:720/format:webp/0*RWkQ0Fziw792xa0S" width=170>](https://pandas.pydata.org/docs/)
  [<img target="_blank" src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSDzf1RMK1iHKjAswDiqbFB8f3by6mLO89eir-Q4LJioPuq9yOrhvpw2d3Ms1u8NLlzsMQ&usqp=CAU" width=280>](https://matplotlib.org/stable/index.html) 
 [<img target="_blank" src="https://icon2.cleanpng.com/20180829/okc/kisspng-flask-python-web-framework-representational-state-flask-stickker-1713946755581.webp" width=170>](https://flask.palletsprojects.com/en/stable/) 
 [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/NumPy_logo_2020.svg/512px-NumPy_logo_2020.svg.png" width=200>](https://aws.amazon.com/s3/) 
 [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/TensorFlow_logo.svg/512px-TensorFlow_logo.svg.png" width=200>](https://www.tensorflow.org/api_docs) 
 [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Keras_logo.svg/512px-Keras_logo.svg.png" width=170>](https://keras.io/) 






## Team
This project was developed by:

Bablu kumar pandey

<!-- Collaborator Name -->




## Credits

Special thanks to the contributors of the scikit-learn library for their fantastic machine learning tools.

