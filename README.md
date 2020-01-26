# Python-Chatbot
A chatbot is an intelligent piece of software that is capable of communicating and performing actions similar to a human. Chatbots are used a lot in customer interaction, marketing on social network sites and instantly messaging the client. There are two basic types of chatbot models based on how they are built; Retrieval based and Generative based models.


# About

In this Python project with source code, we are going to build a chatbot using deep learning techniques. The chatbot will be trained on the dataset which contains categories (intents), pattern and responses. We use a special recurrent neural network (LSTM) to classify which category the user’s message belongs to and then we will give a random response from the list of responses.

Let’s create a retrieval based chatbot using NLTK, Keras, Python, etc.

## Prerequisites

-Tensorflow

-Keras

-NLTK

-Pickle

## File Description

Intents.json – The data file which has predefined patterns and responses.

train_chatbot.py – In this Python file, we wrote a script to build the model and train our chatbot.

Words.pkl – This is a pickle file in which we store the words Python object that contains a list of our vocabulary.

Classes.pkl – The classes pickle file contains the list of categories.

Chatbot_model.h5 – This is the trained model that contains information about the model and has weights of the neurons.

Chatgui.py – This is the Python script in which we implemented GUI for our chatbot. Users can easily interact with the bot.

## Steps

### 1. Import and load the data file

First, make a file name as train_chatbot.py. We import the necessary packages for our chatbot and initialize the variables we will use in our Python project.

The data file is in JSON format so we used the json package to parse the JSON file into Python.

### 2. Preprocess data


When working with text data, we need to perform various preprocessing on the data before we make a machine learning or a deep learning model. Tokenizing is the most basic and first thing you can do on text data. Tokenizing is the process of breaking the whole text into small parts like words.

Here we iterate through the patterns and tokenize the sentence using nltk.word_tokenize() function and append each word in the words list. We also create a list of classes for our tags.

### 3. Create training and testing data

Now, we will create the training data in which we will provide the input and the output. Our input will be the pattern and output will be the class our input pattern belongs to. But the computer doesn’t understand text so we will convert text into numbers.

### 4. Build the model

We have our training data ready, now we will build a deep neural network that has 3 layers. We use the Keras sequential API for this. After training the model for 200 epochs, we achieved 100% accuracy on our model. Let us save the model as ‘chatbot_model.h5’.

### 5. Predict the response (Graphical User Interface)

Now to predict the sentences and get a response from the user to let us create a new file ‘chatapp.py’.

We will load the trained model and then use a graphical user interface that will predict the response from the bot. The model will only tell us the class it belongs to, so we will implement some functions which will identify the class and then retrieve us a random response from the list of responses.

### 6. Run the chatbot

python train_chatbot.py

python chatgui.py
