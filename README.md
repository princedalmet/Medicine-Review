# Medicine-Review

Sentiment analysis for Reviews using NLP


Natural Language Processing
Natural Language Processing or NLP is a branch of Artificial Intelligence which deal with bridging the machines understanding humans in their Natural Language. Natural Language can be in form of text or sound, which are used for humans to communicate each other. NLP can enable humans to communicate to machines in a natural way.

Text Classification is a process involved in Sentiment Analysis. It is classification of peoples opinion or expressions into different sentiments. Sentiments include Positive, Neutral, and Negative, Review Ratings and Happy, Sad. Sentiment Analysis can be done on different consumer centered industries to analyse people's opinion on a particular product or subject.

Sentiment Classification is a perfect problem in NLP for getting started in it. You can really learn a lot of concepts and techniques to master through doing project. 

I will go through all the key and fundament concepts of NLP and Sequence Models, which you will learn in this notebook.Sentiment AnalysisLet's get started with code without furthur ado.


![1](https://user-images.githubusercontent.com/99526815/155457712-3d6d1c8e-f206-4632-bc7b-08422a490608.PNG)


Importing Dependencies

We shall start by importing all the neccessary libraries. I will explain the exact use of each library later in this notebook.

Dataset Preprocessing

In this notebook, Medicine review Sentiment from Kaggle. It contains  review of patients  and I find it a good amount of data to train our model.


We are going to train only on text to classify its sentiment. So we can ditch the rest of the useless columns.

