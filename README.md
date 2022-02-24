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

As both the dataset contains same columns we can combine them for better analysis

We are going to train only on text to classify its sentiment. So we can ditch the rest of the useless columns.



![2](https://user-images.githubusercontent.com/99526815/155458343-f12f4398-68e5-43ae-92a0-2e94f68d1901.PNG)


It's a very good dataset without any skewness. Thank Goodness.


![3](https://user-images.githubusercontent.com/99526815/155458511-1777e666-aa59-4fe0-b2f5-854592c145e6.PNG)

Now let us explore the data we having here...

Text Preprocessing

So we have to clean the text data using various preprocessing and cleansing methods. Let's continue


Stemming/ Lematization

For grammatical reasons, documents are going to use different forms of a word, such as write, writing and writes. Additionally, there are families of derivationally related words with similar meanings. The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally related forms of a word to a common base form.

Stemming usually refers to a process that chops off the ends of words in the hope of achieving goal correctly most of the time and often includes the removal of derivational affixes.

Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base and dictionary form of a word


![4](https://user-images.githubusercontent.com/99526815/155458748-b8a123b8-2f99-4567-a72d-292af6123717.PNG)


Stopwords
Stopwords are commonly used words in English which have no contextual meaning in an sentence. So therefore we remove them before classification. Some stopwords are...Stopwords English


![5](https://user-images.githubusercontent.com/99526815/155458934-0fecf66b-4ab3-41e0-a63a-45af4600b599.PNG)


NLTK is a python library which got functions to perform text processing task for NLP


Aaww.. It is clean and tidy now. Now let's see some word cloud visualizations of it.


Positive Words

Negative Words

Netrual Words


Train and Test Split


train_test_split will shuffle the dataset and split it to gives training and testing dataset. It's important to shuffle our dataset before training.


Tokenization
Given a character sequence and a defined document unit, tokenization is the task of chopping it up into pieces, called tokens , perhaps at the same time throwing away certain characters, such as punctuation. The process is called Tokenization.Tokenization


![6](https://user-images.githubusercontent.com/99526815/155459222-11f2a4c9-f573-4a5f-918e-80334adc8466.PNG)

tokenizer create tokens for every word in the data corpus and map them to a index using dictionary.

word_index contains the index for each word

vocab_size represents the total number of word in the data corpus


Using TensorFlow backend.


Now we got a tokenizer object, which can be used to covert any word into a Key in dictionary (number).

Since we are going to build a sequence model. We should feed in a sequence of numbers to it. And also we should ensure there is no variance in input shapes of sequences. It all should be of same lenght. But texts in tweets have different count of words in it. To avoid this, we seek a little help from pad_sequence to do our job. It will make all the sequence in one constant length MAX_SEQUENCE_LENGTH.


Label Encoding

We are building the model to predict class in enocoded form (0 or 1 as this is a binary classification). We should encode our training labels to encodings.


Word Emdedding

In Language Model, words are represented in a way to intend more meaning and for learning the patterns and contextual meaning behind it.

Word Embedding is one of the popular representation of document vocabulary.It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc.

Basically, it's a feature vector representation of words which are used for other natural language processing applications.

We could train the embedding ourselves but that would take a while to train and it wouldn't be effective. So going in the path of Computer Vision, here we use Transfer Learning. We download the pre-trained embedding and use it in our model.

The pretrained Word Embedding like GloVe & Word2Vec gives more insights for a word which can be used for classification. If you want to learn more about the Word Embedding, please refer some links that I left at the end of this notebook.

In this notebook, I use GloVe Embedding from Stanford AI which can be found here


Model Training - LSTM

We are clear to build our Deep Learning model. While developing a DL model, we should keep in mind of key things like Model Architecture, Hyperparmeter Tuning and Performance of the model.

As you can see in the word cloud, the some words are predominantly feature in both Positive and Negative tweets. This could be a problem if we are using a Machine Learning model like Naive Bayes, SVD, etc.. That's why we use Sequence Models.


![7](https://user-images.githubusercontent.com/99526815/155459672-857628e6-02a0-4fae-be12-f1558c807700.PNG)


Reccurent Neural Networks can handle a seqence of data and learn a pattern of input seqence to give either sequence or scalar value as output. In our case, the Neural Network outputs a scalar value prediction.

For model architecture, we use

1) Embedding Layer - Generates Embedding Vector for each input sequence.

2) Conv1D Layer - Its using to convolve data into smaller feature vectors.

3) LSTM - Long Short Term Memory, its a variant of RNN which has memory state cell to learn the context of words which are at further along the text to carry contextual meaning rather than just neighbouring words as in case of RNN.

4) Dense - Fully Connected Layers for classification


Optimization Algorithm

This notebook uses Adam, optimization algorithm for Gradient Descent. You can learn more about Adam here

Callbacks

Callbacks are special functions which are called at the end of an epoch. We can use any functions to perform specific operation after each epoch. I used two callbacks here,

LRScheduler - It changes a Learning Rate at specfic epoch to achieve more improved result. In this notebook, the learning rate exponentionally decreases after remaining same for first 10 Epoch.

ModelCheckPoint - It saves best model while training based on some metrics. Here, it saves the model with minimum Validity Loss.


Let's start training... It takes a heck of a time if training in CPU,and in my case i am running on cpu so the epoch value i have taken low.

Model Evaluation

Now that we have trained the model, we can evaluate its performance. We will some evaluation metrics and techniques to test the model.

Let's start with the Learning Curve of loss and accuracy of the model on each epoch.

Confusion Matrix

Confusion Matrix provide a nice overlook at the model's performance in classification task.


