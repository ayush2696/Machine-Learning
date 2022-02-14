Repository containing portfolio of data science projects completed by me  self learning purposes. Presented in the form of iPython Notebooks

**Contents**

**Custom Implementation**

  [Tf-idf](https://github.com/ayush2696/Machine-Learning/blob/main/Custom%20Implementation/TF_IDF.ipynb): In this notebook I implement Tf-idf vectorizer and compare its results with sklearn implementation.

  [Stocastic Gradient Descent](https://github.com/ayush2696/Machine-Learning/blob/main/Custom%20Implementation/Stocastic%20Gradient%20Descent.ipynb) : In this notebook I implement Stocastic gradient descent algorithm and compare the results with Sklearn implementation. The difference in weights was in power of 10^(-3). 

  [Performance_metrics](https://github.com/ayush2696/Machine-Learning/blob/main/Custom%20Implementation/Performance_metrics.ipynb) : In this notebook I implement F1-Score and AUC without using sklearn.

  _**Tools:**_ scikit-learn, Pandas,numpy, Seaborn, Matplotlib

**Supervised Learning**

  [Naive Bayes](https://github.com/ayush2696/Machine-Learning/tree/main/Supervised%20Learning/Naive%20Bayes) : In this notebook we apply Naive Bayes on Donors Dataset. We used two different types of vectorization for text data (Bag of Words and TF-IDF) and we use RandomSearchCV to find the best hyper-parameter. We also compare the results for auc values of BoW and Tf-idf in which the Tf-idf performed slightly well than BoW.
  
  [Random Forest](https://github.com/ayush2696/Machine-Learning/tree/main/Supervised%20Learning/Random%20Forest) : In this notebook we bootstrap rows and columns from Boston Housing Dataset to create multiple decision trees. The predictions are averaged over all the trees and then the Mean-Squared error was calculated. Hence here we create a Random-Forest.
  
  [Gradient Boosting Decision Tree(GBDT)](https://github.com/ayush2696/Machine-Learning/tree/main/Supervised%20Learning/Gradient%20Boosting%20Decision%20Tree) : In this notebook we GBDT on Donors Dataset. We used two different types of vectorization for text data (TF-IDF and TF-idf:W2V) and we also try to use Response Coding for numerical features. We also generate four other features using SentimentIntensityAnalyzer which is a part of NLTK. We use RandomSearchCV to find the best hyper-parameter. We also compare the results for auc values of TF-idf:W2V and Tf-idf in which the Tf-idf performed slightly well than TF-idf:W2V.

  _**Tools:**_ NLTK, scikit, Pandas, Seaborn, Matplotlib
  

**Un-Supervised Learning**
  [Clustering on Graph Dataset]() : In this notebook we perform clustering on actor-movie dataset. We consider two cost functions and then after applying clustering on different values of hyperparater we select the one with the highest cost. After which we visualise the clustered movie and actor using TSNE.
  
  [Recommendation Systems and Truncated SVD SGD algorithm to predict ratings]() : In this notebook we perform predict the rating of the movie given a user and movie. First create a adjacency matrix on User-Movie upon which  we used SVD(Single Value Decomposition) to create a dataset. The parameters of Loss Function are found using SGD(Stocastic Gradient Descent) and then we make the final predictions.
   _**Tools:**_ scikit-learn, Pandas,numpy, Seaborn, Matplotlib
  
**Deep-Learning**

  **General Implementation**
  
  [BackPropagation with Gradient Checking](https://github.com/ayush2696/Machine-Learning/blob/main/Unsupervised%20Learning/Clustering%20on%20Graph%20Dataset.ipynb) : In this notebook we take a neural network in which we implement back-propagation to find the weights of the network and to verify if our weights are correct or not we used gradient clipping. We also custom implement different optimizers( ADAM, Momentum and Vanilla).
  
  [Working With Callbacks](https://github.com/ayush2696/Machine-Learning/blob/main/Unsupervised%20Learning/Recommendation%20Systems%20and%20Truncated%20SVD%20SGD%20algorithm%20to%20predict%20ratings.ipynb) : In this notebook we work with different callbacks namely   ModelCheckpoint, LearningRateScheduler, EarlyStopping, Tensorboard, TerminateonNaN. We also implement custom Micro-f1 score function. After which we apply these callbacks and metrics on 4 models with different activations and initilizers.
 
  
  **NLP**
  
  [LSTM on Donors Dataset](https://github.com/ayush2696/Machine-Learning/blob/main/Deep%20Learning/NLP/Applying%20LSTM%20on%20Donor%20Dataset.ipynb) : In this notebook we apply LSTM on donors dataset. We create 3 different models each model takes text data and creates embeddings and then pass it through the LSTM layer whereas we modify how we deal with other features in all the 3 models then we compare the results of all the 3 models. In this notebook we use functional modelling instead of sequential modelling.
  
  [NLP with transfer learning](https://github.com/ayush2696/Machine-Learning/blob/main/Deep%20Learning/NLP/NLP%20with%20transfer%20learning.ipynb) : In this notebook we implement transfer learning on amazon reviews, in which we feed reviews to the model to predict the score they might give. Here we use Pretrained BERT model to generate the embeddings on the reviews which is then feed to a simple Neural Network to predict the score.
  
  [Document Classification using CNN](https://github.com/ayush2696/Machine-Learning/blob/main/Deep%20Learning/NLP/Document%20Classification%20using%20CNN.ipynb) : In this notebook we use CNN(Convolution Neural Network) to predict label on text data. We create embedding on the text data pass it through a CNN network and then we predict the labels for the text. We even experiment with 1-D convolutions with charater embeddings.
  
  [Implementing Encoder - Decoder model with Attention](https://github.com/ayush2696/Machine-Learning/blob/main/Deep%20Learning/NLP/Implementing%20Encoder%20-%20Decoder%20model%20with%20Attention.ipynb) : In this notebook we custom implement simple Encoder-Decoder model as well as Encoder - Decoder model with attention. We even use different scoring functions namely dot, general and concat. We also generate the attention map to see how the input and output words are related.
  
  **Computer Vision**
  
   [Transfer Learning](https://github.com/ayush2696/Machine-Learning/blob/main/Deep%20Learning/General%20Implementations/Transfer%20Learning.ipynb) : In this notebook we implement transfer learning on image dataset where we classify image labels using VGG-16 network. We try to implement different models with different last blocks and see how our accuracy metric gets affected.
   
   [Dense Net on CIFAR](https://github.com/ayush2696/Machine-Learning/blob/main/Deep%20Learning/Computer%20Vision/Dense%20Net%20on%20CIFAR.ipynb) : In this assignment we apply Dense Net on CIFAR dataset. We create a ImageGenerator to generate and augment images and then we train Dense Net to predict the class labels.
**Tools**: Pandas,numpy,tensorflow,keras Seaborn, Matplotlib
  
**Data Analysis and Visualisation**

[Exploratory_data_Analysis_on_Haberman_Dataset](https://github.com/ayush2696/Machine-Learning/blob/main/EDA/Exploratory_data_Analysis_on_Haberman_Dataset.ipynb) : Analysis of Haberman's Dataset, we do univariante and bi-variante analysis to get insights.
**Tools**: Pandas, Seaborn and Matplotlib





If you liked what you saw, want to have a chat with me about the portfolio, work opportunities, or collaboration, shoot an email at mittal.aayush26@gmail.com.
