# Technical documentation

## Notebook work phases

### Data collection

The training data used for the project is a subset of the Reuters 1997 text categorization [dataset](https://www.cs.helsinki.fi/u/yangarbe/Courses/2020-deep-learning/text-training-corpus/).

The original compressed data consists of several XML files corresponding to articles that appeared on the Reuters news service. These XML files contain the ids, dates, languages, titles, topic codes and text bodies of said articles. Our interest were in selecting the text bodies and title codes from each article. The title codes are shorthand codes such as 'ECAT' for 'ECONOMICS' that denote the category/topic area of the article. Each article may have several topic codes.



After parsing the XML files using the XML data extraction features of BeautifulSoup the package, the data is saved in a pandas dataFrame for easier manipulation in Python. Some topic codes are discarded, as a predefined list of topic codes is used to decice which codes are saved alongside the bodytext into the dataFrame.



#### Data pre-processing

Prior to using the data, we still needed to clean the text bodies. All non alphabet symbols were removed and all of the text was converted to a lowercase representation. Using the NLTK 'wordnet' stopwords package for English, we also removed stopwords from the text in order to improve the accuracy of training our model. The WordNetLemmatizer from the NLTK package was also used to lemmatize the words, preventing the same word from appearing in multiple conjugations.



#### Encoding and Tokenizing Data

Using the MultilabelBinarizer and tokenizer from Keras, the cleaned binarized data was fit to a complete list of all the possible topic codes. We padded the encoded text sequences to be 1000 tokens long, while only taking a slice of the 1000 first tokens from those text sequences that would exceed this length. The encoders are also saved as pickles so there is no need to encode the data again if there are no changes.





#### LSTM model Neural Network

A LSTM recurrent neural network architecture was chosen due to its good suitability to classify and make predictions. ADAM (an adaptive learning rate optimization algorithm), was used to optimize the neural network instead of normal gradient descent that would have a fixed learning rate during training.The Sigmoid activation function was used for the output neuron as the project was a multilabel classification and each probability distribution of the neurons are independent of each other. As the loss function, mean squared error was found to work well.



#### Training the model

The model was trained using the cuda GPU option, speeding up the training process. A batch size of 32 was chosen empirically. The model was ran for 3 epochs as the result looked satisfactory.



#### Validation

We set a threshold of 0.5 to decide which values in the tensors in our output were likely to display certain topic codes.

F1-score is used to determine the accuracy of the prediction.

#### Results

Some discussion about our results for training and for the final test data.

[Training Results 25.5.](./training_results.txt)


# Additional research / experimentation

## Trying out different models




