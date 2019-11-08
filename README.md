# SPOKEN LANGUAGE IDENTIFICATION

Spoken language is one of the distinctive characteristics of the human race. Spoken language processing is a branch of computer science that plays an important role in human–computer interaction (HCI), which has made remarkable advancement in the last two decades.
The goal of spoken language identification is to assign language labels to short (usually 10 second) as well as long (usually 45 seconds) audio files containing utterances in one of the languages from a predefined set.

  
## CONVOLUTIONAL RECURRENT NEURAL NETWORK ARCHITECTURE 

![alt text](https://miro.medium.com/max/2358/1*etN2RhEkMJrEtJgWLvD9pQ.png)

 



Convolutional recurrent neural network (CRNN) architecture. The input features are matrix of consecutive frames of log-Mel filter banks (128 filter banks by 4500 time frames). The convolutions and max-poling operations are sequentially applied to extract beneficial features. Then these are fed into the gated recurrent unit (GRU) to capture the temporal information. The network outputs are sigmoid scores, these indicate several active acoustic events in audio signal. 

## METHOD AND ARCHITECTURE OF THE NEURAL NETWORK
 ![alt text](https://www.researchgate.net/profile/Kaveh_Taghipour/publication/305748202/figure/fig1/AS:410488992223237@1474879610604/The-convolutional-recurrent-neural-network-architecture.png)
 
The spectrograms were encoded as images of variable width and a height of 128 pixels (height representing the number of frequency bins).

The spectrograms were provided as an input to a sequential module containing a 2D convolutional layer with 16 filters of 3 x 3 pixels (strides of 1 and ReLU activation functions were used in all convolutional layers), followed by a dropout layer with a 0.2 dropout rate and a max pooling layer of size 2. After the pooling layer, layer normalisation was performed.


This was followed by three more similar convolutional layers with dropout, max pooling and layer normalisation. The parameters of these further convolutional layers are as follows: a layer of 32 5 x 5 filters, and two layers of 32 3 x 3 filters each. The pooling and dropout parameters were identical to the first convolutional layer.

The output of the last layer-normalised max pooling layer served as the input to a GRU cell with 128 hidden nodes followed by layer normalisation and dropout (rate of 0.2). Finally, this served as the input to a fully-connected layer with as many output nodes as the number of languages being considered and a softmax activation function.


A cross-entropy loss function with regularisation (lambda of 0.001) was optimised using an Adam Optimiser. The learning rate was set to 0.0001 for 15 epochs.

## Conclusion 




The main model achieved an accuracy level of 95.2%, identifying one of three languages namely English, German and French from a 10% validation set containing only recordings of speakers not heard during training. This data set contained 500 samples 10 seconds long, so the accuracy of this test set represents an average for test samples of different durations.
The accuracy of the model was measured for each fixed duration sample set. The results are provided in the table below.
Duration	10 seconds
Accuracy	95.05%
 
The project was able to demonstrate a model of spoken language identification from short spoken utterances for 3 languages.

