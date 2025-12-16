

    Data Preparation:
        We start by defining some dummy data: reviews (a list of movie review strings) and labels (corresponding sentiment labels, where 1 represents positive and 0 represents negative sentiment).
        We tokenize the reviews (split them into words) and create a vocabulary (unique words).



    Creating a Vocabulary Set (vocab):
        " ".join(reviews).split() concatenates all the strings in the reviews list, separated by spaces.
        set(...) converts this concatenated string into a set, which automatically removes duplicate words.
        The resulting vocab set contains all unique words from the reviews.

    Creating a Word-to-Index Mapping (word_to_idx):
        {word: idx for idx, word in enumerate(vocab)} constructs a dictionary where each word in vocab is a key, and its corresponding index (position) is the value.
        The enumerate(vocab) function provides both the index and the word during iteration.

    Creating Padded Reviews (padded_reviews):
        [[word_to_idx[word] for word in review.split()] for review in reviews] creates a list of lists.
        For each review in the reviews list:
            review.split() splits the review into individual words.
            word_to_idx[word] maps each word to its index using the word_to_idx dictionary.
            The resulting list of indices represents the encoded version of the original review.
        The outer list contains all the encoded reviews.



    Model Architecture:
        We define a simple linear model using nn.Linear.
        The input size of the linear layer is the vocabulary size (number of unique words), and the output size is 2 (for binary classification: positive/negative sentiment).



    Defining the SentimentModel class:
        class SentimentModel(nn.Module): creates a custom PyTorch module for sentiment analysis.
        It inherits from nn.Module, which provides essential functionality for neural network models.
        The constructor (__init__) initializes the model, taking vocab_size as an argument.

    Initializing the Linear Layer (self.linear):
        self.linear = nn.Linear(vocab_size, 2) creates a linear layer with input size vocab_size and output size 2.
        In sentiment analysis, this layer maps input features (word embeddings) to two output classes (positive and negative sentiment).

    Defining the Forward Pass (forward method):
        The forward(self, x) method specifies how input data flows through the model.
        In this case, it simply returns the result of applying the linear layer to the input x.

    Creating an Instance of the Model (model):
        model = SentimentModel(vocab_size=len(vocab)) instantiates the SentimentModel with the specified vocabulary size.
        This model will be used for sentiment classification.

    Setting Up Loss Function (criterion) and Optimizer (optimizer):
        criterion = nn.CrossEntropyLoss() defines the loss function.
        Cross-entropy loss is commonly used for multi-class classification tasks.
        optimizer = optim.SGD(model.parameters(), lr=0.01) sets up stochastic gradient descent (SGD) optimization with a learning rate of 0.01.




    Optimizer Initialization (optimizer):
        The optimizer is a crucial component in training neural networks. It adjusts the model’s parameters during training to minimize the loss function.
        In this case, we’re using the stochastic gradient descent (SGD) optimizer.
        The SGD optimizer updates the model’s parameters based on the gradients of the loss function with respect to those parameters.

    model.parameters() and Parameter Update:
        model.parameters() returns an iterable containing all the learnable parameters of the model.
        These parameters include weights and biases associated with each layer (e.g., weights in linear layers, convolutional filters, etc.).
        By passing model.parameters() to SGD, we specify that the optimizer should update these parameters during training.

    Learning Rate (lr=0.01):
        Learning rates typically fall in the range between 0.0 and 1.0.
        The lr argument stands for the learning rate.
        It determines the step size taken by the optimizer during parameter updates.
        A higher learning rate means larger steps, which can lead to faster convergence but may overshoot the optimal solution.
        A lower learning rate ensures more stable updates but might slow down convergence.



    Training Loop:
        We set up the training process:
            Initialize an optimizer (Stochastic Gradient Descent) and a loss function (cross-entropy).
            Iterate over the reviews and labels.
            Convert each review to a tensor (numerical representation).
            Zero out the gradients.
            Compute the model’s output (logits) for the review.
            Calculate the loss.
            Backpropagate the gradients and update the model’s weights.

    Inference:
        We test the model on a new review (“This movie is amazing!”):
            Convert the review to a tensor.
            Use torch.no_grad() to avoid gradient computation during inference.
            Get the predicted scores (logits) for each class.
            Choose the class with the highest score as the predicted sentiment.

    Print Result:
        Finally, we print the predicted sentiment (positive or negative) for the test review.
