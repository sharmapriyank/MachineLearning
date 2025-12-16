#Simple Sentiment Analysis Only by Linear NeuralNet

import torch
import torch.nn as nn
import torch.optim as optim

# Dummy data (replace with your actual data)
reviews = ["I love this movie!", "This movie is terrible."]
labels = [1, 0]  # 1 for positive, 0 for negative

# Tokenize and pad reviews (you can use torchtext for more complex preprocessing)
vocab = set(" ".join(reviews).split())
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

padded_reviews = [[word_to_idx[word] for word in review.split()] for review in reviews]

# Create a simple linear model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size):
        super(SentimentModel, self).__init__()
        # Adjust the input size to match the shape of review_tensor
        self.linear = nn.Linear(len(vocab), 2)  # 2 output classes (pos/neg)

    def forward(self, x):
        return self.linear(x)

model = SentimentModel(vocab_size=len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    for review, label in zip(padded_reviews, labels):
        #review_tensor = torch.tensor(review, dtype=torch.float)
        review_tensor = torch.tensor(review, dtype=torch.float).unsqueeze(0)
        optimizer.zero_grad()
        output = model(review_tensor)
        loss = criterion(output.unsqueeze(0), torch.tensor([label]))
        loss.backward()
        optimizer.step()

# Inference
test_review = "This movie is amazing!"
test_review_tensor = torch.tensor([word_to_idx[word] for word in test_review.split()], dtype=torch.float)
with torch.no_grad():
    predicted_scores = model(test_review_tensor)
    predicted_class = torch.argmax(predicted_scores).item()

sentiment = "positive" if predicted_class == 1 else "negative"
print(f"Predicted sentiment for '{test_review}': {sentiment}")
