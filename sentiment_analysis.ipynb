# Neural Network using only Embedding and Linear Layer to Gauge Sentiment
import torch
import torch.nn as nn
import torch.optim as optim

# Dummy data (replace with your actual data)
reviews = ["I love this movie!", "This movie is terrible."]
labels = [1, 0]  # 1 for positive, 0 for negative

# Tokenize and pad reviews (you can use torchtext for more complex preprocessing)
vocab = set(" ".join(reviews).split())
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
word_to_idx['<UNK>'] = len(vocab)  # Add a special token for OOV words

padded_reviews = [[word_to_idx[word] if word in word_to_idx else word_to_idx['<UNK>'] for word in review.split()] for review in reviews]


class SentimentModel(nn.Module):
    def __init__(self, vocab_size, max_length):
        super(SentimentModel, self).__init__()
        # Adjust the input size to match the shape of review_tensor
        self.embedding = nn.Embedding(vocab_size + 1, 8) # +1 for the <UNK> Unknown token
        self.linear = nn.Linear(max_length * 8, 2)  # 2 output classes (pos/neg)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

model = SentimentModel(vocab_size=len(vocab), max_length=max(len(review.split()) for review in reviews))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    for review, label in zip(padded_reviews, labels):
        review_tensor = torch.tensor(review, dtype=torch.long).unsqueeze(0)
        optimizer.zero_grad()
        output = model(review_tensor)
        loss = criterion(output, torch.tensor([label]))
        loss.backward()
        optimizer.step()

# Inference
test_review = "This movie is amazing!"
test_review_tensor = torch.tensor([word_to_idx[word] if word in word_to_idx else word_to_idx['<UNK>'] for word in test_review.split()], dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    predicted_scores = model(test_review_tensor)
    predicted_class = torch.argmax(predicted_scores).item()

sentiment = "positive" if predicted_class == 1 else "negative"
print(f"Predicted sentiment for '{test_review}': {sentiment}")
