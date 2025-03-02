import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin

class DeepLearningModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model, lr=0.001, epochs=10, batch_size=32):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        self.model.to(self.device)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # x shape (n_samples, num_channels, n_samples_per_epoch)
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                                   torch.tensor(y, dtype=torch.long))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            # if x shape (n_samples, num_channels, n_samples_per_epoch) thì không cần thêm dimension
            if X.ndim == 3:
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
            outputs = self.model(X_tensor)
            _, predictions = torch.max(outputs, 1)
        return predictions.cpu().numpy()

    def predict_proba(self, X):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            if X.ndim == 3:
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            else:
                X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()
