import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class DeepLearningModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model, lr=0.001, epochs=10, batch_size=32):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y, calculate_epoch_loss=False): 
        self.model.to(self.device)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                                   torch.tensor(y, dtype=torch.long))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        epoch_losses = None # Initialize epoch_losses to None
        if calculate_epoch_loss: # Conditionally calculate epoch loss
            epoch_losses = [] # List to store average loss for each epoch

        for epoch in range(self.epochs):
            batch_losses = [] # List to store batch losses for current epoch
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                if calculate_epoch_loss: 
                    batch_losses.append(loss.item()) 

            if calculate_epoch_loss: # Only calculate and print epoch loss if calculate_epoch_loss is True
                epoch_loss = np.mean(batch_losses) # Calculate average loss for epoch
                epoch_losses.append(epoch_loss) # Store epoch loss
                print(f"Epoch {epoch+1}/{self.epochs}, Training Loss: {epoch_loss:.4f}") # Print epoch loss

        return self, epoch_losses # Return self and epoch_losses (can be None)

    def predict(self, X):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            else:
                X_tensor = X.to(self.device)
    
            # If the input is already 3D (batch, num_channels, num_samples), use as-is.
            if X_tensor.dim() < 3:
                X_tensor = X_tensor.unsqueeze(0)
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