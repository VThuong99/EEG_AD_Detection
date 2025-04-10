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
        self.num_gpus = torch.cuda.device_count()

        if self.num_gpus > 1:
            print(f"Using {self.num_gpus} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)

    def fit(self, X, y, calculate_epoch_loss=False): 
        self.model.to(self.device)
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                                   torch.tensor(y, dtype=torch.long))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        epoch_losses = [] if calculate_epoch_loss else None

        for epoch in range(self.epochs):
            batch_losses = [] 
            for batch_X, batch_y in dataloader:
                if batch_X.size(0) == 1: # Skip if batch_size=1, for conflict with BatchNorm1d()
                    continue
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                if calculate_epoch_loss: 
                    batch_losses.append(loss.item()) 

            if calculate_epoch_loss: 
                epoch_loss = np.mean(batch_losses) # Calculate average loss for epoch
                epoch_losses.append(epoch_loss) # Store epoch loss
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}") 

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

    def fit_with_validation(self, X, y, val_X=None, val_y=None, patience=None, verbose=1):
        """
        Model training with validation set and early stopping (if needed).
        
        Returns:
            self: Trained model.
            history: Dictionary contains training history (loss and accuracy).
        """

        self.model.to(self.device)
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        # Dataloader for training set
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                                              torch.tensor(y, dtype=torch.long))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Dataloader for validation set
        val_dataloader = None 
        if val_X is not None and val_y is not None:
            val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_X, dtype=torch.float32),
                                                        torch.tensor(val_y, dtype=torch.long))
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        best_val_loss = float('inf')
        patience_counter = 0

        # Training iteration
        for epoch in range(self.epochs):
            self.model.train()
            train_loss, train_correct, total_train = 0.0, 0, 0

            # Training on each batch
            for batch_X, batch_y in dataloader:
                if batch_X.size(0) == 1: # Skip batch_size=1 for BatchNorm error.
                    continue 
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0) 
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == batch_y).sum().item()
                total_train += batch_y.size(0)

            # Average loss, accuracy on training set
            train_loss /= total_train
            train_acc = train_correct / total_train
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Evaluate on validation set (if available)
            if val_dataloader:
                self.model.eval()
                val_loss, val_correct, total_val = 0.0, 0, 0

                with torch.no_grad():
                    for batch_X, batch_y in val_dataloader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_X.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == batch_y).sum().item()
                        total_val += batch_y.size(0)    

                val_loss /= total_val
                val_acc = val_correct / total_val
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # Early stopping (if available)
                if patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss 
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose >= 1:
                                print(f"Early stopping at epoch {epoch+1}")
                            break
            else:
                # If don't use validation, assign None
                history['val_loss'].append(None)
                history['val_acc'].append(None)
            
            # Print log information
            if verbose >= 1:
                log = f"Epoch {epoch+1}/{self.epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                if val_dataloader:
                    log += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                print(log)

        return self, history
