import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Trainer:
    def __init__(self, model, lr=0.01, checkpoint_path="model_checkpoint.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.checkpoint_path = checkpoint_path
        self.start_epoch = 0
        self.load_checkpoint()

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, self.checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}!")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint.get('epoch', 0) + 1
                print(f"Checkpoint loaded! Resuming from epoch {self.start_epoch}.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")

    def clear_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            print("Checkpoint cleared. Training will start from scratch.")

    def train(self, train_loader, epochs=5):
        self.model.train()
        for epoch in range(self.start_epoch, self.start_epoch + epochs):
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()
                
                if i % 10 == 0:
                    print(f'Epoch {epoch + 1}/{self.start_epoch + epochs}, Step {i}, Loss: {loss.item():.4f}')

            print(f'Epoch {epoch + 1}/{self.start_epoch + epochs}, Average Loss: {running_loss / len(train_loader):.4f}')
            
            first_param = next(self.model.parameters())
            print(f'First layer weight mean after epoch {epoch + 1}: {first_param.data.mean().item():.6f}')
            
            self.save_checkpoint(epoch)

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        return accuracy

    def get_weights(self):
        return [param.data.clone() for param in self.model.parameters()]

    def set_weights(self, new_weights):
        for param, new_weight in zip(self.model.parameters(), new_weights):
            param.data = new_weight.clone()
