import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    """Simple feedforward neural network with one hidden layer."""
    def __init__(self, layer_1, layer_2=10):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, layer_1)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        self.fc2 = nn.Linear(layer_1, layer_2)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation before final output
        return x

class Trainer:
    """Trainer class to handle training and weight management."""
    def __init__(self, model, lr=0.01):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device)
        torch.backends.cudnn.benchmark = True  # Snabbare GPU-träning
        for param in self.model.parameters():
            param.requires_grad = True
        # print('All model parameters require grad:', all(p.requires_grad for p in self.model.parameters()))
        for param in self.model.parameters():
            param.requires_grad = True
        self.criterion = nn.CrossEntropyLoss()
        # Optimeraren skapas EFTER att modellen har flyttats till rätt enhet
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)  # LR decay  # Snabbare konvergens
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    def train(self, train_loader, epochs=2):
        self.model.train()  # Se till att modellen är i träningsläge
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.train()
        for epoch in range(epochs):
            
            for i, (images, labels) in enumerate(train_loader):
                if i >= 5: break  # Begränsar antalet klientuppdateringar per runda
                images, labels = images.to(device).float(), labels.to(device).long()  # Flytta till GPU
                self.optimizer.zero_grad()
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    outputs = self.model(images)
                
                    loss = self.criterion(outputs, labels)
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step()  # Uppdatera learning rate
                # if i % 10 == 0: print('Gradientnorm:', torch.norm(torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])))
    
                outputs = self.model(images)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()  # Uppdatera learning rate
                # if i % 10 == 0: print('Gradientnorm:', torch.norm(torch.cat([p.grad.view(-1) for p in self.model.parameters() if p.grad is not None])))
    
    def get_weights(self):
        return [param.data.clone() for param in self.model.parameters()]
    
    def set_weights(self, new_weights):
        with torch.no_grad():
            for param, new_weight in zip(self.model.parameters(), new_weights):
                param.data.copy_(new_weight)
