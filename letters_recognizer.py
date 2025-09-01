import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt 
import pickle
import os

# 1. Simple Neural Network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 26)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(26, 26)

    def forward(self, x):
        x= x.view(-1, 28*28)
        return self.fc2(self.relu(self.fc1(x)))
    
# 2. Training Function
def train_model(batch_size):
    print(f"Actually training with batch size: {batch_size}")
    transform = transforms.ToTensor()

    # #Download Emnist Letters (26 classes: a-z)
    train_set = datasets.EMNIST(root='./data',split='letters',  train=True, download=True, transform=transform)
    test_data = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

    #Dataloaders
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train_losses =[]
    test_accuracies = []
    times =[]
    epochs=10

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start_time = time.time()
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            
            images, labels= images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels-1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        end_time = time.time()
        train_losses.append(total_loss / len(train_loader))
        times.append(end_time - start_time)

        #Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total +=labels.size(0)
                correct +=(predicted ==labels).sum().item()
        test_accuracies.append(100* correct/ total)

    return train_losses, test_accuracies, times


results_file = "results.pkl"

if os.path.exists(results_file):
    with open(results_file,"rb") as f:
        results = pickle.load(f)
    print("Loaded Saved training results")
else:
    results = {}

# 3. Run Experiments
batch_sizes = [8, 64, 512]


for bs in batch_sizes:
    if bs not in results:
        print(f"Training with batch_size={bs}")
        losses, accs, times = train_model(bs)
        results[bs] = {
            "losses": losses, 
            "accuracies" : accs,
            "times" : times
        }
        with open("results.pkl","wb") as f:
            pickle.dump(results,f)
            print("Saved the Model")
    else:
        print(f"Already trained for batch size {bs}, skipping")
    


#4.Plotting Results
for bs in batch_sizes:
    plt.plot(results[bs]["losses"], label=f'Batch {bs}')
plt.title("Training Loss vs Epochs")
plt.xlabel("Loss")
plt.ylabel("Loss")
plt.legend()
plt.show()

for bs in batch_sizes:
    plt.plot(results[bs]["accuracies"], label=f'Batch {bs}')
plt.title("Validation Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

final_accuracies = [results[bs]["accuracies"][-1] for bs in batch_sizes]
plt.figure(figsize=(8,5))
plt.plot(batch_sizes, final_accuracies, marker='o', linestyle='-', color= 'teal')
plt.xlabel("Batch Sizes")
plt.ylabel("Validation Accuracy (%)")
plt.title("Batch Size vs Validation Accuracy")
plt.grid(True)
plt.show()


