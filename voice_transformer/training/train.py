import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.voice_model import VoiceTransformerNet
from utils.audio_utils import VoiceDataset

def train(model, criterion, optimizer, train_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')

def main():
    # Load the voice data
    train_dataset = VoiceDataset('data/train')
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    # Initialize the model
    model = VoiceTransformerNet()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    train(model, criterion, optimizer, train_loader)

    # Save the trained model
    torch.save(model.state_dict(), 'voice_transformer_model.pth')

if __name__ == "__main__":
    main()
