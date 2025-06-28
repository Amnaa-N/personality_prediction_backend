import torch
import torch.nn as nn
import joblib
import os

# Define the model architecture (same as used during training)
class PersonalityModel(nn.Module):
    def __init__(self, input_dim):
        super(PersonalityModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.out(x)

def load_trained_model(path="personality_model.pt", input_dim=9):
    model = PersonalityModel(input_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def load_scaler(path="scaler.pkl"):
    import joblib
    return joblib.load(path)