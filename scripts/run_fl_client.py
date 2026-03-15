import sys 
hospital_id = sys.argv[1]
data_path = f"data/federated/{hospital_id}"
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import flwr as fl
import torch
from src.model import get_model
from src.data_loader import get_data_loaders


class TBClient(fl.client.NumPyClient):

    def __init__(self, data_path):
        self.model = get_model()
        self.train_loader, self.val_loader, _ = get_data_loaders(data_path)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        for images, labels in self.train_loader:
            
            optimizer.zero_grad()
            
            outputs = self.model(images)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()
        
        return self.get_parameters(config), len(self.train_loader.dataset), {"loss": loss.item()}    
            
    

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        correct = 0
        total = 0
        loss_total = 0
        
        criterion = torch.nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, Labels in self.val_loader:
                outputs = self.model(images)
                Loss = criterion(outputs, labels)
                Loss_total += Loss.item()
                
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = correct / total
        avg_loss = loss_total / len(self.val_loader)        
        return loss, len(self.val_loader.dataset), {"accuracy": acc}


def main():
    hospital_id = sys.argv[1]
    data_path = f"data/federated/{hospital_id}"
    client = TBClient(data_path)
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=client,
    )


if __name__ == "__main__":
    main()