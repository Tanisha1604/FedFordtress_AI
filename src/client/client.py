import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple
from src.attacks.malicious import (
    noise_injection,
    weight_scaling,
    random_weights,
    label_flipping
)

class Client:
    def __init__(self,
             client_id,
             model,
             dataloader,
             malicious=False,
             attack_type="noise",
             enable_privacy=False):

             self.client_id = client_id
             self.model = model
             self.dataloader = dataloader
             self.malicious = malicious
             self.attack_type = attack_type
             self.enable_privacy = enable_privacy

             self.criterion = torch.nn.CrossEntropyLoss()
             self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
             self.attack_type = attack_type
             
             # Server connection
             self.server = None
             self.current_version = 0
    
    def connect_to_server(self, server):
        """Connect this client to a federated learning server."""
        self.server = server
    
    def receive_global_model(self, global_state):
        self.model.load_state_dict(global_state)

    def receive_global_model_from_server(self) -> bool:
        """
        Receive global model from connected server.
        Returns True if successful, False otherwise.
        """
        if self.server is None:
            return False
        
        global_model, version = self.server.get_global_model()
        self.current_version = version
        
        # Convert numpy arrays back to PyTorch tensors
        torch_state = {}
        for key, value in global_model.items():
            torch_state[key] = torch.from_numpy(value).float()
        
        self.model.load_state_dict(torch_state)
        return True

    def submit_update_to_server(self, update_dict: Dict[str, np.ndarray], 
                                num_samples: int, loss: float) -> bool:
        """
        Submit model update to the connected server.
        Returns True if successful, False otherwise.
        """
        if self.server is None:
            return False
        
        from src.server.aggregation import ClientUpdate
        import time
        
        update = ClientUpdate(
            client_id=f"client_{self.client_id}",
            update=update_dict,
            num_samples=num_samples,
            version=self.current_version,
            timestamp=time.time(),
            loss=loss,
            reputation=1.0
        )
        
        self.server.submit_update(update)
        return True

    def local_train(self, epochs=1):

        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for _ in range(epochs):

            for data, target in self.dataloader:
                if self.malicious and self.attack_type == "label_flip":
                     target = label_flipping(target)

                self.optimizer.zero_grad()
                output = self.model(data)

                loss = self.criterion(output, target)
                loss.backward()

                # Privacy hook later here

                self.optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct / total

        return {
            "state_dict": self.model.state_dict(),
            "num_samples": total,
            "local_accuracy": accuracy,
            "loss": total_loss
        }
