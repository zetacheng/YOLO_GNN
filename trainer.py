import torch
from tqdm import tqdm
from graph_builder import GraphBuilder

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.graph_builder = GraphBuilder()  # Ensure GraphBuilder is available

    def train_epoch(self, train_loader, presentation, epoch):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)

            # Convert image data to hierarchical graph
            graph = self.graph_builder.build_graph(data, feature_maps=data)

            self.optimizer.zero_grad()
            # Pass graph with edge weights to the model
            output = self.model(graph)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return train_loss / len(train_loader), 100. * correct / total

    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Convert image data to hierarchical graph
                graph = self.graph_builder.build_graph(data, feature_maps=data)

                # Pass graph with edge weights to the model
                output = self.model(graph)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return test_loss / len(test_loader), 100. * correct / total
