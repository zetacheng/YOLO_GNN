import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import logging
from yolo_gnn_model import YOLO_GNN
from para_manager import Params

class Train:
    def __init__(self, model, train_loader, test_loader, params, presentation, overfit_detector):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.params = params
        self.presentation = presentation
        self.overfit_detector = overfit_detector
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=self.params.learning_rate, 
                                    weight_decay=0.01)
        
        # OneCycleLR scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.params.learning_rate,
            epochs=self.params.num_epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,  # Warm-up period
            div_factor=25,  # Initial learning rate = max_lr/25
            final_div_factor=1e4  # Final learning rate = max_lr/10000
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(self.train_loader, desc=f'Training')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
        
        return train_loss / len(self.train_loader), 100. * correct / total

    def evaluate(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return test_loss / len(self.test_loader), 100. * correct / total

    def run(self):
        best_accuracy = 0
        for epoch in range(self.params.num_epochs):
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()
            
            self.scheduler.step()
            
            overfit_tag = self.overfit_detector.tag(train_loss, test_loss, train_acc, test_acc)
            
            self.presentation.display_epoch_results(
                epoch, train_loss, train_acc, test_loss, test_acc,
                epoch_time=0,  # You might want to implement time tracking
                learning_rate=self.scheduler.get_last_lr()[0],
                overfit_tag=overfit_tag
            )
            
            logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, '
                         f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, {overfit_tag}')
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            if self.overfit_detector.check(train_loss, test_loss, train_acc, test_acc)["is_overfit"]:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        print('Training finished.')
        logging.info('Training finished.')

# This allows the script to be run directly or imported as a module
if __name__ == "__main__":
    params = Params()
    model = YOLO_GNN(
        input_size=params.yolo_input_size,
        num_classes=params.yolo_num_classes,
        feature_dim=params.feature_dim,
        gnn_hidden_dim=params.gnn_hidden_dim,
        gnn_output_dim=params.gnn_output_dim,
        top_k=params.top_k,
        knn_neighbors=params.knn_neighbors
    )
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
    
    from presentation import Presentation
    from overfit_detector import OverfitDetector
    
    presentation = Presentation()
    overfit_detector = OverfitDetector()
    
    trainer = Train(model, train_loader, test_loader, params, presentation, overfit_detector)
    trainer.run()
