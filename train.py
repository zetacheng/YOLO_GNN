import torch
import torch.nn as nn
import torch.optim as optim
import logging
from trainer import Trainer
from yolo_gnn_model import YOLO_GNN
from para_manager import Params
from presentation import Presentation
from overfit_detector import OverfitDetector

class Train:
    def __init__(self, model, train_loader, test_loader, params, presentation, overfit_detector):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.params = params
        self.presentation = presentation
        self.overfit_detector = overfit_detector
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params.learning_rate,
            epochs=params.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4
        )
        
        self.trainer = Trainer(model, criterion, optimizer, scheduler, self.device)
        
        logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    def run(self):
        self.presentation.start_profiler()
        best_accuracy = 0

        for epoch in range(self.params.num_epochs):
            print(f"Epoch {epoch+1}/{self.params.num_epochs}")
            train_loss, train_acc = self.trainer.train_epoch(self.train_loader, self.presentation, epoch)
            test_loss, test_acc = self.trainer.evaluate(self.test_loader)
            
            overfit_tag = self.overfit_detector.tag(train_loss, test_loss, train_acc, test_acc)
            
            self.presentation.display_epoch_results(
                epoch, train_loss, train_acc, test_loss, test_acc,
                epoch_time=0,  # You might want to implement time tracking
                learning_rate=self.trainer.scheduler.get_last_lr()[0],
                overfit_tag=overfit_tag
            )
            
            logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, '
                         f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, {overfit_tag}')
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(self.trainer.model.state_dict(), 'best_model.pth')
            
            if self.overfit_detector.check(train_loss, test_loss, train_acc, test_acc)["is_overfit"]:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

        self.presentation.end_profiler()
        
        print('Training finished.')
        logging.info('Training finished.')

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

    from data import CIFAR10DataLoader
    data_loader = CIFAR10DataLoader(batch_size=params.batch_size)
    train_loader, test_loader = data_loader.get_loaders()

    presentation = Presentation()
    overfit_detector = OverfitDetector()

    trainer = Train(model, train_loader, test_loader, params, presentation, overfit_detector)
    trainer.run()
    presentation.finalize_results()
