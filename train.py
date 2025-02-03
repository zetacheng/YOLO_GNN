import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from trainer import Trainer
from yolo_gnn_model import YOLO_GNN
from meta_manager import Metas
from presentation import Presentation
from overfit_detector import OverfitDetector
from graph_builder import GraphBuilder

class Train:
    def __init__(self, model, train_loader, test_loader, Metas, presentation, overfit_detector):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.Metas = Metas
        self.presentation = presentation
        self.overfit_detector = overfit_detector
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=Metas.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Metas.learning_rate,
            epochs=Metas.num_epochs,
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

        for epoch in range(self.Metas.num_epochs):
            print(f"Epoch {epoch+1}/{self.Metas.num_epochs}")
            start_time = time.time()
            
            train_loss, train_acc = self.trainer.train_epoch(self.train_loader, self.presentation, epoch)
            test_loss, test_acc = self.trainer.evaluate(self.test_loader)
            
            epoch_time = time.time() - start_time
            
            overfit_tag = self.overfit_detector.tag(train_loss, test_loss, train_acc, test_acc)
            
            self.presentation.display_epoch_results(
                epoch, train_loss, train_acc, test_loss, test_acc,
                epoch_time=epoch_time,
                learning_rate=self.trainer.scheduler.get_last_lr()[0],
                overfit_tag=overfit_tag
            )
            
            logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, '
                         f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Time: {epoch_time:.2f}s, {overfit_tag}')
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(self.trainer.model.state_dict(), 'best_model.pth')
            
            if self.overfit_detector.check(train_loss, test_loss, train_acc, test_acc)["is_overfit"]:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

        self.presentation.end_profiler()
        
        print('Training finished.')
        logging.info('Training finished.')
