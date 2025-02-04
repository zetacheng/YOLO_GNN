import torch
import torch.nn as nn
import torch.optim as optim
import logging
from trainer import Trainer
import time

class Train:
    def __init__(self, model, train_loader, test_loader, metas, presentation, overfit_detector):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.metas = metas
        self.presentation = presentation
        self.overfit_detector = overfit_detector
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=metas.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=metas.learning_rate,
            epochs=metas.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4
        )
        
        self.trainer = Trainer(model, criterion, optimizer, scheduler, self.device)
        self.early_stop_patience = metas.early_stopping_patience # Number of epochs to wait before early stopping
        self.early_stop_counter = 0  # Tracks how many consecutive overfit detections occur
        
        logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

    def run(self):
        self.presentation.start_profiler()
        best_accuracy = 0

        for epoch in range(self.metas.num_epochs):
            start_time = time.time()  # Start timing
            print(f"Epoch {epoch+1}/{self.metas.num_epochs}")
            train_loss, train_acc = self.trainer.train_epoch(self.train_loader, self.presentation, epoch)
            test_loss, test_acc = self.trainer.evaluate(self.test_loader)
            
            overfit_tag = self.overfit_detector.tag(train_loss, test_loss, train_acc, test_acc)
            end_time = time.time()  # End timing
            epoch_time = end_time - start_time
          
            self.presentation.display_epoch_results(
                epoch, train_loss, train_acc, test_loss, test_acc,
                epoch_time,  #time tracking
                learning_rate=self.trainer.scheduler.get_last_lr()[0],
                overfit_tag=overfit_tag
            )
            
            logging.info(f'Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Accuracy: {train_acc:.2f}% | Test Accuracy: {test_acc:.2f}% | {overfit_tag}')
            
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(self.trainer.model.state_dict(), 'best_model.pth')
                self.early_stop_counter = 0  # Reset counter if there's improvement
            
            if self.overfit_detector.check(train_loss, test_loss, train_acc, test_acc)["is_overfit"]:
                self.early_stop_counter += 1
                print(f'Overfitting detected ({self.early_stop_counter}/{self.early_stop_patience})')
                if self.early_stop_counter >= self.early_stop_patience:    
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            else:
                self.early_stop_counter = 0 # Reset counter if no overfitting detected

        self.presentation.end_profiler()
        
        print('Training finished.')
        logging.info('Training finished.')
