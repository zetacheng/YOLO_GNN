import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, aux_loss_weight=0.3):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.aux_loss_weight = aux_loss_weight

    def train_epoch(self, train_loader, presentation, epoch):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        for data, target in progress_bar:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            final_output, aux_logits = self.model(data)

            # Primary loss (GNN output) + auxiliary loss (YOLO rough classifier)
            # Training aux_logits ensures top-k selection becomes meaningful over time
            loss = self.criterion(final_output, target) + \
                   self.aux_loss_weight * self.criterion(aux_logits, target)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_loss += loss.item()
            _, predicted = final_output.max(1)
            total   += target.size(0)
            correct += predicted.eq(target).sum().item()

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc':  f'{100. * correct / total:.2f}%',
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
                final_output, aux_logits = self.model(data)
                loss = self.criterion(final_output, target) + \
                       self.aux_loss_weight * self.criterion(aux_logits, target)
                test_loss += loss.item()
                _, predicted = final_output.max(1)
                total   += target.size(0)
                correct += predicted.eq(target).sum().item()

        return test_loss / len(test_loader), 100. * correct / total
