# In overfit_detector.py
class OverfitDetector:
    def __init__(self, accuracy_threshold=15, loss_threshold=0.25, patience=5):
        self.accuracy_threshold = accuracy_threshold
        self.loss_threshold = loss_threshold
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        
    def check(self, train_loss, test_loss, train_acc, test_acc):
        acc_diff = train_acc - test_acc
        loss_diff = test_loss - train_loss
        
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            return {"is_overfit": True, "reason": f"no improvement for {self.patience} epochs"}
        elif acc_diff > self.accuracy_threshold:
            return {"is_overfit": True, "reason": f"accuracy diff >{self.accuracy_threshold}%"}
        elif loss_diff > self.loss_threshold:
            return {"is_overfit": True, "reason": f"loss diff >{self.loss_threshold}"}
        else:
            return {"is_overfit": False, "reason": ""}

