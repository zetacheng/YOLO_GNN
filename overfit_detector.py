class OverfitDetector:
    def __init__(self, accuracy_threshold=10, loss_threshold=0.1):
        self.accuracy_threshold = accuracy_threshold
        self.loss_threshold = loss_threshold

    def check(self, train_loss, test_loss, train_acc, test_acc):
        acc_diff = train_acc - test_acc
        loss_diff = test_loss - train_loss
        if acc_diff > self.accuracy_threshold:
            return {"is_overfit": True, "reason": f"accuracy diff >{self.accuracy_threshold}%"}
        elif loss_diff > self.loss_threshold:
            return {"is_overfit": True, "reason": f"loss diff >{self.loss_threshold}"}
        else:
            return {"is_overfit": False, "reason": ""}

    def tag(self, train_loss, test_loss, train_acc, test_acc):
        result = self.check(train_loss, test_loss, train_acc, test_acc)
        if result["is_overfit"]:
            return f"[overfit:{result['reason']}]"
        return ""
