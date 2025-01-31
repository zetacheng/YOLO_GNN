import torch
from para_manager import Params
from yolo_module import YOLO
from gnn_module import EnhancedHierarchicalGNN
from data import CIFAR10DataLoader
from presentation import Presentation
from train import Train
from overfit_detector import OverfitDetector

if __name__ == "__main__":
    params = Params()
    yolo_model = YOLO(input_size=params.yolo_input_size, num_classes=params.yolo_num_classes)
    gnn_model = EnhancedHierarchicalGNN(input_dim=64, hidden_dim=128, output_dim=64, num_classes=params.yolo_num_classes)
    data_loader = CIFAR10DataLoader(batch_size=params.batch_size)
    train_loader, test_loader = data_loader.get_loaders()
    presentation = Presentation()
    presentation.display_meta_parameters(params)
    overfit_detector = OverfitDetector(accuracy_threshold=10, loss_threshold=0.1)
    trainer = Train(yolo_model, gnn_model, train_loader, test_loader, params, presentation, overfit_detector)
    trainer.run()
    presentation.finalize_results()
