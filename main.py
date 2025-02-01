import torch
from para_manager import Params
from yolo_gnn_model import YOLO_GNN
from data import CIFAR10DataLoader
from presentation import Presentation
from train import Train
from overfit_detector import OverfitDetector

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
    data_loader = CIFAR10DataLoader(batch_size=params.batch_size)
    train_loader, test_loader = data_loader.get_loaders()
    presentation = Presentation()
    presentation.display_meta_parameters(params)
    overfit_detector = OverfitDetector(accuracy_threshold=10, loss_threshold=0.1)
    trainer = Train(model, train_loader, test_loader, params, presentation, overfit_detector)
    trainer.run()
    presentation.finalize_results()
