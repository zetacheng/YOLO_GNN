import torch
from meta_manager import Metas
from yolo_gnn_model import YOLO_GNN
from data import CIFAR10DataLoader
from presentation import Presentation
from train import Train
from overfit_detector import OverfitDetector

if __name__ == "__main__":
    Metas = Metas()
    model = YOLO_GNN(
        input_size=Metas.yolo_input_size,
        num_classes=Metas.yolo_num_classes,
        feature_dim=Metas.feature_dim,
        gnn_hidden_dim=Metas.gnn_hidden_dim,
        gnn_output_dim=Metas.gnn_output_dim,
        top_k=Metas.top_k,
        knn_neighbors=Metas.knn_neighbors
    )
    data_loader = CIFAR10DataLoader(batch_size=Metas.batch_size)
    train_loader, test_loader = data_loader.get_loaders()
    presentation = Presentation()
    presentation.display_meta_parameters(Metas)
    overfit_detector = OverfitDetector(accuracy_threshold=10, loss_threshold=0.1)
    trainer = Train(model, train_loader, test_loader, Metas, presentation, overfit_detector)
    trainer.run()
    presentation.finalize_results()
