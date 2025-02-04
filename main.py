import torch
from meta_manager import Meta
from yolo_gnn_model import YOLO_GNN
from data import CIFAR10DataLoader
from presentation import Presentation
from train import Train
from overfit_detector import OverfitDetector

if __name__ == "__main__":
    metas = Meta()
    model = YOLO_GNN(
        input_size=metas.yolo_input_size,
        num_classes=metas.yolo_num_classes,
        feature_dim=metas.feature_dim,
        gnn_hidden_dim=metas.gnn_hidden_dim,
        gnn_output_dim=metas.gnn_output_dim,
        top_k=metas.top_k,
        knn_neighbors=metas.knn_neighbors
    )
    data_loader = CIFAR10DataLoader(batch_size=metas.batch_size)
    train_loader, test_loader = data_loader.get_loaders()
    presentation = Presentation()
    presentation.display_meta_parameters(metas)
    overfit_detector = OverfitDetector(accuracy_threshold=10, loss_threshold=0.1)
    trainer = Train(model, train_loader, test_loader, metas, presentation, overfit_detector)
    trainer.run()
    presentation.finalize_results()
