import torch
from meta_manager import Meta
from simple_vig_gnn import SimpleViG_GNN  # Import the new model
from data import CIFAR10DataLoader
from presentation import Presentation
from train import Train
from overfit_detector import OverfitDetector

if __name__ == "__main__":
    metas = Meta()
    
    # ✅ Initialize ViG_GNN correctly
    model = SimpleViG_GNN(num_classes=metas.vig_num_classes)

    data_loader = CIFAR10DataLoader(batch_size=metas.batch_size)
    train_loader, test_loader = data_loader.get_loaders()
    presentation = Presentation()
    presentation.display_meta_parameters(metas)
    overfit_detector = OverfitDetector(accuracy_threshold=10, loss_threshold=0.1)
    
    trainer = Train(model, train_loader, test_loader, metas, presentation, overfit_detector)
    trainer.run()
    presentation.finalize_results()