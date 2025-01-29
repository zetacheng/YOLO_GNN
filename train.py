import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import time

class Train:
    def __init__(self, yolo_model, gnns, train_loader, test_loader, meta, presentation, overfit_detector):
        self.yolo_model = yolo_model
        self.gnns = gnns  # Dictionary of GNN models per class
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.meta = meta
        self.presentation = presentation
        self.overfit_detector = overfit_detector
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model.to(self.device)
        for gnn in self.gnns.values():
            gnn.to(self.device)

        # Optimizers
        self.yolo_optimizer = torch.optim.Adam(
            self.yolo_model.parameters(),
            lr=self.meta.enquireMetaValue("learning_rate"),
        )
        self.gnn_optimizers = {
            i: torch.optim.Adam(gnn.parameters(), lr=self.meta.enquireMetaValue("learning_rate"))
            for i, gnn in self.gnns.items()
        }

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.yolo_optimizer, T_max=self.meta.enquireMetaValue("num_epochs")
        )

        self.criterion = nn.CrossEntropyLoss()

    def generate_fully_connected_edges(self, num_nodes, device):
        """
        Generate a fully connected edge index for a graph.
        """
        row = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
        col = torch.arange(num_nodes, device=device).repeat(num_nodes)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index

    def train_epoch(self, epoch):
        self.yolo_model.train()
        for gnn in self.gnns.values():
            gnn.train()

        train_loss, correct, total = 0, 0, 0
        start_time = time.time()

        for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # YOLO Forward Pass
            class_logits, _ = self.yolo_model(inputs)  # Unpack YOLO outputs
            preds = class_logits.argmax(dim=1)

            # YOLO Loss and Backward
            loss_yolo = self.criterion(class_logits, labels)
            self.yolo_optimizer.zero_grad()
            loss_yolo.backward()
            self.yolo_optimizer.step()

            # Train GNN for each class
            for label in labels.unique():
                gnn = self.gnns[label.item()]
                optimizer = self.gnn_optimizers[label.item()]

                # Filter inputs and labels for the current class
                label_indices = (labels == label).nonzero(as_tuple=True)[0]
                label_targets = labels[label_indices]  # Targets for this class
                num_instances = len(label_targets)  # Number of instances of this class in the batch

                # Generate dynamic graph data based on the number of instances
                node_features = torch.randn(num_instances, self.meta.enquireMetaValue("gnn_input_dim"), device=self.device)
                edge_index = self.generate_fully_connected_edges(num_instances, device=self.device)

                # GNN Forward Pass
                gnn_output = gnn(node_features, edge_index)

                # Compute loss for the current class
                gnn_loss = self.criterion(gnn_output, label_targets)

                # Backward and optimize GNN
                optimizer.zero_grad()
                gnn_loss.backward()
                optimizer.step()

            # Accumulate metrics
            train_loss += loss_yolo.item() * inputs.size(0)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        epoch_time = time.time() - start_time
        train_loss /= total
        train_acc = 100.0 * correct / total

        return train_loss, train_acc, epoch_time

    def test_epoch(self, epoch):
        self.yolo_model.eval()
        for gnn in self.gnns.values():
            gnn.eval()

        test_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc=f"Epoch {epoch + 1} Testing", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # YOLO Forward Pass
                class_logits, _ = self.yolo_model(inputs)
                loss = self.criterion(class_logits, labels)

                test_loss += loss.item() * inputs.size(0)
                preds = class_logits.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        test_loss /= total
        test_acc = 100.0 * correct / total

        return test_loss, test_acc

    def run(self):
        for epoch in range(self.meta.enquireMetaValue("num_epochs")):
            train_loss, train_acc, epoch_time = self.train_epoch(epoch)
            test_loss, test_acc = self.test_epoch(epoch)

            # Overfit Detection
            overfit_tag = self.overfit_detector.tag(train_loss, test_loss, train_acc, test_acc)

            # Get the current learning rate
            current_lr = self.scheduler.get_last_lr()[0]

            # Log results
            self.presentation.display_epoch_results(
                epoch,
                train_loss,
                train_acc,
                test_loss,
                test_acc,
                epoch_time,
                current_lr,
                overfit_tag,
            )

            # Update the learning rate
            self.scheduler.step()
