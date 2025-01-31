import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time

class Train:
    def __init__(self, yolo_model, gnn_model, train_loader, test_loader, meta, presentation, overfit_detector):
        self.yolo_model = yolo_model
        self.gnn_model = gnn_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.meta = meta
        self.presentation = presentation
        self.overfit_detector = overfit_detector
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model.to(self.device)
        self.gnn_model.to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.yolo_model.parameters()) + list(self.gnn_model.parameters()),
            lr=self.meta.enquireMetaValue("learning_rate"),
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.meta.enquireMetaValue("num_epochs")
        )

        self.criterion = nn.CrossEntropyLoss()

    def build_graph(self, component_features, batch_size):
        num_components = component_features.size(1)
        total_nodes = batch_size * (num_components + 1)  # +1 for the main object node
        
        x = torch.zeros(total_nodes, 64, device=self.device)
        edge_index = []
        batch = []
        
        for i in range(batch_size):
            main_node_idx = i * (num_components + 1)
            x[main_node_idx] = component_features[i].mean(dim=0)  # Main object feature
            x[main_node_idx+1:main_node_idx+num_components+1] = component_features[i]
            
            # Connect main node to all component nodes
            for j in range(num_components):
                edge_index.append([main_node_idx, main_node_idx + j + 1])
                edge_index.append([main_node_idx + j + 1, main_node_idx])
            
            batch.extend([i] * (num_components + 1))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)
        
        return x, edge_index, batch

    def train_epoch(self, epoch):
        self.yolo_model.train()
        self.gnn_model.train()

        train_loss, correct, total = 0, 0, 0
        start_time = time.time()

        for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # YOLO Forward Pass
            class_logits, component_features, _ = self.yolo_model(inputs)

            # Build graph
            x, edge_index, batch = self.build_graph(component_features, inputs.size(0))

            # GNN Forward Pass
            gnn_output = self.gnn_model(x, edge_index, batch)

            # Compute loss
            loss_yolo = self.criterion(class_logits, labels)
            loss_gnn = self.criterion(gnn_output, labels)
            loss = loss_yolo + loss_gnn

            # Backward and optimize
            loss.backward()
            self.optimizer.step()

            # Compute accuracy (using YOLO output for simplicity)
            _, preds = torch.max(class_logits, 1)
            train_loss += loss.item() * inputs.size(0)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        epoch_time = time.time() - start_time
        train_loss /= total
        train_acc = 100.0 * correct / total

        return train_loss, train_acc, epoch_time

    def test_epoch(self, epoch):
        self.yolo_model.eval()
        self.gnn_model.eval()

        test_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc=f"Epoch {epoch + 1} Testing", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # YOLO Forward Pass
                class_logits, component_features, _ = self.yolo_model(inputs)

                # Build graph
                x, edge_index, batch = self.build_graph(component_features, inputs.size(0))

                # GNN Forward Pass
                gnn_output = self.gnn_model(x, edge_index, batch)

                # Compute loss
                loss_yolo = self.criterion(class_logits, labels)
                loss_gnn = self.criterion(gnn_output, labels)
                loss = loss_yolo + loss_gnn

                # Compute accuracy (using YOLO output for simplicity)
                _, preds = torch.max(class_logits, 1)
                test_loss += loss.item() * inputs.size(0)
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
                epoch, train_loss, train_acc, test_loss, test_acc, epoch_time, current_lr, overfit_tag
            )

            # Update the learning rate
            self.scheduler.step()
