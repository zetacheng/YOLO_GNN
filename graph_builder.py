import torch
from torch_geometric.data import Data
from torch_cluster import knn

class GraphBuilder:
    def __init__(self, knn_neighbors=3):
        """
        Initializes the GraphBuilder with hyperparameters.
        :param knn_neighbors: Number of neighbors for intra-level connections.
        """
        self.knn_neighbors = knn_neighbors

    def build_graph(self, yolo_detections, feature_maps):
        """
        Constructs a hierarchical graph with levels:
        Level 0: Entire image (single node)
        Level 1: Object nodes (detected objects from YOLOv8)
        Level 2: Component nodes (parts of detected objects)
        Level 3: Feature nodes (fine-grained details)

        :param yolo_detections: List of detected objects (bounding boxes, class labels, confidence scores)
        :param feature_maps: Extracted deep features from YOLOv8 or another module
        :return: PyG Graph Data object
        """
        nodes = []
        edges = []
        edge_weights = []

        node_features = []
        edge_index = []

        ## 1️⃣ Create Global Image Node (Level 0)
        image_node_idx = 0
        nodes.append(image_node_idx)
        node_features.append(torch.mean(feature_maps, dim=(1,2)))  # Mean pooling

        ## 2️⃣ Create Object Nodes (Level 1)
        object_indices = {}
        object_start_idx = 1

        for i, detection in enumerate(yolo_detections):
            obj_idx = object_start_idx + i
            nodes.append(obj_idx)
            object_indices[i] = obj_idx
            node_features.append(feature_maps[i])

            # Connect to Global Image Node (Parent)
            edge_index.append([image_node_idx, obj_idx])
            edge_index.append([obj_idx, image_node_idx])
            edge_weights.append(1.0)

        ## 3️⃣ Create Component Nodes (Level 2)
        component_indices = {}
        component_start_idx = len(nodes)

        for i, (obj_idx, detection) in enumerate(object_indices.items()):
            components = self.extract_components(yolo_detections[i])
            for j, comp in enumerate(components):
                comp_idx = component_start_idx + len(component_indices)
                nodes.append(comp_idx)
                component_indices[(i, j)] = comp_idx
                node_features.append(self.extract_fine_features(feature_maps[i], comp))

                # Connect to Object Node (Parent)
                edge_index.append([obj_idx, comp_idx])
                edge_index.append([comp_idx, obj_idx])
                edge_weights.append(1.0)

        ## 4️⃣ Create Feature Nodes (Level 3)
        feature_start_idx = len(nodes)

        for (obj_i, comp_j), comp_idx in component_indices.items():
            details = self.extract_fine_features(feature_maps[obj_i], components[comp_j])

            for k, feat in enumerate(details):
                feat_idx = feature_start_idx + k
                nodes.append(feat_idx)
                node_features.append(feat)

                # Connect to Component Node (Parent)
                edge_index.append([comp_idx, feat_idx])
                edge_index.append([feat_idx, comp_idx])
                edge_weights.append(1.0)

        ## 5️⃣ Intra-Level KNN
        node_features_tensor = torch.stack(node_features)

        for level in range(4):
            level_nodes = [n for n in nodes if self.get_node_level(n) == level]
            level_features = node_features_tensor[level_nodes]

            # Compute intra-level KNN connections
            knn_edges = knn(level_features, level_features, k=self.knn_neighbors)
            edge_index.extend(knn_edges.tolist())
            edge_weights.extend([0.5] * knn_edges.size(1))

        ## Convert to PyG Data Object
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight_tensor = torch.tensor(edge_weights, dtype=torch.float)

        data = Data(x=node_features_tensor, edge_index=edge_index_tensor, edge_attr=edge_weight_tensor)
        return data

    def extract_components(self, yolo_detection):
        """
        Extracts object components based on bounding box segmentation.
        :param yolo_detection: Bounding box info (x, y, w, h, class_id)
        :return: List of component bounding boxes [(x1, y1, w1, h1), ...]
        """
        x, y, w, h, class_id = yolo_detection
        components = []

        # Simple heuristic: Divide object into 4 equal parts
        comp_w, comp_h = w // 2, h // 2

        components.append((x, y, comp_w, comp_h))  # Top-left
        components.append((x + comp_w, y, comp_w, comp_h))  # Top-right
        components.append((x, y + comp_h, comp_w, comp_h))  # Bottom-left
        components.append((x + comp_w, y + comp_h, comp_w, comp_h))  # Bottom-right

        return components

    def extract_fine_features(self, feature_map, component_bbox):
        """
        Extracts fine-grained features from a component using edge detection.
        :param feature_map: CNN feature map from YOLO
        :param component_bbox: (x, y, w, h) of the object component
        :return: Feature tensor
        """
        x, y, w, h = component_bbox

        # Extract region from the feature map
        component_feature = feature_map[:, y:y+h, x:x+w]  # Crop feature map

        # Use a simplified feature extraction (e.g., mean pooling)
        fine_features = torch.mean(component_feature, dim=(1, 2))

        return fine_features

    def get_node_level(self, node_idx):
        """
        Returns the level of a node based on its index.
        :param node_idx: Index of the node
        :return: Level (0, 1, 2, or 3)
        """
        if node_idx == 0:
            return 0  # Image Node
        elif 1 <= node_idx < 10:  # Assuming 10 objects max
            return 1  # Object Nodes
        elif 10 <= node_idx < 50:  # Assuming 4 components per object
            return 2  # Component Nodes
        else:
            return 3  # Fine-Grained Feature Nodes
