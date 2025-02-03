import matplotlib.pyplot as plt
from tqdm import tqdm
import cProfile
import pstats
from pstats import SortKey
from torch.profiler import profile, record_function, ProfilerActivity

class Presentation:
    def __init__(self):
        self.epoch_times = []
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.learning_rates = []
        self.overfit_tags = []

    def display_meta_parameters(self, meta):
        print("=== Meta Parameters ===")
        for attr in dir(meta):
            if not attr.startswith("_") and not callable(getattr(meta, attr)):
                print(f"{attr}: {getattr(meta, attr)}")
        print("=======================")

    def display_epoch_results(self, epoch, train_loss, train_acc, test_loss, test_acc, epoch_time, learning_rate, overfit_tag=""):
        self.epoch_times.append(epoch_time)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)
        self.learning_rates.append(learning_rate)
        self.overfit_tags.append(overfit_tag)

        print(f"Epoch {epoch + 1} | Time: {epoch_time:.2f} s | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | {overfit_tag}")

    def finalize_results(self, save_path=None):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.test_losses, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Over Epochs")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_accuracies, label="Train Accuracy")
        plt.plot(epochs, self.test_accuracies, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Over Epochs")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.learning_rates, label="Learning Rate")
        plt.xlabel("Epochs")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Over Epochs")
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"Plots saved to {save_path}")
        plt.show()

    def display_progress_bar(self, phase, epoch, num_epochs, data_loader):
        """
        Displays a progress bar for training or testing.
        :param phase: 'Train' or 'Test'
        :param epoch: Current epoch
        :param num_epochs: Total number of epochs
        :param data_loader: DataLoader for the phase
        """
        return tqdm(data_loader, desc=f"{phase} Progress: Epoch {epoch + 1}/{num_epochs}", leave=False)

    def start_profiler(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def end_profiler(self):
        self.profiler.disable()
        stats = pstats.Stats(self.profiler).sort_stats(SortKey.TIME)
        stats.print_stats(10)  # Print top 10 time-consuming functions

    def profile_train_epoch(self, train_epoch_func, train_loader):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("train_epoch"):
                result = train_epoch_func(train_loader)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        return result
