import matplotlib.pyplot as plt
import os

class TrainingVisualizer:
    def __init__(self, save_dir="logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history = {'total_loss': [], 'tri_loss': [], 'ce_loss': []}

    def update(self, total, tri, ce=0.0):
        self.history['total_loss'].append(total)
        self.history['tri_loss'].append(tri)
        self.history['ce_loss'].append(ce)
        self._plot()

    def _plot(self):
        epochs = range(1, len(self.history['total_loss']) + 1)
        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, self.history['total_loss'], 'b-o', label='Total Loss')
        plt.plot(epochs, self.history['tri_loss'], 'r-s', label='Triplet Loss')
        if any(v > 0 for v in self.history['ce_loss']):
            plt.plot(epochs, self.history['ce_loss'], 'g-^', label='CE Loss')
            
        plt.title('Training Convergence Trends')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.savefig(os.path.join(self.save_dir, "loss_curve.png"))
        plt.close()