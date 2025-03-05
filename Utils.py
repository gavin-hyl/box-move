import matplotlib.pyplot as plt

class LossTracker:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_losses = []
        self.train_losses = []

    def __call__(self,train_loss=None, val_loss=None):
        if val_loss is not None:
            self.val_losses.append(val_loss)
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0
                self.early_stop = False
        if train_loss is not None:
            self.train_losses.append(train_loss)
    
    def render(self, save_path="loss_plot.png"):
        plt.plot(self.val_losses, label="Validation Loss")
        plt.plot(self.train_losses, label="Training Loss")
        plt.legend()
        # plt.show()
        plt.savefig(save_path)