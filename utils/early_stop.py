class EarlyStopping:
    def __init__(self, patience=30):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.is_improved = False  

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.is_improved = True
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.is_improved = True
        else:
            self.counter += 1
            self.is_improved = False
            if self.counter >= self.patience:
                self.early_stop = True
