class TrainingMonitor:

    def __init__(self, patience):
        self.counter = 0
        self.patience = patience
        self.lowest_loss = None

    def __call__(self, loss):
        """returns True if a new lowest loss value was found, False otherwise"""
        if loss is None:
            raise ValueError('loss must not be None')

        if self.lowest_loss is None or loss < self.lowest_loss:
            self.lowest_loss = loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False

    @property
    def early_stopping_enabled(self):
        return self.patience is not None

    @property
    def should_early_stop(self):
        return self.early_stopping_enabled and self.current_patience <= 0

    @property
    def current_patience(self):
        """if this value is less or equal to zero, early stopping should be performed"""
        if self.patience is None:
            return 1  # if patience is None, early stopping will never happen
        else:
            return self.patience - self.counter
