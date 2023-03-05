import numpy as np

class EarlyStopper():
    def __init__(self, limit = 20, delta= 0.5):
        self.limit = limit
        self.delta = delta 
        self.min_loss = np.inf
        self.counter = 0
        print(f"Earlystopper active with limit: {self.limit} steps and delta: {self.delta}.")

    def __call__(self, validation_loss):
        if validation_loss < self.min_loss:
            self.min_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_loss + self.delta:
            self.counter += 1
            if self.counter >= self.limit:
                return True
        return False
