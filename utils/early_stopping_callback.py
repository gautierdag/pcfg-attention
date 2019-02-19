from machine.util.callbacks import Callback


class EarlyStoppingCallback(Callback):
    """
    Taken from https://github.com/ncullen93/torchsample
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self,
                 monitor='eval_loss',
                 min_delta=0,
                 patience=5):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss does not improve by a certain amount for a certain
        number of epochs
        Arguments
        ---------
        monitor : string in {'eval_loss', 'train_losses'}
            whether to monitor train or val loss
        min_delta : float
            minimum change in monitored value to qualify as improvement.
            This number should be positive.
        patience : integer
            number of epochs to wait for improvment before terminating.
            the counter be reset after each improvment
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e-15
        super(EarlyStoppingCallback, self).__init__()

    def on_train_begin(self, info=None):
        self.wait = 0
        self.best_loss = 1e15

    def on_epoch_end(self, info=None):
        """
        Function called at the end of every epoch
        @TODO: allow specifing what eval or train loss to use, or even the use of metrics
        to stop training early
        """

        if self.monitor == 'eval_loss':
            current_loss = info['eval_losses'][0].get_loss()
        elif self.monitor == 'train_losses':
            current_loss = info['train_losses'][0].get_loss()
        else:
            raise NotImplementedError(
                'Monitor not implemented for Early stopping')

        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.trainer._stop_training = True
                self.wait += 1
