from machine.util.callbacks import Callback


class EarlyStoppingCallback(Callback):
    """
    Original callback taken from https://github.com/ncullen93/torchsample
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self,
                 monitor='eval_losses',
                 lm_name=None,
                 min_delta=0,
                 patience=5):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss does not improve by a certain amount for a certain
        number of epochs
        Arguments
        ---------
        monitor : string in {'eval_losses', 'train_losses'}
            whether to monitor train or val loss
        lm_name: loss or metric name eg. 'Avg NLLoss' (default None)
                 If not specified then the first element
                 in the monitor array is used
        min_delta : float
            minimum change in monitored value to qualify as improvement.
            This number should be positive.
        patience : integer
            number of epochs to wait for improvment before terminating.
            the counter be reset after each improvment
        """

        self.monitor = monitor
        self.lm_name = lm_name

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
        This allows specifing what eval or train loss to use
        """

        # if specific loss/metric name is specified
        if self.lm_name is not None:
            for m in info[self.monitor]:
                if m.name == self.lm_name:
                    current_loss = info[self.monitor][0].get_loss()
                    break
        else:  # just use the first metric/loss in the array
            current_loss = info[self.monitor][0].get_loss()

        # compare current loss to previous best
        if (current_loss - self.best_loss) < -self.min_delta:
            self.best_loss = current_loss
            self.wait = 1
        else:
            if self.wait >= self.patience:
                self.trainer._stop_training = True
            self.wait += 1
