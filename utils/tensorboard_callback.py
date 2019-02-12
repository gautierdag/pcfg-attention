from machine.util.callbacks import Callback
from tensorboardX import SummaryWriter


class TensorboardCallback(Callback):
    """
    Callback that is used to store information about
    the training during the training, and writes it to
    a file to the end to be able to read it later.
    """

    def __init__(self, run_path):
        """
        Pass in the run 
        """
        super(TensorboardCallback, self).__init__()
        self.writer = SummaryWriter(run_path)

    def on_epoch_end(self, info=None):
        # self.logs.write_to_log('Train', info['train_losses'],
        #                        info['train_metrics'], info['step'])

    def on_batch_begin(self, batch, info=None):
        pass

    def on_batch_end(self, batch, info=None):
        if info['print']:
            # self.logs.update_step(info['step'])
            # for m_data in self.trainer.monitor_data:
            #     self.logs.write_to_log(m_data,
            #                            info['monitor_losses'][m_data],
            #                            info['monitor_metrics'][m_data],
            #                            info['step'])

    def on_train_begin(self, info=None):
        pass

    def on_train_end(self, info=None):
        pass

# class Callback(object):
#     """
#     Abstract base class to build callbacks.

#     Inspired by keras' callbacks.
#     A callback is a set of functions to be applied at given stages
#     of the training procedure. You can use callbacks to get a view
#     on internal states and statistics of the model during training.
#     You can pass a list of callbacks (as the keyword argument callbacks)
#     to the train() method of the SupervisedTrainer.
#     The relevant methods of the callbacks will then be called at each
#     stage of the training.
#     """

#     def __init__(self):
#         pass

#     def set_trainer(self, trainer):
#         self.trainer = trainer

#     def on_epoch_begin(self, info=None):
#         pass

#     def on_epoch_end(self, info=None):
#         """
#         Function called at the end of every epoch
#         self.info['train_losses'] and self.info['train_metrics'] should be available to use here.
#         self.info['eval_losses'] and self.info['eval_metrics'] should be available to use here.
#         """
#         pass

#     def on_batch_begin(self, batch, info=None):
#         pass

#     def on_batch_end(self, batch, info=None):
#         """
#         Function called at the end of every batch
#         If self.info['print'] = True:
#             Then self.info['monitor_losses'] and self.info['monitor_metrics']
#             should be available to use here.
#         If self.info['checkpoint'] = True:,
#             Then self.info['eval_losses'] and self.info['eval_metrics']
#             should be available to use here.
#         """
#         pass

#     def on_train_begin(self, info=None):
#         """
#         Function called at the very beginning of every training
#         self.info['eval_losses'] and self.info['eval_metrics'] should be available to use here.
#         """
#         pass

#     def on_train_end(self, info=None):
#         pass

#     @staticmethod
#     def get_losses(losses, metrics, step):
#         total_loss = 0
#         model_name = ''
#         log_msg = ''

#         for metric in metrics:
#             val = metric.get_val()
#             log_msg += '%s %.4f ' % (metric.name, val)
#             model_name += '%s_%.2f_' % (metric.log_name, val)

#         for loss in losses:
#             val = loss.get_loss()
#             log_msg += '%s %.4f ' % (loss.name, val)
#             model_name += '%s_%.2f_' % (loss.log_name, val)
#             total_loss += val

#         model_name += 's%d' % step

#         return total_loss, log_msg, model_name
