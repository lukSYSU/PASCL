import torch
from set2tree.utils import InvalidLCAMatrix, lca2adjacency
from torchmetrics import Metric

class PerfectLCAG(Metric):
    """Computes the percentage of the Perfectly predicted LCAs

    - `update` must receive output of the form `(y_pred, y)` or `{'y_pred': y_pred, 'y': y}`.
    - `y_pred` must contain logits and has the following shape (batch_size, num_categories, ...)
    - `y` should have the following shape (batch_size, ...) and contains ground-truth class indices
        with or without the background class. During the computation, argmax of `y_pred` is taken to determine
        predicted classes.

    First the average percentage per batch is computed.
    Then the average of all the batches is returned.
    """

    def __init__(self, batch_size, ignore_index=-1.0):
        super().__init__()

        self.batch_size = batch_size
        self.ignore_index = (
            ignore_index if isinstance(ignore_index, list) else [ignore_index]
        )
        # self.device = device

        self.add_state("_per_corrects", torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("_num_examples", torch.tensor(0), dist_reduce_fx="sum")


    # @reinit__is_reduced
    # def reset(self):

    #     self._per_corrects = 0
    #     self._num_examples = 0

    #     super(PerfectLCA, self).reset()

    """ Computes the percentage of Perfect LCAs PER BATCH!.
    the tensors y_pred and y contain multiple LCAs that belong in a batch.
    """

    # @reinit__is_reduced
    def update(self, output):

        predictions, lcas, edges_batch, node_batch = output  # (N, C), (N)

        # Already log_softmaxed...
        # predictions = torch.softmax(predictions, dim=1)  # (N, C)
        # n = predictions.shape[0]
        # predictions = predictions.reshape((n, n, 8))
        # predictions = predictions + predictions.transpose(0, 1)

        # The diagonal will be equal due to masking
        # Because I've implement mini_batching technique instructed by pyg, I need to
        # seperate one graph into {self.batch_size} graphs
        curr_batch = 0
        for batch in range(self.batch_size):
            pred = predictions[curr_batch:curr_batch + edges_batch[batch]]
            lca = lcas[curr_batch:curr_batch + edges_batch[batch]]
            n = node_batch[batch]
            pred = pred.reshape((n, n, 8))
            lca = lca.reshape((n, n))
            pred = pred + pred.transpose(0, 1)
            fix_pred = torch.argmax(pred, dim=-1).long()
            fix_pred[lca == -1] = -1

            curr_batch += edges_batch[batch]
            # Count the number of zero wrong predictions across the batch.
            # print(fix_pred, lca)
            batch_perfect = torch.equal(fix_pred, lca)

            self._per_corrects += batch_perfect
            self._num_examples += 1

            # print("correct:",self._per_corrects)
            # print("total:",self._num_examples)

        # @sync_all_reduce("_perfect")

    def compute(self):
        if self._num_examples == 0:
            raise ValueError("Must have at least one example before it can be computed because \
                                     the number of examples should not be 0.")
        return self._per_corrects / self._num_examples
