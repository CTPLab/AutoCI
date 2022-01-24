import numpy as np
from typing import Tuple, List
from HOUDINI.Library.NN import NetMLP
from HOUDINI.Synthesizer import AST
from HOUDINI.Synthesizer.Utils import ASTUtils


def get_lib_names(term: AST.PPTerm) -> List[str]:
    if isinstance(term, AST.PPVar):
        name = term.name
        if name[:4] == "lib.":
            return [name[4:]]
        else:
            return []
    else:
        nts = []
        for c in ASTUtils.deconstruct(term):
            cnts = get_lib_names(c)
            nts.extend(cnts)
        return nts


def is_evaluable(st, ns) -> Tuple[bool, int]:
    """The evaluation rules for the type-safe candidates 

    Args:
        st: the program to be examined. 
        ns: the neural synthesizer.
    """

    # The program should not be open (no non-terminals).
    if ASTUtils.isOpen(st):
        return False, 1

    unks = ASTUtils.getUnks(st)

    # At most 3 unks for now.
    if len(unks) > 3:
        return False, 2

    number_of_mlp_nns = 0

    for unk in unks:
        # type variables and dimension variables not allowed.
        if ASTUtils.isAbstract(unk.sort):
            return False, 3

        # Only function types allowed
        if type(unk.sort) != AST.PPFuncSort:
            return False, 4

        # ******* INPUTS ******* :
        # An input to a function can't be a function
        if any([type(arg_sort) == AST.PPFuncSort for arg_sort in unk.sort.args]):
            return False, 11

        fn_input_sort = unk.sort.args[0]
        fn_output_sort = unk.sort.rtpe

        # No more than 2 arguments
        num_input_arguments = unk.sort.args.__len__()
        if num_input_arguments > 1:
            in1_is_2d_tensor = type(
                unk.sort.args[0]) == AST.PPTensorSort and unk.sort.args[0].shape.__len__() == 2
            in2_is_2d_tensor = type(
                unk.sort.args[1]) == AST.PPTensorSort and unk.sort.args[1].shape.__len__() == 2
            out_is_2d_tensor = type(
                unk.sort.rtpe) == AST.PPTensorSort and unk.sort.rtpe.shape.__len__() == 2
            # If a function takes 2 inputs, they'll be concatenated.
            # Thus, we need them to be 2 dimensional tensors
            if num_input_arguments == 2 and in1_is_2d_tensor and in2_is_2d_tensor and out_is_2d_tensor:
                continue
            else:
                return False, 5

        # If the NN's input is a list, it should be: List<2dTensor> -> 2dTensor
        # (as seq-to-seq models aren't supported)
        # We support List<2dTensor> -> 2dTensor
        if type(unk.sort.args[0]) == AST.PPListSort:
            return False, 6

        # If the input to the NN is an image:
        cnn_feature_dim = 64
        input_is_image = type(
            fn_input_sort) == AST.PPTensorSort and fn_input_sort.shape.__len__() == 4
        if input_is_image:
            # if the input is of size [batch_size, _, 28, 28],
            # the output must be of size [batch_size, 32, 4, 4].
            cond0a1 = fn_input_sort.shape[2].value == 28 and fn_input_sort.shape[3].value == 28
            cond0a2 = type(fn_output_sort) == AST.PPTensorSort and fn_output_sort.shape.__len__() == 4 and \
                fn_output_sort.shape[1].value == cnn_feature_dim \
                and fn_output_sort.shape[2].value == 4 and fn_output_sort.shape[3].value == 4
            # if the input is of size [batch_size, 32, 4, 4],
            # the output must be two dimensional.
            cond0b1 = fn_input_sort.shape[1].value == cnn_feature_dim \
                and fn_input_sort.shape[2].value == 4 and fn_input_sort.shape[3].value == 4
            cond0b2 = type(
                fn_output_sort) == AST.PPTensorSort and fn_output_sort.shape.__len__() == 2

            if not ((cond0a1 and cond0a2) or (cond0b1 and cond0b2)):
                return False, 50

            if cond0b1 and cond0b2:
                number_of_mlp_nns += 1
            continue

        # if the input is a 2d tensor:
        in_is_2d_tensor = type(unk.sort.args[0]) == AST.PPTensorSort and \
            unk.sort.args[0].shape.__len__() == 2
        out_is_2d_tensor = type(unk.sort.rtpe) == AST.PPTensorSort and \
            unk.sort.rtpe.shape.__len__() == 2
        if in_is_2d_tensor:
            if out_is_2d_tensor:
                number_of_mlp_nns += 1
            else:
                return False, 51

    lib_names = get_lib_names(st)
    # don't allow multiple repeats,
    # as we could just keep on stacking these.
    if lib_names.count("repeat") > 1:
        return False, 15

    for lib_name in lib_names:
        if type(ns.lib.items[lib_name].obj) == NetMLP:
            number_of_mlp_nns += 1

    # don't allow for multiple MLPs, as we can just keep on stacking them
    # (thus going into architecture search, which is out of scope)
    if number_of_mlp_nns > 2:
        return False, 16

    if lib_names.count('do') != 1:
        return False, 17

    # make sure that the do function directly
    # work on the variable candidate
    if lib_names[-1] != 'do':
        return False, 18

    return True, 0


class NumpyDataSetIterator(object):
    """The class of numpy data iterator.

    Args:
        inputs (ndarray): Array of data input features of shape
            (num_data, input_dim).
        targets (ndarray): Array of data output targets of shape
            (num_data, output_dim) or (num_data,) if output_dim == 1.
        batch_size (int): Number of data points to include in each batch.
        max_num_batches (int): Maximum number of batches to iterate over
            in an epoch. If `max_num_batches * batch_size > num_data` then
            only as many batches as the data can be split into will be
            used. If set to -1 all of the data will be used.
        shuffle_order (bool): Whether to randomly permute the order of
            the data before each epoch.
        rng (RandomState): A seeded random number generator.
    """

    def __init__(self,
                 inputs,
                 targets,
                 batch_size,
                 max_num_batches=-1,
                 shuffle_order=True,
                 rng=None):

        self.inputs = inputs
        self.targets = targets
        if batch_size < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = batch_size
        if max_num_batches == 0 or max_num_batches < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = max_num_batches
        self._update_num_batches()
        self.shuffle_order = shuffle_order
        self._current_order = np.arange(inputs.shape[0])
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.new_epoch()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value < 1:
            raise ValueError('batch_size must be >= 1')
        self._batch_size = value
        self._update_num_batches()

    @property
    def num_datapoints(self):
        num_full_batches = self.inputs.shape[0] // self.batch_size
        if self.inputs.shape[0] % self.batch_size == 0:
            return num_full_batches * self.batch_size
        else:
            return num_full_batches * self.batch_size + self.inputs.shape[0] % self.batch_size

    @property
    def max_num_batches(self):
        return self._max_num_batches

    @max_num_batches.setter
    def max_num_batches(self, value):
        if value == 0 or value < -1:
            raise ValueError('max_num_batches must be -1 or > 0')
        self._max_num_batches = value
        self._update_num_batches()

    def _update_num_batches(self):
        # maximum possible number of batches is equal to number of whole times
        # batch_size divides in to the number of data points which can be
        # found using integer division
        possible_num_batches = self.inputs.shape[0] // self.batch_size + (
            1 if self.inputs.shape[0] % self.batch_size != 0 else 0)
        if self.max_num_batches == -1:
            self.num_batches = possible_num_batches
        else:
            self.num_batches = min(self.max_num_batches, possible_num_batches)

    def __iter__(self):
        """Implements Python iterator interface.
        This should return an object implementing a `next` method which steps
        through a sequence returning one element at a time and raising
        `StopIteration` when at the end of the sequence. Here the object
        returned is the DataProvider itself.
        """

        return self

    def new_epoch(self):
        self._curr_batch = 0
        if self.shuffle_order:
            self.shuffle()

    def __next__(self):
        return self.next()

    def reset(self):
        inv_perm = np.argsort(self._current_order)
        self._current_order = self._current_order[inv_perm]
        self.inputs = self.inputs[inv_perm]
        self.targets = self.targets[inv_perm]
        self.new_epoch()

    def shuffle(self):
        perm = self.rng.permutation(self.inputs.shape[0])
        self._current_order = self._current_order[perm]
        self.inputs = self.inputs[perm]
        self.targets = self.targets[perm]

    def next(self):
        if self._curr_batch + 1 > self.num_batches:
            # no more batches in current iteration through data set so start
            # new epoch ready for another pass and indicate iteration is at end
            self.new_epoch()
            raise StopIteration()
        # create an index slice corresponding to current batch number
        batch_slice = slice(self._curr_batch * self.batch_size,
                            (self._curr_batch + 1) * self.batch_size)
        inputs_batch = self.inputs[batch_slice]
        targets_batch = self.targets[batch_slice]
        self._curr_batch += 1
        return inputs_batch, targets_batch
