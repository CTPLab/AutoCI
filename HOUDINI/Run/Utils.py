import csv
import random
import pickle
import pathlib
import itertools
import pyreadstat
import numpy as np

from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from HOUDINI.Interpreter.Utils.EvalUtils import NumpyDataSetIterator


def mk_tag(tag: str,
           content: str,
           cls: List[str] = [],
           attribs: Dict = {}):
    cls_str = ' '.join(cls)
    if len(cls_str) > 0:
        cls_str = 'class = "%s"' % cls_str

    attrib_str = ' '.join(['%s="%s"' % (k, v) for k, v in attribs.items()])

    return '<%s %s %s>%s</%s>\n' % (tag, cls_str, attrib_str, content, tag)


def mk_div(content: str,
           cls: List[str] = []):
    return mk_tag('div', content, cls)


def append_to_file(a_file,
                   content):
    with open(a_file, "a") as fh:
        fh.write(content)


def write_to_file(a_file,
                  content):
    with open(a_file, "w") as fh:
        fh.write(content)


def sav_to_csv(sav_file,
               csv_file):
    df, _ = pyreadstat.read_sav(str(sav_file))
    df.to_csv(str(csv_file))


def header_lookup(headers):
    """The header lookup table. Assign the index for 
    each candidate as follow, i.e.,
        var_id[patient id] = 0
        var_id[survival rate] = 1

    Args:
        headers: the name list of candidate causal variables,
            outcome, patien id, ,etc.
    """

    var_id = dict()

    for idx, head in enumerate(headers):
        var_id[head] = idx

    return var_id


def pad_array(env,
              env_len):
    """Pad the data array for each environment s.t.
    the array length is the same for all envs.

    Args:
        env: the data array of an environment
        env_len: the length that the data array should reach

    """

    assert env_len >= env.shape[0], \
        'the max len {} should >= data len {}'.format(env_len, env.shape[0])
    q, r = divmod(env_len, env.shape[0])
    if q > 1:
        env = np.repeat(env, q, axis=0)
    if r > 0:
        id_env = list(range(env.shape[0]))
        np.random.shuffle(id_env)
        env = np.concatenate((env, env[id_env[:r]]), axis=0)
    return env


def iterate_diff_training_sizes(train_io_examples,
                                train_percent):
    """iterate different training data based on the percentages

    Args:
        train_io_examples: training data per batch
        train_percent: the percentage of the data for training
    """

    # assuming all lengths are represented equally
    if issubclass(type(train_io_examples), NumpyDataSetIterator) or \
            type(train_io_examples) == list and \
            issubclass(type(train_io_examples[0]), NumpyDataSetIterator):
        num_of_training_dp = train_io_examples[0].inputs.shape[0]
        # raise NotImplementedError("uhm?!")
        yield train_io_examples, num_of_training_dp
        return

    if type(train_io_examples) == list:
        num_of_training_dp = train_io_examples[0][0].shape[0]
    else:
        num_of_training_dp = train_io_examples[0].shape[0]

    for percent in train_percent:
        c_num_items = (percent * num_of_training_dp) // 100
        if type(train_io_examples) == list:
            c_tr_io_examples = [(t[0][:c_num_items],
                                 t[1][:c_num_items]) for t in train_io_examples]
            return_c_num_items = c_num_items * len(train_io_examples)
        else:
            c_tr_io_examples = (train_io_examples[0][:c_num_items],
                                train_io_examples[1][:c_num_items])
            return_c_num_items = c_num_items

        yield c_tr_io_examples, return_c_num_items


def get_lganm_io_examples(lganm_envs: List[np.ndarray],
                          outcome: int,
                          dt_dim: int,
                          max_len: int = 6000) -> Tuple[Tuple, Tuple, Tuple]:
    """Obtain the lganm data 

    Args:
        lganm_envs: dict storing the lganm data 
            collected from different environments
        outcome: outcome variable
        dt_dim: the data feature dimension including 
            candidate causal and outcome variable(s)
        max_len: the maximum number of data in each env

    Returns:
        the train, val, test lganm data 
    """

    dt_input, dt_lab = list(), list()
    lab_msk = np.zeros(dt_dim, dtype=bool)
    lab_msk[outcome] = True
    cfd_msk = np.ones(dt_dim, dtype=bool)
    cfd_msk[outcome] = False

    for env_id, env in enumerate(lganm_envs):
        if env.shape[0] < max_len:
            env = pad_array(env, max_len)
        else:
            env = env[:max_len]
        dt_input.append(env[:, cfd_msk])
        dt_lab.append(env[:, lab_msk])
        # dt_lab[-1][:]= env_id

    dt_trn = (np.stack(dt_input, 1),
              np.stack(dt_lab, 1))
    dt_val, dt_tst = dt_trn, dt_trn
    return dt_trn, dt_val, dt_tst


def main():
    pass


if __name__ == '__main__':
    main()
