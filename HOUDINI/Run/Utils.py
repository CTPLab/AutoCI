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

    dt_trn = (np.stack(dt_input, 1),
              np.stack(dt_lab, 1))
    dt_val, dt_tst = dt_trn, dt_trn
    return dt_trn, dt_val, dt_tst


def get_por123_io_examples(portec_file: pathlib.Path,
                           causal: List[str],
                           outcome: str) -> Tuple[Tuple, Tuple, Tuple]:
    """Obtain the portec data 

    Args:
        portec_file: the *.sav file storing portec data
        causal: the candidate causal variables
        outcome: the outcome data, RFSstatus and RFSyears

    Returns:
        the train, val, test portec data 
    """

    df, _ = pyreadstat.read_sav(str(portec_file))
    df = df.dropna(subset=causal)
    df = df.dropna(subset=outcome)

    # the values of the cau_var are the median of that variable,
    # which is used to create binary lable
    portec = list()
    df_input, df_lab = list(), list()
    max_len = 0
    for i in range(3):
        df_ptc = df.loc[df['PortecStudy'] == i + 1]
        df_ptc = np.concatenate((df_ptc[causal].values,
                                 df_ptc[outcome].values), axis=-1)
        portec.append(df_ptc)
        max_len = max(max_len, df_ptc.shape[0])

    for ptc_id, ptc in enumerate(portec):
        ptc = pad_array(ptc, max_len)
        df_input.append(ptc[:, :-len(outcome)])
        df_lab.append(ptc[:, -len(outcome):])

    df_trn = (np.stack(df_input, 1),
              np.stack(df_lab, 1))
    df_val, df_tst = df_trn, df_trn
    print(df_trn[0].shape, df_trn[1].shape)
    return df_trn, df_val, df_tst


def prep_sav_portec123_age(portec_dir):
    # obtain the dict that maps case_id to spot_id
    sav_file = portec_dir / 'P123_update.sav'

    df, meta = pyreadstat.read_sav(str(sav_file))
    df['Age'] = 0.0
    df.loc[(df['age'] >= 60) & (df['age'] <= 70), 'Age'] = 1.0
    df.loc[df['age'] > 70, 'Age'] = 2.0

    df['LVSI2'] = df['LVSI_2cat']
    # df['Grade3'] = (df['Histgrade_3cat'] - 1) / 2.
    df['Stage2'] = df['stage_2cat'] - 1
    # df['Stage4'] = (df['stage_4cat'] - 1) / 3.
    df['Treat2'] = df['treat_2cat']
    # df['Treat4'] = df['treat'] / 3.

    for gid, gol in enumerate(['g2', 'g3']):
        df[gol] = np.zeros_like(df['Histgrade_3cat'].values)
        # Histgrade_3cat value 1 (ref), 2, 3
        df.loc[df['Histgrade_3cat'] == gid + 2, gol] = 1

    for sid, sol in enumerate(['s2', 's3', 's4']):
        df[sol] = np.zeros_like(df['stage_4cat'].values)
        # stage_4cat value 1 (ref), 2, 3
        df.loc[df['stage_4cat'] == sid + 2, sol] = 1

    for mid, mol in enumerate(['POLE', 'MMRd', 'p53a']):
        df[mol] = df['TCGA_4groups']
        # assign 0 to not nan cases of TCGA groups
        df.loc[~df['TCGA_4groups'].isnull(), mol] = 0
        df.loc[df['TCGA_4groups'] == mid + 1, mol] = 1

    for tid, tol in enumerate(['t1', 't2', 't3']):
        df[tol] = np.zeros_like(df['treat'].values)
        df.loc[df['treat'] == tid + 1, tol] = 1

    new_sav = portec_dir / f'portec_update_age.sav'
    print(new_sav)
    pyreadstat.write_sav(df, str(new_sav))
    new_csv = sav_file.parents[0] / f'portec_update_age.csv'
    print(new_csv)
    df.to_csv(str(new_csv))


def main():
    sav_file = pathlib.Path('/Path/to/PORTEC123')
    prep_sav_portec123_age(sav_file)


if __name__ == '__main__':
    main()
