import pickle
import argparse
import sys
import json
import pathlib
import numpy as np
from typing import Dict

from HOUDINI.Library.OpLibrary import OpLibrary
from HOUDINI.Run.Task import Task, TaskSettings
from HOUDINI.Run.Utils import get_lganm_io_examples
from HOUDINI.Synthesizer.AST import mkFuncSort, mkRealTensorSort, mkListSort
sys.path.append('.')


def get_task_settings(data_dict: Dict) -> TaskSettings:
    """Get the TaskSettings namedtuple, which
    stores important learning parmeters

    Args:
        data_dict: the dict storing
    """

    task_settings = TaskSettings(
        train_size=64,
        val_size=64,
        training_percentages=[100],
        N=5000,
        M=1,
        K=1,
        learning_rate=0.02,
        var_num=data_dict['envs'][0].shape[1] - 1,
        warm_up=8,
        lambda_1=5,
        lambda_2=0.08,
        lambda_cau=10.,
        data_dict=data_dict)
    return task_settings


class IdentifyTask(Task):
    def __init__(self,
                 lganm_dict: Dict,
                 settings: TaskSettings,
                 lib: OpLibrary):
        """The class of the causal(parent) variable
        identification task

        Args:
            lganm_dict: dict storing lganm data info
            settings: the namedtuple storing important
                learning parameters
            lib: library of higher-order functions
        """

        self.outcome = lganm_dict['target']
        self.envs = lganm_dict['envs']
        self.dt_dim = self.envs[0].shape[1]
        input_type = mkListSort(mkRealTensorSort([1, self.dt_dim - 1]))
        output_type = mkRealTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)

        super().__init__(fn_sort,
                         settings,
                         lib)

    def get_io_examples(self):
        return get_lganm_io_examples(self.envs,
                                     self.outcome,
                                     self.dt_dim)

    def name(self):
        return 'idef_Vars'

    def sname(self):
        return 'idef'


def main(lganm_dict: Dict):
    """The main function that runs the lganm experiments

    Args:
        lganm_dict: dict storing lganm data info
    """

    task_settings = get_task_settings(lganm_dict)

    lib = OpLibrary(['do', 'compose',
                     'repeat', 'cat', 'conv'])

    task = IdentifyTask(lganm_dict,
                        task_settings,
                        lib)
    task.run()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lganm-dir',
                        type=pathlib.Path,
                        default='/home/histopath/Data/LGANM/',
                        metavar='DIR')

    parser.add_argument('--res-dir',
                        type=pathlib.Path,
                        default='/home/histopath/Experiment/LGANM/',
                        metavar='DIR')

    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='num of repeated experiments')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    res_dir = args.res_dir / 'proposed'
    res_dir.mkdir(parents=True, exist_ok=True)

    json_out = dict()
    jacads, fwers, errors = list(), list(), list()
    for pkl_id, pkl_file in enumerate(args.lganm_dir.glob('*.pickle')):
        with open(str(pkl_file), 'rb') as pl:
            lganm_dict = pickle.load(pl)
            lganm_dict['truth'] = list(lganm_dict['truth'])
            dag_wei = lganm_dict['case'].sem.W
            assert (dag_wei.shape[1] == dag_wei.shape[0])
            assert np.all(np.asarray(lganm_dict['truth']) < dag_wei.shape[0])
            assert lganm_dict['target'] < dag_wei.shape[1]
            print(dag_wei.shape)
            print('all the parents: {}, outcome: {}'.format(
                lganm_dict['truth'], lganm_dict['target']))
            lganm_parm = {'dict_name': 'lganm',
                          'repeat': args.repeat,
                          'mid_size': lganm_dict['envs'][0].shape[1],
                          'out_type': 'mse',
                          'env_num': len(lganm_dict['envs'])}
            lganm_dict.update(lganm_parm)

        main(lganm_dict)
        res_dict = lganm_dict['json_out']
        for prog_str, prog_dict in res_dict.items():
            if prog_str not in json_out:
                json_out[prog_str] = {'jacads': list(),
                                      'fwers': list(),
                                      'errors': list(),
                                      'rejects': list(),
                                      'accepts': list()}
            json_out[prog_str]['jacads'].extend(prog_dict['jacads'])
            json_out[prog_str]['fwers'].extend(prog_dict['fwers'])
            if prog_dict['jacads'][0][-1] != 1.:
                json_out[prog_str]['errors'].append(pkl_file.stem)

    for prog_str in json_out:
        jacads = np.asarray(json_out[prog_str]['jacads'])
        fwers = np.asarray(json_out[prog_str]['fwers'])
        json_out[prog_str]['jacads_mean'] = np.mean(jacads, axis=0).tolist()
        json_out[prog_str]['jacads_std'] = np.std(jacads, axis=0).tolist()
        json_out[prog_str]['fwers_mean'] = np.mean(fwers, axis=0).tolist()
        json_out[prog_str]['fwers_std'] = np.std(fwers, axis=0).tolist()
        print('Jaccard Similarity (JS) mean: {}, std: {}.'.format(
            np.mean(jacads, axis=0), np.std(jacads, axis=0)))
        print('Family-wise error rate (FWER) mean: {}, std: {}.'.format(
            np.mean(fwers, axis=0), np.std(fwers, axis=0)))

    json_file = res_dir / 'lganm_table.json'
    print('Save results to {}.'.format(json_file))
    with open(str(json_file), 'w', encoding='utf-8') as f:
        json.dump(json_out, f, ensure_ascii=False, indent=4)
