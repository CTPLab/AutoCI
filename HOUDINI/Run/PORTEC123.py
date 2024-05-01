import sys
import json
import pathlib
import argparse
import numpy as np
from typing import Dict

from HOUDINI.Config import config
from HOUDINI.Run.Task import Task
from HOUDINI.Run.Task import TaskSettings
from HOUDINI.Run.Utils import get_por123_io_examples
from HOUDINI.Library.OpLibrary import OpLibrary
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
        var_num=data_dict['clinical_meta']['causal_num'],
        warm_up=8,
        lambda_1=1,
        lambda_2=0.08,
        lambda_cau=10.,
        data_dict=data_dict)
    return task_settings


class SurvTask(Task):
    def __init__(self,
                 portec_dict: Dict,
                 settings: TaskSettings,
                 lib: OpLibrary):

        self.file = portec_dict['file']
        self.causal = list(portec_dict['clinical_meta']['causal'].keys())
        self.outcome = portec_dict['clinical_meta']['outcome']
        input_type = mkListSort(mkRealTensorSort([1, len(self.causal)]))
        output_type = mkRealTensorSort([1, 1])
        fn_sort = mkFuncSort(input_type, output_type)

        super().__init__(fn_sort,
                         settings,
                         lib)

    def get_io_examples(self):
        return get_por123_io_examples(self.file,
                                      self.causal,
                                      self.outcome)

    def name(self):
        return 'surv_RFS'

    def sname(self):
        return 'surv'


def main(portec_dict: Dict):
    """The main function that runs the portec experiments

    Args:
        portec_dict: dict storing portec data info
    """

    task_settings = get_task_settings(portec_dict)

    lib = OpLibrary(['do', 'compose',
                     'repeat', 'cat', 'conv'])

    seq = SurvTask(portec_dict,
                   task_settings,
                   lib)
    seq.run()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--portec-dir',
                        type=pathlib.Path,
                        default='/home/Path/to/Data/PORTEC123/',
                        metavar='DIR')
    parser.add_argument('--confounder',
                        type=str,
                        choices=['baseline', 'treat2', 'binary'],
                        default='baseline',
                        help='the experiments with confounders. (default: %(default)s)')
    parser.add_argument('--outcome',
                        type=str,
                        choices=['over_surv', 'endo_surv', 'orec_recur',
                                 'vrec_recur', 'prec_recur', 'lrec_recur', 'adrec_recur'],
                        default='adrec_recur',
                        help='the experiments with confounders. (default: %(default)s)')
    parser.add_argument('--repeat',
                        type=int,
                        default=16,
                        help='num of repeated experiments')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    yaml_dict = config('HOUDINI/Yaml/PORTEC123.yaml')
    portec_dict = yaml_dict[args.confounder]
    portec_dict['clinical_meta']['outcome'] = yaml_dict[args.outcome]
    print(portec_dict['clinical_meta']['outcome'])
    mid_size = len(portec_dict['clinical_meta']['causal'].keys()) + \
        len(portec_dict['clinical_meta']['outcome'])
    portec_parm = {'dict_name': 'portec',
                   'file': args.portec_dir / 'portec_update_age.sav',
                   'repeat': args.repeat,
                   'mid_size': mid_size,
                   'out_type': 'hazard',
                   'env_num': 3,
                   'results_dir': args.portec_dir / 'final' / f'{args.confounder}_{args.outcome}'}
    portec_dict.update(portec_parm)
    pathlib.Path(portec_dict['results_dir']).mkdir(
        parents=True, exist_ok=True)

    json_out = dict()
    main(portec_dict)

    res_dict = portec_dict['json_out']
    for prog_str, prog_dict in res_dict.items():
        if prog_str not in json_out:
            json_out[prog_str] = dict()

        jacads = np.asarray(prog_dict['jacads'])
        fwers = np.asarray(prog_dict['fwers'])
        warm_dos = np.asarray(prog_dict['warm_dos'])
        caus_dos = np.asarray(prog_dict['val_dos'])
        json_out[prog_str]['jacads_mean'] = np.mean(jacads, axis=0).tolist()
        json_out[prog_str]['jacads_std'] = np.std(jacads, axis=0).tolist()
        json_out[prog_str]['fwers_mean'] = np.mean(fwers, axis=0).tolist()
        json_out[prog_str]['fwers_std'] = np.std(fwers, axis=0).tolist()
        json_out[prog_str]['warm_prob'] = np.mean(warm_dos, axis=0).tolist()
        json_out[prog_str]['caus_prob'] = np.mean(caus_dos, axis=0).tolist()
        json_out[prog_str]['val_grads'] = prog_dict['val_grads']
        json_out[prog_str]['val_dos'] = prog_dict['val_dos']
        json_out[prog_str]['warm_scores'] = prog_dict['warm_scores']
        json_out[prog_str]['val_scores'] = prog_dict['val_scores']
        print(jacads.shape, fwers.shape)
        print('\nprogram: {}'.format(prog_str))
        print('Jaccard Similarity (JS) mean: {}, std: {}.'.format(
            np.mean(jacads, axis=0), np.std(jacads, axis=0)))
        print('Family-wise error rate (FWER) mean: {}, std: {}.'.format(
            np.mean(fwers, axis=0), np.std(fwers, axis=0)))

    json_file = portec_dict['results_dir'] / 'portec_table.json'
    with open(str(json_file), 'w', encoding='utf-8') as f:
        json.dump(json_out, f, ensure_ascii=False, indent=4)
