
import time
import pathlib
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple, Tuple, List, Dict

from HOUDINI.Run.Utils import mk_tag, mk_div, iterate_diff_training_sizes
from HOUDINI.Interpreter.Interpreter import Interpreter
from HOUDINI.Library.OpLibrary import OpLibrary
from HOUDINI.Synthesizer.AST import PPSort, PPTerm, mkRealTensorSort
from HOUDINI.Synthesizer.Utils.ReprUtils import repr_py
from HOUDINI.Synthesizer.Utils.MiscUtils import getElapsedTime, formatTime
from HOUDINI.Synthesizer.SymbolicSynthesizer import SymbolicSynthesizer
from HOUDINI.Synthesizer.NeuralSynthesizer import NeuralSynthesizerSettings, NeuralSynthesizer, NSDebugInfo

matplotlib.use('agg')


class TaskResultSingle:
    """Task Result for a single datasize
    """

    def __init__(self):
        # top k solution-score pairs. descending order
        self.top_k_solutions_results: List[Tuple[PPTerm, Dict]] = []
        # Total number of programs evaluated
        self.num_programs: int = None
        # time taken in seconds
        self.time: int = None
        # genid --> progid --> (progstr, score)
        self.progScores: List[List[Tuple[str, float]]] = None

    def get_top_solution_details(self) -> Tuple[PPTerm, Dict]:
        if len(self.top_k_solutions_results) > 0:
            return self.top_k_solutions_results[0]
        else:
            return None

    def get_top_program(self) -> PPTerm:
        prog_res_dict = self.get_top_solution_details()
        if prog_res_dict is not None:
            return prog_res_dict[0]
        else:
            return None

    def get_top_score(self) -> float:
        prog_res_dict = self.get_top_solution_details()
        if prog_res_dict is not None:
            return prog_res_dict[1]['test_accuracy']
        else:
            return None

    def gen_report(self, task) -> str:
        header_row = mk_tag('tr',
                            mk_tag('th',
                                   'Score Test / Val') +
                            mk_tag('th',
                                   'Top {} Programs'.format(task.settings.K)))

        def mk_row(p, s_test, s_val):
            return mk_tag('tr',
                          mk_tag('td',
                                 '{:.4f} / {:.4f}'.format(s_test, s_val)) +
                          mk_tag('td', repr_py(p)))

        rows = []
        for p, rdict in self.top_k_solutions_results:
            test_acc, val_acc = -10000., -10000.
            if 'test_accuracy' in rdict:
                test_acc = rdict['test_accuracy']
            if 'accuracy' in rdict:
                val_acc = rdict['accuracy']
            rows.append(mk_row(p, test_acc, val_acc))

        table_content = '\n'.join([header_row] + rows)

        res = mk_tag('table',
                     table_content,
                     attribs={'border': '1'})

        def gen_prog_content():
            genReprs = []

            for i, iprogs in enumerate(self.progScores):
                c = mk_div('Generation: {}'.format(i))
                c += ''.join(mk_div('{:.2f} '.format(score + pstr),
                                    cls=['Prog']) for (pstr, score) in iprogs)
                genReprs.append(mk_div(c,
                                       cls=['Generation']))
            return ''.join(genReprs)

        if self.progScores:
            prog_content = gen_prog_content()
            res = res + mk_div(prog_content,
                               cls=['Programs'])
    def get_raw_data(self):
        res = dict()
        res['top_k_solutions_results'] = [(str(prog), res) for
                                          (prog, res) in self.top_k_solutions_results]
        res['num_programs'] = self.num_programs
        res['time'] = self.time
        res['progScores'] = self.progScores
        return res


class TaskResult:
    def __init__(self):
        # Result for various data sizes.
        self.results: List[TaskResultSingle] = []

    def save_plot(self, task, seq_dir):
        xs = np.array(task.settings.training_percentages)
        ys = np.array([single_result.get_top_score()
                       for single_result in self.results])

        plt.figure()
        plt.xlabel("Training Dataset Size")
        plt.ylabel("Accuracy")
        handles = []
        t_line = plt.plot(
            xs,
            ys,
            label=task.name(),
            marker='o')
        handles.append(t_line)

        img_path = pathlib.Path(seq_dir) / '{}.png'.format(task.name())
        plt.savefig(str(img_path))

        xydata = np.array([xs, ys])
        npy_path = pathlib.Path(seq_dir) / '{}_plot.npy'.format(task.name())
        np.save(str(npy_path), xydata)
        plt.close()

    def gen_report(self, task, tid: int) -> str:
        res = mk_tag('h2',
                     'Task {}: {}'.format(tid, task.name()))
        # data size - score plot
        res += mk_div(mk_tag('img',
                             '',
                             attribs={'src': task.name() + '.png'}))

        # Top k Programs and scores
        for task_result_single, percentage in zip(self.results,
                                                  task.settings.training_percentages):
            res += '<br>'
            res += mk_div('Training Data Used: {} %%'.format(percentage))
            res += mk_div('Number of programs evaluated: {}'.
                          format(task_result_single.num_programs))
            res += task_result_single.gen_report(task)
        return res

    def get_raw_data(self):
        return [result.get_raw_data() for result in self.results]


_tset = [
    # Training data size
    ('train_size', int),
    # Validation data size = Test data size
    ('val_size', int),
    # deprecate
    ('training_percentages', List[int]),
    # Max number of solutions to generate
    ('N', int),
    # Max number of solutions evaluated by the interpreter
    ('M', int),
    # Number of top solutions to store
    ('K', int),
    ('learning_rate', float),
    # the amount of causal candidates that need to be looped trough
    ('var_num', int),
    # warm-up epoches
    ('warm_up', int),
    ('lambda_1', float),
    ('lambda_2', float),
    ('lambda_cau', float),
    ('data_dict', dict),
]


class TaskSettings(NamedTuple('TaskSettings', _tset)):
    """A named tuple with default parameters
    """

    def __new__(cls, **kwargs):
        return super(TaskSettings, cls).__new__(cls, **kwargs)


class Task:
    def __init__(self,
                 fn_sort: PPSort,
                 settings: TaskSettings,
                 lib: OpLibrary):
        self.fn_sort = fn_sort
        self.settings = settings
        self.lib = lib
        assert self.lib is not None

    def name(self):
        return NotImplementedError()

    def sname(self):
        return NotImplementedError()

    def _mkNSynth(self):
        # deprecate evolutionary synthesizer
        # only use enumerative synthesizer
        interpreter = Interpreter(self.settings,
                                  self.lib)

        mid_size = self.settings.data_dict['mid_size']
        concreteTypes = [mkRealTensorSort([1, mid_size])]
        synth = SymbolicSynthesizer(self.lib,
                                    self.fn_sort,
                                    self.sname(),
                                    concreteTypes)

        ns_settings = NeuralSynthesizerSettings(self.settings.N,
                                                self.settings.M,
                                                self.settings.K)

        nsynth = NeuralSynthesizer(interpreter,
                                   synth,
                                   self.lib,
                                   self.fn_sort,
                                   ns_settings)
        return nsynth

    def run(self) -> TaskResult:
        tStart = time.time()
        print('BEGIN_TASK, Time: {}'.format(getElapsedTime()))

        nsynth = self._mkNSynth()
        print('Num of programs selected for evaluation: {}'.
              format(len(nsynth.prog_unkinfo_tuples)))

        print('Programs selected for evaluation:')
        for c_prog, c_unkSortMap in nsynth.prog_unkinfo_tuples:
            print(repr_py(c_prog))
            for key, val in c_unkSortMap.items():
                print(key, val)
            print('\n\n')

        train_io, val_io, test_io = self.get_io_examples()
        res = TaskResult()
        for i, (c_tr_io_examples, _) in enumerate(iterate_diff_training_sizes(train_io,
                                                                              self.settings.training_percentages)):
            rStart = time.time()
            print('BEGIN_RUN {}, Time: {}'.format(i, getElapsedTime()))

            c_res = TaskResultSingle()
            try:
                nsynth_res = nsynth.solve(c_tr_io_examples,
                                          val_io,
                                          test_io)

                c_res.top_k_solutions_results = nsynth_res.top_k_solutions_results
                c_res.num_programs = len(nsynth.prog_unkinfo_tuples)
                c_res.time = None  # TODO: implement

            except Exception as e:
                print('Exception in NeuralSynthesizer.solve: {}'.format(str(e)))
                print('# Task Name: {}'.format(self.name()))
                for a in e.args:
                    if isinstance(a, NSDebugInfo):
                        print(a.dprog)
                raise

            res.results.append(c_res)

            print('END_RUN {}, Time: {}'.format(i, getElapsedTime()))
            rEnd = time.time()
            print('TIME_TAKEN_RUN, {}'.format(formatTime(rEnd - rStart)))

        print('END_TASK, Time: {}'.format(getElapsedTime()))
        tEnd = time.time()
        print('TIME_TAKEN_TASK, {}'.format(formatTime(tEnd - tStart)))

        return res
