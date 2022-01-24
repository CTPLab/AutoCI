import logging
import time
import traceback
from typing import NamedTuple, Tuple, Dict, List

from HOUDINI.Library.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Interpreter.Interpreter import Interpreter
from HOUDINI.Interpreter.Utils.EvalUtils import is_evaluable
from HOUDINI.Synthesizer.AST import PPTerm, PPSort, PPFuncSort
from HOUDINI.Synthesizer.Utils.MiscUtils import getElapsedTime, formatTime
from HOUDINI.Synthesizer.Utils.ReprUtils import repr_py
from HOUDINI.Synthesizer.SymbolicSynthesizer import SymbolicSynthesizer


NeuralSynthesizerSettings = NamedTuple("NeuralSynthesizerSettings", [
    ('N', int),  # Generate at most N programs.
    ('M', int),  # Evaluate at most M programs.
    ('K', int),  # Return top-k programs
])


class NSDebugInfo:
    def __init__(self, dprog):
        self.dprog = dprog


def _debug_info(prog: PPTerm,
                unkSortMap,
                lib: FnLibrary,
                fnSort: PPSort):
    lib_items = [PPLibItem(li.name, li.sort, None)
                 for (_, li) in lib.items.items()]
    dprog = """
    io_examples_trn, io_examples_val = None, None
    prog = %s
    unkSortMap = %s
    lib = NewLibrary()
    lib.addItems(%s)
    fn_sort = %s
    interpreter = Interpreter(lib, epochs=1)
    res = interpreter.evaluate(program=prog,
                                        output_type_s=fn_sort.rtpe,
                                        unkSortMap=unkSortMap,
                                        io_examples_trn=io_examples_trn,
                                        io_examples_val=io_examples_val)
    """ % (str(prog), str(unkSortMap), str(lib_items), str(fnSort))

    return NSDebugInfo(dprog),


class NeuralSynthesizerResult:
    def __init__(self,
                 top_k_solutions_results: List[Tuple[PPTerm, Dict]]):
        # A list of top scoring programs
        self.top_k_solutions_results = top_k_solutions_results

    def get_top_solution_score(self):
        """Top scoring program and corresponding score
        """
        top_solution_score = None
        if len(self.top_k_solution_scores):
            top_solution_score = self.top_k_solution_scores[0]
        return top_solution_score


class NeuralSynthesizer:
    def __init__(self,
                 interpreter: Interpreter,
                 synthesizer: SymbolicSynthesizer,
                 lib: FnLibrary,
                 sort: PPFuncSort,
                 settings):
        self.interpreter = interpreter
        self.synthesizer = synthesizer
        self.lib = lib
        self.sort = sort
        self.settings = settings
        self.prog_unkinfo_tuples = []

        self.evaluated_programs_str = []
        self.evaluated_programs_type_info = []

        self.init_progs()

    def init_progs(self):
        n = 0
        m = 0
        pStart = time.time()
        print('BEGIN_PROGRAM_GENERATION, Time: {}'.format(getElapsedTime()))
        for prog, unkSortMap in self.synthesizer.genProgs():
            n += 1
            if n % 100 == 0:
                print('.', end='', flush=True)

            if n > self.settings.N or m >= self.settings.M:
                break

            c_program_str = repr_py(prog)

            try:
                is_ok, ecode = is_evaluable(prog, self)
            except Exception as e:
                self.log_isevaluable_exception(e, prog, unkSortMap)
                continue

            if is_ok:
                self.evaluated_programs_str.append(c_program_str)
                self.evaluated_programs_type_info.append(str(prog))
                self.prog_unkinfo_tuples.append((prog, unkSortMap))
                m += 1

        print('END_PROGRAM_GENERATION, Time: {}'.format(getElapsedTime()))
        pEnd = time.time()
        print('TIME_TAKEN_SYNTH, {}'.format(formatTime(pEnd - pStart)))

    def interpret(self,
                  prog,
                  unkSortMap,
                  io_examples_trn,
                  io_examples_val,
                  io_examples_tst) -> Dict:
        output_type = self.sort.rtpe
        print('BEGIN_EVALUATE, Time: {}'.format(getElapsedTime()))
        eStart = time.time()
        res = self.interpreter.evaluate(program=prog,
                                        output_type_s=output_type,
                                        unkSortMap=unkSortMap,
                                        io_examples_trn=io_examples_trn,
                                        io_examples_val=io_examples_val,
                                        io_examples_tst=io_examples_tst)
        print('END_EVALUATE, Time: {}'.format(getElapsedTime()))
        eEnd = time.time()
        print('TIME_TAKEN_EVALUATE, {}'.format(formatTime(eEnd - eStart)))

        return res

    def update_top_k(self,
                     top_k_solutions_results: List[Tuple[PPTerm, Dict]]):
        top_k_solutions_results.sort(key=lambda x: x[1]['accuracy'],
                                     reverse=True)

        if len(top_k_solutions_results) > self.settings.K:
            del top_k_solutions_results[-1]

        for i in range(1, top_k_solutions_results.__len__()):
            top_k_solutions_results[i][1]["new_fns_dict"] = None

    def solve(self,
              io_examples_trn,
              io_examples_val,
              io_examples_tst) -> List[Tuple[PPTerm, float]]:

        top_k_solutions_results = []
        for prog, unkSortMap in self.prog_unkinfo_tuples:
            try:
                interpreter_res = self.interpret(prog,
                                                 unkSortMap,
                                                 io_examples_trn,
                                                 io_examples_val,
                                                 io_examples_tst)
            except Exception as e:
                e.args += _debug_info(prog,
                                      unkSortMap,
                                      self.lib,
                                      self.sort)
                traceback.print_exc()
                self.log_evaluator_exception(e, prog, unkSortMap)
                continue

            top_k_solutions_results.append((prog, interpreter_res))
            self.update_top_k(top_k_solutions_results)

        print('Exiting NeuralSynthesizer.solve(). '
              'The following programs were evaluated:')
        for idx, program_str in enumerate(self.evaluated_programs_str):
            print(program_str)
            print(self.evaluated_programs_type_info[idx])
            print('..........................')

        return NeuralSynthesizerResult(top_k_solutions_results)

    def log_evaluated_program(self, prog):
        print('Program evaluated: {}'.format(repr_py(prog)))

    def log_rejected_program(self, prog, ecode):
        print('Program rejected (ecode {}): {}'.format(ecode, repr_py(prog)))

    def log_unhandled_program(self, prog):
        print('Program not handled: {}'.format(repr_py(prog)))

    def log_evaluator_exception(self, e, prog, unkSortMap):
        loggerE = logging.getLogger('pp.exceptions')
        loggerE.error('Exception in the Interpreter.\n {}'.format(repr(e)))
        debug_info = _debug_info(prog,
                                 unkSortMap, self.lib, self.sort)[0].dprog
        loggerE.error('DebugInfo.\n {}'.format(debug_info))

    def log_isevaluable_exception(self, e, prog, unkSortMap):
        loggerE = logging.getLogger('pp.exceptions')
        e.args += _debug_info(prog, unkSortMap, self.lib, self.sort)
        loggerE.error('Exception in the Interpreter.\n {}'.format(repr(e)))
