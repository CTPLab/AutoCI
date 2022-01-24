from typing import List

from HOUDINI.Synthesizer import AST
from HOUDINI.Library.FnLibrary import FnLibrary, PPLibItem
from HOUDINI.Library import Op
from HOUDINI.Synthesizer.AST import mkFuncSort, mkListSort


class OpLibrary(FnLibrary):
    def __init__(self, ops):
        self.items = dict()
        self.A = AST.PPSortVar('A')
        self.B = AST.PPSortVar('B')
        self.C = AST.PPSortVar('C')

        self.addOpItems(ops)

    def func(self, *lst):
        return mkFuncSort(*lst)

    def lst(self, t):
        return mkListSort(t)

    def addOpItem(self, op: str):
        if op == 'do':
            self.addItem(PPLibItem('do', self.func(self.func(self.A,
                                                             self.A),
                                                   self.func(self.lst(self.A),
                                                             self.lst(self.A))), Op.pp_do))
        elif op == 'compose':
            self.addItem(PPLibItem('compose', self.func(self.func(self.B,
                                                                  self.C),
                                                        self.func(self.A,
                                                                  self.B),
                                                        self.func(self.A,
                                                                  self.C)), Op.pp_compose))
        elif op == 'repeat':
            self.addItem(PPLibItem('repeat', self.func(AST.PPEnumSort(2, 4),
                                                       self.func(self.A,
                                                                 self.A),
                                                       self.func(self.A,
                                                                 self.A)), Op.pp_repeat))
        elif op == 'cat':
            self.addItem(PPLibItem('cat', self.func(self.func(self.lst(self.A),
                                                              self.lst(self.B)),
                                                    self.func(self.lst(self.A),
                                                              self.B)), Op.pp_cat))
        elif op == 'map':
            self.addItem(PPLibItem('map', self.func(self.func(self.A,
                                                              self.B),
                                                    self.func(self.lst(self.A),
                                                              self.lst(self.B))), Op.pp_map))
        elif op == 'conv':
            self.addItem(PPLibItem('conv', self.func(self.func(self.A,
                                                               self.B),
                                                     self.func(self.lst(self.A),
                                                               self.lst(self.B))), Op.pp_conv))
        elif op == 'fold':
            self.addItem(PPLibItem('fold', self.func(self.func(self.A,
                                                               self.B),
                                                     self.func(self.lst(self.A),
                                                               self.lst(self.B))), Op.pp_reduce))
        else:
            raise NameError(
                'Op name {} does not have corresponding function'.format(op))

    def addOpItems(self, ops: List[str]):
        for op in ops:
            self.addOpItem(op)
