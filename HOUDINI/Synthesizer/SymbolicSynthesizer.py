from queue import PriorityQueue
from typing import Iterable, Tuple, Dict, NamedTuple, List

from HOUDINI.Library.FnLibrary import FnLibrary
from HOUDINI.Synthesizer import AST
from HOUDINI.Synthesizer.BaseSynthesizer import BaseSynthesizer
from HOUDINI.Synthesizer.Utils import ASTUtils, RuleUtils, MiscUtils, SubstUtils

Action = NamedTuple("Action", [('ntId', int), ('ruleId', int)])


class SymbolicSynthesizer(BaseSynthesizer[AST.PPTerm, Action]):
    def __init__(self,
                 lib: FnLibrary,
                 sort: AST.PPFuncSort,
                 nnprefix='',
                 concreteTypes: List[AST.PPSort] = []):

        self.lib = lib
        self.sort = sort
        self._ntNameGen = MiscUtils.getUniqueFn()
        self.nnprefix = nnprefix
        self.concreteTypes = concreteTypes

    def _giveUniqueNamesToUnks(self, st: AST.PPTerm):
        def rename(nt: AST.PPTermUnk):
            return AST.PPTermUnk('nn_fun_{}_{}'.format(self.nnprefix,
                                                       self._ntNameGen()),
                                 nt.sort)

        return ASTUtils.applyTd(st, ASTUtils.isUnk, rename)

    def start(self) -> AST.PPTerm:
        return AST.PPTermNT('Z', self.sort)

    def setEvaluate(self, evaluate: bool):
        self.evaluate = evaluate

    def filterState(self, st: AST.PPTerm):
        if ASTUtils.hasRedundantLambda(st):
            return False
        return True

    def getNextStates(self, st: AST.PPTerm, action: Action) -> List[AST.PPTerm]:
        rule = RuleUtils.getRule(action.ruleId)
        nextSts = rule(self.lib, st, action.ntId)
        nextSts = list(filter(self.filterState, nextSts))
        return nextSts

    def getActionsFirstNT(self, st: AST.PPTerm) -> List[Action]:
        actions = []
        for ruleId in range(len(RuleUtils.rules)):
            actions.append(Action(1, ruleId))
        return actions

    def getActions(self, st: AST.PPTerm) -> List[Action]:
        return self.getActionsFirstNT(st)

    def getActionCost(self, st: AST.PPTerm, action: Action) -> float:
        return ASTUtils.getSize(st)

    def isOpen(self, st: AST.PPTerm) -> bool:
        return ASTUtils.isOpen(st)

    def hasUnk(self, st: AST.PPTerm) -> bool:
        return ASTUtils.hasUnk(st)

    def onEachIteration(self, st: AST.PPTerm, action: Action):
        return None

    def exit(self) -> bool:
        return False

    def genTerms(self) -> Iterable[AST.PPTerm]:
        sn = MiscUtils.getUniqueFn()
        pq = PriorityQueue()

        def addToPQ(aState):
            for cAction in self.getActions(aState):
                stateActionScore = self.getActionCost(aState, cAction)
                pq.put((stateActionScore, sn(), (aState, cAction)))

        state = self.start()
        addToPQ(state)

        while not pq.empty() and not self.exit():
            _, _, (state, action) = pq.get()

            self.onEachIteration(state, action)

            states = self.getNextStates(state, action)

            for state in states:
                if self.isOpen(state):
                    addToPQ(state)
                yield state

    def genProgs(self) -> Iterable[Tuple[AST.PPTerm, Dict[str, AST.PPSort]]]:
        for prog in self.genTerms():
            if self.isOpen(prog):
                continue

            if self.concreteTypes:
                maxSortVarsToBeInstantiated = 2
                eprogs = SubstUtils.substSortVar(
                    prog, self.concreteTypes, maxSortVarsToBeInstantiated)
            else:
                eprogs = [prog]

            for eprog in eprogs:
                unkSortMap = {}
                if self.hasUnk(eprog):
                    eprog = self._giveUniqueNamesToUnks(eprog)
                    unkSortMap = ASTUtils.getUnkNameSortMap(eprog)

                yield eprog, unkSortMap


def main():
    pass


if __name__ == '__main__':
    main()
