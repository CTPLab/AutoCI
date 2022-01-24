from typing import List, TypeVar, Generic, Iterable

S = TypeVar('S')
A = TypeVar('A')


class BaseSynthesizer(Generic[S, A]):
    def start(self) -> S:
        pass

    def getNextStates(self, st: S, action: A) -> List[S]:
        pass

    def getActions(self, st: S) -> List[A]:
        pass

    def getActionCost(self, st: S, action: A) -> float:
        pass

    def isOpen(self, st: S) -> bool:
        pass

    def onEachIteration(self, st: S, action: A):
        pass

    def getScore(self, st: S) -> float:
        pass

    def exit(self) -> bool:
        pass

    def isEvaluable(self) -> bool:
        pass

    def genTerms(self) -> Iterable[S]:
        pass
