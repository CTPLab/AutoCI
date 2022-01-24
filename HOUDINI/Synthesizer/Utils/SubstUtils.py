from typing import Optional, Tuple, Union, List
from HOUDINI.Synthesizer import AST
from HOUDINI.Synthesizer.Utils import ASTUtils

"""
Invariant for substitution: no id on a lhs occurs in any term earlier in the list
"""

Substitution = List[Tuple[AST.PPSortOrDimVar, AST.PPSortOrDim]]


def substOne(a: AST.PPSortOrDimVar,
             b: AST.PPSortOrDim,
             sortTerm: AST.PPSort) -> AST.PPSort:
    """
    substitute b for all the occurrences of variable a in sortTerm
    """
    term1 = ASTUtils.applyTd(sortTerm,
                             lambda x: type(x) == AST.PPSortVar,
                             lambda x: b if x == a else x)
    term2 = ASTUtils.applyTd(term1,
                             lambda x: type(x) == AST.PPDimVar,
                             lambda x: b if x == a else x)
    return term2


def applySubst(subst: Substitution,
               sortTerm: AST.PPSort) -> AST.PPSort:
    """
    Apply a substitution right to left
    """
    curTerm = sortTerm
    for (a, b) in reversed(subst):
        curTerm = substOne(a, b, curTerm)
    return curTerm


def substProg(prog: AST.PPFuncApp,
              cmts: List[AST.PPSort]) -> Tuple[List[AST.PPFuncApp], bool]:
    newProgs = []
    sortVars = ASTUtils.getSortVars(prog)
    if sortVars:
        sortVar = sortVars[0]
        progress = True
        for cmt in cmts:
            newProg = substOne(sortVar, cmt, prog)
            newProgs.append(newProg)
    else:
        # Add the program as it is
        progress = False
        newProgs.append(prog)

    return newProgs, progress


def substProgList(progs: List[AST.PPFuncApp],
                  cmts: List[AST.PPSort]) -> Tuple[List[AST.PPSort], bool]:
    progsNext = []
    progress = False
    for prog in progs:
        newProgs, iProgress = substProg(prog, cmts)
        progress = progress or iProgress
        progsNext.extend(newProgs)

    return progsNext, progress


def substSortVar(prog: AST.PPFuncApp,
                 cmts: List[AST.PPSort],
                 maxSortVarsToBeSubst: int) -> List[AST.PPSort]:
    resProgs = [prog]
    progress = True
    i = 0
    while progress and i < maxSortVarsToBeSubst:
        resProgs, progress = substProgList(resProgs, cmts)
        i += 1
    return resProgs


def occursIn(sv: AST.PPSortVar,
             sort: AST.PPSort) -> bool:
    """
    Check if a variable occurs in a term
    """
    return ASTUtils.exists(sort, lambda x: x == sv)


def unifyOne(s: AST.PPSortOrDim,
             t: AST.PPSortOrDim) -> Optional[Substitution]:
    """ Unify a pair of terms
    """

    def case(ss, tt):
        return isinstance(s, ss) and isinstance(t, tt)

    res = None
    if case(AST.PPSortVar, AST.PPSortVar):
        if s == t:
            res = []
        else:
            res = [(s, t)]
    elif case(AST.PPDimVar, AST.PPDimVar):
        if s == t:
            res = []
        else:
            res = [(s, t)]
    elif case(AST.PPEnumSort, AST.PPEnumSort):
        if s == t:
            res = []
        else:
            # No unification for different enums
            None
    elif case(AST.PPSortVar, AST.PPSortTypes):
        if not occursIn(s, t):
            res = [(s, t)]
        else:
            res = None
    elif case(AST.PPDimVar, AST.PPDimConst):
        res = [(s, t)]
    elif case(AST.PPSortTypes, AST.PPSortVar):
        if not occursIn(t, s):
            res = [(t, s)]
        else:
            res = None
    elif case(AST.PPDimConst, AST.PPDimVar):
        res = [(t, s)]
    elif case(AST.PPInt, AST.PPInt) or \
            case(AST.PPReal, AST.PPReal) or \
            case(AST.PPBool, AST.PPBool) or \
            case(AST.PPImageSort, AST.PPImageSort):
        res = []
    elif case(AST.PPDimConst, AST.PPDimConst) and s == t:
        res = []
    elif case(AST.PPListSort, AST.PPListSort):
        res = unifyOne(s.param_sort, t.param_sort)
    elif case(AST.PPGraphSort, AST.PPGraphSort):
        res = unifyOne(s.param_sort, t.param_sort)
    elif case(AST.PPTensorSort, AST.PPTensorSort):
        res = unifyLists([s.param_sort] + s.shape, [t.param_sort] + t.shape)
    elif case(AST.PPFuncSort, AST.PPFuncSort):
        res = unifyLists(s.args + [s.rtpe], t.args + [t.rtpe])
    else:
        res = None

    return res


def unifyLists(xs: List[AST.PPSortOrDim], 
               ys: List[AST.PPSortOrDim]) -> Optional[Substitution]:
    if len(xs) == len(ys):
        pairs = list(zip(xs, ys))
        res = unify(pairs)
    else:
        res = None
    return res


def unify(pairs: List[Tuple[AST.PPSortOrDim, AST.PPSortOrDim]]) -> Optional[Substitution]:
    """
    Unify a list of pairs
    """
    res = None
    if not pairs:
        res = []
    else:
        (x, y) = pairs[0]
        t = pairs[1:]
        t2 = unify(t)
        if t2 is not None:
            t1 = unifyOne(applySubst(t2, x), applySubst(t2, y))
            if t1 is not None:
                res = t1 + t2
            else:
                res = None
        else:
            res = None
    return res


def main():
    prog1 = AST.PPFuncApp(
        fn=AST.PPVar(name='lib.compose'),
        args=[
            AST.PPTermUnk(name='nn_fun_csc4_2',
                          sort=AST.PPFuncSort(
                              args=[
                                  AST.PPSortVar(name='B')],
                              rtpe=AST.PPTensorSort(param_sort=AST.PPReal(), shape=[AST.PPDimConst(value=1), AST.PPDimConst(value=1)]))),
            AST.PPTermUnk(name='nn_fun_csc4_3',
                          sort=AST.PPFuncSort(
                              args=[AST.PPListSort(param_sort=AST.PPTensorSort(param_sort=AST.PPReal(),
                                                                               shape=[AST.PPDimConst(value=1), AST.PPDimConst(value=1),
                                                                                      AST.PPDimConst(
                                                                                   value=28),
                                  AST.PPDimConst(value=28)]))],
                              rtpe=AST.PPSortVar(name='B')))])

    prog2 = AST.PPFuncApp(
        fn=AST.PPVar(name='lib.compose'),
        args=[
            AST.PPTermUnk(name='nn_fun_csc4_4',
                          sort=AST.PPFuncSort(
                              args=[
                                  AST.PPSortVar(name='C')],
                              rtpe=AST.PPTensorSort(param_sort=AST.PPReal(), shape=[AST.PPDimConst(value=1), AST.PPDimConst(value=1)]))),
            AST.PPTermUnk(name='nn_fun_csc4_5',
                          sort=AST.PPFuncSort(
                              args=[AST.PPListSort(param_sort=AST.PPTensorSort(param_sort=AST.PPReal(),
                                                                               shape=[AST.PPDimConst(value=1), AST.PPDimConst(value=1),
                                                                                      AST.PPDimConst(
                                                                                   value=28),
                                  AST.PPDimConst(value=28)]))],
                              rtpe=AST.PPSortVar(name='C')))])

    prog3 = AST.PPFuncApp(
        fn=AST.PPVar(name='lib.compose'),
        args=[prog1, prog2])

    prog4 = AST.PPFuncApp(fn=AST.PPVar(name='lib.compose'), args=[AST.PPTermUnk(name='nn_fun_csc4_8', sort=AST.PPFuncSort(
        args=[AST.PPListSort(param_sort=AST.PPSortVar(name='B_1'))],
        rtpe=AST.PPTensorSort(param_sort=AST.PPReal(), shape=[AST.PPDimConst(value=1), AST.PPDimConst(value=1)]))),
        AST.PPFuncApp(fn=AST.PPVar(name='lib.map_l'), args=[
            AST.PPTermUnk(name='nn_fun_csc4_9', sort=AST.PPFuncSort(args=[
                AST.PPTensorSort(param_sort=AST.PPReal(),
                                 shape=[AST.PPDimConst(value=1),
                                        AST.PPDimConst(
                                     value=1),
                    AST.PPDimConst(
                                     value=28),
                    AST.PPDimConst(value=28)])],
                rtpe=AST.PPSortVar(
                name='B_1')))])])
    cmts = [AST.PPInt(), AST.PPReal()]
    eprogs = substSortVar(prog1, cmts, 8)
    for i, eprog in enumerate(eprogs):
        print(i, eprog)


if __name__ == '__main__':
    main()
