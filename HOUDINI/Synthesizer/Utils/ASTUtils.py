from typing import Callable, Optional, Dict, Tuple, Union, List
from HOUDINI.Synthesizer import AST 
from HOUDINI.Synthesizer.Utils import MiscUtils, SubstUtils

TreeNodeSort = Union[AST.PPTerm, AST.PPSort, AST.PPVarDecl, tuple, list]
ntNameGen = MiscUtils.getUniqueFn()


def identity(term: AST.PPTerm):
    return construct(term, deconstruct(term))


def deconstruct(obj) -> List[TreeNodeSort]:
    if isinstance(obj, (*AST.PPTermTypes, list, tuple)):
        return list(obj.__iter__())
    else:
        return []


def construct(obj: TreeNodeSort,
              childs: List[object]) -> TreeNodeSort:
    def case(sort):
        return type(obj) == sort

    old_childs = deconstruct(obj)

    modified = False
    if len(childs) == len(old_childs):
        for i in range(len(childs)):
            if id(childs[i]) != id(old_childs[i]):
                modified = True
                break
    else:
        modified = True

    if not modified:
        return obj

    if case(AST.PPIntConst):
        return AST.PPIntConst(childs[0])
    elif case(AST.PPRealConst):
        return AST.PPRealConst(childs[0])
    elif case(AST.PPBoolConst):
        return AST.PPBoolConst(childs[0])
    elif case(AST.PPListTerm):
        return AST.PPListTerm(childs[0])
    elif case(AST.PPLambda):
        return AST.PPLambda(childs[0], childs[1])
    elif case(AST.PPFuncApp):
        return AST.PPFuncApp(childs[0], childs[1])
    elif case(AST.PPVar):
        return AST.PPVar(childs[0])
    elif case(AST.PPTermNT):
        return AST.PPTermNT(childs[0], childs[1])
    elif case(AST.PPTermUnk):
        return AST.PPTermUnk(childs[0], childs[1])
    # Sorts
    elif case(AST.PPInt):
        return AST.PPInt()
    elif case(AST.PPBool):
        return AST.PPBool()
    elif case(AST.PPReal):
        return AST.PPReal()
    elif case(AST.PPListSort):
        return AST.PPListSort(childs[0])
    elif case(AST.PPGraphSort):
        return AST.PPGraphSort(childs[0])
    elif case(AST.PPTensorSort):
        return AST.PPTensorSort(childs[0], childs[1])
    elif case(AST.PPFuncSort):
        return AST.PPFuncSort(childs[0], childs[1])
    elif case(AST.PPSortVar):
        return AST.PPSortVar(childs[0])
    # Other
    elif case(AST.PPVarDecl):
        return AST.PPVarDecl(childs[0], childs[1])
    # Python
    elif case(tuple):
        return tuple(childs)
    elif case(list):
        return childs
    else:
        raise Exception('Unhandled type in construct: {}: '.
                        format(type(obj)))


def deconstructProg(prog: AST.PPTerm):
    """
    Infers type of 'prog' in bottom-up fashion.
    """
    if isinstance(prog, AST.PPTermUnk):
        return []
    elif isinstance(prog, AST.PPVar):
        return []
    elif isinstance(prog, AST.PPFuncApp):
        return prog.args
    elif isinstance(prog, AST.PPIntConst):
        return []
    elif isinstance(prog, AST.PPRealConst):
        return []
    elif isinstance(prog, AST.PPBoolConst):
        return []
    else:
        raise NotImplementedError()


def constructProg(prog: AST.PPTerm, newChilds):
    if isinstance(prog, AST.PPTermUnk):
        return prog
    elif isinstance(prog, AST.PPVar):
        return prog
    elif isinstance(prog, AST.PPFuncApp):
        oldChilds = prog.args

        modified = False
        if len(newChilds) == len(oldChilds):
            for newChild, oldChild in zip(newChilds, oldChilds):
                if id(newChild) != id(oldChild):
                    modified = True
                    break
        else:
            modified = True

        if not modified:
            return prog
        else:
            return AST.PPFuncApp(prog.fn, newChilds)

    elif isinstance(prog, AST.PPIntConst):
        return prog
    elif isinstance(prog, AST.PPRealConst):
        return prog
    elif isinstance(prog, AST.PPBoolConst):
        return prog
    else:
        raise NotImplementedError()


# apply func
def applyTd(term: TreeNodeSort,
            cond: Callable[[TreeNodeSort], bool],
            func: Callable[[TreeNodeSort], TreeNodeSort]) -> TreeNodeSort:
    if cond(term):
        # Found the term
        res = func(term)
    else:
        # Term not found. Go down the tree
        childs = deconstruct(term)
        newChilds = [applyTd(c, cond, func) for c in childs]
        newTerm = construct(term, newChilds)
        res = newTerm

    return res


def applyTdOnce(term: TreeNodeSort,
                cond: Callable[[TreeNodeSort], bool],
                func: Callable[[TreeNodeSort], TreeNodeSort]) -> TreeNodeSort:
    applied = False

    def applyTdOnceRec(termRec: TreeNodeSort) -> TreeNodeSort:
        nonlocal applied

        if applied:
            return termRec

        if cond(termRec):  # Found the term
            res = func(termRec)
            applied = True
        else:
            # Term not found. Go down the tree
            childs = deconstruct(termRec)
            newChilds = [applyTdOnceRec(c) for c in childs]
            newTerm = construct(termRec, newChilds)
            res = newTerm

        return res

    return applyTdOnceRec(term)


def applyTdProg(term: AST.PPTerm,
                cond: Callable[[AST.PPTerm], bool],
                func: Callable[[AST.PPTerm], AST.PPTerm]) -> AST.PPTerm:
    if cond(term):
        # Found the term
        res = func(term)
    else:
        # Term not found. Go down the tree
        childs = deconstructProg(term)
        newChilds = [applyTdProg(c, cond, func) for c in childs]
        newTerm = constructProg(term, newChilds)
        res = newTerm

    return res


def applyTdProgGeneral(term: AST.PPTerm,
                       func: Callable[[AST.PPTerm], AST.PPTerm]) -> AST.PPTerm:
    res = None
    newTerm = func(term)
    if newTerm is None:  # process subterms
        childs = deconstructProg(term)
        newChilds = [applyTdProgGeneral(c, func) for c in childs]
        newTerm = constructProg(term, newChilds)
        res = newTerm
    elif newTerm is not None:  # replace term and proceed to sibling
        res = newTerm

    return res


# predicate func
def exists(term: TreeNodeSort,
           cond: Callable[[TreeNodeSort], bool]) -> bool:
    if cond(term):  # Term found.
        res = True
    else:  # Term not found. Go down the tree
        childs = deconstruct(term)
        res = any(exists(c, cond) for c in childs)

    return res


def isNT(t) -> bool:
    return isinstance(t, AST.PPNTTypes)


def isUnk(t) -> bool:
    return isinstance(t, AST.PPTermUnk)


def isOpen(term: AST.PPTerm) -> bool:
    return exists(term, lambda x: type(x) == AST.PPTermNT)


def hasUnk(term: AST.PPTerm) -> bool:
    return exists(term, lambda x: type(x) == AST.PPTermUnk)


def hasRedundantLambda(term: AST.PPTerm) -> bool:
    def isRedundantLambda(aTerm):
        if type(aTerm) == AST.PPLambda:
            body = aTerm.body
            paramNames = [p.name for p in aTerm.params]

            if len(aTerm.params) == 1:
                paramName = paramNames[0]
                if type(body) == AST.PPVar and body.name == paramName:
                    # dbgPrint("Ignored Lambda: %s" % ReprUtils.repr_py(aTerm))
                    return True

            if type(body) == AST.PPVar and body.name not in paramNames:
                # dbgPrint("Ignored Lambda: %s" % ReprUtils.repr_py(aTerm))
                return True

        return False

    return exists(term, lambda x: isRedundantLambda(x))


def isNthNT(nt_id: int) -> Callable[[AST.PPTerm], bool]:
    nt_cnt = 0

    def cond(term: AST.PPTerm):
        nonlocal nt_cnt
        if type(term) == AST.PPTermNT:
            nt_cnt += 1
            if nt_cnt == nt_id:
                return True
        return False

    return cond


def isAbstract(sort: AST.PPSort) -> bool:
    abstract = False

    def query(x):
        nonlocal abstract
        abstract = True
        return x

    applyTd(sort, lambda x: type(x) == AST.PPDimVar or type(x)
            == AST.PPSortVar or type(x) == AST.PPEnumSort, query)

    return abstract


# def isArgumentOfFun(term: AST.PPTerm,
#                     fName: str,
#                     unkArgName: str) -> bool:
#     """ Checks if the the term has an occurrance
#         where "unkArgName" appears as a direct argument of application of fName
#     """

#     def cond(sterm):
#         if type(sterm) == AST.PPFuncApp:
#             cond1 = sterm.fn == AST.PPVar(fName)
#             cond2 = any(map(lambda x: type(x) ==
#                             AST.PPTermUnk and x.name == unkArgName, sterm.args))
#             return cond1 and cond2
#         else:
#             return False

#     return exists(term, cond)


# def occursIn(sv: AST.PPSortVar,
#              sort: AST.PPSort) -> bool:
#     """
#     Check if a variable occurs in a term
#     """
#     return exists(sort, lambda x: x == sv)


# retrieve func
def getNthNT(term: AST.PPTerm, n: int) -> Optional[AST.PPTermNT]:
    count = 0
    res = None

    def getNthNTRec(termRec: AST.PPTerm):
        nonlocal count, res

        if isNT(termRec):
            count += 1
            if count == n:
                res = termRec
        else:
            for c in deconstruct(termRec):
                getNthNTRec(c)
                if count >= n:
                    break

    getNthNTRec(term)

    return res


def getNTs(term: AST.PPTerm) -> List[AST.PPTermNT]:
    if isinstance(term, AST.PPNTTypes):
        return [term]
    else:
        nts = []
        for c in deconstruct(term):
            cnts = getNTs(c)
            nts.extend(cnts)
        return nts


# def getSortVars(term: AST.PPTerm) -> List[str]:
#     """
#     Returns all SortVars
#     """
#     if isinstance(term, AST.PPSortVar):
#         return [term.name]
#     else:
#         sortVars = []
#         for c in deconstruct(term):
#             cvars = getSortVars(c)
#             sortVars.extend(cvars)
#         return sortVars


def getSortVars(sort: AST.PP) -> List[AST.PPSortVar]:
    svs = []

    def query(x):
        nonlocal svs
        if x not in svs:
            svs.append(x)
        return x

    applyTd(sort, lambda x: type(x) == AST.PPSortVar, query)

    return svs


def getDimVars(sort: AST.PP) -> List[AST.PPSort]:
    svs = []

    def query(x):
        nonlocal svs
        if x not in svs:
            svs.append(x)
        return x

    applyTd(sort, lambda x: type(x) == AST.PPDimVar, query)

    return svs


def getUnks(term: AST.PPTerm) -> List[AST.PPTermUnk]:
    if isinstance(term, AST.PPTermUnk):
        return [term]
    else:
        nts = []
        for c in deconstruct(term):
            cnts = getUnks(c)
            nts.extend(cnts)
        return nts


def getNumNTs(term: AST.PPTerm) -> int:
    return len(getNTs(term))


def getSize(term: AST.PPTerm) -> int:
    if type(term) in AST.PPTermTypes:
        size = 1
    else:
        size = 0

    for c in deconstruct(term):
        if type(c) in AST.PPTermTypes or type(c) is list or type(c) is tuple:
            size += getSize(c)

    return size


# assign func
def giveUniqueNamesToNTs(st: AST.PPTerm) -> TreeNodeSort:
    def rename(nt: AST.PPTermNT):
        return AST.PPTermNT("unk_{}".format(ntNameGen()), nt.sort)

    return applyTd(st, isNT, rename)


def giveUniqueNamesToUnks(st: AST.PPTerm) -> TreeNodeSort:
    def rename(nt: AST.PPTermUnk):
        return AST.PPTermUnk("nn_fun_{}".format(ntNameGen()), nt.sort)

    return applyTd(st, isUnk, rename)


def getNTNameSortMap(term: AST.PPTerm) -> Dict[str, AST.PPSort]:
    retMap = {}

    def id1(nt: AST.PPTermNT):
        nonlocal retMap
        retMap[nt.name] = nt.sort
        return nt

    applyTd(term, isNT, id1)

    return retMap


def getUnkNameSortMap(term: AST.PPTerm) -> Dict[str, AST.PPSort]:
    retMap = {}

    def id1(nt: AST.PPTermUnk):
        nonlocal retMap
        retMap[nt.name] = nt.sort
        return nt

    applyTd(term, isUnk, id1)

    return retMap


def replaceAllSubTerms(term: AST.PP,
                       subTerm: AST.PP,
                       newSubTerm: AST.PP) -> TreeNodeSort:
    def action(st):
        return newSubTerm

    def cond(st):
        return st == subTerm

    newTerm = applyTd(term, cond, action)
    return newTerm


def alphaConvertSorts(sortsA: AST.PP,
                      sortsC: AST.PP) -> TreeNodeSort:
    """
    Renames sort and dim variables in 'sortsA'
    to avoid clash with sort and dim variables in sorts
    TODO: Also alphaconvert dimensions.
    """
    def getSortVarsMulti(sortList):
        res = []

        for s in sortList:
            sVars = getSortVars(s)
            res.extend(sVars)

        res = list(set(res))
        return res

    svsA = getSortVarsMulti(sortsA)
    svsC = getSortVarsMulti(sortsC)

    aMap = {}
    svsB = []
    for sva in svsA:
        if sva in svsC:
            newSV = sva
            i = 0
            while newSV in svsA or newSV in svsB or newSV in svsC:
                newSV = AST.PPSortVar(sva.name + str(i))
                i += 1

            aMap[sva] = newSV
            svsB.append(newSV)

    # print(aMap)

    newSortsA = []
    for sa in sortsA:
        nsa = sa
        for key, value in aMap.items():
            nsa = SubstUtils.substOne(key, value, nsa)
        newSortsA.append(nsa)

    return newSortsA


# @logInferType
def inferType(prog: AST.PPTerm, lib) -> Optional[AST.PPSort]:
    """
    Infers type of 'prog' in bottom-up fashion.
    Only works for concrete leaf node types.
    """
    if isinstance(prog, AST.PPTermUnk):
        return prog.sort
    elif isinstance(prog, AST.PPVar):
        varName = prog.name.replace('lib.', '')
        varSort = lib.get(varName).sort
        return varSort
    elif isinstance(prog, AST.PPFuncApp):
        args = prog.args

        fnName = prog.fn.name.replace('lib.', '')
        li = lib.get(fnName)

        fnSort = li.sort
        argSorts = fnSort.args
        rtpe = fnSort.rtpe
        argSortsConcrete = []
        for arg, argSort in zip(args, argSorts):
            if isinstance(argSort, AST.PPDimVar):
                if not isinstance(arg, AST.PPIntConst):
                    print('prog: ', prog)
                    print('arg: ', arg)
                    raise ValueError(
                        'DimVar arg is not of type AST.PPIntConst')

                ct = AST.PPDimConst(arg.value)
            elif isinstance(argSort, AST.PPEnumSort):
                if not isinstance(arg, AST.PPIntConst):
                    print('prog: ', prog)
                    print('arg: ', arg)
                    raise ValueError('Enum arg is not of type AST.PPIntConst')
                ct = argSort
            else:
                ct = inferType(arg, lib)

            argSortsConcrete.append(ct)

        # Rename argument sort vars to avoid conflict
        sortsToRename = list(argSorts)
        sortsToRename.append(rtpe)
        renamedSorts = alphaConvertSorts(sortsToRename, argSortsConcrete)
        argSorts = renamedSorts[:-1]
        rtpe = renamedSorts[-1]

        subst = SubstUtils.unifyLists(argSorts, argSortsConcrete)
        if subst is not None:
            concreteRtpe = SubstUtils.applySubst(subst, rtpe)
        else:
            concreteRtpe = None
        return concreteRtpe
    elif isinstance(prog, AST.PPIntConst):
        return AST.PPInt()
    elif isinstance(prog, AST.PPRealConst):
        return AST.PPReal()
    elif isinstance(prog, AST.PPBoolConst):
        return AST.PPBool()
    elif isinstance(prog, AST.PPListTerm):
        raise NotImplementedError()
    else:
        raise NotImplementedError()


# get sizes of the prog
def progTreeSize(prog: AST.PPTerm) -> int:
    """
    Size of the 'prog' where only AST.PPTerm nodes are counted.
    """
    if isinstance(prog, AST.PPFuncApp):
        args = prog.args
        size = 1 + sum([progTreeSize(arg) for arg in args])
        return size
    elif isinstance(prog, AST.PPTermUnk):
        return 1
    elif isinstance(prog, AST.PPVar):
        return 1
    elif isinstance(prog, AST.PPIntConst):
        return 1
    elif isinstance(prog, AST.PPRealConst):
        return 1
    elif isinstance(prog, AST.PPBoolConst):
        return 1
    elif isinstance(prog, AST.PPListTerm):
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def progDepth(prog: AST.PPTerm) -> int:
    if isinstance(prog, AST.PPFuncApp):
        args = prog.args
        depth = 1 + max([progDepth(arg) for arg in args])
        return depth
    elif isinstance(prog, AST.PPTermUnk):
        return 1
    elif isinstance(prog, AST.PPVar):
        return 1
    elif isinstance(prog, AST.PPIntConst):
        return 1
    elif isinstance(prog, AST.PPRealConst):
        return 1
    elif isinstance(prog, AST.PPBoolConst):
        return 1
    elif isinstance(prog, AST.PPListTerm):
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def main():
    pass


if __name__ == '__main__':
    main()
