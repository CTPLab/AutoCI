from typing import Callable, Optional, List

from HOUDINI.Library.FnLibrary import PPLibItem, FnLibrary
from HOUDINI.Synthesizer import AST
from HOUDINI.Synthesizer.Utils import ASTUtils, SubstUtils, ReprUtils, ScopeUtils

# global gUseTypes
# gUseTypes = True


def getSortVarNotInTerm(sv, term):
    """
    get a sv that is not in term
    """
    i = 0
    while ASTUtils.exists(term, lambda x: x == sv):
        i = i + 1
        newsv = AST.PPSortVar(sv.name + '_' + str(i))
        sv = newsv

    return sv


def getDimVarNotInTerm(dv, term):
    """
    get a dv that is not in term
    """
    i = 0
    while ASTUtils.exists(term, lambda x: x == dv):
        i = i + 1
        newdv = AST.PPDimVar(dv.name + '_' + str(i))
        dv = newdv

    return dv


def renameSortVars(libItemSort, term):
    i = 0
    svs = ASTUtils.getSortVars(libItemSort)
    for sv in svs:
        newsv = getSortVarNotInTerm(sv, term)
        libItemSort = ASTUtils.replaceAllSubTerms(libItemSort, sv, newsv)

    return libItemSort


def renameDimVars(libItemSort, term):
    i = 0
    dvs = ASTUtils.getDimVars(libItemSort)
    for dv in dvs:
        newdv = getDimVarNotInTerm(dv, term)
        libItemSort = ASTUtils.replaceAllSubTerms(libItemSort, dv, newdv)

    return libItemSort


def alphaConvertLibItem(libItem, term):
    libItemSort = renameSortVars(libItem.sort, term)
    libItemSort = renameDimVars(libItemSort, term)

    libItem = PPLibItem(name=libItem.name, sort=libItemSort, obj=libItem.obj)

    return libItem


def expandToVar(lib: FnLibrary,
                term: AST.PPTerm,
                ntId: int,
                vname: str) -> Optional[AST.PPTerm]:
    """
    Generate a new term by replacing a "ntId"th NT from a "term" with a variable (in scope) with name "fname"
    """
    nt = ASTUtils.getNthNT(term, ntId)
    # libItem = ScopeUtils.getAVarInScope(lib, term, ntId, vname)
    libItem = lib.getWithLibPrefix(vname)
    assert libItem

    libItem = alphaConvertLibItem(libItem, term)
    subst = SubstUtils.unifyOne(libItem.sort, nt.sort)

    # if not gUseTypes:
    #     if subst is None:
    #         subst = []

    termExpanded = None
    if subst is not None:
        termUnified = SubstUtils.applySubst(subst, term)
        termExpanded = ReprUtils.replaceNthNT(
            termUnified, ntId, AST.PPVar(libItem.name))

    return termExpanded


def expandToFuncApp(lib: FnLibrary,
                    term: AST.PPTerm,
                    ntId: int,
                    fname: str) -> Optional[AST.PPTerm]:
    resTerm = None
    # Not needed now as there are no lambda terms
    # libItem = ScopeUtils.getAVarInScope(lib, term, ntId, fname)
    libItem = lib.getWithLibPrefix(fname)
    assert libItem
    nt = ASTUtils.getNthNT(term, ntId)

    libItem = alphaConvertLibItem(libItem, term)
    # print(nt.sort)
    # TODO: expandToVar passed nt.sort as second argument
    subst = SubstUtils.unifyOne(nt.sort, libItem.sort.rtpe)
    # print('subst {}'.format(subst))
    # print()

    # if not gUseTypes:
    #     if subst is None:
    #         subst = []

    if subst is not None:
        nts = [AST.PPTermNT('Z', arg_sort) for arg_sort in libItem.sort.args]
        fnApp = AST.PPFuncApp(AST.PPVar(libItem.name), nts)
        termUnified = SubstUtils.applySubst(subst, term)
        # print('{} \n{}'.format(termUnified, term))
        # print()
        fnAppUnified = SubstUtils.applySubst(subst, fnApp)
        # print('fnAppUnified {}'.format(fnAppUnified))
        # print()
        resTerm = ReprUtils.replaceNthNT(termUnified, ntId, fnAppUnified)
        # print('outterm {}'.format(resTerm))
        # print()
    return resTerm


def expandToUnk(term: AST.PPTerm,
                ntId: int) -> Optional[AST.PPTerm]:
    """
    Generate a new term by replacing a "ntId"th NT from a "term" with a AST.PPTermUnk
    """
    nt = ASTUtils.getNthNT(term, ntId)

    # Avoid generating Unk of type AST.PPDimConst, AST.PPDimVar, AST.PPEumSort, or AST.PPInt()
    if isinstance(nt.sort, AST.PPDimConst) or \
       isinstance(nt.sort, AST.PPDimVar) or \
       isinstance(nt.sort, AST.PPEnumSort) or \
       isinstance(nt.sort, AST.PPInt):
        return None

    unk = AST.mkUnk(nt.sort)

    # subst = unifyOne(unk.sort, nt.sort)
    #
    # if subst != []:
    #     print("non empty subst")
    #
    # termExpanded = None
    # if subst is not None:
    #     termUnified = applySubst(subst, term)
    #     termExpanded = ReprUtils.replaceNthNT(termUnified, ntId, unk)

    termNew = ReprUtils.replaceNthNT(term, ntId, unk)
    return termNew


def expandDimConst(term: AST.PPTerm,
                   ntId: int) -> Optional[AST.PPTerm]:
    """
    Expand dimension constant to integer constants (Required for fold zeros)
    """
    nt = ASTUtils.getNthNT(term, ntId)
    if type(nt.sort) != AST.PPDimConst:
        return None

    subTerm = AST.PPIntConst(nt.sort.value)
    termExpanded = ReprUtils.replaceNthNT(term, ntId, subTerm)
    return termExpanded


def expandEnum(term: AST.PPTerm,
               ntId: int,
               subTerm: AST.PPTerm) -> Optional[AST.PPTerm]:
    nt = ASTUtils.getNthNT(term, ntId)
    termExpanded = ReprUtils.replaceNthNT(term, ntId, subTerm)
    return termExpanded


def applyExpandToVar(lib: FnLibrary,
                     term: AST.PPTerm,
                     ntId: int) -> List[AST.PPTerm]:
    varDecls = ScopeUtils.getAllVarsInScope(lib, term, ntId)

    res = []
    for var in varDecls:
        nxtTerm = expandToVar(lib, term, ntId, var.name)
        if nxtTerm is not None:
            res.append(nxtTerm)

    return res


def applyExpandToUnk(lib: FnLibrary,
                     term: AST.PPTerm,
                     ntId: int) -> List[AST.PPTerm]:
    # Expand to Unk term
    res = []
    nxtTerm = expandToUnk(term, ntId)
    if nxtTerm is not None:
        res.append(nxtTerm)

    return res


def applyExpandToFuncApp(lib: FnLibrary,
                         term: AST.PPTerm,
                         ntId: int) -> List[AST.PPTerm]:
    varDecls = ScopeUtils.getAllFnVarsInScope(lib, term, ntId)

    res = []
    for var in varDecls:
        nxtTerm = expandToFuncApp(lib, term, ntId, var.name)
        if nxtTerm is not None:
            res.append(nxtTerm)

    # # Expand to unk func app.
    # nxtTerm = expandToUnkFuncApp(lib, term, ntId)
    # if nxtTerm is not None:
    #     res.append(nxtTerm)

    return res


def applyExpandEnum(lib: FnLibrary,
                    term: AST.PPTerm,
                    ntId: int) -> List[AST.PPTerm]:
    res = []

    nt = ASTUtils.getNthNT(term, ntId)
    if type(nt.sort) != AST.PPEnumSort:
        return res

    for i in range(nt.sort.start, nt.sort.end + 1):
        subTerm = AST.PPIntConst(i)
        nxtTerm = expandEnum(term, ntId, subTerm)
        if nxtTerm is not None:
            res.append(nxtTerm)

    return res


def applyExpandDimConst(lib: FnLibrary,
                        term: AST.PPTerm,
                        ntId: int) -> List[AST.PPTerm]:
    res = []
    nxtTerm = expandDimConst(term, ntId)
    if nxtTerm is not None:
        res.append(nxtTerm)

    return res


rules = [
    applyExpandToUnk,
    applyExpandToVar,
    applyExpandToFuncApp,
    applyExpandEnum,
    applyExpandDimConst,
]


def getRule(ruleId: int) -> Callable[[AST.PPTerm, int], List[AST.PPTerm]]:
    return rules[ruleId]
