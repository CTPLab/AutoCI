from typing import List, Callable
from HOUDINI.Synthesizer import AST
from HOUDINI.Synthesizer.Utils import ASTUtils


def csv(xs: List[str]) -> str:
    res = ', '.join(xs)
    return res


def repr_py(term: AST.PPTerm) -> str:
    assert term is not None
    term_tpe = type(term)

    def case(tpe):
        return tpe == term_tpe

    if case(AST.PPIntConst):
        return str(term.value)
    elif case(AST.PPRealConst):
        return term.value
    elif case(AST.PPBoolConst):
        return term.value
    elif case(AST.PPVar):
        return term.name
    elif case(AST.PPVarDecl):
        return term.name
    elif case(AST.PPLambda):
        return 'lambda %s: %s' % (csv(map(repr_py, term.params)), repr_py(term.body))
    elif case(AST.PPFuncApp):
        # return '%s(%s)' % (term.fname.value, csv(map(repr_py, term.args)))
        fnRepr = repr_py(term.fn)
        if type(term.fn) != AST.PPVar and type(term.fn) != AST.PPTermUnk:
            fnRepr = '(%s)' % fnRepr
        return '%s(%s)' % (fnRepr, csv(map(repr_py, term.args)))
    elif case(AST.PPListTerm):
        return '[%s]' % csv(map(repr_py, term.items))
    elif case(AST.PPTermNT):
        return term.name
    elif case(AST.PPTermUnk):
        return term.name
    else:
        raise Exception('Unhandled type in printPython: %s' % type(term))


def repr_py_shape(shape: AST.PPDim) -> str:
    res = ''
    if type(shape) == AST.PPDimConst:
        res = str(shape.value)
    elif type(shape) == AST.PPDimVar:
        res = shape.name
    return res


def repr_py_sort(sort: AST.PPSort) -> str:
    def case(tpe):
        return tpe == type(sort)

    if case(AST.PPInt):
        res = 'int'
    elif case(AST.PPReal):
        res = 'real'
    elif case(AST.PPBool):
        res = 'bool'
    elif case(AST.PPListSort):
        res = 'List[%s]' % (repr_py_sort(sort.param_sort),)
    elif case(AST.PPGraphSort):
        res = 'GraphSequences[%s]' % (repr_py_sort(sort.param_sort),)
    elif case(AST.PPTensorSort):
        res = 'Tensor[%s][%s]' % (repr_py_sort(sort.param_sort), ','.join(
            [repr_py_shape(d) for d in sort.shape]))
    elif case(AST.PPFuncSort):
        argsRepr = csv(map(repr_py_sort, sort.args))
        argsRepr = '(%s)' % argsRepr if len(sort.args) > 1 else argsRepr
        res = '(%s --> %s)' % (argsRepr, repr_py_sort(sort.rtpe))
    elif case(AST.PPSortVar):
        res = sort.name
    elif case(AST.PPImageSort):
        res = 'Image'
    elif case(AST.PPEnumSort):
        res = 'EnumSort'
    elif case(AST.PPDimConst) or case(AST.PPDimVar):
        res = repr_py_shape(sort)
    else:
        raise Exception('Unhandled type: %s' % type(sort))

    return res


def repr_py_ann(term: AST.PPTerm) -> str:
    term_tpe = type(term)
    res = ''

    def case(tpe):
        return tpe == term_tpe

    if case(AST.PPIntConst):
        res = str(term.value)
    elif case(AST.PPRealConst):
        res = str(term.value)
    elif case(AST.PPBoolConst):
        res = str(term.value)
    elif case(AST.PPVar):
        res = term.name
    elif case(AST.PPVarDecl):
        res = '%s: %s' % (term.name, term.sort)
    elif case(AST.PPLambda):
        res = 'lambda (%s): %s' % (
            csv(map(repr_py_ann, term.params)), repr_py_ann(term.body))
    elif case(AST.PPFuncApp):
        fnRepr = repr_py_ann(term.fn)
        if type(term.fn) != AST.PPVar and type(term.fn) != AST.PPTermUnk:
            fnRepr = '(%s)' % fnRepr
        res = '%s(%s)' % (fnRepr, csv(map(repr_py_ann, term.args)))
    elif case(AST.PPListTerm):
        res = '[%s]' % csv(map(repr_py_ann, term.items))
    elif case(AST.PPTermNT):
        res = '(%s: %s)' % (term.name, repr_py_sort(term.sort))
    elif case(AST.PPTermUnk):
        res = '(%s: %s)' % (term.name, repr_py_sort(term.sort))
    else:
        raise Exception('Unhandled type: %s' % term_tpe)

    return res


def replaceNthNT(term: AST.PPTerm,
                 ntId: int,
                 newSubTerm: AST.PPTerm) -> AST.PPTerm:
    newTerm = ASTUtils.applyTdOnce(term,
                                   ASTUtils.isNthNT(ntId),
                                   lambda nt: newSubTerm)
    return newTerm


def simplerep(sort: AST.PPSort):
    def case(tpe):
        return type(sort) == tpe

    if case(AST.PPInt):
        return 'int'
    elif case(AST.PPReal):
        return 'real'
    elif case(AST.PPBool):
        return 'bool'
    elif case(AST.PPSortVar):
        return sort.name
    elif case(AST.PPDimVar):
        return sort.name
    elif case(AST.PPDimConst):
        return str(sort.value)
    elif case(AST.PPListSort):
        return 'List[%s]' % simplerep(sort.param_sort)
    elif case(AST.PPGraphSort):
        return 'GraphSequences[%s]' % simplerep(sort.param_sort)
    elif case(AST.PPTensorSort):
        param_sort = simplerep(sort.param_sort)
        shape = ', '.join([simplerep(d) for d in sort.shape])
        return 'Tensor[%s][%s]' % (param_sort, shape)
    elif case(AST.PPFuncSort):
        args = ' * '.join([simplerep(a) for a in sort.args])
        ret = simplerep(sort.rtpe)
        return '( %s -> %s)' % (args, ret)
    elif case(AST.PPImageSort):
        return 'Image'
    elif case(AST.PPEnumSort):
        raise NotImplementedError()
