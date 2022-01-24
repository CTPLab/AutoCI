from typing import NamedTuple, Union, List

# Custom Sort
PPInt = NamedTuple('PPInt', None)
PPReal = NamedTuple('PPReal', None)
PPBool = NamedTuple('PPBool', None)

PPFuncSort = NamedTuple('PPFuncSort', [('args', List['PPSort']),
                                       ('rtpe', 'PPSort')])

PPTensorSort = NamedTuple('PPTensorSort', [('param_sort', 'PPSort'),
                                           ('shape', List['PPDim'])])

PPEnumSort = NamedTuple('PPEnumSort', [('start', int),
                                       ('end', int)])

PPListSort = NamedTuple('PPListSort', [('param_sort', 'PPSort')])
PPGraphSort = NamedTuple('PPGraphSort', [('param_sort', 'PPSort')])
PPImageSort = NamedTuple('PPImageSort', None)

PPSortVar = NamedTuple('PPSortVar', [('name', str)])

PPSortTypes = (PPInt, PPReal, PPBool,
               PPFuncSort, PPTensorSort, PPEnumSort,
               PPListSort, PPGraphSort, PPImageSort,
               PPSortVar)

PPSort = Union[PPInt, PPReal, PPBool,
               PPFuncSort, PPTensorSort, PPEnumSort,
               PPListSort, PPGraphSort, PPImageSort,
               PPSortVar]

# Custom Dim
PPDimVar = NamedTuple('PPDimVar', [('name', str)])
PPDimConst = NamedTuple('PPDimConst', [('value', int)])

PPDimTypes = (PPDimVar, PPDimConst)
PPDim = Union[PPDimVar, PPDimConst]

# Custom Decl
PPSymbol = NamedTuple('PPSymbol', [('value', str)])
PPVarDecl = NamedTuple('PPVarDecl', [('name', str),
                                     ('sort', PPSort)])

PPFuncDecl = NamedTuple('PPFuncDecl', [('fname', PPSymbol),
                                       ('sort', PPFuncSort)])

PPDeclTypes = (PPVarDecl, PPFuncDecl)
PPDecl = Union[PPVarDecl, PPFuncDecl]

# Custom Term
PPIntConst = NamedTuple('PPIntConst', [('value', int)])
PPRealConst = NamedTuple('PPRealConst', [('value', float)])
PPBoolConst = NamedTuple('PPBoolConst', [('value', bool)])

PPVar = NamedTuple('PPVar', [('name', str)])
PPListTerm = NamedTuple('PPListTerm', [('items', List['PPTerm'])])
PPTermNT = NamedTuple('PPTermNT', [('name', str), ('sort', PPSort)])
PPTermUnk = NamedTuple('PPTermUnk', [('name', str), ('sort', PPSort)])

PPFuncApp = NamedTuple('PPFuncApp', [('fn', 'PPTerm'),
                                     ('args', List['PPTerm'])])

PPLambda = NamedTuple('PPLambda', [('params', List['PPVarDecl']),
                                   ('body', 'PPTerm')])

PPTermTypes = (PPIntConst, PPRealConst, PPBoolConst,
               PPVar, PPListTerm,
               PPTermNT, PPTermUnk,
               PPFuncApp, PPLambda)
PPTerm = Union[PPIntConst, PPRealConst, PPBoolConst,
               PPVar, PPListTerm,
               PPTermNT, PPTermUnk,
               PPFuncApp, PPLambda]

PPNTTypes = (PPTermNT)

PPSortOrDimVar = Union[PPSortVar, PPDimVar]
PPSortOrDim = Union[PPSort, PPDim]
PP = Union[PPSort, PPDim, PPDecl, PPTerm]


def mkListSort(sort: PPSort) -> PPListSort:
    return PPListSort(sort)


def mkGraphSort(sort: PPSort) -> PPGraphSort:
    return PPGraphSort(sort)


def mkUnk(sort: PPSort) -> PPTermUnk:
    return PPTermUnk('Unk', sort)


def mkFuncSort(*sortlist) -> PPFuncSort:
    return PPFuncSort(list(sortlist[:-1]), sortlist[-1])


def mkTensorSort(sort: PPSort,
                 rdims: Union[str, int]) -> PPTensorSort:
    dims = []
    for rdim in rdims:
        if type(rdim) == str:
            dims.append(PPDimVar(rdim))
        elif type(rdim) == int:
            dims.append(PPDimConst(rdim))
        else:
            raise Exception("Unhandled dimension")

    return PPTensorSort(sort, dims)


def mkIntTensorSort(rdims) -> PPTensorSort:
    return mkTensorSort(PPInt(), rdims)


def mkRealTensorSort(rdims) -> PPTensorSort:
    return mkTensorSort(PPReal(), rdims)


def mkBoolTensorSort(rdims) -> PPTensorSort:
    return mkTensorSort(PPBool(), rdims)


if __name__ == '__main__':
    print(PP)
