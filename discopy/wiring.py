# -*- coding: utf-8 -*-

"""
Implements wiring diagrams as a free dagger PROP.
"""

from abc import ABC, abstractmethod
import functools
import itertools
import numpy as np

from discopy import cat, messages, monoidal
from discopy.monoidal import PRO, Sum, Ty

def reduce_sequential(arrows):
    return functools.reduce(lambda f, g: f >> g, arrows)

def reduce_parallel(factors, ident=None):
    if ident is None:
        ident = Id(Ty())
    return functools.reduce(lambda x, y: x @ y, factors, ident)

def _dagger_falg(diagram):
    if isinstance(diagram, Box):
        if diagram.name[-1] == '†':
            name = diagram.name[:-1]
        else:
            name = diagram.name + '†'
        return Box(name, diagram.cod, diagram.dom, data=diagram.data)
    if isinstance(diagram, Sequential):
        return reduce_sequential(reversed(diagram.arrows))
    return diagram

class Wiring(ABC, monoidal.Box):
    """
    Implements wiring diagrams in free dagger PROPs.
    """

    @abstractmethod
    def collapse(self, falg):
        """
        Collapse a wiring diagram catamorphically into a single domain,
        codomain, and auxiliary data item.
        """

    @staticmethod
    def id(dom):
        return Id(dom)

    def then(self, *others):
        """
        Implements the sequential composition of wiring diagrams.
        """
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.then(self, *others)
        other = others[0]
        if not isinstance(other, Wiring):
            raise TypeError(messages.type_err(Wiring, other))
        if self.cod != other.dom:
            raise cat.AxiomError(messages.does_not_compose(self, other))

        arrows = [f for f in (self,) + others if not isinstance(f, Id)]
        if not arrows:
            return Id(self.dom)
        if len(arrows) == 1:
            return arrows[0]
        return Sequential(arrows)

    def tensor(self, *others):
        """
        Implements the tensor product of wiring diagrams.
        """
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.tensor(self, *others)
        other = others[0]
        if not isinstance(other, Wiring):
            raise TypeError(messages.type_err(Wiring, other))

        factors = [f for f in (self,) + others if len(f.dom) or len(f.cod)]
        if not factors:
            return Id(Ty())
        if len(factors) == 1:
            return factors[0]
        return Parallel(factors)

    def __matmul__(self, other):
        return self.tensor(other)

    def dagger(self):
        return self.collapse(_dagger_falg)

    def merge_wires(self):
        pass

    def draw(self, *args, **kwargs):
        def ar(f):
            return monoidal.Box(f.name, f.dom, f.cod, data=f.data)
        drawing_functor = Functor(lambda t: t, ar, ob_factory=Ty,
                                  ar_factory=monoidal.Box)
        drawing_functor(self).draw(*args, **kwargs)

class Id(Wiring):
    """ Empty wiring diagram in a free dagger PROP. """
    def __init__(self, dom):
        if not isinstance(dom, Ty):
            raise TypeError(messages.type_err(Ty, dom))
        super().__init__("Id(dom={})".format(dom), dom, dom)

    def __repr__(self):
        return "Id(dom={})".format(repr(self.dom))

    def collapse(self, falg):
        return falg(self)

    def then(self, *others):
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.tensor(self, *others)
        other = others[0]

        if self.cod != other.dom:
            raise cat.AxiomError(messages.does_not_compose(self, other))
        return other

    def tensor(self, *others):
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.tensor(self, *others)
        other = others[0]
        if not isinstance(other, Wiring):
            raise TypeError(messages.type_err(Wiring, other))

        if not self.dom:
            return other
        if isinstance(other, Id):
            return Id(self.dom @ other.dom)
        return super().tensor(other)

    def merge_dom(self, wires=0):
        assert wires <= len(self.dom)
        self._dom = PRO(wires)

    def merge_cod(self, wires=0):
        self.merge_dom(wires)

class Box(Wiring):
    """ Implements boxes in wiring diagrams. """
    def __init__(self, name, dom, cod, **params):
        if not isinstance(dom, Ty):
            raise TypeError(messages.type_err(Ty, dom))
        if not isinstance(cod, Ty):
            raise TypeError(messages.type_err(Ty, cod))
        super().__init__(name, dom, cod, **params)

    def __repr__(self):
        return "Box({}, dom={}, cod={}, data={})".format(
            repr(self.name), repr(self.dom), repr(self.cod), repr(self.data)
        )

    def collapse(self, falg):
        return falg(self)

    def merge_dom(self, wires=0):
        assert wires <= len(self.dom)
        self._dom = PRO(wires)

    def merge_cod(self, wires=0):
        assert wires <= len(self.cod)
        self._cod = PRO(wires)

def _flatten_arrows(arrows):
    for arr in arrows:
        if isinstance(arr, Id):
            continue
        if isinstance(arr, Sequential):
            yield arr.arrows
        else:
            yield [arr]

class Sequential(Wiring):
    """ Sequential composition in a wiring diagram. """
    def __init__(self, arrows, dom=None, cod=None):
        self.arrows = list(itertools.chain(*_flatten_arrows(arrows)))
        if dom is None:
            dom = self.arrows[0].dom
        if cod is None:
            cod = self.arrows[-1].cod
        super().__init__(repr(self), dom, cod)

    def __repr__(self):
        return "Sequential(arrows={})".format(repr(self.arrows))

    def collapse(self, falg):
        return falg(Sequential([f.collapse(falg) for f in self.arrows]))

    def then(self, *others):
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.then(self, *others)
        other = others[0]
        if not isinstance(other, Wiring):
            raise TypeError(messages.type_err(Wiring, other))
        if self.cod != other.dom:
            raise cat.AxiomError(messages.does_not_compose(self, other))

        last = self.arrows[-1] >> other
        last = last.arrows if isinstance(last, Sequential) else [last]
        return Sequential(self.arrows[:-1] + last)

    def merge_wires(self):
        for f, g in zip(self.arrows, self.arrows[1:]):
            fs = f if isinstance(f, Parallel) else Parallel([f])
            gs = g if isinstance(g, Parallel) else Parallel([g])

            adjacency = gs.wire_adjacency(fs)
            for k, factor in enumerate(fs.factors):
                factor.merge_cod(np.count_nonzero(adjacency[k, :]))
            for k, factor in enumerate(gs.factors):
                factor.merge_dom(np.count_nonzero(adjacency[:, k]))

        for f in self.arrows:
            f.merge_wires()

def _flatten_factors(factors):
    for f in factors:
        if isinstance(f, Id):
            for ob in f.dom:
                yield Id(Ty(ob))
        elif isinstance(f, Parallel):
            yield f.factors
        else:
            yield [f]

class Parallel(Wiring):
    """ Parallel composition in a wiring diagram. """
    def __init__(self, factors, dom=None, cod=None):
        self.factors = list(itertools.chain(*_flatten_factors(factors)))
        if dom is None:
            dom = reduce_parallel((f.dom for f in self.factors), Ty())
        if cod is None:
            cod = reduce_parallel((f.cod for f in self.factors), Ty())
        super().__init__(repr(self), dom, cod)

    def __repr__(self):
        return "Parallel(factors={})".format(repr(self.factors))

    def collapse(self, falg):
        return falg(Parallel([f.collapse(falg) for f in self.factors]))

    def wire_adjacency(self, predecessor):
        preds = predecessor.factors if isinstance(predecessor, Parallel) else\
                [predecessor]
        adjacency = np.zeros((len(preds), len(self.factors)), dtype=np.uint)

        l = r = 0
        i = j = 0
        for _ in range(len(self.dom)):
            if i >= len(preds[l].cod):
                l += 1
                i = 0
            if j >= len(self.factors[r].dom):
                r += 1
                j = 0
            adjacency[l, r] += 1
            i += 1
            j += 1
        return adjacency

    def then(self, *others):
        if len(others) != 1 or any(isinstance(other, Sum) for other in others):
            return monoidal.Diagram.tensor(self, *others)
        other = others[0]

        if self.cod != other.dom:
            raise cat.AxiomError(messages.does_not_compose(self, other))

        if isinstance(other, Parallel):
            adjacency = other.wire_adjacency(self)

            f_factors = []
            g_factors = []
            used = set()
            for j, g in enumerate(other.factors):
                incoming = np.flatnonzero(adjacency[:, j])
                fs = [self.factors[i] for i in incoming]

                if all(np.count_nonzero(adjacency[i, :]) == 1 for i
                       in incoming):
                    f_factors.append(reduce_parallel(fs) >> g)
                    g_factors.append(Id(g.cod))
                else:
                    f_factors += [f for i, f in zip(incoming, fs)
                                  if i not in used]
                    g_factors.append(g)
                used |= set(incoming)

            f_factors = reduce_parallel(f_factors)
            g_factors = reduce_parallel(g_factors)
            return Wiring.then(f_factors, g_factors)

        return super().then(other)

    def merge_wires(self):
        dom, cod = Ty(), Ty()
        for f in self.factors:
            f.merge_wires()
            dom = dom @ f.dom
            cod = cod @ f.cod

        self._dom = dom
        self._cod = cod

class Functor(monoidal.Functor):
    def __init__(self, ob, ar, ob_factory=Ty, ar_factory=Box):
        super().__init__(ob, ar, ob_factory, ar_factory)

    def __functor_falg__(self, f):
        if isinstance(f, Id):
            return self.ar_factory.id(f.dom)
        if isinstance(f, Box):
            return self.ar[f]
        if isinstance(f, Sequential):
            return reduce_sequential(f.arrows)
        if isinstance(f, Parallel):
            return reduce_parallel(f.factors, self.ar_factory.id(Ty()))
        raise TypeError(messages.type_err(Wiring, f))

    def __call__(self, diagram):
        if isinstance(diagram, Wiring):
            return diagram.collapse(self.__functor_falg__)
        return super().__call__(diagram)

class WiringFunctor(Functor):
    def __init__(self, typed=False):
        self._typed = typed
        if self._typed:
            ob = lambda t: t
            ar = lambda f: Box(f.name, f.dom, f.cod, data=f.data)
        else:
            ob = lambda t: PRO(len(t))
            ar = lambda f: Box(f.name, PRO(len(f.dom)), PRO(len(f.cod)),
                               data=f.data)
        super().__init__(ob, ar, ob_factory=Ty, ar_factory=Box)

    def __call__(self, diagram):
        result = super().__call__(diagram)
        if isinstance(result, Wiring) and not self._typed:
            result.merge_wires()
        return result