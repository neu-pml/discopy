import os
from pytest import raises

from PIL import Image, ImageChops
from matplotlib import pyplot as plt
from matplotlib.testing.compare import compare_images

from discopy import *
from discopy.quantum import Circuit
from discopy.monoidal import Sum
from discopy.compact import *
from discopy.drawing import *


IMG_FOLDER, TIKZ_FOLDER, TOL = 'test/src/imgs/', 'test/src/tikz/', 10


def draw_and_compare(file, folder=IMG_FOLDER, tol=TOL,
                     draw=Diagram.draw, **params):
    def decorator(func):
        def wrapper():
            true_path = os.path.join(folder, file)
            test_path = os.path.join(folder, '.' + file)
            draw(func(), path=test_path, show=False, **params)
            test = compare_images(true_path, test_path, tol)
            assert test is None
            os.remove(test_path)
        return wrapper
    return decorator


def tikz_and_compare(file, folder=TIKZ_FOLDER, draw=Diagram.draw, **params):
    def decorator(func):
        def wrapper():
            true_paths = [os.path.join(folder, file)]
            test_paths = [os.path.join(folder, '.' + file)]
            if params.get("use_tikzstyles", DEFAULT['use_tikzstyles']):
                true_paths.append(
                    true_paths[0].replace('.tikz', '.tikzstyles'))
                test_paths.append(
                    test_paths[0].replace('.tikz', '.tikzstyles'))
            draw(func(), path=test_paths[0], **dict(params, to_tikz=True))
            for true_path, test_path in zip(true_paths, test_paths):
                with open(true_path, "r") as true:
                    with open(test_path, "r") as test:
                        assert true.read() == test.read()
                os.remove(test_path)
        return wrapper
    return decorator


def spiral(n_cups):
    """
    Implements the asymptotic worst-case for normal_form, see arXiv:1804.07832.
    """
    x = Ty('x')
    unit, counit = Box('unit', Ty(), x), Box('counit', x, Ty())
    cup, cap = Box('cup', x @ x, Ty()), Box('cap', Ty(), x @ x)
    for box in [unit, counit, cup, cap]:
        box.draw_as_spider, box.color, box.drawing_name = True, "black", ""
    result = unit
    for i in range(n_cups):
        result = result >> Id(x ** i) @ cap @ Id(x ** (i + 1))
    result = result >> Id(x ** n_cups) @ counit @ Id(x ** n_cups)
    for i in range(n_cups):
        result = result >>\
            Id(x ** (n_cups - i - 1)) @ cup @ Id(x ** (n_cups - i - 1))
    return result


@draw_and_compare(
    'crack-eggs.png', figsize=(5, 6), fontsize=18, aspect='equal')
def test_draw_eggs():
    def merge(x):
        return Box('merge', x @ x, x)

    egg, white, yolk = Ty('egg'), Ty('white'), Ty('yolk')
    crack = Box('crack', egg, white @ yolk)
    return crack @ crack\
        >> Id(white) @ Swap(yolk, white) @ Id(yolk)\
        >> merge(white) @ merge(yolk)


@draw_and_compare(
    'spiral.png', draw_type_labels=False,
    draw_box_labels=False, aspect='equal')
def test_draw_spiral():
    return spiral(2)


@draw_and_compare('who-ansatz.png', aspect='equal')
def test_draw_who():
    n, s = Ty('n'), Ty('s')
    copy, update = Box('copy', n, n @ n), Box('update', n @ s, s)
    return Cap(n.r, n)\
        >> Id(n.r) @ copy\
        >> Id(n.r @ n) @ Cap(s, s.l) @ Id(n)\
        >> Id(n.r) @ update @ Id(s.l @ n)


@draw_and_compare('sentence-as-diagram.png', aspect='equal')
def test_draw_sentence():
    from discopy.grammar.pregroup import Ty, Cup, Word, Id
    s, n = Ty('s'), Ty('n')
    Alice, Bob = Word('Alice', n), Word('Bob', n)
    loves = Word('loves', n.r @ s @ n.l)
    return Alice @ loves @ Bob >> Cup(n, n.r) @ Id(s) @ Cup(n.l, n)


@draw_and_compare('bialgebra.png', draw=quantum.zx.Sum.draw, aspect='equal')
def test_draw_bialgebra():
    from discopy.quantum.zx import Z, X, Id, SWAP
    bialgebra = Z(1, 2) @ Z(1, 2) >> Id(1) @ SWAP @ Id(1) >> X(2, 1) @ X(2, 1)
    return bialgebra + bialgebra


def draw_equation(diagrams, **params):
    return drawing.Equation(*diagrams).draw(**params)


@draw_and_compare("snake-equation.png", draw=draw_equation,
                  aspect='auto', figsize=(5, 2), draw_type_labels=False)
def test_snake_equation():
    x = Ty('x')
    return Id(x.r).transpose(left=True), Id(x), Id(x.l).transpose()


@draw_and_compare('typed-snake-equation.png', draw=draw_equation,
                  figsize=(5, 2), aspect='auto')
def test_draw_typed_snake():
    x = Ty('x')
    return Id(x.r).transpose(left=True), Id(x), Id(x.l).transpose()


@tikz_and_compare("spiral.tikz", draw_type_labels=False, use_tikzstyles=True)
def test_spiral_to_tikz():
    return spiral(2)


@tikz_and_compare("copy.tikz", use_tikzstyles=True)
def test_copy_to_tikz():
    x, y = map(Ty, ("$x$", "$y$"))
    copy_x, copy_y = Box('COPY', x, x @ x), Box('COPY', y, y @ y)
    copy_x.draw_as_spider, copy_y.draw_as_spider = True, True
    copy_x.drawing_name, copy_y.drawing_name = "", ""
    copy_x.color, copy_y.color = "black", "black"
    return copy_x @ copy_y >> Id(x) @ Swap(x, y) @ Id(y)




def test_Node_repr():
    from discopy.cat import Ob
    assert repr(Node('dom', depth=1, i=0, obj=Ob('x')))\
        == "Node('dom', depth=1, i=0, obj=cat.Ob('x'))"


def test_diagramize():
    x, y = Ty('x'), Ty('y')
    f = Box('f', x, y)
    with raises(cat.AxiomError):
        @diagramize(y, x, [f])
        def diagram(wire):
            return f(wire)
    with raises(cat.AxiomError):
        @diagramize(x, x, [f])
        def diagram(wire):
            return f(wire)
    with raises(cat.AxiomError):
        @diagramize(x @ x, x, [f])
        def diagram(left, right):
            return f(left, right)
    with raises(cat.AxiomError):
        @diagramize(x, x @ y, [f])
        def diagram(wire):
            return wire, f(offset=0)
    with raises(TypeError):
        @diagramize(x, y, [f])
        def diagram(wire):
            return f(x)
    with raises(ValueError):
        diagramize(x, x, [])


@draw_and_compare('empty_diagram.png')
def test_empty_diagram():
    return Id()
