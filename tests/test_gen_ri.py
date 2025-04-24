from chunkanon.generalization_ri import OLA_1
from chunkanon.quasi_identifier import QuasiIdentifier

def test_build_tree():
    qis = [QuasiIdentifier("Age", False, 0, 10)]
    ola = OLA_1(qis, n=1, max_equivalence_classes=1000)
    tree = ola.build_tree()
    assert isinstance(tree, list)
    assert len(tree) > 0
