import pandas as pd
from chunkanon.generalization_rf import OLA_2
from chunkanon.quasi_identifier import QuasiIdentifier

def test_process_chunk_creates_equivalence_classes():
    qis = [QuasiIdentifier("Age", False, 0, 100)]
    ola2 = OLA_2(qis)
    chunk = pd.DataFrame({"Age": [21, 22, 23, 45]})
    histogram = ola2.process_chunk(chunk, [10])
    assert isinstance(histogram, dict)
    assert all(isinstance(k, tuple) for k in histogram)
