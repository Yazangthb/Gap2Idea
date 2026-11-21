import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'bench'))
import five_paper_pilot as p


def test_five_paper_input_uses_unambiguous_ids_without_labels():
    rows = [{'pid': 'a', 'idx': 0, 'sentence': 'First sentence', 'label': 'research_gap'},
            {'pid': 'a', 'idx': 1, 'sentence': 'Second sentence', 'label': 'none'},
            {'pid': 'b', 'idx': 0, 'sentence': 'Third sentence', 'label': 'limitation'}]
    _, text, keys = p.group_prompt(rows)
    assert keys == [['a', 0], ['a', 1], ['b', 0]]
    assert text.splitlines() == ['PAPER a', '[0] First sentence', '[1] Second sentence',
                                 'PAPER b', '[2] Third sentence']
    assert all(label not in text for label in ('research_gap', 'limitation'))
    assert p.metrics(rows, {('a', 0), ('a', 1)}) == {
        'tp': 1, 'fp': 1, 'positives': 2, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
