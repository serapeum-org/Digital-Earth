"""A refusal that lists what does exist has to list what does exist.

Two lookups answer an unknown citation the same way — `clause()` and `roadmap_order()` both name the ones
the contract holds — and that list is the whole value of the refusal, because the way both are reached is a
citation of a number nobody wrote down. Neither rendering was ever read against a table other than the one
shipped today, and on that table both defects are invisible: the clauses run 1 to 14 without a gap, so a
range cannot misdescribe them, and the orders in hand are 23, 24, 26 and 27a, which sort the same as text
as they do as numbers.

So both are asked here over a table the module does not ship. `clause()` rendered the first and last keys
joined by a hyphen, a range it never checked was contiguous (`R2-L7`); `roadmap_order()` sorted its keys as
text, so an ``order 10`` would have been listed ahead of an ``order 9`` (`R2-N1`). Each is a message a
reader is meant to act on, and each would have pointed somewhere that does not exist.

The shipped tables are asserted too, in the test above each, so a substituted table is never the only thing
these read.
"""

from types import MappingProxyType

import pytest

from digitalearth.base import contract, contract_clauses
from digitalearth.base.contract import roadmap_order
from digitalearth.base.contract_clauses import CLAUSES, clause

#: The clause the gapped table drops. Any clause with a neighbour on each side would do; this one is the
#: most cited, so a range that swallowed it would be the most misleading.
_DROPPED = 7

#: An order number no table holds, which is what both lookups are asked for. The same number the shipped
#: docstrings use for "nobody wrote this down".
_ABSENT_ORDER = "order 99"


def _without_a_clause(number):
    """Return the clause table with one clause taken out.

    Args:
        number: The clause to drop.

    Returns:
        A read-only mapping, the same shape as :data:`CLAUSES`, with that number's entry gone.
    """
    return MappingProxyType(
        {key: stated for key, stated in CLAUSES.items() if key != number}
    )


class TestAClauseRefusalNamesOnlyNumbersTheTableHolds:
    """The message's job is to be followed, so everything it lists has to be resolvable."""

    def test_an_unbroken_table_is_named_as_the_one_range_it_is(self):
        """The common case, and the reason the defect went unseen.

        Test scenario:
            Nothing has ever removed a clause, so the shipped table is one unbroken run and the range is
            both correct and the shortest true rendering of it. That is worth keeping: the fix must not
            turn one range into fourteen numbers.
        """
        uncited = max(CLAUSES) + 1
        with pytest.raises(KeyError) as refusal:
            clause(uncited)
        assert "it states C1-C14" in refusal.value.args[0], refusal.value.args[0]

    def test_a_table_with_a_gap_is_not_named_as_a_range_that_covers_it(
        self, monkeypatch
    ):
        """Remove a clause from the middle and the old message claimed it was still there.

        Args:
            monkeypatch: Stands the gapped table in for the canonical one, which is the only way to reach
                a table with a gap: nothing has ever removed a clause.

        Test scenario:
            The rendering read the first and last keys and joined them with a hyphen, so a reader following
            this refusal would be sent to look up a clause the contract no longer states. Nothing else in
            the suite renders a refusal over a table it does not ship, so nothing else could see it.
        """
        gapped = _without_a_clause(_DROPPED)
        uncited = max(gapped) + 1
        monkeypatch.setattr(contract_clauses, "CLAUSES", gapped)
        with pytest.raises(KeyError) as refusal:
            clause(uncited)
        assert "it states C1-C6, C8-C14" in refusal.value.args[0], refusal.value.args[0]


class TestARoadmapRefusalCountsTheOrdersTheWayTheRoadmapDoes:
    """An order is a number with an optional letter, and a number is not sorted as text."""

    def test_the_orders_this_contract_names_are_all_listed(self):
        """The shipped table, whose order happens to be the same either way.

        Test scenario:
            23, 24, 26 and 27a sort identically as text and as numbers, which is why the defect could sit
            here unseen. The listing is pinned all the same: the fix must not drop the letter suffix, and
            27a must still sort after 27 rather than being read as 27 with the letter thrown away.
        """
        with pytest.raises(KeyError) as refusal:
            roadmap_order(_ABSENT_ORDER)
        assert refusal.value.args[0].endswith(
            "order 23, order 24, order 26, order 27a"
        ), refusal.value.args[0]

    def test_a_two_digit_order_is_not_listed_ahead_of_a_one_digit_one(
        self, monkeypatch
    ):
        """The roadmap counts its orders; `sorted()` on the keys only spelled them.

        Args:
            monkeypatch: Stands in a table that reaches two digits, which the shipped one does not — it
                starts at 23, so every key is the same length and text order and numeric order agree.

        Test scenario:
            `", ".join(sorted(ROADMAP_ORDERS))` compares the keys character by character, so `"order 10"`
            sorts before `"order 9"` — right for text, wrong for a plan. The letter suffix is in the table
            too, because it is part of the number and has to sort after the bare one, not before it.
        """
        counted = MappingProxyType(
            {
                "order 10": "the tenth",
                "order 9": "the ninth",
                "order 9a": "a step of its own after the ninth",
            }
        )
        monkeypatch.setattr(contract, "ROADMAP_ORDERS", counted)
        with pytest.raises(KeyError) as refusal:
            roadmap_order(_ABSENT_ORDER)
        assert refusal.value.args[0].endswith("order 9, order 9a, order 10"), (
            refusal.value.args[0]
        )

    def test_an_order_whose_spelling_carries_no_number_is_listed_first(
        self, monkeypatch
    ):
        """A key the counting cannot read has to land somewhere a reader will see it.

        Args:
            monkeypatch: Stands in a table holding a key that is not a number at all, which the shipped
                one cannot supply — every order it names is `order <n>`.

        Test scenario:
            Counting the orders means reading a number off each key, and `ROADMAP_ORDERS` is a table a
            future order is added to by hand. The two tests above only ask what happens when every key
            parses; this asks what happens when one does not. There is no right place in a numbered list
            for a key with no number, so it is sorted to the front, where the reader following the refusal
            meets it rather than finding it buried between two numbers it does not sit between.
        """
        unnumbered = MappingProxyType(
            {
                "order 9": "the ninth",
                "the Core spelling": "no number was ever given to this one",
                "order 10": "the tenth",
            }
        )
        monkeypatch.setattr(contract, "ROADMAP_ORDERS", unnumbered)
        with pytest.raises(KeyError) as refusal:
            roadmap_order(_ABSENT_ORDER)
        assert refusal.value.args[0].endswith("the Core spelling, order 9, order 10"), (
            refusal.value.args[0]
        )
