# Proof (executed) for sphinx-doc__sphinx-12491 f1: sphinx/util/nodes.py:344
# Claim: _lift_sections may raise AttributeError when state_parent.parent is None
# Expected: AttributeError

import pytest
from sphinx.util import nodes as util_nodes

def test_lift_sections_attribute_error():
    """Test that _lift_sections raises AttributeError when walking past a None parent.

    The test passes on the base commit where the function does not exist, and
    fails (by propagating the AttributeError) on the PR commit where the bug is
    present.
    """
    # If the function is not present (base commit), simply pass the test.
    lift = getattr(util_nodes, "_lift_sections", None)
    if lift is None:
        return

    from docutils import nodes as dnodes

    # Dummy node argument (not used by the function for this path).
    dummy_node = dnodes.paragraph()

    # Create a parent node whose own parent is None (top‑level element).
    top_node = dnodes.paragraph()  # parent of state_parent; its .parent is None.
    state_parent = dnodes.section()
    state_parent.parent = top_node

    # Provide title style lists that trigger the nested‑section handling.
    inner_title_styles = ["section"]
    surrounding_title_styles = ["section"]

    # The call should walk up the parent chain and eventually attempt to
    # access ``None.parent``, raising AttributeError.
    lift(
        dummy_node,
        state_parent=state_parent,
        inner_title_styles=inner_title_styles,
        surrounding_title_styles=surrounding_title_styles,
    )
