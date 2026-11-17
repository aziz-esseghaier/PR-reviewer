"""The candidates the evidence agent checks: read from a review run, cleaned up, grouped by PR.

A review run saves each finding's claim in the findings table. The evidence agent reads them
back as Candidates and decides for each one whether a test could show it:
- the claim must be about behavior (an F category), since a test can't show a naming or
  design problem;
- it must name the exception a test would hit. Some replies wrote the strings "None" or
  "null" instead of leaving it empty; those count as no exception.

Imported by other modules; the file isn't run directly.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import TupleRow

from pr_reviewer.core.config import ConfigError
from pr_reviewer.core.schemas import Claim

# Values a model wrote for "no exception" that pass the name check, in lower case.
NO_EXCEPTION = frozenset({"none", "null", "nan", "n/a", "nothing"})


def expected_exception(claim: Claim) -> str | None:
    """Give the exception a claim says a test would hit, or None.

    Args:
        claim (Claim): The claim.

    Returns:
        str | None: Its expected_failure, or None when it's missing or a word such as "None".
    """
    name = claim.expected_failure
    if name is None or name.strip().lower() in NO_EXCEPTION:
        return None
    return name


@dataclass(frozen=True)
class Candidate:
    """One candidate finding of a review run.

    Attributes:
        instance_id (str): The PR.
        finding_id (str): Its ID in the run, such as "f3".
        claim (Claim): What the reviewer said.
    """

    instance_id: str
    finding_id: str
    claim: Claim

    @property
    def expected(self) -> str | None:
        """The exception a test would hit, cleaned up (expected_exception()).

        Returns:
            str | None: Such as "AttributeError", or None.
        """
        return expected_exception(self.claim)

    @property
    def testable(self) -> bool:
        """Whether a test could show the claim: functional, with an expected exception.

        Returns:
            bool: True when the test writer should try.
        """
        return self.claim.is_functional and self.expected is not None


def load_candidates(conn: psycopg.Connection[TupleRow], run: str) -> dict[str, list[Candidate]]:
    """Read a review run's findings as candidates, grouped by PR.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        run (str): The review run's name.

    Returns:
        dict[str, list[Candidate]]: By PR, its candidates in the run's order (f1, f2, ...).
        PRs with no finding are left out.
    """
    rows = conn.execute(
        """
        SELECT instance_id, finding_id, claim FROM findings WHERE run = %s
        ORDER BY instance_id, length(finding_id), finding_id
        """,
        (run,),
    ).fetchall()
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for instance_id, finding_id, claim in rows:
        grouped[instance_id].append(Candidate(instance_id, finding_id, Claim.model_validate(claim)))
    return dict(grouped)


def run_settings(conn: psycopg.Connection[TupleRow], run: str) -> dict[str, Any]:
    """Read the settings a review run was made with.

    Args:
        conn (psycopg.Connection[TupleRow]): Database connection.
        run (str): The review run's name.

    Returns:
        dict[str, Any]: Its config, as saved in review_runs.

    Raises:
        ConfigError: If there's no such run.
    """
    row = conn.execute("SELECT config FROM review_runs WHERE name = %s", (run,)).fetchone()
    if row is None:
        raise ConfigError(f"There's no review run named {run!r}.")
    config: dict[str, Any] = row[0]
    return config
