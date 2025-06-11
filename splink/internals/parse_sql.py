from __future__ import annotations

from collections.abc import Sequence

import sqlglot
import sqlglot.expressions as exp
from sqlglot.expressions import Bracket, Column, Lambda

from splink.internals.sql_transform import remove_quotes_from_identifiers


def get_columns_used_from_sql(
    sql: str, sqlglot_dialect: str = None, retain_table_prefix: bool = False
) -> list[str]:
    """
    Parse the SQL and return a list of column names used.

    - Retains all original functionality (bracketed columns, plain columns, ordering).
    - Skips only lambda variable references, but still captures table columns inside lambdas.
    - Optionally keeps table aliases as prefixes.
    """
    column_names = []
    seen = set()
    syntax_tree = sqlglot.parse_one(sql, read=sqlglot_dialect)

    for subtree in syntax_tree.find_all(exp.Column):
        # Detect if inside a Lambda; only skip unqualified (lambda var) columns
        parent = subtree.parent
        in_lambda = False
        while parent is not None:
            if isinstance(parent, Lambda):
                in_lambda = True
                break
            parent = parent.parent
        if in_lambda and subtree.table is None:
            continue

        # Original bracket vs plain logic
        if subtree.find(Bracket) and isinstance(subtree, Column):
            table = subtree.table
            column = subtree.this.this.this
        elif not isinstance(subtree.parent, Column) and isinstance(subtree, Column):
            table = subtree.table
            column = subtree.this.this
        else:
            table = None
            column = subtree.this.this.this

        # Build final name with optional prefix
        if retain_table_prefix and table:
            key = f"{table}.{column}"
        else:
            key = column

        # Preserve first-seen order
        if key not in seen:
            seen.add(key)
            column_names.append(key)

    return column_names


def parse_columns_in_sql(sql: str, sqlglot_dialect: str, remove_quotes: bool = True) -> Sequence[exp.Column]:
    """Extract all columns found within a SQL expression.

    Args:
        sql (str): A SQL string you wish to parse.

    Returns:
        list[exp.Column]: A list of columns as SQLglot expressions. These can be
            unwrapped with `.sql()`. If the input string is unparseable, None will
            be returned.
    """
    try:
        syntax_tree = sqlglot.parse_one(sql, read=sqlglot_dialect)
    except Exception:  # Consider catching a more specific exception if possible
        # If we can't parse a SQL condition, it's better to just pass.
        return []

    return [
        # Remove quotes if requested by the user
        remove_quotes_from_identifiers(col) if remove_quotes else col
        for col in syntax_tree.find_all(exp.Column)
    ]
