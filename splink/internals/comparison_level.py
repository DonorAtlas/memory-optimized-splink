from __future__ import annotations

import hashlib
import logging
import math
import re
from copy import copy
from statistics import median
from textwrap import dedent
from typing import Any, Optional, Union, cast

import sqlglot
from sqlglot.expressions import Column, Identifier
from sqlglot.optimizer.normalize import normalize
from sqlglot.optimizer.simplify import simplify

from splink.internals.constants import LEVEL_NOT_OBSERVED_TEXT
from splink.internals.input_column import InputColumn
from splink.internals.misc import (
    dedupe_preserving_order,
    interpolate,
    join_list_with_commas_final_and,
    match_weight_to_bayes_factor,
)
from splink.internals.parse_sql import get_columns_used_from_sql
from splink.internals.sql_transform import sqlglot_tree_signature

logger = logging.getLogger(__name__)

total_records_in_field = {
    "middle_name": 62_154_124,
    "middle_initial": 189_090_682,
    "first_last": 226_657_846,
    "tokenized_employers": 19_910_270,
    "occupations": 100_412_596,
    "zip_5s": 226_657_360,
    "city_state_pairs": 226_641_172,
    "addr_digits": 226_657_594,  # false
    "street_addresses": 226_657_846,  # false
    "nonprofit_id": 98_890_398,
    "nonprofit_coordinates": 98_890_398,
}


def _is_exact_match(sql_syntax_tree):
    signature = sqlglot_tree_signature(sql_syntax_tree)

    if signature != sqlglot_tree_signature(sqlglot.parse_one("col_l = col_r")):
        return False

    cols = [s.output_name for s in sql_syntax_tree.find_all(Column)]
    cols_truncated = [c[:-2] for c in cols]
    if cols_truncated[0] == cols_truncated[1]:
        return True
    else:
        return False


def _exact_match_colname(sql_syntax_tree):
    # only interested in expression directly, not context
    sql_syntax_tree.parent = None
    cols = []

    for identifier in sql_syntax_tree.find_all(Identifier):
        identifier.args["quoted"] = False

    cols = [id.sql() for id in sql_syntax_tree.find_all(Identifier) if id.depth == 2]

    cols = [c[:-2] for c in cols]  # Remove _l and _r
    cols = list(set(cols))
    if len(cols) != 1:
        raise ValueError(f"Expected sql condition to refer to one column but got {cols}")
    return cols[0]


def _get_and_subclauses(expr: sqlglot.Expression) -> list[sqlglot.Expression]:
    # get list of subclauses joined together by 'AND' at top-level
    # e.g. 'A AND B AND C' -> ['A', 'B', 'C']
    # or if no AND, return expression as a list, e.g. 'A' -> ['A']
    if isinstance(expr, sqlglot.exp.And):
        return list(expr.flatten())
    return [expr]


def _default_m_values(num_levels: int) -> list[float]:
    proportion_exact_match = 0.95
    remainder = 1 - proportion_exact_match
    split_remainder = remainder / (num_levels - 1)
    return [split_remainder] * (num_levels - 1) + [proportion_exact_match]


def _default_u_values(num_levels: int) -> list[float]:
    m_vals = _default_m_values(num_levels)
    if num_levels == 2:
        match_weights = [-5]
    else:
        match_weights = interpolate(-5, 3, num_levels - 1)
    match_weights = match_weights + [10]

    u_vals = []
    for m, w in zip(m_vals, match_weights):
        p = match_weight_to_bayes_factor(w)
        u = m / p
        u_vals.append(u)

    return u_vals


class ComparisonLevel:
    """Each ComparisonLevel defines a gradation (category) of similarity within a
    `Comparison`.

    For example, a `Comparison` that uses the first_name and surname columns may
    define three `ComparisonLevel`s:
        An exact match on first name and surname
        First name and surname have a JaroWinkler score of above 0.95
        All other comparisons

    The method used to assess similarity will depend on the type of data - for
    instance, the method used to assess similarity of a company's turnover would be
    different to the method used to assess the similarity of a person's first name.

    To summarise:

    ```
    Data Linking Model
    ├─-- Comparison: Name
    │    ├─-- ComparisonLevel: Exact match on first_name and surname
    │    ├─-- ComparisonLevel: first_name and surname have JaroWinkler > 0.95
    │    ├─-- ComparisonLevel: All other
    ├─-- Comparison: Date of birth
    │    ├─-- ComparisonLevel: Exact match
    │    ├─-- ComparisonLevel: One character difference
    │    ├─-- ComparisonLevel: All other
    ├─-- etc.
    ```

    ComparisonLevel is a dialected object.
    """

    def __init__(
        self,
        sql_condition: str,
        sqlglot_dialect: str,
        *,
        label_for_charts: str = None,
        is_null_level: bool = False,
        tf_adjustment_column: str = None,
        tf_adjustment_weight: float = 1.0,
        tf_minimum_u_value: float = 0.0,
        tf_col_is_array: bool = False,
        tf_modifier_custom_sql: str = None,
        m_probability: float = None,
        u_probability: float = None,
        max_epsilon_value: float = 1.0,
        similarity_value: float = 1.0,
        log_base: float = 2.0,
        disable_tf_exact_match_detection: bool = False,
        fix_m_probability: bool = False,
        fix_u_probability: bool = False,
        only_help: bool = False,
    ):
        self.sqlglot_dialect = sqlglot_dialect

        self._sql_condition = sql_condition
        self._is_null_level = is_null_level
        self._label_for_charts = label_for_charts

        self._tf_adjustment_column = tf_adjustment_column
        self._tf_adjustment_weight = tf_adjustment_weight
        self._tf_minimum_u_value = tf_minimum_u_value
        self._tf_col_is_array = tf_col_is_array
        if tf_col_is_array:
            self.log_base = log_base
        self.tf_modifier_custom_sql = tf_modifier_custom_sql
        self._disable_tf_exact_match_detection = disable_tf_exact_match_detection

        self.max_epsilon_value = max_epsilon_value
        # TODO: make this a continuous function
        self.similarity_value = similarity_value
        self.log_base = log_base

        # internally these can be LEVEL_NOT_OBSERVED_TEXT, so allow for this
        self._m_probability: float | None | str = m_probability
        self._u_probability: float | None | str = u_probability
        self.default_m_probability: float | None = None
        self.default_u_probability: float | None = None

        self._fix_m_probability = fix_m_probability
        self._fix_u_probability = fix_u_probability
        self._only_help = only_help

        # TODO: control this in comparison getter setter ?
        # These will be set when the ComparisonLevel is passed into a Comparison
        self._comparison_vector_value: Optional[int] = None
        self._max_level: Optional[bool] = None

        # Enable the level to 'know' when it's been trained
        self._trained_m_probabilities: list[dict[str, Any]] = []
        self._trained_u_probabilities: list[dict[str, Any]] = []
        # controls warnings from model training - ensures we only send once
        self._m_warning_sent = False
        self._u_warning_sent = False

        self._validate()

    def copy(self):
        # define a simple copy method to make copying easy/customisable
        return copy(self)

    @property
    def is_null_level(self) -> bool:
        return self._is_null_level

    @property
    def disable_tf_exact_match_detection(self) -> bool:
        return self._disable_tf_exact_match_detection

    @property
    def only_help(self) -> bool:
        return self._only_help

    @property
    def sql_condition(self) -> str:
        return self._sql_condition

    @property
    def parsed_sql_condition(self):
        expression = sqlglot.parse_one(self.sql_condition)
        mapping = {}

        def replace_complex_expressions(expr):
            # If it's a complex expression (function, lambda, case, etc.)
            if isinstance(
                expr, (sqlglot.exp.Anonymous, sqlglot.exp.Case, sqlglot.exp.Lambda, sqlglot.exp.Reduce)
            ):
                expr_str = expr.sql(dialect="duckdb")  # Ensure consistent dialect
                # Using a deterministic hash based on the expression string
                # This ensures the same expression always gets the same hash
                hash_id = hashlib.md5(expr_str.encode()).hexdigest()[:8]
                col_name = f"__{'_'.join(sorted([col.input_name for col in self._input_columns_used_by_sql_condition]))}_{hash_id}"
                mapping[col_name] = expr_str
                return sqlglot.exp.Identifier(this=col_name)
            return expr

        # Replace all nested expressions
        simplified_expr = expression.transform(replace_complex_expressions)

        # If no complex expressions were found, mapping will be empty and simplified_expr
        # will be the original SQL condition
        return mapping, simplified_expr.sql(dialect="duckdb")

    @property
    def _tf_adjustment_input_column(self):
        val = self._tf_adjustment_column
        if val:
            return InputColumn(
                val, sqlglot_dialect_str=self.sqlglot_dialect, is_array_column=self._tf_col_is_array
            )
        else:
            return None

    @property
    def _tf_adjustment_input_column_name(self):
        input_column = self._tf_adjustment_input_column
        if input_column:
            return input_column.unquote().name

    @property
    def m_probability(self) -> float | None:
        if self.is_null_level:
            raise ValueError("Null levels have no m-probability")
        if self._m_probability == LEVEL_NOT_OBSERVED_TEXT:
            return 1e-6
        m_probability = cast(Union[float, None], self._m_probability)
        if m_probability is None and self.default_m_probability is not None:
            return self.default_m_probability
        return m_probability

    @m_probability.setter
    def m_probability(self, value: float) -> None:
        if self.is_null_level:
            raise AttributeError("Cannot set m_probability when is_null_level is true")

        self._m_probability = value

    @property
    def u_probability(self) -> float | None:
        if self.is_null_level:
            raise ValueError("Null levels have no u-probability")
        if self._u_probability == LEVEL_NOT_OBSERVED_TEXT:
            return 1e-6
        u_probability = cast(Union[float, None], self._u_probability)
        if u_probability is None and self.default_u_probability is not None:
            return self.default_u_probability
        return u_probability

    @u_probability.setter
    def u_probability(self, value: float) -> None:
        if self.is_null_level:
            raise AttributeError("Cannot set u_probability when is_null_level is true")
        self._u_probability = value

    @property
    def _m_probability_description(self) -> str:
        if self.is_null_level:
            return ""
        if self.m_probability is not None:
            percentage = self.m_probability * 100
            one_in_n = 1 / self.m_probability if self.m_probability > 0 else float("inf")
            return (
                "Amongst matching record comparisons, "
                f"{percentage:.4g}% of records (i.e. one in "
                f"{self._num_fmt_dp_or_sf(one_in_n)}) are in the "
                f"{self.label_for_charts.lower()} comparison level"
            )
        else:
            return ""

    @property
    def _u_probability_description(self) -> str:
        if self.is_null_level:
            return ""
        if self.u_probability is not None:
            percentage = self.u_probability * 100
            one_in_n = 1 / self.u_probability if self.u_probability > 0 else float("inf")
            return (
                "Amongst non-matching record comparisons, "
                f"{percentage:.4g}% of records (i.e. one in "
                f"{self._num_fmt_dp_or_sf(one_in_n)}) are in the "
                f"{self.label_for_charts.lower()} comparison level"
            )
        else:
            return ""

    def _add_trained_u_probability(self, val, desc="no description given"):
        self._trained_u_probabilities.append({"probability": val, "description": desc, "m_or_u": "u"})

    def _add_trained_m_probability(self, val, desc="no description given"):
        self._trained_m_probabilities.append({"probability": val, "description": desc, "m_or_u": "m"})

    @property
    def _has_estimated_u_values(self):
        if self.is_null_level:
            return True
        vals = [r["probability"] for r in self._trained_u_probabilities]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return len(vals) > 0

    @property
    def _has_estimated_m_values(self):
        if self.is_null_level:
            return True
        vals = [r["probability"] for r in self._trained_m_probabilities]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return len(vals) > 0

    @property
    def _has_estimated_values(self):
        return self._has_estimated_m_values and self._has_estimated_u_values

    @property
    def _trained_m_median(self):
        vals = [r["probability"] for r in self._trained_m_probabilities]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if len(vals) == 0:
            return None
        return median(vals)

    @property
    def _trained_u_median(self):
        vals = [r["probability"] for r in self._trained_u_probabilities]
        vals = [v for v in vals if isinstance(v, (int, float))]
        if len(vals) == 0:
            return None
        return median(vals)

    @property
    def _m_is_trained(self):
        if self.is_null_level:
            return True
        if self._m_probability == LEVEL_NOT_OBSERVED_TEXT:
            return False
        if self._m_probability is None:
            return False
        return True

    @property
    def _u_is_trained(self):
        if self.is_null_level:
            return True
        if self._u_probability == LEVEL_NOT_OBSERVED_TEXT:
            return False
        if self._u_probability is None:
            return False
        return True

    @property
    def _is_trained(self):
        return self._m_is_trained and self._u_is_trained

    @property
    def _bayes_factor(self):
        if self.is_null_level:
            return 1.0
        if self.m_probability is None or self.u_probability is None:
            return None
        elif self.u_probability == 0:
            return math.inf
        else:
            return self.m_probability / self.u_probability

    @property
    def _log2_bayes_factor(self):
        if self.is_null_level:
            return 0.0
        else:
            return math.log2(self._bayes_factor)

    def _num_fmt_dp_or_sf(self, val):
        if val > 5000:
            return f"{val:,.0f}"
        elif val >= 100:
            return f"{val:,.0f}"
        else:
            return f"{val:,.4g}"

    @property
    def _bayes_factor_description(self):
        text = f"If comparison level is `{self.label_for_charts.lower()}` " "then comparison is"

        if self._bayes_factor == math.inf:
            return f"{text} certain to be a match"
        elif self._bayes_factor == 0.0:
            return f"{text} impossible to be a match"
        elif self._bayes_factor >= 1.0:
            return f"{text} {self._num_fmt_dp_or_sf(self._bayes_factor)} times " "more likely to be a match"
        else:
            mult = 1 / self._bayes_factor
            return f"{text} {self._num_fmt_dp_or_sf(mult)} times " "less likely to be a match"

    @property
    def label_for_charts(self):
        return self._label_for_charts or str(self.comparison_vector_value)

    def _label_for_charts_no_duplicates(self, comparison_levels: list[ComparisonLevel] = None) -> str:
        if comparison_levels is not None:
            labels = []
            for cl in comparison_levels:
                labels.append(cl.label_for_charts)

        if len(labels) == len(set(labels)):
            return self.label_for_charts

        # Make label unique
        cvv = str(self.comparison_vector_value)
        label = self.label_for_charts
        return f"{cvv}. {label}"

    @property
    def _is_else_level(self):
        if self.sql_condition.strip().upper() == "ELSE":
            return True

    @property
    def comparison_vector_value(self) -> int:
        if (cvv := self._comparison_vector_value) is not None:
            return cvv
        raise ValueError(
            "To access a `comparison_vector_value`, a `ComparisonLevel` must " "belong to a `Comparison`"
        )

    @property
    def _has_tf_adjustments(self):
        col = self._tf_adjustment_column
        return col is not None

    def _validate_sql(self):
        sql = self.sql_condition
        if self._is_else_level:
            return True
        dialect = self.sqlglot_dialect
        try:
            sqlglot.parse_one(sql, read=dialect)
        except sqlglot.ParseError as e:
            raise ValueError(f"Error parsing sql_statement:\n{sql}") from e

        return True

    @property
    def _input_columns_used_by_sql_condition(self) -> list[InputColumn]:
        # returns e.g. InputColumn(first_name), InputColumn(surname)
        if self._is_else_level:
            return []

        cols = get_columns_used_from_sql(self.sql_condition, sqlglot_dialect=self.sqlglot_dialect)
        # Parsed order seems to be roughly in reverse order of apearance
        cols = cols[::-1]

        cols = [re.sub(r"_L$|_R$", "", c, flags=re.IGNORECASE) for c in cols]
        cols = dedupe_preserving_order(cols)

        input_cols = []
        for c in cols:
            # We could have tf adjustments for surname on a dmeta_surname column
            # If so, we want to set the tf adjustments against the surname col,
            # not the dmeta_surname one

            input_cols.append(InputColumn(c, sqlglot_dialect_str=self.sqlglot_dialect))

        return input_cols

    @property
    def _input_columns_used_by_tf_modifier_custom_sql(self) -> list[InputColumn]:
        # returns e.g. InputColumn(first_name), InputColumn(surname)
        if self._is_else_level:
            return []

        if not self.tf_modifier_custom_sql:
            return []

        cols = get_columns_used_from_sql(self.tf_modifier_custom_sql, sqlglot_dialect=self.sqlglot_dialect)
        # Parsed order seems to be roughly in reverse order of apearance
        cols = cols[::-1]

        cols = [re.sub(r"_L$|_R$|^tf_", "", c, flags=re.IGNORECASE) for c in cols]
        cols = dedupe_preserving_order(cols)

        input_cols = []
        for c in cols:
            # We could have tf adjustments for surname on a dmeta_surname column
            # If so, we want to set the tf adjustments against the surname col,
            # not the dmeta_surname one

            input_cols.append(InputColumn(c, sqlglot_dialect_str=self.sqlglot_dialect))

        return input_cols

    @property
    def _columns_to_select_for_blocking(self):
        # e.g. l.first_name as first_name_l, r.first_name as first_name_r
        output_cols = []
        cols = self._input_columns_used_by_sql_condition

        for c in cols:
            output_cols.extend(c.l_r_names_as_l_r)
            if self._tf_adjustment_input_column:
                if self._tf_col_is_array:
                    continue
                output_cols.extend(self._tf_adjustment_input_column.l_r_tf_names_as_l_r)

        if self.tf_modifier_custom_sql:
            if self.tf_modifier_custom_sql:
                output_cols.extend(
                    [
                        c
                        for col in self._input_columns_used_by_tf_modifier_custom_sql
                        for c in col.l_r_tf_names_as_l_r
                    ]
                )
        return dedupe_preserving_order(output_cols)

    @property
    def _when_then_comparison_vector_value_sql(self):
        # e.g. when first_name_l = first_name_r then 1
        if not hasattr(self, "_comparison_vector_value"):
            raise ValueError(
                "Cannot get the 'when .. then ...' sql expression because "
                "this comparison level does not belong to a parent Comparison. "
                "The comparison_vector_value is only defined in the "
                "context of a list of ComparisonLevels within a Comparison."
            )
        if self._is_else_level:
            return f"{self.sql_condition} {self.comparison_vector_value}"
        else:
            return f"WHEN {self.sql_condition} THEN {self.comparison_vector_value}"

    @property
    def _is_exact_match(self):
        if self._is_else_level:
            return False

        sql_syntax_tree = sqlglot.parse_one(self.sql_condition.lower(), read=self.sqlglot_dialect)
        sql_cnf = simplify(normalize(sql_syntax_tree))

        exprs = _get_and_subclauses(sql_cnf)
        for expr in exprs:
            if not _is_exact_match(expr):
                return False
        return True

    @property
    def _exact_match_colnames(self):
        sql_syntax_tree = sqlglot.parse_one(self.sql_condition.lower(), read=self.sqlglot_dialect)
        sql_cnf = simplify(normalize(sql_syntax_tree))

        exprs = _get_and_subclauses(sql_cnf)
        for expr in exprs:
            if not _is_exact_match(expr):
                raise ValueError("sql_cond not an exact match so can't get exact match column name")

        cols = []
        for expr in exprs:
            col = _exact_match_colname(expr)
            cols.append(col)
        return cols

    def _u_probability_corresponding_to_exact_match(
        self, comparison_levels: list[ComparisonLevel]
    ) -> float | None:
        if self.disable_tf_exact_match_detection:
            return self.u_probability

        # otherwise, default to looking for an appropriate exact match level:

        # Find a level with a single exact match colname
        # which is equal to the tf adjustment input colname

        for level in comparison_levels:
            if not level._is_exact_match:
                continue
            colnames = level._exact_match_colnames
            if len(colnames) != 1:
                continue
            if colnames[0] == self._tf_adjustment_input_column_name.lower():
                return level.u_probability

        raise ValueError(
            "Could not find an exact match level for "
            f"{self._tf_adjustment_input_column_name}."
            "\nAn exact match level is required to make a term frequency adjustment "
            "on a comparison level that is not an exact match."
        )

    def _bayes_factor_sql(self, gamma_column_name: str) -> str:
        bayes_factor = self._bayes_factor if self._bayes_factor != math.inf else "'Infinity'"
        sql = f"""
        WHEN
        {gamma_column_name} = {self.comparison_vector_value}
        THEN cast({bayes_factor} as float8)
        """
        return dedent(sql)

    def _tf_adjustment_sql(
        self,
        gamma_column_name: str,
        comparison_levels: list[ComparisonLevel],
    ) -> str:
        gamma_colname_value_is_this_level = f"{gamma_column_name} = {self.comparison_vector_value}"

        # A tf adjustment of 1D is a multiplier of 1.0, i.e. no adjustment
        if self.comparison_vector_value == -1:
            sql = f"WHEN  cv.{gamma_colname_value_is_this_level} then cast(1 as float8)"
        elif not self._has_tf_adjustments:
            sql = f"WHEN  cv.{gamma_colname_value_is_this_level} then cast(1 as float8)"
        elif self._tf_adjustment_weight == 0:
            sql = f"WHEN  cv.{gamma_colname_value_is_this_level} then cast(1 as float8)"
        elif self._is_else_level:
            sql = f"WHEN  cv.{gamma_colname_value_is_this_level} then cast(1 as float8)"
        else:
            tf_adj_col = self._tf_adjustment_input_column
            tf_col_is_array = self._tf_col_is_array
            col_name = tf_adj_col.unquote().name.replace(" ", "_")
            N = total_records_in_field[col_name]

            if not tf_adj_col:
                sql = ""

            elif tf_col_is_array:
                # For array TF columns, return the calculated TF adjustment from inference
                tf_adjustment_column = f"{col_name}_tf.tf_adjustment_{col_name}"
                tf_adjustment_exists = f"{tf_adjustment_column} is not null"

                sql = f"""
                WHEN cv.{gamma_colname_value_is_this_level} THEN
                    CASE WHEN {tf_adjustment_exists}
                    THEN {tf_adjustment_column}
                    ELSE cast(1 as float8)
                    END
                """

            elif self._is_exact_match:
                product_l_r = f"(cv.{tf_adj_col.tf_name_l})"

                tf_adjustment_exists = f"{product_l_r} is not null"

                # Using coalesce protects against one of the tf adjustments being null
                # Which would happen if the user provided their own tf adjustment table
                # That didn't contain some of the values in this data

                # In this case rather than taking the greater of the two, we take
                # whichever value exists

                if self._tf_minimum_u_value == 0.0:
                    divisor_sql = f"{product_l_r}"
                else:
                    # This sql works correctly even when the tf_minimum_u_value is 0.0
                    # but is less efficient to execute, hence the above if statement
                    divisor_sql = f"""
                    (CASE
                        WHEN {product_l_r} > cast({self._tf_minimum_u_value} as float8)
                            THEN {product_l_r}
                        ELSE cast({self._tf_minimum_u_value} as float8)
                    END)
                    """

                if self.tf_modifier_custom_sql:
                    multiplier_sql = f"({N} / ({divisor_sql} * ({self.tf_modifier_custom_sql})))"
                else:
                    multiplier_sql = f"({N} / {divisor_sql})"

                sql = f"""
                WHEN  {gamma_colname_value_is_this_level} then
                    (CASE WHEN {tf_adjustment_exists}
                    THEN {multiplier_sql}
                    ELSE cast(1 as float8)
                    END)
                """

            else:
                max_epsilon_value = self.max_epsilon_value
                similarity_value = self.similarity_value
                product_l_r = f"(cv.{tf_adj_col.tf_name_l} * cv.{tf_adj_col.tf_name_r})"

                tf_adjustment_exists = f"{product_l_r} is not null"

                tf_score_sql = f"(({similarity_value * N}/POW({product_l_r}, 0.5)))"

                second_term = (1 - similarity_value) * max_epsilon_value * N**2
                if second_term != 0:
                    tf_score_sql += f"+ ({second_term}/{product_l_r})"

                if self.tf_modifier_custom_sql:
                    multiplier_sql = f"(1/{self.tf_modifier_custom_sql}) * {tf_score_sql}"
                else:
                    multiplier_sql = f"{tf_score_sql}"

                sql = f"""
                WHEN  {gamma_colname_value_is_this_level} then
                    (CASE WHEN {tf_adjustment_exists}
                    THEN {multiplier_sql}
                    ELSE cast(1 as float8)
                    END)
                """
        return dedent(sql).strip()

    def as_dict(self):
        "The minimal representation of this level to use as an input to Splink"
        output: dict[str, Any] = {}

        output["sql_condition"] = self.sql_condition

        if self.label_for_charts:
            output["label_for_charts"] = self.label_for_charts

        if self._m_probability and self._m_is_trained:
            output["m_probability"] = self.m_probability

        if self._u_probability and self._u_is_trained:
            output["u_probability"] = self.u_probability

        output["fix_m_probability"] = self._fix_m_probability
        output["fix_u_probability"] = self._fix_u_probability

        if self._has_tf_adjustments:
            output["tf_adjustment_column"] = self._tf_adjustment_input_column.input_name
            if self._tf_minimum_u_value != 0:
                output["tf_minimum_u_value"] = self._tf_minimum_u_value
            if self._tf_adjustment_weight != 0:
                output["tf_adjustment_weight"] = self._tf_adjustment_weight
            if self._tf_col_is_array:
                output["tf_col_is_array"] = self._tf_col_is_array
                output["log_base"] = self.log_base
            if self.tf_modifier_custom_sql:
                output["tf_modifier_custom_sql"] = self.tf_modifier_custom_sql
        if not self._is_exact_match:
            if self.max_epsilon_value:
                output["max_epsilon_value"] = self.max_epsilon_value
            if self.similarity_value:
                output["similarity_value"] = self.similarity_value

        if self.is_null_level:
            output["is_null_level"] = True

        if self._disable_tf_exact_match_detection:
            output["disable_tf_exact_match_detection"] = True

        if self._only_help:
            output["only_help"] = True

        return output

    def _as_completed_dict(self):
        comp_dict = self.as_dict()
        comp_dict["comparison_vector_value"] = self.comparison_vector_value
        return comp_dict

    def _as_detailed_record(
        self, comparison_num_levels: int, comparison_levels: list[ComparisonLevel]
    ) -> dict[str, Any]:
        "A detailed representation of this level to describe it in charting outputs"
        output: dict[str, Any] = {}
        output["sql_condition"] = self.sql_condition
        output["label_for_charts"] = self._label_for_charts_no_duplicates(comparison_levels)

        if not self._is_null_level:
            output["m_probability"] = self.m_probability
            output["u_probability"] = self.u_probability

            output["m_probability_description"] = self._m_probability_description
            output["u_probability_description"] = self._u_probability_description

        if not self._is_exact_match:
            output["max_epsilon_value"] = self.max_epsilon_value
            output["similarity_value"] = self.similarity_value

        output["has_tf_adjustments"] = self._has_tf_adjustments
        if self._has_tf_adjustments:
            output["tf_adjustment_column"] = self._tf_adjustment_input_column.input_name
            output["tf_col_is_array"] = self._tf_col_is_array
            output["tf_modifier_custom_sql"] = self.tf_modifier_custom_sql
            if self._tf_col_is_array:
                output["log_base"] = self.log_base
        else:
            output["tf_adjustment_column"] = None
        output["tf_adjustment_weight"] = self._tf_adjustment_weight

        output["is_null_level"] = self.is_null_level
        output["bayes_factor"] = self._bayes_factor
        output["log2_bayes_factor"] = self._log2_bayes_factor
        output["comparison_vector_value"] = self.comparison_vector_value
        output["max_comparison_vector_value"] = comparison_num_levels - 1
        output["bayes_factor_description"] = self._bayes_factor_description
        output["m_probability_description"] = self._m_probability_description
        output["u_probability_description"] = self._u_probability_description

        return output

    def _parameter_estimates_as_records(
        self, comparison_num_levels: int, comparison_levels: list[ComparisonLevel]
    ) -> list[dict[str, Any]]:
        output_records = []

        cl_record = self._as_detailed_record(comparison_num_levels, comparison_levels)
        trained_values = self._trained_u_probabilities + self._trained_m_probabilities
        for trained_value in trained_values:
            record = {}
            record["m_or_u"] = trained_value["m_or_u"]
            p = trained_value["probability"]
            record["estimated_probability"] = p
            record["estimate_description"] = trained_value["description"]
            if p is not None and p != LEVEL_NOT_OBSERVED_TEXT and p > 0.0 and p < 1.0:
                record["estimated_probability_as_log_odds"] = math.log2(p / (1 - p))
            else:
                record["estimated_probability_as_log_odds"] = None

            record["sql_condition"] = cl_record["sql_condition"]
            record["comparison_level_label"] = cl_record["label_for_charts"]
            record["comparison_vector_value"] = cl_record["comparison_vector_value"]
            output_records.append(record)

        return output_records

    def _validate(self):
        if not self._is_exact_match:
            if not self.max_epsilon_value and self.similarity_value:
                raise ValueError(
                    "max_epsilon_value and similarity_value must be provided when not an exact match"
                )
        self._validate_sql()

    def _abbreviated_sql(self, cutoff=75):
        sql = self.sql_condition
        return (sql[:cutoff] + "...") if len(sql) > cutoff else sql

    def __repr__(self):
        return f"<{self._human_readable_succinct}>"

    @property
    def _human_readable_succinct(self):
        sql = self._abbreviated_sql(75)
        return f"Comparison level '{self.label_for_charts}' using SQL rule: {sql}"

    @property
    def human_readable_description(self):
        input_cols = join_list_with_commas_final_and(
            [c.name for c in self._input_columns_used_by_sql_condition]
        )
        desc = (
            f"Comparison level: {self.label_for_charts} of {input_cols}\n"
            "Assesses similarity between pairwise comparisons of the input columns "
            f"using the following rule\n{self.sql_condition}"
        )

        return desc
