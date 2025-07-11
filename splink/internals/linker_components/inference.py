from __future__ import annotations

import logging
import math
import multiprocessing
import time
from typing import TYPE_CHECKING, Any

from splink.internals.accuracy import _select_found_by_blocking_rules
from splink.internals.blocking import (
    BlockingRule,
    block_using_rules_sqls,
    materialise_exploded_id_tables,
)
from splink.internals.blocking_rule_creator import BlockingRuleCreator
from splink.internals.blocking_rule_creator_utils import to_blocking_rule_creator
from splink.internals.comparison_vector_values import (
    compute_blocked_candidates_from_id_pairs_sql,
    compute_comparison_metrics_from_blocked_candidates_sql,
    compute_comparison_vector_values_from_id_pairs_sqls,
    compute_comparison_vectors_from_comparison_metrics_sql,
)
from splink.internals.database_api import AcceptableInputTableType
from splink.internals.exceptions import SplinkException
from splink.internals.find_matches_to_new_records import (
    add_unique_id_and_source_dataset_cols_if_needed,
)
from splink.internals.input_column import InputColumn
from splink.internals.misc import ascii_uid, ensure_is_list
from splink.internals.parse_sql import get_columns_used_from_sql
from splink.internals.pipeline import CTEPipeline
from splink.internals.predict import predict_from_comparison_vectors_sqls_using_settings
from splink.internals.shard_sql import shard_comparison_vectors_sql
from splink.internals.splink_dataframe import SplinkDataFrame
from splink.internals.term_frequencies import (
    _join_new_table_to_df_concat_with_tf_sql,
    colname_to_tf_tablename,
)
from splink.internals.unique_id_concat import (
    _composite_unique_id_from_edges_sql,
    _composite_unique_id_from_nodes_sql,
)
from splink.internals.vertically_concatenate import (
    compute_df_concat_with_tf,
    enqueue_df_concat_with_tf,
    split_df_concat_with_tf_into_two_tables_sqls,
)

if TYPE_CHECKING:
    from splink.internals.linker import Linker

logger = logging.getLogger(__name__)


class LinkerInference:
    """Use your Splink model to make predictions (perform inference). Accessed via
    `linker.inference`.
    """

    def __init__(self, linker: Linker):
        self._linker = linker

    def deterministic_link(self) -> SplinkDataFrame:
        """Uses the blocking rules specified by
        `blocking_rules_to_generate_predictions` in your settings to
        generate pairwise record comparisons.

        For deterministic linkage, this should be a list of blocking rules which
        are strict enough to generate only true links.

        Deterministic linkage, however, is likely to result in missed links
        (false negatives).

        Returns:
            SplinkDataFrame: A SplinkDataFrame of the pairwise comparisons.


        Examples:

            ```py
            settings = SettingsCreator(
                link_type="dedupe_only",
                blocking_rules_to_generate_predictions=[
                    block_on("first_name", "surname"),
                    block_on("dob", "first_name"),
                ],
            )

            linker = Linker(df, settings, db_api=db_api)
            splink_df = linker.inference.deterministic_link()
            ```
        """
        pipeline = CTEPipeline()
        # Allows clustering during a deterministic linkage.
        # This is used in `cluster_pairwise_predictions_at_threshold`
        # to set the cluster threshold to 1

        df_concat_with_tf = compute_df_concat_with_tf(self._linker, pipeline)
        pipeline = CTEPipeline([df_concat_with_tf])
        link_type = self._linker._settings_obj._link_type

        blocking_input_tablename_l = "__splink__df_concat_with_tf"
        blocking_input_tablename_r = "__splink__df_concat_with_tf"

        link_type = self._linker._settings_obj._link_type
        if len(self._linker._input_tables_dict) == 2 and self._linker._settings_obj._link_type == "link_only":
            sqls = split_df_concat_with_tf_into_two_tables_sqls(
                "__splink__df_concat_with_tf",
                self._linker._settings_obj.column_info_settings.source_dataset_column_name,
            )
            pipeline.enqueue_list_of_sqls(sqls)

            blocking_input_tablename_l = "__splink__df_concat_with_tf_left"
            blocking_input_tablename_r = "__splink__df_concat_with_tf_right"
            link_type = "two_dataset_link_only"

        exploding_br_with_id_tables = materialise_exploded_id_tables(
            link_type=link_type,
            blocking_rules=self._linker._settings_obj._blocking_rules_to_generate_predictions,
            db_api=self._linker._db_api,
            splink_df_dict=self._linker._input_tables_dict,
            source_dataset_input_column=self._linker._settings_obj.column_info_settings.source_dataset_input_column,
            unique_id_input_column=self._linker._settings_obj.column_info_settings.unique_id_input_column,
            drop_exploded_tables=True,
        )

        sqls = block_using_rules_sqls(
            input_tablename_l=blocking_input_tablename_l,
            input_tablename_r=blocking_input_tablename_r,
            blocking_rules=self._linker._settings_obj._blocking_rules_to_generate_predictions,
            link_type=link_type,
            source_dataset_input_column=self._linker._settings_obj.column_info_settings.source_dataset_input_column,
            unique_id_input_column=self._linker._settings_obj.column_info_settings.unique_id_input_column,
        )
        pipeline.enqueue_list_of_sqls(sqls)
        blocked_pairs = self._linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)

        pipeline = CTEPipeline([blocked_pairs, df_concat_with_tf])

        sqls = compute_comparison_vector_values_from_id_pairs_sqls(
            self._linker._settings_obj._columns_to_select_for_blocking,
            ["*"],
            input_tablename_l="__splink__df_concat_with_tf",
            input_tablename_r="__splink__df_concat_with_tf",
            source_dataset_input_column=self._linker._settings_obj.column_info_settings.source_dataset_input_column,
            unique_id_input_column=self._linker._settings_obj.column_info_settings.unique_id_input_column,
        )
        pipeline.enqueue_list_of_sqls(sqls)

        deterministic_link_df = self._linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)
        deterministic_link_df.metadata["is_deterministic_link"] = True

        [b.drop_materialised_id_pairs_dataframe() for b in exploding_br_with_id_tables]
        blocked_pairs.drop_table_from_database_and_remove_from_cache()

        return deterministic_link_df

    def _set_m_levels_from_only_help(self):
        """
        Set the m_levels from the only_help setting.
        """
        if self._linker._settings_obj.comparisons:
            for c in self._linker._settings_obj.comparisons:
                for cl in c.comparison_levels:
                    if (
                        cl.only_help
                        and cl.is_null_level is False
                        and cl.u_probability is not None
                        and cl.m_probability is not None
                        and cl.u_probability > cl.m_probability
                    ):
                        cl.m_probability = cl.u_probability
                        logger.info(
                            f"Setting m probability = u probability for {c.comparison_description} so that it can't hurt the comparison. (only_help = True)"
                        )

    def predict(
        self,
        threshold_match_probability: float = None,
        threshold_match_weight: float = None,
        materialise_after_computing_term_frequencies: bool = True,
        materialise_blocked_pairs: bool = True,
    ) -> SplinkDataFrame:
        """Create a dataframe of scored pairwise comparisons using the parameters
        of the linkage model.

        Uses the blocking rules specified in the
        `blocking_rules_to_generate_predictions` key of the settings to
        generate the pairwise comparisons.

        Args:
            threshold_match_probability (float, optional): If specified,
                filter the results to include only pairwise comparisons with a
                match_probability above this threshold. Defaults to None.
            threshold_match_weight (float, optional): If specified,
                filter the results to include only pairwise comparisons with a
                match_weight above this threshold. Defaults to None.
            materialise_after_computing_term_frequencies (bool): If true, Splink
                will materialise the table containing the input nodes (rows)
                joined to any term frequencies which have been asked
                for in the settings object.  If False, this will be
                computed as part of a large CTE pipeline.   Defaults to True
            materialise_blocked_pairs: In the blocking phase, materialise the table
                of pairs of records that will be scored

        Examples:
            ```py
            linker = linker(df, "saved_settings.json", db_api=db_api)
            splink_df = linker.inference.predict(threshold_match_probability=0.95)
            splink_df.as_pandas_dataframe(limit=5)
            ```
        Returns:
            SplinkDataFrame: A SplinkDataFrame of the scored pairwise comparisons.
        """

        self._set_m_levels_from_only_help()

        pipeline = CTEPipeline()
        df_concat_with_tf_cte = ""
        if (
            materialise_after_computing_term_frequencies
            or self._linker._sql_dialect.sql_dialect_str == "duckdb"
        ):
            df_concat_with_tf = compute_df_concat_with_tf(self._linker, pipeline)
            pipeline = CTEPipeline([df_concat_with_tf])
            df_concat_with_tf_cte = (
                f"WITH __splink__df_concat_with_tf AS (select * from {df_concat_with_tf.physical_name})"
            )
        else:
            pipeline = enqueue_df_concat_with_tf(self._linker, pipeline)

        start_time = time.time()

        blocking_input_tablename_l = "__splink__df_concat_with_tf"
        blocking_input_tablename_r = "__splink__df_concat_with_tf"

        link_type = self._linker._settings_obj._link_type
        if len(self._linker._input_tables_dict) == 2 and self._linker._settings_obj._link_type == "link_only":
            sqls = split_df_concat_with_tf_into_two_tables_sqls(
                "__splink__df_concat_with_tf",
                self._linker._settings_obj.column_info_settings.source_dataset_column_name,
            )
            pipeline.enqueue_list_of_sqls(sqls)

            blocking_input_tablename_l = "__splink__df_concat_with_tf_left"
            blocking_input_tablename_r = "__splink__df_concat_with_tf_right"
            link_type = "two_dataset_link_only"

        # If exploded blocking rules exist, we need to materialise
        # the tables of ID pairs

        exploding_br_with_id_tables = materialise_exploded_id_tables(
            link_type=link_type,
            blocking_rules=self._linker._settings_obj._blocking_rules_to_generate_predictions,
            db_api=self._linker._db_api,
            splink_df_dict=self._linker._input_tables_dict,
            source_dataset_input_column=self._linker._settings_obj.column_info_settings.source_dataset_input_column,
            unique_id_input_column=self._linker._settings_obj.column_info_settings.unique_id_input_column,
            drop_exploded_tables=True,
        )

        # ------------------------------
        # Blocking
        # ------------------------------
        # blocked id pairs
        sqls = block_using_rules_sqls(
            input_tablename_l=blocking_input_tablename_l,
            input_tablename_r=blocking_input_tablename_r,
            blocking_rules=self._linker._settings_obj._blocking_rules_to_generate_predictions,
            link_type=link_type,
            source_dataset_input_column=self._linker._settings_obj.column_info_settings.source_dataset_input_column,
            unique_id_input_column=self._linker._settings_obj.column_info_settings.unique_id_input_column,
        )
        logger.info(f"Blocking SQL: {sqls[0]['sql']}")

        self._linker._db_api._execute_sql_against_backend(
            f"""CREATE TABLE {sqls[0]['output_table_name']} AS 
            {df_concat_with_tf_cte} 
            {sqls[0]['sql']}"""
        )
        table_size = self._linker._db_api._execute_sql_against_backend(
            f"SELECT COUNT(*) FROM {sqls[0]['output_table_name']}"
        )
        logger.info(f"{sqls[0]['output_table_name']} size: {table_size}")

        blocked_count = table_size.fetchone()[0]
        if blocked_count == 0:
            raise SplinkException(
                "Blocking rules resulted in no blocked id pairs. Exiting early. Please loosen blocking rules or input more data."
            )
        logger.info(f"Processing {blocked_count} blocked pairs")

        # pipeline.enqueue_list_of_sqls(sqls)

        if materialise_blocked_pairs:
            blocked_pairs = self._linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)

            pipeline = CTEPipeline([blocked_pairs, df_concat_with_tf])
            blocking_time = time.time() - start_time
            logger.info(f"Blocking time: {blocking_time:.2f} seconds")
            start_time = time.time()

        sqls = compute_comparison_vector_values_from_id_pairs_sqls(
            self._linker._settings_obj._columns_to_select_for_blocking,
            self._linker._settings_obj._columns_to_select_for_comparison_vector_values,
            input_tablename_l=df_concat_with_tf.physical_name,
            input_tablename_r=df_concat_with_tf.physical_name,
            source_dataset_input_column=self._linker._settings_obj.column_info_settings.source_dataset_input_column,
            unique_id_input_column=self._linker._settings_obj.column_info_settings.unique_id_input_column,
        )

        logger.info(f"Blocked with cols SQL: {sqls[0]['sql']}")
        # blocked with cols df
        self._linker._db_api._execute_sql_against_backend(
            f"CREATE TABLE {sqls[0]['output_table_name']} AS {sqls[0]['sql']}"
        )
        table_size = self._linker._db_api._execute_sql_against_backend(
            f"SELECT COUNT(*) FROM {sqls[0]['output_table_name']}"
        )
        logger.info(f"{sqls[0]['output_table_name']} size: {table_size}")

        tables_result = self._linker._db_api._execute_sql_against_backend("SHOW TABLES")
        logger.info(f"Tables in db: {tables_result.fetchall()}")

        logger.info(f"Comparison vector SQL: {sqls[1]['sql']}")
        # comparison_vectors_df
        self._linker._db_api._execute_sql_against_backend(
            f"CREATE TABLE {sqls[1]['output_table_name']} AS {sqls[1]['sql']}"
        )
        table_size = self._linker._db_api._execute_sql_against_backend(
            f"SELECT COUNT(*) FROM {sqls[1]['output_table_name']}"
        )
        logger.info(f"{sqls[1]['output_table_name']} size: {table_size}")

        logger.info(f"TF array columns: {self._linker._settings_obj._tf_array_columns}")

        # Sort so 'city_state_pairs' goes first every time
        tf_array_columns_items = list(self._linker._settings_obj._tf_array_columns.items())
        tf_array_columns_items.sort(key=lambda x: (x[0] != "city_state_pairs", x[0]))

        for col_name, (
            _,
            gamma_column_name,
            gamma_levels,
        ) in tf_array_columns_items:
            col = next(
                (col for col in self._linker._input_columns() if col.input_name == col_name),
                None,
            )
            if not col:
                continue

            blocked_with_tf_table_name = f"__splink__blocked_ids_pairs_{col_name}_with_tf"
            tf_table_name = f"__splink__df_tf_{col_name}"

            # TODO: @aberdeenmorrow Fix this in the underlying tf array tables for everything except employers
            term_column_name = "term" if not col_name == "tokenized_employers" else "employers"
            tf_column_name = "tf_value" if not col_name == "tokenized_employers" else f"tf_employers"

            # Get TF parameters from comparison levels
            tf_params = self._get_tf_parameters_for_column(col_name, gamma_levels)
            N = tf_params.get("N", 226_657_846)  # Default value
            log_base = tf_params.get("log_base", 2.0)
            exact = tf_params.get("exact", False)
            fuzzy = tf_params.get("fuzzy", False)

            if exact:
                exact_gamma_levels = tf_params.get("exact_gamma_levels", [])
                gamma_count = self._linker._db_api._execute_sql_against_backend(
                    f"SELECT COUNT(*) FROM __splink__df_comparison_vectors WHERE {gamma_column_name} IN ({', '.join(str(l) for l in exact_gamma_levels)})"
                )
                logger.info(f"Gamma count for {col_name}: {gamma_count}")
                # Build the SQL
                exact_cte = f"""base AS (
                    SELECT
                        unique_id_l,
                        unique_id_r,
                        array_intersect({col.name_l}, {col.name_r}) AS common_terms
                    FROM __splink__df_comparison_vectors
                    WHERE {gamma_column_name} IN ({', '.join(str(l) for l in exact_gamma_levels)})
                )

                , {col_name}_flattened AS (
                    SELECT
                        f.unique_id_l,
                        f.unique_id_r,
                        z.term,
                        tf.{tf_column_name} AS tf_value
                    FROM base AS f
                    CROSS JOIN UNNEST(f.common_terms) AS z(term)
                    JOIN {tf_table_name} AS tf ON tf.{term_column_name} = z.term
                )
                SELECT
                    unique_id_l,
                    unique_id_r,
                    array_agg(tf_value ORDER BY tf_value) AS tf_values
                FROM {col_name}_flattened
                GROUP BY unique_id_l, unique_id_r
                HAVING array_length(array_agg(tf_value)) <= 10  -- Limit to max 10 terms for performance
                """

                sql = f"CREATE TABLE {col_name}_values AS WITH {exact_cte}"
                logger.info(f"Exact CTE SQL: {sql}")
                self._linker._db_api._execute_sql_against_backend(sql)

                # Simplified TF calculation - avoid complex subqueries
                ln_base = math.log(log_base)
                exact_table_construction_sql = f"""SELECT
                    unique_id_l,
                    unique_id_r,
                    CASE
                        WHEN array_length(tf_values) = 1 THEN
                            ({N} / tf_values[1])
                        WHEN array_length(tf_values) = 2 THEN
                            ({N} / tf_values[1]) + ((LN(2.0/1.0) / tf_values[2]) * {N / ln_base})
                        WHEN array_length(tf_values) = 3 THEN
                            ({N} / tf_values[1]) + ((LN(2.0/1.0) / tf_values[2]) * {N / ln_base}) + ((LN(3.0/2.0) / tf_values[3]) * {N / ln_base})
                        WHEN array_length(tf_values) = 4 THEN
                            ({N} / tf_values[1]) + ((LN(2.0/1.0) / tf_values[2]) * {N / ln_base}) + ((LN(3.0/2.0) / tf_values[3]) * {N / ln_base}) + ((LN(4.0/3.0) / tf_values[4]) * {N / ln_base})
                        WHEN array_length(tf_values) = 5 THEN
                            ({N} / tf_values[1]) + ((LN(2.0/1.0) / tf_values[2]) * {N / ln_base}) + ((LN(3.0/2.0) / tf_values[3]) * {N / ln_base}) + ((LN(4.0/3.0) / tf_values[4]) * {N / ln_base}) + ((LN(5.0/4.0) / tf_values[5]) * {N / ln_base})
                        ELSE
                            -- For more than 5 terms, use a simplified calculation
                            ({N} / tf_values[1]) + 
                            (SELECT SUM((LN((row_number + 1.0)/row_number) / value) * {N / ln_base})
                             FROM (SELECT value, ROW_NUMBER() OVER (ORDER BY value) AS row_number
                                   FROM UNNEST(tf_values) AS t(value)) AS numbered_values
                             WHERE row_number > 1 AND row_number <= 5)
                    END AS tf_adjustment_{col_name}
                FROM {col_name}_values
                """

                sql = f"""{exact_table_construction_sql}"""

                sql = shard_comparison_vectors_sql(
                    core_sql=sql,
                    num_shards=5,  # Reduced from 10 for better performance
                    table_name=f"{blocked_with_tf_table_name}{'_exact' if fuzzy else ''}",
                    input_table_name=f"{col_name}_values",
                    logger=logger,
                )
                logger.info(f"optimized, sharded sql: {sql}")
                self._linker._db_api._execute_sql_against_backend(sql)
                self._linker._db_api._execute_sql_against_backend(f"DROP TABLE {col_name}_values")

                preview = self._linker._db_api._execute_sql_against_backend(
                    f"SELECT * FROM {blocked_with_tf_table_name}{'_exact' if fuzzy else ''} LIMIT 20"
                )
                logger.info(f"Preview of {col_name} tf intersection table: {preview}")
            if fuzzy:
                fuzzy_gamma_levels = tf_params.get("fuzzy_gamma_levels", [])
                # Build the SQL - optimized fuzzy matching
                filtered_cte = f"""filtered AS (
                    SELECT
                        unique_id_l,
                        unique_id_r,
                        shard,
                        {col.name_l} AS terms_l,
                        {col.name_r} AS terms_r
                    FROM __splink__df_comparison_vectors
                    WHERE {gamma_column_name} IN ({', '.join(str(l) for l in fuzzy_gamma_levels)})
                )
                """

                # Optimized fuzzy matching - avoid double CROSS JOIN
                ln_base = math.log(log_base)

                fuzzy_ctes = f"""
                , fuzzy_pairs AS (
                    SELECT
                        f.unique_id_l,
                        f.unique_id_r,
                        f.shard,
                        t1.term1,
                        t2.term2,
                        GREATEST(tf1.{tf_column_name}, tf2.{tf_column_name}) AS tf_value
                    FROM filtered AS f
                    CROSS JOIN UNNEST(f.terms_l) AS t1(term1)
                    CROSS JOIN UNNEST(f.terms_r) AS t2(term2)
                    LEFT JOIN {tf_table_name} AS tf1 ON tf1.{term_column_name} = t1.term1
                    LEFT JOIN {tf_table_name} AS tf2 ON tf2.{term_column_name} = t2.term2
                    WHERE jaro_winkler_similarity(t1.term1, t2.term2) >= 0.95
                ),
                fuzzy_tf_values AS (
                    SELECT
                        unique_id_l,
                        unique_id_r,
                        shard,
                        array_agg(tf_value ORDER BY tf_value) AS tf_values
                    FROM fuzzy_pairs
                    GROUP BY unique_id_l, unique_id_r, shard
                )
                """

                # Simplified TF calculation for fuzzy matches
                sql = f"""
SELECT
    unique_id_l,
    unique_id_r,
    CASE
        WHEN array_length(tf_values) = 1 THEN
            ({N} / tf_values[1])
        WHEN array_length(tf_values) = 2 THEN
            ({N} / tf_values[1]) + ((LN(2.0/1.0) / tf_values[2]) * {N / ln_base})
        WHEN array_length(tf_values) = 3 THEN
            ({N} / tf_values[1]) + ((LN(2.0/1.0) / tf_values[2]) * {N / ln_base}) + ((LN(3.0/2.0) / tf_values[3]) * {N / ln_base})
        WHEN array_length(tf_values) = 4 THEN
            ({N} / tf_values[1]) + ((LN(2.0/1.0) / tf_values[2]) * {N / ln_base}) + ((LN(3.0/2.0) / tf_values[3]) * {N / ln_base}) + ((LN(4.0/3.0) / tf_values[4]) * {N / ln_base})
        WHEN array_length(tf_values) = 5 THEN
            ({N} / tf_values[1]) + ((LN(2.0/1.0) / tf_values[2]) * {N / ln_base}) + ((LN(3.0/2.0) / tf_values[3]) * {N / ln_base}) + ((LN(4.0/3.0) / tf_values[4]) * {N / ln_base}) + ((LN(5.0/4.0) / tf_values[5]) * {N / ln_base})
        ELSE
            -- For more than 5 terms, use a simplified calculation
            ({N} / tf_values[1]) + 
            (SELECT SUM((LN((row_number + 1.0)/row_number) / value) * {N / ln_base})
             FROM (SELECT value, ROW_NUMBER() OVER (ORDER BY value) AS row_number
                   FROM UNNEST(tf_values) AS t(value)) AS numbered_values
             WHERE row_number > 1 AND row_number <= 5)
    END AS tf_adjustment_{col_name}
FROM fuzzy_tf_values;
                """
                sql = shard_comparison_vectors_sql(
                    core_sql=sql,
                    table_name=f"{blocked_with_tf_table_name}{'_fuzzy' if exact else ''}",
                    input_table_name="__splink__df_comparison_vectors",
                    pre_shard_cte=f"{filtered_cte} {fuzzy_ctes} ",
                    num_shards=5,  # Reduced from 10 for better performance
                    logger=logger,
                )

                logger.info(f"Optimized, sharded SQL:\n{sql}")
                self._linker._db_api._execute_sql_against_backend(sql)

                preview = self._linker._db_api._execute_sql_against_backend(
                    f"SELECT * FROM {blocked_with_tf_table_name}{'_fuzzy' if exact else ''} LIMIT 20"
                )
                logger.info(f"Preview of {col_name} tf intersection table: {preview}")

                if exact:
                    merge_sql = f"""CREATE TABLE {blocked_with_tf_table_name} AS 
                    SELECT * FROM {blocked_with_tf_table_name}_exact
                    UNION ALL
                    SELECT * FROM {blocked_with_tf_table_name}_fuzzy;"""
                    logger.info("Merging fuzzy and exact tables into one bf_tf table")
                    logger.info(merge_sql)
                    self._linker._db_api._execute_sql_against_backend(merge_sql)

                preview = self._linker._db_api._execute_sql_against_backend(
                    f"SELECT * FROM {blocked_with_tf_table_name} LIMIT 20"
                )
                logger.info(f"Preview of {col_name} tf intersection table: {preview}")

        # pipeline.enqueue_list_of_sqls(sqls)
        sqls = predict_from_comparison_vectors_sqls_using_settings(
            self._linker._settings_obj,
            threshold_match_probability,
            threshold_match_weight,
            sql_infinity_expression=self._linker._infinity_expression,
        )
        # __splink__df_match_weight_parts
        try:
            # sql = f"CREATE TABLE {sqls[0]['output_table_name']} AS {sqls[0]['sql']}"
            sql = shard_comparison_vectors_sql(
                core_sql=sqls[0]["sql"],
                table_name=sqls[0]["output_table_name"],
                input_table_name="__splink__df_comparison_vectors",
                logger=logger,
                num_shards=30,  # Reduced from 100 for better performance
            )
            logger.info(f"Optimized Predict SQL: {sql}")
            self._linker._db_api._execute_sql_against_backend(sql)
        except Exception as e:
            logger.error(f"Error creating table {sqls[0]['output_table_name']}: {e}")
            logger.error(f"SQL: {sql}")
            raise e

        table_size = self._linker._db_api._execute_sql_against_backend(
            f"SELECT COUNT(*) FROM {sqls[0]['output_table_name']}"
        )
        logger.info(f"{sqls[0]['output_table_name']} size: {table_size}")
        logger.info(f"Predict SQL 2: {sqls[1]['sql']}")
        # __splink__df_predict
        predictions = self._linker._db_api._execute_sql_against_backend(
            f"CREATE TABLE {sqls[1]['output_table_name']} AS {sqls[1]['sql']}"
        )
        predictions = self._linker._db_api.table_to_splink_dataframe(
            sqls[1]["output_table_name"], sqls[1]["output_table_name"]
        )
        table_size = self._linker._db_api._execute_sql_against_backend(
            f"SELECT COUNT(*) FROM {sqls[1]['output_table_name']}"
        )
        logger.info(f"{sqls[1]['output_table_name']} size: {table_size}")
        # pipeline.enqueue_list_of_sqls(sqls)

        # predictions = self._linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)

        predict_time = time.time() - start_time
        logger.info(f"Predict time: {predict_time:.2f} seconds")

        self._linker._predict_warning()

        [b.drop_materialised_id_pairs_dataframe() for b in exploding_br_with_id_tables]
        if materialise_blocked_pairs:
            blocked_pairs.drop_table_from_database_and_remove_from_cache()

        return predictions

    def _score_missing_cluster_edges(
        self,
        df_clusters: SplinkDataFrame,
        df_predict: SplinkDataFrame = None,
        threshold_match_probability: float = None,
        threshold_match_weight: float = None,
    ) -> SplinkDataFrame:
        """
        Given a table of clustered records, create a dataframe of scored
        pairwise comparisons for all pairs of records that belong to the same cluster.

        If you also supply a scored edges table, this will only return pairwise
        comparisons that are not already present in your scored edges table.

        Args:
            df_clusters (SplinkDataFrame): A table of clustered records, such
                as the output of
                `linker.clustering.cluster_pairwise_predictions_at_threshold()`.
                All edges within the same cluster as specified by this table will
                be scored.
                Table needs cluster_id, id columns, and any columns used in
                model comparisons.
            df_predict (SplinkDataFrame, optional): An edges table, the output of
                `linker.inference.predict()`.
                If supplied, resulting table will not include any edges already
                included in this table.
            threshold_match_probability (float, optional): If specified,
                filter the results to include only pairwise comparisons with a
                match_probability above this threshold. Defaults to None.
            threshold_match_weight (float, optional): If specified,
                filter the results to include only pairwise comparisons with a
                match_weight above this threshold. Defaults to None.

        Examples:
            ```py
            linker = linker(df, "saved_settings.json", db_api=db_api)
            df_edges = linker.inference.predict()
            df_clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
                df_edges,
                0.9,
            )
            df_remaining_edges = linker._score_missing_cluster_edges(
                df_clusters,
                df_edges,
            )
            df_remaining_edges.as_pandas_dataframe(limit=5)
            ```
        Returns:
            SplinkDataFrame: A SplinkDataFrame of the scored pairwise comparisons.
        """

        start_time = time.time()

        source_dataset_input_column = (
            self._linker._settings_obj.column_info_settings.source_dataset_input_column
        )
        unique_id_input_column = self._linker._settings_obj.column_info_settings.unique_id_input_column

        pipeline = CTEPipeline()
        enqueue_df_concat_with_tf(self._linker, pipeline)
        # we need to adjoin tf columns onto clusters table now
        # also alias cluster_id so that it doesn't interfere with existing column
        sql = f"""
        SELECT
            c.cluster_id AS _cluster_id,
            ctf.*
        FROM
            {df_clusters.physical_name} c
        LEFT JOIN
            __splink__df_concat_with_tf ctf
        ON
            c.{unique_id_input_column.name} = ctf.{unique_id_input_column.name}
        """
        if source_dataset_input_column:
            sql += f" AND c.{source_dataset_input_column.name} = " f"ctf.{source_dataset_input_column.name}"
        sqls = [
            {
                "sql": sql,
                "output_table_name": "__splink__df_clusters_renamed",
            }
        ]
        blocking_input_tablename_l = "__splink__df_clusters_renamed"
        blocking_input_tablename_r = "__splink__df_clusters_renamed"

        link_type = self._linker._settings_obj._link_type
        sqls.extend(
            block_using_rules_sqls(
                input_tablename_l=blocking_input_tablename_l,
                input_tablename_r=blocking_input_tablename_r,
                blocking_rules=[BlockingRule("l._cluster_id = r._cluster_id")],
                link_type=link_type,
                source_dataset_input_column=source_dataset_input_column,
                unique_id_input_column=unique_id_input_column,
            )
        )
        # we are going to insert an intermediate table, so rename this
        sqls[-1]["output_table_name"] = "__splink__raw_blocked_id_pairs"

        sql = """
        SELECT ne.*
        FROM __splink__raw_blocked_id_pairs ne
        """
        if df_predict is not None:
            # if we are given edges, we left join them, and then keep only rows
            # where we _didn't_ have corresponding rows in edges table
            if source_dataset_input_column:
                unique_id_columns = [
                    source_dataset_input_column,
                    unique_id_input_column,
                ]
            else:
                unique_id_columns = [unique_id_input_column]
            uid_l_expr = _composite_unique_id_from_edges_sql(unique_id_columns, "l")
            uid_r_expr = _composite_unique_id_from_edges_sql(unique_id_columns, "r")
            sql_predict_with_join_keys = f"""
                SELECT *, {uid_l_expr} AS join_key_l, {uid_r_expr} AS join_key_r
                FROM {df_predict.physical_name}
            """
            sqls.append(
                {
                    "sql": sql_predict_with_join_keys,
                    "output_table_name": "__splink__df_predict_with_join_keys",
                }
            )

            sql = f"""
            {sql}
            LEFT JOIN __splink__df_predict_with_join_keys oe
            ON oe.join_key_l = ne.join_key_l AND oe.join_key_r = ne.join_key_r
            WHERE oe.join_key_l IS NULL AND oe.join_key_r IS NULL
            """

        sqls.append({"sql": sql, "output_table_name": "__splink__blocked_id_pairs"})

        pipeline.enqueue_list_of_sqls(sqls)

        sqls = compute_comparison_vector_values_from_id_pairs_sqls(
            self._linker._settings_obj._columns_to_select_for_blocking,
            self._linker._settings_obj._columns_to_select_for_comparison_vector_values,
            input_tablename_l=blocking_input_tablename_l,
            input_tablename_r=blocking_input_tablename_r,
            source_dataset_input_column=self._linker._settings_obj.column_info_settings.source_dataset_input_column,
            unique_id_input_column=self._linker._settings_obj.column_info_settings.unique_id_input_column,
        )
        pipeline.enqueue_list_of_sqls(sqls)

        sqls = predict_from_comparison_vectors_sqls_using_settings(
            self._linker._settings_obj,
            threshold_match_probability,
            threshold_match_weight,
            sql_infinity_expression=self._linker._infinity_expression,
        )
        sqls[-1]["output_table_name"] = "__splink__df_predict_missing_cluster_edges"
        pipeline.enqueue_list_of_sqls(sqls)

        predictions = self._linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)

        predict_time = time.time() - start_time
        logger.info(f"Predict time: {predict_time:.2f} seconds")

        self._linker._predict_warning()
        return predictions

    def find_matches_to_new_records(
        self,
        records_or_tablename: AcceptableInputTableType | str,
        blocking_rules: (
            list[BlockingRuleCreator | dict[str, Any] | str] | BlockingRuleCreator | dict[str, Any] | str
        ) = [],
        match_weight_threshold: float = -4,
    ) -> SplinkDataFrame:
        """Given one or more records, find records in the input dataset(s) which match
        and return in order of the Splink prediction score.

        This effectively provides a way of searching the input datasets
        for given record(s)

        Args:
            records_or_tablename (List[dict]): Input search record(s) as list of dict,
                or a table registered to the database.
            blocking_rules (list, optional): Blocking rules to select
                which records to find and score. If [], do not use a blocking
                rule - meaning the input records will be compared to all records
                provided to the linker when it was instantiated. Defaults to [].
            match_weight_threshold (int, optional): Return matches with a match weight
                above this threshold. Defaults to -4.

        Examples:
            ```py
            linker = Linker(df, "saved_settings.json", db_api=db_api)

            # You should load or pre-compute tf tables for any tables with
            # term frequency adjustments
            linker.table_management.compute_tf_table("first_name")
            # OR
            linker.table_management.register_term_frequency_lookup(df, "first_name")

            record = {'unique_id': 1,
                'first_name': "John",
                'surname': "Smith",
                'dob': "1971-05-24",
                'city': "London",
                'email': "john@smith.net"
                }
            df = linker.inference.find_matches_to_new_records(
                [record], blocking_rules=[]
            )
            ```

        Returns:
            SplinkDataFrame: The pairwise comparisons.
        """

        original_blocking_rules = self._linker._settings_obj._blocking_rules_to_generate_predictions
        original_link_type = self._linker._settings_obj._link_type

        blocking_rule_list = ensure_is_list(blocking_rules)

        if not isinstance(records_or_tablename, str):
            uid = ascii_uid(8)
            new_records_tablename = f"__splink__df_new_records_{uid}"
            self._linker.table_management.register_table(
                records_or_tablename, new_records_tablename, overwrite=True
            )

        else:
            new_records_tablename = records_or_tablename

        new_records_df = self._linker._db_api.table_to_splink_dataframe(
            "__splink__df_new_records", new_records_tablename
        )

        pipeline = CTEPipeline()
        nodes_with_tf = compute_df_concat_with_tf(self._linker, pipeline)

        pipeline = CTEPipeline([nodes_with_tf, new_records_df])
        if len(blocking_rule_list) == 0:
            blocking_rule_list = ["1=1"]

        blocking_rule_list = [
            to_blocking_rule_creator(br).get_blocking_rule(self._linker._db_api.sql_dialect.sql_dialect_str)
            for br in blocking_rule_list
        ]
        for n, br in enumerate(blocking_rule_list):
            br.add_preceding_rules(blocking_rule_list[:n])

        self._linker._settings_obj._blocking_rules_to_generate_predictions = blocking_rule_list

        pipeline = add_unique_id_and_source_dataset_cols_if_needed(
            self._linker,
            new_records_df,
            pipeline,
            in_tablename="__splink__df_new_records",
            out_tablename="__splink__df_new_records_uid_fix",
        )
        settings = self._linker._settings_obj
        sqls = block_using_rules_sqls(
            input_tablename_l="__splink__df_concat_with_tf",
            input_tablename_r="__splink__df_new_records_uid_fix",
            blocking_rules=blocking_rule_list,
            link_type="two_dataset_link_only",
            source_dataset_input_column=settings.column_info_settings.source_dataset_input_column,
            unique_id_input_column=settings.column_info_settings.unique_id_input_column,
        )
        pipeline.enqueue_list_of_sqls(sqls)

        blocked_pairs = self._linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)

        pipeline = CTEPipeline([blocked_pairs, new_records_df, nodes_with_tf])

        cache = self._linker._intermediate_table_cache
        for tf_col in self._linker._settings_obj._term_frequency_columns:
            tf_table_name = colname_to_tf_tablename(tf_col)
            if tf_table_name in cache:
                tf_table = cache.get_with_logging(tf_table_name)
                pipeline.append_input_dataframe(tf_table)
            else:
                if "__splink__df_concat_with_tf" not in cache:
                    logger.warning(
                        f"No term frequencies found for column {tf_col.name}.\n"
                        "To apply term frequency adjustments, you need to register"
                        " a lookup using "
                        "`linker.table_management.register_term_frequency_lookup`."
                    )

        sql = _join_new_table_to_df_concat_with_tf_sql(self._linker, "__splink__df_new_records")
        pipeline.enqueue_sql(sql, "__splink__df_new_records_with_tf_before_uid_fix")

        pipeline = add_unique_id_and_source_dataset_cols_if_needed(
            self._linker,
            new_records_df,
            pipeline,
            in_tablename="__splink__df_new_records_with_tf_before_uid_fix",
            out_tablename="__splink__df_new_records_with_tf",
        )

        df_comparison_vectors = self._comparison_vectors()

        pipeline = CTEPipeline([df_comparison_vectors, new_records_df])
        sqls = predict_from_comparison_vectors_sqls_using_settings(
            self._linker._settings_obj,
            sql_infinity_expression=self._linker._infinity_expression,
        )
        pipeline.enqueue_list_of_sqls(sqls)

        sql = f"""
        select * from __splink__df_predict
        where match_weight > {match_weight_threshold}
        """

        pipeline.enqueue_sql(sql, "__splink__find_matches_predictions")

        predictions = self._linker._db_api.sql_pipeline_to_splink_dataframe(pipeline, use_cache=False)

        self._linker._settings_obj._blocking_rules_to_generate_predictions = original_blocking_rules
        self._linker._settings_obj._link_type = original_link_type

        blocked_pairs.drop_table_from_database_and_remove_from_cache()

        return predictions

    def compare_two_records(
        self,
        record_1: dict[str, Any] | AcceptableInputTableType,
        record_2: dict[str, Any] | AcceptableInputTableType,
        include_found_by_blocking_rules: bool = False,
    ) -> SplinkDataFrame:
        """Use the linkage model to compare and score a pairwise record comparison
        based on the two input records provided.

        If your inputs contain multiple rows, scores for the cartesian product of
        the two inputs will be returned.

        If your inputs contain hardcoded term frequency columns (e.g.
        a tf_first_name column), then these values will be used instead of any
        provided term frequency lookup tables. or term frequency values derived
        from the input data.

        Args:
            record_1 (dict): dictionary representing the first record.  Columns names
                and data types must be the same as the columns in the settings object
            record_2 (dict): dictionary representing the second record.  Columns names
                and data types must be the same as the columns in the settings object
            include_found_by_blocking_rules (bool, optional): If True, outputs a column
                indicating whether the record pair would have been found by any of the
                blocking rules specified in
                settings.blocking_rules_to_generate_predictions. Defaults to False.

        Examples:
            ```py
            linker = Linker(df, "saved_settings.json", db_api=db_api)

            # You should load or pre-compute tf tables for any tables with
            # term frequency adjustments
            linker.table_management.compute_tf_table("first_name")
            # OR
            linker.table_management.register_term_frequency_lookup(df, "first_name")

            record_1 = {'unique_id': 1,
                'first_name': "John",
                'surname': "Smith",
                'dob': "1971-05-24",
                'city': "London",
                'email': "john@smith.net"
                }

            record_2 = {'unique_id': 1,
                'first_name': "Jon",
                'surname': "Smith",
                'dob': "1971-05-23",
                'city': "London",
                'email': "john@smith.net"
                }
            df = linker.inference.compare_two_records(record_1, record_2)

            ```

        Returns:
            SplinkDataFrame: Pairwise comparison with scored prediction
        """

        linker = self._linker

        retain_matching_columns = linker._settings_obj._retain_matching_columns
        retain_intermediate_calculation_columns = (
            linker._settings_obj._retain_intermediate_calculation_columns
        )
        linker._settings_obj._retain_matching_columns = True
        linker._settings_obj._retain_intermediate_calculation_columns = True

        cache = linker._intermediate_table_cache

        uid = ascii_uid(8)

        # Check if input is a DuckDB relation without importing DuckDB
        if isinstance(record_1, dict):
            to_register_left: AcceptableInputTableType = [record_1]
        else:
            to_register_left = record_1

        if isinstance(record_2, dict):
            to_register_right: AcceptableInputTableType = [record_2]
        else:
            to_register_right = record_2

        df_records_left = linker.table_management.register_table(
            to_register_left,
            f"__splink__compare_two_records_left_{uid}",
            overwrite=True,
        )

        df_records_left.templated_name = "__splink__compare_two_records_left"

        df_records_right = linker.table_management.register_table(
            to_register_right,
            f"__splink__compare_two_records_right_{uid}",
            overwrite=True,
        )
        df_records_right.templated_name = "__splink__compare_two_records_right"

        pipeline = CTEPipeline([df_records_left, df_records_right])

        if "__splink__df_concat_with_tf" in cache:
            nodes_with_tf = cache.get_with_logging("__splink__df_concat_with_tf")
            pipeline.append_input_dataframe(nodes_with_tf)

        tf_cols = linker._settings_obj._term_frequency_columns

        for tf_col in tf_cols:
            tf_table_name = colname_to_tf_tablename(tf_col)
            if tf_table_name in cache:
                tf_table = cache.get_with_logging(tf_table_name)
                pipeline.append_input_dataframe(tf_table)
            else:
                if "__splink__df_concat_with_tf" not in cache:
                    logger.warning(
                        f"No term frequencies found for column {tf_col.name}.\n"
                        "To apply term frequency adjustments, you need to register"
                        " a lookup using "
                        "`linker.table_management.register_term_frequency_lookup`."
                    )

        sql_join_tf = _join_new_table_to_df_concat_with_tf_sql(
            linker, "__splink__compare_two_records_left", df_records_left
        )

        pipeline.enqueue_sql(sql_join_tf, "__splink__compare_two_records_left_with_tf")

        sql_join_tf = _join_new_table_to_df_concat_with_tf_sql(
            linker, "__splink__compare_two_records_right", df_records_right
        )

        pipeline.enqueue_sql(sql_join_tf, "__splink__compare_two_records_right_with_tf")

        pipeline = add_unique_id_and_source_dataset_cols_if_needed(
            linker,
            df_records_left,
            pipeline,
            in_tablename="__splink__compare_two_records_left_with_tf",
            out_tablename="__splink__compare_two_records_left_with_tf_uid_fix",
            uid_str="_left",
        )
        pipeline = add_unique_id_and_source_dataset_cols_if_needed(
            linker,
            df_records_right,
            pipeline,
            in_tablename="__splink__compare_two_records_right_with_tf",
            out_tablename="__splink__compare_two_records_right_with_tf_uid_fix",
            uid_str="_right",
        )

        cols_to_select = self._linker._settings_obj._columns_to_select_for_blocking

        select_expr = ", ".join(cols_to_select)
        sql = f"""
        select {select_expr}, 0 as match_key
        from __splink__compare_two_records_left_with_tf_uid_fix as l
        cross join __splink__compare_two_records_right_with_tf_uid_fix as r
        """
        pipeline.enqueue_sql(sql, "__splink__compare_two_records_blocked")

        cols_to_select = linker._settings_obj._columns_to_select_for_comparison_vector_values
        select_expr = ", ".join(cols_to_select)
        sql = f"""
        select {select_expr}
        from __splink__compare_two_records_blocked
        """
        pipeline.enqueue_sql(sql, "__splink__df_comparison_vectors")

        sqls = predict_from_comparison_vectors_sqls_using_settings(
            linker._settings_obj,
            sql_infinity_expression=linker._infinity_expression,
        )
        pipeline.enqueue_list_of_sqls(sqls)

        if include_found_by_blocking_rules:
            br_col = _select_found_by_blocking_rules(linker._settings_obj)
            sql = f"""
            select *, {br_col}
            from __splink__df_predict
            """

            pipeline.enqueue_sql(sql, "__splink__found_by_blocking_rules")

        predictions = linker._db_api.sql_pipeline_to_splink_dataframe(pipeline, use_cache=False)

        linker._settings_obj._retain_matching_columns = retain_matching_columns
        linker._settings_obj._retain_intermediate_calculation_columns = (
            retain_intermediate_calculation_columns
        )

        return predictions

    def _comparison_vectors(self) -> SplinkDataFrame:
        pipeline = CTEPipeline()
        nodes_with_tf = compute_df_concat_with_tf(self._linker, pipeline)

        settings = self._linker._settings_obj
        source_dataset_input_column = settings.column_info_settings.source_dataset_input_column
        unique_id_input_column = settings.column_info_settings.unique_id_input_column
        unique_id_cols: list[InputColumn] = [unique_id_input_column]
        if source_dataset_input_column:
            unique_id_cols.append(source_dataset_input_column)
        else:
            unique_id_cols.append(
                InputColumn(
                    raw_column_name_or_column_reference="source_dataset",
                    sqlglot_dialect_str=self._linker._db_api.sql_dialect.sqlglot_dialect,
                )
            )

        blocking_rule_sqls = [
            br.blocking_rule_sql for br in self._linker._settings_obj._blocking_rules_to_generate_predictions
        ]
        blocking_rule_sql = " AND ".join(blocking_rule_sqls)
        cols_used_from_br_sql = get_columns_used_from_sql(
            blocking_rule_sql, sqlglot_dialect=self._linker._db_api.sql_dialect.sqlglot_dialect
        )
        column_names = list(
            set(
                cols_used_from_br_sql + [col.input_name for col in unique_id_cols if col is not None],
            )
        )
        required_cols = ", ".join(column_names)

        pipeline = CTEPipeline()
        sql = f"SELECT {required_cols} FROM {nodes_with_tf.physical_name}"
        pipeline.enqueue_sql(sql, "__splink__df_concat_with_tf_select_cols")
        nodes_with_tf_select_cols = self._linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)

        pipeline = CTEPipeline()
        sqls = block_using_rules_sqls(
            input_tablename_l=nodes_with_tf_select_cols.physical_name,
            input_tablename_r=nodes_with_tf_select_cols.physical_name,
            blocking_rules=[self._linker._settings_obj._blocking_rules_to_generate_predictions[0]],
            link_type=settings._link_type,
            source_dataset_input_column=source_dataset_input_column,
            unique_id_input_column=unique_id_input_column,
            join_key_col_name=join_key_col_name,
        )
        pipeline.enqueue_list_of_sqls(sqls)

        logger.info(f"Blocking pairs")
        blocked_pairs = self._linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)

        # pipeline = CTEPipeline([blocked_pairs, nodes_with_tf])

        # Generate blocked candidates
        pipeline = CTEPipeline()
        blocked_candidates_sql = compute_blocked_candidates_from_id_pairs_sql(
            settings._columns_to_select_for_blocking,
            blocked_pairs_table_name=blocked_pairs.physical_name,
            df_concat_with_tf_table_name=nodes_with_tf.physical_name,
            source_dataset_input_column=source_dataset_input_column,
            unique_id_input_column=unique_id_input_column,
            # TODO: @aberdeenmorrow check this
            needs_matchkey_column=True,
        )

        pipeline.enqueue_sql(blocked_candidates_sql, "__splink__df_blocked_candidates")
        logger.info(f"Computing blocked candidates")
        blocked_candidates = self._linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)

        # Generate comparison metrics
        pipeline = CTEPipeline()
        logger.info("Generating comparison metrics")
        comparison_metrics_sql = compute_comparison_metrics_from_blocked_candidates_sql(
            unique_id_input_columns=unique_id_cols,
            comparisons=settings.comparisons,
            retain_matching_columns=False,
            additional_columns_to_retain=[],
            needs_matchkey_column=True,
            blocked_candidates_table_name=blocked_candidates.physical_name,
        )
        pipeline.enqueue_sql(comparison_metrics_sql, "__splink__df_comparison_metrics")
        logger.info(f"Computing comparison metrics")
        comparison_metrics = self._linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)

        # Generate comparison vectors
        pipeline = CTEPipeline()
        logger.info("Generating comparison vectors")
        comparison_vectors_sql = compute_comparison_vectors_from_comparison_metrics_sql(
            comparison_metrics_table_name=comparison_metrics.physical_name,
            unique_id_input_columns=unique_id_cols,
            comparisons=settings.comparisons,
            retain_matching_columns=False,
            additional_columns_to_retain=[],
            needs_matchkey_column=True,
        )
        pipeline.enqueue_sql(comparison_vectors_sql, "__splink__df_comparison_vectors")
        logger.info(f"Computing comparison vector values")
        return self._linker._db_api.sql_pipeline_to_splink_dataframe(pipeline)

    def _get_tf_parameters_for_column(self, col_name: str, gamma_levels: list[int]) -> dict:
        """Get TF parameters for a specific column from comparison levels"""
        tf_params = {}

        # Get the total records in field from the global variable
        from splink.internals.comparison_level import total_records_in_field

        tf_params["N"] = total_records_in_field.get(col_name, 226_657_846)

        # Find the comparison level that uses this column for TF adjustments
        for comparison in self._linker._settings_obj.core_model_settings.comparisons:
            for cl in comparison.comparison_levels:
                if (
                    cl._has_tf_adjustments
                    and cl._tf_adjustment_input_column
                    and cl._tf_adjustment_input_column.unquote().name.replace(" ", "_") == col_name
                    and cl.comparison_vector_value in gamma_levels
                ):
                    tf_params["log_base"] = getattr(cl, "log_base", 2.0)
                    tf_params["exact"] = True
                    if "exact_gamma_levels" not in tf_params:
                        tf_params["exact_gamma_levels"] = []

                    # TODO @aberdeenmorrow this pains me. fix how terrible this is.
                    if not (
                        col_name == "street_addresses" and cl.comparison_vector_value == min(gamma_levels)
                    ):
                        tf_params["exact_gamma_levels"].append(cl.comparison_vector_value)

                    # TODO @aberdeenmorrow fix these to read fuzzy matches dynamically
                    if col_name == "street_addresses" and cl.comparison_vector_value == min(gamma_levels):
                        tf_params["fuzzy"] = True
                        if "fuzzy_gamma_levels" not in tf_params:
                            tf_params["fuzzy_gamma_levels"] = []
                        tf_params["fuzzy_gamma_levels"].append(cl.comparison_vector_value)
            if "log_base" in tf_params:
                break

        logger.info(f"tf_params for {col_name}: {tf_params}")
        return tf_params
