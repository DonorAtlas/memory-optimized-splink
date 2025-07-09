from __future__ import annotations

import logging
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

        if table_size.fetchone()[0] == 0:
            raise SplinkException(
                "Blocking rules resulted in no blocked id pairs. Exiting early. Please loosen blocking rules or input more data."
            )

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

        for col_name, (
            _,
            gamma_column_name,
            gamma_levels,
        ) in self._linker._settings_obj._tf_array_columns.items():
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
            logger.info(f"tf_table_name: {tf_table_name}")

            # TODO: @aberdeenmorrow implement this as a CTE and check for the templated name in comparison_level
            # TODO: @aberdeenmorrow Implement this check
            tf_array_on_any_fuzzy_comparison = True if col_name == "street_addresses" else False
            #     fuzzy_array_construction_sql = f"""
            #         list_filter(
            #             flatten(
            #             list_transform({col.name_l},
            #                 x -> list_transform({col.name_r},
            #                     y -> list_value(x, y)
            #                 )
            #             )
            #             ),
            #             pair -> jaro_winkler_similarity(list_extract(pair, 1), list_extract(pair, 2)) >= 0.95 -- TODO: make this the dynamic parsed threshold
            #         )
            #         """
            #     fuzzy_tf_intersection_sql = f"""
            #     WHEN array_length(fuzzy_array) > 0
            #         THEN
            #             (
            #                 SELECT ARRAY_AGG(
            #                     CASE
            #                         WHEN tf1.{tf_column_name} >= tf2.{tf_column_name} THEN tf1.{tf_column_name}
            #                         ELSE tf2.{tf_column_name}
            #                     END
            #                     ORDER BY
            #                         CASE
            #                             WHEN tf1.{tf_column_name} >= tf2.{tf_column_name} THEN tf1.{tf_column_name}
            #                             ELSE tf2.{tf_column_name}
            #                         END ASC
            #                 )
            #                 FROM (
            #                     SELECT DISTINCT list_extract(pair, 1) AS term1, list_extract(pair, 2) AS term2
            #                     FROM UNNEST(fuzzy_array) AS t(pair)
            #                 ) AS terms
            #                 LEFT JOIN {tf_table_name} AS tf1
            #                     ON tf1.{term_column_name} = terms.term1
            #                 LEFT JOIN {tf_table_name} AS tf2
            #                     ON tf2.{term_column_name} = terms.term2
            #             )
            #         ELSE
            #             NULL"""

            #     fuzzy_size_of_array_sql = f"ELSE array_length(fuzzy_array)"
            #     fuzzy_array_exists_sql = f"OR (array_length(fuzzy_array) > 0)"

            #     sql = f"""WITH cte AS (
            #     SELECT
            #         cv.*,
            #         -- compute fuzzy_array once per row
            #         array_intersect({col.name_l}, {col.name_r}) AS exact_match_array,
            #         {fuzzy_array_construction_sql} AS fuzzy_array
            #     FROM {sqls[0]['output_table_name']} AS cv
            #     )
            #     """
            # else:
            #     fuzzy_tf_intersection_sql, fuzzy_size_of_array_sql = "ELSE NULL", "ELSE NULL"
            #     fuzzy_array_exists_sql = ""
            #     sql = f"""WITH cte AS (SELECT cv.*, array_intersect({col.name_l}, {col.name_r}) AS exact_match_array FROM {sqls[0]['output_table_name']} AS cv)
            #     """

            # # TODO: @aberdeenmorrow pre-calculate arrays in a CTE
            # sql += f"""SELECT DISTINCT
            #     cv.unique_id_l,
            #     cv.unique_id_r,
            #     CASE
            #         WHEN array_length(exact_match_array) > 0 THEN
            #             array_length(exact_match_array)
            #             {fuzzy_size_of_array_sql}
            #     END AS size_of_intersection_{col_name},
            #     CASE
            #         WHEN array_length(exact_match_array) > 0 THEN
            #             (
            #                 SELECT
            #                     ARRAY_AGG(tf.{tf_column_name} ORDER BY tf.{tf_column_name} ASC)
            #                 FROM (
            #                     SELECT DISTINCT t.term
            #                     FROM UNNEST(exact_match_array) AS t(term)
            #                 ) AS terms
            #                 JOIN {tf_table_name} AS tf
            #                 ON tf.{term_column_name} = terms.term
            #             )
            #         {fuzzy_tf_intersection_sql}
            #     END AS sorted_tfs_of_intersection_{col_name},
            #     CASE WHEN ({tf_array_on_any_fuzzy_comparison} AND array_length(exact_match_array) = 0) THEN TRUE ELSE FALSE END as is_fuzzy_comparison
            #     FROM cte AS cv
            #     WHERE
            #         (array_length(exact_match_array) > 0) {fuzzy_array_exists_sql}
            #     """

            # ------------------------------------------------------------------------------------------------------------

            # logger.info(f"Creating index on __splink__df_tf_{col_name} on {term_column_name}")
            # self._linker._db_api._execute_sql_against_backend(
            #     f"CREATE INDEX __splink__df_tf_{col_name}_idx ON __splink__df_tf_{col_name}({term_column_name});"
            # )

            # Build the SQL
            filtered_cte = f"""
            filtered_cv AS (
                SELECT
                    unique_id_l,
                    unique_id_r,
                    {col.name_l},
                    {col.name_r}
                FROM __splink__df_comparison_vectors
                WHERE {gamma_column_name} IN ({', '.join(str(l) for l in gamma_levels)})
            )
            """

            exact_cte = f"""
            , exact_matches AS (
                SELECT
                    cv.unique_id_l,
                    cv.unique_id_r,
                    COUNT(*) AS size_of_intersection_{col_name},
                    ARRAY_AGG(tf.{tf_column_name} ORDER BY tf.{tf_column_name})
                        AS sorted_tfs_of_intersection_{col_name},
                    FALSE AS is_fuzzy_comparison
                FROM filtered_cv AS cv
                CROSS JOIN UNNEST(
                    array_intersect(
                        cv.{col.name_l},
                        cv.{col.name_r}
                    )
                ) AS t(term)
                JOIN {tf_table_name} AS tf
                ON tf.{term_column_name} = t.term
                GROUP BY
                cv.unique_id_l,
                cv.unique_id_r
            )
            """

            fuzzy_ctes = f"""
            , exploded_l AS (
                SELECT unique_id_l, unique_id_r, term1
                FROM filtered_cv
                CROSS JOIN UNNEST(filtered_cv.{col.name_l}) AS t1(term1)
            ),
            exploded_r AS (
                SELECT unique_id_l, unique_id_r, term2
                FROM filtered_cv
                CROSS JOIN UNNEST(filtered_cv.{col.name_r}) AS t2(term2)
            ),
            fuzzy_pairs AS (
                SELECT
                l.unique_id_l,
                l.unique_id_r,
                l.term1,
                r.term2
                FROM exploded_l l
                JOIN exploded_r r
                ON l.unique_id_l = r.unique_id_l
                AND l.unique_id_r = r.unique_id_r
                AND jaro_winkler_similarity(l.term1, r.term2) >= 0.95
            ),
            fuzzy_matches AS (
                SELECT
                fp.unique_id_l,
                fp.unique_id_r,
                COUNT(*) AS size_of_intersection_{col_name},
                ARRAY_AGG(
                    GREATEST(tf1.{tf_column_name}, tf2.{tf_column_name})
                    ORDER BY GREATEST(tf1.{tf_column_name}, tf2.{tf_column_name})
                ) AS sorted_tfs_of_intersection_{col_name},
                TRUE AS is_fuzzy_comparison
                FROM fuzzy_pairs fp
                LEFT JOIN {tf_table_name} tf1
                ON tf1.{term_column_name} = fp.term1
                LEFT JOIN {tf_table_name} tf2
                ON tf2.{term_column_name} = fp.term2
                GROUP BY fp.unique_id_l, fp.unique_id_r
            )
            """

            if tf_array_on_any_fuzzy_comparison:
                sql = f"""
                CREATE TABLE {blocked_with_tf_table_name} AS
                WITH
                {filtered_cte}
                {exact_cte}
                {fuzzy_ctes}
                SELECT * FROM exact_matches
                UNION ALL
                SELECT * FROM fuzzy_matches;
                """
            else:
                sql = f"""
                CREATE TABLE {blocked_with_tf_table_name} AS
                WITH
                {filtered_cte}
                {exact_cte}
                SELECT * FROM exact_matches;
                """

            logger.info(f"Optimized SQL:\n{sql}")
            self._linker._db_api._execute_sql_against_backend(sql)

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
        logger.info(f"Predict SQL: {sqls[0]['sql']}")
        # __splink__df_match_weight_parts
        try:
            sql = f"CREATE TABLE {sqls[0]['output_table_name']} AS {sqls[0]['sql']}"
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
