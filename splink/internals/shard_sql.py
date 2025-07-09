import multiprocessing
import textwrap


def shard_comparison_vectors_sql(
    core_sql: str,
    table_name: str,
    num_shards: int = multiprocessing.cpu_count(),
) -> str:
    """
    Generate a sharded CREATE TABLE SQL string:
      - Sets PRAGMA threads
      - Defines a sharded CTE hashing rows into num_shards buckets
      - Emits num_shards SELECTs UNION ALL'd into a CREATE TABLE AS
    """
    # Normalize and indent the core SQL snippet
    snippet = core_sql.strip().rstrip(";")
    indented = textwrap.indent(snippet + "\n", "  ")

    # Build header: PRAGMA + CREATE WITH CTE
    lines = [
        f"CREATE TABLE {table_name} AS",
        "WITH sharded AS (",
        "  SELECT",
        "    cv.*,",
        f"    hash(cv.unique_id_l, cv.unique_id_r) % {num_shards} AS shard",
        "  FROM __splink__df_comparison_vectors AS cv",
        ")",
        "",
    ]

    # Build UNION ALL blocks
    for i in range(num_shards):
        prefix = "" if i == 0 else "UNION ALL\n"
        lines.append(prefix)
        lines.append(indented)
        lines.append(f"WHERE shard = {i}")
        lines.append("")

    # Replace final blank with semicolon
    lines[-1] = ";"
    return "\n".join(lines)
