#!/usr/bin/env python
"""
Build the documentation for the member comparison of W and Graph


"""

import geopandas as gpd

from libpysal import examples, graph, weights

examples.explain("sids2")


# Read the file in
gdf = gpd.read_file(examples.get_path("sids2.shp"))
gdf = gdf.set_crs("epsg:4326")

# Make weights and graph
w_queen = weights.Queen.from_dataframe(gdf, use_index=False)
g_queen = graph.Graph.build_contiguity(gdf, rook=False)


g_members = set(dir(g_queen))
w_members = set(dir(w_queen))


# filter out private members
g_members = {attr for attr in g_members if not attr.startswith("_")}
w_members = {attr for attr in w_members if not attr.startswith("_")}

compat = []
changed = []

for member in g_members & w_members:
    g_member = getattr(g_queen, member)
    w_member = getattr(w_queen, member)
    print(member, type(g_member), type(w_member), w_member.__class__.__name__)
    if type(g_member) is type(w_member):
        compat.append(member)
    else:
        changed.append(member)

changed.sort()
compat.sort()


changed_content = []
header = "Member, W Type, Graph Type"
changed_content.append(header)
for member in changed:
    line = [member]
    line.append(getattr(w_queen, member).__class__.__name__)
    line.append(getattr(g_queen, member).__class__.__name__)
    changed_content.append(",".join(line))

changed_content = [line.split(",") for line in changed_content]


def create_rst_table(data):
    if not data or not all(isinstance(row, list) for row in data):
        raise ValueError("Input should be a list of lists")

    num_columns = max(len(row) for row in data)

    def format_cell(item):
        return str(item).replace("|", r"\|").replace("\n", " ")

    normalized = [
        [
            format_cell(row[index]) if index < len(row) else ""
            for index in range(num_columns)
        ]
        for row in data
    ]

    col_widths = [
        max(len(row[index]) for row in normalized) for index in range(num_columns)
    ]

    def create_row(row):
        return (
            "| "
            + " | ".join(
                cell.ljust(width) for cell, width in zip(row, col_widths, strict=True)
            )
            + " |"
        )

    separator = "| " + " | ".join("-" * max(width, 3) for width in col_widths) + " |"
    body = [create_row(row) for row in normalized[1:]]

    return "\n".join([create_row(normalized[0]), separator, *body])


changed_table = create_rst_table(changed_content)


content = """
# W to Graph Member Comparisons

## Overview

This guide compares the members (attributes and methods) from the
`W` class and the `Graph` class.

It is intended for developers. Users interested in migrating to the
new Graph class from W should see the
[migration guide](w_g_migration.ipynb).


## Members common to W and Graph
"""


common_content = []
header = "Member,  Type"
common_content.append(header)
for member in compat:
    url = f"#libpysal.graph.Graph.{member}"
    line = [f"[`{member}`]({url})"]
    ga = getattr(g_queen, member)
    class_type = type(ga)
    gat = f"{class_type.__module__}.{class_type.__name__}"
    gs = f" {gat}"

    line.append(gs)

    common_content.append(",".join(line))

common_content = [line.split(",") for line in common_content]


content = f"{content}\n\n{create_rst_table(common_content)}"
head = """
## Members common to W and Graph with different types

"""

changed_content = []
header = "Member, Queen Type, Graph Type"
changed_content.append(header)
for member in changed:
    ms = f"{member}"
    line = [ms]
    ga = getattr(g_queen, member)
    class_type = type(ga)
    gat = f"{class_type.__module__}.{class_type.__name__}"
    gat = f"[`{gat}`](#libpysal.graph.Graph.{member})"

    gs = f"{gat}"

    wa = getattr(w_queen, member)
    class_type = type(wa)
    wat = f"{class_type.__module__}.{class_type.__name__}"
    wat = f"[`{wat}`](#libpysal.weights.W.{member})"

    ws = f"{wat}"
    line.append(ws)
    line.append(gs)

    changed_content.append(",".join(line))

changed_content = [line.split(",") for line in changed_content]
content = f"{content}\n\n{head}\n\n{create_rst_table(changed_content)}"


head = """
## Members unique to W


"""

content = f"{content}\n\n{head}"


w_only = list(w_members - g_members)
w_only.sort()

w_content = []
header = "Member,  Type"
w_content.append(header)
for member in w_only:
    line = [f"[`{member}`](#libpysal.weights.Queen.{member})"]
    wa = getattr(w_queen, member)
    class_type = type(wa)
    wat = f"{class_type.__module__}.{class_type.__name__}"

    ws = f"{wat}"

    line.append(ws)

    w_content.append(",".join(line))

w_content = [line.split(",") for line in w_content]
content = f"{content}\n\n{create_rst_table(w_content)}"


head = """
## Members unique to Graph

"""

content = f"{content}\n\n{head}"


g_only = list(g_members - w_members)
g_only.sort()

g_content = []
header = "Member,  Type"
g_content.append(header)
for member in g_only:
    url = f"#libpysal.graph.Graph.{member}"
    line = [f"[`{member}`]({url})"]
    ga = getattr(g_queen, member)
    class_type = type(ga)
    gat = f"{class_type.__module__}.{class_type.__name__}"

    gs = f"{gat}"

    line.append(gs)

    g_content.append(",".join(line))

g_content = [line.split(",") for line in g_content]
content = f"{content}\n\n{create_rst_table(g_content)}"


with open("../docs/source/user-guide/graph/migration.md", "w") as guide:
    guide.write(content)
