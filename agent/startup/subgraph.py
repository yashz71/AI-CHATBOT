from agent.utils.subgraphs.graph import (
    build_general_subgraph
)
general_subgraph = None


async def initialize_graphs():

    global general_subgraph

    general_subgraph = build_general_subgraph(
    )

