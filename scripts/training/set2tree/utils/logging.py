"""
    Setup for logging framework. Currently supports Comet ML.
    Make sure to create an account and setup the relevant environment
    variables.
"""

import os
import comet_ml

# ---------------------------------------------------------------------
# comet API
# ---------------------------------------------------------------------

os.environ["COMET_API_KEY"]="TGYLzWb3chtAwXikSnKH8EtQk"
os.environ["COMET_WORKSPACE"]="luksysu"

comet_api_key = os.environ["COMET_API_KEY"]
comet_workspace = os.environ["COMET_WORKSPACE"]
# comet_project_name = "set2tree_GNN"
comet_project_name = "set2tree-gnn",
# ---------------------------------------------------------------------


def get_comet_api(
    api_key=comet_api_key,
    project_name=comet_project_name,
    workspace=comet_workspace,
    # api_key="TGYLzWb3chtAwXikSnKH8EtQk",
    # project_name="set2tree-gnn",
    # workspace="luksysu",
    cache=True,
):
    comet_api = comet_ml.API(api_key=api_key, cache=cache)
    return comet_api
