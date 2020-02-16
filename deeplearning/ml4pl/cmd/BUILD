py_binary(
    name = "dump_detailed_batch_logs",
    srcs = ["dump_detailed_batch_logs.py"],
    deps = [
        "//deeplearning/ml4pl/graphs/labelled:graph_tuple",
        "//deeplearning/ml4pl/models:checkpoints",
        "//deeplearning/ml4pl/models:epoch",
        "//deeplearning/ml4pl/models:log_analysis",
        "//deeplearning/ml4pl/models:log_database",
        "//deeplearning/ml4pl/models:logger",
        "//labm8/py:app",
        "//labm8/py:fs",
        "//labm8/py:jsonutil",
        "//labm8/py:labtypes",
        "//labm8/py:ppar",
        "//third_party/py/sqlalchemy",
    ],
)
