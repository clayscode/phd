# The C++ connector for PostgreSQL.
# https://www.postgresql.org/docs/6.5/libpqplusplus.htm

licenses(["notice"])  # BSD

cc_library(
    name = "libpqxx",
    linkopts = [
        "-lpqxx",
        "-lpq",
    ],
    visibility = ["//visibility:public"],
    deps = select({
        "//:darwin": [
            "@libpqxx_darwin//:libpqxx",
        ],
        "//conditions:default": [],
    }),
)
