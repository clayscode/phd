# OpenSSL.
# https://www.openssl.org/

licenses(["notice"])  # Apache

cc_library(
    name = "openssl",
    linkopts = [
        "-lpthread",
        "-lcrypto",
    ],
    visibility = ["//visibility:public"],
    deps = select({
        "//:darwin": [
            "@openssl_darwin//:openssl",
        ],
        "//conditions:default": [],
    }),
)
