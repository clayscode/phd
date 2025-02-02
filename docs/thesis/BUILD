# PhD thesis: "Deep Learning for Compilers".

exports_files(["README.md"])

py_library(
    name = "opencl_lexer",
    srcs = ["opencl_lexer.py"],
)

genrule(
    name = "thesis",
    srcs = [
        ":opencl_lexer",
        "abstract.tex",
        "acknowledgements.tex",
        "alg/clgen-synthesis.tex",
        "alg/maxmunch-tokenization.tex",
        "bibliography.bib",
        "bibliography.tex",
        "chap/background.tex",
        "chap/clgen.tex",
        "chap/conclusions.tex",
        "chap/deepsmith.tex",
        "chap/deeptune.tex",
        "chap/introduction.tex",
        "chap/related-work.tex",
        "declaration.tex",
        "eushield-normal.pdf",
        "eushield.sty",
        "img/activation-relu.pdf",
        "img/activation-sigmoid.pdf",
        "img/activation-tanh.pdf",
        "img/artificial-neural-network.pdf",
        "img/cf-deeptune.pdf",
        "img/cf-magni.pdf",
        "img/cgo-acc.pdf",
        "img/cgo-speedup-amd.pdf",
        "img/cgo-speedup-nvidia.pdf",
        "img/clang-crashes.pdf",
        "img/classification-a.pdf",
        "img/classification-b.pdf",
        "img/classification-c.pdf",
        "img/clgen-pipeline.pdf",
        "img/closeness.pdf",
        "img/deepsmith-vs-clsmith.pdf",
        "img/deepsmith.pdf",
        "img/deeptune.pdf",
        "img/difftest-generator.pdf",
        "img/ex1-A.pdf",
        "img/ex1-B.pdf",
        "img/ex2.pdf",
        "img/lstm-block.pdf",
        "img/motivation-a.pdf",
        "img/motivation-b.pdf",
        "img/motivation-c.pdf",
        "img/nn.pdf",
        "img/opencl-memory.pdf",
        "img/oracle-generator.pdf",
        "img/overview-a.pdf",
        "img/pact-speedup-a.pdf",
        "img/pact-speedup-b.pdf",
        "img/pact-speedup-c.pdf",
        "img/pact-speedup-d.pdf",
        "img/pact-speedup.pdf",
        "img/rnn-recurrence.pdf",
        "img/rnn-unrolled.pdf",
        "img/testcase-flow-chart.pdf",
        "img/training_model_a.pdf",
        "img/training_model_b.pdf",
        "img/viz.pdf",
        "infthesis.cls",
        "lst/amd-fast-walsh-transform-equivalent.cl",
        "lst/amd-fast-walsh-transform.cl",
        "lst/clgen-rewrite-after.cl",
        "lst/clgen-rewrite-before.cl",
        "lst/clgen-sample-a.cl",
        "lst/clgen-sample-b.cl",
        "lst/clgen-sample-c.cl",
        "lst/common-compiler-assertions.tex",
        "lst/compiler-hangs.tex",
        "lst/crash-compilers.tex",
        "lst/crash-intel-compiler-passes-2.tex",
        "lst/crash-intel-compiler-passes.tex",
        "lst/miscompiled-2.tex",
        "lst/miscompiled.tex",
        "lst/opencl-shim-header.h",
        "lst/parser-failures.tex",
        "lst/post-norm.tex",
        "lst/pre-norm.tex",
        "opencl_lexer.py",
        "preamble.tex",
        "tab/cgo17-benchmarks.tex",
        "tab/cgo17-platforms.tex",
        "tab/clang-assert-counts.tex",
        "tab/clgen-cgo13-features.tex",
        "tab/compiler-costs.tex",
        "tab/cpu-gpu-benchmarks-crossvalidate.tex",
        "tab/deepsmith-platforms-a.tex",
        "tab/deepsmith-platforms-b.tex",
        "tab/encoded.tex",
        "tab/grewe-features.tex",
        "tab/magni-features.tex",
        "tab/megatable-clsmith.tex",
        "tab/megatable-deepsmith.tex",
        "tab/nn-size.tex",
        "tab/preliminary-solidity-results.tex",
        "tab/vocabulary.tex",
        "thesis-shield.pdf",
        "thesis.tex",
    ],
    outs = ["thesis.pdf"],
    cmd = (
        "cd docs/thesis && " +
        "pdflatex --shell-escape thesis.tex && " +
        "biber thesis && " +
        "pdflatex --shell-escape thesis.tex && " +
        "pdflatex --shell-escape thesis.tex && " +
        "cp -v thesis.pdf ../../$@"
    ),
)
