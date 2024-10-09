import numpy as np
from mlir.ir import Context, Module
from mlir.execution_engine import ExecutionEngine, ctypes
from mlir.runtime import get_ranked_memref_descriptor
import sys
import os

if __name__ == "__main__":
    code = sys.stdin.read()
    if not code:
        print("No input code was received !", file=sys.stderr)
        sys.exit(1)

    with Context():
        module = Module.parse(code)
    execution_engine = ExecutionEngine(
        module,
        shared_libs=os.getenv("SHARED_LIBS", "").split(","),
    )

    if len(sys.argv) < 2:
        print("Function name is required !", file=sys.stderr)
        sys.exit(1)
    function_name = sys.argv[1]

    full_function_name = os.path.join(
        "lqcd-benchmarks",
        "matrices_inner",
        function_name + ".mlir"
    )
    with open(full_function_name, "r") as f:
        original_code = f.read()

    np_file = np.load(full_function_name + ".npz")
    args_names: list[str] = sorted(
        np_file.files,
        key=lambda s: original_code.index(s)
    )
    args_map: map[str, np.ndarray] = {arr: np_file[arr] for arr in args_names}
    args = []
    for arg_name in args_names:
        args.append(ctypes.pointer(ctypes.pointer(
            get_ranked_memref_descriptor(args_map[arg_name])
        )))

    delta_arg = (ctypes.c_int64 * 1)(0)
    args.append(delta_arg)

    execution_engine.invoke("main", *args)
    print(delta_arg[0] / 1e9)
