import time
from benchopt.benchmark import Benchmark
from benchopt import run_benchmark

bench_enet = Benchmark('./')

# this run fast
start = time.time()
run_benchmark(bench_enet, solver_names=[
              "celer"], dataset_names=["libsvm"], timeout=1, n_repetitions=1)
end = time.time()
print(f"{'*'*10} \n"
      f"Elapsed time {end - start} seconds"
      f"\n{'*'*10}"
      )


# this run slow
start = time.time()
run_benchmark(bench_enet, solver_names=[
              "celer"], dataset_names=["libsvm"], timeout=1, n_repetitions=2)
end = time.time()
print(f"{'*'*10} \n"
      f"Elapsed time {end - start} seconds"
      f"\n{'*'*10}"
      )
