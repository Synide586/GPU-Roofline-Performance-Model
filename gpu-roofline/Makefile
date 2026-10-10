ARCH  ?= sm_75
NVCC  ?= nvcc
FLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo

all: bin/ceilings

bin/ceilings: src/ceilings.cu src/common.cuh
	@mkdir -p bin
	$(NVCC) $(FLAGS) $< -o $@

ceilings: bin/ceilings
	@mkdir -p results
	./bin/ceilings --out results
	python3 scripts/plot_sweep.py results/bandwidth_sweep.csv

model:
	python3 scripts/model.py --ceilings results/ceilings.csv --data data/project1 --out results

counters:
	python3 scripts/counters.py --kernels-repo ../cuda-transformer-kernels --out results

clean:
	rm -rf bin

.PHONY: all ceilings model counters clean
