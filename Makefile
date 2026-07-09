.DEFAULT_GOAL := build-cpu

# -----------------------------------------------------------------------------
# Real targets
# -----------------------------------------------------------------------------

node_modules: package-lock.json
	npm ci
	touch node_modules

package-lock.json: package.json
	npm install --package-lock-only

# -----------------------------------------------------------------------------
# Phony targets
# -----------------------------------------------------------------------------

.PHONY: bench-cpu
bench-cpu:
	cargo bench -p perception_bench_cpu --features cpu

.PHONY: bench-cuda
bench-cuda:
	cargo bench -p perception_bench_cuda --features cuda

.PHONY: build-cpu
build-cpu:
	cargo build --workspace

.PHONY: build-cuda
build-cuda:
	cargo build --all-targets \
		-p perception -p perception_backend_cuda -p perception_backend_cuda_test -p perception_bench_cuda \
		--features perception/cuda,perception_backend_cuda/cuda,perception_backend_cuda_test/cuda,perception_bench_cuda/cuda

.PHONY: clean
clean:
	rm -rf node_modules
	rm -rf target

.PHONY: clippy-cpu
clippy-cpu:
	cargo clippy --workspace --all-targets --features perception_bench_cpu/cpu -- -D warnings

.PHONY: clippy-cuda
clippy-cuda:
	cargo clippy --all-targets \
		-p perception -p perception_backend_cuda -p perception_backend_cuda_test -p perception_bench_cuda \
		--features perception/cuda,perception_backend_cuda/cuda,perception_backend_cuda_test/cuda,perception_bench_cuda/cuda \
		-- -D warnings

.PHONY: coverage-cpu
coverage-cpu: node_modules
	cargo llvm-cov clean --workspace
	cargo llvm-cov nextest --workspace --no-report
	cargo llvm-cov report --json --output-path target/llvm-cov.json
	cargo llvm-cov report --lcov --output-path target/lcov.info
	cargo llvm-cov report
	npx rust-coverage-check target/llvm-cov.json \
		--workspace-root $(CURDIR) \
		--gated perception_metric=100 \
		--gated perception=100 \
		--gated perception_backend=100 \
		--gated perception_backend_cpu=100 \
		--gated perception_metric_bench=100 \
		--gated perception_metric_bench_scenarios=100 \
		--gated perception_metric_test=100 \
		--gated perception_test=100

.PHONY: coverage-cuda
coverage-cuda: node_modules
	cargo llvm-cov clean --workspace
	cargo llvm-cov nextest -p perception_backend_cuda -p perception_backend_cuda_test \
		--features perception_backend_cuda_test/cuda --no-report
	cargo llvm-cov report --json --output-path target/llvm-cov-cuda.json
	cargo llvm-cov report
	npx rust-coverage-check target/llvm-cov-cuda.json \
		--workspace-root $(CURDIR) \
		--gated perception_backend_cuda=100 \
		--gated perception_backend_cuda_test=100

.PHONY: coverage-clean
coverage-clean:
	cargo llvm-cov clean --workspace
	rm -rf target/llvm-cov-target
	rm -f target/llvm-cov.json target/lcov.info

.PHONY: fmt
fmt:
	cargo fmt --all

.PHONY: fmt-check
fmt-check:
	cargo fmt --all -- --check

.PHONY: test-cpu
test-cpu:
	cargo nextest run --workspace

.PHONY: test-cuda
test-cuda:
	cargo nextest run -p perception_backend_cuda -p perception_backend_cuda_test \
		--features perception_backend_cuda_test/cuda
