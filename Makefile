.DEFAULT_GOAL := build

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

.PHONY: build
build:
	cargo build --workspace

.PHONY: clean
clean:
	rm -rf node_modules
	rm -rf target

.PHONY: clippy
clippy:
	cargo clippy --workspace --tests -- -D warnings

.PHONY: coverage
coverage: node_modules
	cargo llvm-cov clean --workspace
	cargo llvm-cov nextest --workspace --no-report
	cargo llvm-cov report --json --output-path target/llvm-cov.json
	cargo llvm-cov report --lcov --output-path target/lcov.info
	cargo llvm-cov report
	npx rust-coverage-check target/llvm-cov.json \
		--workspace-root $(CURDIR) \
		--gated lenses_cli=100

.PHONY: coverage-clean
coverage-clean:
	cargo llvm-cov clean --workspace
	rm -rf target/llvm-cov-target
	rm -f target/llvm-cov.json target/lcov.info

.PHONY: coverage-report
coverage-report:
	cargo llvm-cov --workspace --html

.PHONY: fmt
fmt:
	cargo fmt --all

.PHONY: fmt-check
fmt-check:
	cargo fmt --all -- --check

.PHONY: test
test:
	cargo nextest run --workspace
