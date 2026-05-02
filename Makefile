.PHONY: reset
reset:
	rm -r ./.venv || true
	rm uv.lock || true

.PHONY: install
install:
	uv run maturin develop --release
	uv sync --refresh --reinstall

.PHONY: dev-install
dev-install:
	uv run maturin develop --release
	uv sync --refresh --reinstall --group dev

.PHONY: lint
lint:
	cargo clippy
	uv run ruff check --output-format=full
	uv run ruff format --diff

.PHONY: test
test:
	cargo llvm-cov nextest

# # For headless CI: install xvfb and run ``xvfb-run make document``.
# .PHONY: document
# document:
# 	rm -rf docs/build || true
# 	rm -rf docs/source/api_reference/generated/ || true
# 	rm -rf docs/source/example_gallery/auto_examples || true
# 	rm docs/source/sg_execution_times.rst || true
# 	uv run sphinx-build docs/source docs/build -b html

.PHONY: benchmark
benchmark:
	mkdir -p ./tests/outputs/benchmark
	uv run pytest -v -m benchmark \
		--benchmark-min-rounds=3 \
		--benchmark-save-data \
		--benchmark-time-unit=ms \
		--benchmark-json=./tests/outputs/benchmark/latest.json \
		--benchmark-storage=./tests/outputs/benchmark \
		--benchmark-autosave

