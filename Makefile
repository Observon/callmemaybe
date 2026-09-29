.PHONY: install run debug lint lint-strict clean

FUNCTIONS_DEF = data/input/functions_definition.json
INPUT         = data/input/function_calling_tests.json
OUTPUT        = data/output/function_calling_results.json

install:
	uv sync

run:
	uv run python -m src \
		--functions_definition $(FUNCTIONS_DEF) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	uv run python -m pdb -m src \
		--functions_definition $(FUNCTIONS_DEF) \
		--input $(INPUT) \
		--output $(OUTPUT)

lint:
	uv run flake8 src/ tests/
	uv run mypy src/ tests/ \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 src/ tests/
	uv run mypy src/ tests/ --strict

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"       -delete 2>/dev/null || true
	find . -type f -name ".coverage"   -delete 2>/dev/null || true
