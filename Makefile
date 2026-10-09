PY ?= python
TOOLS ?=
HELM ?= $(TOOLS)helm
TERRAFORM ?= $(TOOLS)terraform

.PHONY: install dev lint fmt typecheck test test-all e2e mutation bench bench-cache demo mock helm terraform docker wheels clean

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check src tests bench

fmt:
	ruff format src tests bench
	ruff check --fix .

typecheck:
	mypy

test:
	$(PY) -m pytest -q -m "not slow"

test-all:
	$(PY) -m pytest -q

e2e:
	$(PY) -m pytest -q -m slow

# Mutation testing of the core modules with testteeth ([tool.testteeth] in pyproject.toml).
mutation:
	testteeth run --workers 2

bench:
	$(PY) bench/loadtest.py --out bench/results/loadtest.json

bench-cache:
	$(PY) bench/semantic_cache_eval.py --embedders hashing,fastembed --out bench/results/semantic_cache.json

mock:
	llmgate mock-upstream --port 9000

demo:
	$(PY) scripts/smoke_test.py --config examples/config.mock.yaml

helm:
	$(HELM) lint deploy/helm/llmgate
	$(HELM) template t deploy/helm/llmgate --set serviceMonitor.enabled=true > /dev/null

terraform:
	cd deploy/terraform/examples/kubernetes && $(TERRAFORM) init -backend=false -input=false && $(TERRAFORM) validate

docker:
	docker build -t llmgate:dev .

# Build wheels of the optional sibling libraries into ./wheels for the Docker image.
wheels:
	rm -rf .wheelsrc && mkdir .wheelsrc
	git clone -q ../costroute .wheelsrc/costroute && git clone -q ../injectguard .wheelsrc/injectguard
	$(PY) -m pip wheel --no-deps -w wheels .wheelsrc/costroute .wheelsrc/injectguard
	rm -rf .wheelsrc

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache *.egg-info src/*.egg-info
