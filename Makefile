.PHONY: install lint test smoke dev-run dashboard docker-build docker-run

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

test:
	pytest tests/ -v

smoke:
	llm-eval prepare-data --split smoke
	llm-eval run --split smoke
	llm-eval report

dev-run:
	llm-eval prepare-data --split dev
	llm-eval run --split dev

dashboard:
	llm-eval dashboard

docker-build:
	docker build -t llm-eval-harness .

docker-run:
	docker-compose up
