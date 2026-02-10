.PHONY: install test lint format dashboard

install:
	pip install -e ".[dashboard]"

test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .

dashboard:
	cd dashboard && python app.py
