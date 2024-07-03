.PHONY: test install

test:
	poetry run pytest -vv tests \
	--cov=. \
	--cov-report=term \
	 --cov-report=xml:coverage.xml

e2e-test:
	poetry run pytest -vv e2e

install:
	poetry install
	
format:
	poetry run isort .
	poetry run black .

lint:
	poetry run flake8 src
	poetry run mypy src