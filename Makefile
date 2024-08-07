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
	poetry run flake8 src --ignore E501,W503,E704
	poetry run mypy src