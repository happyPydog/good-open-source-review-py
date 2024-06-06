.PHONY: test install

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

format:
	poetry run isort .
	poetry run black .