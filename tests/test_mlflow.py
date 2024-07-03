import pytest

from mlfunctools.mlflow import mlflow_run


@pytest.fixture
def mlflow_mocks(mocker):
    """Fixture to mock MLflow's start_run and end_run functions."""
    start_run = mocker.patch(
        "mlflow.start_run", return_value=mocker.MagicMock()
    )
    end_run = mocker.patch("mlflow.end_run")
    return start_run, end_run


def test_mlflow_run_success(mlflow_mocks):
    start_run, end_run = mlflow_mocks

    @mlflow_run(run_name="test_run")
    def function():
        pass

    function()

    start_run.assert_called_once_with(run_name="test_run")
    end_run.assert_called_once()


def test_mlflow_run_starts_and_ends_without_run_name(mlflow_mocks):
    start_run, end_run = mlflow_mocks

    @mlflow_run
    def function():
        pass

    function()

    start_run.assert_called_once()
    end_run.assert_called_once()


def test_mlflow_run_raises_error_when_called_with_fail(mlflow_mocks):
    start_run, end_run = mlflow_mocks

    @mlflow_run(run_name="test_exception")
    def failing_function():
        raise ValueError("Intentional failure")

    with pytest.raises(ValueError, match="Intentional failure"):
        failing_function()

    start_run.assert_called_once()
    end_run.assert_called_once()


def test_mlflow_run_context_manager(mlflow_mocks):
    start_run, end_run = mlflow_mocks

    with mlflow_run(run_name="context_test"):
        pass

    start_run.assert_called_once_with(run_name="context_test")
    end_run.assert_called_once()
