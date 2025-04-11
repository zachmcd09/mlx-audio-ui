# tests/conftest.py
# Shared fixtures for pytest tests will go here.

import pytest
import sys
import os

# Add project root to sys.path to allow importing 'app'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import app as flask_app # Import your Flask app instance

@pytest.fixture(scope='module')
def app():
    """Instance of Flask app"""
    # Configure app for testing
    flask_app.config.update({
        "TESTING": True,
        # Add other test-specific configurations if needed
        # e.g., "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"
    })

    # TODO: Add any other setup needed for the app context

    yield flask_app

    # TODO: Add any cleanup needed after tests run

@pytest.fixture()
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture()
def runner(app):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()

# Add other shared fixtures below (e.g., database setup, mock objects)
