import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import dashboard


class DashboardAuthTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "ALICE_DASHBOARD_USERNAME": "alice-admin",
                "ALICE_DASHBOARD_PASSWORD": "test-password-with-entropy",
            },
            clear=False,
        )
        self.env.start()
        self.client = TestClient(dashboard.app)

    def tearDown(self):
        self.env.stop()

    def test_all_dashboard_routes_require_authentication(self):
        for path in ("/", "/api/status", "/api/config"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 401, path)
            self.assertEqual(response.headers.get("www-authenticate"), 'Basic realm="Alice dashboard"')

    def test_valid_credentials_can_open_dashboard(self):
        response = self.client.get("/", auth=("alice-admin", "test-password-with-entropy"))
        self.assertEqual(response.status_code, 200)

    def test_dashboard_fails_closed_when_password_is_missing(self):
        with patch.dict(os.environ, {"ALICE_DASHBOARD_PASSWORD": ""}):
            response = self.client.get("/")
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
