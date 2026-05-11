import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class FakeSentenceTransformer:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, texts, **_kwargs):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        return np.ones((len(texts), 3), dtype=float)


class PermissionSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = FakeSentenceTransformer
        sys.modules["sentence_transformers"] = fake_module

        cls.server = importlib.import_module("server")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)

        self.server.DB_PATH = root / "emma.db"
        self.server.LOGS_DIR = root / "logs" / "chat_audit"
        self.server.GLOBAL_FILES_DIR = root / "files" / "global"
        self.server.GLOBAL_CHUNKS_DIR = root / "chunks" / "global"

        self.server.init_db()
        self.client = TestClient(self.server.app)

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def login(self, username, password):
        response = self.client.post(
            "/auth/login",
            json={"username": username, "password": password},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["token"]

    def auth_headers(self, token):
        return {"Authorization": f"Bearer {token}"}

    def test_read_only_user_cannot_use_admin_or_file_management_endpoints(self):
        admin_token = self.login("admin", "admin1234")

        create_response = self.client.post(
            "/admin/users",
            headers=self.auth_headers(admin_token),
            json={
                "username": "viewer",
                "password": "viewer1234",
                "full_name": "Read Only Viewer",
                "role": "read_only",
            },
        )
        self.assertEqual(create_response.status_code, 200, create_response.text)

        viewer_token = self.login("viewer", "viewer1234")
        viewer_headers = self.auth_headers(viewer_token)

        admin_response = self.client.get("/admin/users", headers=viewer_headers)
        self.assertEqual(admin_response.status_code, 403)

        upload_response = self.client.post(
            "/upload",
            headers=viewer_headers,
            files={"file": ("policy.txt", b"Local policy text", "text/plain")},
        )
        self.assertEqual(upload_response.status_code, 403)

        delete_response = self.client.delete("/files/user", headers=viewer_headers)
        self.assertEqual(delete_response.status_code, 403)

        download_response = self.client.get(
            "/files/user/policy/download",
            headers=viewer_headers,
        )
        self.assertEqual(download_response.status_code, 403)

    def test_non_admin_cannot_manage_global_rags(self):
        admin_token = self.login("admin", "admin1234")

        create_response = self.client.post(
            "/admin/users",
            headers=self.auth_headers(admin_token),
            json={
                "username": "editor",
                "password": "editor1234",
                "full_name": "Regular Editor",
                "role": "user",
            },
        )
        self.assertEqual(create_response.status_code, 200, create_response.text)

        user_token = self.login("editor", "editor1234")
        user_headers = self.auth_headers(user_token)

        global_upload = self.client.post(
            "/upload?scope=global",
            headers=user_headers,
            files={"file": ("global_policy.txt", b"Global policy", "text/plain")},
        )
        self.assertEqual(global_upload.status_code, 403)

        global_delete = self.client.delete(
            "/files/global/global_policy",
            headers=user_headers,
        )
        self.assertEqual(global_delete.status_code, 403)


if __name__ == "__main__":
    unittest.main()
