from __future__ import annotations

import ssl
import time

import requests
import urllib3
from requests.adapters import HTTPAdapter

from openlifu.cloud.utils import logger_cloud, to_json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- SSL Patch for Python 3.12 / GKE Compatibility ---
class SlicerAdapter(HTTPAdapter):
    """
    Custom Adapter to handle GKE Load Balancer abrupt connection closures
    which trigger SSLEOFError in Python 3.12.
    """
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        # Ignore the 'Unexpected EOF' that Google Cloud LB triggers
        if hasattr(ssl, "OP_IGNORE_UNEXPECTED_EOF"):
            context.options |= ssl.OP_IGNORE_UNEXPECTED_EOF

        # Security settings for compatibility
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

class Request:
    TIMEOUT = (5, 300)

    def __init__(self, api_url: str):
        self._api_url = api_url
        self.headers = {}
        self.session = requests.Session()
        adapter = SlicerAdapter(
        )
        self.session.mount("https://", adapter)

    def _log_request(self, method: str, url: str, start_time: float, status_code: int):
        """Helper to calculate and print request duration."""
        duration = time.perf_counter() - start_time
        logger_cloud.info(f"CLOUD_LOG: {method} {url} | Status: {status_code} | Duration: {duration:.3f}s")

    def get(self, url: str) -> str:
        start = time.perf_counter()
        response = self.session.get(self._api_url + url, headers=self.headers, timeout=self.TIMEOUT, verify=False)
        self._log_request("GET", url, start, response.status_code)

        logger_cloud.debug(f"GET: {url}, status_code: {response.status_code}\nresponse: {response.text}")
        response.raise_for_status()
        return response.text

    def get_bytes(self, url: str) -> bytes:
        start = time.perf_counter()
        response = self.session.get(self._api_url + url, headers=self.headers, timeout=self.TIMEOUT, verify=False)
        self._log_request("GET_BYTES", url, start, response.status_code)

        logger_cloud.debug(f"GET bytes: {url}, status_code: {response.status_code}")
        response.raise_for_status()
        return response.content

    def post(self, url: str, dto) -> str:
        start = time.perf_counter()
        response = self.session.post(self._api_url + url, data=to_json(dto), headers=self.headers, timeout=self.TIMEOUT, verify=False)
        self._log_request("POST", url, start, response.status_code)

        logger_cloud.debug(f"POST: {url}, body: {to_json(dto)}, status_code: {response.status_code}\nresponse: {response.text}")
        response.raise_for_status()
        return response.text

    def post_bytes(self, url: str, data) -> str:
        start = time.perf_counter()
        response = self.session.post(self._api_url + url, data=data, headers=self.headers, timeout=self.TIMEOUT, verify=False)
        self._log_request("POST_BYTES", url, start, response.status_code)

        logger_cloud.debug(f"POST bytes: {url}, status_code: {response.status_code}\nresponse: {response.text}")
        response.raise_for_status()
        return response.text

    def put(self, url: str, dto) -> str:
        start = time.perf_counter()
        response = self.session.put(self._api_url + url, data=to_json(dto), headers=self.headers, timeout=self.TIMEOUT, verify=False)
        self._log_request("PUT", url, start, response.status_code)

        logger_cloud.debug(f"PUT: {url}, body: {to_json(dto)}, status_code: {response.status_code}\nresponse: {response.text}")
        response.raise_for_status()
        return response.text

    def delete(self, url: str) -> str:
        start = time.perf_counter()
        response = self.session.delete(self._api_url + url, headers=self.headers, timeout=self.TIMEOUT, verify=False)
        self._log_request("DELETE", url, start, response.status_code)

        logger_cloud.debug(f"DELETE: {url}, status_code: {response.status_code}\nresponse: {response.text}")
        response.raise_for_status()
        return response.text
