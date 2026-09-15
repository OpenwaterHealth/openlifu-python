from __future__ import annotations

import pytest

from openlifu.cloud.cloud import Cloud
from openlifu.cloud.const import API_URL_DEV, API_URL_PROD


@pytest.mark.parametrize(("kwargs", "api_url"), [
    ({}, API_URL_PROD),
    ({"environment": "prod"}, API_URL_PROD),
    ({"environment": "dev"}, API_URL_DEV),
])
def test_environment_routes_http_and_websocket(mocker, tmp_path, kwargs, api_url):
    send = mocker.patch("requests.Session.send")
    send.return_value.status_code = 200
    send.return_value.text = '{"id": 42}'
    websocket = mocker.patch("openlifu.cloud.ws.socketio.Client")
    mocker.patch("openlifu.cloud.cloud.get_mac_address", return_value="00:11:22:33:44:55")
    mocker.patch("openlifu.cloud.cloud.SyncThread.start")

    cloud = Cloud(**kwargs)
    try:
        cloud.set_access_token("test-token")
        cloud.start(tmp_path)

        request = send.call_args.args[0]
        assert request.method == "PUT"
        assert request.url == api_url + "/databases/claim"
        assert request.headers["Authorization"] == "Bearer test-token"
        connection = websocket.return_value.connect.call_args
        assert connection.args == (api_url + "/socket.io",)
        assert connection.kwargs["auth"] == {"token": "Bearer test-token"}
    finally:
        cloud.stop()


@pytest.mark.parametrize("environment", ["DEV", "deev", "dev ", "staging", "", None])
def test_invalid_environment(environment):
    with pytest.raises(ValueError, match="Unsupported cloud environment"):
        Cloud(environment=environment)
