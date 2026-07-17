from __future__ import annotations

from typing import Callable

import socketio
from socketio import exceptions

from openlifu.cloud.utils import logger_cloud

DATABASE_UPDATES_NS = "/database_updates"

class Websocket:

    def __init__(self, api_url: str, update_callback: Callable[[dict], None]):
        self._api_url = api_url
        self._sio: socketio.Client | None = None
        self._database_id = None
        self._auth = {}
        self._update_callback = update_callback

    def log(self, msg: str):
        """Force message to Slicer console immediately."""
        logger_cloud.debug(f"WS_DEBUG: {msg}\n")

    def authenticate(self, access_token: str):
        self._auth = {"token": f"Bearer {access_token}"}
        if self._database_id is not None:
            self.connect(self._database_id)

    def connect(self, database_id: int):
        self.log(f"Attempting connection to {self._api_url} for DB {database_id}")

        if self._sio is not None:
            self.disconnect()

        self._database_id = database_id

        if len(self._auth) == 0:
            self.log("Abort: No authentication token set.")
            return

        self._sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=1,
            reconnection_delay_max=10
        )

        @self._sio.event(namespace=DATABASE_UPDATES_NS)
        def connect():
            self.log(f"CONNECTED to namespace {DATABASE_UPDATES_NS}")
            try:
                self._sio.emit(
                    "subscribe",
                    {"database_id": database_id},
                    namespace=DATABASE_UPDATES_NS
                )
                self.log(f"Subscribed to database {database_id}")
            except exceptions.BadNamespaceError:
                self.log("Error: Namespace not ready for emit.")

        @self._sio.event(namespace=DATABASE_UPDATES_NS)
        def disconnect():
            self.log("Disconnected from server.")

        @self._sio.on("update", namespace=DATABASE_UPDATES_NS)
        def on_update(data):
            self.log(f"Update received: {data}")
            if self._update_callback is not None:
                self._update_callback(data)

        try:
            self._sio.connect(
                f"{self._api_url}/socket.io",
                auth=self._auth,
                namespaces=[DATABASE_UPDATES_NS],
                transports=["websocket"],
                wait=False,
                socketio_path="/socket.io"
            )
            self.log("Connect call initiated successfully.")
        except exceptions.SocketIOError as e:
            logger_cloud.error(f"WS_FATAL_ERROR: {e!s}\n")

    def disconnect(self):
        if self._sio is not None:
            try:
                self.log("Disconnecting...")
                self._sio.disconnect()
            except exceptions.SocketIOError as e:
                logger_cloud.error(f"WS_FATAL_ERROR: {e!s}\n")
            finally:
                self._sio = None
