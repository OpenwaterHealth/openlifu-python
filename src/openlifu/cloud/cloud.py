from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

from openlifu.cloud.api.api import Api
from openlifu.cloud.api.dto import ClaimDbDto, DatabaseDto
from openlifu.cloud.components.abstract_component import AbstractComponent
from openlifu.cloud.components.photoscans import Photoscans
from openlifu.cloud.components.protocols import Protocols
from openlifu.cloud.components.runs import Runs
from openlifu.cloud.components.sessions import Sessions
from openlifu.cloud.components.solutions import Solutions
from openlifu.cloud.components.subjects import Subjects
from openlifu.cloud.components.systems import Systems
from openlifu.cloud.components.transducers import Transducers
from openlifu.cloud.components.users import Users
from openlifu.cloud.components.volumes import Volumes
from openlifu.cloud.const import API_URL_DEV, API_URL_PROD, ENV_DEV, ENV_PROD
from openlifu.cloud.filesystem_observer import FilesystemObserver
from openlifu.cloud.status import Status
from openlifu.cloud.sync_thread import SyncThread
from openlifu.cloud.utils import from_isoformat, get_mac_address, logger_cloud
from openlifu.cloud.ws import Websocket


class Cloud:

    def __init__(self, environment: str = ENV_PROD):
        if environment == ENV_DEV:
            api_url = API_URL_DEV
        else:
            api_url = API_URL_PROD
        self._filesystem_observer = FilesystemObserver(self._on_file_system_update)
        self._api = Api(api_url)
        self._websocket = Websocket(api_url, self._on_websocket_update)
        self._components: List[AbstractComponent] = []
        self._sync_thread = SyncThread(self._on_status_changed)
        self._db_path: Path | None = None
        self._db: DatabaseDto | None = None
        self._status_callback: Callable[[Status], None] | None = None
        self._sync_idle = True
        self._pending_updates: Dict[Path, datetime] = {}

    def set_access_token(self, token: str):
        self._api.authenticate(token)
        self._websocket.authenticate(token)

    def set_status_callback(self, callback: Callable[[Status], None]):
        self._status_callback = callback

    def start(self, db_path: Path):
        if not db_path.exists():
            raise ValueError("Database path does not exist.")
        if not db_path.is_dir():
            raise ValueError("Database path is not a directory.")

        self._db_path = db_path

        mac_address = get_mac_address()
        if mac_address is None:
            raise ValueError("MAC address is unavailable.")

        logger_cloud.debug(f"Using DB path: {db_path}")
        logger_cloud.debug(f"Mac address: {mac_address}")

        self._db = self._api.databases().claim_database(
            ClaimDbDto(
                db_path=str(db_path),
                mac_address=mac_address,
                description=None
            )
        )

        self._websocket.connect(self._db.id)

        if self._sync_thread is not None and self._sync_thread.is_running():
            self._sync_thread.stop()
        self._sync_thread = SyncThread(self._on_status_changed)

        self._create_components()
        self._sync_thread.start()

    def stop(self):
        if self._sync_thread is not None:
            self._sync_thread.stop()
        self._sync_thread = None
        self._websocket.disconnect()
        self.stop_background_sync()
        self._db_path = None
        self._db = None

    def sync(self):
        for component in self._components:
            self._sync_thread.post(component, None)

    def start_background_sync(self):
        if self._db_path is None:
            raise ValueError("Database path does not exist.")
        self._filesystem_observer.start(self._db_path)

    def stop_background_sync(self):
        self._filesystem_observer.stop()

    def _on_status_changed(self, status: Status):
        self._sync_idle = status.status == Status.STATUS_IDLE
        if self._status_callback is not None:
            self._status_callback(status)

        if self._sync_idle and len(self._pending_updates) > 0:
            logger_cloud.debug("Syncing pending updates...")
            for path, update_date in self._pending_updates.items():
                for component in self._components:
                    component.on_update_from_cloud(path, update_date)
            self._pending_updates.clear()

    def _on_file_system_update(self, path: Path):
        if self._sync_thread.is_path_in_ignore_list(path) or path.name == '.DS_Store':
            return
        if self._sync_idle:
            for component in self._components:
                component.on_filesystem_change(path)
        else:
            self._pending_updates[path] = datetime.now(timezone.utc)

    def _on_websocket_update(self, data: dict):
        if self._db_path is None:
            return
        update_date = from_isoformat(data["update_date"])
        updated_path = data["path"]
        if updated_path == '/':
            return

        path = self._db_path / updated_path

        if self._sync_thread.is_path_in_ignore_list(path):
            return

        if self._sync_idle:
            for component in self._components:
                component.on_update_from_cloud(path, update_date)
        else:
            self._pending_updates[path] = update_date

    def _create_components(self):
        self._components.clear()
        self._components.append(Users(self._api, self._db_path, self._db.id, self._sync_thread))
        self._components.append(Protocols(self._api, self._db_path, self._db.id, self._sync_thread))
        self._components.append(Systems(self._api, self._db_path, self._db.id, self._sync_thread))
        self._components.append(Transducers(self._api, self._db_path, self._db.id, self._sync_thread))
        self._components.append(
            Subjects(self._api, self._db_path, self._db.id, self._sync_thread)
            .add_child(
                Volumes(self._api, self._db_path, self._db.id, self._sync_thread)
            )
            .add_child(
                Sessions(self._api, self._db_path, self._db.id, self._sync_thread)
                .add_child(
                    Photoscans(self._api, self._db_path, self._db.id, self._sync_thread)
                )
                .add_child(
                    Runs(self._api, self._db_path, self._db.id, self._sync_thread)
                )
                .add_child(
                    Solutions(self._api, self._db_path, self._db.id, self._sync_thread)
                )
            )
        )


if __name__ == "__main__":
    logger_cloud.setLevel(logging.DEBUG)
    logger_cloud.addHandler(logging.StreamHandler(sys.stdout))

    cloud = Cloud(ENV_DEV)
    token = os.getenv("TOKEN")
    db_path = os.getenv("DB_PATH")

    cloud.set_access_token(token)
    cloud.set_status_callback(lambda status: logger_cloud.debug(f"Status: {status}"))

    cloud.start(Path(db_path))
    cloud.sync()
    cloud.start_background_sync()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cloud.stop()
