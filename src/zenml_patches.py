"""
Runtime patches for ZenML dependencies to improve compatibility on Windows.

ZenML 0.91.1 relies on sqlalchemy-utils GUID columns that expect UUID objects,
but the stack often passes plain strings when running on SQLite. This causes
``AttributeError: 'str' object has no attribute 'hex'`` when SQLAlchemy tries
to convert the value. To keep the rest of the codebase untouched, we patch the
GUID binder to accept string UUIDs and convert them on the fly.
"""

from __future__ import annotations

import threading
import uuid

from sqlalchemy_utils import UUIDType
try:
    from sqlalchemy.sql.sqltypes import UUID as SAUUID  # SQLAlchemy 2.0
    from sqlalchemy.sql.sqltypes import Uuid as SAUuidBase
except ImportError:  # SQLAlchemy 1.x, skip patch (not needed)
    SAUUID = None  # type: ignore
    SAUuidBase = None  # type: ignore

_patch_lock = threading.Lock()
_patched = False


def ensure_uuid_patch() -> None:
    """Monkey-patch UUIDType to accept string inputs."""
    global _patched
    if _patched:
        return
    if SAUuidBase is None:
        return
    with _patch_lock:
        if _patched:
            return
        original = UUIDType.process_bind_param
        sa_uuid_process = getattr(SAUuidBase, "process_bind_param", None)
        sa_uuid_bind = getattr(SAUuidBase, "bind_processor", None)

        def _maybe_convert(value):
            if isinstance(value, str):
                try:
                    value = uuid.UUID(value)
                except ValueError:
                    pass
            return value

        def process_bind_param(self, value, dialect):
            value = _maybe_convert(value)
            return original(self, value, dialect)

        UUIDType.process_bind_param = process_bind_param  # type: ignore[assignment]
        if sa_uuid_process:
            def process_sa_uuid(self, value, dialect):
                value = _maybe_convert(value)
                return sa_uuid_process(self, value, dialect)

            if SAUUID is not None and hasattr(SAUUID, "process_bind_param"):
                SAUUID.process_bind_param = process_sa_uuid  # type: ignore[assignment]
            SAUuidBase.process_bind_param = process_sa_uuid  # type: ignore[assignment]

        if sa_uuid_bind:
            def bind_processor(self, dialect):
                proc = sa_uuid_bind(self, dialect)
                if proc is None:
                    return None

                def wrapper(value):
                    value = _maybe_convert(value)
                    return proc(value)

                return wrapper

            SAUuidBase.bind_processor = bind_processor  # type: ignore[assignment]
        _patched = True


# Apply patch on import
ensure_uuid_patch()
