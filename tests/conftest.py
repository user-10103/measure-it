"""Shared test isolation for process-global caches.

Two caches in src/ deliberately live for the life of the process — they exist to
stop a sweep repeating expensive work across addresses. That is correct in
production and poison in a test suite: state from one test leaks into the next.

The dead-endpoint memo caused a real hang. A test that recorded a GIS failure
left that endpoint marked dead, so a LATER test skipped GIS entirely and fell
through to the NAIP branch it had not patched — which then attempted a real
network fetch and blocked until the runner timed out. The failure surfaced
nowhere near its cause, which is the usual shape of leaked global state.
"""
import pytest


@pytest.fixture(autouse=True)
def _reset_process_caches():
    from src.ingestion import imagery_select
    from src.roofs import select_candidates

    imagery_select._DEAD_ENDPOINTS.clear()
    select_candidates._SELECT_CACHE.clear()
    yield
    imagery_select._DEAD_ENDPOINTS.clear()
    select_candidates._SELECT_CACHE.clear()
