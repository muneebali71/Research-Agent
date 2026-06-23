"""API client — adds get_chunks and get_documents."""
from __future__ import annotations
import requests

DEFAULT_BASE_URL = "http://localhost:8020"
DEFAULT_TIMEOUT  = 10
UPLOAD_TIMEOUT   = 120
CHAT_TIMEOUT     = 180


class ApiError(Exception):
    pass


def _request(method, url, **kwargs):
    try:
        return requests.request(method, url, **kwargs)
    except requests.exceptions.Timeout as exc:
        raise ApiError("Backend took too long to respond.") from exc
    except requests.exceptions.RequestException as exc:
        raise ApiError(f"Could not reach backend at {url}: {exc}") from exc


def _handle(resp):
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = None
        try: detail = resp.json().get("detail")
        except: pass
        raise ApiError(detail or str(exc)) from exc
    return resp.json()


def create_session(base_url):
    return _handle(_request("POST", f"{base_url}/sessions", timeout=DEFAULT_TIMEOUT))

def list_sessions(base_url):
    return _handle(_request("GET", f"{base_url}/sessions", timeout=DEFAULT_TIMEOUT))

def get_messages(base_url, session_id):
    return _handle(_request("GET", f"{base_url}/sessions/{session_id}/messages", timeout=DEFAULT_TIMEOUT))

def get_documents(base_url, session_id):
    return _handle(_request("GET", f"{base_url}/sessions/{session_id}/documents", timeout=DEFAULT_TIMEOUT))

def get_chunks(base_url, session_id):
    """Fetch all Qdrant chunks for this session."""
    return _handle(_request("GET", f"{base_url}/sessions/{session_id}/chunks", timeout=30))

def upload_pdf(base_url, session_id, filename, file_bytes):
    files = {"file": (filename, file_bytes, "application/pdf")}
    return _handle(_request("POST", f"{base_url}/sessions/{session_id}/upload",
                             files=files, timeout=UPLOAD_TIMEOUT))

def send_chat(base_url, session_id, query):
    return _handle(_request("POST", f"{base_url}/sessions/{session_id}/chat",
                             json={"query": query}, timeout=CHAT_TIMEOUT))