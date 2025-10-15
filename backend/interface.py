
"""Helper interface for interacting with the LLMCompass backend service.

This module provides small wrapper functions to simplify tests and callers:

- get_default_api_url: return the default API base URL
- init_server_command / run_init_server: helpers to build or run the docker command
- wait_for_health: poll /health until the server is responsive
- submit_task: POST /tasks with JSON payload
- query_status: GET /tasks/{task_id}
- get_result: convenience to return the result payload when done

The functions are intentionally lightweight and return dicts with either
`status_code` and `body` on HTTP success, or `error` on exceptions.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any, Dict, Optional, Tuple, Union

import requests

__all__ = [
	"get_default_api_url",
	"init_server_command",
	"run_init_server",
	"wait_for_health",
	"submit_task",
	"query_status",
	"get_result",
	"get_supported_ops",
]


def get_default_api_url(host: str = "127.0.0.1", port: int = 8000) -> str:
	"""Return the default API URL for the backend.

	Default host/port follow the project convention (host: 127.0.0.1, port: 8000).
	"""

	return f"http://{host}:{port}"


def init_server_command(image: str = "llmcompass-backend", host_port: int = 8000) -> Tuple[str, ...]:
	"""Return the docker run command (as a tuple) to start the backend server.

	The canonical command used in README is:
	  sudo docker run --rm -p 8000:8000 llmcompass-backend

	This helper constructs that command. It does not execute it.
	"""

	return ("sudo", "docker", "run", "--rm", "-p", f"{host_port}:{host_port}", image)


def run_init_server(image: str = "llmcompass-backend", host_port: int = 8000, *, background: bool = True) -> Union[subprocess.Popen, subprocess.CompletedProcess]:
	"""Run the docker command to start the backend server.

	If `background` is True, starts the container with subprocess.Popen and
	returns the Popen object so the caller may terminate it later.
	If False, runs the command synchronously and returns the CompletedProcess.

	NOTE: Running docker may require privileges (sudo) and a local docker
	installation. Tests that call this should be prepared to run in an
	environment that supports this.
	"""

	cmd = list(init_server_command(image=image, host_port=host_port))
	if background:
		# start the process and return Popen so the caller can manage it
		return subprocess.Popen(cmd)
	# blocking run
	return subprocess.run(cmd, check=False)


def wait_for_health(api_url: str, timeout: float = 30.0, interval: float = 0.5) -> bool:
	"""Poll the backend /health endpoint until it returns a healthy status or timeout.

	Returns True if healthy was observed within timeout, otherwise False.
	"""

	end = time.time() + timeout
	url = api_url.rstrip("/") + "/health"
	session = requests.Session()
	try:
		while time.time() < end:
			try:
				r = session.get(url, timeout=2.0)
			except Exception:
				time.sleep(interval)
				continue
			if r.status_code == 200:
				try:
					body = r.json()
				except Exception:
					body = {"text": r.text}
				if isinstance(body, dict) and body.get("status") == "ok":
					return True
			time.sleep(interval)
	finally:
		session.close()
	return False


def submit_task(api_url: str, payload: Dict[str, Any], timeout: float = 10.0, session: Optional[requests.Session] = None) -> Dict[str, Any]:
	"""POST the given payload to the backend /tasks endpoint.

	Returns a dict with keys:
	  - status_code: int (when HTTP exchange completed)
	  - body: parsed JSON body or {'text': raw_text}
	On exception returns {'error': str(exception)}.
	"""

	url = api_url.rstrip("/") + "/tasks"
	own_session = False
	if session is None:
		session = requests.Session()
		own_session = True
	try:
		r = session.post(url, json=payload, timeout=timeout)
		try:
			body = r.json()
		except Exception:
			body = {"text": r.text}
		return {"status_code": r.status_code, "body": body}
	except Exception as e:
		return {"error": str(e)}
	finally:
		if own_session:
			session.close()


def query_status(api_url: str, task_id: str, timeout: float = 10.0, session: Optional[requests.Session] = None) -> Dict[str, Any]:
	"""GET the task status from /tasks/{task_id}.

	Returns {'status_code': int, 'body': parsed_json} on success or {'error': ...} on exception.
	"""

	url = api_url.rstrip("/") + f"/tasks/{task_id}"
	own_session = False
	if session is None:
		session = requests.Session()
		own_session = True
	try:
		r = session.get(url, timeout=timeout)
		try:
			body = r.json()
		except Exception:
			body = {"text": r.text}
		return {"status_code": r.status_code, "body": body}
	except Exception as e:
		return {"error": str(e)}
	finally:
		if own_session:
			session.close()


def get_result(api_url: str, task_id: str, timeout: float = 10.0, session: Optional[requests.Session] = None) -> Dict[str, Any]:
	"""Convenience helper that queries status and, when done, returns the result.

	Returns a dict in one of the forms:
	  - {'status': 'done', 'result': {...}, 'status_code': 200}
	  - {'status': <state>, 'body': <full_body>, 'status_code': <code>} when not done
	  - {'error': <string>} on network/other failure
	"""

	resp = query_status(api_url, task_id, timeout=timeout, session=session)
	if "error" in resp:
		return resp
    
	status_code = resp.get("status_code")
	body = resp.get("body")
	return body


def get_supported_ops(api_url: str, timeout: float = 5.0, session: Optional[requests.Session] = None) -> Dict[str, Any]:
	"""Query the backend `/supported_ops` endpoint and return the supported ops.

	Returns a dict with keys on success:
	  - status_code: int
	  - body: parsed JSON (usually {'supported_ops': [...]})
	  - supported_ops: list or None
	On exception returns {'error': str(exception)}.
	"""

	url = api_url.rstrip("/") + "/supported_ops"
	own_session = False
	if session is None:
		session = requests.Session()
		own_session = True
	try:
		r = session.get(url, timeout=timeout)
		try:
			body = r.json()
		except Exception:
			body = {"text": r.text}
		supported = None
		if isinstance(body, dict):
			supported = body.get("supported_ops")
		return {"status_code": r.status_code, "body": body, "supported_ops": supported}
	except Exception as e:
		return {"error": str(e)}
	finally:
		if own_session:
			session.close()

