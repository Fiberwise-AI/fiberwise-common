"""Mini OIDC routes — discovery, JWKS, and token endpoint.

Mounted at /oidc when AGENT_AUTH_MODE=local. Makes Fiberwise look like
an OIDC-compliant IDP to the A2A server.
"""

import hashlib
import json
import logging
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/oidc", tags=["oidc-provider"])

# The provider instance is set during app startup via init_mini_oidc()
_provider = None
_db = None


def init_mini_oidc(provider, db=None):
    """Called at app startup to wire the provider and DB."""
    global _provider, _db
    _provider = provider
    _db = db


@router.get("/.well-known/openid-configuration")
async def discovery():
    """OIDC Discovery document."""
    if not _provider:
        raise HTTPException(503, "Mini OIDC provider not initialized")
    return JSONResponse(_provider.get_discovery())


@router.get("/jwks")
async def jwks():
    """JSON Web Key Set — public keys for JWT verification."""
    if not _provider:
        raise HTTPException(503, "Mini OIDC provider not initialized")
    return JSONResponse(_provider.get_jwks())


@router.post("/token")
async def token(request: Request):
    """Token endpoint — client_credentials grant only."""
    if not _provider:
        raise HTTPException(503, "Mini OIDC provider not initialized")

    # Parse form data (standard OAuth 2.0 token request)
    form = await request.form()
    grant_type = form.get("grant_type")
    client_id = form.get("client_id")
    client_secret = form.get("client_secret")
    audience = form.get("audience", os.getenv("A2A_AUDIENCE", "a2a-server"))

    if grant_type != "client_credentials":
        raise HTTPException(400, {"error": "unsupported_grant_type"})
    if not client_id or not client_secret:
        raise HTTPException(400, {"error": "invalid_request", "error_description": "client_id and client_secret required"})

    # Validate credentials against DB
    secret_hash = hashlib.sha256(client_secret.encode()).hexdigest()
    row = None
    if _db:
        try:
            row = await _db.fetch_one(
                """SELECT agent_id, organization_id, app_id, scopes, a2a_permissions
                   FROM agent_api_keys
                   WHERE idp_client_id = :client_id
                     AND idp_client_secret_hash = :secret_hash
                     AND is_active = true""",
                {"client_id": client_id, "secret_hash": secret_hash},
            )
        except Exception as e:
            logger.error("Token endpoint DB error: %s", e)

    if not row:
        raise HTTPException(401, {"error": "invalid_client"})

    row = dict(row)
    agent_id = row.get("agent_id", "")
    org_id = row.get("organization_id")
    app_id = row.get("app_id")

    # Parse scopes
    scopes_raw = row.get("scopes", "[]")
    if isinstance(scopes_raw, str):
        try:
            scopes = json.loads(scopes_raw)
        except json.JSONDecodeError:
            scopes = []
    else:
        scopes = scopes_raw or []

    # Parse a2a permissions
    perms_raw = row.get("a2a_permissions", "{}")
    if isinstance(perms_raw, str):
        try:
            a2a_perms = json.loads(perms_raw)
        except json.JSONDecodeError:
            a2a_perms = {}
    else:
        a2a_perms = perms_raw or {}

    # Issue JWT
    result = _provider.issue_token(
        subject=f"agent_{agent_id}",
        audience=audience,
        scopes=scopes,
        claims={
            "org_id": org_id,
            "app_id": app_id,
            "agent_id": agent_id,
            "a2a": a2a_perms,
        },
    )

    return JSONResponse(result)
