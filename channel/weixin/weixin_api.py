"""Weixin iLink bot API helpers used by the standalone channel."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import random
from urllib.parse import quote

import requests


logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://ilinkai.weixin.qq.com"
CDN_BASE_URL = "https://novac2c.cdn.weixin.qq.com/c2c"
BOT_TYPE = "3"


def _random_wechat_uin() -> str:
    value = random.randint(0, 0xFFFFFFFF)
    return base64.b64encode(str(value).encode("utf-8")).decode("utf-8")


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else f"{url}/"


def _build_headers(token: str = "") -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "AuthorizationType": "ilink_bot_token",
        "X-WECHAT-UIN": _random_wechat_uin(),
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


class WeixinApi:
    """Small HTTP client for login, polling, sending text, and CDN download."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        token: str = "",
        cdn_base_url: str = CDN_BASE_URL,
        request_timeout: int = 15,
        long_poll_timeout: int = 35,
    ) -> None:
        self.base_url = base_url
        self.token = token
        self.cdn_base_url = cdn_base_url
        self.request_timeout = request_timeout
        self.long_poll_timeout = long_poll_timeout

    def _post(self, endpoint: str, body: dict, *, timeout: int | None = None) -> dict:
        url = f"{_ensure_trailing_slash(self.base_url)}{endpoint}"
        response = requests.post(
            url,
            json=body,
            headers=_build_headers(self.token),
            timeout=timeout or self.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_updates(self, cursor: str = "") -> dict:
        try:
            return self._post(
                "ilink/bot/getupdates",
                {"get_updates_buf": cursor},
                timeout=self.long_poll_timeout + 5,
            )
        except requests.exceptions.Timeout:
            return {"ret": 0, "msgs": []}

    def send_text(self, *, to_user_id: str, text: str, context_token: str) -> dict:
        return self._post(
            "ilink/bot/sendmessage",
            {
                "msg": {
                    "from_user_id": "",
                    "to_user_id": to_user_id,
                    "client_id": os.urandom(8).hex(),
                    "message_type": 2,
                    "message_state": 2,
                    "item_list": [{"type": 1, "text_item": {"text": text}}],
                    "context_token": context_token,
                }
            },
        )

    def fetch_qr_code(self) -> dict:
        url = f"{_ensure_trailing_slash(self.base_url)}ilink/bot/get_bot_qrcode?bot_type={BOT_TYPE}"
        response = requests.get(url, timeout=self.request_timeout)
        response.raise_for_status()
        return response.json()

    def poll_qr_status(self, qrcode: str) -> dict:
        url = (
            f"{_ensure_trailing_slash(self.base_url)}ilink/bot/get_qrcode_status"
            f"?qrcode={quote(qrcode)}"
        )
        response = requests.get(
            url,
            headers={"iLink-App-ClientVersion": "1"},
            timeout=self.long_poll_timeout,
        )
        response.raise_for_status()
        return response.json()


def _aes_ecb_decrypt(data: bytes, key: bytes) -> bytes:
    from Crypto.Cipher import AES

    cipher = AES.new(key, AES.MODE_ECB)
    decrypted = cipher.decrypt(data)
    pad_len = decrypted[-1]
    if 1 <= pad_len <= 16:
        return decrypted[:-pad_len]
    return decrypted


def _decode_aes_key(aes_key: str) -> bytes:
    try:
        decoded_hex = bytes.fromhex(aes_key)
        if len(decoded_hex) == 16:
            return decoded_hex
    except (TypeError, ValueError):
        pass

    decoded = base64.b64decode(aes_key)
    if len(decoded) == 16:
        return decoded
    if len(decoded) == 32:
        return bytes.fromhex(decoded.decode("ascii"))
    msg = f"Unexpected AES key length after decoding: {len(decoded)}"
    raise ValueError(msg)


def download_media_from_cdn(
    *,
    cdn_base_url: str,
    encrypt_query_param: str,
    aes_key: str,
    save_path: str,
    timeout: int = 60,
) -> str:
    """Download and decrypt media from the Weixin CDN into a local file."""
    url = f"{cdn_base_url}/download?encrypted_query_param={quote(encrypt_query_param)}"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    key_bytes = _decode_aes_key(aes_key)
    decrypted = _aes_ecb_decrypt(response.content, key_bytes)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as file_obj:
        file_obj.write(decrypted)

    logger.debug("Downloaded Weixin media to %s (md5=%s)", save_path, hashlib.md5(decrypted).hexdigest())
    return save_path
