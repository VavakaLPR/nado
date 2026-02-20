import asyncio
import os
from dataclasses import dataclass

import aiohttp
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


def _missing(var: str) -> bool:
    return not (os.getenv(var) or "").strip()


async def check_hibachi() -> CheckResult:
    required = ["HIBACHI_API_KEY", "HIBACHI_ACCOUNT_ID"]
    miss = [x for x in required if _missing(x)]
    if miss:
        return CheckResult("hibachi", False, f"Missing env: {', '.join(miss)}")

    api_key = os.getenv("HIBACHI_API_KEY", "").strip()
    account_id = os.getenv("HIBACHI_ACCOUNT_ID", "").strip()
    url = "https://api.hibachi.xyz/trade/account/info"
    params = {"accountId": account_id}
    headers = {
        "Authorization": api_key,
        "Accept": "application/json",
        "User-Agent": "hedge-bot-preflight/1.0",
    }

    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    balance = data.get("balance", "n/a") if isinstance(data, dict) else "n/a"
                    return CheckResult("hibachi", True, f"Auth OK. balance={balance}")
                body = await resp.text()
                if resp.status == 401:
                    return CheckResult(
                        "hibachi",
                        False,
                        "HTTP 401 Unauthorized. Usually means wrong/revoked API key or wrong account id for this key.",
                    )
                return CheckResult("hibachi", False, f"HTTP {resp.status}: {body[:220]}")
        except Exception as e:  # pragma: no cover
            return CheckResult("hibachi", False, f"Request failed: {e}")


async def check_nado() -> CheckResult:
    required = [
        "NADO_SIGNER_PRIVATE_KEY",
        "NADO_MAIN_WALLET_ADDRESS",
        "NADO_SUBACCOUNT",
        "NADO_CHAIN_ID",
    ]
    miss = [x for x in required if _missing(x)]
    if miss:
        return CheckResult("nado", False, f"Missing env: {', '.join(miss)}")

    gateway = os.getenv("NADO_GATEWAY_REST_ENDPOINT", "https://gateway.prod.nado.xyz/v1")
    # Nado read query: contracts (no signature required)
    url = f"{gateway.rstrip('/')}/query"
    payload = {"type": "contracts"}
    timeout = aiohttp.ClientTimeout(total=15)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, br, deflate",
    }

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                text = await resp.text()
                if resp.status != 200:
                    return CheckResult("nado", False, f"HTTP {resp.status}: {text[:220]}")
                try:
                    data = await resp.json()
                except Exception:
                    return CheckResult("nado", False, f"Non-JSON response: {text[:220]}")
                status = data.get("status")
                if status == "success":
                    chain_id = ((data.get("data") or {}).get("chain_id", "n/a"))
                    endpoint = ((data.get("data") or {}).get("endpoint_addr", "n/a"))
                    return CheckResult("nado", True, f"Gateway OK. chain_id={chain_id}, endpoint={endpoint}")
                return CheckResult("nado", False, f"Gateway error: {text[:220]}")
        except Exception as e:  # pragma: no cover
            return CheckResult("nado", False, f"Request failed: {e}")


async def main() -> None:
    results = await asyncio.gather(check_hibachi(), check_nado())
    has_fail = False
    for r in results:
        state = "OK" if r.ok else "FAIL"
        print(f"[{state}] {r.name}: {r.details}")
        if not r.ok:
            has_fail = True
    if has_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
