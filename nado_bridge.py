import argparse
import json
from decimal import Decimal

from nado_protocol.client import create_nado_client
from nado_protocol.engine_client.types.execute import MarketOrderParams, PlaceMarketOrderParams
from nado_protocol.trigger_client.types.execute import CancelProductTriggerOrdersParams
from nado_protocol.utils.bytes32 import subaccount_to_hex
from nado_protocol.utils.nonce import gen_order_nonce
from nado_protocol.utils.subaccount import SubaccountParams


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        choices=["market", "trigger", "price", "cancel_triggers", "position", "balance", "product_meta"],
        default="market",
    )
    parser.add_argument("--mode", required=True)
    parser.add_argument("--signer-key", required=True)
    parser.add_argument("--wallet", required=True)
    parser.add_argument("--subaccount", required=True)
    parser.add_argument("--product-id", required=True, type=int)
    parser.add_argument("--side", choices=["buy", "sell"])
    parser.add_argument("--qty", type=float)
    parser.add_argument("--slippage", type=float, default=0.02)
    parser.add_argument("--reduce-only", default="false")
    parser.add_argument("--price-x18", type=int)
    parser.add_argument("--amount-x18", type=int)
    parser.add_argument("--trigger-price-x18", type=int)
    parser.add_argument("--trigger-type")
    args = parser.parse_args()

    mode = args.mode.lower()
    if mode in {"prod", "production", "main"}:
        mode = "mainnet"

    client = create_nado_client(mode, args.signer_key)
    if args.action == "market":
        if args.side is None or args.qty is None:
            raise ValueError("--side and --qty are required for --action market")
        amount = int(Decimal(str(args.qty)) * (Decimal(10) ** 18))
        if args.side == "sell":
            amount = -amount
        market_order = MarketOrderParams(
            sender=SubaccountParams(subaccount_owner=args.wallet, subaccount_name=args.subaccount),
            amount=amount,
            nonce=gen_order_nonce(),
        )
        params = PlaceMarketOrderParams(
            product_id=args.product_id,
            market_order=market_order,
            slippage=float(args.slippage),
            reduce_only=str(args.reduce_only).strip().lower() in {"1", "true", "yes", "on"},
        )
        res = client.market.place_market_order(params)
    elif args.action == "trigger":
        if (
            args.price_x18 is None
            or args.amount_x18 is None
            or args.trigger_price_x18 is None
            or args.trigger_type is None
        ):
            raise ValueError(
                "--price-x18, --amount-x18, --trigger-price-x18, --trigger-type are required for --action trigger"
            )
        res = client.market.place_price_trigger_order(
            product_id=args.product_id,
            price_x18=str(args.price_x18),
            amount_x18=str(args.amount_x18),
            trigger_price_x18=str(args.trigger_price_x18),
            trigger_type=args.trigger_type,
            subaccount_owner=args.wallet,
            subaccount_name=args.subaccount,
            reduce_only=True,
        )
    elif args.action == "price":
        px = client.market.get_latest_market_price(args.product_id)
        bid_x18 = int(getattr(px, "bid_x18", 0))
        ask_x18 = int(getattr(px, "ask_x18", 0))
        bid = float(Decimal(bid_x18) / (Decimal(10) ** 18))
        ask = float(Decimal(ask_x18) / (Decimal(10) ** 18))
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else (bid or ask or 0.0)
        print(json.dumps({"ok": True, "bid": bid, "ask": ask, "mid": mid}, ensure_ascii=True))
        return 0
    elif args.action == "position":
        sub_hex = subaccount_to_hex(args.wallet, args.subaccount)
        summary = client.subaccount.get_engine_subaccount_summary(sub_hex)
        amount_x18 = 0
        for bal in getattr(summary, "perp_balances", []):
            if int(getattr(bal, "product_id", -1)) != args.product_id:
                continue
            amount_x18 = int(getattr(getattr(bal, "balance", None), "amount", 0))
            break
        position_size = float(Decimal(amount_x18) / (Decimal(10) ** 18))
        print(
            json.dumps(
                {
                    "ok": True,
                    "subaccount": sub_hex,
                    "product_id": args.product_id,
                    "amount_x18": amount_x18,
                    "position_size": position_size,
                },
                ensure_ascii=True,
            )
        )
        return 0
    elif args.action == "balance":
        sub_hex = subaccount_to_hex(args.wallet, args.subaccount)
        summary = client.subaccount.get_engine_subaccount_summary(sub_hex)
        # healths order: [initial, maintenance, unweighted]
        h = getattr(summary, "healths", [])
        if len(h) >= 3:
            assets_x18 = int(getattr(h[2], "assets", 0))
            liabilities_x18 = int(getattr(h[2], "liabilities", 0))
            health_x18 = int(getattr(h[2], "health", 0))
        elif len(h) > 0:
            assets_x18 = int(getattr(h[0], "assets", 0))
            liabilities_x18 = int(getattr(h[0], "liabilities", 0))
            health_x18 = int(getattr(h[0], "health", 0))
        else:
            assets_x18 = 0
            liabilities_x18 = 0
            health_x18 = 0
        assets = float(Decimal(assets_x18) / (Decimal(10) ** 18))
        liabilities = float(Decimal(liabilities_x18) / (Decimal(10) ** 18))
        health = float(Decimal(health_x18) / (Decimal(10) ** 18))
        print(
            json.dumps(
                {
                    "ok": True,
                    "subaccount": sub_hex,
                    "assets_usd": assets,
                    "liabilities_usd": liabilities,
                    "balance_usd": health,
                },
                ensure_ascii=True,
            )
        )
        return 0
    elif args.action == "product_meta":
        all_markets = client.market.get_all_engine_markets()
        found = None
        for p in getattr(all_markets, "perp_products", []) or []:
            try:
                if int(getattr(p, "product_id", -1)) == args.product_id:
                    found = p
                    break
            except Exception:
                continue
        if found is None:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "product_id": args.product_id,
                        "error": "product_not_found",
                    },
                    ensure_ascii=True,
                )
            )
            return 0
        symbol = ""
        try:
            symbols = client.market.get_all_product_symbols()
            for _, s in (getattr(symbols, "symbols", {}) or {}).items():
                if int(getattr(s, "product_id", -1)) == args.product_id:
                    symbol = str(getattr(s, "symbol", ""))
                    break
        except Exception:
            pass
        size_increment = int(getattr(getattr(found, "book_info", None), "size_increment", 0) or 0)
        print(
            json.dumps(
                {
                    "ok": True,
                    "product_id": args.product_id,
                    "symbol": symbol,
                    "size_increment_x18": size_increment,
                },
                ensure_ascii=True,
            )
        )
        return 0
    else:
        params = CancelProductTriggerOrdersParams(
            sender=SubaccountParams(subaccount_owner=args.wallet, subaccount_name=args.subaccount),
            productIds=[args.product_id],
            digest=None,
            nonce=gen_order_nonce(),
        )
        res = client.market.cancel_trigger_product_orders(params)
    order_id = ""
    if isinstance(res, dict):
        order_id = str((res.get("data") or {}).get("digest", "")) or ""
    print(json.dumps({"ok": True, "order_id": order_id}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
