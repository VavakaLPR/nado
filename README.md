# Delta-Neutral Hedge Bot (Extended + Nado)

This project runs a delta-neutral hedge strategy:
- opens opposite positions on two exchanges,
- chooses long/short direction by spread (optional),
- sets TP/SL,
- closes by TP/SL or timeout.

## 1) Requirements
- macOS
- Python 3.11+
- Terminal

## 2) Install
```bash
cd "/Users/damnpluggeddd/Documents/New project"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Nado bridge uses a separate env:
```bash
python3 -m venv .venv_nado
.venv_nado/bin/python -m pip install --upgrade pip
.venv_nado/bin/pip install "nado-protocol==0.3.3"
```

## 3) Configure
1. Copy `.env.example` to `.env`
2. Fill your own API keys and wallet data
3. Verify important fields:
- `EXECUTION_MODE=live`
- `DRY_RUN=false` (or `true` for test)
- `SIDE_MIX_MODE=spread`
- `NATIVE_TPSL_ENABLED=true`

## 4) Run
Recommended overnight run (prevents system sleep):
```bash
cd "/Users/damnpluggeddd/Documents/New project"
caffeinate -i -s -m .venv/bin/python hedge_bot.py
```

Stop:
```bash
Ctrl + C
```

## 5) Logs you should see
- `OPEN ...` (new cycle opened)
- `NADO_TRIGGERS_CLEARED ...`
- `NADO_TP_SET ...`
- `NADO_SL_SET ...`
- `WATCH ...` (periodic status)
- `CYCLE N ...` (cycle completed)
- `STOP ...`

## 6) Safety notes
- Never share `.env` with anyone.
- Use your own API keys per person.
- Keep withdraw permissions disabled on exchange API keys.
- Start with small size and `MAX_CYCLES=1` after any code/config change.

## 7) Share with friends
Use the packaging script:
```bash
./build_share_zip.sh
```

It creates:
- `hedge-bot-share.zip`

The archive excludes:
- `.env`
- virtual environments
- caches
- IDE files
