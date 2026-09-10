"""Automated Data Ingestion and Validation Script.

Usage:
    python scripts/update_data.py [--force-demo]
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add src to sys.path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import yaml
from quant_engine.data.downloader import DataDownloader
from quant_engine.research.store import ResearchStore


def update_all_datasets(force_demo: bool = False) -> None:
    """Read protocol, download/update all market & macro data, and record health status."""
    protocol_path = _ROOT / "configs" / "research_protocol.yaml"
    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol = yaml.safe_load(f)

    start_date = protocol.get("date_range", {}).get("start_date", "2019-01-01")
    end_date = protocol.get("date_range", {}).get("end_date", "2024-12-31")

    downloader = DataDownloader(force_demo=force_demo)
    store = ResearchStore()

    print(f"=== UPDATING DATASETS FOR PROTOCOL {protocol.get('protocol_version')} ===")
    print(f"Time Range: {start_date} to {end_date} | Force Demo: {force_demo}\n")

    total_assets = 0
    healthy_assets = 0

    for m in protocol.get("markets", []):
        m_code = m["market_code"]
        m_name = m["name"]
        print(f"--- Processing {m_name} ({m_code}) ---")

        # Process Universe Equities & Indices
        for asset in m.get("universe", []):
            sym = asset["symbol"]
            name = asset.get("name", sym)
            df, health = downloader.load_price_data(sym, start_date, end_date)

            total_assets += 1
            if health.is_usable:
                healthy_assets += 1

            status_rec = {
                "symbol": sym,
                "name": name,
                "market": m_code,
                "provider": health.provider,
                "start_date": health.start_date,
                "end_date": health.end_date,
                "total_bars": health.total_bars,
                "health_score": health.data_health_score,
                "warnings": health.warnings,
                "is_demo": health.is_demo,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            store.save_data_status(status_rec)
            status_tag = "[OK]" if health.is_usable else "[WARN]"
            demo_tag = "[DEMO]" if health.is_demo else "[LIVE]"
            print(f"  {status_tag} {demo_tag} {sym:<10} | Bars: {health.total_bars:<5} | Score: {health.data_health_score:.1f}%")

        # Process Macro & FX Proxies
        for macro in m.get("macro_proxies", []):
            sym = macro["symbol"]
            name = macro.get("name", sym)
            s, is_demo = downloader.load_macro_data(sym, start_date, end_date)
            status_rec = {
                "symbol": sym,
                "name": name,
                "market": m_code,
                "provider": "Macro Provider",
                "start_date": str(s.index[0].date()) if len(s) > 0 else "",
                "end_date": str(s.index[-1].date()) if len(s) > 0 else "",
                "total_bars": len(s),
                "health_score": 100.0 if len(s) > 100 else 50.0,
                "warnings": [],
                "is_demo": is_demo,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            store.save_data_status(status_rec)
            print(f"  [MACRO] {sym:<10} | Observations: {len(s):<5} | Status: Recorded")

    print(f"\nData Update Complete: {healthy_assets}/{total_assets} assets meet institutional health thresholds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update and validate research market data.")
    parser.add_argument("--force-demo", action="store_true", help="Force synthetic/demo data fallback.")
    args = parser.parse_args()

    update_all_datasets(force_demo=args.force_demo)
