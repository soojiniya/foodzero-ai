from __future__ import annotations

import argparse

from src.foodzero.audit import run_source_audit
from src.foodzero.features import build_model_dataset
from src.foodzero.weather import collect_weather


def main() -> None:
    parser = argparse.ArgumentParser(description="FoodZero AI data collection and preprocessing pipeline")
    parser.add_argument("--skip-weather", action="store_true", help="Use existing data/interim/weather/asos_daily_weather.csv")
    parser.add_argument("--force-weather", action="store_true", help="Refresh KMA API cache")
    args = parser.parse_args()

    run_source_audit()
    if not args.skip_weather:
        collect_weather(force=args.force_weather)
    build_model_dataset()


if __name__ == "__main__":
    main()

