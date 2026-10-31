"""
check_latency.py
Times repeated calls to POST /predict and reports mean / median / p95 / max
latency in milliseconds (evidence for a non-functional requirement such as
"the API responds within X ms").

Start the API first, then from the PROJECT ROOT:
    python src/check_latency.py
"""

import os
import statistics
import sys
import time

import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
N_REQUESTS = 50

PAYLOAD = {
    "amt_income_total": 150000, "amt_credit": 500000, "amt_annuity": 25000,
    "amt_goods_price": 450000, "cnt_children": 1, "age_years": 35,
    "years_employed": 5, "ext_source_1": 0.5, "ext_source_2": 0.6,
    "ext_source_3": 0.55, "name_contract_type": "Cash loans",
    "code_gender": "F", "flag_own_car": "N", "flag_own_realty": "Y",
    "name_income_type": "Working",
    "name_education_type": "Secondary / secondary special",
    "name_family_status": "Married", "name_housing_type": "House / apartment",
}


def main():
    # One warm-up call (the first request is often slower).
    try:
        requests.post(API_URL, json=PAYLOAD, timeout=30).raise_for_status()
    except requests.ConnectionError:
        sys.exit("Could not reach the API. Start it first (see file header).")

    times_ms = []
    for _ in range(N_REQUESTS):
        start = time.perf_counter()
        response = requests.post(API_URL, json=PAYLOAD, timeout=30)
        times_ms.append((time.perf_counter() - start) * 1000)
        if response.status_code != 200:
            sys.exit(f"Request failed: {response.status_code} {response.text}")

    times_ms.sort()
    p95 = times_ms[int(0.95 * len(times_ms)) - 1]
    print(f"\n{N_REQUESTS} requests to {API_URL}")
    print(f"  mean   : {statistics.mean(times_ms):7.1f} ms")
    print(f"  median : {statistics.median(times_ms):7.1f} ms")
    print(f"  p95    : {p95:7.1f} ms")
    print(f"  max    : {times_ms[-1]:7.1f} ms")


if __name__ == "__main__":
    main()