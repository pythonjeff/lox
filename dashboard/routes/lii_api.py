"""Lived Inflation Index API routes."""

import os
import threading

import pandas as pd
from flask import Blueprint, jsonify, request

from dashboard.bls_client import BLSClient
from dashboard.lii_categories import (
    CATEGORIES as LII_CATEGORIES,
    DEBT_CATEGORIES,
    DEBT_FRED_SERIES,
    SHELTER_FRED_SERIES,
    SHELTER_MODES,
    SCENARIO_PROFILES,
    get_all_series_ids as lii_series_ids,
    calculate_lii_weights,
)
from dashboard.lii_calculator import (
    compute_lii_timeseries,
    compute_current as lii_compute_current,
    compute_category_breakdown,
    compute_cumulative,
    compute_lii_with_debt,
    compute_debt_current,
    compute_debt_yoy_series,
    apply_shelter_mode,
    compute_essentials_split,
    compute_purchasing_power,
    compute_wage_gap,
)
from datetime import datetime, timezone

lii_api = Blueprint("lii_api", __name__)

# BLS data cache (in-memory, refreshed via /api/lii/* endpoints)
_bls_client = BLSClient()
_bls_data_cache: dict = {}
_bls_cache_lock = threading.Lock()
_bls_last_fetch: datetime | None = None

# FRED data cache for debt overlay
_fred_debt_cache: dict = {}
_fred_debt_lock = threading.Lock()
_fred_debt_last_fetch: datetime | None = None


def _get_bls_data(refresh: bool = False) -> dict:
    """Get BLS data, fetching if needed. Caches for 24 hours."""
    global _bls_data_cache, _bls_last_fetch
    with _bls_cache_lock:
        now = datetime.now(timezone.utc)
        if (
            not refresh
            and _bls_data_cache
            and _bls_last_fetch
            and (now - _bls_last_fetch).total_seconds() < 86400
        ):
            return _bls_data_cache
    # Fetch outside lock to avoid blocking
    try:
        series_ids = lii_series_ids()
        data = _bls_client.fetch_series(series_ids, start_year=2018, refresh=refresh)
        with _bls_cache_lock:
            _bls_data_cache = data
            _bls_last_fetch = datetime.now(timezone.utc)
        return data
    except Exception as e:
        print(f"[LII] BLS fetch error: {e}")
        return _bls_data_cache or {}


def _get_fred_debt_data(refresh: bool = False) -> dict:
    """Get FRED debt data for debt overlay. Caches for 24 hours."""
    global _fred_debt_cache, _fred_debt_last_fetch
    with _fred_debt_lock:
        now = datetime.now(timezone.utc)
        if (
            not refresh
            and _fred_debt_cache
            and _fred_debt_last_fetch
            and (now - _fred_debt_last_fetch).total_seconds() < 86400
        ):
            return _fred_debt_cache

    try:
        import requests as req
        fred_key = os.environ.get('FRED_API_KEY', '')
        if not fred_key:
            print("[LII] FRED_API_KEY not set — debt overlay disabled")
            return {}

        data: dict = {}
        all_fred_series = list(set(DEBT_FRED_SERIES + SHELTER_FRED_SERIES + ['CES0500000003']))
        for sid in all_fred_series:
            try:
                resp = req.get(
                    'https://api.stlouisfed.org/fred/series/observations',
                    params={
                        'series_id': sid,
                        'api_key': fred_key,
                        'file_type': 'json',
                        'observation_start': '2018-01-01',
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                obs = resp.json().get('observations', [])
                rows = []
                for o in obs:
                    val = o.get('value', '.')
                    if val == '.':
                        continue
                    rows.append({'date': pd.Timestamp(o['date']), 'value': float(val)})
                if rows:
                    data[sid] = pd.DataFrame(rows)
            except Exception as e:
                print(f"[LII] FRED fetch error for {sid}: {e}")

        with _fred_debt_lock:
            _fred_debt_cache = data
            _fred_debt_last_fetch = datetime.now(timezone.utc)
        return data
    except Exception as e:
        print(f"[LII] FRED debt fetch error: {e}")
        return _fred_debt_cache or {}


def _parse_debt_params():
    """Parse debt_overlay and debt_categories from request args."""
    debt_on = request.args.get('debt_overlay', '').lower() in ('true', '1', 'yes')
    cats_str = request.args.get('debt_categories', 'student,credit,auto')
    enabled = [c.strip() for c in cats_str.split(',') if c.strip()]
    return debt_on, enabled


def _parse_shelter_mode():
    """Parse shelter_mode from request args. Default 'oer'."""
    mode = request.args.get('shelter_mode', 'oer').lower()
    return mode if mode in SHELTER_MODES else 'oer'


def _get_effective_bls_data(shelter_mode: str) -> dict:
    """Get BLS data, optionally swapping OER for a shelter alternative."""
    bls_data = _get_bls_data()
    if shelter_mode != 'oer':
        fred_data = _get_fred_debt_data()
        bls_data = apply_shelter_mode(bls_data, fred_data, shelter_mode)
    return bls_data


@lii_api.route('/api/lii/current')
def api_lii_current():
    """Latest LII, CPI, spread, MoM deltas. Supports ?debt_overlay=true&shelter_mode=mortgage."""
    profile = request.args.get('profile', 'default')
    overrides = SCENARIO_PROFILES.get(profile, {}).get('overrides', {})
    debt_on, debt_cats = _parse_debt_params()
    shelter_mode = _parse_shelter_mode()
    bls_data = _get_effective_bls_data(shelter_mode)
    # CPI always uses original BLS data (not affected by shelter mode)
    cpi_bls = _get_bls_data() if shelter_mode != 'oer' else None
    result = lii_compute_current(bls_data, freq_overrides=overrides or None, cpi_bls_data=cpi_bls, shelter_mode=shelter_mode)
    result['profile'] = profile
    result['shelter_mode'] = shelter_mode

    if debt_on:
        fred_data = _get_fred_debt_data()
        df = compute_lii_with_debt(bls_data, fred_data, freq_overrides=overrides or None, enabled_debt=debt_cats, cpi_bls_data=cpi_bls, shelter_mode=shelter_mode)
        if not df.empty:
            latest = df.iloc[-1]
            result['lii_debt'] = round(float(latest.get('lii_debt', latest['lii'])), 2)
            result['spread_debt'] = round(float(latest.get('spread_debt', latest['spread'])), 2)

    return jsonify(result)


@lii_api.route('/api/lii/timeseries')
def api_lii_timeseries():
    """Monthly LII + CPI. Supports ?debt_overlay=true&shelter_mode=mortgage."""
    profile = request.args.get('profile', 'default')
    overrides = SCENARIO_PROFILES.get(profile, {}).get('overrides', {})
    debt_on, debt_cats = _parse_debt_params()
    shelter_mode = _parse_shelter_mode()
    bls_data = _get_effective_bls_data(shelter_mode)

    # CPI always uses original BLS data
    cpi_bls = _get_bls_data() if shelter_mode != 'oer' else None

    if debt_on:
        fred_data = _get_fred_debt_data()
        df = compute_lii_with_debt(bls_data, fred_data, freq_overrides=overrides or None, enabled_debt=debt_cats, cpi_bls_data=cpi_bls, shelter_mode=shelter_mode)
    else:
        df = compute_lii_timeseries(bls_data, freq_overrides=overrides or None, cpi_bls_data=cpi_bls, shelter_mode=shelter_mode)

    rows = []
    for _, row in df.iterrows():
        r = {
            'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
            'lii': round(float(row['lii']), 2),
            'cpi': round(float(row['cpi']), 2),
            'spread': round(float(row['spread']), 2),
        }
        if debt_on and 'lii_debt' in row:
            r['lii_debt'] = round(float(row['lii_debt']), 2)
            r['spread_debt'] = round(float(row['spread_debt']), 2)
        rows.append(r)
    return jsonify({'data': rows, 'profile': profile, 'debt_overlay': debt_on})


@lii_api.route('/api/lii/categories')
def api_lii_categories():
    """All categories with weights, YoY rates, contributions. Supports ?debt_overlay=true&shelter_mode=mortgage."""
    profile = request.args.get('profile', 'default')
    overrides = SCENARIO_PROFILES.get(profile, {}).get('overrides', {})
    debt_on, debt_cats = _parse_debt_params()
    shelter_mode = _parse_shelter_mode()
    bls_data = _get_effective_bls_data(shelter_mode)
    cpi_bls = _get_bls_data() if shelter_mode != 'oer' else None
    cats = compute_category_breakdown(
        bls_data, freq_overrides=overrides or None,
        cpi_bls_data=cpi_bls, shelter_mode=shelter_mode,
    )

    if debt_on:
        fred_data = _get_fred_debt_data()
        debt_yoy = compute_debt_yoy_series(fred_data)
        for dc in DEBT_CATEGORIES:
            if dc["key"] not in debt_cats:
                continue
            yoy_s = debt_yoy.get(dc["key"])
            yoy_pct = 0.0
            if yoy_s is not None and not yoy_s.empty:
                yoy_pct = float(yoy_s.iloc[-1]) * 100 if not pd.isna(yoy_s.iloc[-1]) else 0.0
            cats.append({
                "name": dc["name"],
                "series_id": dc["key"],
                "freq_label": dc["freq_label"],
                "freq_score": dc["freq_score"],
                "cpi_weight": 0.0,
                "lii_weight": round(dc["weight"] * 100, 1),
                "weight_delta": round(dc["weight"] * 100, 1),
                "yoy_pct": round(yoy_pct, 2),
                "cpi_contribution": 0.0,
                "lii_contribution": round(dc["weight"] * yoy_pct / 100, 4),
                "is_debt": True,
            })
        cats.sort(key=lambda r: r.get("lii_contribution", 0), reverse=True)

    return jsonify({'categories': cats, 'profile': profile, 'debt_overlay': debt_on})


@lii_api.route('/api/debt/current')
def api_debt_current():
    """Latest debt balances and rates for callout card."""
    fred_data = _get_fred_debt_data()
    info = compute_debt_current(fred_data)
    return jsonify(info)


@lii_api.route('/api/lii/shelter-modes')
def api_shelter_modes():
    """Available shelter model alternatives."""
    return jsonify(SHELTER_MODES)


@lii_api.route('/api/lii/cumulative')
def api_lii_cumulative():
    """Price level indexed to Jan 2020 = 100."""
    profile = request.args.get('profile', 'default')
    overrides = SCENARIO_PROFILES.get(profile, {}).get('overrides', {})
    bls_data = _get_bls_data()
    df = compute_cumulative(bls_data, freq_overrides=overrides or None)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date'],
            'lii_level': round(float(row['lii_level']), 2),
            'cpi_level': round(float(row['cpi_level']), 2),
        })
    return jsonify({'data': rows, 'profile': profile})


@lii_api.route('/api/lii/custom')
def api_lii_custom():
    """Custom LII with modified frequency scores via query params."""
    profile = request.args.get('profile')
    if profile and profile in SCENARIO_PROFILES:
        overrides = SCENARIO_PROFILES[profile].get('overrides', {})
    else:
        # Parse custom scores from query string: ?scores=CUUR0000SAF11:9,CUUR0000SETA02:0.5
        scores_str = request.args.get('scores', '')
        overrides = {}
        if scores_str:
            for pair in scores_str.split(','):
                parts = pair.strip().split(':')
                if len(parts) == 2:
                    try:
                        overrides[parts[0].strip()] = float(parts[1].strip())
                    except ValueError:
                        pass

    bls_data = _get_bls_data()
    current = lii_compute_current(bls_data, freq_overrides=overrides or None)
    cats = compute_category_breakdown(bls_data, freq_overrides=overrides or None)
    return jsonify({
        'current': current,
        'categories': cats,
        'profile': profile or 'custom',
    })


@lii_api.route('/api/lii/profiles')
def api_lii_profiles():
    """Available scenario profiles."""
    profiles = []
    for key, prof in SCENARIO_PROFILES.items():
        profiles.append({
            'key': key,
            'label': prof['label'],
            'description': prof['description'],
        })
    return jsonify({'profiles': profiles})


@lii_api.route('/api/lii/sentiment')
def api_lii_sentiment():
    """University of Michigan Consumer Sentiment (UMCSENT) aligned with LII monthly data."""
    try:
        from lox.config import load_settings
        settings = load_settings()
        fred_key = getattr(settings, 'fred_api_key', None) or os.environ.get('FRED_API_KEY', '')
        if not fred_key:
            return jsonify({'error': 'FRED_API_KEY not configured', 'data': []})

        import requests as req
        resp = req.get(
            'https://api.stlouisfed.org/fred/series/observations',
            params={
                'series_id': 'UMCSENT',
                'api_key': fred_key,
                'file_type': 'json',
                'observation_start': '2019-01-01',
            },
            timeout=15,
        )
        resp.raise_for_status()
        obs = resp.json().get('observations', [])
        data = []
        for o in obs:
            val = o.get('value', '.')
            if val == '.':
                continue
            data.append({
                'date': o['date'],
                'value': round(float(val), 1),
            })
        return jsonify({'data': data})
    except Exception as e:
        print(f"[LII] Sentiment fetch error: {e}")
        return jsonify({'error': str(e), 'data': []})


@lii_api.route('/api/lii/purchasing-power')
def api_lii_purchasing_power():
    """Cumulative purchasing power loss since Jan 2020."""
    profile = request.args.get('profile', 'default')
    overrides = SCENARIO_PROFILES.get(profile, {}).get('overrides', {})
    shelter_mode = _parse_shelter_mode()
    bls_data = _get_effective_bls_data(shelter_mode)
    result = compute_purchasing_power(bls_data, freq_overrides=overrides or None)
    return jsonify(result)


@lii_api.route('/api/lii/essentials-split')
def api_lii_essentials_split():
    """Essentials vs discretionary inflation breakdown."""
    profile = request.args.get('profile', 'default')
    overrides = SCENARIO_PROFILES.get(profile, {}).get('overrides', {})
    shelter_mode = _parse_shelter_mode()
    bls_data = _get_effective_bls_data(shelter_mode)
    result = compute_essentials_split(bls_data, freq_overrides=overrides or None)
    return jsonify(result)


@lii_api.route('/api/lii/wage-gap')
def api_lii_wage_gap():
    """Real wage gap: LII YoY minus average hourly earnings YoY."""
    profile = request.args.get('profile', 'default')
    overrides = SCENARIO_PROFILES.get(profile, {}).get('overrides', {})
    shelter_mode = _parse_shelter_mode()
    bls_data = _get_effective_bls_data(shelter_mode)
    fred_data = _get_fred_debt_data()
    wage_df = fred_data.get('CES0500000003')
    result = compute_wage_gap(bls_data, wage_data=wage_df, freq_overrides=overrides or None)
    return jsonify(result)


@lii_api.route('/api/cache/refresh')
def api_cache_refresh():
    """Force re-fetch BLS + FRED data."""
    bls_data = _get_bls_data(refresh=True)
    count = len(bls_data)
    return jsonify({'status': 'ok', 'series_fetched': count})
