"""Economic calendar data fetching for the LOX FUND Dashboard."""

from datetime import datetime

from dashboard.news_utils import get_event_source_url


def fetch_trading_economics_calendar(api_key, today_str):
    """Try Trading Economics API first - better timezone handling."""
    import requests
    events = []

    try:
        # Trading Economics API endpoint
        url = f"https://api.tradingeconomics.com/calendar/country/united%20states/{today_str}/{today_str}"
        headers = {"Authorization": f"Client {api_key}"}
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            data = resp.json() or []
            for item in data:
                event_name = item.get("Event", "") or item.get("event", "")
                event_date_str = item.get("Date", "") or item.get("date", "")

                # Parse time - Trading Economics provides timezone-aware times
                event_time = ""
                if event_date_str:
                    try:
                        from zoneinfo import ZoneInfo
                        # Trading Economics format: "2024-01-22T07:30:00-05:00" (already ET)
                        if "T" in event_date_str:
                            dt = datetime.fromisoformat(event_date_str)
                            # Convert to Eastern if needed
                            if dt.tzinfo:
                                dt_et = dt.astimezone(ZoneInfo("America/New_York"))
                            else:
                                dt_et = dt
                            event_time = dt_et.strftime("%I:%M %p ET").lstrip("0")
                    except:
                        pass

                actual = item.get("Actual") or item.get("actual")
                estimate = item.get("Forecast") or item.get("forecast") or item.get("TEForecast")
                previous = item.get("Previous") or item.get("previous")

                events.append({
                    "event": event_name,
                    "time": event_time,
                    "actual": actual,
                    "estimate": estimate,
                    "previous": previous,
                    "source": "tradingeconomics"
                })

        return events
    except Exception as e:
        print(f"[Palmer] Trading Economics error: {e}")
        return []


def fetch_fed_fiscal_calendar(settings):
    """Fetch TODAY's economic releases - tries Trading Economics first, falls back to FMP."""
    events = []
    seen_events = set()

    from datetime import datetime
    import requests

    # Only fetch TODAY's events
    today = datetime.now().strftime("%Y-%m-%d")
    today_display = datetime.now().strftime("%A, %B %d, %Y")

    # Try Trading Economics first (better timezone data)
    te_key = getattr(settings, 'trading_economics_api_key', None) or getattr(settings, 'TRADING_ECONOMICS_API_KEY', None)
    if te_key:
        print("[Palmer] Trying Trading Economics for calendar...")
        te_events = fetch_trading_economics_calendar(te_key, today)
        if te_events:
            print(f"[Palmer] Got {len(te_events)} events from Trading Economics")
            for item in te_events:
                event_name = item.get("event", "")
                dedup_key = f"{event_name[:30].lower().strip()}"
                if dedup_key in seen_events:
                    continue
                seen_events.add(dedup_key)

                # Calculate surprise
                actual = item.get("actual")
                estimate = item.get("estimate")
                surprise_direction = None
                if actual is not None and estimate is not None:
                    try:
                        a = float(str(actual).replace("%", "").replace(",", "").strip())
                        e = float(str(estimate).replace("%", "").replace(",", "").strip())
                        if a > e:
                            surprise_direction = "beat"
                        elif a < e:
                            surprise_direction = "miss"
                    except:
                        pass

                events.append({
                    "time": item.get("time", ""),
                    "event": event_name,
                    "actual": actual,
                    "previous": item.get("previous"),
                    "estimate": estimate,
                    "surprise_direction": surprise_direction,
                    "url": "https://tradingeconomics.com/united-states/calendar",
                    "source": "tradingeconomics"
                })

            if events:
                return events, today_display

    # Fallback to FMP
    print("[Palmer] Using FMP for calendar...")
    fmp_key = settings.fmp_api_key
    if not fmp_key:
        return events, None

    try:
        url = f"https://financialmodelingprep.com/api/v3/economic_calendar?from={today}&to={today}&apikey={fmp_key}"
        resp = requests.get(url, timeout=10)

        if resp.status_code == 200:
            data = resp.json() or []

            for item in data:
                country = item.get("country", "").upper()

                # Only US events
                if country and country != "US":
                    continue

                event_name = item.get("event", "")
                event_name_lower = event_name.lower()
                event_date_str = item.get("date", "")

                # Parse time - FMP may be UTC, convert to ET
                event_time = ""
                if len(event_date_str) > 10:
                    try:
                        from zoneinfo import ZoneInfo
                        dt = datetime.fromisoformat(event_date_str.replace(" ", "T"))
                        # Assume FMP is UTC, convert to Eastern
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
                        dt_et = dt.astimezone(ZoneInfo("America/New_York"))
                        event_time = dt_et.strftime("%I:%M %p ET").lstrip("0")
                    except:
                        pass

                # Dedupe
                dedup_key = f"{event_name[:30].lower().strip()}"
                if dedup_key in seen_events:
                    continue
                seen_events.add(dedup_key)

                # Get values
                actual = item.get("actual")
                estimate = item.get("estimate")
                previous = item.get("previous")

                # Calculate surprise
                surprise_direction = None
                if actual is not None and estimate is not None:
                    try:
                        actual_val = float(actual)
                        estimate_val = float(estimate)
                        diff = actual_val - estimate_val
                        # Jobless claims: lower = better
                        if "jobless" in event_name_lower or "unemployment" in event_name_lower:
                            surprise_direction = "beat" if diff < 0 else "miss" if diff > 0 else "inline"
                        else:
                            surprise_direction = "beat" if diff > 0 else "miss" if diff < 0 else "inline"
                    except:
                        pass

                events.append({
                    "time": event_time,
                    "event": event_name,
                    "actual": actual,
                    "previous": previous,
                    "estimate": estimate,
                    "surprise_direction": surprise_direction,
                    "url": get_event_source_url(event_name),
                })

        # Sort by time
        events.sort(key=lambda x: x.get("time", "99:99"))
        return events, today_display

    except Exception as e:
        print(f"[Palmer] Calendar error: {e}")
        return [], None
