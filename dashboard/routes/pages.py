"""HTML page routes for the LOX FUND Dashboard."""

from flask import Blueprint, render_template, abort
from flask_login import login_required, current_user

pages = Blueprint("pages", __name__)


@pages.route('/')
def index():
    """Main dashboard page -- public (no login required)."""
    return render_template('dashboard.html')


@pages.route('/my-account')
@login_required
def my_account():
    """Personal investor account page."""
    from dashboard.investors import get_investor_data
    investor_code = current_user.investor_code
    account_data = None

    if investor_code:
        try:
            data = get_investor_data()
            investors = data.get("investors", [])
            for inv in investors:
                if inv.get("code") == investor_code:
                    account_data = {
                        "code": inv["code"],
                        "basis": inv.get("basis", 0),
                        "value": inv.get("value", 0),
                        "pnl": inv.get("pnl", 0),
                        "return_pct": inv.get("return_pct", 0),
                        "units": inv.get("units", 0),
                        "ownership": inv.get("ownership", 0),
                        "nav_per_unit": data.get("nav_per_unit", 1.0),
                        "fund_return": data.get("fund_return", 0),
                        "fund_equity": data.get("equity", 0),
                    }
                    break
        except Exception as e:
            print(f"[MyAccount] Error fetching investor data: {e}")

    return render_template("my_account.html", account_data=account_data)


@pages.route('/lived-inflation')
def lived_inflation():
    return render_template('lived_inflation.html')


@pages.route('/inspiration')
def inspiration():
    return render_template('inspiration.html')


@pages.route('/regimes/<regime_name>')
def regime_page(regime_name):
    from dashboard.regime_utils import REGIME_NAMES, REGIME_DISPLAY_NAMES
    if regime_name not in REGIME_NAMES:
        abort(404)
    return render_template(
        'regime.html',
        regime_name=regime_name,
        regimes=REGIME_NAMES,
        regime_display_names=REGIME_DISPLAY_NAMES,
    )
