# LOX FUND Live Dashboard

Investor-facing live P&L dashboard with LOX FUND branding.

## Setup

Install Flask if not already installed:

```bash
pip install flask
```

## Running

From the project root:

```bash
cd dashboard
python app.py
```

Or from anywhere:

```bash
python dashboard/app.py
```

The dashboard will be available at:
- **http://localhost:5001**

## Features

- **Live P&L**: Auto-refreshes every 5 seconds
- **Position Details**: Shows all positions with P&L and percentages
- **NAV Summary**: Current NAV and total portfolio P&L
- **LOX FUND Branding**: Clean, professional design

## Publishing

To make it accessible to investors, you can:

1. **Local Network**: Run with `host='0.0.0.0'` (already set) and share your local IP
2. **Cloud**: Deploy to Heroku, Railway, or similar
3. **Tunnel**: Use ngrok or similar to expose localhost

Example with ngrok:
```bash
ngrok http 5001
```
