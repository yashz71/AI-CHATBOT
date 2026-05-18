import yfinance as yf
from langchain.tools import tool


@tool
def get_hist_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str
):
    """
    Fetch historical market data for a financial asset using Yahoo Finance.

    Args:
        ticker (str):
            The asset ticker symbol (e.g., 'NVDA', 'AAPL', 'BTC-USD').

        start_date (str):
            Start date for the historical data in YYYY-MM-DD format.

        end_date (str):
            End date for the historical data in YYYY-MM-DD format.

        interval (str):
            Data interval frequency.
            Examples:
            - '1m'
            - '5m'
            - '15m'
            - '1h'
            - '1d'
            - '1wk'
            - '1mo'

    Returns:
        pandas.DataFrame | str:
            - A DataFrame containing OHLCV market data if successful.
            - A descriptive error message if no data is found.
    """

    tick = yf.Ticker(ticker)

    hist = tick.history(
        start=start_date,
        end=end_date,
        interval=interval,
        repair=True
    )

    if hist.empty:
        return (
            f"No historical data found for ticker '{ticker}' "
            f"between {start_date} and {end_date} "
            f"with interval '{interval}'."
        )

    latest_price = hist['Close'].iloc[-1]

    print(f"--- {ticker} Historical Data ---")
    print(hist[['Open', 'High', 'Low', 'Close', 'Volume']])
    print(f"\nMost Recent Close: ${latest_price:.2f}")

    return hist
