from flask import Flask, render_template
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd

app = Flask(__name__)

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    return hist

def create_plot(data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name=ticker))
    fig.update_layout(title=f'{ticker} Stock Price', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
    return fig.to_html(full_html=False)

@app.route('/')
def index():
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    plots = {}
    for ticker in tickers:
        data = get_stock_data(ticker)
        plots[ticker] = create_plot(data, ticker)
    return render_template('dashboard.html', plots=plots)

if __name__ == '__main__':
    app.run(debug=True)
