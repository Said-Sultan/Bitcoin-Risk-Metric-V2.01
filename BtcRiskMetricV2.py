from datetime import date
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import nasdaqdatalink
import yfinance as yf

# Download historical data from Nasdaq-Data-Link
df = nasdaqdatalink.get_table("QDL/BCHAIN", code="MKPRU", qopts={"columns":["date", "value"]}, api_key='BWjb71pVxZUtetJWYxFe', paginate=True)

# Convert dates to datetime object for easy use
df['date'] = pd.to_datetime(df['date'])

# restructure quandl dataframe
df = df[['date', 'value']]

# Sort data by date, just in case
df.sort_values(by='date', inplace=True)

# Only include data points with existing price
df = df[df['value'] > 0]

# get data thats not in the Nasdaq-Data-Link database
new_data = yf.download(tickers='BTC-USD', start='2024-01-01', interval='1d')

# remove 'Ticker' from the columns labels
new_data.columns = [col[0] for col in new_data.columns.values]

# restructure yf dataframe to match the Nasdaq-Data-Link one
new_data.reset_index(inplace=True)
new_data.rename(columns={'Date': 'date', 'Open': 'value'}, inplace=True)
new_data = new_data[['date', 'value']]

# append yf dataframe to the Nasdaq-Data-Link dataframe
df = pd.concat([df, new_data], ignore_index=True)

# remove duplicates and sort by date to prevent any issues
df.drop_duplicates(subset='date', keep='first', inplace=True)
df.sort_values(by='date', inplace=True)

# # Get the last price against USD
btcdata = yf.download(tickers='BTC-USD', period='1d', interval='1m')

# Append the latest price data to the dataframe
df.loc[df.index[-1]+1] = [date.today(), btcdata.iloc[-1].loc[('Close', 'BTC-USD')]]
df['date'] = pd.to_datetime(df['date'], utc=True)

diminishing_factor = 0.395
moving_average_days = 365

# Calculate the `Risk Metric`
    # calculate the x day moving average
df['MA'] = df['value'].rolling(moving_average_days, min_periods=1).mean().dropna()
    # calculate log-return adjusted to diminishing returns over time
    # this log-return is the relative price change from the moving average
df['Preavg'] = (np.log(df.value) - np.log(df['MA'])) * df.index**diminishing_factor

# Normalization to 0-1 range
df['avg'] = (df['Preavg'] - df['Preavg'].cummin()) / (df['Preavg'].cummax() - df['Preavg'].cummin())

# Predicting the price according to risk level
price_per_risk = {
    round(risk, 1):round(np.exp(
        (risk * (df['Preavg'].cummax().iloc[-1] - (cummin := df['Preavg'].cummin().iloc[-1])) + cummin) / df.index[-1]**diminishing_factor + np.log(df['MA'].iloc[-1])
    ))
    for risk in np.arange(0.0, 1.0, 0.1)
}

# # Exclude the first 1000 days from the dataframe, because it's pure chaos
# df = df[df.index > 1000]

# Title for the plots
AnnotationText = f"Updated: {btcdata.index[-1]} | Price: {round(df['value'].iloc[-1])} | Risk: {round(df['avg'].iloc[-1], 2)}"

# Plot BTC-USD and Risk on a logarithmic chart
fig = make_subplots(specs=[[{'secondary_y': True}]])

# Add BTC-USD and Risk data to the figure
fig.add_trace(go.Scatter(x=df['date'], y=df['value'], name='Price', line=dict(color='gold')))
fig.add_trace(go.Scatter(x=df['date'], y=df['avg'],   name='Risk',  line=dict(color='white')), secondary_y=True)

# Add green (`accumulation` or `buy`) rectangles to the figure
opacity = 0.2
for i in range(5, 0, -1):
    opacity += 0.05
    fig.add_hrect(y0=i*0.1, y1=((i-1)*0.1), line_width=0, fillcolor='green', opacity=opacity, secondary_y=True)

# Add red (`distribution` or `sell`) rectangles to the figure
opacity = 0.2
for i in range(6, 10):
    opacity += 0.1
    fig.add_hrect(y0=i*0.1, y1=((i+1)*0.1), line_width=0, fillcolor='red', opacity=opacity, secondary_y=True)

fig.update_xaxes(title='Date')
fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
fig.update_yaxes(title='Risk', type='linear', secondary_y=True, showgrid=True, tick0=0.0, dtick=0.1, range=[0, 1])
fig.update_layout(template='plotly_dark', title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
fig.show()

# Plot BTC-USD colored according to Risk values on a logarithmic chart
fig = px.scatter(df, x='date', y='value', color='avg', color_continuous_scale='jet')
fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
fig.update_layout(template='plotly_dark', title={'text': AnnotationText, 'y': 0.9, 'x': 0.5})
fig.show()

# Plot Predicting BTC price according to specific risk
fig = go.Figure(data=[go.Table(
    header=dict(values=['Risk', 'Price'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                align='left'),
    cells=dict(values=[list(price_per_risk.keys()), list(price_per_risk.values())],
               line_color='darkslategray',
               fill_color='lightcyan',
               align='left'))
])
fig.update_layout(width=500, height=500, title={'text': 'Price according to specific risk', 'y': 0.9, 'x': 0.5})
fig.show()
