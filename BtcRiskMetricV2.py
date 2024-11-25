from datetime import date
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import nasdaqdatalink
import yfinance as yf

import streamlit as st

st.set_page_config(
    page_title="Bitcoin Risk Metric",
    page_icon="📈",
    layout="wide"
    )

@st.cache_data(ttl=4*3600)
def load_data():
    # Download historical data from Nasdaq-Data-Link
    df = nasdaqdatalink.get_table("QDL/BCHAIN", code="MKPRU", qopts={"columns":["date", "value"]}, api_key='BWjb71pVxZUtetJWYxFe', paginate=True)

    # Convert dates to datetime object for easy use
    df['date'] = pd.to_datetime(df['date'])

    # restructure Nasdaq-Data-Link dataframe
    df = df[['date', 'value']]

    # Sort data by date, just in case
    df.sort_values(by='date', inplace=True)

    # Only include data points with existing price
    df = df[df['value'] > 0]

    # get data thats not in the Nasdaq-Data-Link database
    new_data = yf.download(tickers='BTC-USD', start='2024-01-01', interval='1d')

    # # remove 'Ticker' from the columns labels
    # new_data.columns = [col[0] for col in new_data.columns.values]

    # restructure yf dataframe to match the Nasdaq-Data-Link one
    new_data.reset_index(inplace=True)
    new_data.rename(columns={'Date': 'date', 'Open': 'value'}, inplace=True)
    new_data = new_data[['date', 'value']]

    # append yf dataframe to the Nasdaq-Data-Link dataframe
    df = pd.concat([df, new_data], ignore_index=True)
    df = df[['date', 'value']]

    # remove duplicates and sort by date to prevent any issues
    df.drop_duplicates(subset='date', keep='first', inplace=True)
    df.dropna(inplace=True)
    df.sort_values(by='date', inplace=True)

    # Get the last price against USD
    btcdata = yf.download(tickers='BTC-USD', period='1d', interval='1m')
    today_data = pd.DataFrame({"date": [btcdata.iloc[-1].name], "value": [btcdata.iloc[-1]['Close']['BTC-USD']]})

    # Add the latest price data to the dataframe
    df = pd.concat([df, today_data], ignore_index=True)

    # Define the custom variables
    diminishing_factor = 0.395
    moving_average_days = 365

    # Calculate the `Risk Metric`
        # calculate the x day moving average
    df['MA'] = df['value'].rolling(moving_average_days, min_periods=1).mean().dropna()
        # calculate log-return adjusted to diminishing returns over time
        # this log-return is the relative price change from the moving average
    df['Preavg'] = (np.log(df.value) - np.log(df['MA'])) * df.index**diminishing_factor

    # Normalization to 0-1 range
    df['avg'] = (df['Preavg'] - df['Preavg'].cummin()) / (df['Preavg'].cummax() - df['Preavg'].cummin()).round(2)

    # Predicting the price according to risk level
    price_per_risk = {
        round(risk, 1):round(np.exp(
            (risk * (df['Preavg'].cummax().iloc[-1] - (cummin := df['Preavg'].cummin().iloc[-1])) + cummin) / df.index[-1]**diminishing_factor + np.log(df['MA'].iloc[-1])
        ))
        for risk in np.arange(0.0, 1.0, 0.1)
    }

    # # Exclude the first 1000 days from the dataframe, because it's pure chaos
    # df = df[df.index > 1000]

    return df, price_per_risk

df, price_per_risk = load_data()

header = "<h2 style='text-align: center; font-family: monospace;'>Bitcoin Risk Level</h2>"
subheader = f"<h6 style='text-align: center;'>Updated: {df['date'].iloc[-1].strftime('%d-%m-%Y %H:%M')} UTC | Price: ${round(df['value'].iloc[-1], 2)}</h6>"

risk_string = f"<h2 style='text-align: center; font-family: monospace;'>Risk level: {round(df['avg'].iloc[-1], 2)}</h2>"

st.markdown(header, unsafe_allow_html=True)
st.markdown("***", unsafe_allow_html=True)
st.markdown(subheader, unsafe_allow_html=True)
st.markdown(risk_string, unsafe_allow_html=True)

# Plot BTC-USD colored according to Risk values on a logarithmic chart
fig = px.scatter(df, x='date', y='value', color='avg', color_continuous_scale='jet', width=800)
fig.update_xaxes(title='Date', showgrid=False)
fig.update_yaxes(title='Price ($USD)', type='log', showgrid=True)
fig.update_layout(template='plotly_dark')
st.plotly_chart(fig, theme="streamlit", use_container_width=True)


st.markdown(subheader, unsafe_allow_html=True)
st.markdown(risk_string, unsafe_allow_html=True)

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
fig.update_layout(template='plotly_dark')
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Plot Predicting BTC price according to specific risk
fig = go.Figure(data=[go.Table(
    header=dict(values=['Risk', 'Price'],
                line_color='darkslategray',
                fill_color='lightskyblue',
                font=dict(color='black', family="monospace", size=20),
                align='center'
            ),
    cells=dict(values=[list(price_per_risk.keys()), list(price_per_risk.values())],
                line_color='darkslategray',
                fill_color='lightcyan',
                font=dict(color='black', family="monospace", size=12, weight="bold"),
                align=['center', 'right'],
                height=20
            )
    )
])
fig.update_layout(width=500, height=500, title={"text": "Price according to specific risk", "x": 0.3})

left, middle, right = st.columns((3, 4, 3))
with middle:
    st.plotly_chart(fig, theme="streamlit")

if st.checkbox('Show Data'):
    st.dataframe(df, use_container_width=True)
