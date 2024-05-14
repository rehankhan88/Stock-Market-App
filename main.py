# import library


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go # type: ignore
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Title
app_name = 'Rehan khan Stock Market App'
st.title(app_name)
st.subheader('This app is create to forecast the stock market price of the selected company')
# add in image
# image_path = "Users\MUHIB\Desktop\ppppp\Python\stuckmarket\images/rehan 6.png"
# st.image(image_path, caption='Your Image', use_column_width=True)
st.image("https://www.marketplace.org/wp-content/uploads/2019/09/stockmarket.jpg?fit=2880%2C1621")
st.image(r"C:\Users\MUHIB\Desktop\ppppp\Python\stuckmarket\images\rehan 6.png")

# take input from the user of appabout the start and end date
# sidebar

st.sidebar.header("Select the parameters from below")
start_date = st.sidebar.date_input("Start date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", date(2021, 1, 1))

# add ticker symbol list

ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "FB", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox("Select the Company", ticker_list)

# fetch data use from user input using yfinance library

data = yf.download(ticker, start=start_date, end=end_date)
# add date as a column to the dateframe
data.insert(0, "Date", data.index, True)
data.reset_index(drop = True, inplace= True)
st.write('Data from', start_date, 'to', end_date)
st.write(data)

# plot the data
st.header('Date visualization')
st.subheader('plot of the data')
st.write('**Note:** Select your specific data range on the sidebar, or zoom in on the plot and select your specific column')
fig = px.line(data, x='Date', y=data.columns, title='Closing price of the stock', width=1000, height=600)
st.plotly_chart(fig)

# add a select box to select column from data
column = st.selectbox('Select the column to use for forcasting', data.columns[1:])
# Subsetting the data 
data = data[['Date', column]]
st.write('Selected Data')
st.write(data)

# ADF test check stationary
st.header('Is data Stationary?')
# st.write('**Note:** If p-value is less than 0.05, than data is stationary')
st.write(adfuller(data[column])[1] < 0.05)
# st.write(adfuller(data[column])[1])

# lets decompose the data
st.header('Decomposition of the data')
# decomposition = seasonal_decompose(data[column], model='additive', period=12)
decompositions = seasonal_decompose(data[column], model='additive', period=12)
st.write(decompositions.plot())


# Replace "Data" with the actual column name containing your dates
date_column = "Date"

# Plotting the decomposition in Plotly
st.write('## Plotting the decomposition in plotly')
st.area_chart(px.line(x=data[date_column], y= decompositions.trend, title='Trend', width=1200, height=400, labels={'x':'Date', 'y': 'Price'}).update_traces(line_color='Blue'))
st.area_chart(px.line(x=data[date_column], y= decompositions.seasonal, title='Seasonality', width=1200, height=400, labels={'x':'Date', 'y': 'Price'}).update_traces(line_color='green'))
st.area_chart(px.line(x=data[date_column], y= decompositions.resid, title='Residuals', width=1200, height=400, labels={'x':'Date', 'y': 'Price'}).update_traces(line_color='red', line_dash='dot'))




#good

# import pandas as pd
# import plotly.express as px
# import statsmodels.api as sm
# import streamlit as st

# Load your data and perform decomposition
# Assuming you have already defined data and decompositions

# Plotting the decomposition in Plotly
st.write('## Plotting the decomposition in plotly')

# Replace "Data" with the actual column name containing your dates
date_column = "Date"

# Trend
fig_trend = px.line(x=data[date_column], y=decompositions.trend, title='Trend', width=1200, height=400,
                    labels={'x':'Date', 'y': 'Price'})
fig_trend.update_traces(line_color='Blue')
st.plotly_chart(fig_trend)

# Seasonality
fig_seasonality = px.line(x=data[date_column], y=decompositions.seasonal, title='Seasonality', width=1200, height=400,
                          labels={'x':'Date', 'y': 'Price'})
fig_seasonality.update_traces(line_color='green')
st.plotly_chart(fig_seasonality)

# Residuals
fig_residuals = px.line(x=data[date_column], y=decompositions.resid, title='Residuals', width=1200, height=400,
                        labels={'x':'Date', 'y': 'Price'})
fig_residuals.update_traces(line_color='red', line_dash='dot')
st.plotly_chart(fig_residuals)

# User input for model parameters
p = st.sidebar.slider("Select the value of p", 0, 5, 2)
d = st.sidebar.slider("Select the value of d", 0, 5, 1)
q = st.sidebar.slider("Select the value of q", 0, 5, 2)
seasonal_order = st.number_input('Select the value of seasonal p', value= 12)

# Fit the SARIMAX model
model = sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q, 12))
model = model.fit(disp=-1)

# Model summary
st.header('##Model Summary')
st.write(model.summary())
st.write("---")

# Forecasting
forecast_period = st.number_input("Enter Forecast period in days", value=10)
predictions = model.get_prediction(start=len(data), end=len(data)+forecast_period)
predictions = predictions.predicted_mean

# Add index to results dataframe as dates
predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index, True)
predictions.reset_index(drop=True, inplace=True)

st.write('Predictions:', predictions)
st.write('Actual data:', data)
st.write('---')

# Assuming you have already defined 'data' and 'predictions'

# Create a new Plotly figure
fig = go.Figure()

# Add actual data to the plot
fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual', line=dict(color='blue')))

# Add predicted data to the plot
fig.add_trace(go.Scatter(x=predictions['Date'], y=predictions['predicted_mean'], mode='lines', name='Predicted', line=dict(color='red')))

# Set the title and axis labels
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)

# Display the plot
st.plotly_chart(fig)

# add button to show and hide separate plots

show_plots = False

# Main Streamlit code
if st.button('Show Separate Plots'):
    show_plots = not show_plots  # Toggle the show_plots variable
    if show_plots:
        # Show separate plots
        for col in data.columns:
            if col != "Date":
                fig = px.line(x=data["Date"], y=data[col], title=f"Plot of {col}")
                st.plotly_chart(fig)
                show_plots = True
    else:
        # st.write("Plots are now hidden")
              show_plots = False
# add hide plots button
hide_plots = False
if st.button("Hide Separate Plots"):
    if not hide_plots:
        hide_plots= True
    else:
        hide_plots=False
st.write("---")



st.write("## Connect with me on my social media")


# Define your social media URLs
linkedin_url = "https://img.icons8.com/color/48/000000/linkedin.png"
github_url = "https://img.icons8.com/color/48/000000/github.png"
facebook_url = "https://img.icons8.com/color/48/000000/facebook.png"
youtube_url = "https://img.icons8.com/color/48/000000/youtube.png"
twitter_url = "https://img.icons8.com/color/48/000000/twitter.png"

# Redirect URLs
linkedin_redirect_url = "https://www.linkedin.com"
github_redirect_url = "https://www.github.com"
facebook_redirect_url = "https://www.facebook.com"
youtube_redirect_url = "https://www.youtube.com"
twitter_redirect_url = "https://www.twitter.com"

# Display social media icons with links
st.markdown(f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}" width="60" height="60"></a>', unsafe_allow_html=True)
st.markdown(f'<a href="{github_redirect_url}"><img src="{github_url}" width="60" height="60"></a>', unsafe_allow_html=True)
st.markdown(f'<a href="{facebook_redirect_url}"><img src="{facebook_url}" width="60" height="60"></a>', unsafe_allow_html=True)
st.markdown(f'<a href="{youtube_redirect_url}"><img src="{youtube_url}" width="60" height="60"></a>', unsafe_allow_html=True)
st.markdown(f'<a href="{twitter_redirect_url}"><img src="{twitter_url}" width="60" height="60"></a>', unsafe_allow_html=True)

            

