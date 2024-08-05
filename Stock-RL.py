import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle as pkl

# Cache the data preparation function
@st.cache
def data_prep(data, name):    
    df = pd.DataFrame(data[data['Name'] == name])
    df.dropna(inplace=True)    
    df.reset_index(drop=True, inplace=True)
    df['5day_MA'] = df['close'].rolling(5).mean()
    df['1day_MA'] = df['close'].rolling(1).mean()
    df['5day_MA'][:4] = 0
    return df

# Cache the state calculation function
@st.cache
def get_state(long_ma, short_ma, t):
    if short_ma < long_ma:
        return (0, 1) if t == 1 else (0, 0)
    else:
        return (1, 1) if t == 1 else (1, 0)

# Cache the trade determination function
@st.cache
def trade_t(num_of_stocks, port_value, current_price):
    return 1 if port_value > current_price else 0

# Cache the next action function
@st.cache
def next_act(state, qtable, epsilon, action=3):
    return np.random.randint(action) if np.random.rand() < epsilon else np.argmax(qtable[state])

# Cache the stock testing function
@st.cache
def test_stock(stocks_test, q_table, invest):
    num_stocks = 0
    epsilon = 0
    net_worth = [invest]
    np.random.seed()

    for dt in range(len(stocks_test)):
        long_ma = stocks_test.iloc[dt]['5day_MA']
        short_ma = stocks_test.iloc[dt]['1day_MA']
        close_price = stocks_test.iloc[dt]['close']
        t = trade_t(num_stocks, net_worth[-1], close_price)
        state = get_state(long_ma, short_ma, t)
        action = next_act(state, q_table, epsilon)

        if action == 0:  # Buy
            num_stocks += 1
            net_worth.append(np.round(net_worth[-1] - close_price, 1))
        elif action == 1:  # Sell
            num_stocks -= 1
            net_worth.append(np.round(net_worth[-1] + close_price, 1))
        elif action == 2:  # Hold
            net_worth.append(np.round(net_worth[-1], 1))

        try:
            next_state = get_state(stocks_test.iloc[dt + 1]['5day_MA'], stocks_test.iloc[dt + 1]['1day_MA'], t)
        except:
            break

    return net_worth

# Function to plot net worth
def plot_net_worth(net_worth):
    net_worth_df = pd.DataFrame(net_worth, columns=['value'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=net_worth_df.index, y=net_worth_df['value'], mode='lines', name='Portfolio Value', line=dict(color='cyan', width=2)))
    fig.update_layout(title='Change in Portfolio Value Day by Day', xaxis_title='Number of Days since Feb 2013', yaxis_title='Value ($)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br> Increase in your net worth as a result of a model decision.</p>', unsafe_allow_html=True)

# Function for the main application
def main():
    st.title("Optimizing Stock Trading Strategy With Reinforcement Learning")
    
    tabs = ["Home", "Data Exploration", "Strategy Simulation", "Performance Metrics"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)

    if selected_tab == "Home":
        st.write("Welcome to the Stock Trading Strategy Optimizer!")
        st.write("Select a tab to get started.")
    
    elif selected_tab == "Data Exploration":
        data_exploration()
    
    elif selected_tab == "Strategy Simulation":
        strategy_simulation()
    
    elif selected_tab == "Performance Metrics":
        performance_metrics()

def data_exploration():
    data = pd.read_csv('all_stocks_5yr.csv')
    names = list(data['Name'].unique())
    names.insert(0, "<Select Names>")
    
    stock = st.sidebar.selectbox("Choose Company Stocks", names, index=0)
    if stock != "<Select Names>":
        stock_df = data_prep(data, stock)
        show_stock_trend(stock, stock_df)

def show_stock_trend(stock, stock_df):
    if st.sidebar.button("Show Stock Trend", key=1):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['close'], mode='lines', name='Stock_Trend', line=dict(color='cyan', width=2)))
        fig.update_layout(title='Stock Trend of ' + stock, xaxis_title='Date', yaxis_title='Price ($)')
        st.plotly_chart(fig, use_container_width=True)
        trend_note = 'Stock is on a solid upward trend. Investing here might be profitable.' if stock_df.iloc[500]['close'] > stock_df.iloc[0]['close'] else 'Stock does not appear to be in a solid uptrend. Better not to invest here; instead, pick a different stock.'
        st.markdown(f'<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br> {trend_note}</p>', unsafe_allow_html=True)

def strategy_simulation():
    st.sidebar.subheader("Enter Your Available Initial Investment Fund")
    invest = st.sidebar.slider('Select a range of values', 1000, 1000000)
    if st.sidebar.button("Calculate", key=2):
        data = pd.read_csv('all_stocks_5yr.csv')
        stock = st.sidebar.selectbox("Choose Company Stocks", list(data['Name'].unique()), index=0)
        stock_df = data_prep(data, stock)
        q_table = pkl.load(open('pickl.pkl', 'rb'))
        net_worth = test_stock(stock_df, q_table, invest)
        plot_net_worth(net_worth)

def performance_metrics():
    st.header("Performance Metrics")
    
    data = pd.read_csv('all_stocks_5yr.csv')
    names = list(data['Name'].unique())
    names.insert(0, "<Select Names>")
    
    stock = st.sidebar.selectbox("Choose Company Stocks", names, index=0)
    if stock != "<Select Names>":
        stock_df = data_prep(data, stock)
        
        invest = st.sidebar.slider('Select a range of values', 1000, 1000000)
        if st.sidebar.button("Calculate Performance Metrics", key=3):
            q_table = pkl.load(open('pickl.pkl', 'rb'))
            net_worth = test_stock(stock_df, q_table, invest)
            
            # Calculate performance metrics
            daily_returns = pd.Series(net_worth).pct_change().dropna()
            cumulative_return = (net_worth[-1] / net_worth[0]) - 1
            annualized_return = np.mean(daily_returns) * 252
            annualized_volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility
            max_drawdown = calculate_max_drawdown(net_worth)
            
            # Display performance metrics
            st.write(f"**Cumulative Return**: {cumulative_return:.2%}")
            st.write(f"**Annualized Return**: {annualized_return:.2%}")
            st.write(f"**Annualized Volatility**: {annualized_volatility:.2%}")
            st.write(f"**Sharpe Ratio**: {sharpe_ratio:.2f}")
            st.write(f"**Maximum Drawdown**: {max_drawdown:.2%}")

            # Plot net worth
            plot_net_worth(net_worth)

def calculate_max_drawdown(net_worth):
    net_worth_series = pd.Series(net_worth)
    rolling_max = net_worth_series.cummax()
    drawdown = net_worth_series / rolling_max - 1
    max_drawdown = drawdown.min()
    return max_drawdown


if __name__ == '__main__':
    main()

