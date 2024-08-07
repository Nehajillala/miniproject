import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pickle as pkl

##############################################################################################################################
############################ Processing Phase ################################################
@st.cache
def data_prep(data, name):    
    df = pd.DataFrame(data[data['Name'] == name])
    df.dropna(inplace=True)    
    df.reset_index(drop=True, inplace=True)
    df['5day_MA'] = df['close'].rolling(5).mean()
    df['1day_MA'] = df['close'].rolling(1).mean()
    df['5day_MA'][:4] = 0
    return df

@st.cache
def get_state(long_ma, short_ma, t):
    if short_ma < long_ma:
        return (0, 1) if t == 1 else (0, 0)
    else:
        return (1, 1) if t == 1 else (1, 0)

@st.cache
def trade_t(num_of_stocks, port_value, current_price):
    return 1 if port_value > current_price else 0

@st.cache
def next_act(state, qtable, epsilon, action=3):
    return np.random.randint(action) if np.random.rand() < epsilon else np.argmax(qtable[state])

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

######################################################################################################################################
############## Designing Phase #############################

def main():
    # Reading the Dataset
    data = pd.read_csv('all_stocks_5yr.csv')
    names = list(data['Name'].unique())
    names.insert(0, "<Select Names>")

    st.title("Optimizing Stock Trading Strategy With Reinforcement Learning")
    st.sidebar.title("Choose Stock and Investment")
    
    tabs = ["Home", "Data Exploration", "Strategy Simulation", "Performance Metrics"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)

    if selected_tab == "Home":
        st.write("Welcome to the Stock Trading Strategy Optimizer!")
        st.write("Select a tab to get started.")
    
    elif selected_tab == "Data Exploration":
        data_exploration(data, names)
    
    elif selected_tab == "Strategy Simulation":
        strategy_simulation(data, names)
    
    elif selected_tab == "Performance Metrics":
        performance_metrics()

def data_exploration(data, names):
    st.header("Data Exploration")
    stock = st.sidebar.selectbox("Choose Company Stocks", names, index=0)
    if stock != "<Select Names>":
        stock_df = data_prep(data, stock)
        if st.sidebar.button("Show Stock Trend", key=1):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['close'], mode='lines', name='Stock_Trend', line=dict(color='cyan', width=2)))
            fig.update_layout(title='Stock Trend of ' + stock, xaxis_title='Date', yaxis_title='Price ($)')
            st.plotly_chart(fig, use_container_width=True)
            
            start_price = stock_df.iloc[0]['close']
            end_price = stock_df.iloc[-1]['close']
            
            if end_price > start_price:
                trend_note = 'Stock is on a solid upward trend. Investing here might be profitable.'
            else:
                trend_note = 'Stock does not appear to be in a solid uptrend. Better not to invest here; instead, pick a different stock.'
            
            st.markdown(f'<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br> {trend_note}</p>', unsafe_allow_html=True)
            st.write(f"Start Price: {start_price}")
            st.write(f"End Price: {end_price}")

    st.header("Summary of Stock Trends")
    summary = []
    for name in names[1:]:
        stock_df = data_prep(data, name)
        start_price = stock_df.iloc[0]['close']
        end_price = stock_df.iloc[-1]['close']
        trend = "Upward" if end_price > start_price else "Downward"
        summary.append((name, trend))
    
    summary_df = pd.DataFrame(summary, columns=["Stock", "Trend"])
    st.dataframe(summary_df)
    
def strategy_simulation(data, names):
    st.header("Strategy Simulation")
    stock = st.sidebar.selectbox("Choose Company Stocks", names, index=0)
    st.sidebar.subheader("Enter Your Available Initial Investment Fund")
    invest = st.sidebar.slider('Select a range of values', 1000, 1000000)
    
    if stock != "<Select Names>" and st.sidebar.button("Calculate", key=2):
        stock_df = data_prep(data, stock)
        q_table = pkl.load(open('pickl.pkl', 'rb'))
        net_worth = test_stock(stock_df, q_table, invest)
        net_worth_df = pd.DataFrame(net_worth, columns=['value'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=net_worth_df.index, y=net_worth_df['value'], mode='lines', name='Portfolio Value', line=dict(color='cyan', width=2)))
        fig.update_layout(title='Change in Portfolio Value Day by Day', xaxis_title='Number of Days since Start', yaxis_title='Value ($)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br> Increase in your net worth as a result of a model decision.</p>', unsafe_allow_html=True)
        
        metrics = calculate_performance_metrics(net_worth, invest)
        display_performance_metrics(metrics)

def calculate_performance_metrics(net_worth, initial_investment):
    net_worth = np.array(net_worth)
    returns = (net_worth[-1] - initial_investment) / initial_investment
    annualized_return = (net_worth[-1] / initial_investment) ** (365 / len(net_worth)) - 1
    daily_returns = np.diff(net_worth) / net_worth[:-1]
    volatility = np.std(daily_returns)
    sharpe_ratio = annualized_return / volatility

    return {
        "Total Return": returns,
        "Annualized Return": annualized_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio
    }

def display_performance_metrics(metrics):
    st.write("### Performance Metrics")
    for key, value in metrics.items():
        st.write(f"**{key}:** {value:.2f}")

def performance_metrics():
    st.header("Performance Metrics")
    st.write("Please select a stock and run the strategy simulation to view performance metrics.")

if __name__ == '__main__':
    main()


