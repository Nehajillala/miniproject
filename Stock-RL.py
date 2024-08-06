import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Data Preparation
@st.cache
def data_prep(data, name):    
    df = pd.DataFrame(data[data['Name'] == name])
    df.dropna(inplace=True)    
    df.reset_index(drop=True, inplace=True)
    df['5day_MA'] = df['close'].rolling(5).mean()
    df['1day_MA'] = df['close'].rolling(1).mean()
    df['5day_MA'].iloc[:4] = 0
    return df

# Build DQN Model
def build_dqn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='linear'))  # 3 actions: buy, sell, hold
    model.compile(optimizer='adam', loss='mse')
    return model

# Get State
def get_state(df, t):
    state = np.array([
        df.iloc[t]['5day_MA'],
        df.iloc[t]['1day_MA'],
        df.iloc[t]['close']
    ])
    return state

# Epsilon-Greedy Strategy
def epsilon_greedy_policy(state, model, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(3)
    else:
        q_values = model.predict(state.reshape(1, -1))
        return np.argmax(q_values)

# Train DQN
def train_dqn(df, episodes, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    model = build_dqn_model(input_dim=3)
    for e in range(episodes):
        state = get_state(df, 0)
        total_profit = 0
        for t in range(len(df) - 1):
            action = epsilon_greedy_policy(state, model, epsilon)
            next_state = get_state(df, t + 1)
            reward = 0
            if action == 0:  # Buy
                total_profit -= df.iloc[t]['close']
            elif action == 1:  # Sell
                total_profit += df.iloc[t]['close']
                reward = total_profit
            elif action == 2:  # Hold
                reward = 0

            q_update = reward + gamma * np.max(model.predict(next_state.reshape(1, -1)))
            q_values = model.predict(state.reshape(1, -1))
            q_values[0][action] = q_update

            model.fit(state.reshape(1, -1), q_values, epochs=1, verbose=0)
            state = next_state

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
                
    return model

# Testing DQN
def test_dqn(df, model):
    state = get_state(df, 0)
    total_profit = 0
    net_worth = [10000]
    for t in range(len(df) - 1):
        action = epsilon_greedy_policy(state, model, epsilon=0.0)
        next_state = get_state(df, t + 1)
        if action == 0:  # Buy
            total_profit -= df.iloc[t]['close']
        elif action == 1:  # Sell
            total_profit += df.iloc[t]['close']
        net_worth.append(net_worth[-1] + total_profit)
        state = next_state
    return net_worth

# Streamlit App
def main():
    # Reading the Dataset
    data = pd.read_csv('all_stocks_5yr.csv')
    names = list(data['Name'].unique())
    names.insert(0, "<Select Names>")

    st.title("Optimizing Stock Trading Strategy With Deep Q-Network")

    st.sidebar.title("Choose Stock and Investment")
    st.sidebar.subheader("Choose Company Stocks")
    stock = st.sidebar.selectbox("(*select one stock only)", names, index=0)
    stock_df = data_prep(data, stock)
    
    if st.sidebar.button("Show Stock Trend", key=1):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['close'], mode='lines', name='Stock_Trend', line=dict(color='cyan', width=2)))
        fig.update_layout(title='Stock Trend of ' + stock, xaxis_title='Date', yaxis_title='Price ($)')
        st.plotly_chart(fig, use_container_width=True)
        
        if stock_df.iloc[500]['close'] > stock_df.iloc[0]['close']:
            st.markdown('<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br>Stock is on a solid upward trend. Investing here might be profitable.</p>', unsafe_allow_html=True)
        else:  
            st.markdown('<b><p style="font-family:Play; color:Red; font-size: 20px;">NOTE:<br>Stock does not appear to be in a solid uptrend. Better not to invest here; instead, pick different stock.</p>', unsafe_allow_html=True)

    st.sidebar.subheader("Enter Your Available Initial Investment Fund")
    invest = st.sidebar.slider('Select a range of values', 1000, 1000000)
    if st.sidebar.button("Calculate", key=2):
        model = train_dqn(stock_df, episodes=100)
        net_worth = test_dqn(stock_df, model)
        net_worth_df = pd.DataFrame(net_worth, columns=['value'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=net_worth_df.index, y=net_worth_df['value'], mode='lines', name='Portfolio Value', line=dict(color='cyan', width=2)))
        fig.update_layout(title='Change in Portfolio Value Day by Day', xaxis_title='Number of Days since Feb 2013', yaxis_title='Value ($)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<b><p style="font-family:Play; color:Cyan; font-size: 20px;">NOTE:<br> Increase in your net worth as a result of a model decision.</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()



            
 
 
       

