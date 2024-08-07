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
    df['5day_MA'] = df['close'].rolling(window=5).mean().fillna(0)
    df['1day_MA'] = df['close'].rolling(window=1).mean().fillna(0)
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
        return np.random.randint(3)  # Random action
    else:
        q_values = model.predict(state.reshape(1, -1))
        return np.argmax(q_values)  # Best action

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
    net_worth = [10000]  # Initial investment
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
    st.title("Stock Trading Strategy Enhancement with Deep Q-Network")

    # Upload CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data['date'] = pd.to_datetime(data['date'])
        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)
        data['5day_MA'] = data['close'].rolling(window=5).mean().fillna(0)
        data['1day_MA'] = data['close'].rolling(window=1).mean().fillna(0)
        
        st.write("Data Overview:")
        st.write(data.head())

        # Sidebar
        st.sidebar.title("Settings")
        st.sidebar.subheader("Choose Stock and Investment")
        stock = st.sidebar.selectbox("Select Stock", data['Name'].unique())
        stock_df = data_prep(data, stock)

        if st.sidebar.button("Show Stock Trend"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['close'], mode='lines', name='Close Price'))
            fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['5day_MA'], mode='lines', name='5-Day MA'))
            fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['1day_MA'], mode='lines', name='1-Day MA'))
            fig.update_layout(title=f'Stock Trend for {stock}', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig, use_container_width=True)

        invest = st.sidebar.slider('Initial Investment', 1000, 1000000, 10000)
        if st.sidebar.button("Calculate"):
            st.spinner('Training DQN model...')
            model = train_dqn(stock_df, episodes=100)
            net_worth = test_dqn(stock_df, model)
            net_worth_df = pd.DataFrame(net_worth, columns=['Value'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=net_worth_df.index, y=net_worth_df['Value'], mode='lines', name='Portfolio Value'))
            fig.update_layout(title='Portfolio Value Over Time', xaxis_title='Days', yaxis_title='Value ($)')
            st.plotly_chart(fig, use_container_width=True)
            st.success('Calculation complete!')
            st.markdown('**NOTE:** The increase in net worth is based on model decisions.')

if __name__ == '__main__':
    main()
