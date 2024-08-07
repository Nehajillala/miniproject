import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Data Preparation
@st.cache
def data_prep(data, name):
    df = data[data['Name'] == name].copy()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['5day_MA'] = df['close'].rolling(window=5).mean().fillna(0)
    df['1day_MA'] = df['close'].rolling(window=1).mean()
    return df

# Build DQN Model
def build_dqn_model(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(3, activation='linear')  # 3 actions: buy, sell, hold
    ])
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
    q_values = model.predict(state.reshape(1, -1))
    return np.argmax(q_values)  # Best action

# Train DQN
def train_dqn(df, episodes, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
    model = build_dqn_model(input_dim=3)
    for _ in range(episodes):
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
    st.set_page_config(page_title="Stock Trading Strategy", layout="wide")
    st.title("Stock Trading Strategy With Deep Q-Network")

    # Upload CSV
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data['date'] = pd.to_datetime(data['date'])
        data = data[['date', 'open', 'high', 'low', 'close', 'volume', 'Name']].dropna()
        data.reset_index(drop=True, inplace=True)

        st.write("Data Overview:")
        st.dataframe(data.head())

        st.sidebar.title("Choose Stock and Investment")
        stock = st.sidebar.selectbox("Select Stock", data['Name'].unique())
        stock_df = data_prep(data, stock)

        if st.sidebar.button("Show Stock Trend"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['close'], mode='lines', name='Close'))
            fig.add_trace(go.Scatter(x=stock_df['date'], y=stock_df['5day_MA'], mode='lines', name='5-Day MA'))
            fig.update_layout(title=f'Stock Trend for {stock}', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig, use_container_width=True)

        invest = st.sidebar.slider('Initial Investment', 1000, 1000000, value=10000)
        if st.sidebar.button("Calculate"):
            with st.spinner("Training the model..."):
                model = train_dqn(stock_df, episodes=100)
                net_worth = test_dqn(stock_df, model)
                net_worth_df = pd.DataFrame(net_worth, columns=['Value'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=net_worth_df.index, y=net_worth_df['Value'], mode='lines', name='Portfolio Value'))
                fig.update_layout(title='Portfolio Value Over Time', xaxis_title='Days', yaxis_title='Value ($)')
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('**NOTE:** Increase in net worth based on model decisions.')

if __name__ == '__main__':
    main()
