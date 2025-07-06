import os
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from datetime import datetime, timedelta

# Load or create mood score data
def load_mood_data(csv_path='data/mood_scores.csv'):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Create sample data for demo
        today = datetime.today()
        dates = [today - timedelta(days=i) for i in range(30)][::-1]
        scores = [60 + 20 * (i % 7 == 0) + 10 * (i % 5 == 0) + (i % 10) for i in range(30)]
        df = pd.DataFrame({'ds': [d.strftime('%Y-%m-%d') for d in dates], 'y': scores})
        df.to_csv(csv_path, index=False)
    return df

# Forecast mood trend
def forecast_mood(df, periods=7):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

# Plot mood trendline
def plot_mood_trend(df, forecast):
    plt.figure(figsize=(10,5))
    plt.plot(df['ds'], df['y'], label='Actual Mood Score', marker='o')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mood Score')
    plt.title('Mood Trend Forecast')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = load_mood_data()
    # Prophet expects columns: ds (date), y (value)
    df['ds'] = pd.to_datetime(df['ds'])
    forecast = forecast_mood(df)
    plot_mood_trend(df, forecast) 