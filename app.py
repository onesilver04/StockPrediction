from flask import Flask, request, render_template
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# 모델 로드
model = load_model('tmp_checkpoint.keras')

def prepare_data(ticker):
    # 주식 데이터를 다운로드
    df = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    
    # 데이터가 비어 있는지 확인
    if df.empty:
        return None, None, None
    
    # 필요한 컬럼만 선택 및 전처리
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    
    # 데이터 정규화
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    
    # 데이터셋 준비
    TEST_SIZE = 200
    data = df_scaled[-TEST_SIZE:]
    
    return df, data, scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    
    # 데이터 준비
    original_df, data, scaler = prepare_data(ticker)
    
    # 데이터가 없을 경우 처리
    if data is None:
        return render_template('index.html', prediction_text="No data found for the given ticker.")
    
    # 데이터셋을 모델 입력에 맞게 변환
    window_size = 20
    input_data = data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
    feature_list = []
    
    for i in range(len(input_data) - window_size):
        feature_list.append(np.array(input_data.iloc[i:i+window_size]))
    
    feature_list = np.array(feature_list)
    
    # 모델 예측
    predictions = model.predict(feature_list)
    
    # 예측된 Close 값을 복원하기 위한 빈 배열 생성
    predictions_extended = np.zeros((predictions.shape[0], data.shape[1]))
    
    # 예측된 Close 값을 빈 배열의 원래 Close 열에 삽입
    predictions_extended[:, data.columns.get_loc('Close')] = predictions[:, 0]

    # 예측 데이터 복원 (역 정규화)
    inv_predictions = scaler.inverse_transform(predictions_extended)

    # 복원된 Close 값만 추출
    final_predictions = inv_predictions[:, data.columns.get_loc('Close')]
    
    # 실제 데이터와 예측 데이터 시각화
    dates = original_df.index[-len(final_predictions):]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dates, original_df['Close'].values[-len(final_predictions):], label='Actual Price')
    plt.plot(dates, final_predictions, label='Predicted Price', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend()
    
    # 그래프를 이미지로 변환
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return render_template('index.html', plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
