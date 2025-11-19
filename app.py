import os
# 關鍵優化：抑制 TensorFlow 的啟動日誌和警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 確保使用 CPU

import numpy as np
import jieba
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google import genai
from google.genai import types
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 0. 全域變數與模型載入 (僅在服務啟動時執行一次) ---

client = genai.Client()

emotion_classes = np.array(['厭惡', '喜悅', '平靜', '悲傷', '憤怒', '期待', '焦慮', '驚訝']) 
max_len = 16 

# 🚨 最終修正：繞過耗時的模型載入，以測試服務是否能啟動
# 因為載入 emotion_model.h5 超過 3 分鐘或造成記憶體崩潰

# 創建一個模擬模型 (Mock Model) 類別
class MockEmotionModel:
    """用於取代 TensorFlow 模型，讓服務快速啟動並模擬一個預測結果 (例如: 焦慮)。"""
    def predict(self, padded_sequence, verbose=0):
        # 模擬一個 '焦慮' (索引 6) 的高信心度結果
        return np.array([[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.65, 0.05]]) 
# 將最終模型設置為模擬模型
final_model = MockEmotionModel() 
print("模型載入已旁路。正在使用模擬模型進行啟動測試。")


# 修正後的 Tokenizer 重建邏輯 (這部分必須成功，因為是純 Python 操作)
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

print("--- 正在重建 Tokenizer... ---")
try:
    # 載入數據用於重建 Tokenizer
    df = pd.read_csv('emotion_data.csv', header=None, names=['text', 'emotion'])
    
    # 重新執行分詞與建立
    df['tokens'] = df['text'].apply(lambda x: list(jieba.cut(x, cut_all=False)))
    texts = [" ".join(tokens) for tokens in df['tokens']]
    
    # 重新建立 Tokenizer 變數
    tokenizer = Tokenizer(num_words=5000, oov_token="<unk>") 
    tokenizer.fit_on_texts(texts)
    
    print("'tokenizer' 已成功從 emotion_data.csv 重建！")

except FileNotFoundError:
    print("FATAL ERROR: 無法找到 'emotion_data.csv' 檔案。請確保它在 app.py 的同一個資料夾中！")

# --- 1. 核心邏輯函式 ---

# 1.1 LSTM 判斷情緒 (現在會使用 MockModel.predict)
def predict_emotion(text_input, model, tokenizer, max_len, emotion_classes):
    """使用 Mock 模型預測輸入文本的情緒，以繞過 TF 載入問題。"""
    # 這裡的分詞和 Padding 仍然是必要的步驟
    tokens = list(jieba.cut(text_input, cut_all=False))
    text_processed = [" ".join(tokens)]
    sequence = tokenizer.texts_to_sequences(text_processed)
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    
    # 呼叫 MockModel.predict，它會回傳我們預設的 '焦慮' 結果
    predictions = model.predict(padded_sequence, verbose=0)
    
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_emotion = emotion_classes[predicted_class]
    confidence = predictions[0][predicted_class]
    
    return predicted_emotion, confidence

# 1.2 推薦邏輯 (保持不變)
recommendation_logic = {
    '喜悅': {'type': '社交型興趣 / 分享', 'reason': '你心情超棒！是時候跟朋友分享這份喜悅，舉辦一場美食聚會吧！'},
    '悲傷': {'type': '創造型 / 發洩型興趣', 'reason': '你現在的情緒需要出口，不如拿起紙筆畫下你的心情，或寫點東西吧！讓創作替你說出那些說不出口的感受。'},
    '憤怒': {'type': '破壞型 / 高強度型興趣', 'reason': '火氣滿滿？找個安全的方式把力量釋放出去！去做高強度運動、打沙包，讓情緒在運動裡被燃燒掉。'},
    '焦慮': {'type': '專注型 / 重複型興趣', 'reason': '心有點亂？試試需要重複動作的小活動吧！像是拼圖、摺紙或分類小物，專注感會讓焦慮慢慢安靜下來。'},
    '平靜': {'type': '探索型 / 放鬆型興趣', 'reason': '你現在散發著穩定的能量～不妨散步探索周遭，或聽點輕音樂，享受這份難得的平和。'},
    '厭惡': {'type': '清理型 / 轉換型興趣', 'reason': '有種被東西惹毛的感覺？那正是整理的好時機！清掉煩人的雜物，讓環境和心情一起煥然一新。'},
    '期待': {'type': '計劃型 / 創作型興趣', 'reason': '你興奮得像準備開啟新關卡！趁這股能量，把你的計畫具體化吧～列清單、找靈感、做腦力激盪，讓期待變成行動。'},
    '驚訝': {'type': '探索型 / 認知型興趣', 'reason': '哇！你的好奇心被點亮了！不如趁勢多了解一下剛剛讓你驚訝的事，查資料、看影片，讓驚訝變成有趣的新發現。'}
}

# 1.3 生成式推薦函式 (Gemini 組織語言與錯誤處理)
def generate_conversational_recommendation(text_input, model, logic, client):
    """結合情緒預測結果，呼叫 Gemini 生成個性化的教練建議。"""
    try:
        # 即使是 Mock 模型，我們也需要運行 predict_emotion
        predicted_emotion, confidence = predict_emotion(text_input, final_model, tokenizer, max_len, emotion_classes)
    except Exception as e:
        return {
            'ai_response': f"LSTM 模型預測失敗：{str(e)}",
            'predicted_emotion': "錯誤",
            'confidence': "0.00%"
        }
    
    recommendation_info = logic.get(predicted_emotion, {'type': '無此類別', 'reason': '請休息'})
    
    prompt = f"""
    你是一個溫暖、專業、幽默的 AI 心理教練。你對重機、電吉他、美食探索等多種興趣有深刻見解。
    你的任務是根據以下資訊，用親切的口語化語氣，鼓勵用戶並給予一個具體的、與他們興趣（重機、電吉他、美食）相關的行動建議。
    請避免使用固定的模板。

    用戶的原始輸入是: "{text_input}"
    系統判斷的情緒是: {predicted_emotion}
    系統建議的活動類型是: {recommendation_info['type']}
    系統建議的固定理由是: {recommendation_info['reason']}
    
    請根據這些資訊，生成一段流暢的鼓勵和推薦語。
    """
    
    try:
        # 嘗試呼叫 Gemini API
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt
        )
        
        return {
            'ai_response': response.text,
            'predicted_emotion': predicted_emotion,
            'confidence': f"{confidence*100:.2f}%"
        }
        
    except genai.errors.PermissionDeniedError:
        # 專門處理 API Key 錯誤 (403)
        return {
            'ai_response': "Gemini API 呼叫失敗：權限遭拒。請檢查 Render 上的 GEMINI_API_KEY 是否設定正確且有效。",
            'predicted_emotion': predicted_emotion,
            'confidence': f"{confidence*100:.2f}%"
        }
    except Exception as e:
        # 處理其他所有 API 錯誤 (如 400, 500, Timeout)
        return {
            'ai_response': f"Gemini API 呼叫失敗：發生未知錯誤 {type(e).__name__}，請檢查 Render 日誌。",
            'predicted_emotion': predicted_emotion,
            'confidence': f"{confidence*100:.2f}%"
        }


# --- 2. Flask API 定義 ---
app = Flask(__name__)
CORS(app) # 啟用 CORS

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        user_text = data.get('text', '')
        
        if not user_text:
            return jsonify({"error": "Missing 'text' in request body"}), 400

        # 呼叫核心處理函式
        result = generate_conversational_recommendation(user_text, final_model, recommendation_logic, client)
        
        return jsonify(result)

    except Exception as e:
        print(f"API 處理錯誤: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
