import os
# 關鍵優化：抑制 TensorFlow 的啟動日誌和警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # 確保使用 CPU

import numpy as np
import jieba
# 為了避免載入錯誤，我們只在必要時導入這些
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    import pandas as pd
except ImportError as e:
    # 如果深度學習庫安裝失敗，將會在這裡捕獲
    print(f"FATAL ERROR: 無法導入深度學習庫 ({e})。請檢查 requirements.txt!")
    tf = None # 將這些模組設為 None 以避免後續崩潰

from google import genai
from google.genai.errors import APIError
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 0. 全域變數與模型載入 (僅在服務啟動時執行一次) ---

# 🚨 務必在 Render 環境變數中設定 GEMINI_API_KEY
client = genai.Client()

emotion_classes = np.array(['厭惡', '喜悅', '平靜', '悲傷', '憤怒', '期待', '焦慮', '驚訝']) 
max_len = 16 

# 初始化 tokenizer 和 model 變數
tokenizer = None
final_model = None

# 🚨 模型載入旁路 (Mock Model) - 保持不變，以確保啟動速度
class MockEmotionModel:
    """用於取代 TensorFlow 模型，讓服務快速啟動並模擬一個預測結果 (例如: 焦慮)。"""
    def predict(self, padded_sequence, verbose=0):
        # 模擬一個 '焦慮' (索引 6) 的高信心度結果
        return np.array([[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.65, 0.05]]) 

final_model = MockEmotionModel() 
print("模型載入已旁路。正在使用模擬模型進行啟動測試。")


# Tokenizer 重建邏輯
print("--- 正在嘗試重建 Tokenizer... ---")
if tf is not None and pd is not None and Tokenizer is not None:
    try:
        # 載入數據用於重建 Tokenizer (假設 emotion_data.csv 存在於根目錄)
        df = pd.read_csv('emotion_data.csv', header=None, names=['text', 'emotion'])
        
        # 重新執行分詞與建立
        df['tokens'] = df['text'].apply(lambda x: list(jieba.cut(x, cut_all=False)))
        texts = [" ".join(tokens) for tokens in df['tokens']]
        
        # 重新建立 Tokenizer 變數
        tokenizer = Tokenizer(num_words=5000, oov_token="<unk>") 
        tokenizer.fit_on_texts(texts)
        
        print("'tokenizer' 已成功從 emotion_data.csv 重建！")

    except FileNotFoundError:
        print("CRITICAL ERROR: 無法找到 'emotion_data.csv' 檔案。Tokenizer 建立失敗。")
        tokenizer = 'FILE_NOT_FOUND' # 設置為錯誤標誌
    except Exception as e:
        print(f"CRITICAL ERROR: Tokenizer 建立失敗，原因: {e}")
        tokenizer = 'BUILD_FAILED' # 設置為錯誤標誌
else:
    print("CRITICAL ERROR: 由於函式庫導入失敗，Tokenizer 無法建立。")
    tokenizer = 'LIB_FAILED'


# --- 1. 核心邏輯函式 ---

# 1.1 LSTM 判斷情緒 (現在會使用 MockModel.predict)
def predict_emotion(text_input, model, tokenizer, max_len, emotion_classes):
    """使用 Mock 模型預測輸入文本的情緒，以繞過 TF 載入問題。"""
    
    if isinstance(tokenizer, str):
         # 如果 Tokenizer 是一個錯誤標誌（字串），直接返回錯誤
         return 'Tokenizer 失敗', 0.0
         
    # 這裡的分詞和 Padding 仍然是必要的步驟
    tokens = list(jieba.cut(text_input, cut_all=False))
    text_processed = [" ".join(tokens)]
    
    # 這裡可能因為 tokenizer 載入不全而崩潰
    try:
        sequence = tokenizer.texts_to_sequences(text_processed)
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    except Exception as e:
        # 如果 Tokenizer 語法級崩潰
        return f'Tokenizer 處理崩潰: {str(e)[:50]}...', 0.0
    
    # 呼叫 MockModel.predict
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
    '驚訝': {'type': '探索型 / 認知型興趣', 'reason': '哇！你的好奇心被點亮了！不如趁勢多了解一下剛剛讓你驚訝的事，查資料、看影片，讓驚訝變成有趣的新發現。'},
    'Tokenizer 失敗': {'type': '錯誤診斷', 'reason': '由於情緒分析模組未啟動，無法提供推薦。'},
    'Tokenizer 處理崩潰': {'type': '錯誤診斷', 'reason': '情緒分析模組處理輸入時崩潰，無法提供推薦。'}
}

# 1.3 生成式推薦函式 (Gemini 組織語言與錯誤處理)
def generate_conversational_recommendation(text_input, model, logic, client):
    """結合情緒預測結果，呼叫 Gemini 生成個性化的教練建議。"""
    try:
        predicted_emotion, confidence = predict_emotion(text_input, final_model, tokenizer, max_len, emotion_classes)
    except Exception as e:
        return {
            'ai_response': f"LSTM 模型預測失敗：{str(e)}",
            'predicted_emotion': "錯誤",
            'confidence': "0.00%"
        }
    
    recommendation_info = logic.get(predicted_emotion, {'type': '無此類別', 'reason': '請休息'})
    
    # 🚨 如果 Tokenizer 失敗，我們不呼叫 Gemini，直接返回診斷結果
    if predicted_emotion in ['Tokenizer 失敗', 'Tokenizer 處理崩潰:']:
        return {
            'ai_response': f"系統錯誤診斷：{predicted_emotion}。請檢查 'emotion_data.csv' 檔案是否正確放置於伺服器。",
            'predicted_emotion': predicted_emotion,
            'confidence': f"{confidence*100:.2f}%"
        }

    
    # 🚨 最終修正：恢復完整的 Gemini 提示詞
    prompt = f"""
    你是一個溫暖、專業、幽默的 AI 心理教練。你對重機、電吉他、美食探索等多種興趣有深刻見解。
    你的任務是根據以下資訊，用輕鬆且鼓勵的語氣給出一個**個性化的興趣推薦**，並**融入一到兩個你提到的專業興趣**（重機、吉他、美食等）作為建議的例子。
    
    用戶原始輸入："{text_input}"
    模型預測情緒：{predicted_emotion}
    情緒建議類型：{recommendation_info['type']}
    建議原因：{recommendation_info['reason']}
    
    請遵循以下格式：
    1. 開頭先用 1-2 句話溫暖地回應用戶的情緒。
    2. 接著用 1-2 句話說明這是屬於哪一種興趣類型（例如：『這時候你需要的是「發洩型」的興趣！』）。
    3. 最後提出至少 3 個具體的興趣建議。
    4. 必須確保你給出的建議是有建設性且正面的。
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
        
    except APIError as e:
        error_message = str(e)
        if "Permission denied" in error_message or "Invalid API key" in error_message:
             return {
                'ai_response': "Gemini API 呼叫失敗：權限遭拒。請檢查 Render 上的 GEMINI_API_KEY 是否設定正確且有效。",
                'predicted_emotion': predicted_emotion,
                'confidence': f"{confidence*100:.2f}%"
            }
        else:
            return {
                'ai_response': f"Gemini API 呼叫失敗：發生 API 錯誤 {type(e).__name__}，錯誤訊息：{error_message[:100]}...",
                'predicted_emotion': predicted_emotion,
                'confidence': f"{confidence*100:.2f}%"
            }
    except Exception as e:
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
    # 🚨 啟動前的最終檢查：如果 Tokenizer 標記為錯誤，直接返回 503
    global tokenizer
    if isinstance(tokenizer, str) and tokenizer in ['FILE_NOT_FOUND', 'BUILD_FAILED', 'LIB_FAILED']:
         # 503 Service Unavailable (服務不可用) 更適合描述服務的核心依賴缺失
         return jsonify({
             "error": "Service Unavailable (503)",
             "message": f"核心依賴遺失或初始化失敗。Tokenizer 狀態: {tokenizer}。請確認 'emotion_data.csv' 檔案已在 GitHub 倉庫中。"
         }), 503
         
    try:
        data = request.get_json()
        user_text = data.get('text', '')
        
        if not user_text:
            return jsonify({"error": "Missing 'text' in request body"}), 400

        # 呼叫核心處理函式
        result = generate_conversational_recommendation(user_text, final_model, recommendation_logic, client)
        
        return jsonify(result)

    except Exception as e:
        # 捕獲所有意料之外的執行期錯誤
        print(f"API 處理錯誤: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
