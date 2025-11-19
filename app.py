import os
import numpy as np
import jieba
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from google import genai
from google.genai import types
from flask import Flask, request, jsonify

# --- 0. 全局變數與模型載入 (僅在服務啟動時執行一次) ---

# 🚨 安全提醒：實際部署時，請使用環境變數或密鑰管理服務。
# os.environ['GEMINI_API_KEY'] = 'AIzaSyA-YzMyQt_BIccMVqnt9t2IjoWq12P5rbQ'
client = genai.Client()

emotion_classes = np.array(['厭惡', '喜悅', '平靜', '悲傷', '憤怒', '期待', '焦慮', '驚訝']) 
max_len = 16 

# 載入你訓練好的模型
try:
    final_model = load_model('emotion_model.h5')
    print("模型載入成功。")
except Exception as e:
    print(f"錯誤：無法載入 emotion_model.h5。請確認檔案是否存在。錯誤訊息: {e}")
    # 服務啟動失敗，應停止

# 🌟 【重要步驟：載入或重建 Tokenizer】🌟
# 由於 tokenizer 變數不會被保存在 .h5 檔案中，你必須在這裡重新定義或載入它！
# 最簡單的方法是從你的 CSV 數據中重建它，像你之前在 Notebook 中做的那樣。

# 假設你已經在這裡重建了 tokenizer 變數：
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

print("--- 正在重建 Tokenizer... ---")
# 警告：如果 emotion_data.csv 不在 app.py 同一資料夾，這裡會失敗
df = pd.read_csv('emotion_data.csv', header=None, names=['text', 'emotion'])
df['tokens'] = df['text'].apply(lambda x: list(jieba.cut(x, cut_all=False)))
texts = [" ".join(tokens) for tokens in df['tokens']]
tokenizer = Tokenizer(num_words=5000, oov_token="<unk>") 
tokenizer.fit_on_texts(texts)

print("'tokenizer' 已成功從 emotion_data.csv 重建！")

# --- 1. 核心邏輯函式 (複製自你的 Notebook) ---

# 1.1 LSTM 判斷情緒
def predict_emotion(text_input, model, tokenizer, max_len, emotion_classes):
    # 這裡假設 tokenizer 已經作為全局變數定義
    tokens = list(jieba.cut(text_input, cut_all=False))
    text_processed = [" ".join(tokens)]
    sequence = tokenizer.texts_to_sequences(text_processed)
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    predictions = model.predict(padded_sequence, verbose=0)
    
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_emotion = emotion_classes[predicted_class]
    confidence = predictions[0][predicted_class]
    
    return predicted_emotion, confidence

# 1.2 推薦邏輯 (你定稿的表格)
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

# 1.3 生成式推薦函式 (Gemini 組織語言)
def generate_conversational_recommendation(text_input, model, logic, client):
    # with graph.as_default(): # 適用於舊版 TensorFlow 多執行緒
    predicted_emotion, confidence = predict_emotion(text_input, model, tokenizer, max_len, emotion_classes)
    
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
    
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=prompt
    )
    
    return {
        'ai_response': response.text,
        'predicted_emotion': predicted_emotion,
        'confidence': f"{confidence*100:.2f}%"
    }


# --- 2. Flask API 定義 ---
app = Flask(__name__)

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        # 接收前端 POST 過來的 JSON 資料
        data = request.get_json()
        user_text = data.get('text', '')
        
        if not user_text:
            return jsonify({"error": "Missing 'text' in request body"}), 400

        # 呼叫核心推薦邏輯
        result = generate_conversational_recommendation(user_text, final_model, recommendation_logic, client)
        
        # 返回 JSON 格式的結果給前端
        return jsonify(result)

    except Exception as e:
        print(f"API 處理錯誤: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

# --- 3. 服務啟動 ---
if __name__ == '__main__':
    # 服務將在本地 5000 埠口運行
    print("Flask 服務啟動中...")
    app.run(host='0.0.0.0', port=5000)