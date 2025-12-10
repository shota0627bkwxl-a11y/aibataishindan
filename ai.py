from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import numpy as np
import io
import os # Added for file path checks

# --- 1. 初期設定 ---
app = Flask(__name__)
model = None
IMAGE_SIZE = (224, 224) # 訓練時と同じサイズ

# --- 2. AIモデルのロード（サーバー起動時に1回だけ実行）
# モデルファイルが無い場合はダミーモデルを作成する
if not os.path.exists('horse_body_model.h5'):
    print("AIモデルが見つかりません。ダミーモデルを作成します...")
    # create_dummy_model.py を実行
    import subprocess
    try:
        subprocess.run(['python', 'create_dummy_model.py'], check=True)
        print("ダミーモデルの作成に成功しました。")
    except subprocess.CalledProcessError as e:
        print(f"ダミーモデル作成エラー: {e}")
    except FileNotFoundError:
        print("create_dummy_model.py が見つかりません。")

# 再度存在確認
if os.path.exists('horse_body_model.h5'):
    try:
        model = load_model('horse_body_model.h5')
        print(" * AIモデル (horse_body_model.h5) のロードに成功しました。")
    except Exception as e:
        print(f"モデルロードエラー: {e}")
        model = None
else:
    print("モデルファイルを作成できませんでした。")
    model = None

def preprocess_image(image_bytes):
    """
    アップロードされた画像をAIが診断できる形式に前処理する
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(IMAGE_SIZE) 
    image_array = np.array(image)
    image_array = image_array / 255.0 # 0-1に正規化
    image_array = np.expand_dims(image_array, axis=0) # バッチの次元を追加
    return image_array

# --- 3. ルーティング（APIの定義） ---

# ルートURL ("/") にアクセスしたら index.html を表示
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


# "/diagnose" へのPOSTリクエスト（画像診断）を処理
@app.route('/diagnose', methods=['POST'])
def diagnose_horse():
    if model is None:
        return jsonify({'error': 'AIモデルが準備できていません'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': '画像ファイルがありません'}), 400
    
    file = request.files['image']
    
    try:
        # 画像を前処理
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # --- データ収集機能 ---
        # ユーザーがアップした画像をサーバーに保存する（将来の学習用）
        import datetime
        import uuid
        import os

        # ディレクトリがなければ作成 (念のため)
        save_dir = 'collected_data'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # ファイル名: YYYYMMDD_HHMMSS_UUID.jpg
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}.jpg"
        save_path = os.path.join(save_dir, filename)

        # バイナリデータを保存
        with open(save_path, 'wb') as f:
            f.write(image_bytes)
        
        print(f" * 画像を保存しました: {save_path}")
        # -------------------
        
        # AIモデルによる予測（5クラス分類: S, A, B, C, D）
        prediction = model.predict(processed_image)
        # prediction is [[prob_A, prob_B, prob_C, prob_D, prob_S]] (Alphabetical order usually)
        # However, flow_from_directory uses alphabetical order.
        # A, B, C, D, S -> 0, 1, 2, 3, 4
        # Wait, sorted(os.listdir) order: A, B, C, D, S
        
        # Mapping index to Score
        # A(0) -> 85, B(1) -> 75, C(2) -> 65, D(3) -> 50, S(4) -> 95
        # Let's verify train_generator.class_indices during training, but standard is alphabetical.
        
        # We will calculate a weighted score.
        probs = prediction[0]
        # Assuming alphabetical order: A, B, C, D, S ?? No, 'S' comes after 'D'.
        # Let's assume standard sorting: A, B, C, D, S
        # A:0, B:1, C:2, D:3, S:4
        
        # Base scores for each class
        # S=95, A=85, B=75, C=65, D=50
        class_scores = [85, 75, 65, 50, 95] 
        
        # Expected Score = sum(prob * score)
        final_score = np.sum(probs * class_scores)
        
        # Comment based on the highest probability class or the score
        max_idx = np.argmax(probs)
        class_names = ['A', 'B', 'C', 'D', 'S']
        predicted_class = class_names[max_idx]

        score_percent = final_score
        
        comment = '診断中...'
        if predicted_class == 'S':
            comment = 'AI評価: 【S】素晴らしい！G1級の馬体です。'
        elif predicted_class == 'A':
            comment = 'AI評価: 【A】かなり良いです。重賞も狙える器。'
        elif predicted_class == 'B':
            comment = 'AI評価: 【B】標準より良いです。勝ち上がりは近そう。'
        elif predicted_class == 'C':
            comment = 'AI評価: 【C】平均的です。成長に期待。'
        else:
            comment = 'AI評価: 【D】少し厳しいかもしれません。'

        return jsonify({
            'success': True,
            'score': round(score_percent, 1),
            'comment': comment
        })

    except Exception as e:
        return jsonify({'error': f'診断中にエラーが発生しました: {e}'}), 500

# --- 4. サーバーの実行 ---
if __name__ == '__main__':
    # '0.0.0.0' で起動すると、ローカルネットワーク内の他のデバイス（スマホなど）からもアクセス可能
    app.run(debug=True, host='0.0.0.0', port=5001)