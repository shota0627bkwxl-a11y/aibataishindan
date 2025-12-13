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

# --- 2. AIモデルのロード（サーバー起動時に1回だけ実行） ---

# Google Driveなどからモデルをダウンロードする機能
# 環境変数 'MODEL_URL' が設定されていれば、そこからダウンロードを試みる
MODEL_URL = os.environ.get('MODEL_URL')

if MODEL_URL:
    print(f"環境変数 MODEL_URL が設定されています。モデルのダウンロードを試みます...")
    try:
        import gdown
        # 既存のモデルがあれば削除して上書き（最新にするため）
        if os.path.exists('horse_body_model.h5'):
            os.remove('horse_body_model.h5')
        
        # gdownを使ってダウンロード (fuzzy=TrueでGoogleDriveのセキュリティ警告を回避)
        output = 'horse_body_model.h5'
        gdown.download(MODEL_URL, output, quiet=False, fuzzy=True)
        print("モデルのダウンロードが完了しました。")
    except Exception as e:
        print(f"モデルのダウンロードに失敗しました: {e}")

# モデルファイルが無い場合はダミーモデルを作成する (ダウンロード失敗時や未設定時)
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

# モデルロード
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
        
        # AIモデルによる予測（6クラス分類: SS, S, A, B, C, D）になる予定
        prediction = model.predict(processed_image)
        
        # アルファベット順: A, B, C, D, S, SS
        # Index: 0, 1, 2, 3, 4, 5
        class_names = ['A', 'B', 'C', 'D', 'S', 'SS']
        class_scores = [85, 75, 65, 50, 95, 100]

        probs = prediction[0]
        
        # もしモデルがまだ5クラス（SS無し）のままだとエラーになる可能性があるので、
        # 配列の長さチェックを入れるのが安全だが、今回はモデルも作り直す前提で進める。
        if len(probs) != 6:
            # 旧モデル(5クラス)の場合のフォールバック
            # A, B, C, D, S -> 0,1,2,3,4
            class_names_old = ['A', 'B', 'C', 'D', 'S']
            class_scores_old = [85, 75, 65, 50, 100] # Sを100にしておく
            final_score = np.sum(probs * class_scores_old)
            max_idx = np.argmax(probs)
            predicted_class = class_names_old[max_idx]
            
            # 擬似SS判定
            if final_score >= 96: predicted_class = 'SS'
            
        else:
            # 新モデル(6クラス)の場合
            final_score = np.sum(probs * class_scores)
            max_idx = np.argmax(probs)
            predicted_class = class_names[max_idx]

        score_percent = final_score
        
        comment = '診断中...'
        
        if predicted_class == 'SS' or score_percent >= 96.0:
             comment = 'AI評価: 【SS】異次元級！歴史的名馬に匹敵するレベルです。。'
             predicted_class = 'SS' # 強制上書き
        elif predicted_class == 'S':
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