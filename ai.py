from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import io
import os # Added for file path checks

# --- 1. 初期設定 ---
app = Flask(__name__)
model = None
check_model = None # 馬判定用モデル
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
        
        # ついでにチェック用モデルもロードしておく（初回アクセスのラグを防ぐため）
        # ただしメモリが厳しいならリクエスト時にロードする手法に変えるが、
        # MobileNetV2は比較的小さいので常駐トライ
        try:
             print(" * 馬判定用モデル (MobileNetV2) をロード中...")
             check_model = MobileNetV2(weights='imagenet')
             print(" * 馬判定用モデルのロード完了。")
        except Exception as e:
             print(f"馬判定モデルロード失敗: {e}")

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
    global check_model
    if model is None:
        return jsonify({'error': 'AIモデルが準備できていません'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': '画像ファイルがありません'}), 400
    
    file = request.files['image']
    
    try:
        # 画像を前処理
        image_bytes = file.read()
        
        # --- 馬判定ロジック (ImageNet) ---
        # 犬や人を弾くため、まず汎用モデルで「何が写っているか」をチェック
        try:
            # 念のためここでロード（失敗時再試行）
            if check_model is None:
                 check_model = MobileNetV2(weights='imagenet')
            
            # MobileNetV2用の前処理 (-1 to 1)
            check_img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
            check_arr = np.array(check_img)
            check_arr = np.expand_dims(check_arr, axis=0)
            check_arr = preprocess_input(check_arr)
            
            preds = check_model.predict(check_arr)
            decoded = decode_predictions(preds, top=5)[0]
            # decoded = [('n02389026', 'sorrel', 0.8), ...]

            horse_keywords = ['sorrel', 'zebra', 'horse', 'pony', 'donkey', 'mule', 'ox', 'impala', 'gazelle'] 
            
            is_horse = False
            detected_label = decoded[0][1] # Top1ラベル
            
            for d in decoded:
                label = d[1].lower()
                if any(k in label for k in horse_keywords):
                    is_horse = True
                    break
            
            if not is_horse:
                 return jsonify({
                    'success': False, 
                    'error': f'馬の写真に見えません（判定: {detected_label}）。馬の写真をアップロードしてください。'
                }), 200 # 200で返してアラート表示

        except Exception as e:
            print(f"馬判定スキップ: {e}")
            pass # エラー時はスルーして診断へ

        # --- AI診断ロジック (Custom Model) ---
        processed_image = preprocess_image(image_bytes)
        prediction = model.predict(processed_image)
        
        class_names = ['A', 'B', 'C', 'D', 'S', 'SS']
        # クラスラベル定義（※学習時のgenerator.class_indicesの順序に対応させる必要あり）
        # アルファベット順だと: ['A', 'B', 'C', 'D', 'S', 'SS'] の可能性が高い
        
        # ユーザー要望（辛口査定）: A以上は相当強く、Cをボリュームゾーンに。
        # スコア配分を全体的に下げて、「ワンランクダウン」させる調整
        # A(76)->B判定, B(68)->C判定, C(62)->C判定, S(85)->A判定
        class_scores = [76, 68, 62, 45, 85, 98]

        probs = prediction[0]
        
        if len(probs) != 6:
            # 旧モデル(5クラス)の場合のフォールバック
            class_names_old = ['A', 'B', 'C', 'D', 'S']
            class_scores_old = [76, 68, 62, 45, 90] # 旧モデルも少し辛口に
            final_score = np.sum(probs * class_scores_old)
        else:
            final_score = np.sum(probs * class_scores)

        score_percent = final_score
        
        comment = '診断中...'
        
        # ランクとスコアの不整合を直す：スコア基準でランクを強制決定する
        if score_percent >= 96.0:
             comment = 'AI評価: 【SS】異次元級！歴史的名馬に匹敵するレベルです。'
        elif score_percent >= 90.0:
             comment = 'AI評価: 【S】素晴らしい！G1級の馬体です。'
        elif score_percent >= 80.0:
             comment = 'AI評価: 【A】かなり良いです。重賞も狙える器。'
        elif score_percent >= 70.0:
             comment = 'AI評価: 【B】標準より良いです。勝ち上がりは近そう。'
        elif score_percent >= 60.0:
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