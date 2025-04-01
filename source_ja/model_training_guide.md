# Qwen2.5モデルの学習ガイド

このドキュメントでは、Qwen2.5モデルのダウンロードから学習、モニタリングまでの一連の流れを説明します。

## 1. 必要なライブラリのインストール

まず、必要なライブラリをインストールします。

```bash
pip install -e . && pip install tf-keras deepspeed slack-sdk && pip install flash-attn --no-build-isolation
```

このコマンドにより、以下のライブラリがインストールされます：
- SWIFTフレームワーク（ローカルインストール）
- TensorFlow Keras
- DeepSpeed（分散学習フレームワーク）
- Slack SDK（通知用）
- Flash Attention（高速なAttention実装）

また、学習結果をクラウドストレージに転送するために必要なrcloneもインストールします：

```bash
curl https://rclone.org/install.sh | sudo bash
```

---

## 2. モデルのダウンロード（初回のみ）

次に、`download_qwen_model.py`スクリプトを使用してQwen2.5-Coder-14B-Instructモデルをダウンロードします。(※一度ダウンロードしている場合は再度実行する必要はありません)

```bash
python download_qwen_model.py
```

このスクリプトは、Hugging Face Hubから「Qwen2.5-Coder-14B-Instruct」モデルをダウンロードし、`models`ディレクトリに保存します。

---

## 3. GPU使用状況のモニタリングと自動インスタンス終了の設定

`GPU_useage_notification.py`スクリプトを使用して、GPU使用状況をモニタリングし、Slackに通知します。学習しているターミナルとは別で、もう一つターミナルを開き、以下を順番に実行します。

### GPU_useage_notification.pyの環境変数の設定

スクリプトを使用するには、以下の環境変数を設定する必要があります：

```bash
export SLACK_API_TOKEN="xoxb-your-token"  # SlackのAPIトークン（管理者から提供されるものを使用）
export SLACK_MENTION_USER_ID="U0123456789"  # 通知するユーザーID（自分で取得）
export LAMBDALABS_API_KEY="your-api-key"  # Lambda LabsのAPIキー（管理者から提供されるものを使用）
export LAMBDALABS_INSTANCE_ID="your-instance-id"  # インスタンスID（自分で取得）
```

**注意**: SlackのAPIトークンとLambda LabsのAPIキーは管理者から提供されるものを使用してください。ユーザーIDとインスタンスIDは以下の手順で自分で取得する必要があります。


#### SlackのユーザーIDの取得方法

SlackのユーザーID（`SLACK_MENTION_USER_ID`）は以下の手順で取得できます：

1. 画面右上のプロフィール写真をクリックする
2. 「プロフィール」をクリックする
3. 「その他」をクリックする
4. 「メンバーIDをコピー」の下でメンバーIDを確認する


#### インスタンスIDの取得方法

Lambda LabsのインスタンスID（`LAMBDALABS_INSTANCE_ID`）は以下のコマンドを実行して取得できます。LAMBDALABS_API_KEYには管理者から提供されたapi-keyを使用してください：

```bash
curl -u LAMBDALABS_API_KEY : https://cloud.lambdalabs.com/api/v1/instances
```

このコマンドを実行すると、現在実行中のインスタンスの情報がJSON形式で表示されます。その中から必要なインスタンスIDを確認できます。

また、必要に応じて以下の設定を変更できます：

- `BASE_PATH`: 学習結果が保存されるディレクトリ
- `threshold`: GPUの使用率の閾値（デフォルト: 5.0%）
- `consecutive_low_usage_count`: 連続して低使用率と判断するカウント数（デフォルト: 2回）

---

## 4. rcloneの設定(初回のみ)

学習結果をクラウドストレージに転送するために、rcloneを使用します。以下の手順でrcloneの設定を行います。

### rclone configの設定

以下のコマンドを実行して設定を行います：

```bash
rclone config
```

対話式の設定プロセスが始まります。以下は、Google Driveをリモートストレージとして設定する例です：

1. `n`を入力して新しいリモートを作成
2. リモート名を入力（例：`nat-cpt-nomu`）
3. ストレージタイプとして`20`（Google Drive）を選択
4. client_idとclient_secretは空のままEnterを押す（デフォルト値を使用）
5. スコープとして`3`（drive.file - rcloneで作成したファイルのみアクセス可能）を選択
6. service_account_fileは空のままEnterを押す
7. 詳細設定は`n`を選択してスキップ
8. ブラウザ認証について、リモートマシンの場合は`n`を選択
9. 以下のようなコマンドが得られるので、別のマシン(ローカルなど)でコマンドを実行して得られたトークンを入力する：
   ```
   rclone authorize "drive" "eyJzY29wZSI6ImRyZlLmZpbGUifQ"
   ```
10. Shared Drive（Team Drive）の設定は`n`を選択
11. 設定を確認して`y`を入力して保存

設定が完了すると、`rclone config`コマンドで設定済みのリモートが表示されます。この設定は`~/.config/rclone/`ディレクトリに保存されるので2回目以降は設定の必要がないです。

`GPU_useage_notification.py`スクリプト内でrcloneが使われ、指定されたgoogle driveに学習済みのモデルが転送されます。

---

## 5. データセットの準備

学習に使用するデータセットを`dataset`フォルダに配置します。('dataset'フォルダがない場合は作成してください)データセットの形式はSWIFTフレームワークの要件に従ってください。

---

## 6. 学習の実行

`Qwen2.5-coder_train.sh`スクリプトを使用して学習を実行します。必要に応じて、このスクリプト内の設定（モデルパス、データセットパス、GPU数など）を調整してください。

```bash
./Qwen2.5-coder_train.sh
```

このスクリプトは、SWIFTフレームワークを使用してQwen2.5-Coder-14B-Instructモデルの学習を行います。学習結果は`output`ディレクトリに保存されます。

詳細な情報については、以下のリンクを参照してください：
- [事前学習と微調整の基本情報](/docs/source_en/Instruction/Pre-training-and-Fine-tuning.md)
- [コマンドラインパラメータの詳細説明](/docs/source_en/Instruction/Command-line-parameters.md)

---

## 7. GPU_useage_notification.pyの実行

学習が始まったら、GPUのモニタリング・インスタンス終了・学習結果の転送のスクリプトを開始します。

```bash
# デフォルト設定で実行
python GPU_useage_notification.py

# 自動終了を無効にして実行
python GPU_useage_notification.py --auto-terminate=False

# すべてのパラメータをカスタマイズして実行
python GPU_useage_notification.py --auto-terminate=True --threshold=5.0 --consecutive-count=2 --interval=15 --notification-interval=4
```

各パラメータの説明：
- `--auto-terminate=True`: インスタンスの自動終了の可否（デフォルト: True）
- `--threshold=5.0`: GPU使用率の閾値（%）この値未満の使用率が続くと終了判定されます。（デフォルト: 5.0%）
- `--consecutive-count=2`: 終了判定に必要な連続低使用率回数。この回数だけ連続してGPU使用率が閾値未満になると、学習が終了したと判断されます。（デフォルト: 2回）
- `--interval=15`: 監視間隔（分）。この間隔でGPU使用率をチェックします。（デフォルト: 15分）
- `--notification-interval=4`: Slack通知間隔（回数ベース）。この回数ごとにSlackに通知が送信されます。（デフォルト: 4回）

このスクリプトは以下の機能を提供します：

1. 定期的にGPUの使用率を監視し、Slackに通知
2. 学習の進捗状況（最新のログエントリ）をSlackに通知
3. GPUの使用率が一定時間低い状態が続いた場合（学習が終了したと判断される場合）：
   - 学習結果をrcloneを使用してGoogle Driveに転送
