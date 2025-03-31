# Qwen2.5モデルの学習ガイド

このドキュメントでは、Qwen2.5モデルのダウンロードから学習、モニタリングまでの一連の流れを説明します。

## モデルのダウンロード

まず、`download_qwen_model.py`スクリプトを使用してQwen2.5-Coder-14B-Instructモデルをダウンロードします。

```bash
python download_qwen_model.py
```

このスクリプトは、Hugging Face Hubから「Qwen2.5-Coder-14B-Instruct」モデルをダウンロードし、`ms-swift/models`ディレクトリに保存します。

## 必要なライブラリのインストール

次に、必要なライブラリをインストールします。

```bash
pip install -e . && pip install tf-keras deepspeed slack-sdk && pip install flash-attn --no-build-isolation
```

このコマンドにより、以下のライブラリがインストールされます：
- SWIFTフレームワーク（ローカルインストール）
- TensorFlow Keras
- DeepSpeed（分散学習フレームワーク）
- Slack SDK（通知用）
- Flash Attention（高速なAttention実装）

## データセットの準備

学習に使用するデータセットを`dataset`フォルダに配置します。データセットの形式はSWIFTフレームワークの要件に従ってください。

## 学習の実行

`Qwen2.5-coder_train.sh`スクリプトを使用して学習を実行します。必要に応じて、このスクリプト内の設定（モデルパス、データセットパス、GPU数など）を調整してください。

```bash
./Qwen2.5-coder_train.sh
```

このスクリプトは、SWIFTフレームワークを使用してQwen2.5-Coder-14B-Instructモデルの学習を行います。学習結果は`output`ディレクトリに保存されます。

## GPU使用状況のモニタリングと自動インスタンス終了

`GPU_useage_notification.py`スクリプトを使用して、GPU使用状況をモニタリングし、Slackに通知します。

```bash
python GPU_useage_notification.py
```

このスクリプトは以下の機能を提供します：

1. 定期的にGPUの使用率を監視し、Slackに通知
2. 学習の進捗状況（最新のログエントリ）をSlackに通知
3. GPUの使用率が一定時間低い状態が続いた場合（学習が終了したと判断される場合）：
   - 学習結果をrcloneを使用して転送
   - Lambda Labsのインスタンスを自動的に終了

### GPU_useage_notification.pyの設定

このスクリプトを使用するには、以下の環境変数を設定する必要があります：

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
