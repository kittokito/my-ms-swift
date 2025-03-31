import subprocess
import json
import time
import os
import statistics
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

###################### INIT ######################
SLACK_TOKEN = os.environ.get("SLACK_API_TOKEN")
SLACK_CHANNEL = "C08ESCTLTGR"
SLACK_MENTION_USER_ID = os.environ.get("SLACK_MENTION_USER_ID", "U0748M9TSUV")
LAMBDA_API_KEY = os.environ.get("LAMBDALABS_API_KEY")
LAMBDA_INSTANCE_ID = os.environ.get("LAMBDALABS_INSTANCE_ID")
client = WebClient(token=SLACK_TOKEN)

# ベースパス（学習結果が保存されるディレクトリ）
BASE_PATH = "/home/ubuntu/nat-cpt-syd/ms-swift/output/Qwen2.5-Coder-14B-Instruct"

def check_gpu_usage(notify_slack=True):
    """
    6秒間、1秒ごとに全GPUの使用率を取得し、
    各GPUの6回分の利用率の平均値を算出する。
    notify_slackがTrueの場合、各GPUの平均使用率をSlackに通知する。
    """
    all_gpu_usages = []  # GPU毎の使用率リスト
    num_samples = 6

    for i in range(num_samples):
        # ヘッダー・単位なしで全GPUの利用率を取得
        gpu_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]
        ).decode("utf-8")
        lines = gpu_output.strip().split("\n")
        
        # 初回サンプリング時にGPU数に合わせてリスト初期化
        if i == 0:
            num_gpus = len(lines)
            all_gpu_usages = [[] for _ in range(num_gpus)]
        
        for j, line in enumerate(lines):
            try:
                usage = float(line.strip())
            except ValueError:
                usage = 0.0
            all_gpu_usages[j].append(usage)
        time.sleep(1)

    # 各GPUについて、6回分の平均値を計算
    avg_gpu_usages = [sum(usages) / len(usages) for usages in all_gpu_usages]

    if notify_slack:
        gpu_usage_str = ", ".join([f"{idx+1}: {usage:.1f}%" for idx, usage in enumerate(avg_gpu_usages)])
        message = f"-------Instance ID-------------------\n{LAMBDA_INSTANCE_ID}\n-------GPU使用率-------------------\n{gpu_usage_str}"
        try:
            client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
            print("GPU usage message sent:", message)
        except SlackApiError as e:
            print("Error sending GPU usage message:", e)

    return avg_gpu_usages

def get_latest_training_dir(base_path=BASE_PATH):
    """
    BASE_PATH直下のサブディレクトリのうち、最新の（最終更新時刻が新しい）ディレクトリ名を返す。
    """
    try:
        subdirs = [
            d for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ]
        if not subdirs:
            return None
        subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(base_path, d)), reverse=True)
        return subdirs[0]
    except Exception as e:
        print("Error in getting latest training dir:", e)
        return None

def get_latest_log_entry():
    """
    最新の学習フォルダ内のlogging.jsonlファイルから、最後の非空行をJSONとして読み込み返す。
    最新のフォルダ名とログエントリを返す。
    """
    latest_dir = get_latest_training_dir(BASE_PATH)
    if latest_dir is None:
        return None, None
    log_file = os.path.join(BASE_PATH, latest_dir, "logging.jsonl")
    if not os.path.exists(log_file):
        return latest_dir, None
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            if line.strip():
                try:
                    log_entry = json.loads(line)
                    return latest_dir, log_entry
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print("Error reading log file:", e)
    return latest_dir, None

def notify_latest_log():
    """
    最新の学習フォルダとlogging.jsonlの最新エントリをSlackに通知する。
    ただし、最新ログがモデルパラメータ情報などの最終まとめログの場合は通知しない。
    """
    latest_dir, log_entry = get_latest_log_entry()
    if latest_dir is None:
        message = "------学習ログ------------------\n学習ディレクトリが見つかりません。"
    elif log_entry is None:
        message = f"------学習ログ------------------\nlogging.jsonlにログエントリが見つかりません。"
    # 最終行がモデルパラメータ情報などの場合は通知しない
    elif "model_parameter_info" in log_entry:
        print("最終ログエントリがモデルパラメータ情報のため、Slack通知をスキップします。")
        return
    else:
        log_str = json.dumps(log_entry, ensure_ascii=False)
        message = f"------学習ログ------------------\n{log_str}"
    try:
        client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
        print("Log notification sent:", message)
    except SlackApiError as e:
        print("Error sending log notification:", e)

def scp_latest_training_results():
    """
    BASE_PATH直下の最新の学習結果ディレクトリを rclone を用いて転送する処理です。
    フォルダ構造を保持するため、転送先に最新ディレクトリ名を付与しています。
    転送先は "nat-cpt-syd:rclone" となります。
    """
    latest_dir = get_latest_training_dir(BASE_PATH)
    if latest_dir is None:
        error_message = "最新の学習結果フォルダが見つかりません。"
        print(error_message)
        try:
            client.chat_postMessage(channel=SLACK_CHANNEL, text=error_message)
        except SlackApiError as e:
            print("Error sending error notification:", e)
        return

    source_dir = os.path.join(BASE_PATH, latest_dir)
    dest_path = f"nat-cpt-syd:rclone/{latest_dir}"  # 転送先に最新ディレクトリ名を付与

    rclone_cmd = [
        "rclone", "copy", "-P",
        source_dir,
        dest_path
    ]
    print("Running rclone command:", " ".join(rclone_cmd))
    try:
        output = subprocess.check_output(rclone_cmd).decode("utf-8")
        print("rclone command output:", output)
        # rcloneが成功した場合にSlackに成功メッセージを送信
        success_message = f"rcloneによる転送が成功しました: {dest_path}"
        client.chat_postMessage(channel=SLACK_CHANNEL, text=success_message)
    except subprocess.CalledProcessError as e:
        error_message = f"rcloneコマンド実行中にエラーが発生しました: {e}"
        print(error_message)
        try:
            client.chat_postMessage(channel=SLACK_CHANNEL, text=error_message)
        except SlackApiError as slack_error:
            print("Error sending Slack error notification:", slack_error)


def terminate_instance():
    """
    Lambda Labsのインスタンスを終了する処理です。
    インスタンス終了に失敗した場合はSlackにエラーメッセージを送信します。
    """
    try:
        message = f"<@{SLACK_MENTION_USER_ID}> Terminating Lambda Labs instance {LAMBDA_INSTANCE_ID}"
        client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
        print("Termination message sent:", message)
        
        _json = {"instance_ids": [f"{LAMBDA_INSTANCE_ID}"]}
        _json_str = json.dumps(_json)
        command = [
            "curl",
            "-u",
            f"{LAMBDA_API_KEY}:",
            "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate",
            "-d",
            _json_str,
            "-H",
            "Content-Type: application/json",
        ]
        output = subprocess.check_output(command).decode("utf-8")
        print("Termination output:", output)
    except subprocess.CalledProcessError as e:
        error_message = f"インスタンス終了に失敗しました: {e}"
        print(error_message)
        try:
            client.chat_postMessage(channel=SLACK_CHANNEL, text=error_message)
        except SlackApiError:
            # Slackへの通知失敗については特にメッセージを出力しない
            pass


if __name__ == '__main__':
    interval_minutes = 1
    consecutive_low_usage_count = 0
    slack_notification_interval = 2  # 通知間隔（回数ベース）
    check_count = 0
    threshold = 5.0  # 5%未満を閾値とする

    while True:
        check_count += 1
        notify = (check_count % slack_notification_interval == 0)
        
        if notify:
            # 通知タイミングの場合、GPU使用率と最新ログの通知を実施
            avg_usages = check_gpu_usage(notify_slack=True)
            notify_latest_log()
        else:
            avg_usages = check_gpu_usage(notify_slack=False)
        
        # 全GPUの平均利用率が全て閾値未満ならカウントアップ
        if all(usage < threshold for usage in avg_usages):
            consecutive_low_usage_count += 1
        else:
            consecutive_low_usage_count = 0

        if consecutive_low_usage_count >= 2:
            # 学習終了と判断：まずrcloneによる結果転送（またはエラー通知）を実施し、
            # その後にインスタンスをterminateする
            scp_latest_training_results()
            terminate_instance()  # 必要に応じてコメント解除
            break

        time.sleep(interval_minutes * 60)
