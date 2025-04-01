import subprocess
import json
import time
import os
import statistics
import argparse
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
# 現在の作業ディレクトリを動的に取得し、パスを構築
current_dir = os.path.dirname(os.path.abspath(__file__))  # スクリプトのディレクトリを取得
BASE_PATH = os.path.join(current_dir, "output/Qwen2.5-Coder-14B-Instruct")

def get_ip_address():
    """
    リモートインスタンスのIPアドレスを取得する
    SSHで接続している場合は接続先のIPアドレスを返す
    """
    try:
        # ホスト名を取得
        hostname = subprocess.check_output(["hostname"]).decode("utf-8").strip()
        
        # ホスト名がIPアドレス形式の場合（例：ubuntu@152-69-170-152）
        import re
        ip_pattern = re.compile(r'(\d+)[.-](\d+)[.-](\d+)[.-](\d+)')
        match = ip_pattern.search(hostname)
        
        if match:
            # ハイフンやドットで区切られたIPアドレスをドット区切りの形式に変換
            ip_parts = match.groups()
            return f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.{ip_parts[3]}"
        
        # 環境変数からSSH_CONNECTION情報を取得
        ssh_connection = os.environ.get("SSH_CONNECTION")
        if ssh_connection:
            # SSH_CONNECTIONは "クライアントIP クライアントポート サーバIP サーバポート" の形式
            parts = ssh_connection.split()
            if len(parts) >= 3:
                return parts[2]  # サーバIP（接続先）を返す
        
        # 上記の方法で取得できない場合は従来の方法を使用
        ip_output = subprocess.check_output(
            ["hostname", "-I"]
        ).decode("utf-8")
        return ip_output.strip().split()[0]
    except Exception as e:
        print("Error getting IP address:", e)
        return "Unknown IP"

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
        ip_address = get_ip_address()
        message = f"-------IP Address-------------------\n{ip_address}\n-------GPU使用率-------------------\n{gpu_usage_str}"
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

def get_available_rclone_remotes():
    """
    rcloneの設定から利用可能なリモートのリストを取得する。
    最初に見つかったリモートを返す。リモートが見つからない場合はNoneを返す。
    """
    try:
        # rclone config showコマンドを実行して設定を取得
        output = subprocess.check_output(["rclone", "config", "show"]).decode("utf-8")
        
        # 出力からリモート名を抽出
        remotes = []
        for line in output.split("\n"):
            if line.strip() and line.strip()[0] == "[" and line.strip()[-1] == "]":
                # [remote_name] の形式からremote_nameを抽出
                remote_name = line.strip()[1:-1]
                remotes.append(remote_name)
        
        if remotes:
            print(f"利用可能なrcloneリモート: {', '.join(remotes)}")
            return remotes[0]  # 最初に見つかったリモートを返す
        else:
            print("rcloneリモートが見つかりません。")
            return None
    except subprocess.CalledProcessError as e:
        print(f"rclone config showコマンドの実行中にエラーが発生しました: {e}")
        return None
    except Exception as e:
        print(f"rcloneリモートの取得中にエラーが発生しました: {e}")
        return None

def scp_latest_training_results():
    """
    BASE_PATH直下の最新の学習結果ディレクトリを rclone を用いて転送する処理です。
    フォルダ構造を保持するため、転送先に最新ディレクトリ名を付与しています。
    転送先は環境変数RCLONE_HOSTで指定されたリモート、または自動検出されたrcloneリモートとなります。
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
    
    # ホスト名を環境変数から取得するか、rclone設定から自動検出
    host_name = os.environ.get("RCLONE_HOST")
    if not host_name:
        host_name = get_available_rclone_remotes()
        if not host_name:
            error_message = "rcloneリモートが見つからないため、転送をスキップします。"
            print(error_message)
            try:
                client.chat_postMessage(channel=SLACK_CHANNEL, text=error_message)
            except SlackApiError as e:
                print("Error sending error notification:", e)
            return
    
    dest_path = f"{host_name}:rclone/{latest_dir}"  # 転送先に最新ディレクトリ名を付与

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


def str2bool(v):
    """文字列をブール値に変換する関数"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    """コマンドライン引数を解析する関数"""
    parser = argparse.ArgumentParser(description='GPU使用率監視とインスタンス終了制御')
    parser.add_argument('--auto-terminate', type=str2bool, default=True, 
                      help='GPUの使用率が低下した際に自動的にインスタンスを終了するかどうか (True/False, yes/no, t/f, y/n, 1/0)')
    parser.add_argument('--threshold', type=float, default=5.0,
                      help='GPU使用率の閾値 (%)、この値未満が続くと終了判定')
    parser.add_argument('--consecutive-count', type=int, default=2,
                      help='終了判定に必要な連続低使用率回数')
    parser.add_argument('--interval', type=int, default=15,
                      help='監視間隔 (分)')
    parser.add_argument('--notification-interval', type=int, default=4,
                      help='Slack通知間隔 (回数ベース)')
    return parser.parse_args()

if __name__ == '__main__':
    # コマンドライン引数の解析
    args = parse_arguments()
    
    interval_minutes = args.interval
    consecutive_low_usage_count = 0
    slack_notification_interval = args.notification_interval
    check_count = 0
    threshold = args.threshold
    consecutive_threshold = args.consecutive_count
    auto_terminate = args.auto_terminate

    print(f"GPU監視を開始します: 閾値={threshold}%, 連続回数={consecutive_threshold}, 自動終了={auto_terminate}")
    
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

        if consecutive_low_usage_count >= consecutive_threshold:
            # 学習終了と判断：まずrcloneによる結果転送を実施
            scp_latest_training_results()
            
            # 自動終了設定に基づいて処理
            if auto_terminate:
                # 自動的にインスタンスを終了
                terminate_instance()
                break
            else:
                # インスタンスを終了せず、監視を継続
                message = f"GPUの使用率が{threshold}%未満の状態が{consecutive_threshold}回連続で続いていますが、auto-terminate=Falseのため終了しません。"
                print(message)
                try:
                    client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
                except SlackApiError as e:
                    print("Error sending message:", e)
                # カウントをリセットして監視継続
                consecutive_low_usage_count = 0

        time.sleep(interval_minutes * 60)
