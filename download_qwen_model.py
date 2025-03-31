#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from huggingface_hub import snapshot_download
import logging

def setup_logging():
    """ロギングの設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """
    Qwen2.5-coder-14B-Instructモデルをダウンロードする関数
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    # モデル情報
    model_name = "Qwen2.5-Coder-14B-Instruct"
    cache_dir = "models"

    # 保存ディレクトリの作成
    os.makedirs(cache_dir, exist_ok=True)
    save_path = os.path.join(cache_dir, model_name)

    logger.info(f"モデル {model_name} のダウンロードを開始します...")
    
    try:
        snapshot_download(
            repo_id=f"Qwen/{model_name}",
            cache_dir=cache_dir,
            local_dir=save_path
        )
        logger.info(f"モデル {model_name} のダウンロードが完了しました")
        logger.info(f"保存先: {save_path}")
    
    except Exception as e:
        logger.error(f"モデルのダウンロード中にエラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main()
