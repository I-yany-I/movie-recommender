"""数据下载模块 — 负责MovieLens和IMDB数据集的下载与解压"""
import gzip
import logging
import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import requests
from tqdm import tqdm

from ..config import Config

logger = logging.getLogger(__name__)


class DataDownloader:
    """数据下载器"""

    def __init__(self):
        self.project_root = Config.get_project_root()
        self.ml_config = Config.get("data.movielens")
        self.imdb_config = Config.get("data.imdb")

    def download_all(self):
        """下载所有数据源"""
        logger.info("=" * 50)
        logger.info("开始下载所有数据源...")
        logger.info("=" * 50)
        self.download_movielens()
        self.download_imdb()
        logger.info("所有数据下载完成！")

    def download_movielens(self):
        """下载 MovieLens 25M 数据集"""
        local_dir = self.project_root / self.ml_config["local_path"]
        local_dir.mkdir(parents=True, exist_ok=True)

        # 检查是否已下载
        ratings_file = local_dir / "ratings.csv"
        if ratings_file.exists():
            logger.info(f"MovieLens 25M 已存在于 {local_dir}")
            return

        url = self.ml_config["url"]
        zip_path = local_dir / "ml-25m.zip"
        logger.info(f"下载 MovieLens 25M 从 {url} ...")
        self._download_with_progress(url, zip_path, "MovieLens 25M")

        logger.info(f"解压 {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(local_dir)

        # 移动解压出的文件到上层
        extracted_dir = local_dir / "ml-25m"
        if extracted_dir.exists():
            for f in extracted_dir.iterdir():
                dest = local_dir / f.name
                if not dest.exists():
                    shutil.move(str(f), str(dest))
            extracted_dir.rmdir()

        zip_path.unlink()  # 删除zip
        logger.info("MovieLens 25M 下载解压完成")

    def download_imdb(self):
        """下载 IMDB 公开数据集"""
        local_dir = self.project_root / self.imdb_config["local_path"]
        local_dir.mkdir(parents=True, exist_ok=True)

        base_url = self.imdb_config["base_url"]
        for filename in self.imdb_config["files"]:
            dest_path = local_dir / filename.replace(".gz", "")
            if dest_path.exists():
                logger.info(f"IMDB {filename} 已存在，跳过")
                continue

            url = base_url + filename
            gz_path = local_dir / filename
            logger.info(f"下载 {filename} 从 {url} ...")
            self._download_with_progress(url, gz_path, filename)

            logger.info(f"解压 {filename} ...")
            with gzip.open(gz_path, "rb") as f_in:
                with open(dest_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            gz_path.unlink()
        logger.info("IMDB 数据集下载完成")

    def _download_with_progress(self, url: str, dest: Path, desc: str):
        """带进度条的下载，优先用 requests，fallback 用 urllib"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(dest, "wb") as f:
                with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        except Exception:
            urlretrieve(url, dest)
