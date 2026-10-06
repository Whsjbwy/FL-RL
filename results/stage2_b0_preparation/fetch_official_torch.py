"""官方大包分段获取；保留官方安装包安全校验，不生成项目源码摘要。"""

import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


def main() -> None:
    """单文件顺序分段，不重试到有利结果；任何区间/校验错误立即失败。"""
    with urllib.request.urlopen("https://download.pytorch.org/whl/cu130/torch/",
                                timeout=30) as response:
        index = response.read().decode()
    match = [link for link in re.findall(r'href="([^"]+)"', index)
             if "2.11.0%2Bcu130-cp313-cp313-win_amd64.whl#sha256=" in link]
    if len(match) != 1:
        raise RuntimeError("官方索引未唯一登记目标轮子")
    url, fragment = urllib.parse.urldefrag(match[0])
    # 官方R2完整请求受阻；同一官方域名镜像支持HTTP区间并用索引校验认证内容。
    url = url.replace("https://download-r2.pytorch.org/", "https://download.pytorch.org/")
    expected = urllib.parse.parse_qs(fragment)["sha256"][0]
    destination = Path(__file__).parent / "development/install_tmp"
    destination.mkdir(parents=True, exist_ok=True)
    wheel = destination / "torch-2.11.0+cu130-cp313-cp313-win_amd64.whl"
    partial = wheel.with_suffix(".official.part")
    total = 1_915_220_980
    checksum = hashlib.sha256()
    with partial.open("xb") as stream:
        for start in range(0, total, 32 * 1024**2):
            end = min(start + 32 * 1024**2, total) - 1
            request = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start}-{end}"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                if (response.status != 206
                        or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}"):
                    raise RuntimeError("官方包下载区间不匹配，停止")
                count = 0
                while chunk := response.read(1024**2):
                    count += len(chunk)
                    checksum.update(chunk)
                    stream.write(chunk)
                if count != end - start + 1:
                    raise RuntimeError("官方包下载长度不匹配，停止")
            print(f"PACKAGE DOWNLOAD {end + 1}/{total} bytes", flush=True)
    if checksum.hexdigest() != expected:
        raise RuntimeError("官方安装包安全校验失败，不安装")
    partial.replace(wheel)
    print(json.dumps(dict(official_url=url, security_checksum_verified=True,
                          bytes=total, wheel=str(wheel)), ensure_ascii=False))


if __name__ == "__main__":
    main()
