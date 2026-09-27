"""本地模型注册表：**离线模型生命周期管理**。

- 扫描本地目录里的 GGUF（及其他本地权重）文件，识别**量化等级**；
- 按「模型家族」聚合（同一模型的不同量化变体），为**自动降级**提供依据；
- 估算每个变体的常驻内存，供内存预算判断是否放得下；
- 完整性用 sha256 校验（按需计算并缓存），服务期**零出网**。
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field

# 量化等级 → 质量分（越高越保真），用于「预算内选最好的」
QUALITY = {
    "F32": 100, "F16": 95, "BF16": 94,
    "Q8_0": 80, "Q6_K": 70,
    "Q5_K_M": 62, "Q5_K_S": 58, "Q5_0": 55, "Q5_1": 57,
    "Q4_K_M": 50, "Q4_K_S": 46, "Q4_0": 42, "Q4_1": 44,
    "Q3_K_L": 34, "Q3_K_M": 30, "Q3_K_S": 26,
    "Q2_K": 20, "IQ4_XS": 40, "IQ3_M": 28,
}
QUANT_RE = re.compile(
    r"(?i)\b(F32|F16|BF16|Q8_0|Q6_K|Q5_K_M|Q5_K_S|Q5_0|Q5_1|"
    r"Q4_K_M|Q4_K_S|Q4_0|Q4_1|Q3_K_L|Q3_K_M|Q3_K_S|Q2_K|IQ4_XS|IQ3_M)\b"
)
MODEL_EXT = {".gguf", ".bin", ".ggml"}


def detect_quant(name: str) -> str:
    m = QUANT_RE.search(name)
    return m.group(1).upper().replace("Q", "Q").replace("_K_M", "_K_M") if m else "UNKNOWN"


def family_of(filename: str, quant: str) -> str:
    """去掉扩展名与量化标记，得到模型家族名。"""
    base = os.path.splitext(os.path.basename(filename))[0]
    if quant != "UNKNOWN":
        base = re.sub(r"(?i)[._\-]*" + re.escape(quant) + r"\b", "", base)
    return base.strip(" ._-") or base


def quality_of(quant: str) -> int:
    return QUALITY.get(quant.upper(), 45 if quant == "UNKNOWN" else 10)


@dataclass
class ModelMeta:
    alias: str                 # 唯一别名（文件名）
    path: str
    family: str
    quant: str
    size_mb: float
    est_ram_mb: float = 0.0
    sha256: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "alias": self.alias, "family": self.family, "quant": self.quant,
            "size_mb": round(self.size_mb, 1), "est_ram_mb": round(self.est_ram_mb, 1),
            "sha256": self.sha256[:16], "path": self.path,
        }


def estimate_ram_mb(size_mb: float, context: int = 2048) -> float:
    """粗略估算常驻内存：权重 + 加载开销 + KV cache。

    对 GGUF 而言权重基本等于文件大小，另加运行时/上下文开销。
    """
    return round(size_mb * 1.15 + 64 + context * 0.0015, 1)


class Registry:
    def __init__(self, model_dir: str = "./models", context: int = 2048):
        self.model_dir = model_dir
        self.context = context
        self._sha_cache: dict = {}

    def scan(self) -> list:
        """扫描本地模型目录，返回 ModelMeta 列表。"""
        out = []
        if not os.path.isdir(self.model_dir):
            return out
        for name in sorted(os.listdir(self.model_dir)):
            ext = os.path.splitext(name)[1].lower()
            if ext not in MODEL_EXT:
                continue
            p = os.path.join(self.model_dir, name)
            if not os.path.isfile(p):
                continue
            size_mb = os.path.getsize(p) / (1024 * 1024)
            quant = detect_quant(name)
            out.append(ModelMeta(
                alias=name, path=os.path.abspath(p), family=family_of(name, quant),
                quant=quant, size_mb=round(size_mb, 1),
                est_ram_mb=estimate_ram_mb(size_mb, self.context),
            ))
        return out

    def families(self) -> dict:
        """家族 → 该家族的量化变体（按质量降序）。"""
        fams: dict = {}
        for m in self.scan():
            fams.setdefault(m.family, []).append(m)
        for k in fams:
            fams[k].sort(key=lambda x: quality_of(x.quant), reverse=True)
        return fams

    def get(self, alias: str):
        for m in self.scan():
            if m.alias == alias or m.family == alias:
                return m
        return None

    def sha256(self, meta: ModelMeta) -> str:
        """按需计算并缓存 sha256（大文件只在显式校验时算一次）。"""
        if meta.alias in self._sha_cache:
            return self._sha_cache[meta.alias]
        h = hashlib.sha256()
        with open(meta.path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        self._sha_cache[meta.alias] = h.hexdigest()
        return h.hexdigest()
