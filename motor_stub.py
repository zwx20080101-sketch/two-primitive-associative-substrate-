"""外部动作/书写模块 stub（占位实现，真实动作模块未来替换）。

宪章定位：本文件属于"外部模块"，不是基底。
职责：读取基底/组块层提供的块记录（顺序位置表），逐字母渲染输出；
读不到块就报错（无中生有闭包由外部同样遵守）。
"""

from __future__ import annotations


class MotorStub:
    """占位动作/书写模块：把块记录还原成逐字母输出。"""

    def spell_out(self, chunk_layer, chunk_id: str) -> list[str]:
        """读块记录, 返回逐字母序列(含双写位置)。未知块 -> KeyError。"""
        return list(chunk_layer.spell(chunk_id))

    def render(self, letters: list[str]) -> str:
        """渲染成字符串输出(占位: 直接拼接)。"""
        return "".join(letters)
