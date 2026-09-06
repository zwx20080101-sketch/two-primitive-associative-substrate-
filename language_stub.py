"""外部语言规则模块 stub（占位实现，真实语言模块未来替换）。

宪章定位：本文件属于"外部模块"，不是基底。
职责：读取基底提供的 规则标记绑定 / 特征类检索 / 重复标记，
决定某个词能否做叠词外推，并渲染输出字符串。
它只读基底数据 + 通过公开 learn 接口把"输出被听到"作为新经验喂回基底，
绝不修改基底内部结构。
"""

from __future__ import annotations


def _neighbors(net, node: str) -> set[str]:
    return {
        other
        for a, b in net.connections
        for other in ((a, b) if node == b else (b, a))
        if node in (a, b)
    }


class ReduplicationStub:
    """占位语言规则：读基底证据，决定能否把 word 叠成 word+word。"""

    def __init__(self, rule_marker: str = "R", feature: str = "F"):
        self.rule_marker = rule_marker
        self.feature = feature

    def taught_words(self, l0) -> set[str]:
        """规则标记 R 绑定的词 = 被教过的叠词例子。"""
        return {w for w in _neighbors(l0, self.rule_marker)}

    def eligible_words(self, l0) -> set[str]:
        """特征类 F 下的候选（含未叠过的词）。"""
        return {w for w in _neighbors(l0, self.feature)}

    def can_reduplicate(self, l0, l1, word: str) -> bool:
        """判定: 规则已从例子习得 + 词有资格 + 还没被教过叠词。"""
        rule_known = len(self.taught_words(l0)) > 0
        eligible = word in self.eligible_words(l0)
        already = l1.count(word, word) > 0
        return rule_known and eligible and not already

    def render(self, word: str) -> str:
        """渲染输出(占位): 把词写两遍。"""
        return word + word
