"""
错字生成器 - 模拟真人打字错误
AstrBot风格的自然错字生成
"""
import random
import re
from typing import Dict, List, Tuple


class TypoGenerator:
    """
    错字生成器

    模拟真人打字时的常见错误：
    1. 同音字替换（的/得/地，在/再）
    2. 相近字形替换
    3. 相近拼音替换
    """

    def __init__(
        self,
        typo_error_rate: float = 0.04,
        homophones: Dict[str, List[str]] = None,
        min_chinese_chars: int = 3,
        min_message_length: int = 10,
        max_typos: int = 2,
    ):
        """
        初始化

        Args:
            typo_error_rate: 错字概率 (0-1)
            homophones: 同音字映射
            min_chinese_chars: 最少中文字符数
            min_message_length: 最少消息长度
            max_typos: 最大错字数量
        """
        self.typo_error_rate = typo_error_rate
        self.homophones = homophones or self._default_homophones()
        self.min_chinese_chars = min_chinese_chars
        self.min_message_length = min_message_length
        self.max_typos = max_typos

    def _default_homophones(self) -> Dict[str, List[str]]:
        """默认同音字映射"""
        return {
            "的": ["得", "地"],
            "得": ["的", "地"],
            "地": ["的", "得"],
            "在": ["再"],
            "再": ["在"],
            "做": ["作"],
            "作": ["做"],
            "已": ["以"],
            "以": ["已"],
            "其": ["起"],
            "起": ["其"],
            "会": ["回"],
            "回": ["会"],
            "像": ["象"],
            "象": ["像"],
            "那": ["哪"],
            "哪": ["那"],
            "它": ["他", "她"],
            "他": ["它", "她"],
            "她": ["他", "它"],
            "您": ["你"],
            "吗": ["嘛"],
            "嘛": ["吗"],
            "呢": ["呐"],
            "就": ["旧"],
            "道": ["到"],
            "到": ["道"],
            "知": ["只"],
            "只": ["知"],
            "说": ["水"],
            "听": ["挺"],
            "挺": ["听"],
            "看": ["坎"],
            "想": ["像"],
            "好": ["号"],
            "号": ["好"],
            "了": ["啦"],
            "啦": ["了"],
            "着": ["了"],
            "啊": ["呀", "哦"],
            "呀": ["啊", "呀"],
            "哦": ["啊", "噢"],
            "噢": ["哦", "哦"],
            "都": ["多"],
            "多": ["都"],
            "很": ["狠"],
            "狠": ["很"],
            "因": ["应"],
            "应": ["因"],
            "要": ["好"],
            "没": ["不"],
            "不": ["没"],
            "有": ["又", "友"],
            "又": ["有"],
            "没": ["每"],
            "每": ["没"],
        }

    def apply_typo(self, text: str) -> str:
        """
        对文本应用错字

        Args:
            text: 原始文本

        Returns:
            str: 可能包含错字的文本
        """
        # 检查是否应该生成错字
        if not self._should_generate_typo(text):
            return text

        # 统计中文字符
        chinese_chars = re.findall(r'[一-鿿]', text)
        if len(chinese_chars) < self.min_chinese_chars:
            return text

        # 找出可以替换的位置
        candidates = self._find_typo_candidates(text)
        if not candidates:
            return text

        # 随机选择替换数量
        num_typos = random.randint(1, min(self.max_typos, len(candidates)))
        selected = random.sample(candidates, num_typos)

        # 执行替换
        result = text
        for position, original, replacements in selected:
            replacement = random.choice(replacements)
            result = result[:position] + replacement + result[position + len(original):]

        return result

    def _should_generate_typo(self, text: str) -> bool:
        """判断是否应该生成错字"""
        # 长度检查
        if len(text) < self.min_message_length:
            return False

        # 概率检查
        return random.random() < self.typo_error_rate

    def _find_typo_candidates(self, text: str) -> List[Tuple[int, str, List[str]]]:
        """找出所有可以替换的位置"""
        candidates = []

        for original, replacements in self.homophones.items():
            # 查找所有出现位置
            start = 0
            while True:
                pos = text.find(original, start)
                if pos == -1:
                    break

                # 检查是否在词语中间（避免破坏词汇）
                # 例如"好的"中的"的"不应该替换
                before = text[pos - 1] if pos > 0 else ' '
                after = text[pos + len(original)] if pos + len(original) < len(text) else ' '

                # 如果前后都是中文字符或标点，可能需要更谨慎
                # 这里简化为只在独立位置替换
                is_word_boundary = (
                    not self._is_chinese(before) and
                    not self._is_chinese(after)
                ) or (
                    before in ' \t\n' and
                    after in ' \t\n。！？，、；：""''（）'
                )

                if is_word_boundary or (not self._is_chinese(before) and not self._is_chinese(after)):
                    candidates.append((pos, original, replacements))

                start = pos + 1

        return candidates

    def _is_chinese(self, char: str) -> bool:
        """判断是否是中文字符"""
        return '一' <= char <= '鿿'

    def batch_apply(self, texts: List[str], apply_probability: float = 1.0) -> List[str]:
        """
        批量应用错字

        Args:
            texts: 文本列表
            apply_probability: 对每个文本应用错字的概率

        Returns:
            List[str]: 处理后的文本列表
        """
        results = []
        for text in texts:
            if random.random() < apply_probability:
                results.append(self.apply_typo(text))
            else:
                results.append(text)
        return results


class TypoPatterns:
    """错字模式"""

    # 常见打字错误模式
    COMMON_PATTERNS = [
        # 手滑连续按
        (r'(.)\1{2,}', r'\1\1'),  # 哈哈哈 → 哈哈
        # 漏字
        (r'(.+?)\1{1,3}', r'\1'),  # 看看看 → 看
    ]

    @classmethod
    def apply_pattern(cls, text: str) -> str:
        """应用错字模式"""
        for pattern, replacement in cls.COMMON_PATTERNS:
            text = re.sub(pattern, replacement, text)
        return text


# 全局实例
_default_generator = None


def get_default_generator() -> TypoGenerator:
    """获取默认错字生成器"""
    global _default_generator
    if _default_generator is None:
        _default_generator = TypoGenerator()
    return _default_generator


def apply_typo(text: str) -> str:
    """快速应用错字"""
    return get_default_generator().apply_typo(text)
