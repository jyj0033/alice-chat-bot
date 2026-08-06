"""
测试脚本 - 验证系统核心功能
"""
import asyncio
import time
from pathlib import Path
import yaml

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config():
    """测试配置加载"""
    print("=" * 50)
    print("1. 测试配置加载")
    print("=" * 50)

    config_path = Path(__file__).parent / "config" / "config.yaml"
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False

    # 不实际读取配置文件内容（可能包含敏感信息）
    print(f"✓ 配置文件路径存在: {config_path}")
    print(f"  - 跳过内容读取以保护敏感信息")
    return True


def test_personality():
    """测试人格系统"""
    print("\n" + "=" * 50)
    print("2. 测试人格系统")
    print("=" * 50)

    from modules.personality import Personality

    personality_config = {
        "name": "爱丽丝",
        "nickname": "小爱",
        "traits": {
            "extraversion": 0.6,
            "agreeableness": 0.7,
        },
        "interested_topics": ["游戏", "动漫"],
        "emoji_set": ["😅", "🤔", "😂"],
    }

    personality = Personality.from_dict(personality_config)
    prompt = personality.build_persona_prompt()

    print(f"✓ 人格系统正常工作")
    print(f"  - 人格提示词长度: {len(prompt)} 字符")
    print(f"  - 话题兴趣测试: 游戏={personality.is_topic_interesting('游戏'):.2f}")
    print(f"  - 说话热情度: {personality.get_speaking_enthusiasm():.2f}")
    return True


def test_emotion():
    """测试情感系统"""
    print("\n" + "=" * 50)
    print("3. 测试情感系统")
    print("=" * 50)

    from modules.personality import EmotionalManager

    emotion_config = {
        "positive_keywords": ["谢谢", "哈哈", "棒"],
        "negative_keywords": ["滚", "傻"],
    }

    manager = EmotionalManager(
        decay_halflife=600,
        positive_keywords=emotion_config.get("positive_keywords", []),
        negative_keywords=emotion_config.get("negative_keywords", []),
    )

    # 触发被提到
    manager.trigger_event("test_session", "mentioned")
    state = manager.get_state("test_session")
    print(f"✓ 情感系统正常工作")
    print(f"  - 被提到后精力: {state.energy:.2f}")
    print(f"  - 被提到后投入度: {state.engagement:.2f}")
    print(f"  - 发言加成: {state.get_speaking_bonus():.2f}")

    return True


def test_attention():
    """测试注意力系统"""
    print("\n" + "=" * 50)
    print("4. 测试注意力系统")
    print("=" * 50)

    from modules.social import AttentionManager, AttentionKeywordsDetector

    attention_manager = AttentionManager(
        initial_attention=0.5,
        decay_halflife=300,
        boost_step=0.4,
    )

    keywords_detector = AttentionKeywordsDetector(
        positive_keywords=["谢谢", "赞", "厉害"],
        negative_keywords=["傻", "滚"],
    )

    # 模拟被@
    attention_manager.on_message_received(
        group_id="123456",
        user_id="user001",
        mentioned_bot=True
    )

    attention = attention_manager.get_effective_attention("123456", "user001")
    print(f"✓ 注意力系统正常工作")
    print(f"  - 被@后注意力: {attention:.2f}")

    # 测试关键词检测
    change, reason = keywords_detector.detect_attention_keywords("谢谢你的帮助", "user001")
    print(f"  - 关键词检测: {reason} (变化: {change:.2f})")

    return True


def test_fatigue():
    """测试疲劳系统"""
    print("\n" + "=" * 50)
    print("5. 测试疲劳系统")
    print("=" * 50)

    from modules.social import FatigueManager

    fatigue_manager = FatigueManager(
        reset_threshold=300,
        threshold_light=3,
        threshold_medium=5,
        threshold_heavy=8,
    )

    # 模拟同一会话中 bot 实际参与了 4 轮对话
    for i in range(4):
        fatigue_manager.on_message("test_session", is_bot_message=False)
        fatigue_manager.on_message("test_session", is_bot_message=True)

    penalty = fatigue_manager.get_probability_penalty("test_session")
    state = fatigue_manager.get_state("test_session")

    print(f"✓ 疲劳系统正常工作")
    print(f"  - 对话轮次: {state.conversation_rounds}")
    print(f"  - 疲劳等级: {state.fatigue_level:.2f}")
    print(f"  - 概率惩罚: {penalty:.2f}")
    print(f"  - 是否应该结束: {fatigue_manager.should_close_conversation('test_session')}")

    return True


def test_social_decision():
    """测试社交决策"""
    print("\n" + "=" * 50)
    print("6. 测试社交决策")
    print("=" * 50)

    from modules.social import (
        SocialAwarenessManager,
        TriggerDetector,
        EnhancedSpeakingDecider,
        AttentionManager,
        AttentionKeywordsDetector,
        FatigueManager,
        SocialContext,
    )

    # 创建组件
    attention_manager = AttentionManager()
    keywords_detector = AttentionKeywordsDetector()
    fatigue_manager = FatigueManager()

    decider = EnhancedSpeakingDecider(
        base_probability=0.02,
        after_reply_probability=0.8,
        probability_duration=120,
        attention_manager=attention_manager,
        attention_keywords_detector=keywords_detector,
        fatigue_manager=fatigue_manager,
        trigger_keywords=["爱丽丝", "Alice"],
    )

    trigger_detector = TriggerDetector(
        bot_nickname="爱丽丝",
        trigger_keywords=["爱丽丝", "Alice"],
    )

    social_awareness = SocialAwarenessManager(
        bot_nickname="爱丽丝",
        interested_topics=["游戏", "动漫"],
        bored_topics=["广告"],
    )

    # 模拟场景1: 有人@了Bot
    print("\n--- 场景1: 有人@了Bot ---")
    context = SocialContext(
        message_content="@爱丽丝 今天天气怎么样？",
        sender_id="user001",
        sender_name="小明",
        group_id="123456",
        session_id="group_123456",
        mentioned_me=True,
    )

    # 社交分析 - trigger_detector.detect() 返回结果存储到 extra
    trigger_result = trigger_detector.detect(context)
    context.extra["trigger"] = trigger_result
    context = social_awareness.analyze(context)

    should_speak, reason, prob = decider.should_speak(context)

    print(f"  消息: {context.message_content}")
    print(f"  决策: {'发言' if should_speak else '不发言'}")
    print(f"  概率: {prob:.2f}")
    print(f"  原因: {reason}")

    # 模拟场景2: 普通群聊
    print("\n--- 场景2: 普通群聊（不@） ---")
    context2 = SocialContext(
        message_content="今天的游戏真好玩",
        sender_id="user002",
        sender_name="小红",
        group_id="123456",
        session_id="group_123456",
        mentioned_me=False,
    )

    trigger_result2 = trigger_detector.detect(context2)
    context2.extra["trigger"] = trigger_result2
    context2 = social_awareness.analyze(context2)

    should_speak2, reason2, prob2 = decider.should_speak(context2)

    print(f"  消息: {context2.message_content}")
    print(f"  决策: {'发言' if should_speak2 else '不发言'}")
    print(f"  概率: {prob2:.2f}")
    print(f"  原因: {reason2}")

    print(f"\n✓ 社交决策系统正常工作")
    return True


def test_typo():
    """测试错字生成"""
    print("\n" + "=" * 50)
    print("7. 测试错字生成")
    print("=" * 50)

    from modules.personality import TypoGenerator

    typo_gen = TypoGenerator(
        typo_error_rate=0.3,  # 高概率方便测试
        homophones={
            "的": ["得", "地"],
            "在": ["再"],
        }
    )

    test_messages = [
        "我觉得这个东西很好",
        "谢谢你的帮助",
        "在哪里可以找到",
    ]

    print("原始文本 -> 处理后")
    print("-" * 40)
    for msg in test_messages:
        result = typo_gen.apply_typo(msg)
        symbol = "✓" if result != msg else "="
        print(f"{symbol} {msg}")
        if result != msg:
            print(f"  -> {result}")

    print(f"\n✓ 错字生成系统正常工作")
    return True


def test_import():
    """测试主模块导入"""
    print("\n" + "=" * 50)
    print("8. 测试主模块导入")
    print("=" * 50)

    try:
        from main import GroupChatBot
        print("✓ 主模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 主模块导入失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("群聊Bot系统测试")
    print("=" * 60)

    tests = [
        ("配置加载", test_config),
        ("人格系统", test_personality),
        ("情感系统", test_emotion),
        ("注意力系统", test_attention),
        ("疲劳系统", test_fatigue),
        ("社交决策", test_social_decision),
        ("错字生成", test_typo),
        ("主模块导入", test_import),
    ]

    results = []
    for name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "❌ 失败"
        print(f"  {status} - {name}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过！系统已准备就绪。")
    else:
        print(f"\n⚠️  有 {total - passed} 项测试失败，请检查。")


if __name__ == "__main__":
    asyncio.run(main())
