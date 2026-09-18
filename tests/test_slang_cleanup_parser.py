import json
import unittest

from main import GroupChatBot


class SlangCleanupParserTests(unittest.TestCase):
    def test_accepts_only_delete_ids_from_the_current_batch(self):
        raw = json.dumps(
            {"action": "delete", "delete_ids": [102, 101]},
            separators=(",", ":"),
        )
        self.assertEqual(
            GroupChatBot._parse_slang_cleanup_ids(raw, [101, 102, 103]),
            [101, 102],
        )

    def test_keep_action_never_deletes(self):
        raw = '{"action":"keep","delete_ids":[]}'
        self.assertEqual(GroupChatBot._parse_slang_cleanup_ids(raw, [101]), [])

    def test_rejects_free_form_negation_and_wrapped_json(self):
        invalid_responses = [
            "无",
            "不要删除 101",
            "do not delete 101",
            '删除 101 | 原因',
            '结果：{"action":"delete","delete_ids":[101]}',
            '```json\n{"action":"delete","delete_ids":[101]}\n```',
            '<think>删除 101</think>{"action":"delete","delete_ids":[101]}',
        ]
        for response in invalid_responses:
            with self.subTest(response=response):
                self.assertEqual(
                    GroupChatBot._parse_slang_cleanup_ids(response, [101]), []
                )

    def test_rejects_wrong_action_schema_and_ambiguous_payloads(self):
        invalid_responses = [
            '{"action":"keep","delete_ids":[101]}',
            '{"action":"DELETE","delete_ids":[101]}',
            '{"action":"delete","delete_ids":[101],"reason":"wrong"}',
            '{"action":"delete","delete_ids":"101"}',
            '{"action":"delete","delete_ids":["101"]}',
            '{"action":"delete","delete_ids":[true]}',
            '{"action":"delete","delete_ids":[101,101]}',
            '{"action":"keep","action":"delete","delete_ids":[101]}',
            '[{"action":"delete","delete_ids":[101]}]',
            '{"delete_ids":[101]}',
        ]
        for response in invalid_responses:
            with self.subTest(response=response):
                self.assertEqual(
                    GroupChatBot._parse_slang_cleanup_ids(response, [101]), []
                )

    def test_rejects_entire_result_if_any_id_is_outside_the_candidate_batch(self):
        raw = '{"action":"delete","delete_ids":[101,999]}'
        self.assertEqual(
            GroupChatBot._parse_slang_cleanup_ids(raw, [101, 102]), []
        )


if __name__ == "__main__":
    unittest.main()
