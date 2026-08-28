import unittest

from a2rl_drone_training.console import format_table


class ConsoleTableTests(unittest.TestCase):
    def test_table_groups_and_aligns_metrics(self):
        output = format_table(
            [
                ("time", [("fps", "1,437"), ("iterations", "3 / 10")]),
                ("rollout", [("ep_rew_mean", "2.500")]),
            ]
        )

        self.assertIn("| time/", output)
        self.assertIn("|     fps", output)
        self.assertIn("1,437", output)
        self.assertIn("| rollout/", output)
        line_lengths = {len(line) for line in output.splitlines()}
        self.assertEqual(len(line_lengths), 1)

    def test_long_values_are_truncated_without_breaking_alignment(self):
        output = format_table(
            [("checkpoint", [("path", "a" * 100)])],
            max_value_width=12,
        )

        self.assertIn("aaaaaaaaa...", output)
        self.assertNotIn("a" * 13, output)


if __name__ == "__main__":
    unittest.main()
