import unittest

from bot import WebhookBot


class AIScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_ai_score_range_and_suffix(self):
        bot = WebhookBot(custom_prefix="!")
        bot._save_persistent_data = lambda: None

        out = bot._prepare_scored_text(
            "Certainly, I can help you with that. Please let me know if you need anything else."
        )
        self.assertIn("is this ai:", out)
        self.assertGreaterEqual(bot._style_reward_model["last_score"], 0.0)
        self.assertLessEqual(bot._style_reward_model["last_score"], 10.0)
        self.assertGreater(bot._style_reward_model["generation_count"], 0)
        await bot.close()

    async def test_formal_text_scores_higher_than_casual(self):
        bot = WebhookBot(custom_prefix="!")
        formal = "Certainly. I can assist you with that request. Please let me know if you need anything else."
        casual = "yeah i got you, send it over"
        formal_score = bot._score_ai_text(formal)
        casual_score = bot._score_ai_text(casual)
        self.assertGreater(formal_score, casual_score)
        await bot.close()


if __name__ == "__main__":
    unittest.main()
