import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord

from bot import WebhookBot


class DummyResponse:
    def __init__(self):
        self.calls = []

    async def send_message(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def is_done(self):
        return False


class DummyInteraction:
    def __init__(self, guild_id: int):
        self.guild = SimpleNamespace(id=guild_id)
        self.response = DummyResponse()


class DummyTextChannel(discord.TextChannel):
    __slots__ = ("sent_calls", "id")

    async def send(self, *args, **kwargs):
        if not hasattr(self, "sent_calls"):
            self.sent_calls = []
        self.sent_calls.append((args, kwargs))


class BotLogicTests(unittest.IsolatedAsyncioTestCase):
    async def test_prefix_dispatch_passes_used_prefix(self):
        bot = WebhookBot(custom_prefix="!,??")
        bot._handle_lq_command = AsyncMock()

        message = SimpleNamespace(content="??lq @user 10m")
        await bot._handle_prefix_command(message)

        bot._handle_lq_command.assert_awaited_once()
        _, kwargs = bot._handle_lq_command.await_args
        self.assertEqual(kwargs["used_prefix"], "??")
        await bot.close()

    async def test_lq_usage_uses_active_prefix(self):
        bot = WebhookBot(custom_prefix="!,??")
        message = SimpleNamespace(
            guild=SimpleNamespace(id=123),
            author=SimpleNamespace(guild_permissions=SimpleNamespace(manage_roles=True)),
            content="??lq",
            mentions=[],
            reply=AsyncMock(),
        )

        await bot._handle_lq_command(message, used_prefix="??")
        message.reply.assert_awaited_once()
        usage_text = message.reply.await_args.args[0]
        self.assertIn("??lq @user [duration]", usage_text)
        await bot.close()

    async def test_ai_score_range(self):
        bot = WebhookBot(custom_prefix="!")
        score = bot._score_ai_text("yo that's fine lol")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)
        await bot.close()

    async def test_server_settings_set_uses_self_methods(self):
        bot = WebhookBot(custom_prefix="!")
        await bot.setup_hook()

        bot.set_guild_setting = MagicMock()

        interaction = DummyInteraction(guild_id=42)
        command = bot.tree.get_command("server_settings")
        self.assertIsNotNone(command)

        await command.callback(
            interaction,
            action="set",
            setting_name="feature_enabled",
            value="true",
        )

        bot.set_guild_setting.assert_called_once_with(42, "feature_enabled", True)
        self.assertTrue(interaction.response.calls)
        await bot.close()

    async def test_apply_defaults_uses_self_methods(self):
        bot = WebhookBot(custom_prefix="!")
        await bot.setup_hook()

        bot.get_all_guild_settings = MagicMock(return_value={})
        bot.apply_default_settings = MagicMock()
        bot.get_default_guild_settings = MagicMock(return_value={"welcome_message": None})

        interaction = DummyInteraction(guild_id=7)
        command = bot.tree.get_command("apply_defaults")
        self.assertIsNotNone(command)

        await command.callback(interaction)

        bot.apply_default_settings.assert_called_once_with(7)
        self.assertTrue(interaction.response.calls)
        await bot.close()

    async def test_humanize_parse_helpers(self):
        bot = WebhookBot(custom_prefix="!")
        self.assertEqual(bot._parse_humanize_choice("2"), 2)
        self.assertIsNone(bot._parse_humanize_choice("4"))
        self.assertEqual(bot._parse_humanize_rating("7"), 7)
        self.assertEqual(bot._parse_humanize_rating("7/10"), 7)
        self.assertEqual(bot._parse_humanize_rating("10 / 10"), 10)
        self.assertIsNone(bot._parse_humanize_rating("11/10"))
        await bot.close()

    async def test_humanize_format_helpers(self):
        bot = WebhookBot(custom_prefix="!")
        statement = bot._format_humanize_statement("server fried after patch")
        options = bot._format_humanize_options(
            [
                {"text": "option one"},
                {"text": "option two"},
                {"text": "option three"},
            ]
        )
        self.assertEqual(statement, "user: server fried after patch")
        self.assertIn("1. option one", options)
        self.assertIn("2. option two", options)
        self.assertIn("3. option three", options)
        self.assertIn("which did you like best?", options)
        await bot.close()

    async def test_humanize_session_flow_select_then_rate(self):
        bot = WebhookBot(custom_prefix="!")
        bot._learn_from_human_rating = MagicMock(return_value=2.4)

        key = bot._humanize_session_key(1, 2, 3)
        bot._humanize_sessions[key] = {
            "created_ts": 9999999999.0,
            "stage": "select",
            "statement": "why this broken",
            "options": [
                {"text": "opt1", "strategy": "identity", "ai_score": 3.0},
                {"text": "opt2", "strategy": "casualize", "ai_score": 2.5},
                {"text": "opt3", "strategy": "shorten", "ai_score": 2.2},
            ],
            "selected": None,
        }

        select_message = SimpleNamespace(
            guild=SimpleNamespace(id=1),
            channel=SimpleNamespace(id=2),
            author=SimpleNamespace(id=3),
            content="2",
            reply=AsyncMock(),
        )
        handled = await bot._handle_humanize_session_message(select_message)
        self.assertTrue(handled)
        self.assertEqual(bot._humanize_sessions[key]["stage"], "rate")
        select_message.reply.assert_awaited()

        rate_message = SimpleNamespace(
            guild=SimpleNamespace(id=1),
            channel=SimpleNamespace(id=2),
            author=SimpleNamespace(id=3),
            content="5/10",
            reply=AsyncMock(),
        )
        handled = await bot._handle_humanize_session_message(rate_message)
        self.assertTrue(handled)
        bot._learn_from_human_rating.assert_called_once_with("opt2", "casualize", 5)
        self.assertNotIn(key, bot._humanize_sessions)
        rate_message.reply.assert_awaited()
        await bot.close()

    async def test_prefix_humanize_sends_statement_then_options(self):
        bot = WebhookBot(custom_prefix="!")
        bot._pick_random_server_statement = AsyncMock(return_value="server died again")
        bot._generate_humanize_candidates = MagicMock(
            return_value=[
                {"text": "option a", "strategy": "identity", "ai_score": 2.0},
                {"text": "option b", "strategy": "casualize", "ai_score": 2.1},
                {"text": "option c", "strategy": "shorten", "ai_score": 2.2},
            ]
        )

        channel = object.__new__(DummyTextChannel)
        channel.sent_calls = []
        channel.id = 99

        message = SimpleNamespace(
            guild=SimpleNamespace(id=1),
            channel=channel,
            author=SimpleNamespace(id=2),
            reply=AsyncMock(),
        )

        await bot._handle_humanize_command(message)

        message.reply.assert_awaited_once_with("user: server died again")
        self.assertEqual(len(channel.sent_calls), 1)
        self.assertIn("1. option a", channel.sent_calls[0][0][0])
        self.assertIn("which did you like best?", channel.sent_calls[0][0][0])
        await bot.close()

    async def test_slash_humanize_sends_two_messages(self):
        bot = WebhookBot(custom_prefix="!")
        await bot.setup_hook()
        bot._pick_random_server_statement = AsyncMock(return_value="voice desynced")
        bot._generate_humanize_candidates = MagicMock(
            return_value=[
                {"text": "first", "strategy": "identity", "ai_score": 2.0},
                {"text": "second", "strategy": "casualize", "ai_score": 2.1},
                {"text": "third", "strategy": "shorten", "ai_score": 2.2},
            ]
        )

        class DummyResponse:
            def __init__(self):
                self.calls = []

            async def send_message(self, *args, **kwargs):
                self.calls.append((args, kwargs))

            def is_done(self):
                return False

        class DummyFollowup:
            def __init__(self):
                self.calls = []

            async def send(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        fake_channel = object.__new__(discord.TextChannel)
        fake_channel.id = 555

        interaction = SimpleNamespace(
            guild=SimpleNamespace(id=10),
            channel=fake_channel,
            user=SimpleNamespace(id=20),
            response=DummyResponse(),
            followup=DummyFollowup(),
        )

        command = bot.tree.get_command("humanize")
        self.assertIsNotNone(command)
        await command.callback(interaction)

        self.assertEqual(len(interaction.response.calls), 1)
        self.assertEqual(len(interaction.followup.calls), 1)
        self.assertEqual(interaction.response.calls[0][0][0], "user: voice desynced")
        self.assertIn("1. first", interaction.followup.calls[0][0][0])
        await bot.close()

    async def test_auto_train_learns_when_enabled(self):
        bot = WebhookBot(custom_prefix="!")
        bot._save_persistent_data = MagicMock()
        bot.set_guild_setting(1, "auto_train_enabled", True, persist=False)
        bot.set_guild_setting(1, "auto_train_target_rating", 8, persist=False)
        bot.set_guild_setting(1, "auto_train_save_every", 99, persist=False)
        bot.set_guild_setting(1, "auto_train_strategy", "identity", persist=False)

        before = bot._style_reward_model["strategy_stats"]["identity"]["count"]
        message = SimpleNamespace(
            guild=SimpleNamespace(id=1),
            author=SimpleNamespace(bot=False),
            webhook_id=None,
            content="yo this patch finally fixed the lag",
            mentions=[],
        )
        await bot._maybe_auto_train_from_message(message)

        after = bot._style_reward_model["strategy_stats"]["identity"]["count"]
        self.assertEqual(after, before + 1)
        self.assertEqual(bot._auto_train_updates, 1)
        await bot.close()

    async def test_auto_train_ignores_prefix_commands(self):
        bot = WebhookBot(custom_prefix="!")
        bot._save_persistent_data = MagicMock()
        bot.set_guild_setting(1, "auto_train_enabled", True, persist=False)
        bot.set_guild_setting(1, "auto_train_target_rating", 8, persist=False)
        bot.set_guild_setting(1, "auto_train_save_every", 99, persist=False)
        bot.set_guild_setting(1, "auto_train_strategy", "identity", persist=False)

        before = bot._style_reward_model["strategy_stats"]["identity"]["count"]
        message = SimpleNamespace(
            guild=SimpleNamespace(id=1),
            author=SimpleNamespace(bot=False),
            webhook_id=None,
            content="!help",
            mentions=[],
        )
        await bot._maybe_auto_train_from_message(message)

        after = bot._style_reward_model["strategy_stats"]["identity"]["count"]
        self.assertEqual(after, before)
        self.assertEqual(bot._auto_train_updates, 0)
        await bot.close()

    async def test_prefix_autotrain_updates_settings(self):
        bot = WebhookBot(custom_prefix="!")
        bot._save_persistent_data = MagicMock()
        message = SimpleNamespace(
            guild=SimpleNamespace(id=1, name="Test Guild"),
            author=SimpleNamespace(guild_permissions=SimpleNamespace(manage_guild=True)),
            reply=AsyncMock(),
        )
        await bot._handle_autotrain_command(message, "on 9")

        self.assertTrue(bot.get_guild_setting(1, "auto_train_enabled", False))
        self.assertEqual(bot.get_guild_setting(1, "auto_train_target_rating", 0), 9)
        message.reply.assert_awaited()
        await bot.close()


if __name__ == "__main__":
    unittest.main()
