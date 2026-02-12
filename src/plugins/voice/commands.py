"""
Voice commands for Discord bot
"""

import discord
from discord.ext import commands
import logging
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class VoiceManager:
    """Manages voice connections and audio processing"""

    def __init__(self):
        self.voice_connections = {}
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Try to initialize voice components
        try:
            import speech_recognition as sr
            import edge_tts

            self.recognizer = sr.Recognizer()
            self.speech_available = True
            self.tts_available = True
            logger.info("Voice components initialized")
        except ImportError as e:
            self.speech_available = False
            self.tts_available = False
            logger.warning(f"Voice components not available: {e}")

    async def join_voice_channel(self, ctx):
        """Join the user's voice channel"""
        if not ctx.author.voice:
            await ctx.send(
                embed=discord.Embed(
                    title="❌ Not in Voice Channel",
                    description="You need to be in a voice channel first!",
                    color=discord.Color.red(),
                )
            )
            return False

        if ctx.guild.id in self.voice_connections:
            await ctx.send(
                embed=discord.Embed(
                    title="❌ Already Connected",
                    description="I'm already in a voice channel!",
                    color=discord.Color.red(),
                )
            )
            return False

        try:
            channel = ctx.author.voice.channel

            # Try to use voice_recv if available
            try:
                from discord.ext import voice_recv

                voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
                voice_client.listen(voice_recv.BasicSink(self._on_voice_receive))
            except ImportError:
                # Fallback to regular voice connection
                voice_client = await channel.connect()

            self.voice_connections[ctx.guild.id] = voice_client

            await ctx.send(
                embed=discord.Embed(
                    title="✅ Joined Voice Channel",
                    description=f"Joined {channel.name} and ready to chat!",
                    color=discord.Color.green(),
                )
            )

            return True

        except Exception as e:
            logger.error(f"Failed to join voice channel: {e}")
            await ctx.send(
                embed=discord.Embed(
                    title="❌ Voice Error",
                    description="Failed to join voice channel. Check permissions.",
                    color=discord.Color.red(),
                )
            )
            return False

    async def leave_voice_channel(self, ctx):
        """Leave the voice channel"""
        if ctx.guild.id not in self.voice_connections:
            await ctx.send(
                embed=discord.Embed(
                    title="❌ Not in Voice Channel",
                    description="I'm not in any voice channel!",
                    color=discord.Color.red(),
                )
            )
            return False

        try:
            voice_client = self.voice_connections[ctx.guild.id]
            await voice_client.disconnect()
            del self.voice_connections[ctx.guild.id]

            await ctx.send(
                embed=discord.Embed(
                    title="✅ Left Voice Channel",
                    description="Successfully left voice channel!",
                    color=discord.Color.green(),
                )
            )

            return True

        except Exception as e:
            logger.error(f"Failed to leave voice channel: {e}")
            await ctx.send(
                embed=discord.Embed(
                    title="❌ Voice Error",
                    description="Failed to leave voice channel.",
                    color=discord.Color.red(),
                )
            )
            return False

    def _on_voice_receive(self, user, audio):
        """Handle incoming voice audio"""
        if user == self.bot.user:
            return  # Ignore own audio

        # Process in background to avoid blocking
        import asyncio
        audio_bytes = audio.data if hasattr(audio, "data") else audio
        audio_meta = {
            "sample_rate": getattr(audio, "sample_rate", 48000),
            "channels": getattr(audio, "channels", 2),
            "sample_width": getattr(audio, "sample_width", 2),
        }
        asyncio.create_task(self._process_voice_input(user, audio_bytes, audio_meta))

    async def _process_voice_input(self, user, audio_bytes, audio_meta):
        """Process voice input and generate response"""
        try:
            if not self.speech_available:
                return

            # Convert speech to text
            text = await self._speech_to_text(audio_bytes, audio_meta)
            if not text:
                return

            logger.info(f"{user.display_name} said: {text}")

            # Generate response using bot's method
            if hasattr(self.bot, "_generate_response"):
                response = await self.bot._generate_response(
                    text,
                    str(user.voice.channel.id),
                    str(user.id),
                    f"Voice chat with {user.display_name}",
                )

                # Convert response to speech and play
                if self.tts_available and user.guild.id in self.voice_connections:
                    audio_data = await self._text_to_speech(response)
                    if audio_data:
                        await self._play_audio(user.guild.id, audio_data)

        except Exception as e:
            logger.error(f"Voice processing error: {e}")

    async def _speech_to_text(self, audio_bytes, audio_meta):
        """Convert audio to text"""
        try:
            import speech_recognition as sr
            import io
            import wave

            def _to_wav_bytes(pcm_bytes, sample_rate, channels, sample_width):
                with io.BytesIO() as buf:
                    with wave.open(buf, "wb") as wf:
                        wf.setnchannels(channels)
                        wf.setsampwidth(sample_width)
                        wf.setframerate(sample_rate)
                        wf.writeframes(pcm_bytes)
                    return buf.getvalue()

            # If it's not a WAV container, wrap raw PCM into WAV
            if isinstance(audio_bytes, (bytes, bytearray)) and not audio_bytes.startswith(b"RIFF"):
                audio_bytes = _to_wav_bytes(
                    audio_bytes,
                    audio_meta.get("sample_rate", 48000),
                    audio_meta.get("channels", 2),
                    audio_meta.get("sample_width", 2),
                )

            # Create temporary file for speech recognition
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name

            try:
                with sr.AudioFile(temp_file_path) as source:
                    audio = self.recognizer.record(source)
                    text = self.recognizer.recognize_whisper(audio, model="base.en")
                    return text
            finally:
                os.unlink(temp_file_path)

        except Exception as e:
            logger.error(f"Speech to text error: {e}")
            return None

    async def _text_to_speech(self, text):
        """Convert text to speech"""
        try:
            import edge_tts

            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            audio_data = b""

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            return audio_data

        except Exception as e:
            logger.error(f"Text to speech error: {e}")
            return None

    async def _play_audio(self, guild_id, audio_data):
        """Play audio in voice channel"""
        try:
            import io

            voice_client = self.voice_connections[guild_id]

            # Create audio source
            audio_source = discord.FFmpegPCMAudio(
                source=io.BytesIO(audio_data),
                pipe=True,
                options="-f mp3 -ac 1 -ar 44100",
            )

            # Play response if not already playing
            if not voice_client.is_playing():
                voice_client.play(audio_source)

        except Exception as e:
            logger.error(f"Audio playback error: {e}")


# Global voice manager
voice_manager = VoiceManager()


def setup_voice_commands(bot):
    """Setup voice-related commands"""
    voice_manager.bot = bot  # Give voice manager access to bot

    @bot.command(name="join")
    async def join_cmd(ctx):
        """Join the voice channel"""
        await voice_manager.join_voice_channel(ctx)

    @bot.command(name="leave")
    async def leave_cmd(ctx):
        """Leave the voice channel"""
        await voice_manager.leave_voice_channel(ctx)

    @bot.command(name="voice")
    async def voice_cmd(ctx):
        """Show voice status"""
        embed = discord.Embed(title="🎙️ Voice Status", color=discord.Color.blue())

        embed.add_field(
            name="Speech Recognition",
            value="✅ Available"
            if voice_manager.speech_available
            else "❌ Unavailable",
            inline=True,
        )

        embed.add_field(
            name="Text to Speech",
            value="✅ Available" if voice_manager.tts_available else "❌ Unavailable",
            inline=True,
        )

        embed.add_field(
            name="Voice Connections",
            value=str(len(voice_manager.voice_connections)),
            inline=True,
        )

        if not voice_manager.speech_available or not voice_manager.tts_available:
            embed.add_field(
                name="⚠️ Dependencies",
                value="Install optional dependencies for full voice support:\\n"
                "- SpeechRecognition>=3.10.0\\n"
                "- edge-tts>=6.1.0\\n"
                "- discord-ext-voice-recv>=0.5.0",
                inline=False,
            )

        await ctx.send(embed=embed)

    logger.info("Voice commands registered")
