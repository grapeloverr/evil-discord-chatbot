#!/usr/bin/env python3
"""
GPT-4chan Style Training Pipeline

Builds a normalized personality dataset from conversation logs and scraped data.
Focus:
- Ingest "4chan-style" / edgy chat logs
- Normalize grammar and structure (without making tone sterile)
- Export style profile + fine-tune-ready examples
"""

import argparse
import asyncio
import html
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


ERROR_LIKE_RESPONSES = {
    "sorry, my brain is lagging.",
    "having trouble thinking right now.",
    "[speech not understood]",
    "sorry, i had trouble processing that.",
}

GENERIC_RESPONSE_MARKERS = [
    "i'm sorry to hear that",
    "let's try to keep the conversation",
    "remember, i'm here to help",
    "how can i help you",
    "as an ai",
    "style:",
]

BAD_STYLE_MARKERS = [
    "/trashcoded/",
    "trashcode",
    "sounds like a real conversation on discord",
    "just kidding",
    "style:",
    "hey jugg",
    "jugg",
    "mdl",
    "esl",
]
HARD_SKIP_PATTERN = re.compile(
    r"\b(rape|rapist|pedo|pedophile|child porn|cp|nigger|faggot|kike|chink)\b",
    re.I,
)

DEFAULT_INPUT_FILES = [
    "training_conversations.txt",
    "scraped_training_data.json",
    "scraped_guild_1431511744748191950.json",
    "sample_messages.json",
    "4chan_posts.json",
    "4chan_posts.jsonl",
    "4chan_posts.txt",
]


@dataclass
class TrainingExample:
    instruction: str
    input_text: str
    response: str
    style_tags: list = field(default_factory=list)
    context_hints: list = field(default_factory=list)


@dataclass
class StyleProfile:
    casual_markers: list = field(default_factory=list)
    meme_references: list = field(default_factory=list)
    humor_style: str = "sarcastic"
    roast_intensity: float = 0.7
    grammar_rules: dict = field(default_factory=dict)
    contraction_map: dict = field(default_factory=dict)


class GPT4chanStyleTrainer:
    MEME_REFERENCES = [
        "pepelaugh",
        "pepe",
        "feelsgoodman",
        "feelsbadman",
        "triggered",
        "sneed",
        "chad",
        "virgin",
        "based",
        "bruh",
        "oof",
        "ratio",
        "cope",
        "ngmi",
    ]

    CASUAL_MARKERS = [
        "tbh",
        "imo",
        "imho",
        "ngl",
        "fr",
        "frfr",
        "lowkey",
        "highkey",
        "bet",
        "w",
        "l",
        "no cap",
        "kinda",
        "gonna",
        "ain't",
    ]

    def __init__(self, min_chars: int = 3):
        self.training_examples: list[TrainingExample] = []
        self.style_profile = StyleProfile()
        self.word_counts = Counter()
        self.bigram_counts = Counter()
        self._seen_pairs = set()
        self.min_chars = max(1, min_chars)

    def load_conversation_data(self, filepath: str):
        path = Path(filepath)
        if not path.exists():
            return 0

        loaded_before = len(self.training_examples)
        suffix = path.suffix.lower()

        if suffix == ".json":
            self._load_json(path)
        elif suffix == ".jsonl":
            self._load_jsonl(path)
        else:
            self._load_text(path)

        loaded_now = len(self.training_examples) - loaded_before
        print(f"Loaded {loaded_now} examples from {path.name}")
        return loaded_now

    def _load_json(self, path: Path):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Failed to read {path}: {exc}")
            return

        if isinstance(data, dict):
            if isinstance(data.get("samples"), list):
                data = data["samples"]
            else:
                data = [data]

        if not isinstance(data, list):
            return

        for item in data:
            if isinstance(item, dict):
                self._add_example(item)

    def _load_jsonl(self, path: Path):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            print(f"Failed to read {path}: {exc}")
            return

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                self._add_example(item)

    def _load_text(self, path: Path):
        raw = path.read_text(encoding="utf-8", errors="ignore")

        # training_conversations.txt format:
        # === timestamp ===
        # USER: ...
        # BOT: ...
        block_pattern = re.compile(
            r"===.*?===\s*USER:\s*(.*?)\s*BOT:\s*(.*?)(?=\n===|\Z)", re.S | re.I
        )
        matches = list(block_pattern.finditer(raw))
        if matches:
            for m in matches:
                self._add_example(
                    {
                        "instruction": "Respond naturally in the room's tone",
                        "input": m.group(1).strip(),
                        "output": m.group(2).strip(),
                    }
                )
            return

        # Fallback for plain text 4chan-like post dumps:
        # Use consecutive non-empty lines as conversational pairs.
        lines = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("==="):
                continue
            # Strip known role prefixes if present.
            line = re.sub(r"^(USER|BOT|ANON|OP)\s*:\s*", "", line, flags=re.I)
            lines.append(line)

        for i in range(len(lines) - 1):
            self._add_example(
                {
                    "instruction": "Reply in context",
                    "input": lines[i],
                    "output": lines[i + 1],
                }
            )

    def _pick_text(self, item: dict, keys: list[str]) -> str:
        for key in keys:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _add_example(self, item: dict):
        response_raw = self._pick_text(
            item,
            [
                "response",
                "output",
                "bot_response",
                "assistant",
                "reply",
                "target",
                "completion",
                "text",
                "com",
                "comment",
                "body",
            ],
        )
        input_raw = self._pick_text(
            item,
            [
                "input",
                "input_text",
                "prompt",
                "user_input",
                "instruction",
                "question",
                "context",
                "parent",
                "title",
            ],
        )

        if not response_raw:
            return

        input_text = self.normalize_input(input_raw)
        response = self.normalize_grammar(response_raw)

        if self._should_skip_pair(input_text, response):
            return

        pair_key = (input_text.lower(), response.lower())
        if pair_key in self._seen_pairs:
            return
        self._seen_pairs.add(pair_key)

        style_tags = self._analyze_style(response)
        example = TrainingExample(
            instruction=item.get("instruction", "Respond in-context, naturally"),
            input_text=input_text,
            response=response,
            style_tags=style_tags,
            context_hints=self._extract_context_hints(input_text, response),
        )
        self.training_examples.append(example)

    def _should_skip_pair(self, input_text: str, response: str) -> bool:
        if HARD_SKIP_PATTERN.search(input_text or ""):
            return True
        if HARD_SKIP_PATTERN.search(response or ""):
            return True
        if len(response) < self.min_chars:
            return True
        if not re.search(r"[A-Za-z]", response):
            return True
        if response.lower().strip() in ERROR_LIKE_RESPONSES:
            return True
        if "sorry, my brain is lagging" in response.lower():
            return True
        if "having trouble thinking right now" in response.lower():
            return True
        if len(response.split()) < 2:
            return True
        if input_text and response.lower() == input_text.lower():
            return True
        lowered = response.lower()
        if any(marker in lowered for marker in GENERIC_RESPONSE_MARKERS):
            return True
        if any(marker in lowered for marker in BAD_STYLE_MARKERS):
            return True
        if len(re.findall(r"[🤣😂😆🤐🔥💀]", response)) >= 2:
            return True
        return False

    def _basic_cleanup(self, text: str) -> str:
        text = html.unescape(text or "")
        text = text.replace("\u200b", " ").replace("\xa0", " ")
        text = re.sub(r"^bot\s*:\s*", "", text, flags=re.I)
        text = re.sub(r"^assistant\s*:\s*", "", text, flags=re.I)
        text = re.sub(r"```[\s\S]*?```", " ", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"<@!?\d+>", "@user", text)
        text = re.sub(r"<#\d+>", "#channel", text)
        text = re.sub(r"https?://\S+", "[link]", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_input(self, text: str) -> str:
        cleaned = self._basic_cleanup(text)
        cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
        cleaned = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

    def normalize_grammar(self, text: str) -> str:
        fixed = self._basic_cleanup(text)

        # Tighten repeated punctuation/noise while keeping conversational tone.
        fixed = re.sub(r"([!?.,])\1{2,}", r"\1", fixed)
        fixed = re.sub(r"(.)\1{4,}", r"\1\1\1", fixed)

        # Normalize spacing around punctuation.
        fixed = re.sub(r"\s+([,.!?;:])", r"\1", fixed)
        fixed = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", fixed)
        fixed = re.sub(r"([!?])\s+\.", r"\1", fixed)
        fixed = re.sub(r"\s{2,}", " ", fixed).strip()

        # Fix lowercase standalone "i".
        fixed = re.sub(r"\bi\b", "I", fixed)

        # Capitalize sentence starts.
        parts = re.split(r"([.!?]+\s*)", fixed)
        rebuilt = []
        for idx, part in enumerate(parts):
            if idx % 2 == 0:
                part = part.strip()
                if part and part[0].isalpha():
                    part = part[0].upper() + part[1:]
            rebuilt.append(part)
        fixed = "".join(rebuilt).strip()

        if fixed and fixed[-1] not in ".!?":
            fixed += "."

        return fixed

    def _analyze_style(self, text: str) -> list:
        tags = []
        text_lower = text.lower()

        if any(x in text_lower for x in ["lol", "lmao", "rofl", "kekw", "haha"]):
            tags.append("humor")
        if any(x in text_lower for x in ["tf", "wtf", "bruh", "oof"]):
            tags.append("relatable")
        if any(x in text_lower for x in ["sure", "obviously", "totally", "yeah right"]):
            tags.append("sarcasm")
        if "?" in text and len(text) < 120:
            tags.append("roast")
        if any(x in text_lower for x in ["n't", "'re", "'ve", "'ll", "'d"]):
            tags.append("casual")

        return tags

    def _extract_context_hints(self, input_text: str, response: str) -> list:
        hints = []
        low_input = (input_text or "").lower()
        if "?" in low_input:
            hints.append("answers_question")
        if any(x in low_input for x in ["think", "opinion", "feel"]):
            hints.append("opinion_given")
        if any(x in low_input for x in ["help", "how to", "what is"]):
            hints.append("provides_help")
        if any(x in low_input for x in ["hi", "hello", "hey", "yo", "sup"]):
            hints.append("greets_back")
        if any(x in low_input for x in ["why", "rate", "roast"]):
            hints.append("banter")
        return hints

    def analyze_style(self) -> StyleProfile:
        profile = StyleProfile()

        for example in self.training_examples:
            text = example.response.lower()

            for marker in self.CASUAL_MARKERS:
                if marker in text:
                    profile.casual_markers.append(marker)

            for meme in self.MEME_REFERENCES:
                if meme in text:
                    profile.meme_references.append(meme)

            words = re.findall(r"\b[a-z']+\b", text)
            for word in words:
                self.word_counts[word] += 1
            for i in range(len(words) - 1):
                self.bigram_counts[f"{words[i]} {words[i+1]}"] += 1

        tag_counts = Counter()
        for example in self.training_examples:
            for tag in example.style_tags:
                tag_counts[tag] += 1

        if tag_counts:
            profile.humor_style = tag_counts.most_common(1)[0][0]

        roast_count = sum(1 for e in self.training_examples if "roast" in e.style_tags)
        profile.roast_intensity = min(
            1.0, roast_count / max(len(self.training_examples), 1) * 2
        )

        profile.contraction_map = self._build_contraction_map()
        profile.grammar_rules = {
            "capitalize_sentence_start": True,
            "enforce_terminal_punctuation": True,
            "compress_repeated_punctuation": True,
            "preserve_casual_tone": True,
        }
        self.style_profile = profile
        return profile

    def _build_contraction_map(self) -> dict:
        return {
            "ain't": ["am not", "is not", "are not"],
            "aren't": ["are not"],
            "can't": ["cannot"],
            "could've": ["could have"],
            "couldn't": ["could not"],
            "didn't": ["did not"],
            "doesn't": ["does not"],
            "don't": ["do not"],
            "gonna": ["going to"],
            "gotta": ["got to"],
            "I'm": ["I am"],
            "I've": ["I have"],
            "isn't": ["is not"],
            "it's": ["it is"],
            "let's": ["let us"],
            "that's": ["that is"],
            "they're": ["they are"],
            "we're": ["we are"],
            "won't": ["will not"],
            "wouldn't": ["would not"],
            "you're": ["you are"],
        }

    def generate_personality_prompt(self) -> str:
        profile = self.style_profile
        top_markers = ", ".join(
            list(dict.fromkeys(profile.casual_markers))[:8]
        ) or "tbh, ngl, kinda, fr"
        top_bigrams = ", ".join(
            [k for k, _ in self.bigram_counts.most_common(6)]
        ) or "yeah what, what do, you need"

        return f"""You are in a Discord voice chat with friends, not customer support.

Tone:
- Conversational, blunt, and meme-aware
- Short direct responses first, detail only when asked
- Casual language is natural, but keep grammar readable
- Keep sarcasm and roasts witty, not repetitive

Style anchors from server data:
- Casual markers: {top_markers}
- Common phrase shapes: {top_bigrams}
- Roast intensity target: {profile.roast_intensity:.2f} (0-1)

Hard rules:
1. No robotic assistant phrasing ("How can I help you today?")
2. No markdown/code blocks for normal replies
3. Keep punctuation + capitalization clean
4. Match conversation context; answer the actual question
"""

    def generate_training_data(
        self, output_file: str = "trained_personality.json", include_synthetic: bool = True
    ):
        dataset = []
        for example in self.training_examples:
            dataset.append(
                {
                    "instruction": example.instruction,
                    "input": self.normalize_input(example.input_text),
                    "output": self.normalize_grammar(example.response),
                    "style_tags": example.style_tags,
                    "context_hints": example.context_hints,
                }
            )

        if include_synthetic:
            dataset.extend(self._synthesize_examples())

        # Dedupe export
        seen = set()
        deduped = []
        for item in dataset:
            key = (item.get("input", "").lower(), item.get("output", "").lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(deduped, f, indent=2, ensure_ascii=False)

        print(f"Generated {len(deduped)} training examples -> {output_file}")
        return deduped

    def _synthesize_examples(self) -> list:
        examples = []

        greetings = [
            ("Hey", "Yo, what's good?", ["casual"]),
            ("What's up", "Not much, you?", ["casual"]),
            ("Hello", "Hey. What's going on?", ["casual"]),
        ]
        for orig, response, tags in greetings:
            examples.append(
                {
                    "instruction": f"Respond to: {orig}",
                    "input": orig,
                    "output": self.normalize_grammar(response),
                    "style_tags": tags,
                    "context_hints": ["greets_back"],
                }
            )

        qa_pairs = [
            ("How are you?", "I'm doing alright, you?", ["opinion", "casual"]),
            ("What do you think?", "Honestly? Kinda mid.", ["opinion", "roast"]),
            ("Can you help me?", "Yeah, what do you need?", ["helpful", "casual"]),
            ("Do you know about X?", "Yeah. What about it?", ["opinion", "casual"]),
        ]
        for question, answer, tags in qa_pairs:
            examples.append(
                {
                    "instruction": f"Answer: {question}",
                    "input": question,
                    "output": self.normalize_grammar(answer),
                    "style_tags": tags,
                    "context_hints": ["answers_question"],
                }
            )

        return examples

    def save_style_profile(self, filename: str = "style_profile.json"):
        profile = self.style_profile
        data = {
            "casual_markers": list(dict.fromkeys(profile.casual_markers))[:50],
            "meme_references": list(dict.fromkeys(profile.meme_references))[:30],
            "humor_style": profile.humor_style,
            "roast_intensity": profile.roast_intensity,
            "grammar_rules": profile.grammar_rules,
            "contraction_map": profile.contraction_map,
            "top_words": dict(self.word_counts.most_common(100)),
            "top_bigrams": dict(self.bigram_counts.most_common(100)),
            "personality_prompt": self.generate_personality_prompt(),
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved style profile to {filename}")
        return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train personality profile from 4chan-style/server conversations."
    )
    parser.add_argument(
        "-i",
        "--input",
        action="append",
        help="Input file(s). Can be used multiple times. Defaults to known local files.",
    )
    parser.add_argument(
        "--output-dataset", default="trained_personality.json", help="Normalized dataset output JSON."
    )
    parser.add_argument(
        "--output-style", default="style_profile.json", help="Style profile output JSON."
    )
    parser.add_argument(
        "--min-chars", type=int, default=3, help="Minimum normalized response length."
    )
    parser.add_argument(
        "--no-synth", action="store_true", help="Disable synthetic helper examples."
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    print("=" * 60)
    print("GPT-4chan Style Training Pipeline")
    print("=" * 60)

    trainer = GPT4chanStyleTrainer(min_chars=args.min_chars)
    files = args.input if args.input else DEFAULT_INPUT_FILES
    files = [f for f in files if Path(f).exists()]

    if not files:
        print("No input files found. Creating starter examples only.")
        trainer.training_examples = [
            TrainingExample(
                "Respond to greeting",
                "hey what's up",
                "not much, you?",
                ["casual", "humor"],
            ),
            TrainingExample(
                "Answer opinion",
                "what do you think",
                "tbh kinda mid",
                ["opinion", "casual"],
            ),
            TrainingExample(
                "Help request",
                "can you help me",
                "yeah what do you need",
                ["helpful", "casual"],
            ),
            TrainingExample(
                "Roast",
                "why is your code like this",
                "lmao cope harder",
                ["roast", "humor"],
            ),
        ]
    else:
        print("Input files:")
        for fp in files:
            print(f"  - {fp}")
            trainer.load_conversation_data(fp)

    print("\nAnalyzing style...")
    profile = trainer.analyze_style()

    print("\nSaving outputs...")
    trainer.save_style_profile(args.output_style)
    dataset = trainer.generate_training_data(
        output_file=args.output_dataset,
        include_synthetic=not args.no_synth,
    )

    print("\n" + "=" * 60)
    print("STYLE PROFILE SUMMARY")
    print("=" * 60)
    print(f"Humor style: {profile.humor_style}")
    print(f"Roast intensity: {profile.roast_intensity:.2f}")
    print(f"Training examples in memory: {len(trainer.training_examples)}")
    print(f"Normalized dataset size: {len(dataset)}")
    print("\nApply to bot:")
    print("  1. Ensure style_profile.json and trained_personality.json are in project root")
    print("  2. Restart bot: python bot.py")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
