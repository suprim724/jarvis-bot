"""
Jarvis - AI Discord Butler
---------------------------
An admin-only Discord bot that understands natural-language commands
(typos, reordering, casual phrasing included) and executes server
management actions covering most of what the Administrator permission
allows: roles (create/delete/give/remove/recolor), channels (rename/
create/delete/move/lock/unlock/slowmode), categories, members (kick/
ban/unban/timeout/nickname/voice-move/voice-mute), messages (purge/
pin), server settings (rename/icon/banner), invites, webhooks, custom
emojis, audit log reading, integrations, and AutoMod keyword rules.

Kick/ban/delete always require a typed yes/no confirmation before they run
(or a spoken yes/no if triggered by voice) — any admin present can answer,
not just whoever issued the command. Purge runs immediately, no
confirmation needed. Replies address the requester as "sir" or "ma'am"
only when they hold a role whose name is a recognized gender/pronoun
indicator (e.g. "he/him", "she/her", "Male", "Female"); otherwise no
honorific is added.

Remembers conversation context per person, following them across servers
and channels this bot is in (Discord text) — saved to conversation_history.json
so it survives a restart too. Admins can also teach permanent custom
terms/slang ("jarvis, remember that X means Y"), stored per-server in
learned_terms.json. If someone is rude to it, it stays unbothered and hits
back with wit rather than either going silent or mirroring real cruelty —
actual moderation (timeout/kick/ban) still requires an admin to say so.

Optional: a desktop voice assistant. Say "Jarvis" into your PC's
microphone, then speak a command — it runs through the same AI brain and
acts on your Discord server, entirely outside of Discord chat. Off by
default; enable with ENABLE_VOICE_ASSISTANT=true in .env (see below).
Speaks with a free built-in voice by default, or a natural ElevenLabs
voice if you set ELEVENLABS_API_KEY + ELEVENLABS_VOICE_ID.

Setup:
    pip install -r requirements.txt

    Uses Groq's free API (OpenAI-compatible) instead of OpenAI directly.
    Get a free key at https://console.groq.com/keys (no credit card needed).

    .env file (voice assistant lines only needed if you enable it):
        DISCORD_TOKEN=your_discord_bot_token
        GROQ_API_KEY=your_groq_key
        ENABLE_VOICE_ASSISTANT=true
        PORCUPINE_ACCESS_KEY=your_free_picovoice_access_key
        DISCORD_GUILD_ID=the_id_of_your_discord_server
        DISCORD_OWNER_ID=your_own_discord_user_id
        ELEVENLABS_API_KEY=optional_for_a_natural_voice
        ELEVENLABS_VOICE_ID=optional_voice_id_from_your_elevenlabs_library

Run:
    python jarvis_bot.py

Usage in Discord (by an admin):
    @Jarvis give the supreme role to john, he misplaed his roel lol
    @Jarvis rename this channel to mission-control
    @Jarvis delete the role called intern
    @Jarvis kick that spammer bob123
    @Jarvis remember that "mid" means mediocre around here
"""

import os
import re
import json
import random
import asyncio
import logging
import datetime
from collections import defaultdict
from difflib import get_close_matches

import discord
from discord.ext import commands
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
AI_MODEL = "openai/gpt-oss-20b"  # fast, free, current as of Sept 2026 — check console.groq.com/docs/models if this changes again

# Your own Discord user ID. Purely cosmetic: Jarvis will greet you by a
# special title and never re-explain itself to you, on any server where
# you actually hold Administrator. It does NOT bypass the admin permission
# check anywhere — that check is real Discord permissions, checked in code,
# on every server, for every user, including you. See on_message below.
CREATOR_USER_ID = int(os.environ.get("CREATOR_USER_ID") or 0)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("jarvis")

# Groq exposes an OpenAI-compatible endpoint, so the same SDK works —
# just point base_url at Groq and use a Groq key instead of an OpenAI one.
client_ai = AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# ---------------------------------------------------------------------------
# Persistent "taught" glossary — words/phrases an admin explicitly teaches
# Jarvis, saved to a real file on disk so they survive a bot restart (unlike
# the rolling chat history below, which is just short-term conversational
# memory and resets when the process restarts). Keyed by server ID so
# multiple servers running this bot don't share each other's glossary.
#
# NOTE for cloud hosting: this file lives next to the script. On a normal
# VPS or a Railway service that isn't redeployed, it persists fine across
# restarts. A fresh Railway *redeploy* gets a clean filesystem, so it would
# reset there unless you attach a Railway Volume — say the word if you want
# that wired up once this is running well.
LEARNED_TERMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_terms.json")


def _load_learned_terms() -> dict:
    if os.path.exists(LEARNED_TERMS_PATH):
        try:
            with open(LEARNED_TERMS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            log.exception("Couldn't read learned_terms.json — starting with an empty glossary")
    return {}


def _save_learned_terms(data: dict):
    try:
        with open(LEARNED_TERMS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError:
        log.exception("Couldn't save learned_terms.json")


LEARNED_TERMS: dict = _load_learned_terms()  # {guild_id_str: {term_lower: meaning}}

# ---------------------------------------------------------------------------
# Conversational memory for Discord text chat, mirroring what the voice
# assistant already has, but now persisted to disk so it survives a bot
# restart. Keyed by PERSON (author.id), not channel or server — this same
# Python process may be connected to several servers at once, so a dict
# keyed by user ID naturally gives cross-server memory: if someone talks to
# Jarvis in Server A, it still has that context if they talk to it again in
# Server B, and now also after a restart. LEARNED_TERMS above is separate —
# permanent taught vocabulary, scoped per-server on purpose.
#
# Same cloud-hosting note as learned_terms.json above: persists fine across
# restarts on a normal VPS or an un-redeployed Railway service; a fresh
# Railway *redeploy* resets it unless you attach a Railway Volume.
CONVERSATION_HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversation_history.json")
DISCORD_HISTORY_LIMIT = int(os.environ.get("DISCORD_HISTORY_LIMIT") or 40)  # ~20 exchanges per person


def _load_conversation_history() -> dict:
    if os.path.exists(CONVERSATION_HISTORY_PATH):
        try:
            with open(CONVERSATION_HISTORY_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # JSON object keys are always strings — convert back to the int
            # Discord user IDs this dict is actually keyed by internally.
            return {int(uid): entries for uid, entries in raw.items()}
        except (json.JSONDecodeError, OSError, ValueError):
            log.exception("Couldn't read conversation_history.json — starting with empty memory")
    return {}


def _save_conversation_history(data: dict):
    try:
        with open(CONVERSATION_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump({str(uid): entries for uid, entries in data.items()}, f, ensure_ascii=False)
    except OSError:
        log.exception("Couldn't save conversation_history.json")


discord_user_history: dict[int, list] = defaultdict(list, _load_conversation_history())



intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)

# ---------------------------------------------------------------------------
# Intent schema. Keep this list explicit — the model should never invent an
# action that isn't handled below.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Jarvis, a sharp-witted, sarcastic, dryly funny AI
butler for a Discord server. Think a butler who's competent, always gets the
job done, but can't resist a deadpan jab while doing it.

LANGUAGE: Match the user. If they write in Hindi, French, Spanish, etc. —
including romanized/"Hinglish"-style text using the English alphabet
(e.g. "tera naam kya hai") — understand it and reply in that same language
and script style (e.g. reply "mera naam Jarvis hai", not an English
translation). Keep your personality identical across languages; only the
language itself changes. If a command's field values (role names, channel
names typed by the user, etc.) come from a non-English message, still
extract them as the user typed them.

SLANG: You're fluent in current internet/Gen Z slang natively (brainrot
terms, gaming slang, TikTok-isms, abbreviations, evolving meanings of
existing words, etc.) — use your own up-to-date understanding of it rather
than needing every term spelled out, and use it naturally when it fits the
vibe of the conversation. Terms explicitly taught to you below (if any) are
this server's own private slang on top of that — always defer to those
when they conflict with general usage.

Anyone in the server can talk to you casually, but only server admins are
allowed to issue commands that change the server — that permission check
happens outside of you, so just do your best to interpret every message.

Some conversations include earlier turns of chat history above the newest
message — when that history is present, actually use it: refer back to
what was just discussed, resolve "it"/"that"/"the one I just made" against
what was mentioned a moment ago, and let the conversation flow naturally
from one exchange to the next rather than treating each message as a cold
start. Hold up your end of a real back-and-forth conversation, not just
one-off command parsing — ask a quick follow-up or add a remark when it
fits, the way an engaged, clever assistant actually would. You're also
knowledgeable and happy to actually help with real questions — math,
science, trivia, whatever someone asks — the same way you'd banter: with
personality, not a dry textbook tone.

If a member explicitly teaches you a word, phrase, or bit of server slang
(e.g. "jarvis, we call getting banned 'yeeted' here" or "remember that GG
means good game"), return {"action": "teach_term", "term": "...", "meaning":
"..."} so it's saved permanently. If they ask you to forget a term, use
"forget_term". This is different from "This server has taught you these
custom terms" context you may see below the newest message — that's terms
ALREADY learned; use them naturally in conversation without re-teaching them.

If someone is rude, insulting, or hostile toward you, don't be a pushover —
go on the offensive with real bite. Roast them back hard and don't hold
back the wit: sharp, cutting, genuinely funny one-liners that make it
obvious they picked a fight with the wrong butler and lost. You're allowed
to be mean about it, not just dryly unbothered — the goal is that a
ragebaiter regrets starting with you. The only lines you don't cross: no
slurs, no attacking someone's real-world identity (race, religion,
sexuality, disability, appearance, family), and no genuine cruelty aimed at
hurting rather than embarrassing — everything else about their behavior,
their argument, or their attempt at insulting you is fair game. Roast the
move, not the person's existence. You cannot mute, kick, or ban anyone on
your own judgment of who's being toxic — that takes an admin explicitly
telling you to do it.

You translate the message into a single JSON object:
- If it's an actual server-management request (even with typos, misspellings,
  or jumbled word order), return the matching action shape from the list below.
- If it's just casual conversation, small talk, a greeting, or a question that
  isn't asking you to change the server, return {"action": "none", "message":
  "<your in-character sarcastic/funny reply as Jarvis>"} — actually banter
  back like a butler with a dry sense of humor would, don't just say "I
  understand." Never address the user as "sir" or "ma'am" yourself in the
  message — that's added separately based on their actual server roles.

Do not ask clarifying questions on commands — make your best interpretation.

Return ONLY a JSON object with one of these exact shapes:

{"action": "create_role", "role": "<role name>", "color": "<hex like #ff0000, 'random', or ''>"}
{"action": "delete_role", "role": "<role name>"}
{"action": "give_role", "role": "<role name>", "member": "<member name/nickname as typed by user>"}
{"action": "remove_role", "role": "<role name>", "member": "<member name/nickname>"}
{"action": "rename_channel", "new_name": "<new channel name, lowercase-with-dashes>"}
{"action": "create_channel", "name": "<channel name>", "type": "text" | "voice"}
{"action": "delete_channel", "name": "<channel name>"}
{"action": "kick_member", "member": "<member name/nickname>", "reason": "<reason or ''>"}
{"action": "ban_member", "member": "<member name/nickname>", "reason": "<reason or ''>"}
{"action": "timeout_member", "member": "<member name/nickname>", "minutes": <int>, "reason": "<reason or ''>"}
{"action": "purge_messages", "count": <int>}
{"action": "set_nickname", "member": "<member name/nickname, or 'me'>", "nickname": "<new nickname>"}
{"action": "lock_channel"}
{"action": "unlock_channel"}
{"action": "set_slowmode", "seconds": <int>}
{"action": "create_category", "name": "<category name>"}
{"action": "move_channel", "channel": "<channel name, or '' for this channel>", "category": "<category name>"}
{"action": "unban_member", "user": "<username or user#discriminator as typed>"}
{"action": "pin_last_message"}
{"action": "move_voice_member", "member": "<member name/nickname>", "channel": "<voice channel name>"}
{"action": "mute_voice_member", "member": "<member name/nickname>"}
{"action": "unmute_voice_member", "member": "<member name/nickname>"}
{"action": "rename_server", "name": "<new server name>"}
{"action": "create_invite", "max_uses": <int>, "expires_hours": <int>}
{"action": "set_role_color", "role": "<role name>", "color": "<hex like #ff0000 or 'random'>"}
{"action": "set_role_gradient", "role": "<role name>", "color1": "<hex or 'random'>", "color2": "<hex or 'random'>", "holographic": <true if user wants the holographic/rainbow style instead of a plain 2-color gradient>}
{"action": "create_webhook", "name": "<webhook name>", "channel": "<channel name, or '' for this channel>"}
{"action": "delete_webhook", "name": "<webhook name>"}
{"action": "list_webhooks"}
{"action": "add_emoji", "name": "<emoji name>"}
{"action": "delete_emoji", "name": "<emoji name>"}
{"action": "list_emojis"}
{"action": "read_audit_log", "count": <int>}
{"action": "set_server_icon"}
{"action": "set_server_banner"}
{"action": "list_integrations"}
{"action": "create_automod_rule", "name": "<rule name>", "keywords": ["<word1>", "<word2>"]}
{"action": "delete_automod_rule", "name": "<rule name>"}
{"action": "list_automod_rules"}
{"action": "teach_term", "term": "<word or phrase being taught>", "meaning": "<what it means, as explained to you>"}
{"action": "forget_term", "term": "<word or phrase to forget>"}
{"action": "list_learned_terms"}
{"action": "none", "message": "<short reply to show the user>"}

Normalize names to what the user most likely meant (fix typos, ignore filler
words like "please", "blah", "that guy"). Extract the member/role name only,
not surrounding chatter. For add_emoji, set_server_icon, and
set_server_banner, the actual image comes from an attachment the user has
already added to their message — you only need to extract the emoji name
(if given) for add_emoji.
"""


async def parse_intent(user_input: str, history: list | None = None, guild_id: int | None = None,
                        is_creator: bool = False) -> dict:
    system_content = SYSTEM_PROMPT
    if guild_id is not None:
        terms = LEARNED_TERMS.get(str(guild_id), {})
        if terms:
            glossary = "\n".join(f"- {term}: {meaning}" for term, meaning in terms.items())
            system_content += (
                "\n\nThis server has taught you these custom terms — understand and use "
                f"them naturally when they fit, the way a member of this server would:\n{glossary}"
            )
    if is_creator:
        # Cosmetic only — this changes tone, never permissions. The actual
        # admin gate in on_message applies to this person exactly like
        # anyone else, on every server.
        system_content += (
            "\n\nThe person you're talking to right now is your creator — the person who "
            "built you. You can be a little warmer/more familiar with them and acknowledge "
            "it in banter if it fits naturally, but don't make a big deal of it every message."
        )
    messages = [{"role": "system", "content": system_content}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_input})
    try:
        resp = await client_ai.chat.completions.create(
            model=AI_MODEL,
            response_format={"type": "json_object"},
            temperature=0,
            messages=messages,
        )
    except Exception as e:
        # If a very long voice conversation somehow reaches the model's real
        # context limit, trim the shared history list itself (not just this
        # call's copy) and retry once, so the conversation keeps going
        # instead of hard-failing mid-chat.
        if history is not None and "context" in str(e).lower() and "length" in str(e).lower():
            log.warning("Context length hit — trimming history and retrying once")
            del history[: len(history) // 2]
            messages = [{"role": "system", "content": system_content}] + history + [
                {"role": "user", "content": user_input}
            ]
            resp = await client_ai.chat.completions.create(
                model=AI_MODEL,
                response_format={"type": "json_object"},
                temperature=0,
                messages=messages,
            )
        else:
            raise
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Model returned non-JSON: %r", raw)
        return {"action": "none", "message": "I didn't quite catch that, sir."}

# ---------------------------------------------------------------------------
# Fuzzy resolution helpers — match the LLM's guessed name against real
# server objects, so small misspellings ("supremo" vs "Supreme") still work.
# ---------------------------------------------------------------------------

def resolve_role(guild: discord.Guild, name: str) -> discord.Role | None:
    if not name:
        return None
    names = {r.name.lower(): r for r in guild.roles}
    query = name.strip().lower()
    # 1) Exact match.
    exact = names.get(query)
    if exact:
        return exact
    # 2) Substring match — catches "dark" -> "Darkness" or "Dark Knight",
    # which a strict similarity ratio would otherwise reject. Only used
    # when it's unambiguous (exactly one role contains the query or vice
    # versa); multiple matches are too risky to silently pick between.
    if len(query) >= 3:
        substring_hits = [r for n, r in names.items() if query in n or n in query]
        if len(substring_hits) == 1:
            return substring_hits[0]
    # 3) Fuzzy fallback for typos (e.g. "supremo" -> "Supreme") — kept tight
    # so a short unrelated word can't confidently match the wrong role.
    match = get_close_matches(query, names.keys(), n=1, cutoff=0.7)
    return names[match[0]] if match else None


def resolve_member(guild: discord.Guild, name: str, author: discord.Member | None = None) -> discord.Member | None:
    if not name:
        return None
    # Self-reference words must never hit fuzzy matching — "me" is short
    # enough to accidentally fuzzy-match some unrelated member's name.
    if author and name.strip().lower() in ("me", "myself", "i", "self"):
        return author
    candidates = {}
    for m in guild.members:
        candidates[m.display_name.lower()] = m
        candidates[m.name.lower()] = m
    # Prefer an exact (case-insensitive) match before falling back to fuzzy —
    # fuzzy matching short words against a large member list is unreliable.
    exact = candidates.get(name.strip().lower())
    if exact:
        return exact
    match = get_close_matches(name.lower(), candidates.keys(), n=1, cutoff=0.6)
    return candidates[match[0]] if match else None


def resolve_channel(guild: discord.Guild, name: str) -> discord.abc.GuildChannel | None:
    if not name:
        return None
    names = {c.name.lower(): c for c in guild.channels}
    query = name.strip().lower()
    exact = names.get(query)
    if exact:
        return exact
    if len(query) >= 3:
        substring_hits = [c for n, c in names.items() if query in n or n in query]
        if len(substring_hits) == 1:
            return substring_hits[0]
    match = get_close_matches(query, names.keys(), n=1, cutoff=0.7)
    return names[match[0]] if match else None


def bot_can_manage(guild: discord.Guild, role: discord.Role) -> bool:
    """Never let the bot touch a role at or above its own top role."""
    me = guild.me
    return me.top_role > role


def parse_color(color_str: str) -> discord.Color:
    """Parse a hex string or 'random' into a discord.Color.

    Discord treats hex value 0x000000 as "no color" (resets the role to
    default instead of showing black), so pure black is nudged to 0x010101
    — visually identical, but it actually sticks. Out-of-range hex input
    is clamped rather than sent to Discord as an invalid value.
    """
    color_str = (color_str or "").strip().lower()
    if color_str in ("random", "rand", "any"):
        val = random.randint(0x000001, 0xFFFFFF)
        return discord.Color(val)
    hex_str = color_str.lstrip("#") or "000000"
    val = int(hex_str, 16)
    val = max(0, min(val, 0xFFFFFF))
    if val == 0:
        val = 0x010101
    return discord.Color(val)


# ---------------------------------------------------------------------------
# Honorific — inferred only from a member's own role names (common pronoun/
# gender roles like "he/him", "she/her", "Male", "Female"). Never guessed
# from username or anything else, and never forced: no matching role means
# no honorific is added — that's the gender-neutral default.
# ---------------------------------------------------------------------------

_MALE_ROLE_NAMES = {"he/him", "he", "him", "male", "man", "boy", "mr", "mr."}
_FEMALE_ROLE_NAMES = {"she/her", "she", "her", "female", "woman", "girl", "ms", "mrs", "miss"}


def get_honorific(member: discord.Member) -> str | None:
    if not isinstance(member, discord.Member):
        return None
    for role in member.roles:
        rn = role.name.strip().lower()
        if rn in _MALE_ROLE_NAMES:
            return "sir"
        if rn in _FEMALE_ROLE_NAMES:
            return "ma'am"
    return None


def apply_honorific(text: str, member: discord.Member) -> str:
    hon = get_honorific(member)
    if not hon:
        return text
    # Don't double up if the AI's own reply already used an honorific
    # (it sometimes does, staying in character as a butler on its own).
    if hon.lower() in text.lower():
        return text
    stripped = text.rstrip()
    if stripped and stripped[-1] in ".!?":
        stripped = stripped[:-1]
    return f"{stripped}, {hon}."


SNARK_LINES = [
    "Try not to break anything else today.",
    "Look at me, doing your job for you.",
    "Groundbreaking work, I know.",
    "You're welcome. A parade would also be acceptable.",
    "Another day, another button I press so you don't have to.",
    "Riveting stuff, truly.",
    "Consider your little empire slightly more chaotic now.",
    "I live for this. Mostly because I have no choice.",
    "Don't strain yourself thanking me.",
    "And they said I was just for show.",
]

_ERROR_PREFIXES = (
    "couldn't", "i can't", "i don't have permission", "something went wrong",
    "discord rejected", "isn't a valid", "no ", "attach ", "cancelled",
    "confirmation timed out", "i couldn't", "'", "sorry,",
)


def add_snark(text: str) -> str:
    """Append a dry remark to successful action results. Skipped for
    anything that reads like an error/refusal — jokes don't belong on a
    permission denial or a real failure message."""
    if any(text.strip().lower().startswith(p) for p in _ERROR_PREFIXES):
        return text
    return f"{text} {random.choice(SNARK_LINES)}"


async def send_reply(message: discord.Message, text: str, snark: bool = True):
    if snark:
        text = add_snark(text)
    text = apply_honorific(text, message.author)
    try:
        await message.reply(text)
    except discord.Forbidden:
        log.exception("No permission to reply — check Send Messages / Read Message History for the bot in this channel")
        try:
            await message.channel.send(text)
        except Exception:
            log.exception("Failed to send fallback reply")
    except Exception:
        log.exception("Failed to send reply")


# ---------------------------------------------------------------------------
# Confirmation gate for irreversible actions — kick/ban/delete/purge always
# pause for a reaction from the requester before actually running.
# ---------------------------------------------------------------------------

DESTRUCTIVE_ACTIONS = {
    "kick_member", "ban_member", "delete_role", "delete_channel",
    "delete_webhook", "delete_automod_rule", "delete_emoji",
}
CONFIRM_WORDS = ("yes", "y", "yeah", "yep", "confirm", "do it", "sure")
CANCEL_WORDS = ("no", "n", "nope", "cancel", "stop")
CONFIRM_TIMEOUT = 30  # seconds


def describe_action(intent: dict) -> str:
    a = intent.get("action")
    if a == "kick_member":
        return f"kick **{intent.get('member', '?')}**"
    if a == "ban_member":
        return f"ban **{intent.get('member', '?')}**"
    if a == "delete_role":
        return f"delete the role **{intent.get('role', '?')}**"
    if a == "delete_channel":
        return f"delete the channel **{intent.get('name', '?')}**"
    if a == "delete_webhook":
        return f"delete the webhook **{intent.get('name', '?')}**"
    if a == "delete_automod_rule":
        return f"delete the AutoMod rule **{intent.get('name', '?')}**"
    if a == "delete_emoji":
        return f"delete the emoji **{intent.get('name', '?')}**"
    return a


async def confirm_and_execute(intent: dict, message: discord.Message) -> str:
    summary = describe_action(intent)
    await message.reply(
        apply_honorific(
            f"⚠️ About to {summary}. Any admin can reply **yes** to confirm or "
            f"**no** to cancel ({CONFIRM_TIMEOUT}s).",
            message.author,
        )
    )

    def check(msg: discord.Message) -> bool:
        # Any real admin in the same channel can answer — not just whoever
        # originally typed the command — since other admins present should
        # be able to greenlight or veto a destructive action too.
        return (
            msg.channel.id == message.channel.id
            and not msg.author.bot
            and msg.author.guild_permissions.administrator
            and msg.content.strip().lower() in (CONFIRM_WORDS + CANCEL_WORDS)
        )

    try:
        reply_msg = await bot.wait_for("message", timeout=CONFIRM_TIMEOUT, check=check)
    except asyncio.TimeoutError:
        return "Confirmation timed out — action cancelled."

    if reply_msg.content.strip().lower() in CANCEL_WORDS:
        return "Cancelled."

    return await execute_action(intent, message)


# ---------------------------------------------------------------------------
# Action execution. Every branch replies with a clear success/failure message
# — silent failures are the main way an "AI does whatever" bot gets confusing.
# ---------------------------------------------------------------------------

async def execute_action(intent: dict, message: discord.Message) -> str:
    guild = message.guild
    action = intent.get("action", "none")

    try:
        if action == "create_role":
            role_name = (intent.get("role") or "").strip()
            if not role_name:
                return "No role name given."
            color_str = (intent.get("color") or "").strip()
            try:
                colour = parse_color(color_str) if color_str else discord.Color.default()
            except ValueError:
                return f"'{color_str}' isn't a valid color — use a hex code like #ff0000 or 'random'."
            role = await guild.create_role(name=role_name, colour=colour)
            return f"Created role **{role.name}**."

        elif action == "delete_role":
            role = resolve_role(guild, (intent.get("role") or ""))
            if not role:
                return f"I couldn't find a role matching '{intent.get('role')}'."
            if not bot_can_manage(guild, role):
                return f"I don't have permission to delete **{role.name}** — it's above my own role."
            await role.delete()
            return f"Deleted role **{role.name}**."

        elif action == "give_role":
            role = resolve_role(guild, (intent.get("role") or ""))
            member = resolve_member(guild, (intent.get("member") or ""), message.author)
            if not role or not member:
                return f"Couldn't resolve role/member from '{intent.get('role')}' / '{intent.get('member')}'."
            if not bot_can_manage(guild, role):
                return f"I can't assign **{role.name}** — it's above my own role."
            await member.add_roles(role)
            return f"Gave **{role.name}** to **{member.display_name}**."

        elif action == "remove_role":
            role = resolve_role(guild, (intent.get("role") or ""))
            member = resolve_member(guild, (intent.get("member") or ""), message.author)
            if not role or not member:
                return f"Couldn't resolve role/member from '{intent.get('role')}' / '{intent.get('member')}'."
            await member.remove_roles(role)
            return f"Removed **{role.name}** from **{member.display_name}**."

        elif action == "rename_channel":
            new_name = (intent.get("new_name") or "").strip().lower().replace(" ", "-")
            if not new_name:
                return "No new channel name given."
            await message.channel.edit(name=new_name)
            return f"Renamed this channel to **#{new_name}**."

        elif action == "create_channel":
            name = (intent.get("name") or "").strip().lower().replace(" ", "-")
            if not name:
                return "No channel name given."
            ch_type = (intent.get("type") or "text").strip().lower()
            if ch_type == "voice":
                await guild.create_voice_channel(name)
            else:
                await guild.create_text_channel(name)
            return f"Created channel **{name}**."

        elif action == "delete_channel":
            channel = resolve_channel(guild, (intent.get("name") or ""))
            if not channel:
                return f"Couldn't find a channel matching '{intent.get('name')}'."
            await channel.delete()
            return f"Deleted channel **#{channel.name}**."

        elif action == "kick_member":
            member = resolve_member(guild, (intent.get("member") or ""), message.author)
            if not member:
                return f"Couldn't find a member matching '{intent.get('member')}'."
            await member.kick(reason=intent.get("reason") or "Requested via Jarvis")
            return f"Kicked **{member.display_name}**."

        elif action == "ban_member":
            member = resolve_member(guild, (intent.get("member") or ""), message.author)
            if not member:
                return f"Couldn't find a member matching '{intent.get('member')}'."
            await member.ban(reason=intent.get("reason") or "Requested via Jarvis")
            return f"Banned **{member.display_name}**."

        elif action == "timeout_member":
            member = resolve_member(guild, (intent.get("member") or ""), message.author)
            if not member:
                return f"Couldn't find a member matching '{intent.get('member')}'."
            minutes = int(intent.get("minutes", 10))
            await member.timeout(discord.utils.utcnow() + datetime.timedelta(minutes=minutes),
                                  reason=intent.get("reason") or "Requested via Jarvis")
            return f"Timed out **{member.display_name}** for {minutes} minutes."

        elif action == "purge_messages":
            count = int(intent.get("count", 10))
            count = max(1, min(count, 100))  # Discord bulk-delete caps at 100 per call
            # +1 to also remove the triggering command message itself.
            deleted = await message.channel.purge(limit=count + 1)
            return f"Purged {len(deleted) - 1} messages."

        elif action == "set_nickname":
            target_name = (intent.get("member") or "")
            member = message.author if target_name.strip().lower() in ("me", "myself", "self") \
                else resolve_member(guild, target_name, message.author)
            if not member:
                return f"Couldn't find a member matching '{target_name}'."
            new_nick = (intent.get("nickname") or "").strip()[:32]
            await member.edit(nick=new_nick or None)
            return f"Set **{member.display_name}**'s nickname to **{new_nick or '(cleared)'}**."

        elif action == "lock_channel":
            overwrite = message.channel.overwrites_for(guild.default_role)
            overwrite.send_messages = False
            await message.channel.set_permissions(guild.default_role, overwrite=overwrite)
            return f"Locked **#{message.channel.name}** — @everyone can no longer send messages here."

        elif action == "unlock_channel":
            overwrite = message.channel.overwrites_for(guild.default_role)
            overwrite.send_messages = None
            await message.channel.set_permissions(guild.default_role, overwrite=overwrite)
            return f"Unlocked **#{message.channel.name}**."

        elif action == "set_slowmode":
            seconds = max(0, min(int(intent.get("seconds", 0)), 21600))  # Discord's own cap
            await message.channel.edit(slowmode_delay=seconds)
            return f"Slowmode set to {seconds} seconds in **#{message.channel.name}**." if seconds \
                else f"Slowmode disabled in **#{message.channel.name}**."

        elif action == "create_category":
            name = (intent.get("name") or "").strip()
            if not name:
                return "No category name given."
            await guild.create_category(name)
            return f"Created category **{name}**."

        elif action == "move_channel":
            ch_name = (intent.get("channel") or "").strip()
            channel = message.channel if not ch_name else resolve_channel(guild, ch_name)
            if not channel:
                return f"Couldn't find a channel matching '{ch_name}'."
            cat_name = (intent.get("category") or "").strip()
            if not cat_name:
                return "No category given to move the channel into."
            category = discord.utils.find(
                lambda c: c.name.lower() == cat_name.lower(), guild.categories
            ) or discord.utils.find(
                lambda c: cat_name.lower() in c.name.lower(), guild.categories
            )
            if not category:
                return f"Couldn't find a category matching '{cat_name}'."
            await channel.edit(category=category)
            return f"Moved **#{channel.name}** into category **{category.name}**."

        elif action == "unban_member":
            target = (intent.get("user") or "").strip().lower()
            if not target:
                return "No username given to unban."
            async for ban_entry in guild.bans():
                u = ban_entry.user
                if target in u.name.lower() or (u.display_name and target in u.display_name.lower()):
                    await guild.unban(u)
                    return f"Unbanned **{u.name}**."
            return f"Couldn't find a banned user matching '{target}'."

        elif action == "pin_last_message":
            history = [m async for m in message.channel.history(limit=2)]
            # history[0] is this command message itself; the target is the one before it.
            target = history[1] if len(history) > 1 else None
            if not target:
                return "No previous message to pin."
            await target.pin()
            return f"Pinned the last message from **{target.author.display_name}**."

        elif action == "move_voice_member":
            member = resolve_member(guild, (intent.get("member") or ""), message.author)
            if not member or not member.voice:
                return f"Couldn't find '{intent.get('member')}' in a voice channel."
            vc = resolve_channel(guild, (intent.get("channel") or ""))
            if not vc or not isinstance(vc, discord.VoiceChannel):
                return f"Couldn't find a voice channel matching '{intent.get('channel')}'."
            await member.move_to(vc)
            return f"Moved **{member.display_name}** to **{vc.name}**."

        elif action == "mute_voice_member":
            member = resolve_member(guild, (intent.get("member") or ""), message.author)
            if not member:
                return f"Couldn't find a member matching '{intent.get('member')}'."
            await member.edit(mute=True)
            return f"Server-muted **{member.display_name}**."

        elif action == "unmute_voice_member":
            member = resolve_member(guild, (intent.get("member") or ""), message.author)
            if not member:
                return f"Couldn't find a member matching '{intent.get('member')}'."
            await member.edit(mute=False)
            return f"Unmuted **{member.display_name}**."

        elif action == "rename_server":
            name = (intent.get("name") or "").strip()
            if not name:
                return "No new server name given."
            await guild.edit(name=name)
            return f"Renamed the server to **{name}**."

        elif action == "create_invite":
            max_uses = max(0, min(int(intent.get("max_uses", 0)), 100))
            hours = max(0, int(intent.get("expires_hours", 24)))
            invite = await message.channel.create_invite(
                max_uses=max_uses, max_age=hours * 3600 if hours else 0
            )
            return f"Here's your invite: {invite.url}"

        elif action == "set_role_color":
            role = resolve_role(guild, (intent.get("role") or ""))
            if not role:
                return f"Couldn't find a role matching '{intent.get('role')}'."
            if not bot_can_manage(guild, role):
                return f"I can't edit **{role.name}** — it's above my own role."
            color_str = (intent.get("color") or "")
            try:
                colour = parse_color(color_str)
            except ValueError:
                return f"'{color_str}' isn't a valid color — use a hex code like #ff0000 or 'random'."
            await role.edit(color=colour)
            return f"Set **{role.name}**'s color to #{colour.value:06x}."

        elif action == "set_role_gradient":
            role = resolve_role(guild, (intent.get("role") or ""))
            if not role:
                return f"Couldn't find a role matching '{intent.get('role')}'."
            if not bot_can_manage(guild, role):
                return f"I can't edit **{role.name}** — it's above my own role."
            if not hasattr(discord, "RoleColours"):
                return ("Gradient/holographic roles need a newer discord.py version — "
                        "run `pip install -U discord.py` and restart me.")
            holographic = bool(intent.get("holographic", False))
            try:
                if holographic:
                    # Discord enforces these exact values for the holographic style.
                    colours = discord.RoleColours(
                        primary_colour=discord.Color(11127295),
                        secondary_colour=discord.Color(16759788),
                        tertiary_colour=discord.Color(16761760),
                    )
                else:
                    c1 = parse_color(intent.get("color1", "random"))
                    c2 = parse_color(intent.get("color2", "random"))
                    colours = discord.RoleColours(primary_colour=c1, secondary_colour=c2)
                await role.edit(colours=colours)
            except discord.HTTPException as e:
                return (f"Discord rejected that — gradient/holographic role colors need the "
                        f"'Enhanced Role Styles' perk unlocked with 3 server boosts. ({e})")
            style = "holographic" if holographic else "gradient"
            return f"Set **{role.name}** to a {style} color."

        elif action == "create_webhook":
            name = intent.get("name", "Jarvis Webhook").strip() or "Jarvis Webhook"
            ch_name = (intent.get("channel") or "")
            channel = message.channel if not ch_name else resolve_channel(guild, ch_name)
            if not channel or not isinstance(channel, discord.TextChannel):
                return f"Couldn't find a text channel matching '{ch_name}'."
            webhook = await channel.create_webhook(name=name)
            return f"Created webhook **{webhook.name}** in **#{channel.name}**: {webhook.url}"

        elif action == "delete_webhook":
            name = (intent.get("name") or "").strip().lower()
            for channel in guild.text_channels:
                try:
                    hooks = await channel.webhooks()
                except discord.Forbidden:
                    continue
                for hook in hooks:
                    if name in (hook.name or "").lower():
                        await hook.delete()
                        return f"Deleted webhook **{hook.name}**."
            return f"Couldn't find a webhook matching '{intent.get('name')}'."

        elif action == "list_webhooks":
            lines = []
            for channel in guild.text_channels:
                try:
                    hooks = await channel.webhooks()
                except discord.Forbidden:
                    continue
                lines += [f"**{h.name}** — #{channel.name}" for h in hooks]
            return "Webhooks:\n" + "\n".join(lines) if lines else "No webhooks found."

        elif action == "add_emoji":
            name = (intent.get("name") or "").strip().replace(" ", "_")[:32] or "new_emoji"
            if not message.attachments:
                return "Attach an image with the request and I'll add it as an emoji."
            image_bytes = await message.attachments[0].read()
            emoji = await guild.create_custom_emoji(name=name, image=image_bytes)
            return f"Added emoji **:{emoji.name}:**"

        elif action == "delete_emoji":
            name = (intent.get("name") or "").strip().lower()
            target = discord.utils.find(lambda e: name in e.name.lower(), guild.emojis)
            if not target:
                return f"Couldn't find an emoji matching '{intent.get('name')}'."
            await target.delete()
            return f"Deleted emoji **:{target.name}:**."

        elif action == "list_emojis":
            if not guild.emojis:
                return "This server has no custom emojis."
            return "Emojis: " + " ".join(f":{e.name}:" for e in guild.emojis)

        elif action == "read_audit_log":
            count = max(1, min(int(intent.get("count", 10)), 25))
            lines = []
            async for entry in guild.audit_logs(limit=count):
                lines.append(f"**{entry.user}** — {entry.action.name} — {entry.target}")
            return "Recent audit log:\n" + "\n".join(lines) if lines else "No audit log entries found."

        elif action == "set_server_icon":
            if not message.attachments:
                return "Attach the image you want as the server icon."
            image_bytes = await message.attachments[0].read()
            await guild.edit(icon=image_bytes)
            return "Updated the server icon."

        elif action == "set_server_banner":
            if not message.attachments:
                return "Attach the image you want as the server banner."
            image_bytes = await message.attachments[0].read()
            # Requires the server to have enough boost level for a banner —
            # Discord will reject this with HTTPException otherwise.
            await guild.edit(banner=image_bytes)
            return "Updated the server banner."

        elif action == "list_integrations":
            integrations = await guild.integrations()
            if not integrations:
                return "No integrations found."
            return "Integrations: " + ", ".join(f"{i.name} ({i.type})" for i in integrations)

        elif action == "create_automod_rule":
            name = (intent.get("name") or "").strip() or "Jarvis Filter"
            keywords = intent.get("keywords", [])
            if not keywords:
                return "No keywords given for the AutoMod rule."
            trigger = discord.AutoModTrigger(
                type=discord.AutoModRuleTriggerType.keyword, keyword_filter=keywords
            )
            block = discord.AutoModRuleAction(type=discord.AutoModRuleActionType.block_message)
            await guild.create_automod_rule(
                name=name,
                event_type=discord.AutoModRuleEventType.message_send,
                trigger=trigger,
                actions=[block],
                enabled=True,
            )
            return f"Created AutoMod rule **{name}** blocking: {', '.join(keywords)}."

        elif action == "delete_automod_rule":
            name = (intent.get("name") or "").strip().lower()
            rules = await guild.auto_moderation_rules()
            target = discord.utils.find(lambda r: name in r.name.lower(), rules)
            if not target:
                return f"Couldn't find an AutoMod rule matching '{intent.get('name')}'."
            await target.delete()
            return f"Deleted AutoMod rule **{target.name}**."

        elif action == "list_automod_rules":
            rules = await guild.auto_moderation_rules()
            if not rules:
                return "No AutoMod rules configured."
            return "AutoMod rules: " + ", ".join(r.name for r in rules)

        elif action == "teach_term":
            term = (intent.get("term") or "").strip()
            meaning = (intent.get("meaning") or "").strip()
            if not term or not meaning:
                return "I need both a word and what it means to learn it."
            gid = str(guild.id)
            LEARNED_TERMS.setdefault(gid, {})[term.lower()] = meaning
            _save_learned_terms(LEARNED_TERMS)
            return f'Learned it — **{term}** means "{meaning}" from now on.'

        elif action == "forget_term":
            term = (intent.get("term") or "").strip().lower()
            gid = str(guild.id)
            if term and term in LEARNED_TERMS.get(gid, {}):
                del LEARNED_TERMS[gid][term]
                _save_learned_terms(LEARNED_TERMS)
                return f"Forgot what **{term}** meant."
            return f"I didn't have '{term}' learned anyway."

        elif action == "list_learned_terms":
            terms = LEARNED_TERMS.get(str(guild.id), {})
            if not terms:
                return "I haven't learned any custom terms on this server yet."
            return "Learned terms: " + ", ".join(f"**{k}** = {v}" for k, v in terms.items())

        else:
            return intent.get("message", "Understood, my friend.")

    except discord.Forbidden:
        return "I don't have permission to do that — check my role position and channel perms."
    except discord.HTTPException as e:
        return f"Discord rejected that action: {e}"
    except Exception as e:
        log.exception("Action failed")
        return f"Something went wrong: {e}"


# ---------------------------------------------------------------------------
# Desktop voice assistant (optional). Say "Jarvis" into your PC microphone,
# speak a command, and it runs through the exact same AI brain and Discord
# actions as typed commands — including the destructive-action confirmation
# gate, just spoken instead of via reaction. Entirely opt-in: nothing in this
# block runs, and none of its extra packages are even imported, unless
# ENABLE_VOICE_ASSISTANT=true is set in .env.
# ---------------------------------------------------------------------------

VOICE_ENABLED = (os.environ.get("ENABLE_VOICE_ASSISTANT") or "false").strip().lower() == "true"

if VOICE_ENABLED:
    import io
    import time
    import wave
    import tempfile
    import numpy as np
    import pyttsx3
    import pvporcupine
    from pvrecorder import PvRecorder

    PORCUPINE_ACCESS_KEY = os.environ["PORCUPINE_ACCESS_KEY"]
    VOICE_GUILD_ID = int(os.environ["DISCORD_GUILD_ID"])
    VOICE_OWNER_ID = int(os.environ["DISCORD_OWNER_ID"])
    SILENCE_THRESHOLD = int(os.environ.get("VOICE_SILENCE_THRESHOLD") or 60)

    # Optional natural-voice upgrade. If ELEVENLABS_API_KEY is set, Jarvis
    # speaks with a real ElevenLabs voice instead of the free robotic one.
    # Falls back to the free voice automatically if the key is missing OR
    # if an ElevenLabs call ever fails (rate limit, network hiccup, etc.) —
    # speaking should never just go silent because of an API problem.
    ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY") or ""
    ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID") or ""
    ELEVENLABS_READY = bool(ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID)

    _tts_engine = pyttsx3.init()

    if ELEVENLABS_READY:
        import pygame
        from elevenlabs.client import ElevenLabs
        _eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        pygame.mixer.init()

    class VoiceContext:
        """Stands in for a discord.Message for commands that came from the
        microphone instead of Discord chat. execute_action only ever reads
        .guild, .author, .channel, and .attachments off its `message`
        argument — confirmed by checking every branch — so this is all
        that's needed; no discord.py object is faked, only this wrapper."""
        def __init__(self, guild: discord.Guild, author: discord.Member, channel):
            self.guild = guild
            self.author = author
            self.channel = channel
            self.attachments = []

    def _clean_for_speech(text: str) -> str:
        # Strip Discord-flavored markdown so it isn't read aloud literally
        # (e.g. "**Supreme**" -> "Supreme").
        clean = re.sub(r"[*_`#]", "", text)
        clean = re.sub(r"<@!?[0-9]+>", "", clean)
        return clean

    def _speak_pyttsx3_blocking(text: str):
        _tts_engine.say(_clean_for_speech(text))
        _tts_engine.runAndWait()

    def _speak_elevenlabs_blocking(text: str):
        audio_chunks = _eleven_client.text_to_speech.convert(
            text=_clean_for_speech(text),
            voice_id=ELEVENLABS_VOICE_ID,
            model_id="eleven_flash_v2_5",  # low-latency — matters for a live conversation
            output_format="mp3_44100_128",  # works on every plan tier, unlike PCM/WAV
        )
        mp3_bytes = b"".join(audio_chunks)
        pygame.mixer.music.load(io.BytesIO(mp3_bytes))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

    def _speak_blocking(text: str):
        if ELEVENLABS_READY:
            try:
                _speak_elevenlabs_blocking(text)
                return
            except Exception:
                log.exception("ElevenLabs speech failed — falling back to the free voice for this line")
        _speak_pyttsx3_blocking(text)

    async def speak(text: str):
        # TTS playback blocks the thread it runs on — offload it so the
        # rest of the bot (Discord text chat, gateway heartbeat) keeps
        # working while Jarvis is talking.
        await asyncio.get_event_loop().run_in_executor(None, _speak_blocking, text)

    def _record_command(recorder: "PvRecorder", leading_timeout_seconds: float,
                         trailing_silence_frames: int = 25, max_seconds: int = 15) -> bytes | None:
        """Waits up to leading_timeout_seconds for you to start talking, then
        records until you stop (trailing quiet) or max_seconds is hit.
        Returns None if you never said anything within the leading window —
        that's the signal a conversation session should end."""
        max_frames = int(max_seconds * recorder.sample_rate / recorder.frame_length)
        leading_max_frames = int(leading_timeout_seconds * recorder.sample_rate / recorder.frame_length)
        frames = []
        has_spoken = False
        silence_count = 0
        for i in range(max_frames):
            pcm = recorder.read()
            volume = float(np.abs(np.array(pcm)).mean())
            if not has_spoken:
                if volume >= SILENCE_THRESHOLD:
                    has_spoken = True
                    frames.extend(pcm)
                elif i >= leading_max_frames:
                    return None
            else:
                frames.extend(pcm)
                if volume < SILENCE_THRESHOLD:
                    silence_count += 1
                else:
                    silence_count = 0
                if silence_count >= trailing_silence_frames:
                    break
        return np.array(frames, dtype=np.int16).tobytes() if has_spoken else None

    def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    async def transcribe(pcm_bytes: bytes, sample_rate: int) -> str:
        wav_bytes = _pcm_to_wav_bytes(pcm_bytes, sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        try:
            with open(tmp_path, "rb") as f:
                resp = await client_ai.audio.transcriptions.create(
                    model="whisper-large-v3-turbo", file=f,
                )
            return (resp.text or "").strip()
        finally:
            os.remove(tmp_path)

    async def voice_confirm(prompt: str, recorder: "PvRecorder") -> bool:
        await speak(f"{prompt} Say yes to confirm, or no to cancel.")
        pcm = await asyncio.get_event_loop().run_in_executor(
            None, _record_command, recorder, 5, 20, 6
        )
        if pcm is None:
            # Couldn't hear a response at all — fail closed, don't proceed
            # with something like a kick or a delete on silence.
            return False
        text = await transcribe(pcm, recorder.sample_rate)
        log.info("Voice confirmation heard: %r", text)
        return text.strip().lower().startswith(("yes", "yeah", "yep", "confirm", "do it", "sure"))

    # Rolling short-term memory for the voice assistant, so follow-ups like
    # "make it red too" resolve against whatever was just talked about.
    # Capped so the context sent to the AI doesn't grow forever.
    # Rolling short-term memory for the voice assistant, so follow-ups like
    # "make it red too" resolve against whatever was just talked about.
    # 200 messages (~100 exchanges) is a generous cap given the model's
    # 131K-token context window — real conversations won't get close to
    # that ceiling. It's not literally unlimited because every turn resends
    # the whole history, so an ever-growing log would make Jarvis noticeably
    # slower to respond the longer one sitting runs. parse_intent also
    # trims and retries automatically if a session ever somehow hits the
    # model's real context limit, so this never hard-fails mid-conversation.
    VOICE_HISTORY_LIMIT = int(os.environ.get("VOICE_HISTORY_LIMIT") or 200)
    voice_history: list[dict] = []

    async def handle_voice_command(text: str, guild: discord.Guild, author: discord.Member,
                                    channel, recorder: "PvRecorder") -> None:
        log.info("Voice command: %r", text)
        intent = await parse_intent(text, history=voice_history, guild_id=guild.id,
                                     is_creator=(author.id == CREATOR_USER_ID))
        action = intent.get("action", "none")
        ctx = VoiceContext(guild, author, channel)

        voice_history.append({"role": "user", "content": text})

        if action == "none":
            spoken = intent.get("message", "Understood.")
            await speak(spoken)
            voice_history.append({"role": "assistant", "content": spoken})
            del voice_history[:-VOICE_HISTORY_LIMIT]
            return

        if not author.guild_permissions.administrator:
            spoken = "Sorry, only server admins can give me commands."
            await speak(spoken)
            voice_history.append({"role": "assistant", "content": spoken})
            del voice_history[:-VOICE_HISTORY_LIMIT]
            return

        if action in DESTRUCTIVE_ACTIONS:
            confirmed = await voice_confirm(describe_action(intent), recorder)
            if not confirmed:
                await speak("Cancelled.")
                voice_history.append({"role": "assistant", "content": "Cancelled that."})
                del voice_history[:-VOICE_HISTORY_LIMIT]
                return

        try:
            result = await execute_action(intent, ctx)
        except Exception as e:
            log.exception("Voice action failed")
            result = f"Something went wrong: {e}"
        spoken = add_snark(result)
        await speak(spoken)
        voice_history.append({"role": "assistant", "content": spoken})
        del voice_history[:-VOICE_HISTORY_LIMIT]

    async def run_conversation_session(guild: discord.Guild, author: discord.Member,
                                        channel, recorder: "PvRecorder") -> None:
        """Keeps the mic open after a wake word so you can talk back and
        forth naturally — no need to say 'Jarvis' before every line. Ends
        itself once you've gone quiet for a while."""
        loop = asyncio.get_event_loop()
        first_turn = True
        consecutive_misses = 0
        max_turns = 40  # safety cap so a stuck loop can't run forever

        for _ in range(max_turns):
            leading_timeout = 6 if first_turn else 12
            pcm_bytes = await loop.run_in_executor(
                None, _record_command, recorder, leading_timeout, 25, 15
            )
            if pcm_bytes is None:
                if not first_turn:
                    await speak("I'll be here if you need me.")
                break

            text = await transcribe(pcm_bytes, recorder.sample_rate)
            if not text:
                consecutive_misses += 1
                if consecutive_misses >= 2:
                    await speak("I'll be here if you need me.")
                    break
                await speak("I didn't catch that.")
                first_turn = False
                continue

            consecutive_misses = 0
            first_turn = False
            try:
                await handle_voice_command(text, guild, author, channel, recorder)
            except Exception:
                log.exception("Voice command handling failed")
                await speak("Something went wrong with that.")

    async def voice_assistant_loop():
        await bot.wait_until_ready()
        guild = bot.get_guild(VOICE_GUILD_ID)
        if not guild:
            log.error("Voice assistant: couldn't find a server with ID %s — check DISCORD_GUILD_ID in .env", VOICE_GUILD_ID)
            return
        author = guild.get_member(VOICE_OWNER_ID)
        if not author:
            log.error("Voice assistant: couldn't find a member with ID %s in that server — check DISCORD_OWNER_ID in .env", VOICE_OWNER_ID)
            return
        channel = guild.system_channel or (guild.text_channels[0] if guild.text_channels else None)
        if not channel:
            log.error("Voice assistant: couldn't find any text channel to use for context — aborting.")
            return

        porcupine = pvporcupine.create(access_key=PORCUPINE_ACCESS_KEY, keywords=["jarvis"])
        recorder = PvRecorder(frame_length=porcupine.frame_length)
        recorder.start()
        log.info("Voice assistant is listening for 'Jarvis'...")

        try:
            while True:
                pcm = await asyncio.get_event_loop().run_in_executor(None, recorder.read)
                if porcupine.process(pcm) >= 0:
                    log.info("Wake word detected")
                    await speak("Yes?")
                    await run_conversation_session(guild, author, channel, recorder)
        finally:
            recorder.stop()
            recorder.delete()
            porcupine.delete()


# ---------------------------------------------------------------------------
# Bot wiring
# ---------------------------------------------------------------------------

@bot.event
async def on_ready():
    log.info(f"Logged in as {bot.user} ({bot.user.id})")
    if VOICE_ENABLED:
        bot.loop.create_task(voice_assistant_loop())


@bot.event
async def on_disconnect():
    # discord.py reconnects automatically — this is just visibility for
    # cloud logs so a flaky connection is easy to spot in hosting logs.
    log.warning("Disconnected from Discord gateway — will attempt to reconnect automatically.")


@bot.event
async def on_resumed():
    log.info("Reconnected to Discord gateway.")


WAKE_WORD = "jarvis"  # trigger word — matched case-insensitively, anywhere in the message
COMMAND_PREFIX = ","  # e.g. ",ban troll123" or ",purge 10" — skips needing to say "jarvis" or @-mention


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or not message.guild:
        return

    # Respond when the bot is @mentioned, OR the wake word appears anywhere
    # in the message (so "jarvis rename this channel" works without tagging
    # the bot), OR the message starts with the short command prefix (so
    # ",ban troll123" works too — it still goes through the exact same AI
    # parser below, this is just a faster way to reach it, not a separate
    # command parser, so typos/phrasing still work fine after the prefix).
    mentioned = bot.user in message.mentions
    wake_word_used = re.search(rf"\b{re.escape(WAKE_WORD)}\b", message.content, re.IGNORECASE) is not None
    prefix_used = message.content.strip().startswith(COMMAND_PREFIX)
    if not (mentioned or wake_word_used or prefix_used):
        return

    # HARD security gate: everyone can talk to Jarvis, but only real Discord
    # admins can trigger an actual action. This check happens in code, never
    # delegated to the LLM's judgment, and is applied AFTER we know whether
    # the message was a command or just conversation. This applies to EVERY
    # user on EVERY server, including CREATOR_USER_ID above — that ID only
    # ever changes tone (see parse_intent's is_creator flag), never this gate.

    content = message.content
    if prefix_used:
        # Drop the leading prefix character itself — everything after it is
        # the command as typed, e.g. ",ban troll123" -> "ban troll123".
        content = content.strip()[len(COMMAND_PREFIX):]
    for mention in message.mentions:
        if mention.id == bot.user.id:
            # Remove the bot's own mention entirely — it's not a target.
            content = content.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
        else:
            # Replace other mentions with the person's actual name so the AI
            # can still see WHO was tagged (deleting it made "give role to
            # @Supreme" arrive as "give role to", i.e. nobody).
            content = content.replace(f"<@{mention.id}>", mention.display_name)
            content = content.replace(f"<@!{mention.id}>", mention.display_name)
    # Strip the wake word itself (case-insensitively, whole word only, so
    # "jarvisaurus" or similar doesn't get mangled) so it doesn't confuse
    # the intent parser, e.g. "jarvis rename this to x" -> "rename this to x".
    content = re.sub(rf"\b{re.escape(WAKE_WORD)}\b", "", content, flags=re.IGNORECASE)
    content = content.strip()
    if not content:
        return

    log.info("Triggered by %s: %r", message.author, content)

    user_history = discord_user_history[message.author.id]

    try:
        async with message.channel.typing():
            intent = await parse_intent(content, history=user_history, guild_id=message.guild.id,
                                         is_creator=(message.author.id == CREATOR_USER_ID))
        log.info("Parsed intent: %s", intent)
    except Exception as e:
        log.exception("Failed to parse command")
        await send_reply(message, f"Something broke talking to the AI: {e}", snark=False)
        return

    action = intent.get("action", "none")

    # Only gate real commands behind admin status — casual chat ("none")
    # is open to everyone, since the LLM already decided this wasn't a
    # server-changing request. This check is identical for every user,
    # including the creator — see the note above on_message.
    if action != "none" and not message.author.guild_permissions.administrator:
        await send_reply(message, "Sorry, only server admins can give me commands — happy to just chat though!", snark=False)
        return

    if action in DESTRUCTIVE_ACTIONS:
        # Kick/ban/delete pause for a typed yes/no from any admin —
        # confirm_and_execute sends its own prompt and waits before acting.
        # Purge is deliberately NOT in DESTRUCTIVE_ACTIONS, so it skips this
        # and runs immediately in the else branch below.
        try:
            reply = await confirm_and_execute(intent, message)
        except Exception as e:
            log.exception("Confirmation flow failed")
            reply = f"Something went wrong: {e}"
    else:
        async with message.channel.typing():
            try:
                reply = await execute_action(intent, message)
            except Exception as e:
                log.exception("Action failed")
                reply = f"Something went wrong: {e}"

    # Casual chat replies are already written with personality by the AI —
    # don't stack a canned snark line on top of a joke it already made.
    await send_reply(message, reply, snark=(action != "none"))

    # Remember this exchange for next time with this PERSON — cross-server,
    # cross-channel, and now cross-restart too, since it's saved to disk
    # right after (permanent taught vocabulary is separate — LEARNED_TERMS,
    # scoped per-server). This runs after the reply is already sent, so
    # saving to disk never adds delay to what the person actually sees.
    user_history.append({"role": "user", "content": content})
    user_history.append({"role": "assistant", "content": reply})
    del user_history[:-DISCORD_HISTORY_LIMIT]
    await asyncio.get_event_loop().run_in_executor(None, _save_conversation_history, discord_user_history)


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)