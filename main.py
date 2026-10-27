"""群聊调度（接力 + 离场 + 点名 + 用词注入）。

四个功能，都只在群聊生效：
1. 接力：同一群里多个机器人同时被唤醒时，一个先开口，另一个等它说完再开口，
   并且后开口的那个在组装 prompt 时已经能看到前一个刚说的话。
2. 离场：主人口令让某个机器人「离场」后，它在群里完全不出声，直到口令叫它回来。
3. 点名：主人点名谁，谁接话；没被点到的闭嘴；都没点就由最近说过话的那个接。
4. 用词注入：判定进入亲密场景后，把该机器人自己私库 08 里的词表段，
   追加进本轮 LLM 请求的 system prompt（不复制文件，运行时读取）。

状态用共享目录里的 json 文件在实例之间同步（同一台机器上的多个实例有效）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import tempfile
import time
import uuid

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

PLUGIN_NAME = "astrbot_plugin_group_handoff"
_EXTRA_FLOOR = "_group_handoff_floor"
_EXTRA_SCENE = "_group_handoff_scene"

DEFAULT_MASTER_IDS = []
DEFAULT_BOT_NAMES = "default=柯莱;furina=芙宁娜"
DEFAULT_BOT_ALIASES = (
    "default=柯莱,小柯莱;furina=芙宁娜,大明星,芙芙,水神"
)
DEFAULT_VOCAB_FILES = ""

DEFAULT_STRONG_WORDS = [
    "内裤", "脱掉", "裸", "含住", "舔", "抽插", "没入", "插入", "射", "高潮",
    "爱液", "湿透", "龟头", "阴道", "阴蒂", "阴唇", "肉棒", "精液", "勃起",
    "呻吟", "闷哼", "水声",
]
DEFAULT_WEAK_WORDS = [
    "抱", "亲", "摸", "贴", "吻", "床", "被窝", "喘", "腰", "腿", "热", "紧",
    "胸口", "贴上来",
]
DEFAULT_END_WORDS = [
    "穿衣服", "起来", "天亮了", "起身", "洗漱", "去做饭", "出门", "收拾",
]
DEFAULT_GROUP_WORDS = ["你们", "你们俩", "你们两个", "你俩"]
DEFAULT_PRESENCE_ENTER = ["离场", "离开"]
DEFAULT_PRESENCE_EXIT = ["回来", "回来了", "回归"]


@register(
    PLUGIN_NAME,
    "柯",
    "群聊调度：机器人轮流说话（后开口的先等前一个说完）、离场静默、点名接话、按场景注入自备用词表。",
    "0.2.0",
    "https://github.com/ke0010/astrbot_plugin_group_handoff",
)
class GroupHandoffPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | dict | None = None):
        super().__init__(context)
        self.config = config or {}

        self._enabled = bool(self.config.get("enabled", True))
        self._groups = self._parse_groups(self.config.get("group_whitelist"))
        self._max_wait = self._to_float(self.config.get("max_wait_seconds"), 20.0)
        self._ttl = self._to_float(self.config.get("floor_ttl_seconds"), 120.0)
        self._poll = max(0.05, self._to_float(self.config.get("poll_interval_seconds"), 0.3))
        self._gap_min = max(0.0, self._to_float(self.config.get("gap_min_seconds"), 1.0))
        self._gap_max = max(self._gap_min, self._to_float(self.config.get("gap_max_seconds"), 2.5))
        self._me = uuid.uuid4().hex[:8]

        # ---- 身份与名字 ----
        self._master_ids = {
            str(item).strip()
            for item in (self.config.get("master_ids") or DEFAULT_MASTER_IDS)
            if str(item).strip()
        }
        self._bot_names = {
            key: (values[0] if values else key)
            for key, values in self._parse_map(
                self.config.get("bot_names"), DEFAULT_BOT_NAMES
            ).items()
        }
        self._bot_aliases = self._parse_aliases(
            self.config.get("bot_aliases"), DEFAULT_BOT_ALIASES
        )

        # ---- 离场 ----
        self._presence_enable = bool(self.config.get("presence_enable", True))
        self._presence_enter = self._to_list(
            self.config.get("presence_enter_words"), DEFAULT_PRESENCE_ENTER
        )
        self._presence_exit = self._to_list(
            self.config.get("presence_exit_words"), DEFAULT_PRESENCE_EXIT
        )

        # ---- 点名接话 ----
        self._mention_enable = bool(self.config.get("mention_enable", True))
        self._group_words = self._to_list(self.config.get("mention_group_words"), DEFAULT_GROUP_WORDS)

        # ---- 亲密场景用词注入 ----
        self._scene_enable = bool(self.config.get("scene_enable", True))
        self._strong_words = self._to_list(self.config.get("scene_strong_words"), DEFAULT_STRONG_WORDS)
        self._weak_words = self._to_list(self.config.get("scene_weak_words"), DEFAULT_WEAK_WORDS)
        self._end_words = self._to_list(self.config.get("scene_end_words"), DEFAULT_END_WORDS)
        self._weak_threshold = max(2, int(self._to_float(self.config.get("scene_weak_threshold"), 3)))
        self._weak_streak_need = max(2, int(self._to_float(self.config.get("scene_weak_streak"), 3)))
        self._end_grace = max(1, int(self._to_float(self.config.get("scene_end_grace_rounds"), 3)))
        self._vocab_files = self._parse_map(self.config.get("scene_vocab_files"), DEFAULT_VOCAB_FILES)
        self._vocab_heading = str(self.config.get("scene_vocab_heading") or "词表")
        self._inject_prefix = str(
            self.config.get("scene_inject_prefix")
            or "【本轮用词参考】下面是你自己那张用词表。说到身体和动作时优先用这里的词，"
            "要具体、直白，不用含糊代称，也不用比喻绕过去："
        )
        self._vocab_cache: dict[str, tuple[float, str]] = {}

        shared_dir = str(self.config.get("shared_dir") or "").strip()
        self._state_dir = shared_dir or os.path.join(tempfile.gettempdir(), "astrbot_group_handoff")
        try:
            os.makedirs(self._state_dir, exist_ok=True)
        except OSError as exc:
            logger.error(f"[group_handoff] 状态目录不可用：{self._state_dir} ({exc})，插件不生效")
            self._enabled = False

        self._cleanup_stale()
        logger.info(
            f"[group_handoff] 已加载 me={self._me} 目录={self._state_dir} "
            f"白名单={sorted(self._groups) or '全部群'} 离场={self._presence_enable} "
            f"点名={self._mention_enable} 用词注入={self._scene_enable}"
        )

    # ------------------------------------------------------------------ 配置
    @staticmethod
    def _to_float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_list(raw, default: list[str]) -> list[str]:
        if not raw:
            return list(default)
        if isinstance(raw, str):
            raw = re.split(r"[,，;；\n]", raw)
        out = [str(item).strip() for item in raw if str(item).strip()]
        return out or list(default)

    @staticmethod
    def _parse_groups(raw) -> set[str]:
        if not raw:
            return set()
        if isinstance(raw, str):
            raw = raw.replace("，", ",").split(",")
        return {str(item).strip() for item in raw if str(item).strip()}

    @staticmethod
    def _parse_map(raw, default: str, value_sep: str = ",") -> dict[str, list[str]]:
        """解析 "key=v1,v2;key2=v3" 形式的映射。"""
        if isinstance(raw, (list, tuple)):
            text = ";".join(str(item) for item in raw)
        else:
            text = str(raw).strip() if raw else default
        result: dict[str, list[str]] = {}
        for chunk in re.split(r"[;；\n]", text):
            chunk = chunk.strip()
            if not chunk or "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            key = key.strip()
            if not key:
                continue
            result[key] = [
                part.strip() for part in re.split(value_sep, value) if part.strip()
            ]
        return result

    @classmethod
    def _parse_aliases(cls, raw, default: str) -> dict[str, list[str]]:
        return cls._parse_map(raw, default, value_sep=r"[,，]")

    # -------------------------------------------------------------- 状态文件
    @staticmethod
    def _safe(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)

    def _floor_path(self, group_key: str) -> str:
        return os.path.join(self._state_dir, f"floor_{self._safe(group_key)}.json")

    def _state_path(self, group_key: str) -> str:
        return os.path.join(self._state_dir, f"state_{self._safe(group_key)}.json")

    def _load(self, path: str) -> dict | None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            return None

    @staticmethod
    def _atomic_write(path: str, payload: dict) -> None:
        tmp = f"{path}.{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def _read_floor(self, group_key: str) -> dict | None:
        return self._load(self._floor_path(group_key))

    def _write_floor(self, group_key: str, owner: str, self_id: str) -> None:
        try:
            self._atomic_write(
                self._floor_path(group_key),
                {"owner": owner, "self_id": self_id, "ts": time.time()},
            )
        except OSError as exc:
            logger.warning(f"[group_handoff] 写状态失败：{group_key} ({exc})")

    def _clear_floor(self, group_key: str) -> None:
        try:
            os.remove(self._floor_path(group_key))
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(f"[group_handoff] 清状态失败：{group_key} ({exc})")

    def _read_state(self, group_key: str) -> dict:
        data = self._load(self._state_path(group_key)) or {}
        data.setdefault("absent", {})
        data.setdefault("last_speaker", "")
        scene = data.setdefault("scene", {})
        scene.setdefault("fp", "")
        scene.setdefault("in_scene", False)
        scene.setdefault("countdown", 0)
        scene.setdefault("weak_streak", 0)
        return data

    def _write_state(self, group_key: str, state: dict) -> None:
        try:
            self._atomic_write(self._state_path(group_key), state)
        except OSError as exc:
            logger.warning(f"[group_handoff] 写场景状态失败：{group_key} ({exc})")

    def _is_fresh(self, floor: dict | None) -> bool:
        if not floor:
            return False
        try:
            ts = float(floor.get("ts", 0))
        except (TypeError, ValueError):
            return False
        return (time.time() - ts) < self._ttl

    def _cleanup_stale(self) -> None:
        try:
            names = os.listdir(self._state_dir)
        except OSError:
            return
        now = time.time()
        for name in names:
            if not (name.endswith(".json") and (name.startswith("floor_") or name.startswith("state_"))):
                continue
            path = os.path.join(self._state_dir, name)
            try:
                if now - os.path.getmtime(path) > 86400 * 3:
                    os.remove(path)
            except OSError:
                pass

    # ------------------------------------------------------------------ 钩子
    def _owner(self, event: AstrMessageEvent) -> str:
        """每个机器人自己的身份：同进程挂多个平台时靠 self_id 区分。"""
        self_id = str(event.get_self_id() or "").strip()
        return self_id or self._me

    def _platform_id(self, event: AstrMessageEvent) -> str:
        try:
            pid = str(event.get_platform_id() or "").strip()
        except AttributeError:
            pid = ""
        return pid or str(event.get_platform_name() or "")

    def _applicable(self, event: AstrMessageEvent) -> str:
        """返回群号（两个机器人在同一个群里必须用同一个标识）；不该生效时返回空串。"""
        if not self._enabled:
            return ""
        group_id = str(event.get_group_id() or "").strip()
        if not group_id:
            return ""
        if self._groups and group_id not in self._groups:
            return ""
        return group_id

    def _text(self, event: AstrMessageEvent) -> str:
        try:
            return str(event.get_message_str() or "")
        except Exception:
            return ""

    def _is_master(self, event: AstrMessageEvent) -> bool:
        sender = str(event.get_sender_id() or "").strip()
        if self._master_ids:
            return sender in self._master_ids
        return bool(sender)

    def _msg_fp(self, event: AstrMessageEvent, text: str) -> str:
        raw = getattr(getattr(event, "message_obj", None), "message_id", "") or ""
        if raw:
            return f"id:{raw}"
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        return f"t:{int(time.time())}:{digest}"

    @filter.on_waiting_llm_request()
    async def on_waiting_llm_request(self, event: AstrMessageEvent):
        """在获取会话锁、组装上下文之前决定：要不要出声、要不要让对方先说。"""
        try:
            await self._dispatch(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("[group_handoff] 判断逻辑出错，按原样继续", exc_info=True)

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """亲密场景：把该机器人自己那份用词表追加进 system prompt。"""
        try:
            group_key = event.get_extra(_EXTRA_SCENE)
            if not group_key:
                return
            vocab = self._vocab_text(self._platform_id(event))
            if not vocab:
                return
            req.system_prompt = (
                f"{req.system_prompt or ''}\n\n{self._inject_prefix}\n{vocab}"
            )
            logger.info(f"[group_handoff] 已注入用词表：{group_key} ({len(vocab)} 字)")
        except Exception:
            logger.warning("[group_handoff] 注入用词表出错", exc_info=True)

    async def _dispatch(self, event: AstrMessageEvent) -> None:
        group_key = self._applicable(event)
        if not group_key:
            return
        platform_id = self._platform_id(event)
        text = self._text(event)
        master = self._is_master(event)
        state = self._read_state(group_key)

        if master:
            self._apply_commands(state, text)
            self._update_scene(state, text, self._msg_fp(event, text))
            self._write_state(group_key, state)

        # ---- 离场静默 ----
        if self._presence_enable and platform_id in state.get("absent", {}):
            logger.info(f"[group_handoff] {platform_id} 离场中，{group_key} 不出声")
            event.stop_event()
            return

        # ---- 点名接话 ----
        if master and self._mention_enable and not self._is_called(event, state, text, platform_id):
            logger.info(f"[group_handoff] 没被点到，{platform_id} 不出声（{group_key}）")
            event.stop_event()
            return

        # ---- 接力 ----
        if await self._maybe_handoff(event, group_key):
            # 记下「最后说话的是谁」，下一次没点名时由它接
            state = self._read_state(group_key)
            state["last_speaker"] = platform_id
            self._write_state(group_key, state)
            if state.get("scene", {}).get("in_scene"):
                event.set_extra(_EXTRA_SCENE, group_key)

    # ---------------------------------------------------------- 离场与点名
    def _apply_commands(self, state: dict, text: str) -> None:
        if not self._presence_enable or not text:
            return
        absent: dict = state.setdefault("absent", {})
        for platform_id, aliases in self._bot_aliases.items():
            names = list(aliases) + ([self._bot_names.get(platform_id, "")] if self._bot_names else [])
            hit = any(name and name in text for name in names)
            if not hit:
                continue
            if any(word in text for word in self._presence_enter):
                if platform_id not in absent:
                    absent[platform_id] = {"since": time.time()}
                    logger.info(f"[group_handoff] {self._bot_names.get(platform_id, platform_id)} 离场")
            elif any(word in text for word in self._presence_exit):
                if absent.pop(platform_id, None):
                    logger.info(f"[group_handoff] {self._bot_names.get(platform_id, platform_id)} 回来")

    def _at_self(self, event: AstrMessageEvent) -> bool:
        self_id = str(event.get_self_id() or "")
        if not self_id:
            return False
        chain = getattr(getattr(event, "message_obj", None), "message", None) or []
        try:
            for comp in chain:
                qq = getattr(comp, "qq", None)
                if qq is not None and str(qq) == self_id:
                    return True
        except TypeError:
            return False
        return False

    def _is_called(
        self,
        event: AstrMessageEvent,
        state: dict,
        text: str,
        platform_id: str,
    ) -> bool:
        if self._at_self(event):
            return True
        if not text:
            return True
        if any(word in text for word in self._group_words):
            return True

        called: set[str] = set()
        for pid, aliases in self._bot_aliases.items():
            for name in aliases:
                if name and self._name_is_call(text, name):
                    called.add(pid)
                    break
        if not called:
            last = str(state.get("last_speaker") or "")
            return platform_id == last if last else True
        return platform_id in called

    @staticmethod
    def _name_is_call(text: str, name: str) -> bool:
        """名字算「点名」的三种位置：句首、称呼位（前面是标点/@）、后面紧跟呼语。"""
        start = 0
        while True:
            idx = text.find(name, start)
            if idx < 0:
                return False
            if idx == 0:
                return True
            prev = text[idx - 1]
            nxt = text[idx + len(name)] if idx + len(name) < len(text) else ""
            if prev in "@，,、。！？:： \n":
                return True
            if nxt in "，,、！？啊呢呀吧哦嘛":
                return True
            start = idx + len(name)

    # ------------------------------------------------------------ 亲密场景
    def _match_strong(self, text: str) -> list[str]:
        hits = []
        for word in self._strong_words:
            if word not in text:
                continue
            if word == "湿透" and any(extra in text for extra in ("雨", "淋")):
                continue
            if word == "射" and any(extra in text for extra in ("箭", "弓", "射门")):
                continue
            if word == "裸" and any(extra in text for extra in ("裸眼", "裸机", "裸辞")):
                continue
            hits.append(word)
        return hits

    def _match_weak(self, text: str) -> list[str]:
        hits = []
        for word in self._weak_words:
            if word not in text:
                continue
            if word == "紧" and any(extra in text for extra in ("紧张", "要紧", "赶紧", "时间紧")):
                continue
            if word == "热" and not any(
                extra in text for extra in ("身体", "身", "腿", "腰", "脸", "皮肤", "下面", "里头", "胸口")
            ):
                continue
            hits.append(word)
        return hits

    def _update_scene(self, state: dict, text: str, fp: str) -> None:
        if not self._scene_enable or not text:
            return
        scene = state.setdefault("scene", {})
        if scene.get("fp") == fp:
            return
        scene["fp"] = fp

        strong = self._match_strong(text)
        weak = self._match_weak(text)
        into_scene = bool(strong) or (
            len(weak) >= self._weak_threshold and len(set(weak)) >= 2
        )

        if scene.get("in_scene"):
            # 场景进行中
            if strong:
                scene["countdown"] = 0
                scene["weak_streak"] = 0
                logger.info(f"[group_handoff] 场景触发词（强）：{'、'.join(strong[:3])}")
                return
            countdown = int(scene.get("countdown") or 0)
            if countdown > 0:
                countdown -= 1
                scene["countdown"] = countdown
                if countdown <= 0:
                    scene["in_scene"] = False
                    scene["weak_streak"] = 0
                    logger.info("[group_handoff] 收尾延迟结束，退出场景")
                return
            if any(word in text for word in self._end_words):
                scene["countdown"] = self._end_grace + 1
                logger.info(f"[group_handoff] 命中收尾词，再延续 {self._end_grace} 轮")
                return
            if weak:
                scene["weak_streak"] = 0
            return

        # 还没进场景
        if into_scene:
            scene["in_scene"] = True
            scene["countdown"] = 0
            scene["weak_streak"] = 0
            why = "强" if strong else "弱词≥阈值"
            logger.info(f"[group_handoff] 进入亲密场景（{why}）：{'、'.join((strong or weak)[:3])}")
            return
        if weak:
            scene["weak_streak"] = int(scene.get("weak_streak") or 0) + 1
            if scene["weak_streak"] >= self._weak_streak_need:
                scene["in_scene"] = True
                scene["weak_streak"] = 0
                logger.info(f"[group_handoff] 进入亲密场景（连续 {self._weak_streak_need} 轮弱词）")
        else:
            scene["weak_streak"] = 0

    def _vocab_text(self, platform_id: str) -> str:
        path = (self._vocab_files.get(platform_id) or [""])[0]
        if not path:
            return ""
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return ""
        cached = self._vocab_cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except OSError:
            return ""
        collected: list[str] = []
        inside = False
        for line in lines:
            if line.startswith("## "):
                inside = self._vocab_heading in line
                continue
            if line.startswith("### ") and inside:
                collected.append("")
                continue
            if inside:
                collected.append(line)
        text = "\n".join(collected).strip()
        if not text:
            return ""
        self._vocab_cache[path] = (mtime, text)
        return text

    # ------------------------------------------------------------------ 接力
    async def _maybe_handoff(self, event: AstrMessageEvent, group_key: str) -> bool:
        owner = self._owner(event)
        self_id = str(event.get_self_id() or "")
        deadline = time.monotonic() + self._max_wait
        waited = False
        claimed = False

        while True:
            floor = self._read_floor(group_key)
            holder = (floor or {}).get("owner")
            if self._is_fresh(floor) and holder and holder != owner:
                if time.monotonic() >= deadline:
                    logger.info(f"[group_handoff] 等满 {self._max_wait}s，{group_key} 仍被占用，先开口")
                    break
                waited = True
                await asyncio.sleep(self._poll)
                continue

            self._write_floor(group_key, owner, self_id)
            await asyncio.sleep(0.02)
            current = self._read_floor(group_key)
            if (current or {}).get("owner") == owner:
                claimed = True
                break

        if not claimed:
            return False

        if waited:
            gap = random.uniform(self._gap_min, self._gap_max)
            await asyncio.sleep(gap)
            logger.info(f"[group_handoff] {group_key} 等对方说完，再停 {gap:.1f}s 后开口")

        event.set_extra(_EXTRA_FLOOR, group_key)
        return True

    @filter.after_message_sent()
    async def on_after_message_sent(self, event: AstrMessageEvent):
        """说完了：释放发言权，并记住最后说话的是谁。"""
        try:
            group_key = event.get_extra(_EXTRA_FLOOR)
            if not group_key:
                return
            floor = self._read_floor(group_key)
            if (floor or {}).get("owner") == self._owner(event):
                self._clear_floor(group_key)
            state = self._read_state(group_key)
            state["last_speaker"] = self._platform_id(event)
            self._write_state(group_key, state)
        except Exception:
            logger.warning("[group_handoff] 释放发言权出错", exc_info=True)
