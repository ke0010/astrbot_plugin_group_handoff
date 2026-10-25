"""群聊轮流说话（握手）。

同一群里多个 AstrBot 实例（多个机器人）同时被同一句话唤醒时，
让其中一个先开口、另一个等它说完再开口，并且后开口的那个在组装 prompt 时
已经能看到前一个刚说的话——因为等待发生在 OnWaitingLLMRequestEvent，
而 prompt 是在那之后才组装的。

状态用一个共享目录里的 json 文件在实例间同步（同一台机器上的多个实例有效）。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import tempfile
import time
import uuid

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

PLUGIN_NAME = "astrbot_plugin_group_handoff"
_EXTRA_KEY = "_group_handoff_floor"


@register(
    PLUGIN_NAME,
    "柯（本地）",
    "群聊里多个机器人轮流说话：后开口的先等前一个说完，再带着对方刚说的话接上。",
    "0.1.0",
    "",
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

        shared_dir = str(self.config.get("shared_dir") or "").strip()
        self._state_dir = shared_dir or os.path.join(
            tempfile.gettempdir(), "astrbot_group_handoff"
        )
        try:
            os.makedirs(self._state_dir, exist_ok=True)
        except OSError as exc:
            logger.error(
                f"[group_handoff] 状态目录不可用：{self._state_dir} ({exc})，插件不生效"
            )
            self._enabled = False

        self._cleanup_stale()
        logger.info(
            f"[group_handoff] 已加载 me={self._me} 目录={self._state_dir} "
            f"白名单={sorted(self._groups) or '全部群'} 最长等待={self._max_wait}s"
        )

    # ------------------------------------------------------------------ 配置
    @staticmethod
    def _to_float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_groups(raw) -> set[str]:
        if not raw:
            return set()
        if isinstance(raw, str):
            raw = [part for part in raw.replace("，", ",").split(",")]
        return {str(item).strip() for item in raw if str(item).strip()}

    # -------------------------------------------------------------- 状态文件
    def _floor_path(self, group_key: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in group_key)
        return os.path.join(self._state_dir, f"floor_{safe}.json")

    def _read_floor(self, group_key: str) -> dict | None:
        path = self._floor_path(group_key)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            return None

    def _write_floor(self, group_key: str, owner: str, self_id: str) -> None:
        path = self._floor_path(group_key)
        payload = {"owner": owner, "self_id": self_id, "ts": time.time()}
        tmp = f"{path}.{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning(f"[group_handoff] 写状态失败：{path} ({exc})")
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _clear_floor(self, group_key: str) -> None:
        try:
            os.remove(self._floor_path(group_key))
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(f"[group_handoff] 清状态失败：{group_key} ({exc})")

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
            if not (name.startswith("floor_") and name.endswith(".json")):
                continue
            path = os.path.join(self._state_dir, name)
            try:
                if now - os.path.getmtime(path) > self._ttl * 2:
                    os.remove(path)
            except OSError:
                pass

    # ------------------------------------------------------------------ 钩子
    def _owner(self, event: AstrMessageEvent) -> str:
        """每个机器人自己的身份。

        用账号 self_id 而不是进程 id：同一个 AstrBot 进程里挂两个平台
        （柯莱、芙宁娜）时，两个平台的 self_id 不同，必须能区分开。
        """
        self_id = str(event.get_self_id() or "").strip()
        return self_id or self._me

    def _applicable(self, event: AstrMessageEvent) -> str:
        """返回群标识；不该生效时返回空串。"""
        if not self._enabled:
            return ""
        group_id = str(event.get_group_id() or "").strip()
        if not group_id:
            return ""  # 私聊不管
        if self._groups and group_id not in self._groups:
            return ""
        return f"{event.get_platform_name()}:{group_id}"

    @filter.on_waiting_llm_request()
    async def on_waiting_llm_request(self, event: AstrMessageEvent):
        """在获取会话锁、组装上下文之前决定：要不要让对方先说。"""
        try:
            await self._maybe_handoff(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("[group_handoff] 等待逻辑出错，按原样继续", exc_info=True)

    async def _maybe_handoff(self, event: AstrMessageEvent) -> None:
        group_key = self._applicable(event)
        if not group_key:
            return

        owner = self._owner(event)
        self_id = str(event.get_self_id() or "")
        deadline = time.monotonic() + self._max_wait
        waited = False
        claimed = False

        while True:
            floor = self._read_floor(group_key)
            holder = (floor or {}).get("owner")
            if self._is_fresh(floor) and holder and holder != owner:
                # 别的机器人在说：等它说完
                if time.monotonic() >= deadline:
                    logger.info(
                        f"[group_handoff] 等满 {self._max_wait}s，{group_key} 仍被占用，先开口"
                    )
                    break
                waited = True
                await asyncio.sleep(self._poll)
                continue

            # 没人占位 / 标记过期 / 就是自己：试着占位，再回读确认
            # （两边同时被触发时都写一次，谁后写谁拿到；没拿到的下一轮进等待分支）
            self._write_floor(group_key, owner, self_id)
            await asyncio.sleep(0.02)
            current = self._read_floor(group_key)
            if (current or {}).get("owner") == owner:
                claimed = True
                break

        if not claimed:
            return

        if waited:
            gap = random.uniform(self._gap_min, self._gap_max)
            await asyncio.sleep(gap)
            logger.info(f"[group_handoff] {group_key} 等对方说完，再停 {gap:.1f}s 后开口")

        event.set_extra(_EXTRA_KEY, group_key)

    @filter.after_message_sent()
    async def on_after_message_sent(self, event: AstrMessageEvent):
        """说完了，把发言权还回去。"""
        try:
            group_key = event.get_extra(_EXTRA_KEY)
            if not group_key:
                return
            floor = self._read_floor(group_key)
            if (floor or {}).get("owner") == self._owner(event):
                self._clear_floor(group_key)
        except Exception:
            logger.warning("[group_handoff] 释放发言权出错", exc_info=True)
