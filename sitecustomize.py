"""Runtime patches for SkillsBench launches.

This module is loaded automatically by Python when it is on ``PYTHONPATH``.
The ACP-result collector is opt-in for Phase-2 runs; the no-timeout patch is
independently controlled by ``SKILLSBENCH_DISABLE_TIMEOUTS=1``.  Normal local
scripts remain unaffected.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any


_phase2_capture_marker = ""
_phase2_batch_root = os.environ.get("PHASE2_BATCH_ROOT", "").strip()
if _phase2_batch_root:
    _phase2_capture_marker = str(Path(_phase2_batch_root) / ".capture_acp_results")
_phase2_capture_enabled = (
    os.environ.get("PHASE2_CAPTURE_ACP_RESULTS") == "1"
    or bool(_phase2_capture_marker and Path(_phase2_capture_marker).is_file())
)


def _install_phase2_acp_capture_fallback() -> None:
    """Install ACP result capture independently of the timeout patch.

    ``sitecustomize`` is imported before BenchFlow starts.  In some virtualenv
    layouts the optional timeout imports are not available at that moment,
    while ``benchflow.acp.session`` is importable a little later.  Keeping this
    collector independent prevents a missing timeout dependency from silently
    turning every tool result into ``content: []``.  The richer in-branch
    collector below remains the primary implementation; the marker makes the
    two paths idempotent.
    """
    if not _phase2_capture_enabled:
        return
    try:
        from benchflow.acp.session import ACPSession
    except Exception:
        return
    if getattr(ACPSession, "_phase2_capture_installed", False):
        return

    original_handle_update = ACPSession.handle_update

    def capture_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return "\n".join(
                part for part in (capture_text(item) for item in value) if part
            )
        if isinstance(value, dict):
            for key in (
                "formatted_output",
                "aggregated_output",
                "stdout",
                "stderr",
                "output",
                "result",
                "error",
                "data",
                "text",
            ):
                if key in value and value.get(key) not in (None, "", [], {}):
                    rendered = capture_text(value.get(key))
                    if rendered:
                        return rendered
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    def output_values(value: Any) -> list[str]:
        if value in (None, "", [], {}):
            return []
        if isinstance(value, (str, int, float, bool)):
            rendered = capture_text(value)
            return [rendered] if rendered else []
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                out.extend(output_values(item))
            return out
        if not isinstance(value, dict):
            rendered = capture_text(value)
            return [rendered] if rendered else []
        out: list[str] = []
        for key in (
            "formatted_output",
            "aggregated_output",
            "terminal_output",
            "command_output",
            "stdout",
            "stderr",
            "output",
            "result",
            "error",
            "data",
            "text",
            "content",
        ):
            if key in value:
                out.extend(output_values(value.get(key)))
        return out

    def append_output(record: Any, rendered: str) -> None:
        if not rendered:
            return
        # BenchFlow already stores ordinary ``content`` blocks.  The shim is
        # needed for ACP adapters that put the real terminal text in
        # ``rawOutput`` or metadata; do not append a second copy when both
        # representations are present.
        existing_blocks = getattr(record, "content", [])
        existing_text = "\n".join(
            str(block.get("text", ""))
            for block in existing_blocks
            if isinstance(block, dict) and block.get("text")
        )
        if rendered == existing_text or rendered in existing_text:
            return
        block = getattr(record, "_phase2_result_block", None)
        if block is None:
            block = {"type": "text", "text": ""}
            record.content.append(block)
            record._phase2_result_block = block
        current = str(block.get("text", ""))
        if rendered == current or rendered in current:
            return
        if current and rendered.startswith(current):
            block["text"] = rendered
        else:
            block["text"] = current + rendered

    def capture_update(self: Any, update: dict[str, Any]) -> None:
        original_handle_update(self, update)
        if not isinstance(update, dict):
            return
        tool_id = str(update.get("toolCallId", "") or "")
        if not tool_id:
            return
        tool_map = getattr(self, "_tool_call_map", {})
        record = tool_map.get(tool_id)
        if record is None:
            for key, candidate in tool_map.items():
                if str(key) == tool_id:
                    record = candidate
                    break
        if record is None:
            return

        values: list[str] = []
        # ``content`` is intentionally excluded: the stock ACP handler has
        # already copied it into ``record.content``.  Reading it here would
        # duplicate every normal result in the trajectory.
        for key in ("rawOutput", "raw_output", "result", "output"):
            values.extend(output_values(update.get(key)))
        meta = update.get("_meta")
        if isinstance(meta, dict):
            for key, value in meta.items():
                lowered = str(key).lower()
                if any(token in lowered for token in ("output", "stdout", "stderr", "result")):
                    values.extend(output_values(value))
        # Preserve streaming terminal deltas in the order received.  A later
        # aggregate is replaced only when it is the same prefix, avoiding
        # duplicate completion text.
        seen: set[str] = set()
        for value in values:
            if value and value not in seen:
                seen.add(value)
                append_output(record, value)

    ACPSession.handle_update = capture_update
    ACPSession._phase2_capture_installed = True


if os.environ.get("SKILLSBENCH_DISABLE_TIMEOUTS") == "1":
    try:
        import asyncio

        import benchflow.rollout as rollout
        from benchflow.sandbox.daytona import DaytonaSandbox
        from benchflow.sandbox.lockdown import _build_cleanup_cmd, harden_before_verify
        from benchflow.task import Verifier
    except Exception:
        # If BenchFlow is not available yet, leave the process unchanged.
        pass
    else:
        def _no_timeout(value: Any) -> Any:
            return None if value in (0, 0.0) else value

        original_sandbox_exec = DaytonaSandbox._sandbox_exec
        original_create_sandbox = DaytonaSandbox._create_sandbox

        async def _sandbox_exec_no_timeout(
            self,
            command: str,
            cwd: str | None = None,
            env: dict[str, str] | None = None,
            timeout_sec: int | None = None,
            shell: str = "bash -c",
            user: str | int | None = None,
        ):
            return await original_sandbox_exec(
                self,
                command,
                cwd=cwd,
                env=env,
                timeout_sec=_no_timeout(timeout_sec),
                shell=shell,
                user=user,
            )

        async def _create_sandbox_no_timeout(self, params):
            if not self._client_manager:
                raise RuntimeError(
                    "Client manager not initialized. This should never happen."
                )

            daytona = await self._client_manager.get_client()
            create_task = asyncio.ensure_future(
                daytona.create(
                    params=params,
                    timeout=_no_timeout(round(self.task_env_config.build_timeout_sec)),
                )
            )
            try:
                self._sandbox = await asyncio.shield(create_task)
            except asyncio.CancelledError:
                try:
                    self._sandbox = await asyncio.wait_for(create_task, timeout=30)
                except (TimeoutError, asyncio.CancelledError, Exception):
                    create_task.cancel()
                raise

        async def _verify_rollout_no_timeout(
            env: Any,
            task: Any,
            rollout_paths: Any,
            timing: dict,
            sandbox_user: str | None = None,
            workspace: str | None = None,
        ) -> tuple[dict | None, str | None]:
            rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
            await harden_before_verify(env, task, sandbox_user, workspace=workspace)
            rollout_logger = getattr(rollout, "logger", None)
            if rollout_logger is not None:
                rollout_logger.info("Running verifier...")
            t0 = datetime.now()
            verifier_error = None
            try:
                verifier = Verifier(task=task, rollout_paths=rollout_paths, sandbox=env)
                verifier_result = await verifier.verify()
                timing["verifier"] = (datetime.now() - t0).total_seconds()
                rewards = verifier_result.rewards
                if rollout_logger is not None:
                    rollout_logger.info(f"Rewards: {rewards}")
            except Exception as e:
                timing["verifier"] = (datetime.now() - t0).total_seconds()
                verifier_error = f"verifier crashed: {e}"
                rewards = None
                if rollout_logger is not None:
                    rollout_logger.error(verifier_error)
            return rewards, verifier_error

        async def _soft_verify_no_timeout(self) -> tuple[dict | None, str | None, str | None]:
            from benchflow.sandbox.lockdown import _build_cleanup_cmd, _read_hardening_config
            from benchflow.task import Verifier

            self._rollout_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
            await self._env.exec(
                "rm -rf /logs/verifier && mkdir -p /logs/verifier /app && chmod 777 /logs/verifier",
                user="root",
                timeout_sec=0,
            )
            hardening = _read_hardening_config(getattr(self._task, "task_dir", None))
            await self._env.exec(
                _build_cleanup_cmd(hardening),
                user="root",
                timeout_sec=0,
            )

            rewards = None
            verifier_output = None
            verifier_error = None
            try:
                verifier = Verifier(
                    task=self._task,
                    rollout_paths=self._rollout_paths,
                    sandbox=self._env,
                )
                verifier_result = await verifier.verify()
                rewards = verifier_result.rewards
                cat = await self._env.exec(
                    "cat /logs/verifier/*.log 2>/dev/null || "
                    "cat /logs/verifier/output.txt 2>/dev/null || true",
                    timeout_sec=0,
                )
                verifier_output = (cat.stdout or "").strip() or None
            except Exception as e:
                verifier_error = f"soft verifier crashed: {e}"
            return rewards, verifier_output, verifier_error

        DaytonaSandbox._sandbox_exec = _sandbox_exec_no_timeout
        DaytonaSandbox._create_sandbox = _create_sandbox_no_timeout
        rollout._verify_rollout = _verify_rollout_no_timeout
        rollout.Rollout.soft_verify = _soft_verify_no_timeout

        # Codex ACP reports command output in ``rawOutput`` and incremental
        # terminal deltas under ``_meta``.  Older BenchFlow releases only
        # copied the update's ``content`` field, which is usually just a
        # terminal handle; consequently every tool-call result in
        # trajectory/acp_trajectory.jsonl was recorded as ``[]``.  Enable the
        # small compatibility shim only for Phase-2 runs that explicitly ask
        # for complete execution memory.  Baseline runs keep the stock
        # collector and are not affected.
        if _phase2_capture_enabled:
            try:
                from benchflow.acp.session import ACPSession
            except Exception:
                ACPSession = None  # type: ignore[assignment]
            if ACPSession is not None:
                _original_handle_update = ACPSession.handle_update

                def _capture_text(value: Any) -> str:
                    if value is None:
                        return ""
                    if isinstance(value, str):
                        return value
                    if isinstance(value, (int, float, bool)):
                        return str(value)
                    if isinstance(value, list):
                        chunks = [_capture_text(item) for item in value]
                        return "\n".join(chunk for chunk in chunks if chunk)
                    if isinstance(value, dict):
                        # ACP command completion uses formatted_output; MCP
                        # calls use result/error.  Prefer the human-readable
                        # fields and retain structured values when necessary.
                        for key in (
                            "formatted_output",
                            "stdout",
                            "stderr",
                            "output",
                            "result",
                            "error",
                            "data",
                            "text",
                        ):
                            if key in value and value.get(key) not in (None, "", [], {}):
                                rendered = _capture_text(value.get(key))
                                if rendered:
                                    return rendered
                        try:
                            return json.dumps(value, ensure_ascii=False)
                        except Exception:
                            return str(value)
                    return str(value)

                def _append_tool_output(record: Any, text: str) -> None:
                    if not text:
                        return
                    block = getattr(record, "_phase2_result_block", None)
                    if block is None:
                        block = {"type": "text", "text": ""}
                        record.content.append(block)
                        record._phase2_result_block = block
                    block["text"] += text

                def _output_candidates(value: Any) -> list[str]:
                    """Extract command/MCP output from ACP's version variants.

                    ACP 0.0.45 normally uses ``rawOutput``.  Older/newer
                    adapters have used ``aggregated_output``, ``stdout`` or
                    a nested ``result`` object instead.  Walk only output
                    named fields; never serialize the complete update because
                    that would leak protocol metadata and tool arguments into
                    the result text.
                    """
                    if value is None:
                        return []
                    if isinstance(value, str):
                        return [value] if value else []
                    if isinstance(value, (int, float, bool)):
                        return [str(value)]
                    if isinstance(value, list):
                        out: list[str] = []
                        for item in value:
                            out.extend(_output_candidates(item))
                        return out
                    if not isinstance(value, dict):
                        return [str(value)]
                    # Prefer the fields that ACP implementations use for
                    # completed terminal/MCP output.  ``content`` is handled
                    # too, but empty content must not suppress rawOutput.
                    names = (
                        "formatted_output",
                        "aggregated_output",
                        "terminal_output",
                        "command_output",
                        "stdout",
                        "stderr",
                        "output",
                        "result",
                        "error",
                        "data",
                        "text",
                        "content",
                    )
                    out: list[str] = []
                    for name in names:
                        if name not in value or value.get(name) in (None, "", [], {}):
                            continue
                        out.extend(_output_candidates(value.get(name)))
                    return out

                def _capture_acp_update(self: Any, update: dict[str, Any]) -> None:
                    _original_handle_update(self, update)
                    if not isinstance(update, dict):
                        return
                    tool_id = str(update.get("toolCallId", "") or "")
                    if not tool_id:
                        return
                    tool_map = getattr(self, "_tool_call_map", {})
                    record = tool_map.get(tool_id)
                    if record is None:
                        # A few ACP adapters keep numeric IDs internally while
                        # serializing them as strings on the wire.
                        for key, candidate in tool_map.items():
                            if str(key) == tool_id:
                                record = candidate
                                break
                    if record is None:
                        return

                    meta = update.get("_meta")
                    if not isinstance(meta, dict):
                        meta = {}
                    delta_text = ""
                    for key in (
                        "terminal_output_delta",
                        "mcp_output_delta",
                        "terminal_output",
                        "mcp_output",
                        "aggregated_output",
                    ):
                        payload = meta.get(key)
                        if isinstance(payload, dict):
                            delta_text += _capture_text(payload.get("data"))
                        elif payload is not None:
                            delta_text += _capture_text(payload)
                    # Some adapters place a complete result under a custom
                    # meta key instead of rawOutput.  Only inspect keys whose
                    # names clearly denote output; protocol bookkeeping is
                    # intentionally ignored.
                    meta_values: list[str] = []
                    for key, payload in meta.items():
                        key_text = str(key).lower()
                        if any(token in key_text for token in ("output", "stdout", "stderr", "result")):
                            meta_values.extend(_output_candidates(payload))
                    if delta_text:
                        _append_tool_output(record, delta_text)
                        record._phase2_has_output_delta = True

                    raw_values: list[str] = []
                    for key in ("rawOutput", "raw_output", "result", "output", "content"):
                        value = update.get(key)
                        if value in (None, "", [], {}):
                            continue
                        raw_values.extend(_output_candidates(value))
                    raw_values.extend(meta_values)
                    if raw_values:
                        # Completion notifications often repeat the aggregate
                        # after deltas.  Preserve order but avoid appending the
                        # exact same payload twice.
                        unique_values: list[str] = []
                        seen_values: set[str] = set()
                        for value in raw_values:
                            if value and value not in seen_values:
                                seen_values.add(value)
                                unique_values.append(value)
                        rendered = "\n".join(unique_values)
                        block = getattr(record, "_phase2_result_block", None)
                        current = str(block.get("text", "")) if block is not None else ""
                        if rendered and rendered != current and rendered not in current:
                            if current and rendered.startswith(current):
                                block["text"] = rendered
                            else:
                                _append_tool_output(record, rendered)

                ACPSession.handle_update = _capture_acp_update
                ACPSession._phase2_capture_installed = True


# Do not make ACP capture contingent on the optional timeout imports above.
# When the timeout patch cannot load, this fallback still runs at module end;
# when the primary collector did load, the marker keeps this call a no-op.
_install_phase2_acp_capture_fallback()
