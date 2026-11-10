"""Offline checks: no paid API calls and no model downloads."""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat

from run_notebook import ROOT, definitions


SCOPE = definitions()


class NotebookChecks(unittest.TestCase):
    def test_notebook_schema_and_python(self):
        notebook = nbformat.read(ROOT / "LLM_Theatre.ipynb", as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, "notebook", "exec")

    def test_evidence_is_released_by_act(self):
        tools = SCOPE["TOOLS"]
        early = tools.file_search("контроллер 21:14 поправка", 1)
        self.assertTrue(all(hit["doc_id"] == "statement" for hit in early))
        with self.assertRaises(ValueError):
            tools.read_excerpt("clock", 1, 3, 2)
        with self.assertRaises(ValueError):
            tools.read_excerpt("../../.env", 1, 2, 3)
        self.assertEqual(tools.file_search("контроллера 21:14", 2)[0]["doc_id"], "access")

    def test_clock_and_invalid_tool_calls(self):
        tools = SCOPE["TOOLS"]
        self.assertEqual(tools.timeline_check("21:14", 12)["actual"], "21:02")
        self.assertEqual(tools.timeline_check("00:05", 12)["actual"], "23:53")
        for name, args, act in [
            ("timeline_check", {"recorded_time": "21:14", "offset_minutes": 12}, 2),
            ("timeline_check", {"recorded_time": "25:00", "offset_minutes": 12}, 3),
            ("timeline_check", {"recorded_time": "21:14", "offset_minutes": -12}, 3),
            ("file_search", {"query": "ключ", "path": ".env"}, 3),
            ("shell", {"command": "dir"}, 3),
        ]:
            with self.assertRaises(ValueError):
                tools.execute(name, args, act)

    def test_addressed_question_has_priority(self):
        history = [{"speaker": "orlov", "text": "Вы вернулись?", "address_to": "sokolov"}]
        self.assertEqual(SCOPE["candidates"](history), ["sokolov"])
        history = [{"speaker": "orlov", "text": "Понятно.", "address_to": "sokolov"},
                   {"speaker": "sokolov", "text": "Да.", "address_to": "orlov"},
                   {"speaker": "orlov", "text": "Запишем.", "address_to": "sokolov"},
                   {"speaker": "sokolov", "text": "Хорошо.", "address_to": "orlov"}]
        self.assertEqual(SCOPE["candidates"](history), ["volkova"])

    def test_director_fallback_is_visible(self):
        class InvalidDirector:
            def generate(self, *args, **kwargs):
                return '{"speaker":"unknown","reason":"oops"}'
        history = [{"speaker": "orlov", "text": "Понятно.", "address_to": "sokolov"}]
        speaker, source, reason = SCOPE["choose_speaker"](InvalidDirector(), history, SCOPE["BEATS"][0])
        self.assertIn(speaker, ("volkova", "sokolov"))
        self.assertEqual(source, "fallback")
        self.assertTrue(reason)

    def test_local_actor_has_no_evidence_access(self):
        class LocalProbe:
            def generate(self, system, prompt, **kwargs):
                self.prompt = prompt
                return "Я зарядку забыл."
        local = LocalProbe()
        actor = SCOPE["Actor"]("sokolov", local)
        actor.evidence.append({"private": "PRIVATE_SENTINEL"})
        log = []
        line = actor.speak([{"speaker": "orlov", "text": "Зачем вернулись?"}], SCOPE["BEATS"][1], None, log)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(local.prompt, ensure_ascii=False))
        self.assertNotIn("Поправка", json.dumps(local.prompt, ensure_ascii=False))
        self.assertEqual(local.prompt[-1]["role"], "user")
        self.assertEqual(line["address_to"], "orlov")
        self.assertEqual(log, [])

    def test_local_actor_receives_own_lines_as_assistant(self):
        class LocalProbe:
            def generate(self, system, prompt, **kwargs):
                self.prompt = prompt
                return "Зарядка осталась на столе."
        local = LocalProbe()
        actor = SCOPE["Actor"]("sokolov", local)
        history = [{"speaker": "orlov", "text": "Вы вернулись?"},
                   {"speaker": "sokolov", "text": "Да, за зарядкой."},
                   {"speaker": "volkova", "text": "Где она лежала?"}]
        actor.speak(history, SCOPE["BEATS"][1], None, [])
        self.assertEqual([m["role"] for m in local.prompt], ["user", "assistant", "user"])

    def test_editor_preserves_raw_local_output(self):
        class LocalProbe:
            def generate(self, *args, **kwargs):
                return "Я за зарядка вернулся."
        class EditorProbe:
            def generate(self, system, prompt, **kwargs):
                self.prompt = prompt
                return '{"text":"Я вернулся за зарядкой."}'
        editor = EditorProbe()
        actor = SCOPE["Actor"]("sokolov", LocalProbe(), editor=editor)
        line = actor.speak([{"speaker": "orlov", "text": "Зачем вернулись?"}], SCOPE["BEATS"][0], None, [])
        self.assertEqual(line["raw_text"], "Я за зарядка вернулся.")
        self.assertEqual(line["text"], "Я вернулся за зарядкой.")
        self.assertIn(line["raw_text"], editor.prompt)

    def test_investigator_uses_tool_result_before_speaking(self):
        class ScriptedModel:
            answers = iter([
                '{"text":"В журнале есть вход.","address_to":"sokolov"}',
                '{"tool":"file_search","arguments":{"query":"вход 21:14"}}',
                '{"text":"В журнале вход в 21:14. Вы возвращались?","address_to":"sokolov"}',
            ])
            def generate(self, *args, **kwargs):
                return next(self.answers)
        actor = SCOPE["Actor"]("volkova", ScriptedModel())
        log = []
        line = actor.speak([], SCOPE["BEATS"][1], SCOPE["TOOLS"], log)
        self.assertIn("21:14", line["text"])
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["result"][0]["doc_id"], "access")

    def test_plan_validation_and_export(self):
        plan = {"title": "Поправка к протоколу", "acts": [
            {"id": i, "title": "Сцена", "goal": "Проверить версию.", "stage": "Папка на столе."}
            for i in range(1, 4)]}
        SCOPE["validate_plan"](plan)
        directions = {str(i): {"opening": "Пауза.", "closing": "Папку закрывают."} for i in range(1, 4)}
        history = [{"act": 1, "speaker": "sokolov", "text": "Не помню.", "backend": "local"}]
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            result = SCOPE["export_play"](plan, history, directions, [], [], temporary, [])
            data = json.loads((Path(temporary) / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(data["turns"], result["turns"])
            self.assertIn("**Соколов:** Не помню.", (Path(temporary) / "play.md").read_text(encoding="utf-8"))
        plan["acts"][1]["id"] = 1
        with self.assertRaises(ValueError):
            SCOPE["validate_plan"](plan)

    def test_resume_keeps_finished_acts_and_private_evidence(self):
        plan = {"title": "Поправка к протоколу", "acts": [
            {"id": i, "title": "Сцена", "goal": "Проверить версию.", "stage": "Папка на столе."}
            for i in range(1, 4)]}
        finished_count = sum(beat["turns"] for beat in SCOPE["BEATS"][:2])
        roles = (["orlov", "sokolov", "volkova", "sokolov"] * finished_count)[:finished_count]
        history = [{"act": 1 if i < SCOPE["BEATS"][0]["turns"] else 2,
                    "speaker": r, "text": f"Реплика {i}.",
                    "address_to": "orlov" if r == "sokolov" else "sokolov"}
                   for i, r in enumerate(roles)]
        resume = {"plan": plan, "turns": list(history), "decisions": [], "tool_calls": [],
                  "directions": {str(i): {"opening": "Начало.", "closing": "Пауза."} for i in (1, 2)},
                  "actor_evidence": {"volkova": [{"saved": "PRIVATE_SENTINEL"}]},
                  "models": {"cloud": SCOPE["CLOUD_MODEL"], "local": SCOPE["LOCAL_MODEL"]}}
        class DummyActor:
            def __init__(self, role, llm, editor=None):
                self.role = role
                self.evidence = []
            def speak(self, history, beat, tools, log):
                if not log:
                    log.append({"tool": "timeline_check", "result": {"actual": "21:02"}})
                return {"text": "Продолжим.", "raw_text": "Продолжим.",
                        "address_to": "orlov" if self.role == "sokolov" else "sokolov"}
        selections = iter(["orlov", "sokolov", "volkova", "sokolov"] * 3)
        def choose(*args):
            return next(selections), "rule", "Тест продолжения."
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            with patch.dict(SCOPE, Actor=DummyActor, choose_speaker=choose,
                            stage_direction=lambda *a, **kw: "Пауза."), contextlib.redirect_stdout(io.StringIO()):
                result = SCOPE["run_theatre"](None, None, plan, output=temporary, resume=resume)
            self.assertEqual(result["turns"][:finished_count], history)
            self.assertEqual(len(result["turns"]), sum(beat["turns"] for beat in SCOPE["BEATS"]))
            state = json.loads((Path(temporary) / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertTrue(state["complete"])
            self.assertEqual(state["actor_evidence"]["volkova"][0]["saved"], "PRIVATE_SENTINEL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
