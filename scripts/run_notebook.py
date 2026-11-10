"""Execute the real notebook and retain genuine cell outputs, without nbconvert."""
import argparse
import contextlib
import io
import os
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def definitions():
    notebook = nbformat.read(ROOT / "LLM_Theatre.ipynb", as_version=4)
    namespace = {"__name__": "__notebook__"}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and "live" not in cell.metadata.get("tags", []):
            exec(compile(cell.source, f"notebook_cell_{index}", "exec"), namespace)
    return namespace


class Tee(io.StringIO):
    def write(self, value):
        sys.__stdout__.write(value)
        sys.__stdout__.flush()
        return super().write(value)

    def flush(self):
        sys.__stdout__.flush()


def execute():
    path = ROOT / "LLM_Theatre.ipynb"
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    namespace = {"__name__": "__notebook__"}
    count = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        count += 1
        cell.execution_count = count
        captured = Tee()
        try:
            with contextlib.redirect_stdout(captured):
                exec(compile(cell.source, f"notebook_cell_{index}", "exec"), namespace)
        except Exception as exc:
            if captured.getvalue():
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.getvalue()))
            # No full HTTP response or environment dump is written into the notebook.
            message = str(exc)
            for key in ("api_key", "API_KEY", "YANDEX_API_KEY", "HF_TOKEN", "DEEPSEEK_API_KEY"):
                secret = os.getenv(key)
                if secret:
                    message = message.replace(secret, "<hidden>")
            cell.outputs.append(nbformat.v4.new_output(
                "error", ename=type(exc).__name__, evalue=message, traceback=[message]))
            nbformat.write(notebook, path)
            print(f"Stopped at cell {index}: {type(exc).__name__}: {message}")
            return 1
        if captured.getvalue():
            cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.getvalue()))
        nbformat.write(notebook, path)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-cloud", action="store_true")
    parser.add_argument("--smoke-local", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.smoke_local:
        scope = definitions()
        local = scope["LocalLLM"](scope["LOCAL_MODEL"])
        actor = scope["Actor"]("sokolov", local)
        history = [{"speaker": "orlov", "text": "После совещания вы сразу пошли к выходу?"}]
        print("Local smoke:", actor.speak(history, scope["BEATS"][0], None, [])["text"])
    elif arguments.smoke_cloud:
        scope = definitions()
        cloud = scope["CloudLLM"](scope["API_KEY"], scope["FOLDER"], scope["CLOUD_MODEL"])
        reply = cloud.generate("Ответь одним словом: готов.", "Проверка связи.", max_tokens=30)
        print("Cloud smoke:", reply)
    else:
        if arguments.resume:
            os.environ["THEATRE_RESUME"] = "1"
        raise SystemExit(execute())
