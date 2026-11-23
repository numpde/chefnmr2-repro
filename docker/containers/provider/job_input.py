"""Extract ChefNMR2 observations; the SDK owns model turns and endpoint cleanup."""

import asyncio
from dataclasses import dataclass
import json
import pathlib
import tomllib

from provider_core.adapters.chat_completions.tools import function_tool
from provider_core.conversation import ToolFinished, ToolRejected
from provider_core.interpreter import ModelReportedInterpretationProblem, ModelWindowExpired
from provider_core.payload import PayloadRejected

if __package__:
    from .._shared.wire import Peaks
else:
    from wire import Peaks


@dataclass
class JobInput:
    """Scientific observations; the deployment owns sampling budget and seed."""

    formula: str
    peaks: Peaks


async def interpret(interpreter, source, *, codec, require_execution):
    """Accept one typed proposal without running science or repairing missing facts."""

    prompts = pathlib.Path(__file__).with_name("prompts")
    prompt = (prompts / "interpreter.md").read_text(encoding="utf-8")
    tool_descriptions = tomllib.loads((prompts / "tools.toml").read_text(encoding="utf-8"))

    tools = [
        function_tool(name="submit_input", description=tool_descriptions["submit_input"], parameters=codec.json_schema()),
        function_tool(
            name="report_interpretation_problem",
            description=tool_descriptions["report_interpretation_problem"],
            parameters={
                "type": "object",
                "properties": {"explanation": {"type": "string", "minLength": 1}},
                "required": ["explanation"],
                "additionalProperties": False,
            },
        ),
    ]

    async def dispatch(name, arguments, *, not_after):
        require_execution()

        if asyncio.get_running_loop().time() >= not_after:
            raise ModelWindowExpired()

        if name == "report_interpretation_problem":
            raise ModelReportedInterpretationProblem(model_summary="The model could not establish one complete formula and peak set from the source.")

        try:
            supplied = codec.decode(json.dumps(arguments, allow_nan=False).encode())
        except PayloadRejected as error:
            raise ToolRejected(str(error)) from error

        return ToolFinished(supplied)

    require_execution()

    interpretation = await interpreter.run(
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": source}],
        tools=tools,
        dispatch=dispatch,
    )

    return interpretation.value
