# Agent integration example

This page shows how an LLM agent can call the `wasmagent-train-replay` tool
interface from a `tool_use` message, validate the arguments with JSON Schema,
and return a `tool_result` containing root-cause evidence.

The current agent-facing tool is `trace_tensor`. It wraps
`EpochReplayer.find_root_cause()` for one tensor entity in a PyTorch Flight
Recorder dump.

## Tool definition

Register this tool with the agent runtime:

```json
{
  "name": "trace_tensor",
  "description": "Trace causal ancestors for a tensor entity in a Flight Recorder dump.",
  "input_schema": {
    "type": "object",
    "additionalProperties": false,
    "required": ["entity_id"],
    "properties": {
      "entity_id": {
        "type": "string",
        "minLength": 1,
        "description": "PROV entity id for the anomalous tensor, for example tensor:0:1:out."
      }
    }
  },
  "output_schema": {
    "type": "object",
    "additionalProperties": false,
    "required": ["tool", "entity_id", "causal_ancestors"],
    "properties": {
      "tool": {
        "const": "trace_tensor"
      },
      "entity_id": {
        "type": "string"
      },
      "causal_ancestors": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Activity ids returned by EpochReplayer.find_root_cause()."
      }
    }
  }
}
```

`dump_path` is intentionally supplied by the host integration rather than by
the model. That keeps the model's arguments limited to the schema-validated
debugging request while the host controls local file access.

## Message schemas

The agent runtime should accept a `tool_use` envelope that identifies the tool
and carries the schema-validated input:

```json
{
  "$id": "https://wasmagent.dev/schemas/train-replay/tool-use.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["type", "id", "name", "input"],
  "properties": {
    "type": {
      "const": "tool_use"
    },
    "id": {
      "type": "string",
      "minLength": 1
    },
    "name": {
      "const": "trace_tensor"
    },
    "input": {
      "type": "object",
      "additionalProperties": false,
      "required": ["entity_id"],
      "properties": {
        "entity_id": {
          "type": "string",
          "minLength": 1
        }
      }
    }
  }
}
```

The host returns a matching `tool_result` envelope. The `tool_use_id` links the
result to the original request, and `content[0].json` is the root-cause payload
returned by `trace_tensor`:

```json
{
  "$id": "https://wasmagent.dev/schemas/train-replay/tool-result.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["type", "tool_use_id", "content"],
  "properties": {
    "type": {
      "const": "tool_result"
    },
    "tool_use_id": {
      "type": "string",
      "minLength": 1
    },
    "content": {
      "type": "array",
      "minItems": 1,
      "maxItems": 1,
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["type", "json"],
        "properties": {
          "type": {
            "const": "json"
          },
          "json": {
            "type": "object",
            "additionalProperties": false,
            "required": ["tool", "entity_id", "causal_ancestors"],
            "properties": {
              "tool": {
                "const": "trace_tensor"
              },
              "entity_id": {
                "type": "string"
              },
              "causal_ancestors": {
                "type": "array",
                "items": {
                  "type": "string"
                }
              }
            }
          }
        }
      }
    }
  }
}
```

## Example `tool_use`

When the agent identifies an anomalous output tensor, it emits a tool call like
this:

```json
{
  "type": "tool_use",
  "id": "toolu_train_trace_001",
  "name": "trace_tensor",
  "input": {
    "entity_id": "tensor:0:1:out"
  }
}
```

## Host dispatch

The host receives the `tool_use`, validates `input` against the JSON Schema
above, and dispatches it to the local Python interface:

```python
from pathlib import Path

from train_replay.agent.tools import dispatch_tool

tool_use = {
    "type": "tool_use",
    "id": "toolu_train_trace_001",
    "name": "trace_tensor",
    "input": {"entity_id": "tensor:0:1:out"},
}

dump_path = Path("examples/sample_trace.pkl")
result = dispatch_tool(tool_use["name"], dump_path, tool_use["input"])
tool_result = {
    "type": "tool_result",
    "tool_use_id": tool_use["id"],
    "content": [{"type": "json", "json": result}],
}
```

The same call is also available through the CLI:

```bash
train-replay agent-query examples/sample_trace.pkl \
  --tool trace_tensor \
  --args '{"entity_id":"tensor:0:1:out"}'
```

## Sample `tool_result`

For a dump containing sequence `1` on rank `0`, the result returned to the
agent is:

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_train_trace_001",
  "content": [
    {
      "type": "json",
      "json": {
        "tool": "trace_tensor",
        "entity_id": "tensor:0:1:out",
        "causal_ancestors": ["act:0:all_reduce:1"]
      }
    }
  ]
}
```

The root-cause signal is the `causal_ancestors` array. In this example,
`act:0:all_reduce:1` is the collective activity that generated
`tensor:0:1:out`, so the agent can explain that the suspect tensor traces back
to rank `0`'s `all_reduce` at sequence `1`.
