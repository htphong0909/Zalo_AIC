The data has three splits with the same layout: **train** (about 1,600 with labels), **public\_test** (about 600 samples, for the public leaderboard) and **private\_test** (about 600 samples, released after the final Docker solution is frozen).

 

**Folder structure**

```
<split>/                                  # train | public_test | private_test
├── audio/<id>/<id>.wav                   # spoken request of each sample
├── <split>.jsonl                         # one sample per line
├── <split>_tools_catalog.json            # schema of every app tool
├── <split>_environment/<id>/state.json   # initial app data of each sample
├── sample_submission.jsonl               # public_test only
└── zaic_tools/                           # execution environment library for this split

```

 

**Files**

• **(1) Audio:** the user’s latest request and every earlier user turn in history, in Vietnamese with some common English words (e.g. "book", "AC"). WAV, 16 kHz, mono, 16-bit; about 100% recorded by real speakers of different ages, genders and regions (North, Central, South).

• **(2) &lt;split&gt;.jsonl:** the input of each sample — id, audio path, user context (current\_time, timezone, current location, user\_profile, installed\_apps) and earlier conversation turns (history). Train samples also include audio\_text (the transcript of every user turn) and the label: the gold tool\_calls, their tool\_results and the final\_state (the full data of every app the gold calls **changed**).

• **(3) &lt;split&gt;\_tools\_catalog.json:** the apps of the split and the schema of every tool — name (app.tool), description, kind (read, write, or meta for sys.ask\_missing\_info), parameters as JSON Schema and a short summary of the returned fields. 

    ○ Parameters may carry x-ref (must be a valid ID in the app data), x-match: "fuzzy" (free text, scored by token-F1) and x-one-of (at least one of the listed groups is required, e.g. contact\_id or phone\_number). The catalog covers the whole split; the tools a sample may call are those of its context.installed\_apps.

• **(4) state.json:** the initial app data of a sample (devices, accounts, contacts, reminders, …), with the actions of history already applied, plus a *meta block (question*id, current\_time, tools). It is loaded by the environment; the agent must not read it directly.

 

**Examples**

**Sample record in &lt;split&gt;.jsonl**

```
{"id": "train_xxxx", "audio": "audio/train_xxxx/train_xxxx.wav",
"context": {"current_time": "2026-11-16T17:00:00+07:00", "timezone": "Asia/Ho_Chi_Minh",
"user_profile": {"home_address": "123 Trần Hưng Đạo, Quận 5, TP.HCM"},
"installed_apps": {"smart_home": {"tools": [...]}, "ride": {"tools": [...]}}},
"history": [],
"label": {"tool_calls": [...],"tool_results": [...], "final_state": {...}}}   # label: train only

```

 

**Sample tool in &lt;split&gt;_tools_catalog.json (inside apps[].tools)**

```
{"schema_version": "2.7.0",
"note": "kind của mỗi tool: read = ...; write = ...; meta = ...",
"apps": [
  {"id": "smart_home", "name_vi": "Nhà thông minh", "group": "home", "description": "...",
  "tools": [
    {
      "name": "smart_home.light_set",
      "kind": "write",
      "description": "Bật/tắt/chỉnh sáng đèn",
      "parameters": {
        "type": "object",
        "properties": {
          "device_id": {"type": "string", "description": "ID thiết bị, lấy từ smart_home.get_status", "x-ref": "smart_home.devices"},
          "power": {"type": "string", "enum": ["on", "off"]},
          "brightness": {"type": "integer"},
          "color": {"type": "string"}
        },
        "required": ["device_id"]
      },
      "returns": "",
      "app": "smart_home"
    },
    ...
  ]},
  ...
]}

```

 

**Sample state.json**

 

```
{
  "_meta": {"question_id": "train_0248", "current_time": "2026-11-19T18:30:00+07:00", "tools": [...]},
  "smart_home": {
    "devices": [
      {"id": "light_living", "name": "đèn phòng khách", "type": "light", "room": "phòng khách", "floor": 1,
      "state": {"power": "off", "brightness": 40}},
      ...
    ]
  },
  "contacts": {"contacts": [{"id": "c_01", "name": "Hương", "relation": "bạn", "phone": "09701957839"}, ...]},
  ...
}

```

 

**Execution environment**

The organisers provide the **zaic\_tools** Python library, which simulates the apps locally. Each sample runs in its own ToolEnv, which must be created only once. The environment checks each call against the tool schema, executes it, updates the app data and returns the result, which the agent can use in its next step. A call fails with "ok": false and an error message, typically when the tool does not exist or an argument has the wrong data type. Only tools from the apps listed in context.installed\_apps (plus sys.ask\_missing\_info) may be called. Every call, successful or not, is logged; this call log, together with the final state hash, forms the submission. Each split comes with its own version of zaic\_tools.

> import sys
>
> sys.path.insert(0, "public\_test")              # zaic\_tools of the split
>
> from zaic\_tools import ToolEnv
>
>
>
> # Create one environment per sample; root is the folder that holds the split folders
>
> env = ToolEnv("public\_test", "public\_0001", root=".")
>
>
>
> # Check arguments against the tool schema (nothing is executed)
>
> env.validate("smart\_home.light\_set", {"power": "off"})
>
> # -&gt; {"ok": false, "error": "thiếu tham số bắt buộc: device\_id"}
>
>
>
> # Execute a call
>
> # -&gt; {"ok": true, ...} or {"ok": false, "error": "..."}
>
> [env.call](http://env.call)("smart\_home.get\_status", {})
>
>
>
> # Call log and final state hash for submission
>
> env.get\_tool\_calls(), env.get\_state\_hash()

Rule: The agent/solution **must not** read, copy or parse any file under &lt;split&gt;\_environment/ (including state.json); all app data is accessed only through the organisers' zaic\_tools, with exactly one ToolEnv per sample, and every call made on it must appear in the submission.

 

**Inference and submission template**

The template below runs the agent on every sample of a split and writes submission.jsonl: one line per sample with id, tool\_calls and state\_hash. Run it from the folder that holds public\_test/. Create exactly one ToolEnv per sample inside infer().

 

> import json
>
> import sys
>
> from pathlib import Path
>
>
>
> ROOT = Path(".")                                  # folder that holds public\_test/
>
> SPLIT = "public\_test"
>
> sys.path.insert(0, str(ROOT / SPLIT))             # zaic\_tools of the split
>
> from zaic\_tools import ToolEnv
>
>
>
> def infer(item):
>
>     env = ToolEnv(SPLIT, item\["id"\], root=ROOT)   # one environment per sample
>
>     # Your pipeline: ASR (ROOT / SPLIT / item\["audio"\]) -&gt; LLM -&gt; [env.call](http://env.call)(tool, arguments) -&gt; LLM -&gt; ...
>
>     # e.g. [env.call](http://env.call)("smart\_home.get\_status", {})
>
>     #      [env.call](http://env.call)("smart\_home.light\_set", {"device\_id": "light\_kitchen", "power": "off"})
>
>     return env.get\_tool\_calls(), env.get\_state\_hash()
>
>
>
> with open(ROOT / SPLIT / f"{SPLIT}.jsonl", encoding="utf-8") as fin, \\
>
>      open("submission.jsonl", "w", encoding="utf-8") as fout:
>
>     for line in fin:
>
>         item = json.loads(line)
>
>         tool\_calls, state\_hash = infer(item)
>
>         fout.write(json.dumps({"id": item\["id"\], "tool\_calls": tool\_calls, "state\_hash": state\_hash}, ensure\_ascii=False) + "\\n")

 

**Submission file format**

The file must be UTF-8 JSON Lines with exactly one line per sample in the split; see sample\_submission.jsonl for the format.

Submission file example (three lines of public\_test/sample\_submission.jsonl):

> {"id": "public\_0039", "tool\_calls": \[{"name": "alarm.list", "arguments": {}}, {"name": "alarm.create", "arguments": {"time": "2026-11-11T05:30:00+07:00"}}\], "state\_hash": "461fcd40…"}
>
> {"id": "public\_0001", "tool\_calls": \[{"name": "sys.ask\_missing\_info", "arguments": {"tool\_name": "music.add\_to\_playlist", "params": \["playlist\_id", "track\_id"\]}}\], "state\_hash": "c83200c2…"}
>
> {"id": "public\_0002", "tool\_calls": \[\], "state\_hash": "0c9a2ec0…"}

Other notes are described in [README.md](http://README.md) inside the data package.