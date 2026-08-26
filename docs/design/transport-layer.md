# Transport Layer

Recall separates retrieval from model transport. Retrieval planning, search,
timeline selection, context compilation, and prompt formatting produce a prompt.
The OpenRouter client owns only the protocol request and HTTP transport used to
send that prompt.

## Boundaries

The current request path is:

```text
PromptBuilder
  -> OpenRouterClient protocol request
  -> Transport
  -> ureq
```

`OpenRouterClient` concerns are provider protocol details: model name, messages,
temperature, optional OpenRouter metadata headers, response JSON parsing, and
provider diagnostics.

`Transport` concerns are HTTP details: endpoint, authorization header, timeout
policy, buffered body reading, response status, retry headers, and timing
diagnostics.

Keeping those concerns separate means future providers can reuse the transport
shape without depending on Recall retrieval internals.

## Timeouts

Transport configuration uses independent timeout values for:

- connect
- request write
- response headers
- response body

`RECALL_OPENROUTER_TIMEOUT_SECS` remains as a compatibility default for all four
timeouts. Each stage can be overridden independently with:

- `RECALL_OPENROUTER_CONNECT_TIMEOUT_SECS`
- `RECALL_OPENROUTER_REQUEST_WRITE_TIMEOUT_SECS`
- `RECALL_OPENROUTER_RESPONSE_HEADER_TIMEOUT_SECS`
- `RECALL_OPENROUTER_RESPONSE_BODY_TIMEOUT_SECS`

The transport no longer relies on one global timeout around the full request.

## Diagnostics

Transport timing diagnostics are collected at the HTTP boundary and returned
through the existing opt-in diagnostics path. Normal command output remains
unchanged.

The available stages are:

- request creation
- upload through response headers
- first body byte
- body completion
- total request time
- response body size

`ureq` does not expose a separate "upload complete" timestamp. The current
measurement records the elapsed time from starting `send_json()` until response
headers are available, which includes request upload and upstream header wait.

## Streaming

OpenRouter supports streaming responses by adding `stream: true` to the chat
completion request. The current transport keeps streaming disabled by default.

The streaming path belongs in the transport layer because it changes HTTP body
handling from buffered JSON to Server-Sent Events. The OpenRouter protocol layer
would still decide whether streaming is requested, but the transport would own
incremental body reads and SSE frame parsing.

Future streaming support should route incremental deltas through a callback or
iterator-like response type. The first consumer can still accumulate chunks into
the existing answer string, preserving the current CLI output while making
first-token latency observable.
