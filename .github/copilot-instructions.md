<!-- mermaid-ai-skills:start -->
## Mermaid Diagrams

When the user asks to create, edit, or visualize a diagram, follow the
instructions in `.github/instructions/mermaid.instructions.md`.
<!-- mermaid-ai-skills:end -->

## Engie AI Bridge delegation

When the user requests a consultation on engie.ai, delegate only the question to
Engie AI Bridge using `Claude Haiku 4.5 (copilot)` for the subagent. Do not
substitute Qwen or infer the effective model from the agent configuration;
report which model actually handled the browser interaction when verifiable.
Leave the main chat model and context unchanged.
