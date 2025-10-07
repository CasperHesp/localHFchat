Local HF Chat — v4

New in this build
• **Markdown rendering** in the chat (headings, lists, code blocks, inline code, links, blockquotes).
• **Thinking spinner** while the backend is generating.
• **Idle kickstart (10s)**: if a conversation is empty, we auto-post a conversation starter pulled from `/static/conversation_starters.json`, based on the selected context (Market → company_market, Geography → company_geography, Matching → company_matching).

All previous features retained
• Default refinement = Model rewrite (with heuristic cleanup).
• Loading overlay + auto-reload after the model becomes ready.
• Auto-length (or manual cap) to reduce cut‑offs.
• M3/MPS acceleration on macOS (PyTorch MPS).
• Company context + dropdown extra context (Market/Geography/Matching).
• Random port + browser auto-open on macOS; random published port in Docker.

Quick start (macOS)
  chmod +x run_macos.sh
  ./run_macos.sh
  # optional model override
  ./run_macos.sh Qwen/Qwen2.5-0.5B-Instruct

Docker
  chmod +x run_docker.sh
  ./run_docker.sh
