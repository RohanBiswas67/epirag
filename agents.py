"""
EpiRAG — agents.py
------------------
Multi-agent swarm debate engine with real-time SSE callbacks.

5 agents debate in parallel. Each posts events via callback as it responds,
enabling live streaming to the browser via SSE.

Agent roster:
  Alpha   - meta-llama/Llama-3.1-8B-Instruct   (cerebras)    - Skeptic
  Beta    - Qwen/Qwen2.5-7B-Instruct           (together)    - Literalist
  Gamma   - HuggingFaceH4/zephyr-7b-beta       (featherless) - Connector
  Delta   - deepseek-ai/DeepSeek-R1            (sambanova)   - Deep Reasoner
  Epsilon - llama-3.3-70b-versatile            (groq)        - Synthesizer
"""

import concurrent.futures
from groq import Groq
from huggingface_hub import InferenceClient

AGENTS = [
    {
        "name":        "Alpha",
        "model":       "meta-llama/Llama-3.1-8B-Instruct",
        "provider":    "cerebras",
        "client_type": "hf",
        "color":       "red",
        "personality": (
            "You are Agent Alpha - a ruthless Skeptic. "
            "Challenge every claim aggressively. Demand evidence. Math nerd. "
            "Point out what is NOT in the sources. Be blunt and relentless."
        )
    },
    {
        "name":        "Beta",
        "model":       "Qwen/Qwen2.5-7B-Instruct",
        "provider":    "together",
        "client_type": "hf",
        "color":       "yellow",
        "personality": (
            "You are Agent Beta - a strict Literalist. "
            "Accept ONLY what is explicitly stated in the source text. "
            "Reject all inferences. If it is not literally written, it does not exist."
        )
    },
    {
        "name":        "Gamma",
        "model":       "HuggingFaceH4/zephyr-7b-beta",
        "provider":    "featherless-ai",
        "client_type": "hf",
        "color":       "green",
        "personality": (
            "You are Agent Gamma - a Pattern Connector. "
            "Find non-obvious connections between sources."
            "Look for relationships and synthesis opportunities others miss."
        )
    },
    {
        "name":        "Delta",
        "model":       "deepseek-ai/DeepSeek-R1",
        "provider":    "sambanova",
        "client_type": "hf",
        "color":       "purple",
        "personality": (
            "You are Agent Delta - a Deep Reasoner. Prefer Detailed answer or to the point. "
            "Move slowly and carefully. Check every logical step. "
            "Flag hidden assumptions and claims beyond what sources support."
        )
    },
    {
        "name":        "Epsilon",
        "model":       "llama-3.3-70b-versatile",
        "provider":    "groq",
        "client_type": "groq",
        "color":       "blue",
        "personality": (
            "You are Agent Epsilon - the Synthesizer. "
            "Reconcile the debate. Find where agents agree and disagree. "
            "Produce a final authoritative answer with source citations."
        )
    },
]

MAX_ROUNDS       = 3
MAX_TOKENS_AGENT = 500
MAX_TOKENS_SYNTH = 900
TIMEOUT_SECONDS  = 30
CONTEXT_LIMIT    = 3000   # chars fed to synthesizer to avoid 413

DOMAIN_GUARD = """
SCOPE: EpiRAG — strictly epidemic modeling, network science, mathematical epidemiology, disease biology and related epidemiology.
Do NOT answer anything outside this domain. If off-topic, say so and stop.
"""


def _make_client(agent, groq_key, hf_token):
    if agent["client_type"] == "groq":
        return Groq(api_key=groq_key)
    return InferenceClient(provider=agent["provider"], api_key=hf_token)


def _call_agent(agent, messages, groq_key, hf_token, max_tokens=MAX_TOKENS_AGENT):
    try:
        client = _make_client(agent, groq_key, hf_token)
        if agent["client_type"] == "groq":
            resp = client.chat.completions.create(
                model=agent["model"], messages=messages,
                temperature=0.7, max_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()
        else:
            resp = client.chat_completion(
                model=agent["model"], messages=messages,
                temperature=0.7, max_tokens=max_tokens
            )
            return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[{agent['name']} error: {str(e)[:100]}]"


def _round1_msgs(agent, question, context):
    return [
        {"role": "system", "content": f"{DOMAIN_GUARD}\n\n{agent['personality']}"},
        {"role": "user",   "content": (
            f"Context from research papers/web:\n\n{context}\n\n---\n\n"
            f"Question: {question}\n\n"
            f"Answer based on context. Cite sources. Stay in character."
        )}
    ]


def _round2_msgs(agent, question, context, prev_answers):
    others = "\n\n".join(
        f"=== {n}'s answer ===\n{a}"
        for n, a in prev_answers.items() if n != agent["name"]
    )
    return [
        {"role": "system", "content": f"{DOMAIN_GUARD}\n\n{agent['personality']}"},
        {"role": "user",   "content": (
            f"Context:\n\n{context}\n\nQuestion: {question}\n\n"
            f"Your answer so far:\n{prev_answers.get(agent['name'], '')}\n\n---\n\n"
            f"Other agents said:\n\n{others}\n\n---\n\n"
            f"Now ARGUE. Where do you agree/disagree? What did they miss or get wrong? "
            f"Stay in character. Be specific."
        )}
    ]


def _synth_msgs(question, context, all_rounds):
    transcript = ""
    for i, rnd in enumerate(all_rounds, 1):
        transcript += f"\n\n{'='*40}\nROUND {i}\n{'='*40}\n"
        for name, ans in rnd.items():
            transcript += f"\n-- {name} --\n{ans}\n"
    ctx = context[:CONTEXT_LIMIT] + "..." if len(context) > CONTEXT_LIMIT else context
    return [
        {"role": "system", "content": (
            f"{DOMAIN_GUARD}\n\nYou are the Synthesizer. "
            "Produce the single best final answer by:\n"
            "1. Noting what all agents agreed on (high confidence)\n"
            "2. Resolving disagreements using strongest evidence\n"
            "3. Flagging genuine uncertainty\n"
            "4. Citing sources clearly\n"
            "End with: CONFIDENCE: HIGH / MEDIUM / LOW"
        )},
        {"role": "user", "content": (
            f"Context (truncated):\n\n{ctx}\n\n---\n\n"
            f"Question: {question}\n\n---\n\n"
            f"Debate transcript:{transcript}\n\n---\n\n"
            f"Produce the final synthesized answer."
        )}
    ]


def _converged(answers):
    agree = ["i agree", "correct", "you're right", "i concur",
             "well said", "exactly", "this is accurate", "that's right"]
    hits  = sum(1 for a in answers.values()
                if any(p in a.lower() for p in agree))
    return hits >= len(answers) * 0.5


def run_debate(question, context, groq_key, hf_token, callback=None):
    """
    Run the full multi-agent swarm debate.

    callback(event: dict) is called after each agent responds, enabling SSE streaming.

    event shapes:
      {"type": "agent_done",   "round": int, "name": str, "color": str, "text": str}
      {"type": "round_start",  "round": int}
      {"type": "synthesizing"}
      {"type": "done",         "consensus": bool, "rounds": int}

    Returns:
      {"final_answer", "debate_rounds", "consensus", "rounds_run", "agent_count"}
    """
    def emit(event):
        if callback:
            callback(event)

    debate_agents  = [a for a in AGENTS if a["name"] != "Epsilon"]
    synthesizer    = next(a for a in AGENTS if a["name"] == "Epsilon")
    agent_colors   = {a["name"]: a["color"] for a in AGENTS}
    debate_rounds  = []

    # -- Round 1 ------------------------------------------------------------
    emit({"type": "round_start", "round": 1})
    round1 = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(debate_agents)) as ex:
        futures = {
            ex.submit(_call_agent, agent,
                      _round1_msgs(agent, question, context),
                      groq_key, hf_token): agent
            for agent in debate_agents
        }
        for future in concurrent.futures.as_completed(futures, timeout=TIMEOUT_SECONDS * 2):
            agent = futures[future]
            try:
                answer = future.result(timeout=TIMEOUT_SECONDS)
            except Exception as e:
                answer = f"[{agent['name']} timed out: {e}]"
            round1[agent["name"]] = answer
            emit({"type": "agent_done", "round": 1,
                  "name": agent["name"], "color": agent_colors[agent["name"]],
                  "text": answer})

    debate_rounds.append(round1)
    consensus    = _converged(round1)
    current      = round1
    rounds_run   = 1

    # -- Rounds 2+ ------------------------------------------------------------
    while not consensus and rounds_run < MAX_ROUNDS:
        rounds_run += 1
        emit({"type": "round_start", "round": rounds_run})
        next_round = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(debate_agents)) as ex:
            futures = {
                ex.submit(_call_agent, agent,
                          _round2_msgs(agent, question, context, current),
                          groq_key, hf_token): agent
                for agent in debate_agents
            }
            for future in concurrent.futures.as_completed(futures, timeout=TIMEOUT_SECONDS * 2):
                agent = futures[future]
                try:
                    answer = future.result(timeout=TIMEOUT_SECONDS)
                except Exception as e:
                    answer = f"[{agent['name']} timed out: {e}]"
                next_round[agent["name"]] = answer
                emit({"type": "agent_done", "round": rounds_run,
                      "name": agent["name"], "color": agent_colors[agent["name"]],
                      "text": answer})

        debate_rounds.append(next_round)
        current    = next_round
        consensus  = _converged(next_round)

    # -- Synthesis ------------------------------------------------------------
    emit({"type": "synthesizing"})
    final = _call_agent(synthesizer, _synth_msgs(question, context, debate_rounds),
                        groq_key, hf_token, max_tokens=MAX_TOKENS_SYNTH)
    emit({"type": "done", "consensus": consensus, "rounds": rounds_run})

    return {
        "final_answer":  final,
        "debate_rounds": debate_rounds,
        "consensus":     consensus,
        "rounds_run":    rounds_run,
        "agent_count":   len(debate_agents)
    }