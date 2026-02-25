import math
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"

MODELS = {
    "qwen3": "qwen3",
    "llama3.2": "llama3.2",
    "gemma3": "gemma3",
}


class OllamaTokenPredictor:
    """Predict next tokens with log-probabilities using a local Ollama model.

    Uses the OpenAI-compatible ``/v1/chat/completions`` endpoint exposed by
    Ollama, with ``logprobs=True`` and ``top_logprobs`` to retrieve the
    model's probability distribution at each generation step.

    Args:
        model_name: Ollama model tag (e.g. ``"qwen3"``, ``"llama3.2"``).
        base_url:   Ollama server URL.  Defaults to ``http://localhost:11434/v1``.
    """

    def __init__(self, model_name: str, base_url: str = OLLAMA_BASE_URL):
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.model_name = model_name

    def predict_tokens(
        self,
        prompt: str,
        max_tokens: int = 100,
        top_logprobs: int = 3,
    ) -> List[Dict]:
        """Generate text and collect per-token predictions with alternatives.

        Args:
            prompt:       The user message to send to the model.
            max_tokens:   Maximum number of tokens to generate.
            top_logprobs: How many top alternatives to request per token
                          (capped by model / Ollama support).

        Returns:
            A list of dicts, one per generated token::

                {
                    "token": str,
                    "probability": float,   # 0-1 scale
                    "alternatives": [(token, prob), ...],
                }
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
            logprobs=True,
            top_logprobs=top_logprobs,
            stream=True,
        )

        predictions: List[Dict] = []
        for chunk in response:
            choice = chunk.choices[0]
            if not choice.delta.content:
                continue
            if not choice.logprobs or not choice.logprobs.content:
                continue

            token = choice.delta.content
            top_lps = choice.logprobs.content[0].top_logprobs
            lp_map = {item.token: item.logprob for item in top_lps}

            chosen_logprob = lp_map.get(token)
            if chosen_logprob is None:
                continue

            alternatives = sorted(
                [(t, math.exp(lp)) for t, lp in lp_map.items() if t != token],
                key=lambda x: x[1],
                reverse=True,
            )

            predictions.append({
                "token": token,
                "probability": math.exp(chosen_logprob),
                "alternatives": alternatives[:2],
            })

        return predictions


def create_token_graph(
    model_name: str,
    predictions: List[Dict],
) -> nx.DiGraph:
    """Build a directed graph of chosen tokens and their top alternatives."""
    G = nx.DiGraph()
    G.add_node("START", token=model_name, prob="START", color="lightgreen", size=4000)

    for i, pred in enumerate(predictions):
        tid = f"t{i}"
        G.add_node(
            tid,
            token=pred["token"],
            prob=f"{pred['probability'] * 100:.1f}%",
            color="lightblue",
            size=6000,
        )
        G.add_edge("START" if i == 0 else f"t{i - 1}", tid)

    last_parent = None
    for i, pred in enumerate(predictions):
        parent = "START" if i == 0 else f"t{i - 1}"
        for j, (alt_token, alt_prob) in enumerate(pred["alternatives"]):
            alt_id = f"t{i}_alt{j}"
            G.add_node(
                alt_id,
                token=alt_token,
                prob=f"{alt_prob * 100:.1f}%",
                color="lightgray",
                size=6000,
            )
            G.add_edge(parent, alt_id)
            last_parent = parent

    if last_parent is not None:
        G.add_node("END", token="END", prob="100%", color="red", size=6000)
        G.add_edge(last_parent, "END")

    return G


def visualize_predictions(
    G: nx.DiGraph,
    figsize: tuple = (14, 80),
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """Draw the token prediction graph with a vertical layout.

    Args:
        G:       A graph returned by :func:`create_token_graph`.
        figsize: Figure size (only used when *ax* is ``None``).
        title:   Plot title.  Defaults to ``"Token prediction"``.
        ax:      Optional Matplotlib axes to draw into (for multi-panel figures).

    Returns:
        The current ``matplotlib.pyplot`` module (caller can ``.show()``).
    """
    standalone = ax is None
    if standalone:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    spacing_y = 5
    spacing_x = 5
    pos = {}

    main_nodes = [n for n in G.nodes() if "_alt" not in n]
    for i, node in enumerate(main_nodes):
        pos[node] = (0, -i * spacing_y)

    for node in G.nodes():
        if "_alt" in node:
            main_token = node.split("_")[0]
            alt_num = int(node.split("_alt")[1])
            if main_token in pos:
                x_offset = -spacing_x if alt_num == 0 else spacing_x
                pos[node] = (x_offset, pos[main_token][1] + 0.05)

    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    node_sizes = [G.nodes[n]["size"] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20, alpha=0.7, ax=ax)

    labels = {n: f"{G.nodes[n]['token']}\n{G.nodes[n]['prob']}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=14, ax=ax)

    ax.set_title(title or "Token prediction", fontsize=16)
    ax.axis("off")

    margin = 8
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    return plt


def compare_models(
    prompt: str,
    model_names: Optional[List[str]] = None,
    max_tokens: int = 50,
    base_url: str = OLLAMA_BASE_URL,
):
    """Run the same prompt through multiple Ollama models and display side by side.

    Args:
        prompt:      The user message to send to each model.
        model_names: List of Ollama model tags.  Defaults to all models in
                     :data:`MODELS`.
        max_tokens:  Maximum tokens per model.
        base_url:    Ollama server URL.

    Returns:
        The ``matplotlib.pyplot`` module with the comparison figure.
    """
    model_names = model_names or list(MODELS.values())
    n = len(model_names)

    results = {}
    for name in model_names:
        print(f"Generating predictions for {name}…")
        predictor = OllamaTokenPredictor(name, base_url=base_url)
        predictions = predictor.predict_tokens(prompt, max_tokens=max_tokens)
        results[name] = predictions

    max_preds = max(len(p) for p in results.values()) if results else 1
    fig_height = max(max_preds * 5 + 4, 20)

    fig = plt.figure(figsize=(14 * n, fig_height))
    gs = gridspec.GridSpec(1, n, figure=fig)

    for idx, name in enumerate(model_names):
        ax = fig.add_subplot(gs[idx])
        G = create_token_graph(name, results[name])
        visualize_predictions(G, title=f"{name}", ax=ax)

    fig.suptitle(
        f'Model comparison — \u201c{prompt[:60]}{"…" if len(prompt) > 60 else ""}\u201d',
        fontsize=18,
        y=1.01,
    )
    plt.tight_layout()
    return plt
