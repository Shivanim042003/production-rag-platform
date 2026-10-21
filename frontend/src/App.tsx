import { useState } from "react";
import "./App.css";

const API_URL: string =
  import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const EXAMPLE_QUESTIONS = [
  "What are PostgreSQL indexes used for?",
  "How does Redis key expiration work?",
  "How does PostgreSQL use statistics for query planning?",
  "What factors affect Redis performance?",
];

type Source = {
  chunk_id: string;
  score: number;
};

type StreamEvent =
  | { type: "status"; message: string }
  | { type: "token"; text: string }
  | { type: "answer"; answer: string }
  | { type: "grounding"; grounded: boolean }
  | { type: "sources"; sources: Source[] }
  | { type: "complete" }
  | { type: "error"; message: string };

type Timings = {
  firstTokenMs: number | null;
  totalMs: number;
};

function parseEvents(buffer: string): {
  events: StreamEvent[];
  rest: string;
} {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";
  const events: StreamEvent[] = [];

  for (const block of blocks) {
    const line = block
      .split("\n")
      .find((item) => item.startsWith("data:"));

    if (!line) {
      continue;
    }

    const json = line.slice("data:".length).trim();

    if (json) {
      events.push(JSON.parse(json) as StreamEvent);
    }
  }

  return { events, rest };
}

function formatSeconds(ms: number | null): string {
  return ms === null ? "n/a" : `${(ms / 1000).toFixed(1)} s`;
}

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [grounded, setGrounded] = useState<boolean | null>(null);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [timings, setTimings] = useState<Timings | null>(null);

  const askQuestion = async (question: string) => {
    const text = question.trim();

    if (!text || loading) {
      return;
    }

    setQuery(text);
    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);
    setGrounded(null);
    setTimings(null);
    setStatus("Connecting...");

    const startedAt = performance.now();
    let firstTokenMs: number | null = null;

    try {
      const response = await fetch(
        `${API_URL}/api/v1/rag/query/stream`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: text }),
        },
      );

      if (!response.ok) {
        throw new Error(
          `RAG request failed with status ${response.status}.`,
        );
      }

      if (!response.body) {
        throw new Error("Streaming response body is unavailable.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        const parsed = parseEvents(buffer);
        buffer = parsed.rest;

        for (const event of parsed.events) {
          switch (event.type) {
            case "status":
              setStatus(event.message);
              break;

            case "token":
              if (firstTokenMs === null) {
                firstTokenMs = performance.now() - startedAt;
                setStatus("Generating answer...");
              }
              setAnswer((current) => current + event.text);
              break;

            case "answer":
              // Authoritative answer: replaces the streamed text
              // (for example with the fallback when grounding fails).
              setAnswer(event.answer);
              break;

            case "grounding":
              setGrounded(event.grounded);
              setStatus(
                event.grounded
                  ? "Answer verified against the sources."
                  : "The answer could not be verified.",
              );
              break;

            case "sources":
              setSources(event.sources);
              break;

            case "complete":
              setTimings({
                firstTokenMs,
                totalMs: performance.now() - startedAt,
              });
              break;

            case "error":
              throw new Error(event.message);
          }
        }
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Something went wrong.",
      );
      setStatus("");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Production RAG Platform</h1>
          <p>
            Hybrid retrieval &middot; Reranking &middot; Grounded
            generation
          </p>
        </div>
      </header>

      <main className="main">
        <section className="query-section">
          <label htmlFor="query">Ask a question</label>

          <textarea
            id="query"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
            }}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                (event.ctrlKey || event.metaKey)
              ) {
                event.preventDefault();
                void askQuestion(query);
              }
            }}
            placeholder="What are PostgreSQL indexes used for?"
            rows={4}
            disabled={loading}
          />

          <button
            type="button"
            onClick={() => {
              void askQuestion(query);
            }}
            disabled={loading || !query.trim()}
          >
            {loading ? "Processing..." : "Ask"}
          </button>

          <p className="hint">Ctrl+Enter to submit</p>

          <div className="examples">
            {EXAMPLE_QUESTIONS.map((example) => (
              <button
                type="button"
                key={example}
                className="example"
                disabled={loading}
                onClick={() => {
                  void askQuestion(example);
                }}
              >
                {example}
              </button>
            ))}
          </div>

          {status && <p className="status">{status}</p>}
        </section>

        {error && <section className="error">{error}</section>}

        {answer && (
          <section className="answer-section">
            <div className="section-header">
              <h2>Answer</h2>

              {grounded !== null && (
                <span
                  className={
                    grounded ? "badge grounded" : "badge ungrounded"
                  }
                >
                  {grounded
                    ? "\u2713 Grounded"
                    : "\u26A0 Not grounded"}
                </span>
              )}
            </div>

            <div className="answer">
              {answer}
              {loading && (
                <span className="cursor" aria-hidden="true" />
              )}
            </div>

            {timings && (
              <p className="timings">
                First token {formatSeconds(timings.firstTokenMs)}{" "}
                &middot; Total {formatSeconds(timings.totalMs)}
              </p>
            )}
          </section>
        )}

        {sources.length > 0 && (
          <section className="sources-section">
            <h2>Sources used</h2>

            <div className="sources">
              {sources.map((source) => (
                <div className="source" key={source.chunk_id}>
                  <div className="source-id">{source.chunk_id}</div>

                  <div className="source-score">
                    Score: {source.score.toFixed(4)}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;