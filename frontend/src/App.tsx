import { useState } from "react";
import "./App.css";

type Source = {
  chunk_id: string;
  score: number;
};

type StreamEvent =
  | {
      type: "status";
      message: string;
    }
  | {
      type: "answer";
      answer: string;
    }
  | {
      type: "grounding";
      grounded: boolean;
    }
  | {
      type: "sources";
      sources: Source[];
    }
  | {
      type: "complete";
    }
  | {
      type: "error";
      message: string;
    };

function App() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [grounded, setGrounded] = useState<boolean | null>(null);
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const askQuestion = async () => {
    if (!query.trim()) {
      return;
    }

    setLoading(true);
    setError("");
    setAnswer("");
    setSources([]);
    setGrounded(null);
    setStatus("Connecting...");

    try {
      const response = await fetch(
        "http://localhost:8000/api/v1/rag/query/stream",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            query,
          }),
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

        buffer += decoder.decode(value, {
          stream: true,
        });

        const events = buffer.split("\n\n");

        buffer = events.pop() ?? "";

        for (const event of events) {
          const line = event
            .split("\n")
            .find((item) => item.startsWith("data:"));

          if (!line) {
            continue;
          }

          const jsonData = line
            .slice("data:".length)
            .trim();

          if (!jsonData) {
            continue;
          }

          const streamEvent: StreamEvent =
            JSON.parse(jsonData);

          switch (streamEvent.type) {
            case "status":
              setStatus(streamEvent.message);
              break;

            case "answer":
              setAnswer(streamEvent.answer);
              setStatus("Answer generated.");
              break;

            case "grounding":
              setGrounded(streamEvent.grounded);
              break;

            case "sources":
              setSources(streamEvent.sources);
              break;

            case "complete":
              setStatus("Complete.");
              break;

            case "error":
              throw new Error(streamEvent.message);
          }
        }
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong.",
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
            Hybrid retrieval · Reranking · Grounded generation
          </p>
        </div>
      </header>

      <main className="main">
        <section className="query-section">
          <label htmlFor="query">
            Ask a question
          </label>

          <textarea
            id="query"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
            }}
            placeholder="What are PostgreSQL indexes used for?"
            rows={4}
            disabled={loading}
          />

          <button
            type="button"
            onClick={askQuestion}
            disabled={loading || !query.trim()}
          >
            {loading ? "Processing..." : "Ask"}
          </button>

          {status && (
            <p className="status">
              {status}
            </p>
          )}
        </section>

        {error && (
          <section className="error">
            {error}
          </section>
        )}

        {answer && (
          <section className="answer-section">
            <div className="section-header">
              <h2>Answer</h2>

              {grounded !== null && (
                <span
                  className={
                    grounded
                      ? "badge grounded"
                      : "badge ungrounded"
                  }
                >
                  {grounded
                    ? "✓ Grounded"
                    : "⚠ Not grounded"}
                </span>
              )}
            </div>

            <div className="answer">
              {answer}
            </div>
          </section>
        )}

        {sources.length > 0 && (
          <section className="sources-section">
            <h2>Retrieved Sources</h2>

            <div className="sources">
              {sources.map((source) => (
                <div
                  className="source"
                  key={source.chunk_id}
                >
                  <div className="source-id">
                    {source.chunk_id}
                  </div>

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
