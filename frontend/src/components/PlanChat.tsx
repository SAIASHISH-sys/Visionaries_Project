import { useRef, useState } from "react";
import { useQueryChat } from "../api/hooks";
import type { Track } from "../api/types";

interface Message {
  role: "user" | "assistant";
  text: string;
}

interface Props {
  runId: string;
  track: Track;
}

export function PlanChat({ runId, track }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const query = useQueryChat(runId, track);
  const bottomRef = useRef<HTMLDivElement>(null);

  const send = () => {
    const q = input.trim();
    if (!q || query.isPending) return;
    const next: Message[] = [...messages, { role: "user", text: q }];
    setMessages(next);
    setInput("");
    query.mutate(q, {
      onSuccess: (res) => {
        setMessages((m) => [...m, { role: "assistant", text: res.answer }]);
        setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
      },
      onError: () => {
        setMessages((m) => [...m, { role: "assistant", text: "Could not get an answer. Check that the manifest exists and GEMINI_API_KEY is set." }]);
      },
    });
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 border border-rule bg-surface p-4" style={{ minHeight: "12rem", maxHeight: "28rem", overflowY: "auto" }}>
        {messages.length === 0 && (
          <p className="m-0 text-sm italic text-muted">Ask anything about this floor plan — rooms, areas, violations, adjacency…</p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`flex flex-col gap-1 ${m.role === "user" ? "items-end" : "items-start"}`}>
            <span className="label">{m.role === "user" ? "You" : "Assistant"}</span>
            <p className={`m-0 max-w-prose rounded px-3 py-2 text-sm leading-relaxed ${m.role === "user" ? "bg-ink text-paper" : "border border-rule bg-paper text-ink"}`}>
              {m.text}
            </p>
          </div>
        ))}
        {query.isPending && (
          <div className="flex items-start flex-col gap-1">
            <span className="label">Assistant</span>
            <div className="flex gap-1 px-3 py-2 border border-rule">
              <span className="h-2 w-2 animate-pulse bg-ink rounded-full" />
              <span className="h-2 w-2 animate-pulse bg-ink rounded-full" style={{ animationDelay: "150ms" }} />
              <span className="h-2 w-2 animate-pulse bg-ink rounded-full" style={{ animationDelay: "300ms" }} />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          placeholder="e.g. Which rooms are below minimum area?"
          disabled={query.isPending}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          className="flex-1 border border-rule bg-surface px-3 py-2 text-sm focus:border-ink focus:outline-none disabled:opacity-50"
        />
        <button
          type="button"
          disabled={!input.trim() || query.isPending}
          onClick={send}
          className="bg-ink px-5 py-2 text-sm font-semibold text-paper transition-opacity hover:opacity-85 disabled:opacity-40"
        >
          Send
        </button>
      </div>
    </div>
  );
}
