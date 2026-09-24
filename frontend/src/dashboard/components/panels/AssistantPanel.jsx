import React, { useEffect, useRef, useState } from 'react';
import { Bot, CircleAlert, CircleStop, ImageIcon, SendHorizontal, User } from 'lucide-react';
import { Button, EmptyState, FormattedAnswer, Notice, cx } from '../ui';
import ConfidenceBadge from '../ConfidenceBadge';

const SUGGESTIONS = [
  'What land cover is visible?',
  'Is there any water or built-up area?',
  'Describe the scene.',
];

// Shown once, before any real conversation, so the panel doesn't feel unresponsive on first open. This
// is rendered locally only - it is never pushed into ws.chat, so it never reaches the model as history,
// never appears in the PDF report, and disappears as soon as a real message exists.
const GREETING = "Hi! I'm the SatQuery-AI assistant. Add a satellite image and ask me anything about it — land cover, features, or what's visible in the scene.";

function Message({ msg }) {
  if (msg.role === 'system') {
    return <p className="py-1 text-center text-[11px] italic text-slate-500">{msg.content}</p>;
  }
  const isUser = msg.role === 'user';
  return (
    <div className={cx('flex items-start gap-2.5', isUser && 'flex-row-reverse')}>
      <div className={cx('mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border', isUser ? 'border-slate-600/50 bg-slate-800/60 text-slate-300' : 'border-blue-700/40 bg-blue-950/40 text-blue-300')}>
        {isUser ? <User size={14} /> : <Bot size={14} />}
      </div>
      <div className={cx(
        'max-w-[85%] rounded-xl border px-3.5 py-2.5 text-[13px] leading-relaxed',
        isUser ? 'whitespace-pre-wrap border-slate-700/50 bg-slate-800/60 text-slate-100' : 'border-blue-700/40 bg-blue-950/30 text-blue-50',
      )}>
        {isUser ? msg.content : <FormattedAnswer text={msg.content} />}
        {!isUser && <ConfidenceBadge info={msg.confidence} />}
      </div>
    </div>
  );
}

export default function AssistantPanel({ ws, goTo }) {
  const { chat, isExecuting, errorFor, dismissError, sendMessage, stopAnalysis, slots } = ws;
  const error = errorFor('assistant');
  const [text, setText] = useState('');
  const endRef = useRef(null);
  const hasImage = Boolean(slots.a.file);
  // Results of two-image runs are in the same conversation (and the PDF) but have their own panels
  const visible = chat.filter((entry) => entry.scope !== 'pair');
  const canSend = hasImage && !isExecuting && text.trim().length > 0;

  // Keep the newest message (or the spinner/error) in view
  useEffect(() => {
    endRef.current?.scrollIntoView?.({ behavior: 'smooth', block: 'end' });
  }, [visible.length, isExecuting, error]);

  const submit = (e) => {
    e?.preventDefault();
    if (!canSend) return;
    const message = text.trim();
    setText('');
    sendMessage(message);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 space-y-3.5 overflow-y-auto pr-1 sq-scroll" aria-live="polite">
        {visible.length === 0 && (
          <Message msg={{ role: 'ai', content: GREETING }} />
        )}

        {visible.length === 0 && !isExecuting && (
          hasImage ? (
            <div className="space-y-3 pt-2">
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <button key={s} type="button" onClick={() => setText(s)} className="cursor-pointer rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300 transition hover:border-blue-500/50 hover:text-blue-200">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <EmptyState
              icon={ImageIcon}
              title="Add an image to start"
              action={<Button variant="subtle" size="sm" className="mt-2" onClick={() => goTo('imagery')}>Open Imagery</Button>}
            />
          )
        )}

        {visible.map((msg, i) => <Message key={i} msg={msg} />)}

        {isExecuting && (
          <div className="flex items-center gap-2 px-1 text-xs text-blue-300">
            <span className="h-2 w-2 animate-pulse rounded-full bg-blue-400" />
            <span className="italic">Analysing&hellip;</span>
          </div>
        )}
        {error && <Notice tone="error" icon={CircleAlert} title="Something went wrong" onDismiss={dismissError}>{error}</Notice>}
        <div ref={endRef} />
      </div>

      <form onSubmit={submit} className="mt-3 shrink-0">
        <div className="flex items-end gap-2 rounded-xl border border-white/15 bg-black/40 p-2 transition focus-within:border-blue-500">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) submit(e);
            }}
            rows={2}
            disabled={isExecuting}
            aria-label="Your question"
            title="Enter to send, Shift+Enter for a new line"
            placeholder={hasImage ? 'Ask about the imagery…' : 'Add an image to ask questions'}
            className="max-h-32 min-h-[2.75rem] flex-1 resize-none bg-transparent px-2 py-1 text-[13px] text-slate-100 placeholder:text-slate-500 focus:outline-none disabled:opacity-50"
          />
          {isExecuting ? (
            <Button
              type="button" size="sm" variant="rose" icon={CircleStop} onClick={stopAnalysis}
              title="Stop the running analysis" aria-label="Stop analysing" className="h-9 w-9 !p-0"
            />
          ) : (
            <Button type="submit" size="sm" icon={SendHorizontal} disabled={!canSend} aria-label="Send" className="h-9 w-9 !p-0" />
          )}
        </div>
      </form>
    </div>
  );
}
