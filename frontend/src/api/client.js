/**
 * API client for the Agentic RAG backend.
 *
 * All calls proxy through Vite dev server → http://localhost:8000.
 * In production, set VITE_API_BASE to the backend URL.
 */

const BASE = import.meta.env.VITE_API_BASE ?? ''

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------

/** @returns {Promise<Array<{doc_id, filename, collection, file_type, vector_count}>>} */
export async function listDocuments(collection = 'demo') {
  const res = await fetch(`${BASE}/api/documents?collection=${encodeURIComponent(collection)}`)
  if (!res.ok) throw new Error(`listDocuments failed: ${res.status}`)
  return res.json()
}

/**
 * Upload a file and ingest it into Pinecone.
 * @param {File} file
 * @param {string} collection
 * @param {string} accessScope
 * @returns {Promise<{doc_id, filename, collection, vectors_upserted}>}
 */
export async function uploadDocument(file, collection = 'demo', accessScope = 'public') {
  const form = new FormData()
  form.append('file', file)
  form.append('collection', collection)
  form.append('access_scope', accessScope)
  const res = await fetch(`${BASE}/api/documents/upload`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? `Upload failed: ${res.status}`)
  }
  return res.json()
}

/**
 * Delete all Pinecone vectors for a document.
 * @returns {Promise<{deleted, doc_id, collection}>}
 */
export async function deleteDocument(docId, collection = 'demo') {
  const res = await fetch(
    `${BASE}/api/documents/${encodeURIComponent(docId)}?collection=${encodeURIComponent(collection)}`,
    { method: 'DELETE' },
  )
  if (!res.ok) throw new Error(`Delete failed: ${res.status}`)
  return res.json()
}

// ---------------------------------------------------------------------------
// Ingest jobs
// ---------------------------------------------------------------------------

export async function triggerIngest(collection = 'demo', sourceType = 'azure') {
  const res = await fetch(`${BASE}/api/ingest/trigger`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ collection, source_type: sourceType }),
  })
  if (!res.ok) throw new Error(`Trigger failed: ${res.status}`)
  return res.json()
}

export async function getIngestStatus(jobId) {
  const res = await fetch(`${BASE}/api/ingest/status/${jobId}`)
  if (!res.ok) throw new Error(`Status failed: ${res.status}`)
  return res.json()
}

// ---------------------------------------------------------------------------
// Query — SSE stream
// ---------------------------------------------------------------------------

/**
 * Stream the agent trace for a question.
 *
 * Calls `onEvent` for every SSE event (node_complete, done, error).
 * Returns an AbortController — call .abort() to cancel.
 *
 * @param {string} question
 * @param {string} collection
 * @param {Array<{question: string, rewritten_query?: string}>} conversationHistory
 * @param {function} onEvent
 *
 * Event shapes:
 *   {event:'node_complete', node:'rewrite', rewritten_query, changed}
 *   {event:'node_complete', node:'router',  route, reason}
 *   {event:'node_complete', node:'retrieve_vector'|'retrieve_cag'|'retrieve_graph', chunk_count, chunks}
 *   {event:'node_complete', node:'grade',   grade, retry_count}
 *   {event:'node_complete', node:'generate', answer, citations}
 *   {event:'done',          answer, route, citations, ...}
 *   {event:'error',         detail}
 */
export function streamQuery(question, collection, conversationHistory = [], onEvent, role = 'general') {
  const controller = new AbortController()

  ;(async () => {
    let res
    try {
      res = await fetch(`${BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, collection, conversation_history: conversationHistory, role }),
        signal: controller.signal,
      })
    } catch (err) {
      if (err.name !== 'AbortError') onEvent({ event: 'error', detail: err.message })
      return
    }

    if (!res.ok) {
      onEvent({ event: 'error', detail: `HTTP ${res.status}` })
      return
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        // SSE lines are separated by '\n\n'
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw) continue
          try {
            onEvent(JSON.parse(raw))
          } catch {
            // malformed JSON — skip
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') onEvent({ event: 'error', detail: err.message })
    }
  })()

  return controller
}

// ---------------------------------------------------------------------------
// Inline RAGAS eval
// ---------------------------------------------------------------------------

/**
 * Run referenceless RAGAS faithfulness + answer_relevancy for one answer.
 * @param {string} question
 * @param {string} answer
 * @param {string[]} contexts  — retrieved source passages used to generate the answer
 * @returns {Promise<{faithfulness: number|null, answer_relevancy: number|null, latency_ms: number}>}
 */
export async function evalAnswer(question, answer, contexts) {
  const res = await fetch(`${BASE}/api/eval`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, answer, contexts }),
  })
  if (!res.ok) throw new Error(`Eval failed: ${res.status}`)
  return res.json()
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function healthCheck() {
  try {
    const res = await fetch(`${BASE}/healthz`)
    return res.ok
  } catch {
    return false
  }
}
