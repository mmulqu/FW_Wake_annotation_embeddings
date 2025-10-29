// cloudflare-worker.js
// Endpoints:
// - POST /search_chunk        → search a SLICE of embeddings.bin (range-read from R2; low memory/CPU)
// - POST /search              → optional single-slice shortcut (falls back to /search_chunk for small sets)
// - /upload/<key>?action=...  → multipart API (mpu-create / mpu-uploadpart / mpu-complete / mpu-abort / get)

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, PUT, GET, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Access-Control-Max-Age": "86400",
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    // 🔝 EARLY: /meta
    if (request.method === "GET" && (url.pathname === "/meta" || url.pathname === "/meta/")) {
      console.log("hit /meta");
      try {
        const meta = await loadMeta(env, 1536);
        return jsonResponse(meta);
      } catch (e) {
        console.error("meta error:", e);
        return jsonResponse({ error: String(e?.message || e) }, 500);
      }
    }

    if (url.pathname.startsWith("/upload/")) {
      return handleUpload(request, env, url);
    }

    // One-off admin import route: stream from R2 -> Vectorize index in batches
    if (request.method === "POST" && url.pathname === "/admin/import") {
      if (request.headers.get("Authorization") !== `Bearer ${env.ADMIN_TOKEN}`) {
        return jsonResponse({ error: "unauthorized" }, 401);
      }
      try {
        const res = await importFromR2ToVectorize(env, { batchSize: 1000 });
        return jsonResponse({ ok: true, ...res });
      } catch (e) {
        return jsonResponse({ error: String(e?.message || e) }, 500);
      }
    }

    // Tolerant search route (kept for backwards compat). Will only work for tiny files.
    const isSearchPath =
      url.pathname === "/search" ||
      url.pathname === "/search/" ||
      url.pathname === "" ||
      url.pathname === "/";

      if (request.method === "POST" && isSearchPath) {
      try {
        const { query, top_k = 20 } = await request.json();
          if (!query?.trim()) return jsonResponse({ error: "Query required" }, 400);
      
          // 1) embed
          const embedding = await embedQuery(query, env.OPENAI_API_KEY);
      
          // Guard: must be 1536-d
          if (!embedding || !Array.isArray(embedding) || embedding.length !== 1536) {
            console.error("Embedding length bad:", Array.isArray(embedding) ? embedding.length : embedding);
            return jsonResponse({
              error: "Bad embedding from provider",
              got_length: Array.isArray(embedding) ? embedding.length : null,
              expected_length: 1536
            }, 500);
          }
      
          // NEW: cast & sanitize
          const vec = new Float32Array(embedding.map((x) => Number.isFinite(x) ? x : 0));
          console.log("vector typed length:", vec.length);
      
          // 2) query Vectorize
          const resp = await env.FW_INDEX.query(vec, {
            topK: top_k,
            returnValues: false,
            returnMetadata: true,
          });
      
          // 3) enrich
          let textLookup = {};
          const tlObj = await env.FW_BUCKET.get('text_lookup.json');
          if (tlObj) textLookup = JSON.parse(await tlObj.text());
      
          const results = (resp.matches || []).map(m => {
            const entry = m?.metadata?.page_line ? textLookup[m.metadata.page_line] : null;
            return {
              similarity: m.score,
              ...(m.metadata || {}),
              wake_text: typeof entry === "string" ? entry : (entry?.text ?? null),
            };
          });
      
          return jsonResponse({ results, query, num_results: results.length });
        } catch (e) {
          console.error("search error:", e);
          return jsonResponse({ error: String(e?.message || e) }, 500);
        }
      }

    // New: chunked search API
    if (request.method === "POST" && url.pathname === "/search_chunk") {
      try {
        const { query, start = 0, count, top_k = 20 } = await request.json();
        if (!query || !query.trim()) return jsonResponse({ error: "Query required" }, 400);
        if (count == null || count <= 0) return jsonResponse({ error: "count must be > 0" }, 400);
        if (start < 0) return jsonResponse({ error: "start must be >= 0" }, 400);

        const queryEmbedding = await embedQuery(query, env.OPENAI_API_KEY);
        const queryNorm = normalize(queryEmbedding);

        const meta = await loadMeta(env, queryEmbedding.length);
        const { dim, total_vectors } = meta;

        const safeStart = Math.min(start, Math.max(0, total_vectors));
        const safeCount = Math.max(0, Math.min(count, total_vectors - safeStart));
        if (safeCount === 0) {
          return jsonResponse({ results: [], slice: { start: safeStart, count: safeCount, total_vectors } });
        }

        const top = await searchSlice(env, "embeddings.bin", queryNorm, dim, safeStart, safeCount, top_k);

        // Just return indices + scores (small). Client will enrich with annotations via a separate call or once per UI load.
        return jsonResponse({
          results: top, // [{index, similarity}]
          slice: { start: safeStart, count: safeCount, total_vectors, dim },
        });
      } catch (error) {
        console.error("search_chunk error:", error);
        return jsonResponse({ error: error.message, details: error.stack }, 500);
      }
    }

    return jsonResponse({ error: "Not found. Use POST /search_chunk, POST /search, or /upload/*" }, 404);
  },
};

/* ------------------------ Multipart upload handlers ------------------------ */
async function handleUpload(request, env, url) {
  const key = url.pathname.replace(/^\/upload\//, "").trim();
  let action = url.searchParams.get("action");
  if (!key) return jsonResponse({ error: "Missing key in path" }, 400);

  if (!action) {
    if (request.method === "POST") {
      const body = await tryJson(request.clone());
      action = body && body.parts ? "mpu-complete" : "mpu-create";
    } else if (request.method === "PUT") action = "mpu-uploadpart";
  }

  try {
    switch (request.method) {
      case "POST":
        if (action === "mpu-create") {
          const mpu = await env.FW_BUCKET.createMultipartUpload(key);
          return jsonResponse({ key: mpu.key, uploadId: mpu.uploadId });
        }
        if (action === "mpu-complete") {
          const uploadId = url.searchParams.get("uploadId");
          if (!uploadId) return jsonResponse({ error: "Missing uploadId" }, 400);
          const body = await tryJson(request);
          if (!body?.parts || !Array.isArray(body.parts)) return jsonResponse({ error: "Missing parts" }, 400);
          const mpu = env.FW_BUCKET.resumeMultipartUpload(key, uploadId);
            const obj = await mpu.complete(body.parts);
            return new Response(null, { status: 200, headers: { ...CORS_HEADERS, etag: obj.httpEtag } });
        }
        return jsonResponse({ error: `Unknown POST action: ${action}` }, 400);

      case "PUT":
        if (action === "mpu-uploadpart") {
          const uploadId = url.searchParams.get("uploadId");
          const partNumber = Number(url.searchParams.get("partNumber"));
          if (!uploadId || !partNumber) return jsonResponse({ error: "Missing uploadId or partNumber" }, 400);
          if (!request.body) return jsonResponse({ error: "Missing request body" }, 400);
          const mpu = env.FW_BUCKET.resumeMultipartUpload(key, uploadId);
            const uploaded = await mpu.uploadPart(partNumber, request.body);
          return jsonResponse(uploaded);
        }
        return jsonResponse({ error: `Unknown PUT action: ${action}` }, 400);

      case "GET":
        if (action === "get") {
          const obj = await env.FW_BUCKET.get(key);
          if (!obj) return jsonResponse({ error: "Not Found" }, 404);
          const headers = new Headers(CORS_HEADERS);
          obj.writeHttpMetadata(headers);
          headers.set("etag", obj.httpEtag);
          return new Response(obj.body, { headers });
        }
        if (action === "ping") return jsonResponse({ ok: true, key });
        return jsonResponse({ error: `Unknown GET action: ${action}` }, 400);

      case "DELETE":
        if (action === "mpu-abort") {
          const uploadId = url.searchParams.get("uploadId");
          if (!uploadId) return jsonResponse({ error: "Missing uploadId" }, 400);
          const mpu = env.FW_BUCKET.resumeMultipartUpload(key, uploadId);
          await mpu.abort();
          return new Response(null, { status: 204, headers: CORS_HEADERS });
        }
        if (action === "delete") {
          await env.FW_BUCKET.delete(key);
          return new Response(null, { status: 204, headers: CORS_HEADERS });
        }
        return jsonResponse({ error: `Unknown DELETE action: ${action}` }, 400);
    }
  } catch (err) {
    console.error("Upload route error:", err);
    return jsonResponse({ error: String(err?.message || err) }, 500);
  }

  return new Response("Method Not Allowed", {
    status: 405,
    headers: { ...CORS_HEADERS, Allow: "PUT, POST, GET, DELETE, OPTIONS" },
  });
}

/* ----------------------------- Search helpers ----------------------------- */

async function embedQuery(text, apiKey) {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model: "text-embedding-3-small", input: text }),
  });

  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(`OpenAI API error: ${res.status} ${msg}`);
  }

  const data = await res.json().catch(() => ({}));
  const v = data?.data?.[0]?.embedding;
  console.log("embed length:", Array.isArray(v) ? v.length : null);
  return v;
}

async function loadAnnotations(env) {
  const annotationsObj = await env.FW_BUCKET.get("annotations.json");
  const textLookupObj = await env.FW_BUCKET.get("text_lookup.json");
  if (!annotationsObj) throw new Error("annotations.json not found");
  const annotations = JSON.parse(await annotationsObj.text());
  const textLookup = textLookupObj ? JSON.parse(await textLookupObj.text()) : {};
  return { annotations, textLookup };
}

async function loadMeta(env, defaultDim) {
  const metaObj = await env.FW_BUCKET.get("metadata.json");
  if (metaObj) {
    const m = JSON.parse(await metaObj.text());
    if (m && m.dim && m.total_vectors) return m;
    if (m && m.dim) return { dim: m.dim, total_vectors: m.total_vectors ?? 0 };
  }
  // Fallback: try to derive total_vectors from object size
  const head = await env.FW_BUCKET.head("embeddings.bin");
  const size = head?.size ?? 0;
  const dim = defaultDim;
  const total_vectors = size ? Math.floor(size / (dim * 4)) : 0;
  return { dim, total_vectors };
}

// One-off import helper: stream embeddings from R2 and upsert to Vectorize
async function importFromR2ToVectorize(env, { batchSize = 1000 } = {}) {
  const annObj = await env.FW_BUCKET.get('annotations.json');
  if (!annObj) throw new Error('annotations.json not found');
  const annotations = JSON.parse(await annObj.text());

  const head = await env.FW_BUCKET.head('embeddings.bin');
  if (!head?.size) throw new Error('embeddings.bin missing/empty');
  const dim = 1536; // embedding dim
  const totalVectors = Math.floor(head.size / (dim * 4));

  if (annotations.length !== totalVectors) {
    throw new Error(`Count mismatch: annotations=${annotations.length} vs vectors=${totalVectors}`);
  }

  const BYTES_PER_VECTOR = dim * 4;
  let start = 0;

  while (start < totalVectors) {
    const count = Math.min(batchSize, totalVectors - start);
    const offset = start * BYTES_PER_VECTOR;
    const length = count * BYTES_PER_VECTOR;

    const part = await env.FW_BUCKET.get('embeddings.bin', { range: { offset, length } });
    if (!part) throw new Error(`Range read failed at start=${start}`);

    const buf = await part.arrayBuffer();
    const floats = new Float32Array(buf); // length = count * dim

    const payload = new Array(count);
    for (let i = 0; i < count; i++) {
      const id = String(start + i);
      const values = floats.subarray(i * dim, (i + 1) * dim);
      const meta = annotations[start + i] || {};
      payload[i] = { id, values, metadata: meta };
    }

    await env.FW_INDEX.upsert(payload);
    start += count;
  }

  return { imported: totalVectors };
}

// Search a slice: [start, start+count) vectors
async function searchSlice(env, key, queryNorm, dim, start, count, k) {
  const BYTES_PER_VECTOR = dim * 4;
  const offset = start * BYTES_PER_VECTOR;
  const length = count * BYTES_PER_VECTOR;

  const obj = await env.FW_BUCKET.get(key, { range: { offset, length } });
  if (!obj) throw new Error(`Missing ${key}`);
  const reader = obj.body.getReader();

  let idx = start;
  const heap = [];
  let leftover = new Uint8Array(0);

  for (;;) {
    const { value, done } = await reader.read();
    const chunk = value ? (leftover.length ? concat(leftover, value) : value) : leftover;
    if (!chunk || chunk.length === 0) { if (done) break; leftover = new Uint8Array(0); continue; }

    const usable = chunk.length - (chunk.length % BYTES_PER_VECTOR);
    const dv = new DataView(chunk.buffer, chunk.byteOffset, usable);
    let off = 0;
    while (off < usable) {
      let dot = 0;
      for (let i = 0; i < dim; i++) {
        dot += queryNorm[i] * dv.getFloat32(off + i * 4, true);
      }
      minHeapPush(heap, { index: idx, similarity: dot }, k);
      idx++;
      off += BYTES_PER_VECTOR;
    }
    leftover = chunk.slice(usable);
    if (done) break;
  }

  heap.sort((a, b) => b.similarity - a.similarity);
  return heap;
}

/* ------------------------------- Min-heap K ------------------------------- */
function minHeapPush(heap, item, k) {
  if (heap.length < k) { heap.push(item); bubbleUp(heap); }
  else if (item.similarity > heap[0].similarity) { heap[0] = item; sinkDown(heap); }
}
function bubbleUp(h){ for(let i=h.length-1;i>0;){const p=(i-1)>>1; if(h[p].similarity<=h[i].similarity)break; [h[p],h[i]]=[h[i],h[p]]; i=p; } }
function sinkDown(h){ for(let i=0;;){ let l=i*2+1,r=l+1,s=i; if(l<h.length&&h[l].similarity<h[s].similarity)s=l; if(r<h.length&&h[r].similarity<h[s].similarity)s=r; if(s===i)break; [h[i],h[s]]=[h[s],h[i]]; i=s; } }

/* --------------------------- Vector math helpers --------------------------- */
function normalize(vector) {
  let sum = 0;
  for (let i = 0; i < vector.length; i++) sum += vector[i] * vector[i];
  const norm = Math.sqrt(sum) || 1;
  // Return a plain Array to keep everything serializable
  const out = new Array(vector.length);
  for (let i = 0; i < vector.length; i++) out[i] = vector[i] / norm;
  return out;
}

/* -------------------------------- Utilities -------------------------------- */
function concat(a, b) { const out = new Uint8Array(a.length + b.length); out.set(a, 0); out.set(b, a.length); return out; }
function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json", ...CORS_HEADERS } });
}
async function tryJson(request) { try { return await request.json(); } catch { return null; } }
