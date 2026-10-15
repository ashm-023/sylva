const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1'
// Security: never put GEMINI_API_KEY / OPENTOPO / tokens in VITE_* vars.
// Those are embedded in the public JS bundle. Secrets belong on the server only.

const GEOCODE_TIMEOUT_MS = 12_000
// Must outlast the server's own deadline (RECOMMEND_TIMEOUT_S = 80 s in
// app/routers/recommendations.py) so the user sees the server's 504 message, not a blind abort.
const RECOMMEND_TIMEOUT_MS = 85_000

async function fetchWithTimeout(url, options = {}, timeoutMs = 30_000) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } catch (e) {
    if (e?.name === 'AbortError') {
      throw new Error('Request timed out — the server took too long. Try again.')
    }
    throw e
  } finally {
    clearTimeout(timer)
  }
}

export async function geocode(query) {
  const url = `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(query)}`
  const res = await fetchWithTimeout(
    url,
    {
      headers: {
        Accept: 'application/json',
      },
    },
    GEOCODE_TIMEOUT_MS,
  )
  if (!res.ok) throw new Error('Geocoding failed')
  const data = await res.json()
  if (!data.length) throw new Error('Location not found — try a town or coordinates')
  const parts = (data[0].display_name || '').split(',').map((s) => s.trim()).filter(Boolean)
  return {
    lat: parseFloat(data[0].lat),
    lon: parseFloat(data[0].lon),
    label: parts.slice(0, 3).join(', '),
    country: parts.at(-1) || null,
  }
}

export async function fetchRecommendations({ lat, lon, country, topN = 10 }) {
  const params = new URLSearchParams({
    lat: String(lat),
    lon: String(lon),
    radius_km: '5',
    top_n: String(topN),
  })
  if (country) params.set('country', country)

  const res = await fetchWithTimeout(
    `${API_BASE}/farm/recommendations?${params}`,
    {},
    RECOMMEND_TIMEOUT_MS,
  )
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      detail = typeof body.detail === 'string' ? body.detail : detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json()
}

// ── Nature library / industry R&D map ────────────────────────────────────────

async function natureJson(path, options = {}, timeoutMs = 30_000) {
  const res = await fetchWithTimeout(`${API_BASE}/nature${path}`, options, timeoutMs)
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    if (res.status === 504) detail = 'The question took too long to answer. Try again or shorten it.'
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail)) {
        const msgs = body.detail.map((d) => d?.msg).filter(Boolean)
        if (msgs.length) detail = msgs.join('; ')
      }
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json()
}

function withParams(path, params) {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null && v !== '') sp.set(k, String(v))
  }
  const qs = sp.toString()
  return qs ? `${path}?${qs}` : path
}

export const natureExportUrl = (db, format) => `${API_BASE}/nature/export/${db}?format=${format}`

export const fetchNatureStats = () => natureJson('/stats')
export const fetchNatureMeta = () => natureJson('/meta')
export const searchLibrary = (params) => natureJson(withParams('/library', params))
export const fetchLibraryRecord = (id) => natureJson(`/library/${encodeURIComponent(id)}`)
export const searchMap = (params) => natureJson(withParams('/map', params))
export const fetchSectorGaps = (sectorId, limit = 10) =>
  natureJson(withParams(`/sectors/${encodeURIComponent(sectorId)}/gaps`, { limit }))
export const askNature = (body) =>
  natureJson(
    '/ask',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    75_000,
  )

// ── Sylva-wild ───────────────────────────────────────────────────────────────

async function wildJson(path, options = {}, timeoutMs = 30_000) {
  const res = await fetchWithTimeout(`${API_BASE}/wild${path}`, options, timeoutMs)
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    if (res.status === 504) detail = 'That took too long to answer. Try again or shorten your description.'
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail)) {
        const msgs = body.detail.map((d) => d?.msg).filter(Boolean)
        if (msgs.length) detail = msgs.join('; ')
      }
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json()
}

export const fetchWildMeta = () => wildJson('/meta')
export const askWild = (body) =>
  wildJson(
    '/ask',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
    90_000,
  )
