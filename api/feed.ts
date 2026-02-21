import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'
import { parsePainting, PAINTING_FIELDS, hmgetToHash } from './_lib/parse'

const LIKES_KEY = 'stv:likes'
const DIM = 768
const DEFAULT_SIZE = 20
const MAX_SEEN = 300

/**
 * Smart feed algorithm with live taste learning.
 *
 * POST /api/feed  { seen?: string[], size?: number }
 *
 * Taste strategy scales with likes:
 *   0 likes:  70% quality, 30% explore (cold start)
 *   1-2 likes: single averaged taste vector → KNN
 *   3+ likes:  taste poles (farthest-point sampling) → multi-KNN
 *              each pole = a distinct facet of your taste
 *
 * Light signals (not hard filters):
 *   - Style affinity: soft boost for paintings matching your preferred styles
 *   - Price range: gentle nudge toward your typical price band
 *
 * Interleaving: taste slots rotate across poles, mixed with quality + explore.
 * Seen-set exclusion prevents repeats across fetches.
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  let seen: string[] = []
  let size = DEFAULT_SIZE

  if (req.method === 'POST') {
    seen = req.body?.seen || []
    size = Math.min(req.body?.size || DEFAULT_SIZE, 60)
  } else if (req.method === 'GET') {
    const seenParam = (req.query.seen as string) || ''
    seen = seenParam ? seenParam.split(',').filter(Boolean) : []
    size = Math.min(parseInt(req.query.size as string) || DEFAULT_SIZE, 60)
  } else {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const redis = getRedis()
  const likedIds = await redis.smembers(LIKES_KEY)
  const likedSet = new Set(likedIds)
  const seenSet = new Set(seen.slice(-MAX_SEEN))
  const excludeSet = new Set([...Array.from(likedSet), ...Array.from(seenSet)])

  // ── Taste candidates ──
  let tasteCandidates: FeedItem[] = []
  let numPoles = 0
  let styleAffinity: Record<string, number> = {}
  let priceCenter = 0

  if (likedIds.length >= 3) {
    const result = await getPoleTaste(redis, likedIds, excludeSet)
    tasteCandidates = result.candidates
    numPoles = result.poles
    styleAffinity = result.styleAffinity
    priceCenter = result.priceCenter
  } else if (likedIds.length > 0) {
    const result = await getSimpleTaste(redis, likedIds, excludeSet)
    tasteCandidates = result.candidates
    styleAffinity = result.styleAffinity
    priceCenter = result.priceCenter
  }

  // ── Quality candidates ──
  const tasteIdSet = new Set(tasteCandidates.map(t => t.id))
  const qualityExclude = new Set([...Array.from(excludeSet), ...Array.from(tasteIdSet)])
  const qualityCandidates = await getQualityCandidates(redis, qualityExclude, size)

  // ── Explore candidates ──
  const allUsedIds = new Set([
    ...Array.from(qualityExclude),
    ...qualityCandidates.map(q => q.id),
  ])
  const exploreCandidates = await getExploreCandidates(redis, allUsedIds, size)

  // ── Interleave ──
  const feedItems = interleave(tasteCandidates, qualityCandidates, exploreCandidates, size)

  if (!feedItems.length) {
    return res.json({ paintings: [], total: 0, liked: likedIds, meta: { poles: 0, strategy: 'empty' } })
  }

  // ── Fetch painting data ──
  const pipe = redis.pipeline()
  for (const item of feedItems) {
    pipe.hmget(`stv:p:${item.id}`, ...PAINTING_FIELDS)
  }
  const results = await pipe.exec()

  let paintings = feedItems
    .map((item, i) => {
      const [err, values] = results![i]
      if (err || !values) return null
      const hash = hmgetToHash(values as (string | null)[])
      if (!Object.keys(hash).length) return null
      const p = parsePainting(hash, item.id)
      if (!p.images.length) return null
      return { ...p, _reason: item.reason, _pole: item.pole }
    })
    .filter((p): p is NonNullable<typeof p> => p !== null)

  // ── Soft re-rank: light style/price boost ──
  if (likedIds.length >= 3 && (Object.keys(styleAffinity).length || priceCenter > 0)) {
    paintings = softRerank(paintings, styleAffinity, priceCenter)
  }

  const strategy = likedIds.length === 0 ? 'cold' : likedIds.length < 3 ? 'simple' : 'poles'
  res.setHeader('Cache-Control', 'no-cache')
  return res.json({
    paintings,
    total: paintings.length,
    liked: likedIds,
    meta: { poles: numPoles, strategy, styles: styleAffinity },
  })
}


// ─── Types ───

interface FeedItem {
  id: string
  reason: 'taste' | 'quality' | 'explore'
  pole?: number
}


// ─── Taste: simple averaged vector (1-2 likes) ───

async function getSimpleTaste(redis: any, likedIds: string[], excludeSet: Set<string>) {
  const { embeddings, styleAffinity, priceCenter } = await fetchLikedData(redis, likedIds)
  if (!embeddings.length) return { candidates: [] as FeedItem[], styleAffinity, priceCenter }

  // Average all embeddings
  const taste = averageVectors(embeddings.map(e => e.vec))
  const tasteBuffer = Buffer.from(taste.buffer)

  const candidates = await knnQuery(redis, tasteBuffer, 80, excludeSet)
  return {
    candidates: candidates.map(id => ({ id, reason: 'taste' as const })),
    styleAffinity,
    priceCenter,
  }
}


// ─── Taste: diverse poles (3+ likes) ───

async function getPoleTaste(redis: any, likedIds: string[], excludeSet: Set<string>) {
  const { embeddings, styleAffinity, priceCenter } = await fetchLikedData(redis, likedIds)
  if (!embeddings.length) {
    return { candidates: [] as FeedItem[], poles: 0, styleAffinity, priceCenter }
  }

  // Find diverse poles via farthest-point sampling
  const maxPoles = Math.min(3, Math.max(1, Math.floor(embeddings.length / 2)))
  const poleIndices = findPoles(embeddings.map(e => e.vec), maxPoles)

  // KNN from each pole
  const poleResults: string[][] = []
  const seenInTaste = new Set<string>()

  for (const poleIdx of poleIndices) {
    const poleBuffer = Buffer.from(embeddings[poleIdx].vec.buffer)
    const results = await knnQuery(redis, poleBuffer, 60, excludeSet, seenInTaste)
    for (const id of results) seenInTaste.add(id)
    poleResults.push(results)
  }

  // Round-robin across poles: P0[0], P1[0], P2[0], P0[1], P1[1], ...
  const candidates: FeedItem[] = []
  const maxLen = Math.max(...poleResults.map(p => p.length))
  for (let i = 0; i < maxLen; i++) {
    for (let p = 0; p < poleResults.length; p++) {
      if (i < poleResults[p].length) {
        candidates.push({ id: poleResults[p][i], reason: 'taste', pole: p })
      }
    }
  }

  return {
    candidates,
    poles: poleIndices.length,
    styleAffinity,
    priceCenter,
  }
}


// ─── Shared: fetch liked embeddings + compute style/price signals ───

async function fetchLikedData(redis: any, likedIds: string[]) {
  // Fetch embeddings + clip_styles + price in one pipeline
  const pipe = redis.pipeline()
  for (const id of likedIds) {
    (pipe as any).hgetBuffer(`stv:p:${id}`, 'embedding')
    pipe.hmget(`stv:p:${id}`, 'clip_styles', 'price')
  }
  const results = await pipe.exec()

  const embeddings: Array<{ id: string, vec: Float32Array }> = []
  const styleCounts: Record<string, number> = {}
  const prices: number[] = []

  for (let i = 0; i < likedIds.length; i++) {
    const embIdx = i * 2
    const metaIdx = i * 2 + 1

    // Embedding
    const [embErr, embData] = results![embIdx]
    if (!embErr && embData && Buffer.isBuffer(embData) && embData.length >= DIM * 4) {
      const vec = new Float32Array(
        embData.buffer.slice(embData.byteOffset, embData.byteOffset + DIM * 4)
      )
      embeddings.push({ id: likedIds[i], vec })
    }

    // Style + price
    const [metaErr, metaValues] = results![metaIdx]
    if (!metaErr && metaValues) {
      const [stylesStr, priceStr] = metaValues as [string | null, string | null]
      if (stylesStr) {
        try {
          const styles = JSON.parse(stylesStr) as Array<{ style: string, confidence: number }>
          for (const s of styles) {
            styleCounts[s.style] = (styleCounts[s.style] || 0) + s.confidence
          }
        } catch {}
      }
      if (priceStr) {
        const p = parseFloat(priceStr)
        if (p > 0) prices.push(p)
      }
    }
  }

  // Normalize style counts to 0-1
  const totalStyle = Object.values(styleCounts).reduce((a, b) => a + b, 0)
  const styleAffinity: Record<string, number> = {}
  if (totalStyle > 0) {
    for (const [style, count] of Object.entries(styleCounts)) {
      styleAffinity[style] = count / totalStyle
    }
  }

  // Median price as center
  const sortedPrices = prices.sort((a, b) => a - b)
  const priceCenter = sortedPrices.length
    ? sortedPrices[Math.floor(sortedPrices.length / 2)]
    : 0

  return { embeddings, styleAffinity, priceCenter }
}


// ─── Vector math ───

function averageVectors(vecs: Float32Array[]): Float32Array {
  const avg = new Float32Array(DIM)
  for (const v of vecs) {
    for (let i = 0; i < DIM; i++) avg[i] += v[i]
  }
  let norm = 0
  for (let i = 0; i < DIM; i++) {
    avg[i] /= vecs.length
    norm += avg[i] * avg[i]
  }
  norm = Math.sqrt(norm)
  if (norm > 0) for (let i = 0; i < DIM; i++) avg[i] /= norm
  return avg
}

function cosineSim(a: Float32Array, b: Float32Array): number {
  let dot = 0
  for (let i = 0; i < DIM; i++) dot += a[i] * b[i]
  return dot // already normalized
}

function findPoles(vecs: Float32Array[], maxPoles: number): number[] {
  if (vecs.length <= maxPoles) return vecs.map((_, i) => i)

  // Start with first vector
  const poles = [0]

  while (poles.length < maxPoles) {
    let bestIdx = -1
    let bestMinDist = -Infinity

    for (let i = 0; i < vecs.length; i++) {
      if (poles.includes(i)) continue
      // Min distance to any existing pole
      let minDist = Infinity
      for (const pi of poles) {
        const dist = 1 - cosineSim(vecs[i], vecs[pi])
        if (dist < minDist) minDist = dist
      }
      if (minDist > bestMinDist) {
        bestMinDist = minDist
        bestIdx = i
      }
    }

    if (bestIdx >= 0) poles.push(bestIdx)
    else break
  }

  return poles
}


// ─── KNN query ───

async function knnQuery(
  redis: any,
  queryBuffer: Buffer,
  count: number,
  excludeSet: Set<string>,
  alsoExclude?: Set<string>,
): Promise<string[]> {
  try {
    const knnResults = await (redis as any).call(
      'FT.SEARCH', 'stv:vec_idx',
      `*=>[KNN ${count} @embedding $BLOB AS dist]`,
      'PARAMS', '2', 'BLOB', queryBuffer,
      'RETURN', '1', 'dist',
      'SORTBY', 'dist', 'ASC',
      'LIMIT', '0', String(count),
      'DIALECT', '2',
    )

    const ids: string[] = []
    for (let i = 1; i < knnResults.length; i += 2) {
      const docId = String(knnResults[i]).replace('stv:p:', '')
      if (!excludeSet.has(docId) && (!alsoExclude || !alsoExclude.has(docId))) {
        ids.push(docId)
      }
    }
    return ids
  } catch {
    return []
  }
}


// ─── Quality + explore candidates ───

async function getQualityCandidates(redis: any, excludeSet: Set<string>, size: number): Promise<FeedItem[]> {
  const topIds = await redis.zrevrange('stv:idx:art_score', 0, 299)
  const filtered = topIds.filter((id: string) => !excludeSet.has(id))
  return filtered.slice(0, size).map((id: string) => ({ id, reason: 'quality' as const }))
}

async function getExploreCandidates(redis: any, excludeSet: Set<string>, size: number): Promise<FeedItem[]> {
  const totalPaintings: number = await redis.zcard('stv:idx:art_score')
  if (totalPaintings === 0) return []

  const midStart = Math.floor(totalPaintings * 0.15)
  const midEnd = Math.floor(totalPaintings * 0.65)
  const midIds: string[] = await redis.zrevrange('stv:idx:art_score', midStart, midEnd)
  const available = midIds.filter(id => !excludeSet.has(id))

  // Shuffle
  for (let i = available.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [available[i], available[j]] = [available[j], available[i]]
  }

  return available.slice(0, size).map(id => ({ id, reason: 'explore' as const }))
}


// ─── Interleave ───

function interleave(taste: FeedItem[], quality: FeedItem[], explore: FeedItem[], maxSize: number): FeedItem[] {
  const result: FeedItem[] = []
  let ti = 0, qi = 0, ei = 0
  const seen = new Set<string>()

  function add(item: FeedItem): boolean {
    if (!seen.has(item.id)) {
      seen.add(item.id)
      result.push(item)
      return true
    }
    return false
  }

  const hasTaste = taste.length > 0

  while (result.length < maxSize) {
    const before = result.length

    if (hasTaste) {
      // Pattern: T Q T T E
      while (ti < taste.length && !add(taste[ti])) ti++
      if (ti < taste.length) ti++
      while (qi < quality.length && !add(quality[qi])) qi++
      if (qi < quality.length) qi++
      while (ti < taste.length && !add(taste[ti])) ti++
      if (ti < taste.length) ti++
      while (ti < taste.length && !add(taste[ti])) ti++
      if (ti < taste.length) ti++
      while (ei < explore.length && !add(explore[ei])) ei++
      if (ei < explore.length) ei++
    } else {
      // Cold start: Q Q Q E
      while (qi < quality.length && !add(quality[qi])) qi++
      if (qi < quality.length) qi++
      while (qi < quality.length && !add(quality[qi])) qi++
      if (qi < quality.length) qi++
      while (qi < quality.length && !add(quality[qi])) qi++
      if (qi < quality.length) qi++
      while (ei < explore.length && !add(explore[ei])) ei++
      if (ei < explore.length) ei++
    }

    if (result.length === before) break
  }

  return result.slice(0, maxSize)
}


// ─── Soft re-rank: light style/price nudge ───

function softRerank(
  paintings: any[],
  styleAffinity: Record<string, number>,
  priceCenter: number,
): any[] {
  if (!paintings.length) return paintings

  // Light boost: add a small bonus to paintings matching style/price preferences
  // This is a NUDGE, not a hard sort — max ±5% position shift
  const scored = paintings.map((p, originalIdx) => {
    let boost = 0

    // Style boost: +0.03 per matching style weighted by affinity
    if (p.clip_styles && styleAffinity) {
      try {
        const styles = typeof p.clip_styles === 'string' ? JSON.parse(p.clip_styles) : p.clip_styles
        for (const s of styles) {
          if (styleAffinity[s.style]) {
            boost += styleAffinity[s.style] * s.confidence * 0.03
          }
        }
      } catch {}
    }

    // Price proximity: tiny boost for paintings near your price center
    // Only if price exists and center is known
    if (priceCenter > 0 && p.price) {
      const ratio = Math.min(p.price, priceCenter) / Math.max(p.price, priceCenter)
      boost += ratio * 0.01 // max +0.01 when exact match
    }

    return { painting: p, originalIdx, boost }
  })

  // Sort within small windows (groups of 5) to preserve overall interleave structure
  // but allow light shuffling within each window
  const result: any[] = []
  for (let i = 0; i < scored.length; i += 5) {
    const window = scored.slice(i, i + 5)
    window.sort((a, b) => b.boost - a.boost)
    result.push(...window.map(s => s.painting))
  }

  return result
}
