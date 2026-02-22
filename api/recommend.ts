import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'
import { parsePainting, PAINTING_FIELDS, hmgetToHash, isSpamTitle } from './_lib/parse'

const LIKES_KEY = 'stv:likes'
const DIM = 768

/**
 * "For You" recommendations — tight taste-matching, no exploration.
 *
 * Unlike the feed (which mixes taste + quality + explore for discovery),
 * this returns paintings most similar to what you've liked, period.
 *
 * Strategy scales with likes:
 *   1-2 likes: single averaged taste vector → KNN
 *   3+ likes:  taste poles (farthest-point sampling) → multi-KNN, merged by similarity
 *
 * Quality floor (art_score >= 30) prevents junk, but ranking is pure taste.
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const redis = getRedis()
  const likedIds = await redis.smembers(LIKES_KEY)
  if (!likedIds.length) {
    return res.json({ paintings: [], total: 0 })
  }

  // Fetch embeddings for liked paintings
  const embPipe = redis.pipeline()
  for (const id of likedIds) {
    (embPipe as any).hgetBuffer(`stv:p:${id}`, 'embedding')
  }
  const embResults = await embPipe.exec()

  const embeddings: Array<{ id: string; vec: Float32Array }> = []
  for (let i = 0; i < likedIds.length; i++) {
    const [err, data] = embResults![i]
    if (!err && data && Buffer.isBuffer(data) && data.length >= DIM * 4) {
      const vec = new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + DIM * 4))
      embeddings.push({ id: likedIds[i], vec })
    }
  }

  if (!embeddings.length) {
    return res.json({
      paintings: [], total: 0,
      message: 'No embeddings available yet — run scorer with embeddings first',
    })
  }

  const likedSet = new Set(likedIds)

  // Find taste poles and KNN from each
  let candidates: Array<{ id: string; dist: number; pole: number }>

  if (embeddings.length < 3) {
    // Simple: average all embeddings
    const taste = averageVectors(embeddings.map(e => e.vec))
    const results = await knnQuery(redis, Buffer.from(taste.buffer), 200, likedSet)
    candidates = results.map(r => ({ ...r, pole: 0 }))
  } else {
    // Poles: find diverse facets of taste, KNN from each
    const maxPoles = Math.min(4, Math.max(2, Math.floor(embeddings.length / 2)))
    const poleIndices = findPoles(embeddings.map(e => e.vec), maxPoles)

    const allCandidates: Array<{ id: string; dist: number; pole: number }> = []
    const seen = new Set<string>()

    for (let p = 0; p < poleIndices.length; p++) {
      const poleVec = embeddings[poleIndices[p]].vec
      const results = await knnQuery(redis, Buffer.from(poleVec.buffer), 120, likedSet, seen)
      for (const r of results) {
        seen.add(r.id)
        allCandidates.push({ ...r, pole: p })
      }
    }

    // Sort by distance (similarity) — tightest matches first
    allCandidates.sort((a, b) => a.dist - b.dist)
    candidates = allCandidates
  }

  if (!candidates.length) {
    return res.json({ paintings: [], total: 0 })
  }

  // Fetch painting data
  const fetchIds = candidates.slice(0, 500).map(c => c.id)
  const distMap = new Map(candidates.map(c => [c.id, c.dist]))

  const dataPipe = redis.pipeline()
  for (const id of fetchIds) {
    dataPipe.hmget(`stv:p:${id}`, ...PAINTING_FIELDS)
  }
  const dataResults = await dataPipe.exec()

  const paintings: any[] = []
  for (let i = 0; i < fetchIds.length; i++) {
    const [err, values] = dataResults![i]
    if (err || !values) continue

    const hash = hmgetToHash(values as (string | null)[])
    if (!Object.keys(hash).length) continue

    const painting = parsePainting(hash, fetchIds[i])
    if (!painting.images.length) continue
    if (isSpamTitle(painting.title)) continue

    const artScore = painting.art_score || 0
    if (artScore < 30) continue

    const dist = distMap.get(fetchIds[i]) ?? 2.0
    const similarity = 1 - dist

    // Pure taste ranking: similarity is king, art_score is just a light tiebreaker
    const score = similarity * 100 + artScore * 0.1

    paintings.push({
      ...painting,
      taste_similarity: Math.round(similarity * 100) / 100,
      blended_score: Math.round(score * 10) / 10,
    })
  }

  paintings.sort((a: any, b: any) => b.blended_score - a.blended_score)

  res.setHeader('Cache-Control', 'no-cache')
  return res.json({
    paintings,
    total: paintings.length,
    poles: embeddings.length >= 3 ? Math.min(4, Math.floor(embeddings.length / 2)) : 1,
  })
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
  return dot
}

function findPoles(vecs: Float32Array[], maxPoles: number): number[] {
  if (vecs.length <= maxPoles) return vecs.map((_, i) => i)

  const poles = [0]
  while (poles.length < maxPoles) {
    let bestIdx = -1
    let bestMinDist = -Infinity

    for (let i = 0; i < vecs.length; i++) {
      if (poles.includes(i)) continue
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
): Promise<Array<{ id: string; dist: number }>> {
  try {
    const knnResults = await redis.call(
      'FT.SEARCH', 'stv:vec_idx',
      `*=>[KNN ${count} @embedding $BLOB AS dist]`,
      'PARAMS', '2', 'BLOB', queryBuffer,
      'RETURN', '1', 'dist',
      'SORTBY', 'dist', 'ASC',
      'LIMIT', '0', String(count),
      'DIALECT', '2',
    )

    const results: Array<{ id: string; dist: number }> = []
    for (let i = 1; i < knnResults.length; i += 2) {
      const docId = String(knnResults[i]).replace('stv:p:', '')
      if (excludeSet.has(docId) || (alsoExclude && alsoExclude.has(docId))) continue

      const fields = knnResults[i + 1]
      let dist = 2.0
      if (Array.isArray(fields)) {
        for (let j = 0; j < fields.length; j += 2) {
          if (String(fields[j]) === 'dist') dist = parseFloat(String(fields[j + 1]))
        }
      }
      results.push({ id: docId, dist })
    }
    return results
  } catch {
    return []
  }
}
