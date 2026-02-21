import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'
import { parsePainting, PAINTING_FIELDS, hmgetToHash } from './_lib/parse'

const LIKES_KEY = 'stv:likes'
const DIM = 768
const FEED_SIZE = 60

/**
 * Feed algorithm: returns a curated sequence of paintings optimized for discovery.
 *
 * Mix strategy (when likes exist):
 *   - 50% taste-similar (KNN from taste vector)
 *   - 30% top quality (high art_score, not already in taste results)
 *   - 20% exploration (random from mid-quality pool)
 *
 * Interleaving: taste, quality, taste, taste, explore, repeat
 * This prevents monotony while keeping the feed feeling personalized.
 *
 * When no likes: 70% top quality, 30% random exploration.
 */
export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const redis = getRedis()
  const likedIds = await redis.smembers(LIKES_KEY)
  const likedSet = new Set(likedIds)

  let tasteIds: string[] = []
  let qualityIds: string[] = []
  let exploreIds: string[] = []

  if (likedIds.length > 0) {
    // Try to get taste-similar paintings via KNN
    tasteIds = await getTasteIds(redis, likedIds, likedSet)
  }

  // Get top quality paintings (high art_score, not liked, not in taste results)
  const tasteSet = new Set(tasteIds)
  const topIds = await redis.zrevrange('stv:idx:art_score', 0, 299)
  qualityIds = topIds.filter(id => !likedSet.has(id) && !tasteSet.has(id))

  // Get random exploration paintings (mid-quality range)
  const totalPaintings = await redis.zcard('stv:idx:art_score')
  if (totalPaintings > 0) {
    // Sample from the middle of the quality range
    const midStart = Math.floor(totalPaintings * 0.2)
    const midEnd = Math.floor(totalPaintings * 0.7)
    const midIds = await redis.zrevrange('stv:idx:art_score', midStart, midEnd)
    const usedSet = new Set([...Array.from(likedSet), ...Array.from(tasteSet), ...qualityIds])
    const available = midIds.filter(id => !usedSet.has(id))
    // Shuffle and take some
    for (let i = available.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [available[i], available[j]] = [available[j], available[i]]
    }
    exploreIds = available.slice(0, 40)
  }

  // Interleave into feed sequence
  const feedIds = interleave(tasteIds, qualityIds, exploreIds, FEED_SIZE)

  // Fetch full painting data
  if (!feedIds.length) {
    return res.json({ paintings: [], total: 0, liked: likedIds })
  }

  const pipe = redis.pipeline()
  for (const id of feedIds) {
    pipe.hmget(`stv:p:${id}`, ...PAINTING_FIELDS)
  }
  const results = await pipe.exec()

  const paintings = feedIds
    .map((id, i) => {
      const [err, values] = results![i]
      if (err || !values) return null
      const hash = hmgetToHash(values as (string | null)[])
      if (!Object.keys(hash).length) return null
      const p = parsePainting(hash, id)
      if (!p.images.length) return null
      return p
    })
    .filter((p): p is NonNullable<typeof p> => p !== null)

  res.setHeader('Cache-Control', 'no-cache')
  return res.json({ paintings, total: paintings.length, liked: likedIds })
}


async function getTasteIds(redis: any, likedIds: string[], likedSet: Set<string>): Promise<string[]> {
  // Fetch embeddings for liked paintings
  const embPipe = redis.pipeline()
  for (const id of likedIds) {
    (embPipe as any).hgetBuffer(`stv:p:${id}`, 'embedding')
  }
  const embResults = await embPipe.exec()

  const validEmbeddings: Buffer[] = []
  for (const [err, data] of embResults || []) {
    if (!err && data && Buffer.isBuffer(data)) {
      validEmbeddings.push(data)
    }
  }

  if (!validEmbeddings.length) return []

  // Average into taste vector
  const taste = new Float32Array(DIM)
  for (const buf of validEmbeddings) {
    const floats = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + DIM * 4))
    for (let i = 0; i < DIM; i++) taste[i] += floats[i]
  }
  let norm = 0
  for (let i = 0; i < DIM; i++) {
    taste[i] /= validEmbeddings.length
    norm += taste[i] * taste[i]
  }
  norm = Math.sqrt(norm)
  for (let i = 0; i < DIM; i++) taste[i] /= norm

  const tasteBuffer = Buffer.from(taste.buffer)

  try {
    const knnResults = await (redis as any).call(
      'FT.SEARCH', 'stv:vec_idx',
      '*=>[KNN 100 @embedding $BLOB AS dist]',
      'PARAMS', '2', 'BLOB', tasteBuffer,
      'RETURN', '1', 'dist',
      'SORTBY', 'dist', 'ASC',
      'LIMIT', '0', '100',
      'DIALECT', '2',
    )

    const ids: string[] = []
    for (let i = 1; i < knnResults.length; i += 2) {
      const docId = String(knnResults[i]).replace('stv:p:', '')
      if (!likedSet.has(docId)) ids.push(docId)
    }
    return ids
  } catch {
    return []
  }
}


function interleave(taste: string[], quality: string[], explore: string[], maxSize: number): string[] {
  const result: string[] = []
  let ti = 0, qi = 0, ei = 0
  const seen = new Set<string>()

  function add(id: string) {
    if (!seen.has(id)) {
      seen.add(id)
      result.push(id)
    }
  }

  // Pattern: T Q T T E (repeating)
  // If no taste results, pattern: Q Q Q E (repeating)
  const hasTaste = taste.length > 0

  while (result.length < maxSize) {
    const before = result.length

    if (hasTaste) {
      if (ti < taste.length) add(taste[ti++])
      if (qi < quality.length) add(quality[qi++])
      if (ti < taste.length) add(taste[ti++])
      if (ti < taste.length) add(taste[ti++])
      if (ei < explore.length) add(explore[ei++])
    } else {
      if (qi < quality.length) add(quality[qi++])
      if (qi < quality.length) add(quality[qi++])
      if (qi < quality.length) add(quality[qi++])
      if (ei < explore.length) add(explore[ei++])
    }

    // No more items to add
    if (result.length === before) break
  }

  return result.slice(0, maxSize)
}
