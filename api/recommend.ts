import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'
import { parsePainting, PAINTING_FIELDS, hmgetToHash } from './_lib/parse'

const LIKES_KEY = 'stv:likes'
const DIM = 768

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const redis = getRedis()

  // Get liked painting IDs from Redis
  const likedIds = await redis.smembers(LIKES_KEY)
  if (!likedIds.length) {
    return res.json({ paintings: [], total: 0 })
  }

  // Fetch embeddings for liked paintings (binary)
  const embPipe = redis.pipeline()
  for (const id of likedIds) {
    (embPipe as any).hgetBuffer(`stv:p:${id}`, 'embedding')
  }
  const embResults = await embPipe.exec()

  // Collect valid embeddings
  const validEmbeddings: Buffer[] = []
  for (const [err, data] of embResults || []) {
    if (!err && data && Buffer.isBuffer(data)) {
      validEmbeddings.push(data)
    }
  }

  if (!validEmbeddings.length) {
    return res.json({
      paintings: [], total: 0,
      message: 'No embeddings available yet — run scorer with embeddings first',
    })
  }

  // Average embeddings into taste vector (then normalize)
  const taste = new Float32Array(DIM)
  for (const buf of validEmbeddings) {
    // Slice to guarantee 4-byte alignment for Float32Array
    const floats = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + DIM * 4))
    for (let i = 0; i < DIM; i++) {
      taste[i] += floats[i]
    }
  }
  let norm = 0
  for (let i = 0; i < DIM; i++) {
    taste[i] /= validEmbeddings.length
    norm += taste[i] * taste[i]
  }
  norm = Math.sqrt(norm)
  for (let i = 0; i < DIM; i++) {
    taste[i] /= norm
  }

  const tasteBuffer = Buffer.from(taste.buffer)

  // KNN search via RediSearch
  let knnResults: any[]
  try {
    knnResults = await (redis as any).call(
      'FT.SEARCH', 'stv:vec_idx',
      '*=>[KNN 200 @embedding $BLOB AS dist]',
      'PARAMS', '2', 'BLOB', tasteBuffer,
      'RETURN', '1', 'dist',
      'SORTBY', 'dist', 'ASC',
      'LIMIT', '0', '200',
      'DIALECT', '2',
    )
  } catch (e: any) {
    if (e.message?.includes('no such index') || e.message?.includes('Unknown Index')) {
      return res.json({
        paintings: [], total: 0,
        message: 'Vector index not ready — run scorer --embed-only',
      })
    }
    throw e
  }

  // Parse FT.SEARCH response: [total, docId, [field, val, ...], docId, [...], ...]
  const candidates: { id: string; dist: number }[] = []
  for (let i = 1; i < knnResults.length; i += 2) {
    const rawId = String(knnResults[i])
    const docId = rawId.replace('stv:p:', '')
    const fields = knnResults[i + 1]
    let dist = 2.0
    if (Array.isArray(fields)) {
      for (let j = 0; j < fields.length; j += 2) {
        if (String(fields[j]) === 'dist') dist = parseFloat(String(fields[j + 1]))
      }
    }
    candidates.push({ id: docId, dist })
  }

  // Filter out liked paintings
  const likedSet = new Set(likedIds)
  const filtered = candidates.filter(c => !likedSet.has(c.id))

  // Fetch full painting data for top results
  const fetchIds = filtered.slice(0, 100).map(c => c.id)
  const distMap = new Map(filtered.map(c => [c.id, c.dist]))

  const dataPipe = redis.pipeline()
  for (const id of fetchIds) {
    dataPipe.hmget(`stv:p:${id}`, ...PAINTING_FIELDS)
  }
  const dataResults = await dataPipe.exec()

  // Parse and compute blended ranking
  const paintings: any[] = []
  for (let i = 0; i < fetchIds.length; i++) {
    const [err, values] = dataResults![i]
    if (err || !values) continue

    const hash = hmgetToHash(values as (string | null)[])
    if (!Object.keys(hash).length) continue

    const painting = parsePainting(hash, fetchIds[i])
    if (!painting.images.length) continue

    const dist = distMap.get(fetchIds[i]) ?? 2.0
    const similarity = 1 - dist  // cosine: 0=orthogonal, 1=identical
    const artScore = painting.art_score || 0

    // Quality floor — don't recommend low-quality paintings
    if (artScore < 30) continue

    // Blended: taste similarity dominates, quality is a gate + boost
    const blendedScore = artScore * 0.3 + similarity * 70

    paintings.push({
      ...painting,
      taste_similarity: Math.round(similarity * 100) / 100,
      blended_score: Math.round(blendedScore * 10) / 10,
    })
  }

  // Sort by blended score descending
  paintings.sort((a: any, b: any) => b.blended_score - a.blended_score)

  res.setHeader('Cache-Control', 'no-cache')
  return res.json({ paintings, total: paintings.length })
}
