import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'
import { parsePainting, PAINTING_FIELDS, hmgetToHash, isSpamTitle } from './_lib/parse'

const DIM = 768

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const q = ((req.query.q as string) || '').toLowerCase().trim()
  if (!q) return res.status(400).json({ error: 'Missing q parameter' })

  const redis = getRedis()

  // Try exact match first
  let queryEmb: Buffer | null = await (redis as any).hgetBuffer('stv:search_vocab', q)

  // Fallback: split into words, average their embeddings
  if (!queryEmb) {
    const words = q.split(/\s+/)
    const wordEmbs: Buffer[] = []
    for (const w of words) {
      const emb: Buffer | null = await (redis as any).hgetBuffer('stv:search_vocab', w)
      if (emb && Buffer.isBuffer(emb)) wordEmbs.push(emb)
    }
    if (wordEmbs.length) {
      queryEmb = averageBuffers(wordEmbs)
    }
  }

  if (!queryEmb) {
    return res.json({ paintings: [], total: 0 })
  }

  // KNN search (same pattern as feed.ts)
  let knnResults: any[]
  try {
    knnResults = await (redis as any).call(
      'FT.SEARCH', 'stv:vec_idx',
      '*=>[KNN 60 @embedding $BLOB AS dist]',
      'PARAMS', '2', 'BLOB', queryEmb,
      'RETURN', '1', 'dist',
      'SORTBY', 'dist', 'ASC',
      'LIMIT', '0', '60',
      'DIALECT', '2',
    )
  } catch (e: any) {
    if (e.message?.includes('no such index') || e.message?.includes('Unknown Index')) {
      return res.json({ paintings: [], total: 0, message: 'Vector index not ready' })
    }
    throw e
  }

  // Parse FT.SEARCH response
  const candidateIds: string[] = []
  for (let i = 1; i < knnResults.length; i += 2) {
    const docId = String(knnResults[i]).replace('stv:p:', '')
    candidateIds.push(docId)
  }

  if (!candidateIds.length) {
    return res.json({ paintings: [], total: 0 })
  }

  // Fetch painting data
  const pipe = redis.pipeline()
  for (const id of candidateIds) {
    pipe.hmget(`stv:p:${id}`, ...PAINTING_FIELDS)
  }
  const results = await pipe.exec()

  const paintings = candidateIds
    .map((id, i) => {
      const [err, values] = results![i]
      if (err || !values) return null
      const hash = hmgetToHash(values as (string | null)[])
      if (!Object.keys(hash).length) return null
      const p = parsePainting(hash, id)
      if (!p.images.length) return null
      if (isSpamTitle(p.title)) return null
      return p
    })
    .filter((p): p is NonNullable<typeof p> => p !== null)

  res.setHeader('Cache-Control', 'no-cache')
  return res.json({ paintings, total: paintings.length, query: q })
}

function averageBuffers(buffers: Buffer[]): Buffer {
  const avg = new Float32Array(DIM)
  for (const buf of buffers) {
    const floats = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + DIM * 4))
    for (let i = 0; i < DIM; i++) avg[i] += floats[i]
  }
  let norm = 0
  for (let i = 0; i < DIM; i++) {
    avg[i] /= buffers.length
    norm += avg[i] * avg[i]
  }
  norm = Math.sqrt(norm)
  if (norm > 0) for (let i = 0; i < DIM; i++) avg[i] /= norm
  return Buffer.from(avg.buffer)
}
