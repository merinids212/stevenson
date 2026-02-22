import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'
import { parsePainting, PAINTING_FIELDS, hmgetToHash, isSpamTitle } from './_lib/parse'

const DIM = 768

/** Generate stem variants for common English suffixes */
function stemVariants(word: string): string[] {
  const v = [word]
  if (word.endsWith('ies') && word.length > 4) v.push(word.slice(0, -3) + 'y')
  if (word.endsWith('es') && word.length > 3) v.push(word.slice(0, -2))
  if (word.endsWith('s') && !word.endsWith('ss') && !word.endsWith('us') && !word.endsWith('ies') && word.length > 2) v.push(word.slice(0, -1))
  return Array.from(new Set(v))
}

/** Look up query embedding from vocab, with stemming fallback */
async function resolveVocab(q: string, redis: any): Promise<Buffer | null> {
  // Try exact phrase
  let emb: Buffer | null = await redis.hgetBuffer('stv:search_vocab', q)
  if (emb) return emb

  // Split into words, try stems for each
  const words = q.split(/\s+/)
  const wordEmbs: Buffer[] = []
  for (const word of words) {
    const variants = stemVariants(word)
    for (const v of variants) {
      const e: Buffer | null = await redis.hgetBuffer('stv:search_vocab', v)
      if (e && Buffer.isBuffer(e)) { wordEmbs.push(e); break }
    }
  }
  return wordEmbs.length ? averageBuffers(wordEmbs) : null
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const redis = getRedis()
  let queryEmb: Buffer | null = null
  let queryText = ''

  if (req.method === 'POST') {
    // Browser CLIP path: receive raw 768-dim embedding
    const { embedding } = req.body || {}
    if (!embedding || !Array.isArray(embedding) || embedding.length !== DIM) {
      return res.status(400).json({ error: 'Expected 768-dim embedding array' })
    }
    queryEmb = Buffer.from(new Float32Array(embedding).buffer)
    queryText = req.body.text || 'semantic'
  } else if (req.method === 'GET') {
    // Vocab path with stemming
    queryText = ((req.query.q as string) || '').toLowerCase().trim()
    if (!queryText) return res.status(400).json({ error: 'Missing q parameter' })
    queryEmb = await resolveVocab(queryText, redis as any)
    if (!queryEmb) return res.json({ paintings: [], total: 0 })
  } else {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  // KNN search
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
    candidateIds.push(String(knnResults[i]).replace('stv:p:', ''))
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
  return res.json({ paintings, total: paintings.length, query: queryText })
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
