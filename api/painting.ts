import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'
import { parsePainting, PAINTING_FIELDS, hmgetToHash } from './_lib/parse'

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const id = typeof req.query.id === 'string' ? req.query.id : ''
  if (!id) {
    return res.status(400).json({ error: 'Missing id' })
  }

  const redis = getRedis()
  const values = await redis.hmget(`stv:p:${id}`, ...PAINTING_FIELDS)
  const hash = hmgetToHash(values as (string | null)[])

  if (!Object.keys(hash).length) {
    return res.status(404).json({ error: 'Not found' })
  }

  const painting = parsePainting(hash, id)

  res.setHeader('Cache-Control', 's-maxage=3600, stale-while-revalidate=86400')
  return res.json({ painting })
}
