import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'
import { parsePainting } from './_lib/parse'

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const redis = getRedis()
  const q = req.query as Record<string, string | string[]>

  const region = typeof q.region === 'string' ? q.region : undefined
  const source = typeof q.source === 'string' ? q.source : undefined
  const sort = typeof q.sort === 'string' ? q.sort : 'art_score'
  const order = typeof q.order === 'string' ? q.order : 'desc'
  const page = parseInt(typeof q.page === 'string' ? q.page : '1') || 1
  const limit = parseInt(typeof q.limit === 'string' ? q.limit : '0') || 0
  const minPrice = typeof q.min_price === 'string' ? parseFloat(q.min_price) : undefined
  const maxPrice = typeof q.max_price === 'string' ? parseFloat(q.max_price) : undefined
  const filter = typeof q.filter === 'string' ? q.filter : undefined

  // Choose the right sorted index
  let indexKey: string
  if (region) {
    indexKey = `stv:idx:region:${region}`
  } else if (source) {
    indexKey = `stv:idx:source:${source}`
  } else if (sort === 'value_score' || filter === 'gems') {
    indexKey = 'stv:idx:value_score'
  } else if (sort === 'price') {
    indexKey = 'stv:idx:price'
  } else {
    indexKey = 'stv:idx:art_score'
  }

  // Get sorted IDs from the index
  const ids = order === 'asc'
    ? await redis.zrange(indexKey, 0, -1)
    : await redis.zrevrange(indexKey, 0, -1)

  if (!ids.length) {
    res.setHeader('Cache-Control', 's-maxage=60, stale-while-revalidate=300')
    return res.json({ paintings: [], total: 0 })
  }

  // Fetch all painting hashes via pipeline
  const pipe = redis.pipeline()
  for (const id of ids) {
    pipe.hgetall(`stv:p:${id}`)
  }
  const results = await pipe.exec()

  // Parse and filter
  let paintings = (results || [])
    .map(([err, data], i) => {
      if (err || !data || typeof data !== 'object' || !Object.keys(data as object).length) return null
      return parsePainting(data as Record<string, string>, ids[i])
    })
    .filter((p): p is NonNullable<typeof p> => p !== null)
    .filter(p => p.images.length > 0)

  // Additional filters
  if (minPrice !== undefined) paintings = paintings.filter(p => p.price !== null && p.price >= minPrice)
  if (maxPrice !== undefined) paintings = paintings.filter(p => p.price !== null && p.price <= maxPrice)
  if (filter === 'best') paintings = paintings.filter(p => p.art_score !== null)
  if (filter === 'gems') paintings = paintings.filter(p => p.value_score !== null)

  const total = paintings.length

  // Paginate if limit > 0
  if (limit > 0) {
    const start = (page - 1) * limit
    paintings = paintings.slice(start, start + limit)
  }

  res.setHeader('Cache-Control', 's-maxage=60, stale-while-revalidate=300')
  return res.json({ paintings, total })
}
