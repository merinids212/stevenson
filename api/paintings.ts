import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'
import { parsePainting, PAINTING_FIELDS, hmgetToHash, isSpamTitle } from './_lib/parse'

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const redis = getRedis()
  const q = req.query as Record<string, string | string[]>

  // Fast path: fetch specific paintings by ID list
  const idsParam = typeof q.ids === 'string' ? q.ids : undefined
  if (idsParam) {
    const requestedIds = idsParam.split(',').filter(Boolean).slice(0, 100)
    if (!requestedIds.length) return res.json({ paintings: [], total: 0 })

    const pipe = redis.pipeline()
    for (const id of requestedIds) {
      pipe.hmget(`stv:p:${id}`, ...PAINTING_FIELDS)
    }
    const results = await pipe.exec()

    const paintings = (results || [])
      .map(([err, data], i) => {
        if (err || !data || !Array.isArray(data)) return null
        const hash = hmgetToHash(data as (string | null)[])
        if (!Object.keys(hash).length) return null
        return parsePainting(hash, requestedIds[i])
      })
      .filter((p): p is NonNullable<typeof p> => p !== null)
      .filter(p => p.images.length > 0)

    res.setHeader('Cache-Control', 'no-cache')
    return res.json({ paintings, total: paintings.length })
  }

  const region = typeof q.region === 'string' ? q.region : undefined
  const state = typeof q.state === 'string' ? q.state : undefined
  const source = typeof q.source === 'string' ? q.source : undefined
  const sort = typeof q.sort === 'string' ? q.sort : 'art_score'
  const order = typeof q.order === 'string' ? q.order : 'desc'
  const page = parseInt(typeof q.page === 'string' ? q.page : '1') || 1
  const limit = parseInt(typeof q.limit === 'string' ? q.limit : '0') || 0
  const minPrice = typeof q.min_price === 'string' ? parseFloat(q.min_price) : undefined
  const maxPrice = typeof q.max_price === 'string' ? parseFloat(q.max_price) : undefined
  const filter = typeof q.filter === 'string' ? q.filter : undefined
  const subject = typeof q.subject === 'string' ? q.subject : undefined
  const mood = typeof q.mood === 'string' ? q.mood : undefined
  const medium = typeof q.medium === 'string' ? q.medium : undefined
  const color = typeof q.color === 'string' ? q.color : undefined

  // Choose the right sorted index
  let indexKey: string
  if (subject) {
    indexKey = `stv:idx:subject:${subject}`
  } else if (mood) {
    indexKey = `stv:idx:mood:${mood}`
  } else if (medium) {
    indexKey = `stv:idx:medium:${medium}`
  } else if (color) {
    indexKey = `stv:idx:color:${color}`
  } else if (region) {
    indexKey = `stv:idx:region:${region}`
  } else if (state) {
    indexKey = `stv:idx:state:${state}`
  } else if (source) {
    indexKey = `stv:idx:source:${source}`
  } else if (sort === 'value_score' || filter === 'gems') {
    indexKey = 'stv:idx:value_score'
  } else if (sort === 'price') {
    indexKey = 'stv:idx:price'
  } else {
    indexKey = 'stv:idx:art_score'
  }

  // Get sorted IDs from the index (cap at 3000 for performance)
  const maxFetch = 3000
  const [ids, indexTotal] = await Promise.all([
    order === 'asc'
      ? redis.zrange(indexKey, 0, maxFetch - 1)
      : redis.zrevrange(indexKey, 0, maxFetch - 1),
    redis.zcard(indexKey),
  ])

  if (!ids.length) {
    res.setHeader('Cache-Control', 's-maxage=60, stale-while-revalidate=300')
    return res.json({ paintings: [], total: 0, totalIndex: 0 })
  }

  // Fetch painting fields via pipeline (excludes embedding blob)
  const pipe = redis.pipeline()
  for (const id of ids) {
    pipe.hmget(`stv:p:${id}`, ...PAINTING_FIELDS)
  }
  const results = await pipe.exec()

  // Parse and filter
  let paintings = (results || [])
    .map(([err, data], i) => {
      if (err || !data || !Array.isArray(data)) return null
      const hash = hmgetToHash(data as (string | null)[])
      if (!Object.keys(hash).length) return null
      return parsePainting(hash, ids[i])
    })
    .filter((p): p is NonNullable<typeof p> => p !== null)
    .filter(p => p.images.length > 0)
    .filter(p => !isSpamTitle(p.title))

  // Additional filters
  if (minPrice !== undefined) paintings = paintings.filter(p => p.price !== null && p.price >= minPrice)
  if (maxPrice !== undefined) paintings = paintings.filter(p => p.price !== null && p.price <= maxPrice)
  if (filter === 'best') paintings = paintings.filter(p => p.art_score !== null)
  if (filter === 'gems') paintings = paintings.filter(p => p.value_score !== null)

  const total = paintings.length

  // Weighted shuffle: skew toward high scores but mix in variety
  // Each painting gets a random key weighted by its rank position
  // Top paintings are very likely to stay near the top, but not locked in
  if (sort === 'art_score' && order === 'desc' && !region && !state && !source) {
    paintings = paintings.map((p, i) => ({
      painting: p,
      key: Math.random() * Math.pow(0.998, i),
    }))
    .sort((a, b) => b.key - a.key)
    .map(w => w.painting)
  }

  // Paginate if limit > 0
  if (limit > 0) {
    const start = (page - 1) * limit
    paintings = paintings.slice(start, start + limit)
  }

  res.setHeader('Cache-Control', 'no-cache')
  return res.json({ paintings, total, totalIndex: indexTotal })
}
