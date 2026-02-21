import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'

const LIKES_KEY = 'stv:likes'
const MAP_FIELDS = [
  'art_score', 'price', 'clip_styles', 'title',
  'region', 'source', 'images', 'url', 'artist',
  'value_score', 'umap_x', 'umap_y', 'umap_z',
] as const

/**
 * GET /api/taste-map
 *
 * Returns all paintings with UMAP 3D coordinates for the taste visualization.
 * Response: { ids, coords (flat xyz), meta, liked }
 * Cached for 5 minutes.
 */
export default async function handler(_req: VercelRequest, res: VercelResponse) {
  const redis = getRedis()

  // Get all embedded painting IDs (those with UMAP coords)
  const allIds = await redis.smembers('stv:embedded')
  if (!allIds.length) {
    return res.json({ ids: [], coords: [], meta: [], liked: [] })
  }

  // Get liked IDs
  const likedIds = await redis.smembers(LIKES_KEY)

  // Fetch UMAP coords + metadata in batches
  const BATCH = 500
  const ids: string[] = []
  const coords: number[] = []
  const meta: Array<Record<string, unknown>> = []

  for (let i = 0; i < allIds.length; i += BATCH) {
    const batch = allIds.slice(i, i + BATCH)
    const pipe = redis.pipeline()
    for (const id of batch) {
      pipe.hmget(`stv:p:${id}`, ...MAP_FIELDS)
    }
    const results = await pipe.exec()

    for (let j = 0; j < batch.length; j++) {
      const [err, values] = results![j]
      if (err || !values) continue
      const v = values as (string | null)[]

      const x = v[10] ? parseFloat(v[10]) : null
      const y = v[11] ? parseFloat(v[11]) : null
      const z = v[12] ? parseFloat(v[12]) : null
      if (x === null || y === null || z === null) continue

      ids.push(batch[j])
      coords.push(
        Math.round(x * 10000) / 10000,
        Math.round(y * 10000) / 10000,
        Math.round(z * 10000) / 10000,
      )

      // Parse primary style
      let primaryStyle = ''
      if (v[2]) {
        try {
          const styles = JSON.parse(v[2])
          if (styles[0]?.style) primaryStyle = styles[0].style
        } catch {}
      }

      // Parse first image
      let thumb = ''
      if (v[6]) {
        try {
          const imgs = JSON.parse(v[6])
          if (imgs[0]) thumb = imgs[0]
        } catch {}
      }

      meta.push({
        s: v[0] ? Math.round(parseFloat(v[0]) * 10) / 10 : 0,
        p: v[1] ? Math.round(parseFloat(v[1])) : 0,
        st: primaryStyle,
        t: (v[3] || '').slice(0, 60),
        r: v[4] || '',
        src: v[5] || '',
        img: thumb,
        u: v[7] || '',
        a: v[8] || '',
        v: v[9] ? Math.round(parseFloat(v[9]) * 10) / 10 : 0,
      })
    }
  }

  res.setHeader('Cache-Control', 's-maxage=300, stale-while-revalidate=600')
  return res.json({ ids, coords, meta, liked: likedIds })
}
