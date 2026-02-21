import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'

const LIKES_KEY = 'stv:likes'

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const redis = getRedis()

  if (req.method === 'GET') {
    // Return all liked painting IDs
    const liked = await redis.smembers(LIKES_KEY)
    return res.json({ liked })
  }

  if (req.method === 'POST') {
    const { id, liked } = req.body as { id?: string; liked?: boolean }
    if (!id || typeof id !== 'string') {
      return res.status(400).json({ error: 'id required' })
    }

    if (liked === false) {
      await redis.srem(LIKES_KEY, id)
    } else {
      await redis.sadd(LIKES_KEY, id)
    }

    const count = await redis.scard(LIKES_KEY)
    return res.json({ id, liked: liked !== false, total_likes: count })
  }

  return res.status(405).json({ error: 'Method not allowed' })
}
