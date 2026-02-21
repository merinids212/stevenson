import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const id = typeof req.query.id === 'string' ? req.query.id : ''
  if (!id) {
    return res.redirect(302, '/')
  }

  const redis = getRedis()

  // Fetch just the fields we need for OG tags
  const [title, price, images] = await redis.hmget(`stv:p:${id}`, 'title', 'price', 'images')

  if (!title) {
    return res.redirect(302, '/')
  }

  // Parse image
  let imageUrl = '/og.jpg'
  try {
    const parsed = JSON.parse(images || '[]')
    if (Array.isArray(parsed) && parsed.length > 0) {
      imageUrl = parsed[0]
    }
  } catch {}

  // Clean title
  const cleanTitle = (title || 'Untitled')
    .replace(/free shipping!?/gi, '')
    .replace(/[!]+/g, '')
    .trim()

  const ogTitle = `STEVENSON — ${cleanTitle}`
  const priceStr = price && price !== 'null' ? `$${Math.round(parseFloat(price))}` : ''
  const ogDescription = priceStr
    ? `${cleanTitle} — ${priceStr}`
    : cleanTitle

  // Escape for HTML
  const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

  const html = `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>${esc(ogTitle)}</title>
<meta property="og:title" content="${esc(ogTitle)}">
<meta property="og:description" content="${esc(ogDescription)}">
<meta property="og:image" content="${esc(imageUrl)}">
<meta property="og:type" content="website">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="${esc(ogTitle)}">
<meta name="twitter:description" content="${esc(ogDescription)}">
<meta name="twitter:image" content="${esc(imageUrl)}">
<meta http-equiv="refresh" content="0;url=/#${esc(id)}">
</head>
<body>
<script>window.location.replace('/#${id.replace(/'/g, "\\'")}');</script>
</body>
</html>`

  res.setHeader('Content-Type', 'text/html; charset=utf-8')
  res.setHeader('Cache-Control', 's-maxage=3600, stale-while-revalidate=86400')
  return res.send(html)
}
