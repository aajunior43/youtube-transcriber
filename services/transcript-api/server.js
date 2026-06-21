const http = require('http');
const url = require('url');

const PORT = 8771;

async function fetchYouTubePage(videoId) {
  const res = await fetch(`https://www.youtube.com/watch?v=${videoId}`, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8'
    }
  });
  return res.text();
}

function extractCaptionTracks(html) {
  const match = html.match(/ytInitialPlayerResponse\s*=\s*({.+?});/);
  if (!match) return null;
  let data;
  try {
    data = JSON.parse(match[1]);
  } catch (e) {
    return null;
  }
  const tracks = data?.captions?.playerCaptionsTracklistRenderer?.captionTracks;
  if (!tracks || tracks.length === 0) return null;
  return tracks.map(t => ({
    languageCode: t.languageCode,
    language: t.name?.simpleText || t.languageCode,
    baseUrl: t.baseUrl.replace(/\\u0026/g, '&')
  }));
}

async function fetchTranscriptXml(url) {
  const res = await fetch(url);
  return res.text();
}

function parseTranscriptXml(xml) {
  const segments = [];
  const textMatches = xml.matchAll(/<text start="([\d.]+)"(?: dur="([\d.]+)")?[^>]*>([\s\S]*?)<\/text>/g);
  for (const m of textMatches) {
    segments.push({
      start: parseFloat(m[1]) || 0,
      dur: m[2] ? parseFloat(m[2]) : 0,
      text: m[3].replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&#39;/g, "'").replace(/&quot;/g, '"')
    });
  }
  return segments;
}

async function getOembed(videoId) {
  try {
    const res = await fetch(`https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`);
    if (res.ok) return res.json();
  } catch (e) {}
  return null;
}

const server = http.createServer(async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }
  
  const parsed = url.parse(req.url, true);
  const path = parsed.pathname;
  
  if (path === '/transcript' || path === '/api/transcript') {
    const videoId = parsed.query.videoId || parsed.query.v;
    if (!videoId || !/^[\w-]{11}$/.test(videoId)) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Invalid video ID' }));
      return;
    }
    
    try {
      const [html, oembed] = await Promise.all([
        fetchYouTubePage(videoId),
        getOembed(videoId)
      ]);
      
      const tracks = extractCaptionTracks(html);
      if (!tracks) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'No captions available', oembed }));
        return;
      }
      
      // Fetch transcript from first track
      const lang = parsed.query.lang || tracks[0].languageCode;
      const track = tracks.find(t => t.languageCode === lang) || tracks[0];
      const xml = await fetchTranscriptXml(track.baseUrl);
      const segments = parseTranscriptXml(xml);
      
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        videoId,
        segments,
        tracks,
        currentLang: track.languageCode,
        oembed
      }));
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Internal error', detail: e.message }));
    }
    return;
  }
  
  if (path === '/health') {
    res.writeHead(200);
    res.end('ok');
    return;
  }
  
  res.writeHead(404);
  res.end('Not found');
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`Transcript API running on port ${PORT}`);
});
