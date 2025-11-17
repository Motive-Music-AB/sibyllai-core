# Beta Testing Roadmap

This document outlines deployment options for beta testing SibyllAI.

## Overview

SibyllAI consists of two components:
- **Frontend**: React/TypeScript app (Vite)
- **Backend**: FastAPI Python server with ML models (YAMNet, CLAP, Music2Emo)

The backend performs CPU-intensive music analysis that can take several minutes for long files.

## Deployment Options

### Option 1: Local Network Testing (Simplest)

**Best for**: Quick demos with people in the same location

**Steps**:
1. Find your computer's local IP address:
   ```bash
   ipconfig getifaddr en0  # macOS
   ipconfig               # Windows
   ```
2. Start both servers (already running on ports 8001 and 5173)
3. Share URL with testers: `http://YOUR_IP:5173`

**Pros**:
- Immediate, no setup
- Free

**Cons**:
- Only works on same WiFi network
- Requires your computer running continuously
- No internet access

### Option 2: Render.com (Free Tier)

**Best for**: Initial beta testing on a budget

**Backend Deployment**:
1. Go to https://render.com
2. Sign up and connect GitHub account
3. Click "New Web Service"
4. Select `sibyllai-core` repository
5. Configure:
   - **Name**: `sibyllai-backend`
   - **Root Directory**: Leave blank
   - **Build Command**: `pip install -e .`
   - **Start Command**: `cd sibyllai-web/backend && uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free
6. Click "Create Web Service"
7. Copy the deployed URL (e.g., `https://sibyllai-backend.onrender.com`)

**Frontend Deployment**:
1. Update `sibyllai-web/frontend/vite.config.ts`:
   ```typescript
   proxy: {
     '/api': {
       target: 'https://sibyllai-backend.onrender.com',  // Your backend URL
       changeOrigin: true,
     }
   }
   ```
2. Commit and push changes
3. Render.com → "New Static Site"
4. Select repository
5. Configure:
   - **Name**: `sibyllai-frontend`
   - **Root Directory**: `sibyllai-web/frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `dist`
6. Click "Create Static Site"
7. Share the frontend URL with beta testers

**Pros**:
- Completely free
- Public URL accessible anywhere
- Auto-deploys from GitHub

**Cons**:
- Free tier sleeps after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds to wake up
- Limited to 512 MB RAM on free tier

### Option 3: Railway.app (Paid but Better Experience)

**Best for**: Production-quality beta testing without wake-up delays

**Deployment**:
1. Go to https://railway.app
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `sibyllai-core`
5. Railway auto-detects and creates services:
   - Backend service (Python/FastAPI)
   - Frontend service (Node/Vite)
6. Configure environment variables if needed
7. Railway generates public URLs for both
8. Update frontend to point to backend URL
9. Share frontend URL with testers

**Pricing**:
- $5 free credits per month
- After free credits: ~$10-15/month for this app
- No sleep delays
- Charged by actual usage (RAM/CPU minutes)

**Pros**:
- No sleep delays (instant response)
- Better developer experience
- Scales automatically
- Fast deployments

**Cons**:
- Costs money after free $5 credits
- More expensive than Render for continuous use

### Option 4: Hybrid (Vercel + Railway/Render)

**Best for**: Optimizing cost while keeping frontend fast

**Setup**:
- **Frontend**: Deploy to Vercel (free, global CDN)
- **Backend**: Deploy to Railway (~$10/month) or Render (free with sleep)

**Benefits**:
- Vercel's free tier is excellent for React frontends
- CDN provides instant global access to UI
- Backend can handle long-running ML processing

**Note**: Cannot deploy backend to Vercel because serverless function timeout (60s max on Pro) is too short for music analysis.

## Recommendation

**For initial beta testing**:
- Start with **Render.com (Option 2)** - completely free
- Warn testers about potential 30-60s initial load time after inactivity
- If testers complain about wake-up delays, upgrade to Railway

**For serious beta testing**:
- Use **Railway.app (Option 3)** for better user experience
- Budget ~$10-15/month
- No delays, professional experience

## Next Steps

1. Choose deployment option
2. Deploy backend first, get URL
3. Update frontend config with backend URL
4. Deploy frontend
5. Test end-to-end with a sample file
6. Share URL with beta testers
7. Collect feedback

## Technical Considerations

**Backend Requirements**:
- Python 3.11+
- TensorFlow, PyTorch, YAMNet models
- Minimum 1-2 GB RAM recommended
- Long-running processes (can take minutes per file)

**Frontend Requirements**:
- Node.js build
- Static file serving
- Proxy to backend API

**CORS Setup**:
Both deployment platforms handle CORS automatically when using their proxy configurations.

## Support

For deployment issues, refer to:
- Render docs: https://render.com/docs
- Railway docs: https://docs.railway.app
- Vercel docs: https://vercel.com/docs

For SibyllAI-specific issues, see main README.md
