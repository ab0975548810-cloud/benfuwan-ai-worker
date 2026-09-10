本福丸自架 AI 去背 Worker（Runpod Serverless）
============================================

用途
- 這不是 remove.bg API。
- AI 模型是 ZhengPeng7/BiRefNet，模型頁標示 MIT License。
- Runpod 只提供 GPU Serverless 運算環境；你的網站把照片送到你自己的 Worker。
- 前端同一張照片去背結果會存在該裝置 IndexedDB，重複去背優先讀快取，減少 GPU 次數。

檔案
- handler.py
- requirements.txt
- Dockerfile
- .dockerignore

建議部署方式（iPad 也能做）
1. GitHub 新建一個 repo，例如：benfuwan-ai-worker
2. 把這個資料夾裡 4 個檔案放到 repo 根目錄
3. Runpod → Settings → Connections → Connect GitHub
4. Runpod → Serverless → New Endpoint → Import Git Repository
5. 選 benfuwan-ai-worker、branch main、Dockerfile Path 填 Dockerfile
6. Endpoint Type 選 Queue
7. GPU 先選便宜、VRAM 至少 16GB 的型號測試（例如 T4 16GB / L4 24GB / A5000 24GB，實際可選項依 Runpod 當下庫存）
8. Active Workers 設 0（沒人使用時不常駐燒錢）
9. Max Workers 測試期先設 1
10. Idle Timeout / FlashBoot 依 Runpod 畫面預設即可；之後再針對速度調整
11. Deploy Endpoint，等 Build = Completed
12. 記下 Endpoint ID
13. Runpod → Settings / API Keys 建一把 API Key，自己保存，不要貼 GitHub

接回 Zeabur
Zeabur fan-phone 服務新增：
RUNPOD_ENDPOINT_ID=你的 Endpoint ID
RUNPOD_API_KEY=你的 Runpod API Key
AI_MODEL_NAME=ZhengPeng7/BiRefNet
AI_REMOVE_BG_TIMEOUT=120

舊的 REMOVEBG_API_KEY 可以刪除；v2.1 不會再讀它。

重新部署 fan-phone 後測：
https://你的網站/api/health

應看到：
"ai_background_removal": true
"ai_provider": "self-hosted-runpod"
"ai_model": "ZhengPeng7/BiRefNet"

安全提醒
- RUNPOD_API_KEY 只能放 Zeabur 環境變數，不可放 index.html/GitHub。
- 圖片是從瀏覽器 → 你的 Zeabur 後端 → 你的 Runpod Worker；不再傳給 remove.bg。
