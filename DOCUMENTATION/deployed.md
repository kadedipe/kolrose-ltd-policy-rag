markdown
# 🚀 Deployed Application

The Kolrose Limited AI Policy Assistant is deployed and accessible at the following URL:

## 🔗 Live Application Link
**https://kolrose-policy-rag-production-d0bc.up.railway.app/**

---

## 📱 Accessing the Application

### Web Interface
Open the link above in any modern browser to access the chat interface.

### Railway Dashboard
Monitor your deployment at [railway.app](https://railway.app)

---

## 🔄 Deployment Status

| Platform | Status | URL |
|----------|--------|-----|
| **Railway** | ✅ **Primary** | [https://kolrose-policy-rag-production-d0bc.up.railway.app/](https://kolrose-policy-rag-production-d0bc.up.railway.app/) |
| Streamlit Cloud | 🔄 Backup | kolrose-policy-rag.streamlit.app |
| Render | 🔄 Backup | kolrose-policy-rag.onrender.com |

---

## 🏢 About Kolrose Limited

📍 Suite 10, Bataiya Plaza, Area 2 Garki, Opposite FCDA, Abuja, FCT, Nigeria

This RAG system answers employee questions about **12 company policy documents** using:

| Component | Technology | Cost |
|-----------|-----------|------|
| **LLM** | OpenRouter (Google Gemini Flash) | **Free** |
| **Embeddings** | all-MiniLM-L6-v2 (Local) | **Free** |
| **Vector DB** | ChromaDB (Local) | **Free** |
| **Framework** | Streamlit | **Free** |
| **Hosting** | Railway | **Free Tier** |
| **Total Monthly Cost** | | **$0.00** |

---

## 📋 Policy Documents Covered

- KOL-HR-001: Employee Handbook
- KOL-HR-002: Leave and Time-Off Policy
- KOL-HR-003: Code of Conduct and Ethics
- KOL-HR-005: Remote Work Policy
- KOL-IT-001: IT Security Policy
- KOL-FIN-001: Expenses and Reimbursement
- KOL-HR-006: Performance Management
- KOL-HR-007: Training and Development
- KOL-ADMIN-001: Business Travel Policy
- KOL-FIN-002: Procurement Policy
- KOL-ADMIN-002: Health and Safety Policy
- KOL-HR-008: Grievance and Dispute Resolution

---

## 🔑 Environment Variables (Railway)

| Variable | Value |
|----------|-------|
| `OPENROUTER_API_KEY` | `sk-or-v1-your-key-here` |

---

## 🚀 How to Redeploy

1. **Push changes to GitHub:**
   ```bash
   git add .
   git commit -m "Update app"
   git push origin main
Railway auto-deploys from the main branch

Manual deploy (if needed):

bash
railway up
📧 Contact
For HR inquiries: hr@kolroselimited.com.ng

For technical support: IT Support Desk

Last deployed: May 7, 2026

text

---

## ✅ Key Updates Made

| Before | After |
|--------|-------|
| Streamlit Cloud as Primary | **Railway as Primary** |
| Placeholder URL | **Actual Railway URL** |
| Multiple deployment links | Clear Primary/Backup status |
| Basic info | Added policy list, env vars, redeploy instructions |

Save this as `deployed.md` or update your `README.md` with this information!