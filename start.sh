#!/bin/bash
export CHROMA_DB_PATH=/tmp/chroma_db
mkdir -p $CHROMA_DB_PATH
streamlit run BACKEND/app/app.py --server.port=$PORT --server.address=0.0.0.0
