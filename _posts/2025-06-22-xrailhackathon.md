---
layout: post
title: X-RAIL: Our Multi-Agent Insurance AI Platform Built with Google Cloud & ADK
description: "Built for the adkhackathon - X-RAIL is an explainable, multi-agent AI system for insurance risk analysis and simulation, powered by Google Cloud and ADK."
date: 2025-06-23 21:00:00 +0530
image: '/img/posts/xrail/XRAIL22.png'
tags: [Google Cloud, ADK, Insurance AI, Explainable AI, Multi-Agent Systems, Vertex AI, SHAP, BigQuery]
---

# 🚂 Building X-RAIL: A Multi-Agent AI Insurance Platform Powered by Google Cloud & ADK

> This project was built as an official submission to the **Google ADK Hackathon 2025**.  
> We created **X-RAIL** — an explainable, multi-agent AI system for insurance analytics — to demonstrate the real-world potential of orchestrated LLM agents in high-stakes, regulated environments.
>
> #adkhackathon

## 🧠 The Problem We Set Out to Solve

Insurance underwriting often runs on opaque black-box models. In a $7 trillion global industry, that's a transparency problem regulators, customers, and underwriters can't afford.

Our goal? Build a fully transparent, explainable, and automated AI platform for insurance risk analytics — one that orchestrates tasks just like a real-world underwriting team.

## ✨ Introducing X-RAIL

**X-RAIL** (Xplainable Risk Assessment & Insights Loop) is a multi-agent AI system designed for structured + unstructured insurance claims.

It computes risk scores, explains them with SHAP, simulates "what-if" scenarios, generates reports, and trains models — all in real-time, all auditable.

## 🧱 Architecture Overview

X-RAIL is built with Google's Agent Development Kit (ADK), powered by Gemini Flash for intent recognition, and backed by Google Cloud's Vertex AI, BigQuery, and Cloud Storage.

Here's how all the agents connect in X-RAIL:

![X-RAIL Architecture Diagram](/img/posts/xrail/ARC_XRAIL1.png)

_Above: Each agent has a dedicated role — from risk scoring to simulation to analytics — coordinated by a central Conductor Agent via ADK._

## 🤖 Agents at Work

| Agent                   | Role                                                             |
| ----------------------- | ---------------------------------------------------------------- |
| 🧠 Conductor Agent      | Manages workflows and context via ADK + Gemini                   |
| 🔢 Risk Agent           | Predicts risk using calibrated XGBoost hosted on Vertex AI       |
| 🔍 Explainability Agent | Generates SHAP-based explanations with visual + narrative output |
| 📊 Dashboard Agent      | Builds live Streamlit dashboards with Plotly                     |
| 🔁 Impact Simulator     | Supports what-if changes, multi-scenario comparison              |
| 📄 PDF Report Agent     | Summarizes SHAP output into branded, compliant reports           |
| 📈 BigQuery Agent       | Converts NL to SQL with ChaseSQL                                 |
| 🧮 BQML Agent           | Trains ML models in BigQuery                                     |
| 📊 Analytics Agent      | Runs Python-based analysis via Vertex AI Code Executor           |

## 📊 Key Capabilities

- ✅ Real-time calibrated risk scoring
- ✅ Transparent SHAP explanations
- ✅ Live scenario simulation
- ✅ NL-powered SQL and analytics
- ✅ Streamlit dashboards + PDF reports
- ✅ Complete audit logging and traceability

## 🔧 Technologies Used

- **Google Cloud**: BigQuery, Vertex AI, Cloud Storage
- **ML**: XGBoost, SHAP, BigQuery ML, Scikit-learn
- **LLMs**: Gemini Flash, Vertex AI RAG
- **UI & Reporting**: Streamlit, Plotly, FPDF
- **Backend**: Python, Pandas, NumPy, Pydantic, Poetry
- **DevOps**: Docker, Cloud Run, IPython, Pytest

## 🛠️ Challenges We Tackled

- 🔄 Multi-agent orchestration with shared state
- 🔍 Translating SHAP into readable explanations
- 📦 Deploying dependency-heavy agents via Cloud Run
- 🔐 Managing IAM and service integrations across Cloud
- 🧪 Generating realistic synthetic insurance datasets

## 🏆 Outcomes & Impact

- ⏱️ Reduced manual review time by 60–70%
- 🔒 Full audit trail for regulators and compliance teams
- 📊 Instant dashboards for underwriters
- 🧠 Transparent explanations for non-technical users
- 📈 Scalable model training and analytics

## 🔮 What's Next

- 🚗 Telematics-based risk scoring
- 🧾 Regulatory mapping agent
- 🛡️ Fraud detection and anomaly detection
- 📱 Mobile-first risk dashboards for field agents
- 🔁 Counterfactual reasoning: "What could've improved this score?"

## 💬 Final Thoughts

X-RAIL is more than a model — it's a **thinking system**. By combining transparency, intelligence, and automation, it redefines how insurance underwriting and analytics can operate at scale.

Stay tuned — we're just getting started.
