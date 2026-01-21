# Shin Gateway - LLM Proxy Improvement Plan

This document outlines the roadmap to transform Shin Gateway into an enterprise-grade LLM proxy.

## 1. Recommended Tech Stack Additions

To achieve high performance, scalability, and observability, we recommend integrating the following technologies:

| Component | Recommendation | Purpose |
| :--- | :--- | :--- |
| **High-Perf Caching** | **Redis** | **Distributed State & Caching**. Moves RateLimiter/CircuitBreaker state out of memory to allow horizontal scaling. Enables sub-millisecond response caching. |
| **Vector DB** | **Qdrant** or **pgvector** | **Semantic Caching**. Instead of exact string matching, use embeddings to find "similar" queries (e.g., "Explain React" vs "What is React") and return cached responses. |
| **Analytics DB** | **ClickHouse** | **High-Volume Logging**. Replaces SQLite for request/response logging. Essential for analyzing millions of tokens/requests without performance degradation. |
| **Tracing** | **OpenTelemetry** | **Full Observability**. Provides end-to-end tracing across Python Core, Rust Admin, and upstream providers. |

## 2. Core Logic & Feature Roadmap

### A. Reliability & Smart Routing ("The Brain")
*   **Cascading Fallbacks**: Never fail a request. If the primary provider (e.g., OpenAI) fails/timeouts, automatically retry with a secondary provider (e.g., Azure or Anthropic).
*   **Latency-Based Routing**: Dynamically route traffic to the fastest provider in real-time.
*   **Load Balancing**: Distribute requests across multiple API keys/organizations to avoid hitting provider rate limits.

### B. Caching Strategy ("The Cost Saver")
*   **L1 Cache (Exact Match)**: Redis-based key-value store for identical prompts.
*   **L2 Cache (Semantic)**: Vector-based similarity search to catch rephrased queries.
*   **Tool Definition Caching**: Cache large tool schemas (commonly sent by agents like Claude Code) to reduce input token usage/latency.

### C. Security & Governance
*   **PII Redaction**: Middleware to detect and redact sensitive info (Emails, Credit Cards, API Keys) *before* sending to external providers.
*   **Audit Logging**: Encrypted, immutable logs of all interactions for compliance.
*   **Budgeting**: Set hard limits on spend per user/project.

### D. Agentic Optimizations
*   **Context Compression**: When history exceeds limits, use a small model (GPT-4o-mini) to summarize older messages.
*   **Virtual Context**: maintain stateful conversation history on the proxy side, allowing "stateless" clients to send shorter requests.

## 3. Implementation Steps

### Phase 1: Foundation (Current Focus)
1.  [ ] **Redis Integration**: Add Redis client for shared state.
2.  [ ] **Fallback Logic**: Implement retry mechanism with alternative providers.
3.  [ ] **PII Middleware**: Add regex-based scrubber.

### Phase 2: Intelligence
4.  [ ] **Semantic Caching**: Integrate Qdrant/pgvector.
5.  [ ] **Smart Routing**: Implement latency monitoring and dynamic routing.

### Phase 3: Scale
6.  [ ] **ClickHouse Migration**: Move analytics from SQLite to ClickHouse.
7.  [ ] **OpenTelemetry**: Add full tracing instrumentation.
