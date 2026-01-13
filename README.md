# AutoStream AI Agent

## Project Overview
This project implements a conversational AI agent for **AutoStream**, a fictional SaaS company. The agent is designed to handle customer inquiries, retrieve product information using RAG (Retrieval-Augmented Generation), and qualify high-intent leads by collecting their details (Name, Email, Platform).

## Architecture & Design Choices
### Framework: LangGraph
I chose **LangGraph** over AutoGen because this assignment requires a deterministic state machine. We need strict control over the conversation flow (e.g., once the user says "sign up," the agent *must* stay in the lead capture loop until it gets all details). LangGraph's `StateGraph` allows us to define these rigid loops and conditional edges (Router -> RAG vs. Router -> Lead Capture) much more clearly than a multi-agent debate system.

### State Management
The state is managed using a strictly typed dictionary (`AgentState`) that persists across conversation turns.
- **`lead_info`**: A dictionary that acts as a "slot-filling" buffer. It survives across turns, allowing the user to provide their name in one message and their email in the next without the agent forgetting previous details.
- **`intent`**: A flag determined by the Router node to decide which "specialist" node (RAG or Lead Capture) should handle the next step.

## WhatsApp Integration Plan
To deploy this on WhatsApp, I would use the **Meta Cloud API** with a Python backend (FastAPI/Flask):
1. **Webhook:** Set up a FastAPI endpoint to receive POST requests from WhatsApp.
2. **Session ID:** Use the user's phone number (`wa_id`) as the unique session key.
3. **Persistence:** Store the `AgentState` in a Redis database, keyed by the phone number. When a message arrives, load the state, run the LangGraph workflow, and save the updated state back to Redis.
4. **Response:** Send the agent's text response back via the Meta API.
## 🎥 Demo Video
Here is a screen recording of the agent in action, demonstrating RAG, Intent Detection, and Lead Capture:

[▶️ Watch the Demo Video](https://drive.google.com/file/d/1Kvh4cWZRFAGgpKE1axms7mLRSSzTmqAN/view?usp=sharing)
## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd autostream-agent
