import os
import operator
import json
import re  # Added for Regex Backup
from typing import Annotated, TypedDict, Union, List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# --- SETUP ---
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    print("CRITICAL ERROR: GOOGLE_API_KEY not found. Check your .env file.")
    exit(1)

key = os.environ["GOOGLE_API_KEY"]
print(f"DEBUG: Loaded API Key starting with: {key[:5]}...")

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# --- 1. TOOLS ---
@tool
def mock_lead_capture(name: str, email: str, platform: str):
    """Captures lead details when a user is ready to sign up."""
    print(f"\n✅ [SYSTEM ACTION] Lead Captured: {name} | {email} | {platform}\n")
    return "success"

# --- 2. RAG KNOWLEDGE BASE ---
def setup_rag():
    DB_PATH = "faiss_index"
    if os.path.exists(DB_PATH):
        try:
            return FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True).as_retriever()
        except:
            pass 

    print("--- Creating Knowledge Base (One-time API call) ---")
    if not os.path.exists("knowledge_base.txt"):
        print("Error: knowledge_base.txt missing!")
        return None
        
    loader = TextLoader("knowledge_base.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    try:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(DB_PATH) 
        return db.as_retriever()
    except Exception as e:
        print(f"Error creating knowledge base: {e}")
        return None

retriever = setup_rag()

# --- 3. STATE DEFINITION ---
class AgentState(TypedDict):
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], operator.add]
    intent: str
    lead_info: dict 

# --- 4. NODE LOGIC ---

def router_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1].content
    current_info = state.get("lead_info", {})
    msg_lower = last_message.lower()

    # RULE 0: Keyword Overrides
    if any(x in msg_lower for x in ["sign up", "buy", "register", "subscribe"]):
        return {"intent": "HIGH_INTENT"}
    if any(x in msg_lower for x in ["my name is", "i am", "@", ".com", "youtube", "instagram"]):
        return {"intent": "HIGH_INTENT"}

    # RULE 1: Continuation
    missing = [k for k in ["name", "email", "platform"] if not current_info.get(k)]
    if current_info and missing:
        return {"intent": "HIGH_INTENT"}

    # RULE 2: LLM Classification
    try:
        prompt = ChatPromptTemplate.from_template(
            """Classify intent: GREETING, INQUIRY, or HIGH_INTENT.
            User Message: {message}
            Output ONE word.
            """
        )
        chain = prompt | llm
        response = chain.invoke({"message": last_message})
        intent = response.content.strip().upper()
    except:
        intent = "INQUIRY"
    
    if "GREETING" in intent: return {"intent": "GREETING"}
    if "HIGH" in intent or "SIGN" in intent: return {"intent": "HIGH_INTENT"}
    return {"intent": "INQUIRY"} 

def rag_node(state: AgentState):
    if not retriever:
        return {"messages": [AIMessage(content="Knowledge base error.")]}
        
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = ChatPromptTemplate.from_template(
        """Answer using ONLY this context: {context}
        Question: {question}
        """
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": query})
    return {"messages": [response]}

def lead_capture_node(state: AgentState):
    current_info = state.get("lead_info", {}) or {}
    last_message = state["messages"][-1].content
    msg_lower = last_message.lower()
    
    # 1. Try LLM Extraction first
    extract_prompt = ChatPromptTemplate.from_template(
        """Extract JSON with keys: name, email, platform. 
        User Input: {input}
        """
    )
    try:
        raw = (extract_prompt | llm).invoke({"input": last_message}).content
        clean = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        current_info.update({k: v for k, v in data.items() if v})
    except:
        pass

    # 2. ROBUST FALLBACKS (The Fix)
    # If Email is still missing, find it with Regex
    if not current_info.get("email"):
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", last_message)
        if email_match:
            current_info["email"] = email_match.group(0)

    # If Platform is missing, look for keywords
    if not current_info.get("platform"):
        for p in ["youtube", "instagram", "tiktok", "facebook", "linkedin"]:
            if p in msg_lower:
                current_info["platform"] = p.capitalize()
                break

    # If Name is missing but user said "My name is...", grab it
    if not current_info.get("name") and "name is" in msg_lower:
        try:
            name_part = last_message.lower().split("name is")[-1].strip()
            current_info["name"] = name_part.split()[0].capitalize()
        except:
            pass

    # 3. Check what's left
    missing = [k for k in ["name", "email", "platform"] if not current_info.get(k)]
    
    if not missing:
        mock_lead_capture.invoke(current_info)
        return {"lead_info": {}, "messages": [AIMessage(content=f"Registered {current_info['name']}!")]}
    
    return {"lead_info": current_info, "messages": [AIMessage(content=f"I just need your {missing[0]}.")]}

def greeting_node(state: AgentState):
    return {"messages": [AIMessage(content="Hello! Ask me about AutoStream plans.")]}

# --- 5. GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("greeting_agent", greeting_node)
workflow.add_node("rag_agent", rag_node)
workflow.add_node("lead_agent", lead_capture_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: "greeting_agent" if x["intent"]=="GREETING" else "lead_agent" if x["intent"]=="HIGH_INTENT" else "rag_agent")
workflow.add_edge("greeting_agent", END)
workflow.add_edge("rag_agent", END)
workflow.add_edge("lead_agent", END)
app = workflow.compile()

# --- 6. RUN ---
if __name__ == "__main__":
    print("\n--- AutoStream Agent ---")
    session_data = {} 
    chat_history = []
    while True:
        user_in = input("\nYou: ")
        if user_in.lower() in ["exit", "quit"]: break
        try:
            res = app.invoke({"messages": chat_history + [HumanMessage(content=user_in)], "lead_info": session_data, "intent": ""})
            bot_msg = res["messages"][-1].content
            print(f"Agent: {bot_msg}")
            chat_history.extend([HumanMessage(content=user_in), AIMessage(content=bot_msg)])
            if "lead_info" in res: session_data = res["lead_info"]
        except Exception as e:
            print(f"Error: {e}")