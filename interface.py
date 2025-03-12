"""Streamlit interface for Ollama-based reasoning chain."""

try:
    import streamlit as st
except ImportError:
    raise ImportError(
        "streamlit is required for the UI. "
        "Install it with `pip install streamlit>=1.0.0`"
    )

import asyncio
import nest_asyncio
from engine import ReasonChain, Step, ReasoningError
from models import model_registry
from src.evaluation.simple_evaluator import SimpleEvaluator
from src.utils.logging_utils import setup_logger, log_direct_interaction, configure_root_logger

DEFAULT_TIMEOUT = 30
MIN_STEPS = 3

# Apply nest_asyncio to allow nested event loops
try:
    nest_asyncio.apply()
except Exception:
    pass

# Configure logger
configure_root_logger()
logger = setup_logger("reasoning_chain")

def check_ollama_running():
    """Check if Ollama is running and provide guidance if not."""
    try:
        import httpx
        with httpx.Client(timeout=2.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                return True
    except Exception:
        return False
    return False

def _format_step(step: Step) -> None:
    """Format and display a reasoning step."""
    if step.is_final:
        st.success("Final Answer")

        st.session_state.final_answer = step.content

        # Log metrics if evaluation is enabled
        if st.session_state.final_answer and st.session_state.human_ref:
            evaluator = SimpleEvaluator()
            human_ref = st.session_state.human_ref.strip()
            st.session_state.evaluation_metrics = evaluator.evaluate_answer(
                human_ref,
                st.session_state.final_answer,
                st.session_state.context,
                st.session_state.query,
            )

        # Log reasoning steps and final answer
        logger.info(f"Final Answer: {st.session_state.final_answer}")
        logger.info(f"Reasoning Steps:\n{st.session_state.reasoning_text}")
        logger.info(f"Evaluation Metrics: {st.session_state.evaluation_metrics}")
    else:
        st.subheader(f"Step {step.number}: {step.title}")

    st.write(step.content)
    st.progress(step.confidence)
    st.caption(f"Confidence: {step.confidence:.2f} | Thinking time: {step.thinking_time:.2f}s")

    st.session_state.reasoning_text += f"\nStep {step.number}: {step.title}\n{step.content}\n"
        

# def extract_final_answer(steps: list) -> str:
#     """Extract the final answer from reasoning steps."""
#     if not steps:
#         return ""
    
#     # Try to find a step marked as final
#     for step in reversed(steps):
#         if step.is_final:
#             return step.content
    
#     # If no final step is marked, return the last step
#     return steps[-1].content if steps else ""

# def _extract_json_content(text: str) -> Optional[str]:
#     """Extract content from JSON objects in text if present."""
#     try:
#         # Look for JSON objects using a regex pattern that matches JSON structure
#         json_pattern = r'(\{[^{}]*(\{[^{}]*\})*[^{}]*\})'
#         json_objects = re.findall(json_pattern, text)
        
#         if json_objects:
#             for json_str, _ in json_objects:
#                 try:
#                     # Try to parse as JSON
#                     parsed_obj = json.loads(json_str)
                    
#                     # Check if this is a "Final Answer" object
#                     if isinstance(parsed_obj, dict) and parsed_obj.get("title") == "Final Answer":
#                         return parsed_obj.get("content", "")
#                 except json.JSONDecodeError:
#                     continue
#     except Exception:
#         pass
    
#     return None

# async def _stream_reasoning_async(chain: ReasonChain, query: str, context: str = "") -> list:
#     """Stream reasoning steps and update UI using async, return all steps."""
#     placeholder = st.empty()
#     steps = []
    
#     with st.spinner("Generating reasoning chain..."):
#         try:
#             async for step in chain.generate_with_metadata(query):
#                 steps.append(step)
#                 with placeholder.container():
#                     _format_step(step)
#             return steps
#         except Exception as e:
#             st.error(f"Error during reasoning: {str(e)}")
#             return []

# def _stream_reasoning(chain: ReasonChain, query: str, context: str = "") -> list:
#     """Stream reasoning steps with proper asyncio handling, return all steps."""
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
    
#     try:
#         return loop.run_until_complete(_stream_reasoning_async(chain, query))
#     except Exception as e:
#         st.error(f"Error in async execution: {str(e)}")
#         return []
#     finally:
#         loop.close()

async def _stream_reasoning(chain: ReasonChain, query: str) -> None:
    """Stream reasoning steps and update UI."""
    placeholder = st.empty()
    # st.session_state.reasoning_texts = ""
    with st.spinner("Generating reasoning chain..."):
        try:
            async for step in chain.generate_with_metadata(query):
                with placeholder.container():
                    _format_step(step)
        except Exception as e:
            st.error(f"Error during reasoning: {str(e)}")

def _initialize_session_state() -> None:
    """Initialize session state variables."""
    if 'chain' not in st.session_state:
        st.session_state.chain = None
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'show_reasoning' not in st.session_state:
        st.session_state.show_reasoning = True
    if 'reasoning_texts' not in st.session_state:
        st.session_state.reasoning_texts = ""
    if 'final_answer' not in st.session_state:
        st.session_state.final_answer = ""
    if 'evaluation_metrics' not in st.session_state:
        st.session_state.evaluation_metrics = {}
    if 'human_ref' not in st.session_state:
        st.session_state.human_ref = ""
    if 'context' not in st.session_state:
        st.session_state.context = ""
    if 'query' not in st.session_state:
        st.session_state.query = ""

def render_ui() -> None:
    """Render the main UI components."""
    st.title("Local LLM Reasoning Chain")
    st.write("Step-by-step reasoning using local Ollama models")

    # Check if Ollama is running
    if not check_ollama_running():
        st.error("⚠️ Ollama server is not running or not reachable. Please start Ollama first.")
        st.info("""
        **How to start Ollama:**
        1. Open a new command prompt/terminal
        2. Run the command: `ollama serve`
        3. Wait for Ollama to start, then refresh this page
        """)
        return

    with st.sidebar:
        st.header("Model Settings")
        
        # Display model info
        with st.expander("Available Models"):
            available_models = model_registry.get_available_models()
            for model_name, ollama_model in available_models.items():
                st.code(f"• {model_name} → {ollama_model}")
            
            st.markdown("""
            **How it works:**
            - Selecting a different model in the dropdown will automatically use that model for your next query
            - All models listed should already be loaded in Ollama
            - You don't need to restart anything when switching models
            """)
        
        model_template = st.selectbox(
            "Select Model",
            model_registry.list_models()
        )
        
        # Show which actual Ollama model will be used
        st.info(f"Using Ollama model: **{model_registry.get_model(model_template)}**")
        
        min_steps = st.slider("Minimum reasoning steps:", 1, 10, MIN_STEPS)
        timeout = st.slider("Timeout (seconds):", 5, 300, DEFAULT_TIMEOUT)
        
        # Add context input in the sidebar, similar to streamlit_app.py
        context = st.text_area("Provide context (optional)", height=200)
        st.session_state.context = context
        
        # Add human reference answer for evaluation
        human_ref = st.text_area(
            "Human Reference Answer for Evaluation", 
            height=100, 
            placeholder="Enter human answer to compare...",
            key="human_ref"
        )
        
        # Add toggle for showing reasoning steps
        st.session_state.show_reasoning = st.checkbox(
            "Show reasoning steps", 
            value=st.session_state.show_reasoning
        )

    query = st.text_area("Enter your question:", height=100)
    st.session_state.query = query
    
    if st.button("Generate Reasoning Chain"):
        if query:
            st.session_state.reasoning_text = ""
            st.session_state.final_answer = ""
            st.session_state.evaluation_metrics = {}
            st.session_state.query = ""
            st.session_state.context = ""
            with st.spinner(f"Initializing reasoning with {model_registry.get_model(model_template)}..."):
                # Include context in the query if provided
                full_query = query
                if context:
                    full_query = f"Context: {context}\n\nQuestion: {query}"
                
                # Initialize the reasoning chain
                print("model_template:", model_template)
                chain = ReasonChain(
                    model=model_template,
                    timeout=timeout,
                    min_steps=min_steps
                )
                
                try:
                    # Get all reasoning steps
                    asyncio.run(_stream_reasoning(chain, full_query))
                    log_direct_interaction(
                    logger,
                    question=query,
                    context=context,
                    response_data={
                        "llm-answer": st.session_state.final_answer,
                        "reasoning": st.session_state.reasoning_text,
                        "human_reference": st.session_state.get("human_ref", ""),
                        "metrics": st.session_state.evaluation_metrics
                        }
                    )
                    if 'query_history' in st.session_state:
                        st.session_state.query_history.append(query)
                    
                    # Extract the final answer
                    # final_answer = extract_final_answer(steps)

                    # print("Final Answer:", final_answer)
                    
                    # Try to get cleaner content if the answer is in JSON format
                    # final_answer_content = _extract_json_content(final_answer) or final_answer
                    
                    # Display the LLM's final answer explicitly
                    # st.subheader("LLM Answer")
                    # st.info(final_answer_content)  # Clearly mark this as the LLM answer
                    
                    # # Collect reasoning steps as text
                    # reasoning_text = "\n\n".join([
                    #     f"Step {step.number}: {step.title}\n{step.content}"
                    #     for step in steps
                    # ])
                    
                    # # If a human reference is provided, perform evaluation
                    # if human_ref and human_ref.strip():
                    #     evaluator = SimpleEvaluator()
                    #     scores = evaluator.evaluate_answer(
                    #         human_ref, 
                    #         final_answer_content, 
                    #         context, 
                    #         query
                    #     )
                        
                    #     st.subheader("Evaluation Results")
                    #     col1, col2, col3 = st.columns(3)
                    #     with col1:
                    #         st.metric("Final Score", f"{scores.get('final_score', 0):.3f}")
                    #     with col2:
                    #         st.metric("Similarity", f"{scores.get('similarity', 0):.3f}")
                    #     with col3:
                    #         st.metric("ROUGE-L", f"{scores.get('rougeL', 0)::.3f}")
                        
                    #     # Log the evaluation results
                    #     log_evaluation_data = {
                    #         "llm-answer": final_answer_content,  # This is the LLM answer used for evaluation
                    #         "reasoning": reasoning_text,
                    #         "human_reference": human_ref,
                    #         "metrics": scores
                    #     }
                    #     log_direct_interaction(logger, query, context, log_evaluation_data)
                    # else:
                    #     # Log without evaluation
                    #     log_data = {
                    #         "llm-answer": final_answer_content,  # This is the LLM answer being logged
                    #         "reasoning": reasoning_text,
                    #         "full_response": final_answer
                    #     }
                    #     log_direct_interaction(logger, query, context, log_data)
                    
                    # # Add to query history
                    # if 'query_history' in st.session_state:
                    #     st.session_state.query_history.append(query)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please enter a question first.")

def main() -> None:
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="LLM Reasoning Chain",
        page_icon="🤔",
        layout="wide"
    )
    _initialize_session_state()
    render_ui()

if __name__ == '__main__':
    main()
