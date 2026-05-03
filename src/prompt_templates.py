"""
Prompt template builder.

This file defines the four prompt styles used in the experiment:
- concise
- structured
- verbose
- cot
"""

from typing import Literal

# Allowed prompt style types

PromptStyle = Literal["concise", "structured", "verbose", "cot"]


def build_prompt(task_type: str, prompt_style: PromptStyle, input_text: str) -> str:
    """
    Build a full prompt from:
    - task_type
    - prompt_style
    - input_text

    Parameters
    ----------
    task_type : str
        qa, reasoning, or summarization
    prompt_style : PromptStyle
        concise, structured, verbose, cot
    input_text : str
        The actual input example content

    Returns
    -------
    str
        Full final prompt to send to the model
    """
     # Format base input depending on task type
    
    if task_type in {"qa", "reasoning"}:
        base = f"Question/Input:\n{input_text}\n"
    elif task_type == "summarization":
        base = f"Document:\n{input_text}\n"
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

     # Define prompt instruction based on prompt style
    
    if prompt_style == "concise":
        
         # Minimal direct instruction
        
        instruction = "Respond as briefly as possible while staying correct."

    elif prompt_style == "structured":

        # Explicit formatted response guidance
        
        if task_type == "summarization":
            instruction = (
                "Summarize in this format:\n"
                "Summary: ...\n"
                "Key Point: ...\n"
            )
        else:
            instruction = (
                "Answer in this format:\n"
                "Final Answer: ...\n"
                "Key Evidence: ...\n"
            )

    elif prompt_style == "verbose":
        
        # Expanded context-heavy response style
        
        instruction = "Provide a detailed answer with explanation and context."

    elif prompt_style == "cot":

        # Step-by-step reasoning prompt
        
        instruction = "Reason step by step, then provide the final answer clearly."

    else:
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")

    # Combine instruction with task input
    
    return f"{instruction}\n\n{base}"
