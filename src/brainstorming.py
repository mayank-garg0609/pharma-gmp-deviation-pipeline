
### * importing * ###
import os
import json
import requests
from src.files.helperfunc import import_data, load_active_prompts, processing_content,process_description
from src.files.agents import llm, summarizerAgent, instructionAnsweringAgent
from src.files.brainstorminghelper import summary_qa
from src.files.vectorstores import  MongoVectorStore
from src.files.deviation_store import DeviationSimilarityService
from src.files.redis_repo import DeviationRedisRepository, DeviationUpstashRedisRepository
from dotenv import load_dotenv
#! brainstorming function
load_dotenv()

base_url = os.getenv("CUSTOM_LLM_URL")
if not base_url:
    raise EnvironmentError("CUSTOM_LLM_URL environment variable is not set")


def brain(input_data: dict):
    summary=summary_qa(input_data['Problem Description and Immediate Action']) 
    # Use absolute path relative to this file's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(current_dir, "prompts", "Prompts Output 2 1.xlsx")
    prompts=load_active_prompts(prompts_path) 
    answer=process_description(summary,llm)
    answers=[answer]
    vector_store = MongoVectorStore()
    similarity_service = DeviationSimilarityService(vector_store)
    similar_results = similarity_service.find_similar(
                                                    answers=answers
                                                )           
    similar_ids = set()
    for result in similar_results:
        hit = result["matches"]
        for sid in hit:
            i=sid.get("metadata", {}).get("summary_id")
            print(i)
            similar_ids.add(i)
    similarfile = list(similar_ids)
    redis_repo =  DeviationUpstashRedisRepository()
    rootcause_content = "Previous similar root causes for brainstorming:\n"
    for deviation_id in similarfile:
        data = redis_repo.get_deviation(deviation_id)

        if not data:
            continue
        
        text = f"""Problem description: {data['problem_description']}\n"
                Root cause: {data['root_cause']}\n
                ########################\n"""

        
        
        rootcause_content += text        
    results = {}

    QUESTION_KEYS = {
        "Root Cause Brainstorming": "root_cause",
        "Recommended Corrective Action /Preventive Action": "capa",
        "Recommend Corrective Action Effectiveness Check": "capa_effectiveness",
        "Recommend Preventive Action Effectiveness Check": "pa_effectiveness"
    }

    # Prepare all questions for batch processing
    batch_questions = []
    question_keys_order = []
    
    for prompte in prompts:
        _, question, prompt = prompte
        question_key = QUESTION_KEYS.get(question, question)
        
        additional_content = ""
        if question_key == "root_cause":
            additional_content = rootcause_content

        query = f"""
        You are a gmp expert in pharma industry, your tasked with providing a complete and accurate answer to a user's question by strictly following their specific instructions. content should be in markdown format(compulsory) and strictly as writen

        ---

        **THE ACTUAL QUESTION YOU MUST ANSWER:**
        {question}

        **CONTEXT/KNOWLEDGE BASE TO USE:**
        {summary}

        **HOW TO ANSWER (USER'S INSTRUCTIONS):**
        {prompt}

        {additional_content}

        ---

        ## Question:
        {question}

        ## Answer:
        """
        
        batch_questions.append(query)
        question_keys_order.append(question_key)
    
    # Make batch API call

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/batch_chat",
            json={
                "questions": batch_questions,
                "max_token": 256
            },
            timeout=300  # 5 minutes for all questions
        )
        response.raise_for_status()
        batch_result = response.json()
        
        # Map responses back to question keys
        for i, question_key in enumerate(question_keys_order):
            results[question_key] = batch_result["responses"][i]
            print(f"Completed question: {question_key}")
            
    except requests.exceptions.RequestException as e:
        print(f"Batch API call failed: {e}")
        print("Falling back to sequential processing...")
        
        # Fallback to sequential processing if batch fails
        for prompte in prompts:
            _, question, prompt = prompte
            question_key = QUESTION_KEYS.get(question, question)
            
            additional_content = ""
            if question_key == "root_cause":
                additional_content = rootcause_content
            elif question_key == "capa":
                additional_content = "The root cause brainstorming is:\n" + results.get("root_cause", "")
            elif question_key in ("capa_effectiveness", "pa_effectiveness"):
                additional_content = "The recommended corrective and preventive actions are:\n" + results.get("capa", "")

            query = f"""
            You are a gmp expert in pharma industry, your tasked with providing a complete and accurate answer to a user's question by strictly following their specific instructions. content should be in markdown format(compulsory) and strictly as writen

            ---

            **THE ACTUAL QUESTION YOU MUST ANSWER:**
            {question}

            **CONTEXT/KNOWLEDGE BASE TO USE:**
            {summary}

            **HOW TO ANSWER (USER'S INSTRUCTIONS):**
            {prompt}

            {additional_content}

            ---

            ## Question:
            {question}

            ## Answer:
            """
            
            output = llm.call(query)
            print(f"Completed question: {question_key}")
            results[question_key] = output

    return results



def save_ans_to_json(ans, filename="brain_output.json"):
    """Save the brain function answers to a JSON file."""
    # Use absolute path relative to project root
    if not os.path.isabs(filename):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filename = os.path.join(project_root, filename)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ans, f, indent=2, ensure_ascii=False)
    print(f"Answers saved to {filename}")


if __name__ == "__main__":
    ans = brain("../reference document/second(complete set)/question-answer2.json")
    save_ans_to_json(ans)

