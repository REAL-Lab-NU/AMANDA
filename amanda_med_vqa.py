import argparse
from PIL import Image
from tqdm import tqdm
from MLLM.models import load_model_and_preprocess
from MLLM.conversation.conversation import conv_templates
import math
import random
from amanda_prompts import *
from Retriever.utility import *
import yaml
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


client = OpenAI(api_key="OPENAI_API_KEY")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_gpt(messages, temperature=0.0, answer_type="open"):
    try:
        response = client.chat.completions.create(
            model=args.engine,
            messages=messages,
            temperature=temperature,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API错误: {e}")
        time.sleep(5)
        raise


def load_chat_template(template_path):
    with open(template_path, 'r') as f:
        return f.read().replace('    ', '').replace('\n', '')


def generate_initial_answer(model, image_tensor, question, answer_type):
    answer = \
        model.generate({"image": image_tensor, "prompt": [question]}, max_length=100 if answer_type == "open" else 50)[
            0].strip().replace("<s>", "").replace("</s>", "")
    return answer


def generate_reasoned_answer(caption, question, initial_answer, answer_type, examples, dataset_name):
    if answer_type == "open":
        system_prompt = OPEN_ENDED_GENERATOR_SYSTEM_PROMPT
        user_prompt = OPEN_ENDED_GENERATOR_PROMPT
    else:
        system_prompt = get_closed_ended_system_prompt(dataset_name, is_initial=True)
        user_prompt = CLOSED_ENDED_GENERATOR_PROMPT

    examples_str = ""
    if examples:
        examples_str = "Here are some similar examples:\n"
        for i, example in enumerate(examples, 1):
            examples_str += f"Example {i}:\nImage description: {example['caption']}\nQuestion: {example['question']}\nAnswer: {example['answer']}\n\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(
            caption=caption,
            question=question,
            initial_answer=initial_answer,
            examples=examples_str
        )}
    ]
    response = call_gpt(messages, answer_type=answer_type)
    analysis, answer = parse_reasoned_answer(response)
    return analysis, answer


def parse_reasoned_answer(response):
    analysis = re.search(r'Analysis:(.*?)Answer:', response, re.DOTALL)
    answer = re.search(r'Answer:(.*)', response, re.DOTALL)
    return (analysis.group(1).strip() if analysis else "",
            answer.group(1).strip() if answer else "")


def evaluate_answer(caption, question, answer, history):
    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": EVALUATOR_PROMPT.format(
            caption=caption,
            question=question,
            answer=answer,
            history=history
        )}
    ]
    response = call_gpt(messages)
    return parse_evaluation(response)


def parse_evaluation(response):
    confidence = re.search(r'Score: (\d+)', response)
    explanation = re.search(r'Explanation: (.*)', response, re.DOTALL)
    return int(confidence.group(1)) if confidence else 0, explanation.group(1).strip() if explanation else ""


def parse_follow_up_question(response):
    questions = re.findall(r'Sub-question\s*\d*:?\s*(.*?)(?=\nSub-question|\Z)', response, re.DOTALL | re.IGNORECASE)
    return [q.strip() for q in questions if q.strip()]


def generate_follow_up_question(question, history, max_sub_questions):
    system_prompt = EXPLORER_SYSTEM_PROMPT.format(max_sub_questions=max_sub_questions)
    prompt = EXPLORER_PROMPT
    formatted_prompt = prompt.format(
        question=question,
        history=history)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_prompt}
    ]
    response = call_gpt(messages)
    return parse_follow_up_question(response)


vectorstore = load_chroma("./vectorDB/disease_nodes_db", 'sentence-transformers/all-MiniLM-L6-v2')
embedding_function_for_context_retrieval = load_sentence_transformer('pritamdeka/S-PubMedBert-MS-MARCO')
node_context_df = pd.read_csv('./context_of_disease.csv')


def retriever(input):
    try:
        context = retrieve_context(input, vectorstore, embedding_function_for_context_retrieval, node_context_df,
                                   100, 95, 0.9, False)
        if not context or context.strip() in ['.', ',', '。',
                                              '，'] or context == "No disease entities found in the question.":
            return None
        return context
    except KeyError as e:
        print(f"RAG Error: {e}")
        return None


def generate_final_answer(caption, question, history, rag_context, answer_type, initial_answer, dataset_name):
    if answer_type == "open":
        system_prompt = OPEN_ENDED_REASONER_SYSTEM_PROMPT
        user_prompt = OPEN_ENDED_REASONER_PROMPT
    else:
        system_prompt = get_closed_ended_system_prompt(dataset_name, is_initial=False)
        user_prompt = CLOSED_ENDED_REASONER_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(
            caption=caption,
            question=question,
            initial_answer=initial_answer,
            history=history
        )}
    ]
    response = call_gpt(messages, answer_type=answer_type)
    return parse_final_answer(response)


def parse_final_answer(response):
    analysis = re.search(r'Analysis:(.*?)Answer:', response, re.DOTALL)
    answer = re.search(r'Answer:(.*)', response, re.DOTALL)
    return (analysis.group(1).strip() if analysis else "",
            answer.group(1).strip() if answer else "")


caption_prompts = [
    "What is the overall appearance of the image?",
    "Provide a detailed description of the given image",
    "Give an elaborate explanation of the image you see",
    "Share a comprehensive rundown of the presented image",
    "Offer a thorough analysis of the image",
    "Explain the various aspects of the image before you",
    "Clarify the contents of the displayed image with great detail",
    "Characterize the image using a well-detailed description",
    "Break down the elements of the image in a detailed manner",
    "Walk through the important details of the image",
    "Portray the image with a rich, descriptive narrative",
    "Narrate the contents of the image with precision",
    "Analyze the image in a comprehensive and detailed manner",
    "Illustrate the image through a descriptive explanation",
    "Examine the image closely and share its details",
    "Write an exhaustive depiction of the given image"
]


def generate_caption(model, image_tensor, model_type):
    question = random.choice(caption_prompts)
    caption = generate_initial_answer(model, image_tensor, question, "None", model_type, "open")
    return caption


def main(args):
    with open(args.test_file, 'r') as f:
        test_set = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, vis_processors, _ = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True,
                                                         device=device)
    vis_processor = vis_processors["eval"]

    test_set_chunk = get_chunk(test_set, args.num_chunks, args.chunk_idx)

    save_path = os.path.join(args.save_root, f'amanda_med_vqa_{args.exp_tag}')
    os.makedirs(os.path.join(save_path, 'result'), exist_ok=True)
    with open(os.path.join(save_path, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    with open(args.train_file, 'r') as f:
        train_set = json.load(f)

    with open(args.answers_file, 'w', encoding='utf-8') as ans_file:
        for test_item in tqdm(test_set_chunk, desc=f"Processing chunk {args.chunk_idx}"):

            image_path = os.path.join(args.image_folder, test_item["image"])
            image = Image.open(image_path).convert('RGB')
            image_tensor = vis_processor(image).unsqueeze(0).to(device)
            question = test_item["conversations"][0]['value'].replace('<image>', '').strip()
            answer_type = "open" if test_item['answer_type'] in ['open', 'OPEN'] else "close"

            caption = generate_caption(model, image_tensor, args.model_type)

            # 生成初始答案
            initial_answer = generate_initial_answer(model, image_tensor, question, args.conv_mode, args.model_type,
                                                     answer_type)

            history = [f"Initial model answer: {initial_answer}"]
            confidence = 0
            iteration = 0
            rag_context = None
            final_analysis = ""
            final_answer = ""

            # 获取相似例子
            examples = []
            if args.n > 0:
                similar_example_indices = test_item.get("similar_indices", [])[:args.n]
                for idx in similar_example_indices:
                    example = train_set[idx]
                    example_image_path = os.path.join(args.image_folder, example["image"])
                    example_image = Image.open(example_image_path).convert('RGB')
                    example_image_tensor = vis_processor(example_image).unsqueeze(0).to(device)
                    caption_prompt = random.choice(caption_prompts)
                    example_caption = generate_initial_answer(model, example_image_tensor, caption_prompt, "None",
                                                              args.model_type, "open")
                    examples.append({
                        "question": example["conversations"][0]['value'].replace('<image>', '').strip(),
                        "answer": example["conversations"][1]['value'],
                        "caption": example_caption
                    })
            sub_question_counter = 1
            while confidence < (
                    args.open_confidence_threshold if answer_type == "open" else args.close_confidence_threshold) and iteration < args.max_iterations:
                if iteration == 0:
                    analysis, answer = generate_reasoned_answer(caption, question, initial_answer, answer_type,
                                                                examples, args.dataset_name)
                    history.append(f"Iteration {iteration} answer: {answer}")
                    history.append(f"in_context_examples: {examples}")
                else:
                    analysis, answer = generate_final_answer(caption, question, "\n".join(history), rag_context,
                                                             answer_type, initial_answer, args.dataset_name)
                    history.append(f"Iteration {iteration} answer: {answer}")

                confidence, explanation = evaluate_answer(caption, question, answer, "\n".join(history))

                if confidence >= (
                        args.open_confidence_threshold if answer_type == "open" else args.close_confidence_threshold):
                    final_analysis = analysis
                    final_answer = answer
                    break

                follow_up_questions = generate_follow_up_question(question, "\n".join(history), args.max_sub_questions)
                for follow_up_question in follow_up_questions:
                    if follow_up_question:
                        history.append(f"Sub-Question {sub_question_counter}: {follow_up_question}")
                        follow_up_answer = generate_initial_answer(model, image_tensor, follow_up_question,
                                                                   args.conv_mode, args.model_type, answer_type)
                        history.append(f"Sub-Answer {sub_question_counter}: {follow_up_answer}")
                        sub_question_counter += 1

                        retriever_input = f"{caption}\n{question}\n" + "\n".join(history)
                        current_rag_context = retriever(retriever_input)

                        if current_rag_context:
                            rag_context = current_rag_context
                            history.append(f"Additional Grounded Medical Information: {rag_context}")
                            print(f"Successfully retrieved RAG context in iteration {iteration}")
                        else:
                            # print(f"No valid RAG context retrieved in iteration {iteration}")
                            rag_context = None
                else:
                    rag_context = None

                iteration += 1

            if not final_answer:
                final_analysis, final_answer = analysis, answer

            # 准备结果
            result = {
                "id": test_item["id"],
                "iteration": iteration,
                "question": question,
                "answer": test_item["conversations"][1]['value'],
                "caption": caption,
                "history": history,
                "final_analysis": final_analysis,
                "final_answer": final_answer,
            }

            # 保存详细结果
            result_path = os.path.join(save_path, 'result', f'{test_item["id"]}.yaml')
            with open(result_path, 'w') as f:
                yaml.dump(result, f)

            # 保存摘要结果
            ans_file.write(json.dumps({
                "question_id": test_item["id"],
                "prompt": question,
                "iteration": iteration,
                "answer": test_item["conversations"][1]['value'],
                "text": final_answer
            }) + "\n")
            ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amanda Medical VQA")
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--answers_file", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_chunks", type=int, default=1, help="Total number of chunks to process")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Index of the current chunk to process")
    parser.add_argument("--conv_mode", type=str, default="None", help="Conversation mode")
    parser.add_argument("--save_root", type=str, default="./exp_result/", help="Root path for saving results")
    parser.add_argument("--exp_tag", type=str, required=True, help="Tag for this experiment")
    parser.add_argument("--open_confidence_threshold", type=int, default=3,
                        help="Confidence threshold for open-ended questions (1-5)")
    parser.add_argument("--close_confidence_threshold", type=int, default=3,
                        help="Confidence threshold for closed-ended questions (1-5)")
    parser.add_argument("--max_iterations", type=int, default=3, help="Maximum number of iterations")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--n", type=int, default=0, help="Number of examples to use (0-10)")
    parser.add_argument("--engine", type=str)
    parser.add_argument("--max_sub_questions", type=int, default=3,
                        help="Maximum number of sub-questions to generate in each iteration")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name to determine the appropriate closed-ended prompt")
    args = parser.parse_args()
    main(args)
