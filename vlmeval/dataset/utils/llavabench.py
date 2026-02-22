import numpy as np
import pandas as pd
from ...smp import *

rule_dict = {
    # 'llava_bench_conv': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'},  # noqa: E501
    # 'llava_bench_detail': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'},  # noqa: E501
    # 'llava_bench_complex': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.\nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.'}  # noqa: E501
    'llava_bench_conv': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.'},  # noqa: E501
    'llava_bench_detail': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.'},  # noqa: E501
    'llava_bench_complex': {'role': 'Assistant', 'prompt': 'We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. The user asks the question on observing an image. For your reference, the visual content in the image is represented with a few sentences describing the image. \nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.'}  # noqa: E501
}


def get_eval(judge, content):
    return judge.generate(content)


def parse_score(review):
    logger = get_logger('Evaluation')
    try:
        # score_pair = review.split('\n')[0]
        # score_pair = score_pair.replace(',', ' ')
        # sp = score_pair.split(' ')
        review = review.replace(', ', '\n')
        sp = review.split('Assistant 1: ')[-1].split('\nAssistant 2: ')
        if len(sp) == 2:
            if '/' in sp[0]:
                sp[0] = sp[0].split('/')[0]
            if '/' in sp[1]:
                sp[1] = sp[1].split('/')[0]
            # return [float(sp[0]), float(sp[1])]
            return [float(sp[0]), float(sp[1]), None]
        else:
            logger.error('error', review)
            # return [-1, -1]
            return [-1, -1, review]
    except Exception as e:
        logger.error(e, 'error', review)
        # return [-1, -1]
        return [-1, -1, review]

def build_prompt(line):
    cap_str = line['caption']
    question = line['question']
    ans1 = line['gpt4_ans']
    ans2 = line['prediction']
    category = 'llava_bench_' + line['category']
    rule = rule_dict[category]
    role, prompt = rule['role'], rule['prompt']

    content = (f'[Context]\n{cap_str}\n\n'
               f'[Question]\n{question}\n\n'
               f'[{role} 1]\n{ans1}\n\n[End of {role} 1]\n\n'
               f'[{role} 2]\n{ans2}\n\n[End of {role} 2]\n\n'
               f'[System]\n{prompt}\n\n')
    return content

def build_prompt_llama_judge(line):
    cap_str = line['caption']
    question = line['question']
    ans1 = line['gpt4_ans']
    ans2 = line['prediction']
    category = 'llava_bench_' + line['category']
    rule = rule_dict[category]
    role, prompt = rule['role'], rule['prompt']

    content = (f'[INST] <<SYS>>\n{prompt}\nDo not include labels, headings, or extra text. Do not output anything else.\n<</SYS>>\n'
               f'Context: ' + cap_str + '\n'
               f'Question: ' + question + '\n'
               f'Assistant 1: ' + ans1 + '\n'
               f'Assistant 2: ' + ans2 + '\n'
               f'[/INST]'
            )
    return content

def LLaVABench_atomeval(model, prompt):
    review = get_eval(model, prompt)
    scores = parse_score(review)
    return scores

def get_eval_batch(judge, content):
    return judge.generate_batch_vllm(content)

def LLaVABench_atomeval_batch_vllm(tups, model):
    rets = []
    review = get_eval_batch(model, tups)
    for rev in review:
        rets.append(parse_score(rev))
    return rets

def LLaVABench_score(data):
    cates = ['overall'] + list(set(data['category']))
    ret = defaultdict(list)

    for c in cates:
        ret['split'].append(c)
        sub = data[data['category'] == c] if c != 'overall' else data
        ret['Relative Score (main)'].append(np.mean(sub['score']) / np.mean(sub['gpt4_score']) * 100)
        ret['VLM Score'].append(np.mean(sub['score']) * 10)
        ret['GPT4 Score'].append(np.mean(sub['gpt4_score']) * 10)
    return pd.DataFrame(ret)
