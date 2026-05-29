import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleHQQQuantizedCache, FlexibleVanillaQuantizedCache

dataset2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}

CACHE_DIR = "./models_storage"

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    # parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-3B-Instruct-AWQ")
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--k_bits', type=int, default=8)
    parser.add_argument('--v_bits', type=int, default=8)
    parser.add_argument('--residual_length', type=int, default=128)
    parser.add_argument('--group_size', type=int, default=64)
    parser.add_argument('--asym', type=bool, default=True)
    # in HQQ, 0 for per-channel, 1 for per-token
    parser.add_argument('--axis_key', type=int, default=0)
    parser.add_argument('--axis_value', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=7500)
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    # if "chatglm3" in model_name:
    #     prompt = tokenizer.build_chat_input(prompt)
    # elif "chatglm" in model_name:
    #     prompt = tokenizer.build_prompt(prompt)
    # elif "longchat" in model_name or "vicuna" in model_name:
    #     from fastchat.model import get_conversation_template
    #     conv = get_conversation_template("vicuna")
    #     conv.append_message(conv.roles[0], prompt)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()
    # elif "llama2" in model_name:
    #     prompt = f"[INST]{prompt}[/INST]"
    # elif "xgen" in model_name:
    #     header = (
    #         "A chat between a curious human and an artificial intelligence assistant. "
    #         "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
    #     )
    #     prompt = header + f" ### Human: {prompt}\n###"
    # elif "internlm" in model_name:
    #     prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    # if "xgen" in model_name:
    #     response = response.strip().replace("Assistant:", "")
    # elif "internlm" in model_name:
    #     response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format, dataset, device, model_name, out_path, cache_config):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        # if "chatglm3" in model_name:
        #     tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        # if len(tokenized_prompt) > max_length:
        #     half = int(max_length/2)
        #     prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        # if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
        #     prompt = build_chat(tokenizer, prompt, model_name)
        # if "chatglm3" in model_name:
        #     if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        #         input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        #     else:
        #         input = prompt.to(device)
        # else:
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        past_key_values = FlexibleVanillaQuantizedCache(cache_config=cache_config)
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                # num_beams=1,
                # do_sample=False,
                # temperature=1.0,
                min_length=context_length+1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                past_key_values=past_key_values,
                use_cache=True
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                # num_beams=1,
                # do_sample=False,
                # temperature=1.0,
                past_key_values=past_key_values,
                use_cache=True
            )[0]
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    # dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_name, device):
    # if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
    #     tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    #     model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    # elif "llama2" in model_name:
    #     replace_llama_attn_with_flash_attn()
    #     tokenizer = LlamaTokenizer.from_pretrained(path)
    #     model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    # elif "longchat" in model_name or "vicuna" in model_name:
    #     from fastchat.model import load_model
    #     replace_llama_attn_with_flash_attn()
    #     model, _ = load_model(
    #         path,
    #         device='cpu',
    #         num_gpus=0,
    #         load_8bit=False,
    #         cpu_offloading=False,
    #         debug=False,
    #     )
    #     model = model.to(device)
    #     model = model.bfloat16()
    #     tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=CACHE_DIR, torch_dtype=torch.float16).to(device)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    cache_config = FlexibleQuantizedCacheConfig(nbits_key=args.k_bits, nbits_value=args.v_bits, residual_length=args.residual_length, q_group_size=args.group_size,
                                                asym=args.asym, axis_key=args.axis_key, axis_value=args.axis_value, device='cuda')
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    # model2path = json.load(open("config/model2path.json", "r"))
    # model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    # define your model
    # max_length = model2maxlen[model_name]
    max_length = args.max_length
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    # dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    # dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            if not os.path.exists("pred_e"):
                os.makedirs("pred_e")
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists("pred"):
                os.makedirs("pred")
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            out_path = f"pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        assert world_size == 1
        get_pred(0, world_size, data_subsets[0], max_length, max_gen, prompt_format, dataset, device, model_name, out_path, cache_config)
        # processes = []
        # for rank in range(world_size):
        #     p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
        #                 max_gen, prompt_format, dataset, device, model_name, model2path, out_path))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
